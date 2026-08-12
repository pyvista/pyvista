"""Dynamic hyperlinking of identifiers inside ``.. pyvista-plot::`` output.

This is a from-scratch, PyVista-specific replacement for the static-analysis
approach of ``sphinx-codeautolink``, which cannot correctly resolve most
method chains (``mesh.warp_by_scalar(...).extract_surface()``) since it has
to *guess* a call's return type from static annotations. This module instead
mirrors how Sphinx-Gallery's own ``reference_url`` feature resolves names --
by looking at what an identifier *actually* evaluated to once the code
already ran, rather than inferring it -- which sidesteps that whole class of
problem for free: a real executed object always has exactly one real type.

Scope (v1): wired directly into :mod:`pyvista.ext.plot_directive`, which
already executes every docstring example to render its figures. This module
never executes anything itself; it only analyses the namespace the plot
directive produces, once, right after it runs -- so nothing is ever run
twice. It is not (yet) usable by projects that don't already have their own
execution step; see ``setup()`` for wiring, or the module docstring section
below for what would need to change to make this a standalone extension.

Design, two phases, closely modeled on ``sphinx_gallery.backreferences``:

1. :func:`record_namespace` (called by the plot directive, once per page,
   right after it finishes executing a docstring's examples): walks the
   source with :mod:`ast`, and for every accessed dotted name
   (``mesh.warp_by_scalar``), resolves it against the *live* namespace the
   code just ran in. Since the object is real, its true class -- and every
   one of that class's base classes -- gives a set of *candidate* documented
   names, by trying the object's module path at every truncation depth
   (``pyvista.core.dataset.DataSet.plot``, ``pyvista.core.DataSet.plot``,
   ``pyvista.DataSet.plot``, ...). Only plain strings are stored on the
   build environment here -- never the namespace or any live object, which
   may not be picklable (a live VTK object almost certainly isn't) and
   wouldn't survive Sphinx's parallel-build worker merging.

2. :func:`_embed_links` (``build-finished``, once the whole site's object
   inventory is actually known): for each page with recorded candidates,
   checks each candidate against the local Python domain and intersphinx
   inventories, and rewrites the already-built HTML to wrap whichever
   candidate matched (if any) in a hyperlink. Never wraps text already
   inside an anchor tag -- its own, from a page an incremental rebuild left
   untouched on disk, or another extension's (e.g. ``sphinx-codeautolink``,
   if still active on the same site) -- so re-running this, or running
   alongside another autolinker, never nests a second ``<a>`` inside one
   already there.

Known limitations, matching Sphinx-Gallery's own resolver rather than
improving on it (see the "Namespace capture" design decision):

- Only the *final* state of the namespace is used, so a variable reused
  both before and after being reassigned to a different type resolves
  against its last type everywhere it's used.
- Only chains rooted at a plain name are followed (``mesh.plot()``, where
  ``mesh`` is a variable) -- a one-off chain with no intermediate variable
  (``pv.Sphere().plot()``) only resolves its first hop, since there is
  nothing in the namespace to look the call's result up under.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect
from pathlib import Path
import re
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment

#: ``env`` attribute holding recorded candidates, keyed by docname.
_ENV_ATTR = 'pyvista_autolink_records'

#: Matches any already-rendered anchor tag, ours or another extension's
#: (e.g. ``sphinx-codeautolink``'s). Used to skip re-wrapping text that's
#: already inside a link -- see :func:`_embed_links`.
_ANCHOR_RE = re.compile(r'<a\b[^>]*>.*?</a>', re.DOTALL)

# Pygments token classes seen in practice for bare identifiers across the
# python/pycon lexers (``n`` for plain names, ``nn``/``nc``/... for names
# Pygments has a more specific guess for, e.g. ``nn`` for a module named in
# an ``import`` line). The dot between attribute parts is consistently
# rendered as ``o``.
_NAME_SPAN = '<span class="n[a-zA-Z]{{0,2}}">{}</span>'
_DOT_SPAN = '<span class="o">.</span>'


def _name_pattern_source(accessed: str) -> str:
    """Build a regex source matching how Pygments is likely to render ``accessed``."""
    parts = (_NAME_SPAN.format(re.escape(part)) for part in accessed.split('.'))
    return _DOT_SPAN.join(parts)


@dataclass(frozen=True)
class _Candidate:
    """One accessed name and the documented names it might resolve to."""

    accessed: str
    candidates: tuple[str, ...]


# ---------------------------------------------------------------------------
# Phase 1: collect accessed names from source, resolve each against the
# namespace it executed in to a list of candidate documented names.
# ---------------------------------------------------------------------------


class _NameCollector(ast.NodeVisitor):
    """Collect every dotted name accessed in a chain rooted at a plain name.

    Ported from ``sphinx_gallery.backreferences.NameFinder``: only ``a.b.c``
    forms are collected, not ``a().b`` -- the result of a call that isn't
    stored in a variable has nothing to look up in the namespace afterwards.
    """

    def __init__(self) -> None:
        self.accessed: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        """Record a bare name access."""
        self.accessed.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Record a dotted chain rooted at a plain name."""
        parts = []
        cursor: ast.expr = node
        while isinstance(cursor, ast.Attribute):
            parts.append(cursor.attr)
            cursor = cursor.value
        if isinstance(cursor, ast.Name):
            parts.append(cursor.id)
            self.accessed.add('.'.join(reversed(parts)))
        else:
            # e.g. `pv.Sphere().plot` -- nothing to look up for the call
            # result, but keep walking in case the call's own arguments
            # reference something resolvable.
            self.visit(cursor)


def _accessed_names(source: str) -> set[str]:
    """Return every dotted name accessed in ``source``, or none on a parse error."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    collector = _NameCollector()
    collector.visit(tree)
    return collector.accessed


def _module_path_candidates(thing: type | Any, method: list[str]) -> Iterator[str]:
    """Yield ``thing``'s qualified name at every module-path truncation depth.

    ``thing`` is a class, or a plain function/builtin reached without going
    through a bound instance method. Either way, it's documented under its
    public re-exported path (``pyvista.PolyData``, ``pyvista.Sphere``), but
    usually lives, at runtime, several packages deeper
    (``pyvista.core.pointset.PolyData``) than where it's documented. Trying
    every prefix, rather than only the full or only the top-level one, means
    whichever one is actually documented gets found without having to know
    in advance how many packages were re-exported through.
    """
    module = inspect.getmodule(thing)
    if module is None:
        return
    module_parts = module.__name__.split('.')
    for depth in range(len(module_parts), 0, -1):
        yield '.'.join([*module_parts[:depth], thing.__qualname__, *method])


def _candidate_names(accessed: str, namespace: dict[str, Any]) -> list[str]:
    """Return candidate documented names for one dotted name access.

    Tries every prefix of ``accessed`` against ``namespace`` -- the longest
    prefix bound to a real object wins -- then walks the remaining
    attributes on that live object. A module resolves directly to its own
    name. Otherwise, whatever the walk lands on (and, if it's an inherited
    method or property, every base class it might actually be documented
    on) is turned into module-path-truncated candidates by
    :func:`_module_path_candidates`.
    """
    parts = accessed.split('.')
    for split in range(len(parts)):
        head = '.'.join(parts[: split + 1])
        if head not in namespace:
            continue
        obj = namespace[head]
        remainder = parts[split + 1 :]

        if inspect.ismodule(obj):
            return ['.'.join([obj.__name__, *remainder])]

        is_class_attr = False
        method: list[str] = []
        for level in remainder:
            owner = obj
            prop = getattr(type(owner), level, None) if not inspect.isclass(owner) else None
            if isinstance(prop, property):
                obj = owner
                is_class_attr, method = True, [level]
                break
            try:
                obj = getattr(obj, level)
            except Exception:  # noqa: BLE001
                # be as lenient as the object under inspection demands --
                # attribute access on arbitrary live objects can raise
                # anything, not just AttributeError. The chain is broken
                # either way, so there's nothing further to walk.
                break
            if inspect.ismethod(obj):
                obj = owner
                is_class_attr, method = True, [level]
                break

        is_class = inspect.isclass(obj)
        if is_class or is_class_attr:
            classes = [obj if is_class else obj.__class__]
            offset = 0
            while offset < len(classes):
                for base in classes[offset].__bases__:
                    if base is not object and base not in classes:
                        classes.append(base)
                offset += 1
            return [name for cc in classes for name in _module_path_candidates(cc, method)]

        if inspect.isroutine(obj):
            # A plain function/builtin reached without going through a bound
            # instance method -- e.g. `pv.Sphere` itself, not `mesh.plot`.
            # Its own module + qualname is what's documented, not
            # `type(obj)` (which would just be `builtins.function`).
            return list(_module_path_candidates(obj, []))

        return list(_module_path_candidates(obj.__class__, []))
    return []


def record_namespace(
    *, env: BuildEnvironment, docname: str, source: str, namespace: dict[str, Any]
) -> None:
    """Record candidate documented names for every identifier in ``source``.

    Called once per page, right after :mod:`pyvista.ext.plot_directive`
    finishes executing that page's docstring examples. Stores only plain
    strings -- see the module docstring for why.
    """
    all_records: dict[str, list[_Candidate]] | None = getattr(env, _ENV_ATTR, None)
    if all_records is None:
        all_records = {}
        setattr(env, _ENV_ATTR, all_records)
    records = all_records.setdefault(docname, [])
    for accessed in sorted(_accessed_names(source)):
        candidates = _candidate_names(accessed, namespace)
        if candidates:
            records.append(_Candidate(accessed, tuple(candidates)))


def _merge_records(  # noqa: PLR0917
    app: Sphinx,  # noqa: ARG001
    env: BuildEnvironment,
    docnames: list[str],  # noqa: ARG001
    other: BuildEnvironment,
) -> None:
    """Merge records collected in a parallel-reading worker process.

    Signature fixed by Sphinx's ``env-merge-info`` event.
    """
    ours = getattr(env, _ENV_ATTR, {})
    theirs = getattr(other, _ENV_ATTR, {})
    ours.update(theirs)
    setattr(env, _ENV_ATTR, ours)


def _purge_doc(app: Sphinx, env: BuildEnvironment, docname: str) -> None:  # noqa: ARG001
    """Drop stale records for a document being re-read."""
    getattr(env, _ENV_ATTR, {}).pop(docname, None)


# ---------------------------------------------------------------------------
# Phase 2: once the whole site's objects are known, match candidates against
# the real inventory and rewrite the already-built HTML.
# ---------------------------------------------------------------------------


def _local_inventory(app: Sphinx) -> dict[str, tuple[str, str]]:
    """Return ``{name: (docname, anchor)}`` for every locally documented Python object."""
    return {
        name: (entry.docname, entry.node_id)
        for name, entry in app.env.domains['py'].objects.items()
    }


def _intersphinx_inventory(app: Sphinx) -> dict[str, str]:
    """Return ``{name: absolute_url}`` for every intersphinx-mapped object."""
    from sphinx.ext.intersphinx import InventoryAdapter  # noqa: PLC0415

    urls: dict[str, str] = {}
    for by_objtype in InventoryAdapter(app.env).main_inventory.values():
        for name, item in by_objtype.items():
            urls.setdefault(name, item[2])
    return urls


def _resolve_link(
    candidates: tuple[str, ...],
    *,
    docname: str,
    app: Sphinx,
    local: dict[str, tuple[str, str]],
    external: dict[str, str],
) -> str | None:
    """Return the first candidate's URL, local names taking priority, or ``None``."""
    for name in candidates:
        if name in local:
            target_docname, anchor = local[name]
            return f'{app.builder.get_relative_uri(docname, target_docname)}#{anchor}'
        if name in external:
            return external[name]
    return None


def _embed_links(app: Sphinx, exception: Exception | None) -> None:
    """Rewrite built HTML pages with links for every resolved recorded name.

    ``build-finished`` handler: the whole site's objects (and every
    intersphinx inventory) are only fully known once every page has been
    read, so this is the earliest point at which candidates collected by
    :func:`record_namespace` can actually be checked against real targets.
    """
    if exception is not None or app.builder.format != 'html':
        return

    records: dict[str, list[_Candidate]] = getattr(app.env, _ENV_ATTR, {})
    if not records:
        return

    local = _local_inventory(app)
    external = _intersphinx_inventory(app)

    for docname, candidates in records.items():
        out_file = Path(app.outdir) / (app.builder.get_target_uri(docname))
        if not out_file.exists():
            continue

        # Deduplicate first: the same accessed name can be recorded once for
        # every documented function that happens to reference it.
        resolved: dict[str, str] = {}
        for candidate in candidates:
            if candidate.accessed in resolved:
                continue
            link = _resolve_link(
                candidate.candidates, docname=docname, app=app, local=local, external=external
            )
            if link is not None:
                resolved[candidate.accessed] = link
        if not resolved:
            continue

        # One combined pattern, longest name first, rather than one `.sub()`
        # call per name: a sequence of separate passes would let `mesh`
        # alone match -- and get wrapped a second time -- inside text a
        # longer `mesh.plot` pass already wrapped a link around.
        names = sorted(resolved, key=len, reverse=True)
        combined = re.compile(
            '|'.join(f'(?P<n{i}>{_name_pattern_source(name)})' for i, name in enumerate(names))
        )

        html = out_file.read_text(encoding='utf-8')

        # Positions already inside *any* anchor -- ours from a previous,
        # unchanged incremental build that left this exact file on disk, or
        # another extension's (e.g. sphinx-codeautolink, if still active on
        # the same site) that already linked this name. Either way, nothing
        # here should be wrapped a second time.
        already_linked = [m.span() for m in _ANCHOR_RE.finditer(html)]

        def _wrap(  # noqa: PLR0917
            match: re.Match[str],
            names: list[str] = names,
            resolved: dict[str, str] = resolved,
            already_linked: list[tuple[int, int]] = already_linked,
        ) -> str:
            if any(start <= match.start() < end for start, end in already_linked):
                return match.group(0)
            link = resolved[names[int(match.lastgroup[1:])]]
            return f'<a class="pyvista-autolink-a" href="{link}">{match.group(0)}</a>'

        out_file.write_text(combined.sub(_wrap, html), encoding='utf-8')


def setup(app: Sphinx) -> None:
    """Wire up dynamic autolinking.

    Called by :mod:`pyvista.ext.plot_directive`; this module is not a
    Sphinx extension of its own.
    """
    app.connect('env-merge-info', _merge_records)
    app.connect('env-purge-doc', _purge_doc)
    app.connect('build-finished', _embed_links)
