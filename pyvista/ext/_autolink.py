"""Dynamic hyperlinking of identifiers inside ``.. pyvista-plot::`` output.

Resolves each accessed identifier against the real namespace it executed
in, rather than inferring its type statically (cf. ``sphinx-codeautolink``).
Wired into :mod:`pyvista.ext.plot_directive`; opt-in via
``pyvista_plot_autolink``. Never executes code itself.

Two phases:

1. :func:`record_namespace` -- walks the executed source with :mod:`ast`
   and resolves each accessed name to a list of candidate documented names.
2. :func:`_embed_links` (``build-finished``) -- matches candidates against
   the local Python domain and intersphinx inventories, and rewrites the
   built HTML.

Limitations:

- Only the final namespace state is used: a variable reassigned to a
  different type resolves against its last type everywhere it's used.
- A one-off chain with no intermediate variable (``pv.Sphere().plot()``)
  resolves its trailing attribute (``.plot``) only if the called function's
  return annotation is a plain, resolvable class name -- not a string
  expression (``Widget | str``), and only one call deep.
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

#: Matches any anchor tag, ours or another extension's.
_ANCHOR_RE = re.compile(r'<a\b[^>]*>.*?</a>', re.DOTALL)

# Pygments token classes for bare identifiers (``n`` plain names, ``nn``/``nc``/...
# more specific guesses, e.g. module names in an ``import`` line). Dots render as ``o``.
_NAME_SPAN = '<span class="n[a-zA-Z]{{0,2}}">{}</span>'
_DOT_SPAN = '<span class="o">.</span>'

#: A call's closing paren -- either its own token (``foo(x)``) or merged with the
#: opening one for a no-argument call (``foo()``).
_CALL_END = r'<span class="p">\(?\)</span>'


def _dotted_span_source(parts: tuple[str, ...]) -> str:
    """Build a regex source matching how Pygments is likely to render a dotted chain."""
    return _DOT_SPAN.join(_NAME_SPAN.format(re.escape(part)) for part in parts)


def _name_pattern_source(accessed: str) -> str:
    """Build a regex source matching how Pygments is likely to render ``accessed``."""
    return _dotted_span_source(tuple(accessed.split('.')))


@dataclass(frozen=True)
class _Candidate:
    """One accessed name and the documented names it might resolve to."""

    accessed: str
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class _CallCandidate:
    """A trailing attribute chain on a call's result, and its candidate names."""

    call_target: str
    trailing: tuple[str, ...]
    candidates: tuple[str, ...]


# ---------------------------------------------------------------------------
# Phase 1: collect accessed names from source, resolve each against the
# namespace it executed in to a list of candidate documented names.
# ---------------------------------------------------------------------------


def _dotted_name(node: ast.expr) -> str | None:
    """Return ``a.b.c`` for a chain rooted at a plain name, or ``None``."""
    parts: list[str] = []
    cursor = node
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if isinstance(cursor, ast.Name):
        parts.append(cursor.id)
        return '.'.join(reversed(parts))
    return None


class _NameCollector(ast.NodeVisitor):
    """Collect dotted names, and trailing attributes on a call's result."""

    def __init__(self) -> None:
        self.accessed: set[str] = set()
        #: (call target, trailing attrs), e.g. ``('pv.Sphere', ('plot',))``
        #: for ``pv.Sphere().plot``.
        self.call_chains: set[tuple[str, tuple[str, ...]]] = set()

    def visit_Name(self, node: ast.Name) -> None:
        """Record a bare name access."""
        self.accessed.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Record a dotted chain rooted at a plain name, or a call's trailing chain."""
        parts = []
        cursor: ast.expr = node
        while isinstance(cursor, ast.Attribute):
            parts.append(cursor.attr)
            cursor = cursor.value
        if isinstance(cursor, ast.Name):
            parts.append(cursor.id)
            self.accessed.add('.'.join(reversed(parts)))
            return
        if isinstance(cursor, ast.Call):
            call_target = _dotted_name(cursor.func)
            if call_target is not None and parts:
                self.call_chains.add((call_target, tuple(reversed(parts))))
        # e.g. `pv.Sphere().plot` -- keep walking the call's own arguments.
        self.visit(cursor)


def _accessed_names(source: str) -> set[str]:
    """Return every dotted name accessed in ``source``, or none on a parse error."""
    return _collect(source).accessed


def _call_chains(source: str) -> set[tuple[str, tuple[str, ...]]]:
    """Return every ``(call target, trailing attrs)`` pair, or none on a parse error."""
    return _collect(source).call_chains


def _collect(source: str) -> _NameCollector:
    collector = _NameCollector()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return collector
    collector.visit(tree)
    return collector


def _module_path_candidates(thing: type | Any, method: list[str]) -> Iterator[str]:
    """Yield ``thing``'s qualified name at every module-path truncation depth.

    E.g. ``pyvista.core.pointset.PolyData``, ``pyvista.core.PolyData``,
    ``pyvista.PolyData``, ... so whichever depth is actually documented
    gets found.
    """
    qualname = getattr(thing, '__qualname__', None)
    if qualname is None:
        # e.g. a functools.partial instance: isroutine() is true (it's a
        # method descriptor), but it has no qualified name of its own.
        return
    module = inspect.getmodule(thing)
    if module is None:
        return
    module_parts = module.__name__.split('.')
    for depth in range(len(module_parts), 0, -1):
        yield '.'.join([*module_parts[:depth], qualname, *method])


def _class_candidates(cls: type, method: list[str]) -> list[str]:
    """Return module-path-truncated candidates for ``cls`` and every base class."""
    classes = [cls]
    offset = 0
    while offset < len(classes):
        for base in classes[offset].__bases__:
            if base is not object and base not in classes:
                classes.append(base)
        offset += 1
    return [name for cc in classes for name in _module_path_candidates(cc, method)]


def _candidate_names(accessed: str, namespace: dict[str, Any]) -> list[str]:
    """Return candidate documented names for one dotted name access.

    Tries every prefix of ``accessed`` against ``namespace``; the longest
    prefix bound to a real object wins, then walks the remaining attributes
    on that live object.
    """
    parts = accessed.split('.')
    for split in range(len(parts)):
        head = '.'.join(parts[: split + 1])
        if head not in namespace:
            continue
        obj = namespace[head]
        remainder = parts[split + 1 :]

        if inspect.ismodule(obj) and not remainder:
            return [obj.__name__]

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
                break
            if inspect.ismethod(obj):
                obj = owner
                is_class_attr, method = True, [level]
                break

        if inspect.ismodule(obj):
            # `obj` is itself a (sub)module reached via attribute access
            # (e.g. `pv.examples`) -- not further resolved into a class,
            # function, or method, so nothing built below applies to it.
            return [obj.__name__]

        is_class = inspect.isclass(obj)
        if is_class or is_class_attr:
            return _class_candidates(obj if is_class else obj.__class__, method)

        if inspect.isroutine(obj):
            return list(_module_path_candidates(obj, []))

        return list(_module_path_candidates(obj.__class__, []))
    return []


#: Bare, dotted class name -- e.g. ``PolyData`` or ``pyvista.PolyData`` -- rejecting
#: anything that isn't (``Widget | str``, ``list[int]``, ...).
_SIMPLE_NAME_RE = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\Z')


def _resolve_object(accessed: str, namespace: dict[str, Any]) -> Any | None:
    """Resolve a dotted name to the live object it refers to, or ``None``."""
    parts = accessed.split('.')
    for split in range(len(parts)):
        head = '.'.join(parts[: split + 1])
        if head not in namespace:
            continue
        obj = namespace[head]
        for level in parts[split + 1 :]:
            try:
                obj = getattr(obj, level)
            except Exception:  # noqa: BLE001
                return None
        return obj
    return None


def _call_return_type(func: Any, namespace: dict[str, Any]) -> type | None:
    """Return ``func``'s return type, if its annotation names one plain, resolvable class.

    Checked against ``func``'s own module first, then every module already
    present in ``namespace`` -- covering re-exported aliases (e.g.
    ``pyvista.PolyData``) that ``func``'s own module never actually imports,
    only references under ``TYPE_CHECKING``.
    """
    annotation = getattr(func, '__annotations__', {}).get('return')
    if isinstance(annotation, type):
        return annotation
    if not isinstance(annotation, str) or not _SIMPLE_NAME_RE.match(annotation):
        return None
    name = annotation.rsplit('.', 1)[-1]
    namespaces = [getattr(func, '__globals__', {})]
    namespaces.extend(vars(obj) for obj in namespace.values() if inspect.ismodule(obj))
    for ns in namespaces:
        candidate = ns.get(name)
        if isinstance(candidate, type):
            return candidate
    return None


def _call_chain_candidates(
    call_target: str, trailing: tuple[str, ...], namespace: dict[str, Any]
) -> list[str]:
    """Return candidate documented names for a call's trailing attribute chain."""
    func = _resolve_object(call_target, namespace)
    if func is None or not inspect.isroutine(func):
        return []
    return_type = _call_return_type(func, namespace)
    if return_type is None:
        return []
    return _class_candidates(return_type, list(trailing))


def record_namespace(
    *, env: BuildEnvironment, docname: str, source: str, namespace: dict[str, Any]
) -> None:
    """Record candidate documented names for every identifier in ``source``."""
    all_records: dict[str, list[_Candidate | _CallCandidate]] | None = getattr(
        env, _ENV_ATTR, None
    )
    if all_records is None:
        all_records = {}
        setattr(env, _ENV_ATTR, all_records)
    records = all_records.setdefault(docname, [])
    collected = _collect(source)
    for accessed in sorted(collected.accessed):
        candidates = _candidate_names(accessed, namespace)
        if candidates:
            records.append(_Candidate(accessed, tuple(candidates)))
    for call_target, trailing in sorted(collected.call_chains):
        candidates = _call_chain_candidates(call_target, trailing, namespace)
        if candidates:
            records.append(_CallCandidate(call_target, trailing, tuple(candidates)))


def _merge_records(  # noqa: PLR0917
    app: Sphinx,  # noqa: ARG001
    env: BuildEnvironment,
    docnames: list[str],  # noqa: ARG001
    other: BuildEnvironment,
) -> None:
    """Merge records collected in a parallel-reading worker process."""
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
    """Rewrite built HTML pages with links for every resolved recorded name."""
    if exception is not None or app.builder.format != 'html':
        return

    records: dict[str, list[_Candidate | _CallCandidate]] = getattr(app.env, _ENV_ATTR, {})
    if not records:
        return

    local = _local_inventory(app)
    external = _intersphinx_inventory(app)

    for docname, candidates in records.items():
        out_file = Path(app.outdir) / (app.builder.get_target_uri(docname))
        if not out_file.exists():
            continue

        # Deduplicate: the same accessed name, or the same call chain, can be
        # recorded once per documented function that references it.
        resolved_names: dict[str, str] = {}
        resolved_calls: dict[tuple[str, tuple[str, ...]], str] = {}
        for candidate in candidates:
            if isinstance(candidate, _CallCandidate):
                call_key = (candidate.call_target, candidate.trailing)
                if call_key in resolved_calls:
                    continue
                link = _resolve_link(
                    candidate.candidates, docname=docname, app=app, local=local, external=external
                )
                if link is not None:
                    resolved_calls[call_key] = link
            else:
                if candidate.accessed in resolved_names:
                    continue
                link = _resolve_link(
                    candidate.candidates, docname=docname, app=app, local=local, external=external
                )
                if link is not None:
                    resolved_names[candidate.accessed] = link
        if not resolved_names and not resolved_calls:
            continue

        # One combined pattern, longest name first, so `mesh` can't match -- and get
        # wrapped again -- inside a `mesh.plot` match's own text. Each call-chain
        # pattern gets a nested `w{i}` group marking the part to actually wrap, since
        # its leading closing-paren context must stay outside the link.
        group_kind: dict[int, str] = {}
        group_link: dict[int, str] = {}
        sources: list[str] = []
        for i, name in enumerate(sorted(resolved_names, key=len, reverse=True)):
            group_kind[i] = 'name'
            group_link[i] = resolved_names[name]
            sources.append(f'(?P<n{i}>{_name_pattern_source(name)})')
        offset = len(sources)
        for j, key in enumerate(sorted(resolved_calls, key=lambda k: len(k[1]), reverse=True)):
            i = offset + j
            group_kind[i] = 'call'
            group_link[i] = resolved_calls[key]
            _, trailing = key
            sources.append(
                f'(?P<n{i}>{_CALL_END}(?P<w{i}>{_DOT_SPAN}{_dotted_span_source(trailing)}))'
            )
        combined = re.compile('|'.join(sources))

        html = out_file.read_text(encoding='utf-8')

        # Skip matches already inside an anchor (ours, from an unchanged
        # incremental rebuild, or another extension's).
        already_linked = [m.span() for m in _ANCHOR_RE.finditer(html)]

        def _wrap(  # noqa: PLR0917
            match: re.Match[str],
            group_kind: dict[int, str] = group_kind,
            group_link: dict[int, str] = group_link,
            already_linked: list[tuple[int, int]] = already_linked,
        ) -> str:
            i = int(match.lastgroup[1:])
            link = group_link[i]
            if group_kind[i] == 'call':
                wrap_start, wrap_end = match.span(f'w{i}')
                if any(start <= wrap_start < end for start, end in already_linked):
                    return match.group(0)
                prefix = match.string[match.start() : wrap_start]
                wrapped = match.string[wrap_start:wrap_end]
                return f'{prefix}<a class="pyvista-autolink-a" href="{link}">{wrapped}</a>'
            if any(start <= match.start() < end for start, end in already_linked):
                return match.group(0)
            return f'<a class="pyvista-autolink-a" href="{link}">{match.group(0)}</a>'

        out_file.write_text(combined.sub(_wrap, html), encoding='utf-8')


def setup(app: Sphinx) -> None:
    """Wire up dynamic autolinking. Called by :mod:`pyvista.ext.plot_directive`."""
    app.connect('env-merge-info', _merge_records)
    app.connect('env-purge-doc', _purge_doc)
    app.connect('build-finished', _embed_links)
