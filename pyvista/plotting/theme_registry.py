"""Registry for named pyvista themes."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from importlib.metadata import EntryPoint
from importlib.metadata import entry_points
from typing import TYPE_CHECKING
from typing import Literal
from typing import NamedTuple
from typing import TypedDict

from pyvista._warn_external import warn_external
from pyvista.core.utilities._registry_helpers import handler_source

if TYPE_CHECKING:
    from .themes import Theme


THEME_ENTRY_POINT_GROUP = 'pyvista.themes'


class ThemeRegistration(NamedTuple):
    """Describe how a theme name became registered.

    Returned by :func:`~pyvista.registered_themes`. ``kind`` distinguishes
    the registration path: a :class:`~pyvista.plotting.themes.Theme`
    subclass with a class-level ``_default_name`` (``'subclass'``), a
    ``pyvista.themes`` entry point from an installed plugin
    (``'entry_point'``), or a built-in legacy alias such as ``'default'``
    or ``'vtk'`` (``'alias'``).

    Attributes
    ----------
    name : str
        The registered (normalized) theme name.
    kind : {'subclass', 'entry_point', 'alias'}
        How the name was registered.
    source : str
        Human-readable origin (e.g. ``'my_package.theme.MyTheme'``).

    Examples
    --------
    Inspect how a name became registered.

    >>> import pyvista as pv
    >>> next(r for r in pv.registered_themes() if r.name == 'dark')
    ThemeRegistration(name='dark', kind='subclass', source='pyvista.plotting.themes.DarkTheme')

    """

    name: str
    kind: Literal['subclass', 'entry_point', 'alias']
    source: str


class _ThemeRegistryState(TypedDict):
    """Stored registry state used by tests."""

    classes: dict[str, type[Theme]]
    classes_sources: dict[str, str]
    aliases: set[str]
    discovered: dict[str, type[Theme] | Theme]
    discovered_sources: dict[str, str]
    pending: dict[str, list[EntryPoint]]
    loaded: bool


_registered_theme_classes: dict[str, type[Theme]] = {}
_registered_theme_classes_sources: dict[str, str] = {}
_registered_theme_aliases: set[str] = set()
_discovered_entry_point_themes: dict[str, type[Theme] | Theme] = {}
_discovered_entry_point_sources: dict[str, str] = {}
# Entry-point metadata, populated by ``_ensure_entry_points``. Maps each
# theme name to the list of ``EntryPoint`` records that declared it. The
# plugin module itself is *not* imported until that name is actually
# requested via :func:`_resolve_theme`, keeping ``set_plot_theme`` calls
# for built-in themes free of third-party plugin import cost.
_pending_ep_themes: dict[str, list[EntryPoint]] = {}
_entry_points_loaded: bool = False


def _normalize_theme_name(name: str) -> str:
    """Normalize a theme name."""
    return name.strip().lower()


def _save_registry_state() -> _ThemeRegistryState:
    """Snapshot the current registry state for later restoration."""
    return {
        'classes': _registered_theme_classes.copy(),
        'classes_sources': _registered_theme_classes_sources.copy(),
        'aliases': _registered_theme_aliases.copy(),
        'discovered': _discovered_entry_point_themes.copy(),
        'discovered_sources': _discovered_entry_point_sources.copy(),
        'pending': {k: list(v) for k, v in _pending_ep_themes.items()},
        'loaded': _entry_points_loaded,
    }


def _restore_registry_state(state: _ThemeRegistryState) -> None:
    """Restore registry state from a snapshot."""
    global _entry_points_loaded  # noqa: PLW0603
    _registered_theme_classes.clear()
    _registered_theme_classes.update(state['classes'])
    _registered_theme_classes_sources.clear()
    _registered_theme_classes_sources.update(state['classes_sources'])
    _registered_theme_aliases.clear()
    _registered_theme_aliases.update(state['aliases'])
    _discovered_entry_point_themes.clear()
    _discovered_entry_point_themes.update(state['discovered'])
    _discovered_entry_point_sources.clear()
    _discovered_entry_point_sources.update(state['discovered_sources'])
    _pending_ep_themes.clear()
    _pending_ep_themes.update({k: list(v) for k, v in state['pending'].items()})
    _entry_points_loaded = state['loaded']


def _register_theme_class(name: str, cls: type[Theme], *, source: str) -> None:
    """Register a ``Theme`` subclass. Called from ``Theme.__init_subclass__``.

    Collisions warn (do not raise) so that a subclass definition at
    import time cannot crash the interpreter. Last wins, mirroring the
    behavior of every other PyVista plugin registry.
    """
    normalized = _normalize_theme_name(name)
    if not normalized:
        warn_external(
            f'Theme class {source} declared an empty ``_default_name``; skipped.',
        )
        return
    if normalized in _registered_theme_classes:
        existing = _registered_theme_classes_sources.get(normalized, '<unknown>')
        warn_external(
            f'Registering theme "{normalized}" from {source} replaces an '
            f'existing registration from {existing}.',
        )
    _registered_theme_classes[normalized] = cls
    _registered_theme_classes_sources[normalized] = source


def _register_alias(name: str, cls: type[Theme]) -> None:
    """Register a built-in name alias (e.g. ``'default' → DocumentTheme``).

    Bypasses collision detection so built-in aliases can coexist with
    auto-registered subclasses without warning.
    """
    normalized = _normalize_theme_name(name)
    _registered_theme_classes[normalized] = cls
    _registered_theme_classes_sources[normalized] = handler_source(cls)
    _registered_theme_aliases.add(normalized)


def _lookup_explicit(normalized: str) -> Theme | None:
    """Look up a normalized name in the explicit subclass registry.

    Classes are instantiated fresh every call so callers never share a
    registered instance with the global theme.
    """
    cls = _registered_theme_classes.get(normalized)
    if cls is not None:
        return cls()
    return None


def _lookup_discovered(normalized: str) -> Theme | None:
    """Look up a normalized name in the loaded entry-point registry."""
    discovered = _discovered_entry_point_themes.get(normalized)
    if isinstance(discovered, type):
        return discovered()
    return discovered


def _resolve_theme(name: str) -> Theme | None:
    """Look up a theme by name and return a usable ``Theme`` instance.

    Explicit subclass registrations win over entry-point discoveries.
    Entry-point plugins are imported lazily — only the plugin claiming
    *name* loads, sibling plugins stay pending.
    """
    normalized = _normalize_theme_name(name)

    resolved = _lookup_explicit(normalized)
    if resolved is not None:
        return resolved

    _ensure_entry_points()

    # Fast path: load the plugin claiming this exact name.
    if normalized in _pending_ep_themes:
        _resolve_pending_theme(normalized)

    # Loading an entry point's module may have triggered __init_subclass__
    # on a Theme subclass, which populates the explicit registry. Re-check
    # before falling back to the discovered registry.
    resolved = _lookup_explicit(normalized)
    if resolved is not None:
        return resolved
    resolved = _lookup_discovered(normalized)
    if resolved is not None:
        return resolved

    # Final fallback: a Mapping-style entry point may expose names that
    # differ from its ``ep.name``. The fast path above only matches by
    # ``ep.name``, so resolve any remaining pending EPs as a last resort.
    for pending_name in list(_pending_ep_themes):
        _resolve_pending_theme(pending_name)
        resolved = _lookup_explicit(normalized)
        if resolved is not None:
            return resolved
        resolved = _lookup_discovered(normalized)
        if resolved is not None:
            return resolved
    return None


def _available_theme_names() -> tuple[str, ...]:
    """Return all currently registered theme names.

    Pending entry-point names appear in the result without triggering
    plugin imports — only metadata is consulted.
    """
    _ensure_entry_points()
    names = (
        set(_registered_theme_classes)
        | set(_discovered_entry_point_themes)
        | set(_pending_ep_themes)
    )
    return tuple(sorted(names))


def registered_themes() -> tuple[ThemeRegistration, ...]:
    """Return all registered themes.

    Use this to discover which names can be passed to
    :func:`~pyvista.set_plot_theme` or the ``PYVISTA_PLOT_THEME``
    environment variable. Entry-point plugins are loaded on the first
    call so they appear in the result.

    Returns
    -------
    tuple[pyvista.ThemeRegistration, ...]
        One record per registered theme, sorted by name. Each record
        exposes ``name``, ``kind``, and ``source``.

    Examples
    --------
    List available theme names.

    >>> import pyvista as pv
    >>> for record in pv.registered_themes():
    ...     print(record.name)
    dark
    default
    document
    document_build
    document_pro
    paraview
    testing
    vtk

    Inspect how a name became registered.

    >>> next(r for r in pv.registered_themes() if r.name == 'dark').kind
    'subclass'

    """
    _ensure_entry_points()
    # Force every pending plugin to load so the result reflects every
    # theme visible to PyVista. A plugin that fails to import emits a
    # ``UserWarning`` and is skipped; the rest still appear.
    for pending_name in list(_pending_ep_themes):
        _resolve_pending_theme(pending_name)
    records: list[ThemeRegistration] = []
    for name, cls in _registered_theme_classes.items():
        kind: Literal['subclass', 'alias'] = (
            'alias' if name in _registered_theme_aliases else 'subclass'
        )
        source = _registered_theme_classes_sources.get(name, handler_source(cls))
        records.append(ThemeRegistration(name=name, kind=kind, source=source))
    seen = {r.name for r in records}
    for name in _discovered_entry_point_themes:
        if name in seen:
            # Explicit registrations always win over entry-point providers.
            continue
        records.append(
            ThemeRegistration(
                name=name,
                kind='entry_point',
                source=_discovered_entry_point_sources.get(name, '<unknown>'),
            )
        )
    return tuple(sorted(records, key=lambda r: r.name))


def _resolve_dotted_path(spec: str) -> type[Theme]:
    """Resolve a ``'package.module:ClassName'`` spec to a ``Theme`` subclass."""
    from .themes import Theme  # local import breaks circular dependency  # noqa: PLC0415

    if ':' not in spec:
        msg = f'Theme spec "{spec}" must use "package.module:ClassName" form.'
        raise ValueError(msg)
    module_path, _, class_name = spec.partition(':')
    if not module_path or not class_name:
        msg = f'Invalid theme spec "{spec}".'
        raise ValueError(msg)
    try:
        module = import_module(module_path)
    except ImportError as exc:
        msg = f'Cannot import "{module_path}" from theme spec "{spec}": {exc}'
        raise ValueError(msg) from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        msg = f'"{class_name}" not found in module "{module_path}".'
        raise ValueError(msg)
    if not (isinstance(cls, type) and issubclass(cls, Theme)):
        # ValueError (not TypeError) keeps parity with the other dotted-path
        # failure modes and lets _set_plot_theme_from_env catch it uniformly.
        msg = f'"{spec}" does not resolve to a pyvista Theme subclass.'
        raise ValueError(msg)  # noqa: TRY004
    return cls


def _ensure_entry_points() -> None:
    """Scan ``pyvista.themes`` entry-point metadata once.

    Populates :data:`_pending_ep_themes` with every name declared by an
    installed plugin. The plugin modules themselves are **not** imported
    here; the cost is one ``importlib.metadata.entry_points`` call.
    Plugin imports are deferred to :func:`_resolve_pending_theme`, which
    runs only when a theme with that specific name is actually requested.
    """
    global _entry_points_loaded  # noqa: PLW0603
    if _entry_points_loaded:
        return
    _entry_points_loaded = True

    for ep in entry_points(group=THEME_ENTRY_POINT_GROUP):
        normalized = _normalize_theme_name(ep.name)
        if not normalized:
            warn_external(
                f'Ignoring {THEME_ENTRY_POINT_GROUP} entry point from '
                f'{ep.value}: theme name must not be empty.',
            )
            continue
        if normalized in _registered_theme_classes:
            # Explicit registration wins silently — user-owned registrations
            # shouldn't be shouted at just because a plugin also defines it.
            continue
        _pending_ep_themes.setdefault(normalized, []).append(ep)


def _resolve_pending_theme(name: str) -> bool:
    """Import the plugin claiming *name*, if any.

    Returns
    -------
    bool
        ``True`` if a plugin loaded successfully and contributed at least
        one registration. ``False`` if no pending plugin matches, or if
        the plugin failed to import.

    Notes
    -----
    A plugin that fails to import emits a ``UserWarning`` and is dropped
    from the pending list, so subsequent lookups of the same name fall
    straight through without re-triggering the import or re-emitting the
    warning.

    """
    eps = _pending_ep_themes.pop(name, None)
    if not eps:
        return False
    winner = eps[0]
    source = winner.value
    try:
        # ep.load() runs third-party import machinery — it can raise
        # literally anything. Convert to a warning so one broken plugin
        # cannot take down every theme lookup.
        loaded = winner.load()
    except Exception as exc:  # noqa: BLE001
        warn_external(
            f'Failed to load {THEME_ENTRY_POINT_GROUP} entry point '
            f'"{winner.name}" from {source}: {exc}',
        )
        return False

    if isinstance(loaded, Mapping):
        for theme_name, theme_obj in loaded.items():
            if not isinstance(theme_name, str):
                warn_external(
                    f'Ignoring {THEME_ENTRY_POINT_GROUP} entry point from '
                    f'{source}: theme names must be strings.',
                )
                continue
            _register_discovered_theme(theme_name, theme_obj, source=source)
    else:
        _register_discovered_theme(winner.name, loaded, source=source)

    if len(eps) > 1:
        providers = ', '.join(ep.value for ep in eps)
        warn_external(
            f'Multiple {THEME_ENTRY_POINT_GROUP} providers registered for '
            f'"{name}": {providers}. Using {source}.',
        )
    return name in _registered_theme_classes or name in _discovered_entry_point_themes


def _register_discovered_theme(name: str, obj: object, *, source: str) -> None:
    """Register an entry-point-discovered theme."""
    from .themes import Theme  # local import breaks circular dependency  # noqa: PLC0415

    normalized = _normalize_theme_name(name)
    if not normalized:
        warn_external(
            f'Ignoring {THEME_ENTRY_POINT_GROUP} entry point from '
            f'{source}: theme name must not be empty.',
        )
        return

    if normalized in _registered_theme_classes:
        # Explicit registration wins silently — user-owned registrations
        # shouldn't be shouted at just because a plugin also defines it.
        return

    if normalized in _discovered_entry_point_themes:
        existing = _discovered_entry_point_sources[normalized]
        warn_external(
            f'Multiple {THEME_ENTRY_POINT_GROUP} providers registered for '
            f'"{normalized}": {existing}, {source}. Using {existing}.',
        )
        return

    is_class = isinstance(obj, type) and issubclass(obj, Theme)
    is_instance = isinstance(obj, Theme)
    if not (is_class or is_instance):
        warn_external(
            f'Ignoring {THEME_ENTRY_POINT_GROUP} entry point "{normalized}" '
            f'from {source}: value is not a pyvista Theme subclass or instance.',
        )
        return

    # ``obj`` was just narrowed to ``type[Theme] | Theme`` by the
    # ``is_class or is_instance`` check, but mypy can't see through the
    # boolean union. Safe to assign.
    _discovered_entry_point_themes[normalized] = obj  # type: ignore[assignment]
    _discovered_entry_point_sources[normalized] = source
