"""Registry for named pyvista themes."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from importlib.metadata import entry_points
from typing import TYPE_CHECKING
from typing import Literal
from typing import NamedTuple
from typing import TypedDict

from pyvista._warn_external import warn_external

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
    >>> pv.registered_themes()['dark']
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
    loaded: bool


_registered_theme_classes: dict[str, type[Theme]] = {}
_registered_theme_classes_sources: dict[str, str] = {}
_registered_theme_aliases: set[str] = set()
_discovered_entry_point_themes: dict[str, type[Theme] | Theme] = {}
_discovered_entry_point_sources: dict[str, str] = {}
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
    _entry_points_loaded = state['loaded']


def _register_theme_class(name: str, cls: type[Theme], *, source: str) -> None:
    """Register a ``Theme`` subclass. Called from ``Theme.__init_subclass__``.

    Collisions warn (do not raise) so that a subclass definition at
    import time cannot crash the interpreter.
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
            f'Theme name "{normalized}" is already registered by {existing}; '
            f'ignoring duplicate registration from {source}.',
        )
        return
    _registered_theme_classes[normalized] = cls
    _registered_theme_classes_sources[normalized] = source


def _register_alias(name: str, cls: type[Theme]) -> None:
    """Register a built-in name alias (e.g. ``'default' → DocumentTheme``).

    Bypasses collision detection so built-in aliases can coexist with
    auto-registered subclasses without warning.
    """
    normalized = _normalize_theme_name(name)
    _registered_theme_classes[normalized] = cls
    _registered_theme_classes_sources[normalized] = f'{cls.__module__}.{cls.__qualname__}'
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


def _resolve_theme(name: str) -> Theme | None:
    """Look up a theme by name and return a usable ``Theme`` instance.

    Explicit subclass registrations win over entry-point discoveries.
    """
    normalized = _normalize_theme_name(name)

    resolved = _lookup_explicit(normalized)
    if resolved is not None:
        return resolved

    _ensure_entry_points()

    # Loading an entry point's module may have triggered __init_subclass__
    # on a Theme subclass, which populates the explicit registry. Re-check
    # before falling back to the discovered registry.
    resolved = _lookup_explicit(normalized)
    if resolved is not None:
        return resolved

    discovered = _discovered_entry_point_themes.get(normalized)
    if isinstance(discovered, type):
        return discovered()
    return discovered


def _available_theme_names() -> tuple[str, ...]:
    """Return all currently registered theme names."""
    _ensure_entry_points()
    names = set(_registered_theme_classes) | set(_discovered_entry_point_themes)
    return tuple(sorted(names))


def registered_themes() -> dict[str, ThemeRegistration]:
    """Return all registered themes, keyed by name.

    Use this to discover which names can be passed to
    :func:`~pyvista.set_plot_theme` or the ``PYVISTA_PLOT_THEME``
    environment variable. Entry-point plugins are loaded on the first
    call so they appear in the result.

    Returns
    -------
    dict[str, pyvista.ThemeRegistration]
        Mapping of theme name to a :class:`~pyvista.ThemeRegistration`
        record describing how the name was registered.

    Examples
    --------
    List available theme names.

    >>> import pyvista as pv
    >>> for name in sorted(pv.registered_themes()):
    ...     print(name)
    dark
    default
    document
    document_build
    document_pro
    paraview
    testing
    vtk

    Inspect how a name became registered.

    >>> pv.registered_themes()['dark'].kind
    'subclass'

    """
    _ensure_entry_points()
    out: dict[str, ThemeRegistration] = {}
    for name, cls in _registered_theme_classes.items():
        kind: Literal['subclass', 'alias'] = (
            'alias' if name in _registered_theme_aliases else 'subclass'
        )
        source = _registered_theme_classes_sources.get(
            name, f'{cls.__module__}.{cls.__qualname__}'
        )
        out[name] = ThemeRegistration(name=name, kind=kind, source=source)
    for name in _discovered_entry_point_themes:
        if name in out:
            # Explicit registrations always win over entry-point providers.
            continue
        out[name] = ThemeRegistration(
            name=name,
            kind='entry_point',
            source=_discovered_entry_point_sources.get(name, '<unknown>'),
        )
    return dict(sorted(out.items()))


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
    """Discover ``pyvista.themes`` entry points once."""
    global _entry_points_loaded  # noqa: PLW0603
    if _entry_points_loaded:
        return

    _entry_points_loaded = True
    for theme_entry_point in entry_points(group=THEME_ENTRY_POINT_GROUP):
        source = theme_entry_point.value
        try:
            loaded = theme_entry_point.load()
        except Exception as exc:  # noqa: BLE001
            msg = (
                f'Failed to load {THEME_ENTRY_POINT_GROUP} entry point '
                f'"{theme_entry_point.name}" from {source}: {exc}'
            )
            warn_external(msg)
            continue

        if isinstance(loaded, Mapping):
            for theme_name, theme_obj in loaded.items():
                if not isinstance(theme_name, str):
                    warn_external(
                        f'Ignoring {THEME_ENTRY_POINT_GROUP} entry point from '
                        f'{source}: theme names must be strings.',
                    )
                    continue
                _register_discovered_theme(theme_name, theme_obj, source=source)
            continue

        _register_discovered_theme(theme_entry_point.name, loaded, source=source)


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
