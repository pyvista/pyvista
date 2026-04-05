"""Registry for named interactor styles."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import entry_points
from typing import TYPE_CHECKING
from typing import TypedDict

from pyvista._warn_external import warn_external

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    InteractorStyleFactory = Callable[..., Any]


class _InteractorStyleRegistryState(TypedDict):
    """Stored registry state used by tests."""

    custom: dict[str, InteractorStyleFactory]
    discovered: dict[str, InteractorStyleFactory]
    discovered_sources: dict[str, str]
    loaded: bool


INTERACTOR_STYLE_ENTRY_POINT_GROUP = 'pyvista.interactor_styles'

_BUILTIN_INTERACTOR_STYLE_METHODS = {
    '2d_style': 'enable_2d_style',
    'custom_trackball_style': 'enable_custom_trackball_style',
    'image_style': 'enable_image_style',
    'joystick_actor_style': 'enable_joystick_actor_style',
    'joystick_style': 'enable_joystick_style',
    'rubber_band_2d_style': 'enable_rubber_band_2d_style',
    'rubber_band_style': 'enable_rubber_band_style',
    'terrain_style': 'enable_terrain_style',
    'trackball_actor_style': 'enable_trackball_actor_style',
    'trackball_style': 'enable_trackball_style',
    'zoom_style': 'enable_zoom_style',
}

_custom_interactor_styles: dict[str, InteractorStyleFactory] = {}
_discovered_entry_point_styles: dict[str, InteractorStyleFactory] = {}
_discovered_entry_point_sources: dict[str, str] = {}
_entry_points_loaded: bool = False


def _normalize_interactor_style_name(name: str) -> str:
    """Normalize an interactor style name."""
    return name.strip().lower()


def _save_registry_state() -> _InteractorStyleRegistryState:
    """Snapshot the current registry state for later restoration."""
    return {
        'custom': _custom_interactor_styles.copy(),
        'discovered': _discovered_entry_point_styles.copy(),
        'discovered_sources': _discovered_entry_point_sources.copy(),
        'loaded': _entry_points_loaded,
    }


def _restore_registry_state(state: _InteractorStyleRegistryState) -> None:
    """Restore registry state from a snapshot."""
    global _entry_points_loaded  # noqa: PLW0603
    _custom_interactor_styles.clear()
    _custom_interactor_styles.update(state['custom'])
    _discovered_entry_point_styles.clear()
    _discovered_entry_point_styles.update(state['discovered'])
    _discovered_entry_point_sources.clear()
    _discovered_entry_point_sources.update(state['discovered_sources'])
    _entry_points_loaded = state['loaded']


def register_interactor_style(name: str, handler: InteractorStyleFactory) -> None:
    """Register a custom interactor style.

    Parameters
    ----------
    name : str
        Name of the interactor style. Built-in PyVista styles use names
        that mirror the public ``enable_*_style`` API, such as
        ``'terrain_style'``.

    handler : callable
        Factory for the interactor style. This can be a
        :vtk:`vtkInteractorStyle` subclass or any callable with
        signature ``handler(interactor)`` that returns an interactor
        style instance.

    Raises
    ------
    ValueError
        If ``name`` is empty or collides with a built-in PyVista
        interactor style.

    Examples
    --------
    Register a custom interactor style and use it from a theme.

    >>> import pyvista as pv
    >>> def custom_style(interactor): ...
    >>> pv.register_interactor_style(
    ...     'custom_style', custom_style
    ... )  # doctest: +SKIP
    >>> pv.global_theme.interactor_style = 'custom_style'  # doctest: +SKIP

    """
    normalized_name = _normalize_interactor_style_name(name)
    if not normalized_name:
        msg = 'Interactor style name must not be empty.'
        raise ValueError(msg)
    if normalized_name in _BUILTIN_INTERACTOR_STYLE_METHODS:
        msg = (
            f'Cannot register custom interactor style "{normalized_name}": '
            'collides with a built-in interactor style.'
        )
        raise ValueError(msg)

    _custom_interactor_styles[normalized_name] = handler


def _get_interactor_style_handler(name: str) -> str | InteractorStyleFactory | None:
    """Look up a registered interactor style by name."""
    normalized_name = _normalize_interactor_style_name(name)
    handler = _custom_interactor_styles.get(normalized_name)
    if handler is not None:
        return handler

    builtin_method = _BUILTIN_INTERACTOR_STYLE_METHODS.get(normalized_name)
    if builtin_method is not None:
        return builtin_method

    _ensure_entry_points()
    return _discovered_entry_point_styles.get(normalized_name)


def _available_interactor_style_names() -> tuple[str, ...]:
    """Return all currently available interactor style names."""
    _ensure_entry_points()
    names = (
        set(_BUILTIN_INTERACTOR_STYLE_METHODS)
        | set(_custom_interactor_styles)
        | set(_discovered_entry_point_styles)
    )
    return tuple(sorted(names))


def _has_interactor_style(name: str) -> bool:
    """Return ``True`` when an interactor style name is available."""
    normalized_name = _normalize_interactor_style_name(name)
    if normalized_name in _custom_interactor_styles:
        return True
    if normalized_name in _BUILTIN_INTERACTOR_STYLE_METHODS:
        return True

    _ensure_entry_points()
    return normalized_name in _discovered_entry_point_styles


def _validate_interactor_style(name: object) -> str:
    """Validate and normalize an interactor style name."""
    if not isinstance(name, str):
        msg = f'Interactor style must be a string, not {type(name).__name__}.'
        raise TypeError(msg)

    normalized_name = _normalize_interactor_style_name(name)
    if _has_interactor_style(normalized_name):
        return normalized_name

    available_styles = ', '.join(f'"{item}"' for item in _available_interactor_style_names())
    msg = f'Invalid interactor style "{name}".\nUse one of the following:\n{available_styles}'
    raise ValueError(msg)


def _ensure_entry_points() -> None:
    """Discover ``pyvista.interactor_styles`` entry points once."""
    global _entry_points_loaded  # noqa: PLW0603
    if _entry_points_loaded:
        return

    _entry_points_loaded = True
    for interactor_style_entry_point in entry_points(group=INTERACTOR_STYLE_ENTRY_POINT_GROUP):
        source = interactor_style_entry_point.value
        try:
            loaded_handler = interactor_style_entry_point.load()
        except Exception as exc:  # noqa: BLE001
            msg = (
                f'Failed to load {INTERACTOR_STYLE_ENTRY_POINT_GROUP} entry point '
                f'"{interactor_style_entry_point.name}" from {source}: {exc}'
            )
            warn_external(msg)
            continue

        if isinstance(loaded_handler, Mapping):
            for style_name, style_handler in loaded_handler.items():
                if not isinstance(style_name, str):
                    msg = (
                        f'Ignoring {INTERACTOR_STYLE_ENTRY_POINT_GROUP} entry point from '
                        f'{source}: style names must be strings.'
                    )
                    warn_external(msg)
                    continue
                _register_discovered_interactor_style(
                    style_name,
                    style_handler,
                    source=source,
                )
            continue

        _register_discovered_interactor_style(
            interactor_style_entry_point.name,
            loaded_handler,
            source=source,
        )


def _register_discovered_interactor_style(
    name: str,
    handler: object,
    *,
    source: str,
) -> None:
    """Register a discovered interactor style handler from an entry point."""
    normalized_name = _normalize_interactor_style_name(name)
    if not normalized_name:
        msg = (
            f'Ignoring {INTERACTOR_STYLE_ENTRY_POINT_GROUP} entry point from '
            f'{source}: style name must not be empty.'
        )
        warn_external(msg)
        return

    if normalized_name in _BUILTIN_INTERACTOR_STYLE_METHODS:
        msg = (
            f'Ignoring {INTERACTOR_STYLE_ENTRY_POINT_GROUP} entry point '
            f'"{normalized_name}" from {source}: collides with a built-in interactor style.'
        )
        warn_external(msg)
        return

    if normalized_name in _custom_interactor_styles:
        return

    if normalized_name in _discovered_entry_point_styles:
        existing_source = _discovered_entry_point_sources[normalized_name]
        msg = (
            f'Multiple {INTERACTOR_STYLE_ENTRY_POINT_GROUP} providers registered for '
            f'"{normalized_name}": {existing_source}, {source}. Using {existing_source}.'
        )
        warn_external(msg)
        return

    if not callable(handler):
        msg = (
            f'Ignoring {INTERACTOR_STYLE_ENTRY_POINT_GROUP} entry point '
            f'"{normalized_name}" from {source}: handler is not callable.'
        )
        warn_external(msg)
        return

    _discovered_entry_point_styles[normalized_name] = handler
    _discovered_entry_point_sources[normalized_name] = source
