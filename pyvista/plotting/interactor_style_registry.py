"""Registry for named interactor styles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyvista.core.utilities._registry import _HandlerRegistry

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    InteractorStyleHandler = Callable[..., Any]


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


def _normalize_interactor_style_name(name: str) -> str:
    """Normalize an interactor style name."""
    return name.strip().lower()


_registry: _HandlerRegistry[InteractorStyleHandler] = _HandlerRegistry(
    INTERACTOR_STYLE_ENTRY_POINT_GROUP,
    label='interactor style',
    normalize=_normalize_interactor_style_name,
    builtin=lambda name: name in _BUILTIN_INTERACTOR_STYLE_METHODS,
    builtin_label='a built-in interactor style',
    accept=lambda key, handler: (
        'collides with a built-in interactor style.'
        if key in _BUILTIN_INTERACTOR_STYLE_METHODS
        else None
        if callable(handler)
        else 'handler is not callable.'
    ),
    separate_discovered=True,
)
_custom_interactor_styles = _registry.explicit
_discovered_entry_point_styles = _registry.discovered
_plugins = _registry.plugins
_save_registry_state = _registry.snapshot
_restore_registry_state = _registry.restore


def register_interactor_style(
    name: str,
    handler: InteractorStyleHandler,
    *,
    override: bool = False,
) -> None:
    """Register a custom interactor style.

    Parameters
    ----------
    name : str
        Name of the interactor style. Built-in PyVista styles use names
        that mirror the public ``enable_*_style`` API, such as
        ``'terrain_style'``.

    handler : callable
        Handler for the interactor style. This can be a
        :vtk:`vtkInteractorStyle` subclass (instantiated automatically)
        or any callable with signature ``handler(interactor)`` that
        returns an interactor style instance.

    override : bool, default: False
        If ``True``, allow registering a name that collides with a
        built-in interactor style. Also silences the warning emitted
        when replacing an existing custom registration.

    Raises
    ------
    ValueError
        If ``name`` is empty, or collides with a built-in PyVista
        interactor style and ``override`` is ``False``.

    Warns
    -----
    UserWarning
        If ``name`` already refers to a registered custom interactor
        style. The new registration replaces the old one (last wins);
        pass ``override=True`` to silence the warning.

    Examples
    --------
    Register a :vtk:`vtkInteractorStyle` subclass directly and use
    it from a theme.

    >>> import pyvista as pv
    >>> from vtkmodules.vtkInteractionStyle import (
    ...     vtkInteractorStyleFlight,
    ... )
    >>> pv.register_interactor_style(
    ...     'flight_style', vtkInteractorStyleFlight
    ... )  # doctest: +SKIP
    >>> pv.global_theme.interactor_style = 'flight_style'  # doctest: +SKIP

    A callable that takes the interactor as an argument can also be
    registered.

    >>> def custom_style(interactor): ...
    >>> pv.register_interactor_style(
    ...     'custom_style', custom_style
    ... )  # doctest: +SKIP

    """
    if not _normalize_interactor_style_name(name):
        msg = 'Interactor style name must not be empty.'
        raise ValueError(msg)
    _registry.register(name, handler, override=override)


def _get_interactor_style_handler(name: str) -> str | InteractorStyleHandler | None:
    """Look up a registered interactor style by name."""
    normalized_name = _normalize_interactor_style_name(name)
    handler = _custom_interactor_styles.get(normalized_name)
    if handler is not None:
        return handler

    builtin_method = _BUILTIN_INTERACTOR_STYLE_METHODS.get(normalized_name)
    if builtin_method is not None:
        return builtin_method

    _plugins.ensure_loaded()
    return _registry.get(normalized_name)


def _available_interactor_style_names() -> tuple[str, ...]:
    """Return all currently available interactor style names."""
    _plugins.ensure_loaded()
    return tuple(sorted(set(_BUILTIN_INTERACTOR_STYLE_METHODS) | _registry.keys()))


def _has_interactor_style(name: str) -> bool:
    """Return ``True`` when an interactor style name is available."""
    normalized_name = _normalize_interactor_style_name(name)
    if normalized_name in _custom_interactor_styles:
        return True
    if normalized_name in _BUILTIN_INTERACTOR_STYLE_METHODS:
        return True

    _plugins.ensure_loaded()
    return _registry.get(normalized_name) is not None


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
