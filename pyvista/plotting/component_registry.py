"""Registry for plotter components.

This module implements the namespaced *component* extension point on
:class:`pyvista.BasePlotter`. It mirrors the dataset accessor pattern in
:mod:`pyvista.core.utilities.accessor_registry` line for line on the
registration API and adds two optional lifecycle hooks
(``__plotter_close__`` and ``__plotter_deep_clean__``) so plugin authors
can release VTK observers, websockets, background threads, and any other
state that needs explicit teardown when the plotter shuts down.

Plugin authors learn one decorator shape and one mental model: the dataset
accessor decorator. The component variant adds two optional dunder hooks
for cleanup. Everything else aligns.

Two registration paths are supported:

1. **Explicit import.** A plugin's module runs the
   :func:`register_plotter_component` decorator at import time, so any
   user that does ``import plugin`` gets the component attached.
2. **Entry points.** A plugin declares a ``pyvista.plotter_components``
   entry point in its ``pyproject.toml`` pointing at its component
   module. PyVista reads the entry-point metadata lazily (no plugin
   module is imported at ``import pyvista`` time). The plugin module
   imports only when a user first accesses ``plotter.<name>`` on any
   plotter and the normal attribute lookup misses.

See Also
--------
pyvista.register_plotter_component
pyvista.unregister_plotter_component
pyvista.registered_plotter_components

"""

from __future__ import annotations

import contextlib
from importlib import import_module
from importlib.metadata import entry_points
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import NamedTuple
from typing import Protocol
from typing import TypedDict
from typing import runtime_checkable

from pyvista._warn_external import warn_external
from pyvista.core.utilities._registry_helpers import handler_source

if TYPE_CHECKING:
    from collections.abc import Callable


COMPONENT_ENTRY_POINT_GROUP = 'pyvista.plotter_components'


@runtime_checkable
class PlotterComponent(Protocol):
    """Structural protocol for plotter component classes.

    A component class must accept the plotter instance as its single
    ``__init__`` argument. Any public methods on the class become
    available as ``plotter.<namespace>.<method>(...)`` once the class is
    registered with :func:`~pyvista.register_plotter_component`.

    Two optional dunder hooks participate in the plotter lifecycle and
    are detected by ``getattr`` at lifecycle time, not by the protocol
    (since they are optional):

    - ``__plotter_close__(self) -> None`` — called when the plotter
      closes. Only fires on components that were actually constructed
      (i.e. accessed at least once). Use this to release VTK observers,
      close websockets, stop background threads, or otherwise undo any
      side effects the component is responsible for.
    - ``__plotter_deep_clean__(self) -> None`` — called from
      ``BasePlotter.deep_clean``. Optional; if absent, deep
      clean falls through to the normal close path on the next plotter
      shutdown.

    .. versionadded:: 0.48.0

    Examples
    --------
    Declare a component that satisfies the protocol.

    >>> from pyvista import PlotterComponent
    >>> class MyComponent:
    ...     def __init__(self, plotter):
    ...         self._plotter = plotter
    >>> isinstance(MyComponent(None), PlotterComponent)
    True

    """

    def __init__(self, plotter: Any) -> None: ...


class ComponentRegistration(NamedTuple):
    """Describe one registered plotter component.

    Returned by :func:`~pyvista.registered_plotter_components`.

    .. versionadded:: 0.48.0

    Attributes
    ----------
    name : str
        The namespace the component is attached under (e.g.
        ``'scalar_bars'``).
    target : type
        The PyVista plotter class (or base class) the component is
        registered against.
    component : type
        The component class itself.
    source : str
        Human-readable origin in the form ``'module.qualname'`` for
        explicit registrations, or the entry-point ``value`` for plugin-
        discovered ones — useful for debugging which plugin registered
        a given component.

    """

    name: str
    target: type
    component: type
    source: str


class _ComponentRegistryState(TypedDict):
    """Stored registry state used for test isolation.

    Records every ``(target_cls, name)`` the registry has touched along
    with the value that was present on the class dictionary before the
    component descriptor replaced it (sentinel if the attribute did not
    exist).
    """

    registrations: list[ComponentRegistration]
    prior: dict[tuple[type, str], Any]
    attached: dict[tuple[type, str], Any]
    entry_points_loaded: bool
    pending: dict[str, str]


class _CachedComponent:
    """Non-data descriptor implementing the lazy-component pattern.

    The first time ``obj.<name>`` is accessed, ``component_cls(obj)`` is
    constructed and stored in ``obj.__dict__[name]``. The component is
    also appended to ``obj._components`` for close-time iteration.
    Subsequent lookups bypass the descriptor and hit ``__dict__``
    directly because a non-data descriptor yields to instance
    ``__dict__``.

    The ``obj.__dict__`` write goes through the dictionary directly so
    it bypasses :class:`pyvista.core.utilities.misc._NoNewAttrMixin`'s
    ``__setattr__`` freeze without weakening the freeze for any other
    attribute. When the target class uses ``__slots__`` and no
    ``__dict__`` is available, the component is constructed fresh on
    each access — slower but still correct *as long as* the component
    has no lifecycle hooks. A component that declares
    ``__plotter_close__`` or ``__plotter_deep_clean__`` cannot be
    attached to a slots target because there is no stable instance to
    track for close-time iteration; that combination raises
    :class:`TypeError` on first access rather than silently leaking
    whatever the hook was meant to release (VTK observers, websockets,
    background threads, etc.).
    """

    _LIFECYCLE_HOOKS: ClassVar[tuple[str, ...]] = (
        '__plotter_close__',
        '__plotter_deep_clean__',
    )

    def __init__(self, name: str, component_cls: type) -> None:
        self._name = name
        self._component_cls = component_cls

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f'<_CachedComponent name={self._name!r} cls={self._component_cls!r}>'

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self._component_cls
        component_instance = self._component_cls(obj)
        try:
            obj.__dict__[self._name] = component_instance
        except AttributeError:
            # ``__slots__`` target: no ``__dict__`` to cache into and no
            # stable identity across accesses. Safe to fall through
            # *only* if the component has no lifecycle hooks — otherwise
            # the hooks would never fire and any cleanup the component
            # is responsible for would silently leak.
            declared = [
                hook
                for hook in self._LIFECYCLE_HOOKS
                if getattr(self._component_cls, hook, None) is not None
            ]
            if declared:
                msg = (
                    f'Plotter component {self._name!r} '
                    f'({self._component_cls.__qualname__}) declares '
                    f'lifecycle hook(s) {declared} but is attached to '
                    f'{type(obj).__qualname__}, which uses ``__slots__`` '
                    'and has no ``__dict__`` to cache the component '
                    "instance in. Lifecycle hooks can't fire on a slots "
                    'target because each attribute access constructs a '
                    'fresh component, so close-time teardown would have '
                    'no stable receiver. Either drop the lifecycle '
                    'hooks or give the target class a ``__dict__`` slot.'
                )
                raise TypeError(msg)
            return component_instance
        # Track for close / deep-clean iteration. ``_components`` is
        # initialized on the plotter in ``BasePlotter.__init__``; if a
        # subclass forgot to chain through, fall back to creating the
        # list lazily so the descriptor cannot crash bare attribute
        # access.
        components = getattr(obj, '_components', None)
        if components is None:
            with contextlib.suppress(AttributeError):
                obj.__dict__['_components'] = []
                components = obj.__dict__['_components']
        if components is not None:
            components.append(component_instance)
        return component_instance


_MISSING: Any = object()

_registrations: list[ComponentRegistration] = []
_prior_values: dict[tuple[type, str], Any] = {}
_entry_points_loaded: bool = False
_pending_components: dict[str, str] = {}


def _save_registry_state() -> _ComponentRegistryState:
    """Snapshot the current registry state for later restoration."""
    attached: dict[tuple[type, str], Any] = {}
    for record in _registrations:
        key = (record.target, record.name)
        attached[key] = record.target.__dict__.get(record.name, _MISSING)
    return {
        'registrations': list(_registrations),
        'prior': dict(_prior_values),
        'attached': attached,
        'entry_points_loaded': _entry_points_loaded,
        'pending': dict(_pending_components),
    }


def _restore_registry_state(state: _ComponentRegistryState) -> None:
    """Restore registry state from a snapshot.

    Removes any components that have been added since the snapshot was
    taken and re-attaches any that were removed, putting prior built-in
    attributes back if override had shadowed them.
    """
    global _entry_points_loaded  # noqa: PLW0603
    current_keys = {(r.target, r.name) for r in _registrations}
    snapshot_keys = {(r.target, r.name) for r in state['registrations']}

    for record in _registrations:
        key = (record.target, record.name)
        if key in snapshot_keys:
            continue
        attr = record.target.__dict__.get(record.name)
        if isinstance(attr, _CachedComponent):
            delattr(record.target, record.name)
        prior = _prior_values.pop(key, _MISSING)
        if prior is not _MISSING:
            setattr(record.target, record.name, prior)

    for record in state['registrations']:
        key = (record.target, record.name)
        attached_at_snapshot = state['attached'].get(key, _MISSING)
        if attached_at_snapshot is _MISSING:
            if record.name in record.target.__dict__:
                delattr(record.target, record.name)
        elif (
            key not in current_keys
            or record.target.__dict__.get(record.name) is not attached_at_snapshot
        ):
            setattr(record.target, record.name, attached_at_snapshot)

    _registrations.clear()
    _registrations.extend(state['registrations'])
    _prior_values.clear()
    _prior_values.update(state['prior'])
    _entry_points_loaded = state['entry_points_loaded']
    _pending_components.clear()
    _pending_components.update(state['pending'])


def _find_component_on_mro(target_cls: type, name: str) -> type | None:
    """Return the class whose ``__dict__`` owns the component descriptor."""
    for klass in target_cls.__mro__:
        attr = klass.__dict__.get(name)
        if isinstance(attr, _CachedComponent):
            return klass
    return None


def _find_shadow_on_mro(target_cls: type, name: str) -> type | None:
    """Return the class shadowing ``name`` with a non-component attribute.

    Walks ``target_cls.__mro__`` (skipping ``object``) and returns the
    first class that defines a non-component attribute under ``name``,
    or ``None`` if none exists. Used to detect built-in attribute
    shadowing.
    """
    for klass in target_cls.__mro__:
        if klass is object:
            continue
        if name not in klass.__dict__:
            continue
        attr = klass.__dict__[name]
        if isinstance(attr, _CachedComponent):
            return None
        return klass
    return None


def _validate_name(name: object) -> str:
    """Normalize and validate a component name."""
    if not isinstance(name, str):
        msg = f'Component name must be a string, got {type(name).__name__}.'
        raise TypeError(msg)
    normalized = name.strip()
    if not normalized:
        msg = 'Component name must not be empty.'
        raise ValueError(msg)
    if normalized.startswith('_'):
        msg = (
            f'Component name must not start with an underscore, got {name!r}. '
            'Leading underscores are reserved for PyVista internals.'
        )
        raise ValueError(msg)
    if not normalized.isidentifier():
        msg = f'Component name must be a valid Python identifier, got {name!r}.'
        raise ValueError(msg)
    return normalized


def _validate_target(target_cls: Any) -> type:
    """Validate that ``target_cls`` is a class."""
    if not isinstance(target_cls, type):
        msg = f'target_cls must be a class, got {type(target_cls).__name__}.'
        raise TypeError(msg)
    return target_cls


def register_plotter_component(
    name: str,
    target_cls: type | None = None,
    *,
    override: bool = False,
) -> Callable[[type], type]:
    """Register a custom namespaced component on a PyVista plotter class.

    Once registered, the component becomes available as
    ``plotter.<name>.<method>(...)`` on every instance of ``target_cls``
    and its subclasses. The component class is instantiated lazily on
    first access and cached on the plotter instance, so subsequent
    accesses return the same component object.

    Components participate in the plotter lifecycle: if the class
    defines ``__plotter_close__`` it is invoked when the plotter
    closes; if it defines ``__plotter_deep_clean__`` it is invoked from
    ``BasePlotter.deep_clean``. Both hooks fire only on
    components that were actually constructed (touched at least once).

    Mirrors :func:`~pyvista.register_dataset_accessor` on the
    registration surface so plugin authors learn one decorator pattern.

    .. versionadded:: 0.48.0

    Parameters
    ----------
    name : str
        The namespace the component will be attached under. Must be a
        valid Python identifier that does not start with an underscore.

    target_cls : type, optional
        The plotter class (or base class) to attach the component to.
        Defaults to :class:`pyvista.BasePlotter`, which exposes the
        component on every plotter subclass (including
        ``pyvistaqt.QtInteractor``).

    override : bool, default: False
        If ``True``, allow registering a name that shadows an existing
        built-in attribute on ``target_cls`` (a method, property, or any
        other non-component attribute). The prior attribute is restored
        on :func:`~pyvista.unregister_plotter_component`.

    Returns
    -------
    callable
        Decorator that registers the component class passed to it and
        returns that class unchanged.

    Raises
    ------
    ValueError
        If ``name`` is empty, not a valid identifier, starts with ``_``,
        or shadows a built-in attribute on ``target_cls`` without
        ``override=True``.

    TypeError
        If ``target_cls`` is not a class, or ``name`` is not a string.

    Warns
    -----
    UserWarning
        If ``name`` already refers to a registered component on
        ``target_cls`` or any ancestor. Pass ``override=True`` to
        silence the warning when replacement is intentional.

    See Also
    --------
    pyvista.unregister_plotter_component
    pyvista.registered_plotter_components
    pyvista.PlotterComponent

    Examples
    --------
    Register a minimal component on the plotter.

    >>> import pyvista as pv
    >>> @pv.register_plotter_component('demo')
    ... class DemoComponent:
    ...     def __init__(self, plotter):
    ...         self._plotter = plotter
    ...
    ...     def title(self):
    ...         return self._plotter.title
    >>> pl = pv.Plotter()
    >>> pl.demo.title() == pl.title
    True

    Components are cached per plotter. Repeated access returns the same
    object.

    >>> pl.demo is pl.demo
    True

    Clean up so the namespace is not visible to later doctest examples.

    >>> pv.unregister_plotter_component('demo')
    >>> pl.close()

    """
    normalized = _validate_name(name)
    target = _resolve_default_target(target_cls)

    def decorator(component_cls: object) -> type:
        if not isinstance(component_cls, type):
            msg = f'Component must be a class, got {type(component_cls).__name__}.'
            raise TypeError(msg)
        _attach_component(normalized, target, component_cls, override=override)
        return component_cls

    return decorator


def _resolve_default_target(target_cls: type | None) -> type:
    """Return the default target class when none is supplied."""
    if target_cls is None:
        # Lazy import to avoid a circular dependency at module load.
        from pyvista.plotting.plotter import BasePlotter  # noqa: PLC0415

        return BasePlotter
    return _validate_target(target_cls)


def _attach_component(
    name: str,
    target_cls: type,
    component_cls: type,
    *,
    override: bool,
    source: str | None = None,
) -> None:
    """Attach ``component_cls`` as a ``_CachedComponent`` on ``target_cls``."""
    component_owner = _find_component_on_mro(target_cls, name)
    if component_owner is not None:
        if not override:
            if component_owner is target_cls:
                location = target_cls.__qualname__
            else:
                location = (
                    f'{component_owner.__qualname__} (inherited by {target_cls.__qualname__})'
                )
            warn_external(
                f'Registering plotter component {name!r} on '
                f'{target_cls.__qualname__} replaces an existing registered '
                f'component on {location}.',
            )
    else:
        shadow_owner = _find_shadow_on_mro(target_cls, name)
        if shadow_owner is not None and not override:
            if shadow_owner is target_cls:
                location = target_cls.__qualname__
            else:
                location = f'{shadow_owner.__qualname__} (inherited by {target_cls.__qualname__})'
            msg = (
                f'Cannot register plotter component {name!r} on '
                f'{target_cls.__qualname__}: shadows built-in attribute on '
                f'{location}. Pass override=True to force.'
            )
            raise ValueError(msg)

    key = (target_cls, name)
    if key not in _prior_values:
        _prior_values[key] = target_cls.__dict__.get(name, _MISSING)

    setattr(target_cls, name, _CachedComponent(name, component_cls))

    resolved_source = source if source is not None else handler_source(component_cls)
    _registrations[:] = [
        r for r in _registrations if not (r.target is target_cls and r.name == name)
    ]
    _registrations.append(
        ComponentRegistration(
            name=name,
            target=target_cls,
            component=component_cls,
            source=resolved_source,
        ),
    )


def unregister_plotter_component(name: str, target_cls: type | None = None) -> None:
    """Remove a component previously attached to a plotter class.

    The inverse of :func:`~pyvista.register_plotter_component`. Restores
    any built-in attribute that was shadowed via ``override=True`` when
    the component was registered.

    .. versionadded:: 0.48.0

    Parameters
    ----------
    name : str
        Namespace of the component to remove.
    target_cls : type, optional
        Target class the component was registered against. Defaults to
        :class:`pyvista.BasePlotter`. Only components attached directly
        on ``target_cls`` can be unregistered; a component inherited
        from a parent class must be unregistered on that parent.

    Raises
    ------
    ValueError
        If no component named ``name`` is attached directly to
        ``target_cls``.

    TypeError
        If ``target_cls`` is not a class, or ``name`` is not a string.

    Examples
    --------
    >>> import pyvista as pv
    >>> @pv.register_plotter_component('tmp')
    ... class TmpComponent:
    ...     def __init__(self, plotter):
    ...         self._plotter = plotter
    >>> pv.unregister_plotter_component('tmp')

    """
    normalized = _validate_name(name)
    target = _resolve_default_target(target_cls)

    attr = target.__dict__.get(normalized)
    if not isinstance(attr, _CachedComponent):
        msg = (
            f'No registered plotter component {name!r} attached directly to {target.__qualname__}.'
        )
        raise ValueError(msg)  # noqa: TRY004

    key = (target, normalized)
    prior = _prior_values.pop(key, _MISSING)
    if prior is _MISSING:
        delattr(target, normalized)
    else:
        setattr(target, normalized, prior)

    _registrations[:] = [
        r for r in _registrations if not (r.target is target and r.name == normalized)
    ]


def _ensure_entry_points() -> None:
    """Scan ``pyvista.plotter_components`` entry-point metadata once."""
    global _entry_points_loaded  # noqa: PLW0603
    if _entry_points_loaded:
        return

    _entry_points_loaded = True
    for component_entry_point in entry_points(group=COMPONENT_ENTRY_POINT_GROUP):
        _pending_components[component_entry_point.name] = component_entry_point.value


def _resolve_pending_component(name: str) -> bool:
    """Import the plugin module registered under ``name``, if any.

    Called from :meth:`pyvista.BasePlotter.__getattr__` when a normal
    attribute lookup misses. Ensures entry-point metadata has been
    scanned, pops the pending entry for ``name``, and imports the
    corresponding plugin module. Importing the module triggers any
    ``@register_plotter_component`` decorators inside it and attaches
    the component as a side effect.

    Returns
    -------
    bool
        ``True`` if a plugin was loaded for ``name`` (and the attribute
        lookup should be retried). ``False`` if no pending plugin
        matches ``name``, or if the plugin failed to import.

    """
    _ensure_entry_points()
    module_path = _pending_components.pop(name, None)
    if module_path is None:
        return False
    try:
        import_module(module_path)
    except Exception as exc:  # noqa: BLE001
        msg = (
            f'Failed to load {COMPONENT_ENTRY_POINT_GROUP} entry point '
            f'"{name}" from {module_path}: {exc}'
        )
        warn_external(msg)
        return False
    return True


def _pending_component_names() -> tuple[str, ...]:
    """Return every pending component name without importing any plugin."""
    _ensure_entry_points()
    return tuple(_pending_components)


def registered_plotter_components() -> tuple[ComponentRegistration, ...]:
    """Return every plotter component currently registered.

    Forces discovery of any pending entry-point plugins so the returned
    list reflects every plugin visible to PyVista, not just the ones
    that have been touched already via attribute access. A plugin that
    fails to import emits a ``UserWarning`` and is skipped; the rest
    still appear in the result.

    .. versionadded:: 0.48.0

    Returns
    -------
    tuple[ComponentRegistration, ...]
        Ordered by registration time. Each record exposes ``name``,
        ``target``, ``component``, and ``source``.

    Examples
    --------
    >>> import pyvista as pv
    >>> @pv.register_plotter_component('demo_listed')
    ... class DemoListedComponent:
    ...     def __init__(self, plotter):
    ...         self._plotter = plotter
    >>> [
    ...     r.name
    ...     for r in pv.registered_plotter_components()
    ...     if r.name == 'demo_listed'
    ... ]
    ['demo_listed']
    >>> pv.unregister_plotter_component('demo_listed')

    """
    _ensure_entry_points()
    for pending_name in list(_pending_components):
        _resolve_pending_component(pending_name)
    return tuple(_registrations)
