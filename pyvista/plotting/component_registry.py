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
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import NamedTuple
from typing import Protocol
from typing import runtime_checkable

from pyvista.core.utilities._registry import _DescriptorRegistry

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

    - ``__plotter_close__(self) -> None``—called when the plotter
      closes. Only fires on components that were actually constructed
      (i.e. accessed at least once). Use this to release VTK observers,
      close websockets, stop background threads, or otherwise undo any
      side effects the component is responsible for.
    - ``__plotter_deep_clean__(self) -> None``—called from
      ``BasePlotter.deep_clean``. Optional; if absent, deep
      clean falls through to the normal close path on the next plotter
      shutdown.

    .. versionadded:: 0.48.0

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter the component is attached to.

    Examples
    --------
    Declare a component that satisfies the protocol.

    >>> import pyvista as pv
    >>> class MyComponent:
    ...     def __init__(self, plotter):
    ...         self._plotter = plotter
    >>> isinstance(MyComponent(None), pv.PlotterComponent)
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
        The namespace the component is attached under (for example
        ``'scalar_bars'``).
    target : type
        The PyVista plotter class (or base class) the component is
        registered against.
    component : type
        The component class itself.
    source : str
        Human-readable origin in the form ``'module.qualname'`` for
        explicit registrations, or the entry-point ``value`` for plugin-
        discovered ones—useful for debugging which plugin registered
        a given component.

    """

    name: str
    target: type
    component: type
    source: str


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
    each access—slower but still correct *as long as* the component
    has no lifecycle hooks. A component that declares
    ``__plotter_close__`` or ``__plotter_deep_clean__`` cannot be
    attached to a slots target because there is no stable instance to
    track for close-time iteration; that combination raises
    :class:`TypeError` on first access rather than silently leaking
    whatever the hook was meant to release (VTK observers, websockets,
    background threads, etc.).

    Parameters
    ----------
    name : str
        Attribute name the component is registered under.

    component_cls : type
        Component class to instantiate on first access.

    """

    _LIFECYCLE_HOOKS: ClassVar[tuple[str, ...]] = (
        '__plotter_close__',
        '__plotter_deep_clean__',
    )

    def __init__(self, name: str, component_cls: type) -> None:
        self._name = name
        self._cls = component_cls

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f'<_CachedComponent name={self._name!r} cls={self._cls!r}>'

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self._cls
        component_instance = self._cls(obj)
        try:
            obj.__dict__[self._name] = component_instance
        except AttributeError:
            # ``__slots__`` target: no ``__dict__`` to cache into and no
            # stable identity across accesses. Safe to fall through
            # *only* if the component has no lifecycle hooks—otherwise
            # the hooks would never fire and any cleanup the component
            # is responsible for would silently leak.
            declared = [
                hook
                for hook in self._LIFECYCLE_HOOKS
                if getattr(self._cls, hook, None) is not None
            ]
            if declared:
                msg = (
                    f'Plotter component {self._name!r} '
                    f'({self._cls.__qualname__}) declares '
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


_registry: _DescriptorRegistry[ComponentRegistration] = _DescriptorRegistry(
    COMPONENT_ENTRY_POINT_GROUP,
    descriptor=_CachedComponent,
    record=ComponentRegistration,
    noun='plotter component',
    label='Component',
)
_plugins = _registry.plugins
_save_registry_state = _registry.snapshot
_restore_registry_state = _registry.restore


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
    normalized = _registry.validate_name(name)
    target = _resolve_default_target(target_cls)

    def decorator(component_cls: object) -> type:
        cls = _registry.validate_class(component_cls)
        _registry.attach(normalized, target, cls, override=override)
        return cls

    return decorator


def _resolve_default_target(target_cls: type | None) -> type:
    """Return the default target class when none is supplied."""
    if target_cls is None:
        # Lazy import to avoid a circular dependency at module load.
        from pyvista.plotting.plotter import BasePlotter  # noqa: PLC0415

        return BasePlotter
    return _registry.validate_target(target_cls)


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
    _registry.detach(_registry.validate_name(name), _resolve_default_target(target_cls))


_resolve_pending_component = _registry.resolve_pending
_pending_component_names = _registry.plugins.pending_keys


def registered_plotter_components() -> tuple[ComponentRegistration, ...]:
    """Return every plotter component currently registered.

    Forces discovery of any pending entry-point plugins so the returned
    list reflects every plugin visible to PyVista, not just the ones
    that have been touched already via attribute access. A plugin that
    fails to import emits a ``UserWarning`` and is skipped; the rest
    still appear in the result.

    .. versionadded:: 0.48.0

    .. versionchanged:: 0.50.0
        A plugin whose import failed stays pending and is retried on
        every call, so a component whose dependency was installed after
        the failure becomes available again.

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
    return _registry.registered()
