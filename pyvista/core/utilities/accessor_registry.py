"""Registry for third-party dataset accessors.

This module implements the pandas/xarray-style accessor pattern for PyVista
dataset classes. Plugin packages attach a namespaced accessor to a target
class at import time; the accessor is constructed lazily on first access
per dataset instance and cached on the instance.

Two registration paths are supported:

1. **Explicit import.** A plugin's module runs the
   :func:`register_dataset_accessor` decorator at import time, so any
   user that does ``import plugin`` gets the accessor attached.
2. **Entry points.** A plugin declares a ``pyvista.accessors`` entry
   point in its ``pyproject.toml`` pointing at its accessor module.
   PyVista reads the entry-point metadata lazily (no plugin module is
   imported at ``import pyvista`` time). The plugin module imports only
   when a user first accesses ``dataset.<name>`` on any dataset and the
   normal attribute lookup misses. That isolates broken or slow plugins
   from the main ``import pyvista`` path.

See Also
--------
pyvista.register_dataset_accessor
pyvista.unregister_dataset_accessor
pyvista.registered_accessors

"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import Protocol
from typing import runtime_checkable
import weakref

from pyvista import _vtk
from pyvista.core.utilities._registry import _MISSING
from pyvista.core.utilities._registry import _DescriptorRegistry

if TYPE_CHECKING:
    from collections.abc import Callable


ACCESSOR_ENTRY_POINT_GROUP = 'pyvista.accessors'


@runtime_checkable
class DataSetAccessor(Protocol):
    """Structural protocol for third-party accessor classes.

    An accessor class must accept the dataset instance as its single
    ``__init__`` argument. Any public methods on the class become
    available as ``dataset.<namespace>.<method>(...)`` once the class is
    registered with :func:`~pyvista.register_dataset_accessor`.

    This Protocol exists primarily for static type analysis: plugin
    authors can annotate ``accessor_cls: type[DataSetAccessor]`` (or
    use it as a return annotation) to have mypy/pyright check that
    their accessor class conforms. The runtime ``isinstance`` check is
    intentionally permissive (every class has ``__init__``), so prefer
    the static-typing use.

    .. versionadded:: 0.48.0

    Parameters
    ----------
    dataset : pyvista.DataSet
        Dataset the accessor is attached to.

    Examples
    --------
    Declare an accessor that satisfies the protocol.

    >>> import pyvista as pv
    >>> class MyAccessor:
    ...     def __init__(self, dataset):
    ...         self._dataset = dataset
    >>> isinstance(MyAccessor(None), pv.DataSetAccessor)
    True

    """

    def __init__(self, dataset: Any) -> None: ...


class AccessorRegistration(NamedTuple):
    """Describe one registered dataset accessor.

    Returned by :func:`~pyvista.registered_accessors`.

    .. versionadded:: 0.48.0

    Attributes
    ----------
    name : str
        The namespace the accessor is attached under (for example, ``'meshfix'``).
    target : type
        The PyVista dataset class (or base class) the accessor is
        registered against.
    accessor : type
        The accessor class itself.
    source : str
        Human-readable origin in the form ``'module.qualname'``—useful
        for debugging which plugin registered a given accessor.

    """

    name: str
    target: type
    accessor: type
    source: str


# Cached-accessor bookkeeping lives in the instance dict under this prefix. It is
# stripped from the pickled state (see ``pyvista.core.dataobject``): a weak reference
# cannot be pickled, and a cache is not state worth carrying to another process.
_ACCESSOR_CACHE_PREFIX = '_pyvista_accessor_'

# The anchor below observes an event nothing invokes -- not VTK, not PyVista. It exists
# only to hold a reference. ``UserEvent`` is VTK's documented base for ids that VTK
# itself will never fire; should another library fire this exact id, the anchor is a
# no-op callable and nothing happens.
_ACCESSOR_ANCHOR_EVENT = _vtk.vtkCommand.UserEvent + 1701


class _AccessorAnchor:
    """Hold an accessor from its dataset's observer list.

    See :class:`_CachedAccessor` for why the observer list, of all places.

    Parameters
    ----------
    accessor : object
        Accessor instance to anchor.

    """

    __slots__ = ('accessor',)

    def __init__(self, accessor: Any) -> None:
        self.accessor = accessor

    def __call__(self, *args: Any) -> None:
        """Do nothing: the event this observes is never invoked."""


class _CachedAccessor:
    """Data descriptor implementing the pandas/xarray accessor pattern.

    The first time ``obj.<name>`` is accessed, ``accessor_cls(obj)`` is constructed
    and cached for as long as ``obj`` lives, so repeated access returns the same
    accessor and ``__init__`` runs once per dataset.

    The cache is deliberately indirect: the instance ``__dict__`` holds a weak
    reference, and the accessor is kept alive by an observer on ``obj``. Caching it
    the obvious way, as ``obj.__dict__[name] = accessor``, leaks the dataset outright.
    An accessor that stores the dataset -- the documented pattern, ``self._mesh =
    mesh`` -- closes a reference cycle through the instance dict of a VTK object, and
    VTK's ``tp_traverse`` reports observers but never that dict (nor is there a
    ``tp_clear``), so the collector cannot see the cycle and neither object is ever
    freed. The observer list is the one place a VTK object's Python references *are*
    reported, so a cycle anchored there collects normally.

    When the target uses ``__slots__`` and has no ``__dict__``, or is not a
    ``vtkObject`` and so has nowhere to anchor, the accessor is constructed fresh on
    each access -- slower but still correct.

    Parameters
    ----------
    name : str
        Attribute name the accessor is registered under.

    accessor_cls : type
        Accessor class to instantiate on first access.

    """

    def __init__(self, name: str, accessor_cls: type) -> None:
        self._name = name
        self._cache_key = _ACCESSOR_CACHE_PREFIX + name
        self._cls = accessor_cls

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f'<_CachedAccessor name={self._name!r} cls={self._cls!r}>'

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            # Class-level access returns the accessor class itself so
            # ``help(cls.<name>)`` shows the accessor docstring.
            return self._cls
        obj_dict = getattr(obj, '__dict__', None)
        if obj_dict is None:
            return self._cls(obj)
        if self._name in obj_dict:
            # A value assigned over the accessor wins, as it did when this was a
            # non-data descriptor and the instance dict shadowed it.
            return obj_dict[self._name]
        cached = obj_dict.get(self._cache_key)
        if cached is not None:
            owner_id, accessor_ref, _tag = cached
            accessor = accessor_ref()
            # The id guards against an entry that reached a *copy* of ``obj``: a live
            # accessor holds its own dataset, so while the weak reference is alive that
            # dataset cannot have been freed and its id cannot have been reused.
            if accessor is not None and owner_id == id(obj):
                return accessor
        accessor = self._cls(obj)
        self._cache(obj, obj_dict, accessor)
        return accessor

    def _cache(self, obj: Any, obj_dict: dict[str, Any], accessor: Any) -> None:
        """Anchor the accessor to ``obj`` and record how to find it again."""
        add_observer = getattr(obj, 'AddObserver', None)
        if add_observer is None:
            return  # not a vtkObject: nowhere to anchor it, so do not cache
        try:
            accessor_ref = weakref.ref(accessor)
        except TypeError:
            return  # accessor class is not weak-referenceable (``__slots__``)
        tag = add_observer(_ACCESSOR_ANCHOR_EVENT, _AccessorAnchor(accessor))
        obj_dict[self._cache_key] = (id(obj), accessor_ref, tag)

    def __set__(self, obj: Any, value: Any) -> None:
        # Defining __delete__ makes this a data descriptor, which takes precedence over
        # the instance dict, so assignment has to be forwarded there explicitly to keep
        # shadowing an accessor working as it did.
        obj.__dict__[self._name] = value

    def __delete__(self, obj: Any) -> None:
        obj_dict = getattr(obj, '__dict__', None)
        deleted = False
        if obj_dict is not None:
            if obj_dict.pop(self._name, _MISSING) is not _MISSING:
                deleted = True
            cached = obj_dict.pop(self._cache_key, None)
            if cached is not None:
                obj.RemoveObserver(cached[2])  # drop the anchor, so the accessor dies
                deleted = True
        if not deleted:
            msg = self._name
            raise AttributeError(msg)


def _clear_accessor_cache(dict_: dict[str, Any]) -> None:
    """Drop cached-accessor bookkeeping from an instance dict."""
    for key in [key for key in dict_ if key.startswith(_ACCESSOR_CACHE_PREFIX)]:
        del dict_[key]


_registry: _DescriptorRegistry[AccessorRegistration] = _DescriptorRegistry(
    ACCESSOR_ENTRY_POINT_GROUP,
    descriptor=_CachedAccessor,
    record=AccessorRegistration,
    noun='accessor',
    label='Accessor',
)
_plugins = _registry.plugins
_save_registry_state = _registry.snapshot
_restore_registry_state = _registry.restore


def register_dataset_accessor(
    name: str,
    target_cls: type,
    *,
    override: bool = False,
) -> Callable[[type], type]:
    """Register a custom namespaced accessor on a PyVista dataset class.

    Mirrors the accessor pattern used by
    `pandas <https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors>`_
    and `xarray <https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`_.
    Once registered, the accessor becomes available as
    ``dataset.<name>.<method>(...)`` on every instance of ``target_cls``
    and its subclasses. The accessor class is instantiated lazily on
    first access and cached on the dataset instance, so subsequent
    accesses return the same accessor object.

    .. versionadded:: 0.48.0

    .. versionchanged:: 0.50.0
        ``override=True`` also silences the warning when replacing a
        registered accessor.

    Parameters
    ----------
    name : str
        The namespace the accessor will be attached under. Must be a
        valid Python identifier that does not start with an underscore.

    target_cls : type
        The PyVista dataset class (or base class) to attach the accessor
        to. Registering against :class:`~pyvista.DataSet` exposes the
        accessor on every subclass (``PolyData``, ``UnstructuredGrid``,
        ``ImageData``, and so on). Registering against
        :class:`~pyvista.DataObject` additionally covers
        :class:`~pyvista.MultiBlock`.

    override : bool, default: False
        If ``True``, allow registering a name that shadows an existing
        built-in attribute on ``target_cls`` (a filter method, property,
        or any other non-accessor attribute). The prior attribute is
        restored on :func:`~pyvista.unregister_dataset_accessor`. It also
        replaces a registered accessor of the same name without a warning.

    Returns
    -------
    callable
        Decorator that registers the accessor class passed to it and
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
        If ``name`` already refers to a registered accessor on
        ``target_cls`` or any ancestor. Emitting a warning rather than
        raising matches pandas' behavior and avoids hard-failing when
        two plugins collide. Pass ``override=True`` to silence the
        warning when replacement is intentional.

    See Also
    --------
    pyvista.unregister_dataset_accessor
    pyvista.registered_accessors
    pyvista.DataSetAccessor

    Examples
    --------
    Register a minimal accessor on :class:`~pyvista.PolyData`.

    >>> import pyvista as pv
    >>> @pv.register_dataset_accessor('demo', pv.PolyData)
    ... class DemoAccessor:
    ...     def __init__(self, mesh):
    ...         self._mesh = mesh
    ...
    ...     def n_points_doubled(self):
    ...         return self._mesh.n_points * 2
    >>> sphere = pv.Sphere()
    >>> sphere.demo.n_points_doubled() == sphere.n_points * 2
    True

    The accessor is cached per instance. Repeated access returns the
    same object.

    >>> sphere.demo is sphere.demo
    True

    Clean up so the namespace is not visible to later doctest examples.

    >>> pv.unregister_dataset_accessor('demo', pv.PolyData)

    """
    normalized = _registry.validate_name(name)
    target = _registry.validate_target(target_cls)

    def decorator(accessor_cls: object) -> type:
        cls = _registry.validate_class(accessor_cls)
        _registry.attach(normalized, target, cls, override=override)
        return cls

    return decorator


def unregister_dataset_accessor(name: str, target_cls: type) -> None:
    """Remove an accessor previously attached to a target class.

    The inverse of :func:`~pyvista.register_dataset_accessor`. Restores
    any built-in attribute that was shadowed via ``override=True`` when
    the accessor was registered.

    .. versionadded:: 0.48.0

    Parameters
    ----------
    name : str
        Namespace of the accessor to remove.
    target_cls : type
        Target class the accessor was registered against. Only accessors
        attached directly on ``target_cls`` can be unregistered; an
        accessor inherited from a parent class must be unregistered on
        that parent.

    Raises
    ------
    ValueError
        If no accessor named ``name`` is attached directly to
        ``target_cls``.

    TypeError
        If ``target_cls`` is not a class, or ``name`` is not a string.

    Examples
    --------
    >>> import pyvista as pv
    >>> @pv.register_dataset_accessor('tmp', pv.PolyData)
    ... class TmpAccessor:
    ...     def __init__(self, mesh):
    ...         self._mesh = mesh
    >>> pv.unregister_dataset_accessor('tmp', pv.PolyData)

    """
    _registry.detach(_registry.validate_name(name), _registry.validate_target(target_cls))


_resolve_pending_accessor = _registry.resolve_pending
_pending_accessor_names = _registry.plugins.pending_keys


def registered_accessors() -> tuple[AccessorRegistration, ...]:
    """Return every accessor currently registered.

    Forces discovery of any pending entry-point plugins so the returned
    list reflects every plugin visible to PyVista, not just the ones
    that have been touched already via attribute access. A plugin that
    fails to import emits a ``UserWarning`` and is skipped; the rest
    still appear in the result.

    .. versionadded:: 0.48.0

    .. versionchanged:: 0.49.0
        A plugin whose import failed stays pending and is retried on
        every call, so an accessor whose dependency was installed after
        the failure becomes available again.

    Returns
    -------
    tuple[AccessorRegistration, ...]
        Ordered by registration time. Each record exposes ``name``,
        ``target``, ``accessor``, and ``source``.

    Examples
    --------
    >>> import pyvista as pv
    >>> @pv.register_dataset_accessor('demo_listed', pv.PolyData)
    ... class DemoListedAccessor:
    ...     def __init__(self, mesh):
    ...         self._mesh = mesh
    >>> [r.name for r in pv.registered_accessors() if r.name == 'demo_listed']
    ['demo_listed']
    >>> pv.unregister_dataset_accessor('demo_listed', pv.PolyData)

    """
    return _registry.registered()
