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

import contextlib
from importlib import import_module
from importlib.metadata import entry_points
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import Protocol
from typing import TypedDict
from typing import runtime_checkable

from pyvista._warn_external import warn_external

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

    Examples
    --------
    Declare an accessor that satisfies the protocol.

    >>> from pyvista import DataSetAccessor
    >>> class MyAccessor:
    ...     def __init__(self, dataset):
    ...         self._dataset = dataset
    >>> isinstance(MyAccessor(None), DataSetAccessor)
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
        The namespace the accessor is attached under (e.g. ``'meshfix'``).
    target : type
        The PyVista dataset class (or base class) the accessor is
        registered against.
    accessor : type
        The accessor class itself.
    source : str
        Human-readable origin in the form ``'module.qualname'`` — useful
        for debugging which plugin registered a given accessor.

    """

    name: str
    target: type
    accessor: type
    source: str


class _AccessorRegistryState(TypedDict):
    """Stored registry state used for test isolation.

    Records every ``(target_cls, name)`` the registry has touched along
    with the value that was present on the class dictionary before the
    accessor replaced it (sentinel if the attribute did not exist).
    """

    registrations: list[AccessorRegistration]
    # Per (target, name), the exact attribute that was in target.__dict__
    # before registration, or _MISSING if no attribute existed.
    prior: dict[tuple[type, str], Any]
    # Snapshot of each target class's current binding for each registered
    # name so restore can diff against "live" state.
    attached: dict[tuple[type, str], Any]
    # Whether entry-point metadata has already been scanned. Tests that
    # mock ``entry_points`` reset this to force a re-scan.
    entry_points_loaded: bool
    # Accessor names declared via entry points that have not yet had
    # their plugin module imported. Populated by ``_ensure_entry_points``
    # and consumed by ``_resolve_pending_accessor`` on first attribute
    # miss.
    pending: dict[str, str]


class _CachedAccessor:
    """Non-data descriptor implementing the pandas/xarray accessor pattern.

    The first time ``obj.<name>`` is accessed, ``accessor_cls(obj)`` is
    constructed and stored in ``obj.__dict__[name]``. Subsequent lookups
    bypass the descriptor and hit ``__dict__`` directly because a
    non-data descriptor yields to instance ``__dict__``.

    When the target class uses ``__slots__`` and no ``__dict__`` is
    available, the accessor is constructed fresh on each access — slower
    but still correct.
    """

    def __init__(self, name: str, accessor_cls: type) -> None:
        self._name = name
        self._accessor_cls = accessor_cls

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f'<_CachedAccessor name={self._name!r} cls={self._accessor_cls!r}>'

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            # Class-level access returns the accessor class itself so
            # ``help(cls.<name>)`` shows the accessor docstring.
            return self._accessor_cls
        accessor_instance = self._accessor_cls(obj)
        # Skip the cache on ``__slots__`` targets (no ``__dict__``) and
        # construct fresh on each access.
        with contextlib.suppress(AttributeError):
            obj.__dict__[self._name] = accessor_instance
        return accessor_instance


# ``_MISSING`` marks "attribute did not exist on the target class dict
# before we touched it" so unregister can tell the difference between
# "restore nothing" and "restore a prior value of None".
_MISSING: Any = object()

_registrations: list[AccessorRegistration] = []
_prior_values: dict[tuple[type, str], Any] = {}
_entry_points_loaded: bool = False
_pending_accessors: dict[str, str] = {}


def _save_registry_state() -> _AccessorRegistryState:
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
        'pending': dict(_pending_accessors),
    }


def _restore_registry_state(state: _AccessorRegistryState) -> None:
    """Restore registry state from a snapshot.

    Removes any accessors that have been added since the snapshot was
    taken and re-attaches any that were removed, putting prior built-in
    attributes back if override had shadowed them.
    """
    global _entry_points_loaded  # noqa: PLW0603
    current_keys = {(r.target, r.name) for r in _registrations}
    snapshot_keys = {(r.target, r.name) for r in state['registrations']}

    # Remove anything that was not in the snapshot.
    for record in _registrations:
        key = (record.target, record.name)
        if key in snapshot_keys:
            continue
        attr = record.target.__dict__.get(record.name)
        if isinstance(attr, _CachedAccessor):
            delattr(record.target, record.name)
        prior = _prior_values.pop(key, _MISSING)
        if prior is not _MISSING:
            setattr(record.target, record.name, prior)

    # Re-add anything that was in the snapshot but has since been
    # removed, restoring the descriptor binding that was live at
    # snapshot time.
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
    _pending_accessors.clear()
    _pending_accessors.update(state['pending'])


def _find_accessor_on_mro(target_cls: type, name: str) -> type | None:
    """Return the class whose ``__dict__`` owns the accessor.

    Walks ``target_cls.__mro__`` and returns the first class whose
    ``__dict__`` holds a :class:`_CachedAccessor` under ``name``, or
    ``None`` if none exists.
    """
    for klass in target_cls.__mro__:
        attr = klass.__dict__.get(name)
        if isinstance(attr, _CachedAccessor):
            return klass
    return None


def _find_shadow_on_mro(target_cls: type, name: str) -> type | None:
    """Return the class shadowing ``name`` with a non-accessor attribute.

    Walks ``target_cls.__mro__`` (skipping ``object``) and returns the
    first class that defines a non-accessor attribute under ``name``, or
    ``None`` if none exists. Used to detect built-in attribute
    shadowing.
    """
    for klass in target_cls.__mro__:
        if klass is object:
            continue
        if name not in klass.__dict__:
            continue
        attr = klass.__dict__[name]
        if isinstance(attr, _CachedAccessor):
            return None  # an accessor wins over any further shadow search
        return klass
    return None


def _validate_name(name: object) -> str:
    """Normalize and validate an accessor name.

    Accepts ``object`` so the ``isinstance`` check is reachable for
    callers that bypass static typing (e.g. dynamic plugin loaders or
    tests that intentionally pass the wrong type).
    """
    if not isinstance(name, str):
        msg = f'Accessor name must be a string, got {type(name).__name__}.'
        raise TypeError(msg)
    normalized = name.strip()
    if not normalized:
        msg = 'Accessor name must not be empty.'
        raise ValueError(msg)
    if normalized.startswith('_'):
        msg = (
            f'Accessor name must not start with an underscore, got {name!r}. '
            'Leading underscores are reserved for PyVista internals.'
        )
        raise ValueError(msg)
    if not normalized.isidentifier():
        msg = f'Accessor name must be a valid Python identifier, got {name!r}.'
        raise ValueError(msg)
    return normalized


def _validate_target(target_cls: Any) -> type:
    """Validate that ``target_cls`` is a class."""
    if not isinstance(target_cls, type):
        msg = f'target_cls must be a class, got {type(target_cls).__name__}.'
        raise TypeError(msg)
    return target_cls


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
        restored on :func:`~pyvista.unregister_dataset_accessor`.

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
        two plugins collide.

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
    normalized = _validate_name(name)
    target = _validate_target(target_cls)

    # Accept ``object`` so the runtime guard is reachable when callers
    # (e.g. dynamic plugin loaders) bypass static typing.
    def decorator(accessor_cls: object) -> type:
        if not isinstance(accessor_cls, type):
            msg = f'Accessor must be a class, got {type(accessor_cls).__name__}.'
            raise TypeError(msg)
        _attach_accessor(normalized, target, accessor_cls, override=override)
        return accessor_cls

    return decorator


def _attach_accessor(
    name: str,
    target_cls: type,
    accessor_cls: type,
    *,
    override: bool,
) -> None:
    """Attach ``accessor_cls`` as a ``_CachedAccessor`` on ``target_cls``.

    Performs collision detection before attaching.
    """
    accessor_owner = _find_accessor_on_mro(target_cls, name)
    if accessor_owner is not None:
        # Accessor-vs-accessor collision — warn and replace (pandas style).
        if accessor_owner is target_cls:
            location = target_cls.__qualname__
        else:
            location = f'{accessor_owner.__qualname__} (inherited by {target_cls.__qualname__})'
        warn_external(
            f'Registering accessor {name!r} on {target_cls.__qualname__} '
            f'replaces an existing registered accessor on {location}.',
        )
    else:
        shadow_owner = _find_shadow_on_mro(target_cls, name)
        if shadow_owner is not None and not override:
            if shadow_owner is target_cls:
                location = target_cls.__qualname__
            else:
                location = f'{shadow_owner.__qualname__} (inherited by {target_cls.__qualname__})'
            msg = (
                f'Cannot register accessor {name!r} on '
                f'{target_cls.__qualname__}: shadows built-in attribute on '
                f'{location}. Pass override=True to force.'
            )
            raise ValueError(msg)

    # Record the prior attribute value only on first registration for
    # this (target, name) pair so repeated override+register cycles do
    # not forget the very first built-in attribute.
    key = (target_cls, name)
    if key not in _prior_values:
        _prior_values[key] = target_cls.__dict__.get(name, _MISSING)

    setattr(target_cls, name, _CachedAccessor(name, accessor_cls))

    source = f'{accessor_cls.__module__}.{accessor_cls.__qualname__}'
    # Remove any previous registration record for this (target, name) pair
    # so the registry state never reports duplicates.
    _registrations[:] = [
        r for r in _registrations if not (r.target is target_cls and r.name == name)
    ]
    _registrations.append(
        AccessorRegistration(
            name=name,
            target=target_cls,
            accessor=accessor_cls,
            source=source,
        ),
    )


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
    normalized = _validate_name(name)
    target = _validate_target(target_cls)

    attr = target.__dict__.get(normalized)
    if not isinstance(attr, _CachedAccessor):
        # ValueError (not TypeError) — the shape of the target is valid,
        # the caller just did not have an accessor to remove.
        msg = f'No registered accessor {name!r} attached directly to {target.__qualname__}.'
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
    """Scan ``pyvista.accessors`` entry-point metadata once.

    Populates :data:`_pending_accessors` with a mapping of
    ``name -> module_path`` for every declared entry point. The plugin
    modules themselves are **not** imported here; the cost of this
    function is one ``importlib.metadata.entry_points`` call. The
    plugin import is deferred to :func:`_resolve_pending_accessor`,
    which runs only when the corresponding accessor is actually
    accessed on a dataset.
    """
    global _entry_points_loaded  # noqa: PLW0603
    if _entry_points_loaded:
        return

    # Flip the flag before iterating so a plugin that happens to import
    # pyvista recursively does not loop back into this function.
    _entry_points_loaded = True
    for accessor_entry_point in entry_points(group=ACCESSOR_ENTRY_POINT_GROUP):
        _pending_accessors[accessor_entry_point.name] = accessor_entry_point.value


def _resolve_pending_accessor(name: str) -> bool:
    """Import the plugin module registered under ``name``, if any.

    Called from :meth:`pyvista.DataObject.__getattr__` when a normal
    attribute lookup misses. Ensures entry-point metadata has been
    scanned, pops the pending entry for ``name``, and imports the
    corresponding plugin module. Importing the module triggers any
    ``@register_dataset_accessor`` decorators inside it and attaches
    the accessor as a side effect.

    Returns
    -------
    bool
        ``True`` if a plugin was loaded for ``name`` (and the attribute
        lookup should be retried). ``False`` if no pending plugin
        matches ``name``, or if the plugin failed to import.

    Notes
    -----
    A plugin that fails to import emits a ``UserWarning`` and is
    dropped from the pending list, so subsequent lookups of the same
    name fall straight through without re-triggering the import or
    re-emitting the warning.

    """
    _ensure_entry_points()
    module_path = _pending_accessors.pop(name, None)
    if module_path is None:
        return False
    try:
        import_module(module_path)
    except Exception as exc:  # noqa: BLE001
        msg = (
            f'Failed to load {ACCESSOR_ENTRY_POINT_GROUP} entry point '
            f'"{name}" from {module_path}: {exc}'
        )
        warn_external(msg)
        return False
    return True


def registered_accessors() -> tuple[AccessorRegistration, ...]:
    """Return every accessor currently registered.

    Forces discovery of any pending entry-point plugins so the returned
    list reflects every plugin visible to PyVista, not just the ones
    that have been touched already via attribute access. A plugin that
    fails to import emits a ``UserWarning`` and is skipped; the rest
    still appear in the result.

    .. versionadded:: 0.48.0

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
    _ensure_entry_points()
    for pending_name in list(_pending_accessors):
        _resolve_pending_accessor(pending_name)
    return tuple(_registrations)
