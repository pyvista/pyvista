.. _extending-pyvista:

Extending PyVista
=================

PyVista exposes a pandas/xarray-style accessor mechanism so third-party
packages can attach custom filter methods to dataset classes without
monkey-patching or subclassing. This is the recommended way to plug
domain-specific operations (mesh repair, tetrahedralization, format
conversion, remote IO) into the fluent filter API.

.. contents::
   :local:
   :depth: 2


Why accessors
-------------

PyVista inherits the problem every extensible scientific library in
Python eventually faces: users want to add methods to a core type
without forking it. pandas solved this first with
`registered accessors <https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors>`_
and xarray adopted the same pattern, growing an ecosystem of
downstream plugins such as
`rioxarray <https://github.com/corteva/rioxarray>`_ and
`pint-xarray <https://github.com/xarray-contrib/pint-xarray>`_
that hook in via ``ds.rio.reproject(...)`` and ``ds.pint.quantify()``
without ever subclassing xarray's core objects.

PyVista's accessor mechanism follows the same contract. A plugin
registers an accessor class against a target dataset type. When the
plugin is imported, the accessor becomes available on every instance
of that type (and its subclasses) under a single namespace.

The advantages over subclassing and monkey-patching:

- **Namespaced**: ``mesh.meshfix.repair(...)`` cannot collide with
  PyVista built-ins or with a second plugin's methods.
- **Lazy**: the accessor class is instantiated on first access per
  instance, so unused plugins cost nothing.
- **Composable**: accessor methods that return PyVista datasets chain
  naturally with core filters.
- **Per-instance caching**: subsequent accesses on the same dataset
  return the same accessor object, which can hold per-mesh computed
  state without leaking across instances.


Writing an accessor
-------------------

An accessor class accepts the dataset instance as its single
``__init__`` argument and exposes any methods that should be callable
under the namespace.

.. code-block:: python

    # pymeshfix/_pyvista_plugin.py
    import pyvista as pv
    from pymeshfix import MeshFix


    @pv.register_dataset_accessor("meshfix", pv.PolyData)
    class MeshFixAccessor:
        """Accessor exposing pymeshfix on PolyData."""

        def __init__(self, mesh):
            self._mesh = mesh

        def repair(
            self,
            *,
            verbose=False,
            joincomp=False,
            remove_smallest_components=True
        ):
            fix = MeshFix(self._mesh)
            fix.repair(
                verbose=verbose,
                joincomp=joincomp,
                remove_smallest_components=remove_smallest_components,
            )
            return fix.mesh

        def has_holes(self):
            return MeshFix(self._mesh).has_holes()

Once the plugin is imported, every :class:`~pyvista.PolyData` instance
exposes the ``meshfix`` namespace:

.. code-block:: python

    import pyvista as pv
    import pymeshfix  # noqa: F401 — registers the .meshfix accessor

    mesh = pv.PolyData("broken.ply")
    result = mesh.clean().meshfix.repair(verbose=True).decimate(0.5)

The accessor registered against :class:`~pyvista.PolyData` is visible
only on ``PolyData`` and its subclasses. To expose a method across
every dataset type, register against :class:`~pyvista.DataSet`. To
cover :class:`~pyvista.MultiBlock` as well, register against
:class:`~pyvista.DataObject`.


Registration paths
------------------

PyVista supports two ways to register an accessor. Both use the same
``@register_dataset_accessor`` decorator. The difference is only
*when* the decorator runs.

**Import-time.** The plugin's module runs the decorator at import
time, so any user who does ``import plugin`` gets the accessor
attached. This is the simplest pattern and is ideal for scripts,
notebooks, and tests:

.. code-block:: python

    # script.py
    import pyvista as pv
    import pymeshfix  # noqa: F401 — registers the ``.meshfix`` accessor

    pv.PolyData("broken.ply").meshfix.repair()

**Entry points.** The plugin declares a ``pyvista.accessors`` entry
point in its ``pyproject.toml`` pointing at its accessor module.
``import pyvista`` reads the entry-point metadata but does not import
any plugin module. The plugin module imports on demand: on the first
``dataset.<name>`` access whose normal attribute lookup misses,
PyVista resolves the pending entry, imports the module, lets its
top-level decorator attach the accessor, and completes the lookup.
Users never need an explicit ``import plugin``:

.. code-block:: toml

   # pymeshfix/pyproject.toml
   [project.entry-points."pyvista.accessors"]
   meshfix = "pymeshfix._accessor"

.. code-block:: python

    # script.py
    import pyvista as pv

    pv.PolyData(
        "broken.ply"
    ).meshfix.repair()  # works with no explicit import

Both paths populate the same registry. A plugin that declares an
entry point AND imports its accessor module from its ``__init__`` is
safe (the second import is a no-op against ``sys.modules``).

The lazy resolution means installing an accessor plugin does not
affect ``import pyvista`` performance or stability. A broken plugin
only surfaces when a user actually accesses its namespace: they get
a ``UserWarning`` pointing at the specific plugin and an
``AttributeError`` on the call, and no other code is affected.
``pv.registered_accessors()`` is the one call that explicitly forces
discovery of every pending plugin so the returned list reflects the
full picture.

For production plugins on PyPI, prefer the entry-point path: users
get zero-config discovery without any startup cost on ``import
pyvista``. For in-script or in-notebook experimentation, the
decorator alone is simpler.

The entry-point key is the accessor namespace (``meshfix`` above),
and the value must be a module path (no ``:ClassName`` suffix). A
plugin that registers multiple accessors from one module should
declare one entry-point line per namespace, all pointing at the same
module. Put the accessor in a small module like ``_accessor.py``
that runs the decorator at import time and does nothing else; heavy
compute dependencies should be lazy-imported inside the accessor
methods, not at the module top.


Chaining and return types
-------------------------

Accessor methods can return three kinds of things:

1. **A PyVista dataset** (the common case). Chaining continues
   seamlessly into core filters and into other plugins' accessors.
2. **A different PyVista type** (for example, a filter that converts
   ``PolyData`` to ``UnstructuredGrid``). Chaining continues on the
   new type; any accessors registered against that type become
   available.
3. **A non-dataset value** (for example, a ``bool``). Chaining
   terminates. Documented as a query method, not a filter.

For consistency with PyVista's core filters, which are overwhelmingly
functional, plugins should prefer returning a new dataset rather than
mutating the input in place. The caller then decides whether to
assign the result back.


Collision policy
----------------

Two collision cases are handled differently:

- **Accessor-vs-accessor** (two plugins register the same name on
  the same target): emits :class:`UserWarning` and replaces the
  previous accessor. Matches pandas' behavior so a collision does not
  hard-break user scripts.
- **Accessor shadowing a built-in attribute** (a filter method, a
  property, or any other non-accessor attribute on the target or one
  of its ancestors): raises :class:`ValueError` unless
  ``override=True`` is passed. This prevents accidental replacement
  of core PyVista methods.

.. code-block:: python

    @pv.register_dataset_accessor("clip", pv.PolyData)
    class MyClipAccessor: ...


    # ValueError: Cannot register accessor 'clip' on PolyData:
    #   shadows built-in attribute on DataSetFilters
    #   (inherited by PolyData). Pass override=True to force.


Typing and autocomplete
-----------------------

Because accessors are attached at import time via a decorator, static
type checkers do not see the new attribute on the target class.
PyVista exports a :class:`~pyvista.DataSetAccessor` structural
protocol so plugin authors can have type checkers verify their own
accessor class conforms to the expected shape:

.. code-block:: python

    import pyvista as pv


    class MeshFixAccessor:
        def __init__(self, mesh: pv.PolyData) -> None:
            self._mesh = mesh

        def repair(self) -> pv.PolyData: ...


    # mypy / pyright verify that MeshFixAccessor's __init__ signature
    # is compatible with the DataSetAccessor protocol.
    _accessor_cls: type[pv.DataSetAccessor] = MeshFixAccessor

To give downstream users autocomplete on ``mesh.meshfix.repair(...)``,
plugin packages should ship a small ``.pyi`` stub that declares the
attribute on the target class. For example, ``pymeshfix/__init__.pyi``:

.. code-block:: python

    from pyvista import PolyData

    from pymeshfix._pyvista_plugin import MeshFixAccessor

    # Surface the attribute that the import-time decorator installs at
    # runtime. Type checkers do not execute decorators, so the stub is
    # the only signal they see.
    PolyData.meshfix: MeshFixAccessor  # type: ignore[attr-defined]

This is the same pattern used by ``rioxarray`` and ``pint-xarray`` on
the xarray side, where the plugin ships stubs that teach editors and
type checkers about the attribute.


Deregistration
--------------

For tests and interactive sessions, an accessor can be removed with
:func:`~pyvista.unregister_dataset_accessor`. Any built-in attribute
that was shadowed via ``override=True`` is restored.

.. code-block:: python

    pv.unregister_dataset_accessor("meshfix", pv.PolyData)

To inspect the current registry, call
:func:`~pyvista.registered_accessors`, which returns a tuple of
:class:`~pyvista.AccessorRegistration` records describing each active
registration.


Cache semantics
---------------

The first access of ``dataset.<name>`` constructs the accessor and
caches the result in the dataset instance's ``__dict__``. All
subsequent accesses return the same accessor object, which allows the
accessor to hold per-mesh computed state without re-running any setup
work.

If the underlying dataset is mutated in a way the accessor needs to
observe, evict the cache with ``del dataset.<name>``; the next access
will rebuild the accessor. Avoid mutating the dataset from inside
accessor methods unless you explicitly document and test that
behavior — functional methods that return a new dataset are easier to
reason about.


subclassing (advanced)
----------------------

For use cases that genuinely require a custom class (for example,
adding persistent state that travels with the dataset through
filters), direct subclassing of :class:`~pyvista.PolyData` or
:class:`~pyvista.UnstructuredGrid` is still supported. See
:ref:`Extending PyVista Example <extending_pyvista_example>` for the
subclassing pattern.

For everything else, prefer accessors. They compose more cleanly,
cost nothing when unused, and give plugins a clear boundary the
PyVista core can rely on.
