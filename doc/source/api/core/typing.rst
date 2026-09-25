Typing
======

.. module:: pyvista.typing

Type aliases and type variable for annotating code that uses PyVista.

.. versionadded:: 0.50
   The ``pyvista.typing`` module.

.. deprecated:: 0.50
   Accessing these aliases from ``pyvista``, for example ``pyvista.VectorLike``,
   is deprecated. Use ``pyvista.typing.VectorLike`` instead.


Numeric Array-Like Types
------------------------

pyvista.typing.NumberType
~~~~~~~~~~~~~~~~~~~~~~~~~
Type variable for numeric data types.

.. currentmodule:: pyvista.typing

.. autotypevar:: NumberType

pyvista.typing.Number
~~~~~~~~~~~~~~~~~~~~~
Integer or float value.

.. currentmodule:: pyvista.typing

.. autodata:: Number

pyvista.typing.ArrayLike
~~~~~~~~~~~~~~~~~~~~~~~~
Any-dimensional array-like object with numerical values.

Includes sequences, nested sequences, and numpy arrays. Scalar values are not included.

.. currentmodule:: pyvista.typing

.. autodata:: ArrayLike

pyvista.typing.MatrixLike
~~~~~~~~~~~~~~~~~~~~~~~~~
Two-dimensional array-like object with numerical values.

Includes singly nested sequences and numpy arrays.

.. currentmodule:: pyvista.typing

.. autodata:: MatrixLike


pyvista.typing.VectorLike
~~~~~~~~~~~~~~~~~~~~~~~~~
One-dimensional array-like object with numerical values.

Includes sequences and numpy arrays.

.. currentmodule:: pyvista.typing

.. autodata:: VectorLike


VTK Related Types
-----------------

pyvista.typing.WrappableType
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Object accepted by :func:`~pyvista.wrap`.

Includes PyVista and VTK data objects, VTK data arrays, NumPy arrays, ``trimesh``
meshes, ``meshio`` meshes, and ``None``.

.. currentmodule:: pyvista.typing

.. autodata:: WrappableType

pyvista.BoundsTuple
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pyvista

.. autoclass:: BoundsTuple

pyvista.typing.CellsLike
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pyvista.typing

.. autodata:: CellsLike

pyvista.typing.CellArrayLike
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pyvista.typing

.. autodata:: CellArrayLike

pyvista.typing.RotationLike
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Array or object representing a spatial rotation.

Includes 3x3 arrays and SciPy Rotation objects.

.. currentmodule:: pyvista.typing

.. autodata:: RotationLike

pyvista.typing.TransformLike
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Array or object representing a spatial transformation.

Includes 3x3 and 4x4 arrays as well as SciPy Rotation objects.

.. currentmodule:: pyvista.typing

.. autodata:: TransformLike

pyvista.typing.InteractionEventType
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Interaction event mostly used for widgets.

Includes both strings such as ``'end'``, ``'start'`` and ``'always'``
and :vtk:`vtkCommand.EventIds`.

.. currentmodule:: pyvista.typing

.. autodata:: InteractionEventType

pyvista.typing.LineStyle
~~~~~~~~~~~~~~~~~~~~~~~~
Named style of a line, shared by the charts and the line filters.

.. currentmodule:: pyvista.typing

.. autodata:: LineStyle

pyvista.typing.CameraPositionOptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Any object used to set a :class:`~pyvista.Camera`.

.. currentmodule:: pyvista.typing

.. autodata:: CameraPositionOptions

pyvista.typing.JupyterBackendOptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Jupyter backend to use.

.. currentmodule:: pyvista.typing

.. autodata:: JupyterBackendOptions

pyvista.typing.MeshValidationFields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Field options for :meth:`~pyvista.DataObjectFilters.validate_mesh`.

.. currentmodule:: pyvista.typing

.. autodata:: MeshValidationFields


Plotting Types
--------------

pyvista.typing.ColorLike
~~~~~~~~~~~~~~~~~~~~~~~~
Any object that can be converted to a :class:`~pyvista.Color`.

.. currentmodule:: pyvista.typing

.. autodata:: ColorLike

pyvista.typing.Chart
~~~~~~~~~~~~~~~~~~~~
Any of :class:`~pyvista.Chart2D`, :class:`~pyvista.ChartBox`, :class:`~pyvista.ChartPie`
or :class:`~pyvista.ChartMPL`, as accepted by :meth:`~pyvista.Plotter.add_chart`.

.. currentmodule:: pyvista.typing

.. autodata:: Chart
   :no-value:

pyvista.typing.PlottableType
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Object accepted by :func:`~pyvista.plot` or :meth:`~pyvista.Plotter.add_mesh`.

Includes PyVista and VTK datasets and composite datasets, ``trimesh`` and ``meshio``
meshes, NumPy arrays of points or of volume values, and the path of a mesh file.

.. currentmodule:: pyvista.typing

.. autodata:: PlottableType
