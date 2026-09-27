Typing
======

.. module:: pyvista.typing

Type aliases for annotating code that uses PyVista.

.. versionadded:: 0.50
   The ``pyvista.typing`` module.

.. deprecated:: 0.50
   Accessing these aliases from ``pyvista``, for example ``pyvista.ColorLike``,
   is deprecated. Use ``pyvista.typing.ColorLike`` instead.

.. deprecated:: 0.50
   ``pyvista.Number``, ``pyvista.NumberType`` and ``pyvista.NumpyArray`` are
   deprecated. Use ``float``, a :class:`~typing.TypeVar` and
   :data:`numpy.typing.NDArray` instead.

.. deprecated:: 0.50
   ``pyvista.ArrayLike``, ``pyvista.MatrixLike`` and ``pyvista.VectorLike`` are
   deprecated. Use the ``Float``, ``Int`` or ``Bool`` array-like types instead, for
   example ``VectorLikeFloat`` in place of ``VectorLike[float]``.


Numeric Array-Like Types
------------------------
Each array-like accepts NumPy arrays and sequences of one kind of value. As in Python type
hints, ``Float`` means :class:`float`, which also accepts :class:`int` and :class:`bool`, so
the ``Float`` types accept integer and boolean arrays too; likewise the ``Int`` types accept
boolean arrays. Complex, string, and object arrays are not included; annotate an input that
accepts them as ``NDArray[Any] | Sequence[Any]``.

.. currentmodule:: pyvista.typing

pyvista.typing.VectorLikeFloat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One-dimensional array-like object with numerical values.

Accepts sequences of :class:`float`, :class:`int` and :class:`bool` values, and NumPy
arrays of :class:`numpy.floating`, :class:`numpy.integer` or :class:`numpy.bool` dtype.

Includes sequences and one-dimensional NumPy arrays.

.. autodata:: VectorLikeFloat

pyvista.typing.VectorLikeInt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One-dimensional array-like object with integer values.

Accepts sequences of :class:`int` and :class:`bool` values, and NumPy arrays of
:class:`numpy.integer` or :class:`numpy.bool` dtype. Floating values are not included.

Includes sequences and one-dimensional NumPy arrays.

.. autodata:: VectorLikeInt

pyvista.typing.VectorLikeBool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One-dimensional array-like object with boolean values.

Accepts sequences of :class:`bool` values and NumPy arrays of :class:`numpy.bool` dtype
only. Integer and floating values are not included.

Includes sequences and one-dimensional NumPy arrays.

.. autodata:: VectorLikeBool

pyvista.typing.MatrixLikeFloat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Two-dimensional array-like object with numerical values.

Accepts sequences of :class:`float`, :class:`int` and :class:`bool` values, and NumPy
arrays of :class:`numpy.floating`, :class:`numpy.integer` or :class:`numpy.bool` dtype.

Includes sequences of vectors and two-dimensional NumPy arrays.

.. autodata:: MatrixLikeFloat

pyvista.typing.MatrixLikeInt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Two-dimensional array-like object with integer values.

Accepts sequences of :class:`int` and :class:`bool` values, and NumPy arrays of
:class:`numpy.integer` or :class:`numpy.bool` dtype. Floating values are not included.

Includes sequences of vectors and two-dimensional NumPy arrays.

.. autodata:: MatrixLikeInt

pyvista.typing.MatrixLikeBool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Two-dimensional array-like object with boolean values.

Accepts sequences of :class:`bool` values and NumPy arrays of :class:`numpy.bool` dtype
only. Integer and floating values are not included.

Includes sequences of vectors and two-dimensional NumPy arrays.

.. autodata:: MatrixLikeBool

pyvista.typing.ArrayLikeFloat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Any-dimensional array-like object with numerical values.

Accepts sequences of :class:`float`, :class:`int` and :class:`bool` values, and NumPy
arrays of :class:`numpy.floating`, :class:`numpy.integer` or :class:`numpy.bool` dtype.

Includes NumPy arrays and sequences nested up to four deep. Scalar values are not included.

.. autodata:: ArrayLikeFloat

pyvista.typing.ArrayLikeInt
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Any-dimensional array-like object with integer values.

Accepts sequences of :class:`int` and :class:`bool` values, and NumPy arrays of
:class:`numpy.integer` or :class:`numpy.bool` dtype. Floating values are not included.

Includes NumPy arrays and sequences nested up to four deep. Scalar values are not included.

.. autodata:: ArrayLikeInt

pyvista.typing.ArrayLikeBool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Any-dimensional array-like object with boolean values.

Accepts sequences of :class:`bool` values and NumPy arrays of :class:`numpy.bool` dtype
only. Integer and floating values are not included.

Includes NumPy arrays and sequences nested up to four deep. Scalar values are not included.

.. autodata:: ArrayLikeBool


VTK Related Types
-----------------

pyvista.typing.WrappableType
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Object accepted by :func:`~pyvista.wrap`.

Includes PyVista and VTK data objects, VTK data arrays, NumPy arrays, sequences of
points, ``trimesh`` meshes, ``meshio`` meshes, and ``None``.

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

Includes PyVista and VTK datasets, multiblock and partitioned datasets, ``trimesh`` and
``meshio`` meshes, NumPy arrays of points or of volume values, sequences of points,
and the path of a mesh file.

.. currentmodule:: pyvista.typing

.. autodata:: PlottableType
