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
boolean arrays. A sequence holds either Python values or NumPy scalars, not a mix of both,
and a lone scalar or zero-dimensional array is not array-like. Complex, string, and object
arrays are not included; annotate an input that accepts them as
``NDArray[Any] | Sequence[Any]``.

.. currentmodule:: pyvista.typing

pyvista.typing.VectorLikeFloat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One-dimensional array-like object with numerical values, such as a list of numbers or a 1D
NumPy array.

.. autodata:: VectorLikeFloat

pyvista.typing.VectorLikeInt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One-dimensional array-like object with integer values, such as a list of integers or a 1D
NumPy array.

.. autodata:: VectorLikeInt

pyvista.typing.VectorLikeBool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One-dimensional array-like object with boolean values, such as a list of booleans or a 1D
NumPy array.

.. autodata:: VectorLikeBool

pyvista.typing.MatrixLikeFloat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Two-dimensional array-like object with numerical values, such as a list of vectors or a 2D
NumPy array.

.. autodata:: MatrixLikeFloat

pyvista.typing.MatrixLikeInt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Two-dimensional array-like object with integer values, such as a list of vectors or a 2D
NumPy array.

.. autodata:: MatrixLikeInt

pyvista.typing.MatrixLikeBool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Two-dimensional array-like object with boolean values, such as a list of vectors or a 2D
NumPy array.

.. autodata:: MatrixLikeBool

pyvista.typing.ArrayLikeFloat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Any-dimensional array-like object with numerical values, such as nested lists of numbers or
a NumPy array.

.. autodata:: ArrayLikeFloat

pyvista.typing.ArrayLikeInt
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Any-dimensional array-like object with integer values, such as nested lists of integers or a
NumPy array.

.. autodata:: ArrayLikeInt

pyvista.typing.ArrayLikeBool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Any-dimensional array-like object with boolean values, such as nested lists of booleans or a
NumPy array.

.. autodata:: ArrayLikeBool


VTK Related Types
-----------------

pyvista.typing.WrappableType
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Object accepted by :func:`~pyvista.wrap`, such as a VTK data object, a NumPy array of
points, or a mesh from another library.

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
Array or object representing a spatial rotation, such as a 3x3 matrix.

.. currentmodule:: pyvista.typing

.. autodata:: RotationLike

pyvista.typing.TransformLike
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Array or object representing a spatial transformation, such as a 4x4 matrix or a
:data:`RotationLike`.

.. currentmodule:: pyvista.typing

.. autodata:: TransformLike

pyvista.typing.InteractionEventType
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Interaction event mostly used for widgets, given by name or as a VTK event ID.

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
Chart accepted by :meth:`~pyvista.Plotter.add_chart`.

.. currentmodule:: pyvista.typing

.. autodata:: Chart

pyvista.typing.PlottableType
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Object accepted by :func:`~pyvista.plot` or :meth:`~pyvista.Plotter.add_mesh`, such as a
dataset, a NumPy array of points, or the path of a mesh file.

.. currentmodule:: pyvista.typing

.. autodata:: PlottableType
