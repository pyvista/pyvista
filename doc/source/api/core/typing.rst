Typing
======

Type aliases and type variable used by PyVista.


Numeric Array-Like Types
------------------------

pyvista.NumberType
~~~~~~~~~~~~~~~~~~
Type variable for numeric data types.

.. currentmodule:: pyvista

.. autotypevar:: NumberType

pyvista.ArrayLike
~~~~~~~~~~~~~~~~~
Any-dimensional array-like object with numerical values.

Includes sequences, nested sequences, and numpy arrays. Scalar values are not included.

.. currentmodule:: pyvista

.. autodata:: ArrayLike

pyvista.MatrixLike
~~~~~~~~~~~~~~~~~~
Two-dimensional array-like object with numerical values.

Includes singly nested sequences and numpy arrays.

.. currentmodule:: pyvista

.. autodata:: MatrixLike


pyvista.VectorLike
~~~~~~~~~~~~~~~~~~
One-dimensional array-like object with numerical values.

Includes sequences and numpy arrays.

.. currentmodule:: pyvista

.. autodata:: VectorLike


VTK Related Types
-----------------

pyvista.BoundsTuple
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pyvista

.. autoclass:: BoundsTuple

pyvista.CellsLike
~~~~~~~~~~~~~~~~~

.. currentmodule:: pyvista

.. autodata:: CellsLike

pyvista.CellArrayLike
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pyvista

.. autodata:: CellArrayLike

pyvista.RotationLike
~~~~~~~~~~~~~~~~~~~~
Array or object representing a spatial rotation.

Includes 3x3 arrays and SciPy Rotation objects.

.. currentmodule:: pyvista

.. autodata:: RotationLike

pyvista.TransformLike
~~~~~~~~~~~~~~~~~~~~~
Array or object representing a spatial transformation.

Includes 3x3 and 4x4 arrays as well as SciPy Rotation objects.

.. currentmodule:: pyvista

.. autodata:: TransformLike

pyvista.InteractionEventType
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Interaction event mostly used for widgets.

Includes both strings such as ``'end'``, ``'start'`` and ``'always'``
and :vtk:`vtkCommand.EventIds`.

.. currentmodule:: pyvista

.. autodata:: InteractionEventType

pyvista.CameraPositionOptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Any object used to set a :class:`Camera`.

.. currentmodule:: pyvista

.. autodata:: CameraPositionOptions

pyvista.JupyterBackendOptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Jupyter backend to use.

.. currentmodule:: pyvista

.. autodata:: JupyterBackendOptions


Data Object Types
-----------------

pyvista._GridType
~~~~~~~~~~~~~~~~~

.. currentmodule:: pyvista

.. autotypevar:: _GridType

pyvista._PointGridType
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pyvista

.. autotypevar:: _PointGridType

pyvista._PointSetType
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pyvista

.. autotypevar:: _PointSetType


pyvista._DataSetType
~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pyvista

.. autotypevar:: _DataSetType

pyvista._DataSetOrMultiBlockType
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pyvista

.. autotypevar:: _DataSetOrMultiBlockType

pyvista._DataObjectType
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pyvista

.. autotypevar:: _DataObjectType
