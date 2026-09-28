"""Type aliases for type hints."""

from __future__ import annotations

from pyvista_validation.typing import ArrayLikeBool as ArrayLikeBool
from pyvista_validation.typing import ArrayLikeFloat as ArrayLikeFloat
from pyvista_validation.typing import ArrayLikeInt as ArrayLikeInt
from pyvista_validation.typing import MatrixLikeBool as MatrixLikeBool
from pyvista_validation.typing import MatrixLikeFloat as MatrixLikeFloat
from pyvista_validation.typing import MatrixLikeInt as MatrixLikeInt
from pyvista_validation.typing import VectorLikeBool as VectorLikeBool
from pyvista_validation.typing import VectorLikeFloat as VectorLikeFloat
from pyvista_validation.typing import VectorLikeInt as VectorLikeInt

# Aliases for PyVista and VTK objects
from ._aliases import BoundsTuple as BoundsTuple
from ._aliases import CellArrayLike as CellArrayLike
from ._aliases import CellsLike as CellsLike
from ._aliases import InteractionEventType as InteractionEventType
from ._aliases import LineStyle as LineStyle
from ._aliases import RotationLike as RotationLike
from ._aliases import TransformLike as TransformLike
from ._aliases import WrappableType as WrappableType
from ._aliases import _MeshLike as _MeshLike

# NumPy scalar and array aliases and TypeVars, built only from NumPy and validation types
from ._array_types import _AnyArrayLike as _AnyArrayLike
from ._array_types import _ArrayLikeOrScalar as _ArrayLikeOrScalar
from ._array_types import _Floating as _Floating
from ._array_types import _FloatingT as _FloatingT
from ._array_types import _GenericT as _GenericT
from ._array_types import _Integer as _Integer
from ._array_types import _IntegerT as _IntegerT
from ._array_types import _MatrixSequence as _MatrixSequence
from ._array_types import _NumericArray as _NumericArray
from ._array_types import _NumericScalar as _NumericScalar
from ._array_types import _Real as _Real
from ._array_types import _RealT as _RealT
from ._array_types import _Scalar as _Scalar
from ._array_types import _ScalarT as _ScalarT
from ._array_types import _VectorSequence as _VectorSequence
from ._array_types import _VolumeArray as _VolumeArray
from ._dataset_types import _DataObjectType as _DataObjectType
from ._dataset_types import _DataSetOrMultiBlockType as _DataSetOrMultiBlockType
from ._dataset_types import _DataSetType as _DataSetType
from ._dataset_types import _GridType as _GridType
from ._dataset_types import _MultiBlockType as _MultiBlockType
from ._dataset_types import _OutputDataObject as _OutputDataObject
from ._dataset_types import _OutputDataSet as _OutputDataSet
from ._dataset_types import _PointGridType as _PointGridType
from ._dataset_types import _PointSetBaseType as _PointSetBaseType
