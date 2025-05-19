"""Core routines."""

from __future__ import annotations

from . import _vtk_core as _vtk_core
from ._typing_core import *
from ._typing_core._dataset_types import ConcreteDataObjectType as ConcreteDataObjectType
from ._typing_core._dataset_types import ConcreteDataSetType as ConcreteDataSetType
from ._typing_core._dataset_types import ConcreteGridType as ConcreteGridType
from ._typing_core._dataset_types import ConcretePointGridType as ConcretePointGridType
from ._typing_core._dataset_types import ConcretePointSetType as ConcretePointSetType
from .cell import Cell  # noqa: F401
from .cell import CellArray  # noqa: F401
from .celltype import CellType  # noqa: F401
from .composite import MultiBlock  # noqa: F401
from .dataobject import DataObject as DataObject
from .dataset import DataObject  # noqa: F401, F811
from .dataset import DataSet  # noqa: F401
from .datasetattributes import DataSetAttributes  # noqa: F401
from .errors import AmbiguousDataError  # noqa: F401
from .errors import DeprecationError  # noqa: F401
from .errors import MissingDataError  # noqa: F401
from .errors import NotAllTrianglesError  # noqa: F401
from .errors import PointSetCellOperationError  # noqa: F401
from .errors import PointSetDimensionReductionError  # noqa: F401
from .errors import PointSetNotSupported  # noqa: F401
from .errors import PyVistaDeprecationWarning  # noqa: F401
from .errors import PyVistaEfficiencyWarning  # noqa: F401
from .errors import PyVistaFutureWarning  # noqa: F401
from .errors import PyVistaPipelineError  # noqa: F401
from .errors import VTKVersionError  # noqa: F401
from .filters import CompositeFilters  # noqa: F401
from .dataset import DataSet as DataSet
from .datasetattributes import DataSetAttributes as DataSetAttributes
from .errors import AmbiguousDataError as AmbiguousDataError
from .errors import DeprecationError as DeprecationError
from .errors import MissingDataError as MissingDataError
from .errors import NotAllTrianglesError as NotAllTrianglesError
from .errors import PointSetCellOperationError as PointSetCellOperationError
from .errors import PointSetDimensionReductionError as PointSetDimensionReductionError
from .errors import PointSetNotSupported as PointSetNotSupported
from .errors import PyVistaAttributeError as PyVistaAttributeError
from .errors import PyVistaDeprecationWarning as PyVistaDeprecationWarning
from .errors import PyVistaEfficiencyWarning as PyVistaEfficiencyWarning
from .errors import PyVistaFutureWarning as PyVistaFutureWarning
from .errors import PyVistaPipelineError as PyVistaPipelineError
from .errors import VTKVersionError as VTKVersionError
from .filters import CompositeFilters as CompositeFilters
from .filters import DataObjectFilters as DataObjectFilters
from .filters import DataSetFilters  # noqa: F401
from .filters import ImageDataFilters  # noqa: F401
from .filters import PolyDataFilters  # noqa: F401
from .filters import UnstructuredGridFilters  # noqa: F401
from .grid import Grid  # noqa: F401
from .grid import ImageData  # noqa: F401
from .grid import RectilinearGrid  # noqa: F401
from .molecule import Molecule  # noqa: F401
from .objects import Table  # noqa: F401
from .partitioned import PartitionedDataSet  # noqa: F401
from .pointset import ExplicitStructuredGrid  # noqa: F401
from .pointset import PointGrid  # noqa: F401
from .pointset import PointSet  # noqa: F401
from .pointset import PolyData  # noqa: F401
from .pointset import StructuredGrid  # noqa: F401
from .pointset import UnstructuredGrid  # noqa: F401
from .pyvista_ndarray import pyvista_ndarray  # noqa: F401
from .utilities import *
from .wrappers import _wrappers as _wrappers
