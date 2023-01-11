"""Internal vtkAlgorithm support helpers."""
import traceback

import numpy as np

import pyvista
from pyvista import _vtk

from .helpers import wrap


def algorithm_to_mesh_handler(mesh_or_algo, port=0):
    """Handle vtkAlgorithms where mesh objects are expected."""
    if isinstance(mesh_or_algo, (_vtk.vtkAlgorithm, _vtk.vtkAlgorithmOutput)):
        if isinstance(mesh_or_algo, _vtk.vtkAlgorithmOutput):
            algo = mesh_or_algo.GetProducer()
            # If vtkAlgorithmOutput, override port argument
            port = mesh_or_algo.GetIndex()
            output = mesh_or_algo
        else:
            algo = mesh_or_algo
            output = algo.GetOutputPort(port)
        algo.Update()  # NOTE: this could be expensive... but we need it to get the mesh
        #                      for legacy implementation. This can be refactored.
        mesh_or_algo = wrap(algo.GetOutputDataObject(port))
        if mesh_or_algo is None:
            # This is known to happen with vtkPointSet and VTKPythonAlgorithmBase
            raise RuntimeError('The passed algorithm is failing to produce an output.')
        # NOTE: Return the vtkAlgorithmOutput only if port is non-zero. Segfaults can sometimes
        #       happen with vtkAlgorithmOutput. This logic will mostly avoid those issues.
        #       See https://gitlab.kitware.com/vtk/vtk/-/issues/18776
        return mesh_or_algo, output if port != 0 else algo
    return mesh_or_algo, None


def set_algorithm_input(alg, inp, port=0):
    """Set the input to a vtkAlgorithm.

    Parameters
    ----------
    alg : vtk.vtkAlgorith
        The algorithm who's input is being set

    inp : vtk.vtkAlgorithm or vtk.vtkAlgorithmOutput or vtk.vtkDataObject
        The input to the algorithm

    port : int, default: 0
        The input port

    """
    if isinstance(inp, _vtk.vtkAlgorithm):
        alg.SetInputConnection(port, inp.GetOutputPort())
    elif isinstance(inp, _vtk.vtkAlgorithmOutput):
        alg.SetInputConnection(port, inp)
    else:
        alg.SetInputDataObject(port, inp)


class PreserveTypeAlgorithmBase(_vtk.VTKPythonAlgorithmBase):
    """Base algorithm to preserve type."""

    def __init__(self, nInputPorts=1, nOutputPorts=1):
        """Initialize algorithm."""
        _vtk.VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=nInputPorts,
            nOutputPorts=nOutputPorts,
        )

    def GetInputData(self, inInfo, port, idx):
        """Get input data object.

        This will convert ``vtkPointSet`` to ``vtkPolyData``
        """
        inp = wrap(_vtk.VTKPythonAlgorithmBase.GetInputData(self, inInfo, port, idx))
        if isinstance(inp, pyvista.PointSet):
            return inp.cast_to_polydata()
        return inp

    # THIS IS CRUCIAL to preserve data type through filter
    def RequestDataObject(self, request, inInfo, outInfo):
        """Preserve data type."""
        class_name = self.GetInputData(inInfo, 0, 0).GetClassName()
        if class_name == 'vtkPointSet':
            # See https://gitlab.kitware.com/vtk/vtk/-/issues/18771
            self.OutputType = 'vtkPolyData'
        else:
            self.OutputType = class_name
        self.FillOutputPortInformation(0, outInfo.GetInformationObject(0))
        return 1


class ActiveScalarsAlgorithm(PreserveTypeAlgorithmBase):
    """Internal helper algorithm to control active scalars.

    Parameters
    ----------
    name : str
        Name of scalars used to "color" the mesh.  Accepts a
        string name of an array that is present on the mesh.
        Array should be sized as a single vector.

    preference : str, optional
        When ``mesh.n_points == mesh.n_cells`` and setting
        scalars, this parameter sets how the scalars will be
        mapped to the mesh.  Default ``'point'``, causes the
        scalars will be associated with the mesh points.  Can be
        either ``'point'`` or ``'cell'``.

    """

    def __init__(self, name, preference='point'):
        """Initialize algorithm."""
        super().__init__()
        self.scalars_name = name
        self.preference = preference

    def RequestData(self, request, inInfo, outInfo):
        """Perform algorithm execution."""
        try:
            inp = wrap(self.GetInputData(inInfo, 0, 0))
            out = self.GetOutputData(outInfo, 0)
            output = inp.copy()
            if output.n_arrays:
                output.set_active_scalars(self.scalars_name, preference=self.preference)
            out.ShallowCopy(output)
        except Exception as e:  # pragma: no cover
            traceback.print_exc()
            raise e
        return 1


class PointSetToPolyDataAlgorithm(_vtk.VTKPythonAlgorithmBase):
    """Internal helper algorithm to cast PointSets."""

    def __init__(self):
        """Initialize algorithm."""
        _vtk.VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=1,
            nOutputPorts=1,
            inputType='vtkPointSet',
            outputType='vtkPolyData',
        )

    def RequestData(self, request, inInfo, outInfo):
        """Perform algorithm execution."""
        try:
            inp = wrap(self.GetInputData(inInfo, 0, 0))
            out = self.GetOutputData(outInfo, 0)
            output = inp.cast_to_polydata(deep=False)
            out.ShallowCopy(output)
        except Exception as e:  # pragma: no cover
            traceback.print_exc()
            raise e
        return 1


class AddIDsAlgorithm(PreserveTypeAlgorithmBase):
    """Internal helper algorithm to add point or cell IDs."""

    def __init__(self, point_ids=True, cell_ids=True):
        """Initialize algorithm."""
        super().__init__()
        if not point_ids and not cell_ids:  # pragma: no cover
            raise ValueError('IDs must be set for points or cells or both.')
        self.point_ids = point_ids
        self.cell_ids = cell_ids

    def RequestData(self, request, inInfo, outInfo):
        """Perform algorithm execution."""
        try:
            inp = wrap(self.GetInputData(inInfo, 0, 0))
            out = self.GetOutputData(outInfo, 0)
            output = inp.copy()
            if self.point_ids:
                output.point_data['point_ids'] = np.arange(0, output.n_points, dtype=int)
            if self.cell_ids:
                output.cell_data['cell_ids'] = np.arange(0, output.n_cells, dtype=int)
            if output.active_scalars_name in ['point_ids', 'cell_ids']:
                output.active_scalars_name = inp.active_scalars_name
            out.ShallowCopy(output)
        except Exception as e:  # pragma: no cover
            traceback.print_exc()
            raise e
        return 1


class CrinkleAlgorithm(_vtk.VTKPythonAlgorithmBase):
    """Internal helper algorithm to crinkle cell IDs."""

    def __init__(self):
        """Initialize algorithm."""
        super().__init__(
            nInputPorts=2,
            outputType='vtkUnstructuredGrid',
        )

    def RequestData(self, request, inInfo, outInfo):
        """Perform algorithm execution."""
        try:
            clipped = wrap(self.GetInputData(inInfo, 0, 0))
            source = wrap(self.GetInputData(inInfo, 1, 0))
            out = self.GetOutputData(outInfo, 0)
            output = source.extract_cells(np.unique(clipped.cell_data['cell_ids']))
            out.ShallowCopy(output)
        except Exception as e:  # pragma: no cover
            traceback.print_exc()
            raise e
        return 1


def outline_algorithm(inp, generate_faces=False):
    """Add vtkOutlineFilter to pipeline."""
    alg = _vtk.vtkOutlineFilter()
    set_algorithm_input(alg, inp)
    alg.SetGenerateFaces(generate_faces)
    return alg


def extract_surface_algorithm(inp, pass_pointid=False, pass_cellid=False, nonlinear_subdivision=1):
    """Add vtkDataSetSurfaceFilter to pipeline."""
    surf_filter = _vtk.vtkDataSetSurfaceFilter()
    surf_filter.SetPassThroughPointIds(pass_pointid)
    surf_filter.SetPassThroughCellIds(pass_cellid)
    if nonlinear_subdivision != 1:
        surf_filter.SetNonlinearSubdivisionLevel(nonlinear_subdivision)
    set_algorithm_input(surf_filter, inp)
    return surf_filter


def active_scalars_algorithm(inp, name, preference='point'):
    """Add a filter that sets the active scalars."""
    alg = ActiveScalarsAlgorithm(
        name=name,
        preference=preference,
    )
    set_algorithm_input(alg, inp)
    return alg


def pointset_to_polydata_algorithm(inp):
    """Add a filter that casts PointSet to PolyData."""
    alg = PointSetToPolyDataAlgorithm()
    set_algorithm_input(alg, inp)
    return alg


def add_ids_algorithm(inp, point_ids=True, cell_ids=True):
    """Add a filter that adds point or cell IDs."""
    alg = AddIDsAlgorithm(point_ids=point_ids, cell_ids=cell_ids)
    set_algorithm_input(alg, inp)
    return alg


def crinkle_algorithm(clip, source):
    """Add a filter that crinkles a clip."""
    alg = CrinkleAlgorithm()
    set_algorithm_input(alg, clip, 0)
    set_algorithm_input(alg, source, 1)
    return alg


def cell_data_to_point_data_algorithm(inp, pass_cell_data=False):
    """Add a filter that converts cell data to point data."""
    alg = _vtk.vtkCellDataToPointData()
    alg.SetPassCellData(pass_cell_data)
    set_algorithm_input(alg, inp)
    return alg


def point_data_to_cell_data_algorithm(inp, pass_point_data=False):
    """Add a filter that converts point data to cell data."""
    alg = _vtk.vtkPointDataToCellData()
    alg.SetPassPointData(pass_point_data)
    set_algorithm_input(alg, inp)
    return alg
