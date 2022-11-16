"""Internal vtkAlgorithm support helpers."""
import traceback

import numpy as np

from pyvista import _vtk

from .helpers import wrap


def algorithm_to_mesh_handler(mesh_or_algo, default_port=0):
    """Handle vtkAlgorithms where mesh objects are expected."""
    algo = None
    if isinstance(mesh_or_algo, (_vtk.vtkAlgorithm, _vtk.vtkAlgorithmOutput)):
        if isinstance(mesh_or_algo, _vtk.vtkAlgorithmOutput):
            algo = mesh_or_algo.GetProducer()
            port = mesh_or_algo.GetIndex()
        else:
            algo = mesh_or_algo
            port = default_port
        algo.Update()
        mesh_or_algo = wrap(algo.GetOutputDataObject(port))
    return mesh_or_algo, algo


class BasePreserveTypeAlgorithm(_vtk.VTKPythonAlgorithmBase):
    """Base algorithm to preserve type."""

    def __init__(self, nInputPorts=1, nOutputPorts=1):
        """Initialize algorithm."""
        _vtk.VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=nInputPorts,
            nOutputPorts=nOutputPorts,
        )

    # THIS IS CRUCIAL to preserve data type through filter
    def RequestDataObject(self, request, inInfo, outInfo):
        """Preserve data type."""
        self.OutputType = self.GetInputData(inInfo, 0, 0).GetClassName()
        self.FillOutputPortInformation(0, outInfo.GetInformationObject(0))
        return 1


class ActiveScalarsAlgorithm(BasePreserveTypeAlgorithm):
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

    # THIS IS CRUCIAL to preserve data type through filter
    def RequestDataObject(self, request, inInfo, outInfo):
        """Preserve data type."""
        self.OutputType = self.GetInputData(inInfo, 0, 0).GetClassName()
        self.FillOutputPortInformation(0, outInfo.GetInformationObject(0))
        return 1

    def RequestData(self, request, inInfo, outInfo):
        """Perform algorithm execution."""
        try:
            inp = self.GetInputData(inInfo, 0, 0)
            out = self.GetOutputData(outInfo, 0)
            output = wrap(inp).copy()
            if output.n_arrays:
                output.set_active_scalars(self.scalars_name, preference=self.preference)
            out.ShallowCopy(output)
        except Exception as e:
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
        except Exception as e:
            traceback.print_exc()
            raise e
        return 1


class AddIDsAlgorithm(BasePreserveTypeAlgorithm):
    """Internal helper algorithm to add point or cell IDs."""

    def __init__(self, point_ids=True, cell_ids=True):
        """Initialize algorithm."""
        super().__init__()
        if not point_ids and not cell_ids:
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
            out.ShallowCopy(output)
        except Exception as e:
            traceback.print_exc()
            raise e
        return 1


class CrinkleAlgorithm(_vtk.VTKPythonAlgorithmBase):
    """Internal helper algorithm to crinkle cell IDs."""

    def __init__(self, point_ids=True, cell_ids=True):
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
        except Exception as e:
            traceback.print_exc()
            raise e
        return 1


def outline_algorithm(inp, generate_faces=False):
    """Add vtkOutlineFilter to pipeline."""
    alg = _vtk.vtkOutlineFilter()
    if isinstance(inp, _vtk.vtkAlgorithm):
        alg.SetInputConnection(0, inp.GetOutputPort())
    elif isinstance(inp, _vtk.vtkAlgorithmOutput):
        alg.SetInputConnection(0, inp)
    else:
        alg.SetInputDataObject(0, inp)
    alg.SetGenerateFaces(generate_faces)
    return alg


def active_scalars_algorithm(inp, name, preference='point'):
    """Add a filter that sets the active scalars."""
    alg = ActiveScalarsAlgorithm(
        name=name,
        preference=preference,
    )
    if isinstance(inp, _vtk.vtkAlgorithm):
        alg.SetInputConnection(0, inp.GetOutputPort())
    elif isinstance(inp, _vtk.vtkAlgorithmOutput):
        alg.SetInputConnection(0, inp)
    else:
        alg.SetInputDataObject(0, inp)
    return alg


def pointset_to_polydata_algorithm(inp):
    """Add a filter that casts PointSet to PolyData."""
    alg = PointSetToPolyDataAlgorithm()
    if isinstance(inp, _vtk.vtkAlgorithm):
        alg.SetInputConnection(0, inp.GetOutputPort())
    elif isinstance(inp, _vtk.vtkAlgorithmOutput):
        alg.SetInputConnection(0, inp)
    else:
        alg.SetInputDataObject(0, inp)
    return alg


def add_ids_algorithm(inp, point_ids=True, cell_ids=True):
    """Add a filter that adds point or cell IDs."""
    alg = AddIDsAlgorithm(point_ids=point_ids, cell_ids=cell_ids)
    if isinstance(inp, _vtk.vtkAlgorithm):
        alg.SetInputConnection(0, inp.GetOutputPort())
    elif isinstance(inp, _vtk.vtkAlgorithmOutput):
        alg.SetInputConnection(0, inp)
    else:
        alg.SetInputDataObject(0, inp)
    return alg


def crinkle_algorithm(clip, source, point_ids=True, cell_ids=True):
    """Add a filter that crinkles a clip."""
    alg = CrinkleAlgorithm(point_ids=point_ids, cell_ids=cell_ids)
    if isinstance(clip, _vtk.vtkAlgorithm):
        alg.SetInputConnection(0, clip.GetOutputPort())
    elif isinstance(clip, _vtk.vtkAlgorithmOutput):
        alg.SetInputConnection(0, clip)
    else:
        alg.SetInputDataObject(0, clip)
    if isinstance(source, _vtk.vtkAlgorithm):
        alg.SetInputConnection(1, source.GetOutputPort())
    elif isinstance(source, _vtk.vtkAlgorithmOutput):
        alg.SetInputConnection(1, source)
    else:
        alg.SetInputDataObject(1, source)
    return alg
