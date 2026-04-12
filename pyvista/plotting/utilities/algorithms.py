"""Internal :vtk:`vtkAlgorithm` support helpers."""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core._vtk_utilities import DisableVtkSnakeCase
from pyvista.core.errors import PyVistaPipelineError
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import _NoNewAttrMixin
from pyvista.plotting import _vtk

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyvista import DataSet
    from pyvista.core.utilities.arrays import CellLiteral
    from pyvista.core.utilities.arrays import PointLiteral


def algorithm_to_mesh_handler(
    mesh_or_algo, port=0
) -> tuple[DataSet, _vtk.vtkAlgorithm | _vtk.vtkAlgorithmOutput | None]:
    """Handle :vtk:`vtkAlgorithms` where mesh objects are expected.

    This is a convenience method to handle :vtk:`vtkAlgorithms` when passed to methods
    that expect a :class:`~pyvista.DataSet`. This method will check if the passed
    object is a :vtk:`vtkAlgorithm` or :vtk:`vtkAlgorithmOutput` and if so,
    return that algorithm's output dataset (mesh) as the mesh to be used by the
    calling function.

    Parameters
    ----------
    mesh_or_algo : DataSet | :vtk:`vtkAlgorithm` | :vtk:`vtkAlgorithmOutput`
        The input to be used as a data set (mesh) or :vtk:`vtkAlgorithm` object.

    port : int, default: 0
        If the input (``mesh_or_algo``) is an algorithm, this specifies which output
        port to use on that algorithm for the returned mesh.

    Returns
    -------
    mesh : pyvista.DataSet
        The resulting mesh data set from the input.

    algorithm : :vtk:`vtkAlgorithm` | :vtk:`vtkAlgorithmOutput` | None
        If an algorithm is passed, it will be returned. Otherwise returns ``None``.

    """
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
        mesh = wrap(algo.GetOutputDataObject(port))
        if mesh is None:
            # This is known to happen with vtkPointSet and VTKPythonAlgorithmBase
            #     see workaround in PreserveTypeAlgorithmBase.
            #     This check remains as a fail-safe.
            msg = 'The passed algorithm is failing to produce an output.'  # type: ignore[unreachable]
            raise PyVistaPipelineError(msg)
        # NOTE: Return the vtkAlgorithmOutput only if port is non-zero. Segfaults can sometimes
        #       happen with vtkAlgorithmOutput. This logic will mostly avoid those issues.
        #       See https://gitlab.kitware.com/vtk/vtk/-/issues/18776
        return mesh, output if port != 0 else algo
    return mesh_or_algo, None


def set_algorithm_input(alg, inp, port=0):
    """Set the input to a :vtk:`vtkAlgorithm`.

    Parameters
    ----------
    alg : :vtk:`vtkAlgorithm`
        The algorithm whose input is being set.

    inp : :vtk:`vtkAlgorithm` | :vtk:`vtkAlgorithmOutput` | :vtk:`vtkDataObject`
        The input to the algorithm.

    port : int, default: 0
        The input port.

    """
    if isinstance(inp, _vtk.vtkAlgorithm):
        alg.SetInputConnection(port, inp.GetOutputPort())
    elif isinstance(inp, _vtk.vtkAlgorithmOutput):
        alg.SetInputConnection(port, inp)
    else:
        alg.SetInputDataObject(port, inp)


class PreserveTypeAlgorithmBase(_NoNewAttrMixin, DisableVtkSnakeCase, _vtk.VTKPythonAlgorithmBase):
    """Base algorithm to preserve type.

    Parameters
    ----------
    nInputPorts : int, default: 1
        Number of input ports for the algorithm.

    nOutputPorts : int, default: 1
        Number of output ports for the algorithm.

    """

    def __init__(self, nInputPorts=1, nOutputPorts=1):
        """Initialize algorithm."""
        _vtk.VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=nInputPorts,
            nOutputPorts=nOutputPorts,
        )

    def GetInputData(self, inInfo, port, idx):
        """Get input data object.

        This will convert :vtk:`vtkPointSet` to :vtk:`vtkPolyData`.

        Parameters
        ----------
        inInfo : :vtk:`vtkInformation`
            The information object associated with the input port.

        port : int
            The index of the input port.

        idx : int
            The index of the data object within the input port.

        Returns
        -------
        :vtk:`vtkDataObject`
            The input data object.

        """
        inp = wrap(_vtk.VTKPythonAlgorithmBase.GetInputData(self, inInfo, port, idx))
        if isinstance(inp, pv.PointSet):
            return inp.cast_to_polydata()
        return inp

    # THIS IS CRUCIAL to preserve data type through filter
    def RequestDataObject(self, _request, inInfo, outInfo) -> int:
        """Preserve data type.

        Parameters
        ----------
        _request : :vtk:`vtkInformation`
            The request object for the filter.

        inInfo : :vtk:`vtkInformationVector`
            The input information vector for the filter.

        outInfo : :vtk:`vtkInformationVector`
            The output information vector for the filter.

        Returns
        -------
        int
            Returns 1 if successful.

        """
        class_name = self.GetInputData(inInfo, 0, 0).GetClassName()
        if class_name == 'vtkPointSet':
            # See https://gitlab.kitware.com/vtk/vtk/-/issues/18771
            self.OutputType = 'vtkPolyData'
        else:
            self.OutputType = class_name
        self.FillOutputPortInformation(0, outInfo.GetInformationObject(0))
        return 1


def _resolve_output_type(output_type: str | type) -> str:
    """Resolve an output type to a VTK class name string.

    Parameters
    ----------
    output_type : str | type[pv.DataSet]
        Output type specification.  Accepts a VTK class name string (e.g.
        ``'vtkPolyData'``) or a PyVista :class:`~pyvista.DataSet` subclass
        (e.g. ``pv.PolyData``).

    Returns
    -------
    str
        VTK class name (e.g. ``'vtkPolyData'``).

    """
    if isinstance(output_type, str):
        return output_type
    try:
        is_dataset_subclass = issubclass(output_type, pv.DataSet)
    except TypeError:
        is_dataset_subclass = False
    if is_dataset_subclass:
        return output_type().GetClassName()
    msg = (
        f'Invalid output_type: {output_type!r}. '
        'Expected a VTK class name string or a pyvista.DataSet subclass.'
    )
    raise TypeError(msg)


class SourceAlgorithm(_NoNewAttrMixin, DisableVtkSnakeCase, _vtk.VTKPythonAlgorithmBase):
    """Algorithm that generates data from a callable with no input.

    The callable is invoked on each :meth:`RequestData` and must return a
    :class:`~pyvista.DataSet`.

    Parameters
    ----------
    generator : callable
        ``generator() -> dataset``.  Called each time the pipeline requests
        data.

    output_type : str | type[pyvista.DataSet], default: :class:`pyvista.UnstructuredGrid`
        Output type.  Accepts a VTK class name string (e.g.
        ``'vtkPolyData'``) or a PyVista :class:`~pyvista.DataSet` subclass
        (e.g. :class:`pyvista.PolyData`).

    """

    def __init__(
        self,
        generator: Callable[[], DataSet],
        output_type: str | type = pv.UnstructuredGrid,
    ):
        """Initialize algorithm."""
        resolved = _resolve_output_type(output_type)
        _vtk.VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=0,
            nOutputPorts=1,
            outputType=resolved,
        )
        self._generator = generator

    def RequestData(self, _request, _inInfo, outInfo) -> int:
        """Perform algorithm execution.

        Parameters
        ----------
        _request : :vtk:`vtkInformation`
            The request object.
        _inInfo : :vtk:`vtkInformationVector`
            Information about the input data (unused — no input ports).
        outInfo : :vtk:`vtkInformationVector`
            Information about the output data.

        Returns
        -------
        int
            1 on success.

        """
        try:
            out = self.GetOutputData(outInfo, 0)
            out.ShallowCopy(self._generator())
        except Exception:  # pragma: no cover
            traceback.print_exc()
            raise
        return 1


class CallbackFilterAlgorithm(PreserveTypeAlgorithmBase):
    """Algorithm that delegates processing to a user-supplied callable.

    The callable receives a :class:`~pyvista.DataSet` (the wrapped input) and
    must return a :class:`~pyvista.DataSet` of the appropriate type.

    By default the output type is preserved from the input (via
    :class:`PreserveTypeAlgorithmBase`). Pass ``output_type`` to override.

    Parameters
    ----------
    callback : callable
        ``callback(dataset) -> dataset``.  Called on each
        :meth:`RequestData` invocation with the wrapped input.

    output_type : str | type[pyvista.DataSet] | None, default: ``None``
        Fixed output type. Accepts a VTK class name string (e.g.
        ``'vtkPolyData'``) or a PyVista :class:`~pyvista.DataSet` subclass
        (e.g. :class:`pyvista.PolyData`). When ``None``, the output type is
        inferred from the input.

    nInputPorts : int, default: 1
        Number of input ports.

    nOutputPorts : int, default: 1
        Number of output ports.

    """

    def __init__(  # noqa: PLR0917
        self,
        callback: Callable[[DataSet], DataSet],
        output_type: str | type | None = None,
        nInputPorts: int = 1,
        nOutputPorts: int = 1,
    ):
        """Initialize algorithm."""
        if output_type is not None:
            self._fixed_output_type: str | None = _resolve_output_type(output_type)
        else:
            self._fixed_output_type = None
        if self._fixed_output_type is not None:
            _vtk.VTKPythonAlgorithmBase.__init__(
                self,
                nInputPorts=nInputPorts,
                nOutputPorts=nOutputPorts,
                outputType=self._fixed_output_type,
            )
        else:
            super().__init__(nInputPorts=nInputPorts, nOutputPorts=nOutputPorts)
        self._callback = callback

    def RequestDataObject(self, _request, inInfo, outInfo) -> int:
        """Preserve or override data type.

        Parameters
        ----------
        _request : :vtk:`vtkInformation`
            The request object for the filter.
        inInfo : :vtk:`vtkInformationVector`
            The input information vector for the filter.
        outInfo : :vtk:`vtkInformationVector`
            The output information vector for the filter.

        Returns
        -------
        int
            Returns 1 if successful.

        """
        if self._fixed_output_type is not None:
            return 1
        return super().RequestDataObject(_request, inInfo, outInfo)

    def RequestData(self, _request, inInfo, outInfo) -> int:
        """Perform algorithm execution.

        Parameters
        ----------
        _request : :vtk:`vtkInformation`
            The request object.
        inInfo : :vtk:`vtkInformationVector`
            Information about the input data.
        outInfo : :vtk:`vtkInformationVector`
            Information about the output data.

        Returns
        -------
        int
            1 on success.

        """
        try:
            inp = self.GetInputData(inInfo, 0, 0)
            out = self.GetOutputData(outInfo, 0)
            result = self._callback(wrap(inp))
            out.ShallowCopy(result)
        except Exception:  # pragma: no cover
            traceback.print_exc()
            raise
        return 1


class ActiveScalarsAlgorithm(PreserveTypeAlgorithmBase):
    """Algorithm to control active scalars.

    The output of this filter is a shallow copy of the input data
    set with the active scalars set as specified.

    Parameters
    ----------
    name : str
        Name of scalars used to set as active on the output mesh.
        Accepts a string name of an array that is present on the mesh.
        Array should be sized as a single vector.

    preference : str, default: 'point'
        When ``mesh.n_points == mesh.n_cells`` and setting
        scalars, this parameter sets how the scalars will be
        mapped to the mesh.  The default, ``'point'``, causes the
        scalars to be associated with the mesh points.  Can be
        either ``'point'`` or ``'cell'``.

    """

    def __init__(self, name: str, preference: PointLiteral | CellLiteral = 'point'):
        """Initialize algorithm."""
        super().__init__()
        self._scalars_name = name
        self._preference: PointLiteral | CellLiteral = preference

    @property
    def scalars_name(self) -> str:  # numpydoc ignore=RT01
        """Return or set the name of the active scalars array."""
        return self._scalars_name

    @scalars_name.setter
    def scalars_name(self, name: str) -> None:
        if name != self._scalars_name:
            self._scalars_name = name
            self.Modified()

    @property
    def preference(self) -> PointLiteral | CellLiteral:  # numpydoc ignore=RT01
        """Return or set the preferred field association (``'point'`` or ``'cell'``)."""
        return self._preference

    @preference.setter
    def preference(self, preference: PointLiteral | CellLiteral) -> None:
        if preference != self._preference:
            self._preference = preference
            self.Modified()

    def RequestData(self, _request, inInfo, outInfo) -> int:
        """Perform algorithm execution.

        Parameters
        ----------
        _request : :vtk:`vtkInformation`
            The request object.
        inInfo : :vtk:`vtkInformationVector`
            Information about the input data.
        outInfo : :vtk:`vtkInformationVector`
            Information about the output data.

        Returns
        -------
        int
            1 on success.

        """
        try:
            inp = self.GetInputData(inInfo, 0, 0)
            out = self.GetOutputData(outInfo, 0)
            out.ShallowCopy(inp)
            # Set active scalars directly via VTK API on the output object.
            # Using wrap(out) would create a new VTK object rather than
            # wrapping the existing one, so changes would be lost.
            if self.preference == 'cell':
                out.GetCellData().SetActiveScalars(self.scalars_name)
            else:
                out.GetPointData().SetActiveScalars(self.scalars_name)
        except Exception:  # pragma: no cover
            traceback.print_exc()
            raise
        return 1


class SmoothShadingAlgorithm(_NoNewAttrMixin, DisableVtkSnakeCase, _vtk.VTKPythonAlgorithmBase):
    """Algorithm to compute point normals for smooth shading.

    The output is always a :vtk:`vtkPolyData`. Non-polydata inputs have their
    external surface extracted first. Normals are computed via
    :meth:`~pyvista.PolyDataFilters.compute_normals`, with ``split_vertices``
    controlled by ``split_sharp_edges`` to get crisp edges at feature-angle
    boundaries.

    The output carries a ``vtkOriginalPointIds`` point-data array that maps
    each output point back to its index in the original input mesh. Callers
    that need to remap input-length arrays onto the (potentially longer)
    output topology — for example, raw numpy scalars passed to ``add_mesh``
    — can do so via this tracker.

    Parameters
    ----------
    split_sharp_edges : bool, default: False
        Whether to use feature-angle splitting when computing normals.
        When ``True``, shared vertices on sharp edges are duplicated so
        each face has its own normal, producing crisp feature lines.

    feature_angle : float, default: 30.0
        Angle (in degrees) above which an edge is considered sharp. Only
        used when ``split_sharp_edges`` is ``True``.

    """

    ORIGINAL_POINT_IDS_NAME = 'vtkOriginalPointIds'

    def __init__(
        self,
        split_sharp_edges: bool = False,  # noqa: FBT001, FBT002
        feature_angle: float = 30.0,
    ):
        """Initialize algorithm."""
        _vtk.VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=1,
            nOutputPorts=1,
            outputType='vtkPolyData',
        )
        self._split_sharp_edges = split_sharp_edges
        self._feature_angle = feature_angle

    @property
    def split_sharp_edges(self) -> bool:  # numpydoc ignore=RT01
        """Return or set whether to split sharp edges when computing normals."""
        return self._split_sharp_edges

    @split_sharp_edges.setter
    def split_sharp_edges(self, value: bool) -> None:
        if value != self._split_sharp_edges:
            self._split_sharp_edges = value
            self.Modified()

    @property
    def feature_angle(self) -> float:  # numpydoc ignore=RT01
        """Return or set the feature angle (degrees) for splitting sharp edges."""
        return self._feature_angle

    @feature_angle.setter
    def feature_angle(self, value: float) -> None:
        if value != self._feature_angle:
            self._feature_angle = value
            self.Modified()

    def RequestData(self, _request, inInfo, outInfo) -> int:
        """Perform algorithm execution.

        Parameters
        ----------
        _request : :vtk:`vtkInformation`
            The request object.
        inInfo : :vtk:`vtkInformationVector`
            Information about the input data.
        outInfo : :vtk:`vtkInformationVector`
            Information about the output data.

        Returns
        -------
        int
            1 on success.

        """
        try:
            inp = self.GetInputData(inInfo, 0, 0)
            out = self.GetOutputData(outInfo, 0)

            wrapped = wrap(inp)
            if isinstance(wrapped, pv.PointSet):
                wrapped = wrapped.cast_to_polydata()

            if not isinstance(wrapped, pv.DataSet) or wrapped.n_points == 0:
                return 1

            # Respect user-provided point normals on a polydata input when no
            # splitting is requested: pass the input through unchanged so the
            # custom normals survive.  Splitting always re-runs compute_normals
            # because it needs to know which points to duplicate.
            # No ``vtkOriginalPointIds`` tracker is needed here because the
            # topology is unchanged (n_points stays the same).
            if (
                isinstance(wrapped, pv.PolyData)
                and not self._split_sharp_edges
                and wrapped.point_data.active_normals is not None
            ):
                out.ShallowCopy(wrapped)
                return 1

            # Extract the external surface when the input is not polydata.
            # ``pass_pointid=True`` attaches ``vtkOriginalPointIds`` that maps
            # each surface point back to its index in the input mesh. For
            # polydata inputs we install an identity tracker on a shallow
            # copy so that compute_normals(split_vertices=True) will split
            # it alongside the points, giving us a chained mapping for free.
            if isinstance(wrapped, pv.PolyData):
                surface = wrapped.copy(deep=False)
                surface.point_data[self.ORIGINAL_POINT_IDS_NAME] = np.arange(
                    surface.n_points, dtype=pv.ID_TYPE
                )
            else:
                surface = wrapped.extract_surface(
                    algorithm=None,
                    pass_pointid=True,
                    pass_cellid=True,
                )
                if surface.n_points == 0:
                    out.ShallowCopy(surface)
                    return 1

            try:
                result = surface.compute_normals(
                    cell_normals=False,
                    split_vertices=self._split_sharp_edges,
                    feature_angle=self._feature_angle,
                )
            except TypeError as exc:
                if 'Normals cannot be computed' in repr(exc):
                    # No renderable 2D cells (point cloud, lines, etc.).
                    # Pass the surface through so downstream stages still
                    # see a valid polydata.
                    out.ShallowCopy(surface)
                    return 1
                raise

            # ``compute_normals(split_vertices=True)`` leaves behind a
            # ``pyvistaOriginalPointIds`` helper array that mirrors our own
            # ``vtkOriginalPointIds``. Drop it so downstream code isn't
            # surprised by the duplicate tracker.
            if 'pyvistaOriginalPointIds' in result.point_data:
                del result.point_data['pyvistaOriginalPointIds']

            out.ShallowCopy(result)
        except Exception:  # pragma: no cover
            traceback.print_exc()
            raise
        return 1


class PointSetToPolyDataAlgorithm(
    _NoNewAttrMixin, DisableVtkSnakeCase, _vtk.VTKPythonAlgorithmBase
):
    """Algorithm to cast PointSet to PolyData.

    This is implemented with :func:`pyvista.PointSet.cast_to_polydata`.

    """

    def __init__(self):
        """Initialize algorithm."""
        _vtk.VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=1,
            nOutputPorts=1,
            inputType='vtkPointSet',
            outputType='vtkPolyData',
        )

    def RequestData(self, _request, inInfo, outInfo) -> int:
        """Perform algorithm execution.

        Parameters
        ----------
        _request : :vtk:`vtkInformation`
            Information associated with the request.
        inInfo : :vtk:`vtkInformationVector`
            Information about the input data.
        outInfo : :vtk:`vtkInformationVector`
            Information about the output data.

        Returns
        -------
        int
            1 when successful.

        """
        try:
            inp = wrap(self.GetInputData(inInfo, 0, 0))
            out = self.GetOutputData(outInfo, 0)
            output = inp.cast_to_polydata(deep=False)
            out.ShallowCopy(output)
        except Exception:  # pragma: no cover
            traceback.print_exc()
            raise
        return 1


class AddIDsAlgorithm(PreserveTypeAlgorithmBase):
    """Algorithm to add point or cell IDs.

    Output of this filter is a shallow copy of the input with
    point and/or cell ID arrays added.

    Parameters
    ----------
    point_ids : bool, default: True
        Whether to add point IDs.

    cell_ids : bool, default: True
        Whether to add cell IDs.

    Raises
    ------
    ValueError
        If neither point IDs nor cell IDs are set.

    """

    @_deprecate_positional_args
    def __init__(self, point_ids: bool = True, cell_ids: bool = True):  # noqa: FBT001, FBT002
        """Initialize algorithm."""
        super().__init__()
        if not point_ids and not cell_ids:  # pragma: no cover
            msg = 'IDs must be set for points or cells or both.'
            raise ValueError(msg)
        self.point_ids = point_ids
        self.cell_ids = cell_ids

    def RequestData(self, _request, inInfo, outInfo) -> int:
        """Perform algorithm execution.

        Parameters
        ----------
        _request : :vtk:`vtkInformation`
            Information associated with the request.
        inInfo : :vtk:`vtkInformationVector`
            Information about the input data.
        outInfo : :vtk:`vtkInformationVector`
            Information about the output data.

        Returns
        -------
        int
            Returns 1 if the algorithm was successful.

        Raises
        ------
        Exception
            If the algorithm fails to execute properly.

        """
        try:
            inp = self.GetInputData(inInfo, 0, 0)
            out = self.GetOutputData(outInfo, 0)
            out.ShallowCopy(inp)
            # AddArray does not modify active scalars (unlike PyVista's
            # __setitem__), so no fixup is needed after insertion.
            if self.point_ids:
                n = out.GetNumberOfPoints()
                arr = _vtk.numpy_to_vtk(np.arange(n, dtype=int))
                arr.SetName('point_ids')
                out.GetPointData().AddArray(arr)
            if self.cell_ids:
                n = out.GetNumberOfCells()
                arr = _vtk.numpy_to_vtk(np.arange(n, dtype=int))
                arr.SetName('cell_ids')
                out.GetCellData().AddArray(arr)
        except Exception:  # pragma: no cover
            traceback.print_exc()
            raise
        return 1


class CrinkleAlgorithm(_NoNewAttrMixin, DisableVtkSnakeCase, _vtk.VTKPythonAlgorithmBase):
    """Algorithm to crinkle cell IDs."""

    def __init__(self):
        """Initialize algorithm."""
        super().__init__(
            nInputPorts=2,
            outputType='vtkUnstructuredGrid',
        )

    def RequestData(self, _request, inInfo, outInfo) -> int:
        """Perform algorithm execution based on the input data and produce the output.

        Parameters
        ----------
        _request : :vtk:`vtkInformation`
            The request information associated with the algorithm.
        inInfo : :vtk:`vtkInformationVector`
            Information vector describing the input data.
        outInfo : :vtk:`vtkInformationVector`
            Information vector where the output data should be placed.

        Returns
        -------
        int
            Status of the execution. Returns 1 on successful completion.

        """
        try:
            clipped = wrap(self.GetInputData(inInfo, 0, 0))
            source = wrap(self.GetInputData(inInfo, 1, 0))
            out = self.GetOutputData(outInfo, 0)
            output = source.extract_cells(np.unique(clipped.cell_data['cell_ids']))
            out.ShallowCopy(output)
        except Exception:  # pragma: no cover
            traceback.print_exc()
            raise
        return 1


@_deprecate_positional_args(allowed=['inp'])
def outline_algorithm(inp, generate_faces: bool = False):  # noqa: FBT001, FBT002
    """Add :vtk:`vtkOutlineFilter` to pipeline.

    Parameters
    ----------
    inp : pyvista.Common
        Input data to be filtered.
    generate_faces : bool, default: False
        Whether to generate faces for the outline.

    Returns
    -------
    :vtk:`vtkOutlineFilter`
        Outline filter applied to the input data.

    """
    alg = _vtk.vtkOutlineFilter()
    set_algorithm_input(alg, inp)
    alg.SetGenerateFaces(generate_faces)
    return alg


def source_algorithm(
    generator: Callable[[], DataSet],
    output_type: str | type = pv.UnstructuredGrid,
):
    """Create a source algorithm that generates data from a callable.

    Unlike filter algorithms, a source has no input port — it produces
    data from scratch via *generator*.

    Parameters
    ----------
    generator : callable
        ``generator() -> dataset``.

    output_type : str | type[pyvista.DataSet], default: :class:`pyvista.UnstructuredGrid`
        Output type.  Accepts a VTK class name string or a PyVista
        :class:`~pyvista.DataSet` subclass.

    Returns
    -------
    SourceAlgorithm
        The source algorithm.

    """
    return SourceAlgorithm(generator=generator, output_type=output_type)


def callback_algorithm(
    inp,
    callback: Callable[[DataSet], DataSet],
    output_type: str | type | None = None,
):
    """Add a filter that delegates to a user-supplied callable.

    Parameters
    ----------
    inp : pyvista.DataSet | :vtk:`vtkAlgorithm`
        Input data or algorithm.

    callback : callable
        ``callback(dataset) -> dataset``.

    output_type : str | type[pyvista.DataSet] | None, default: ``None``
        Fixed output type.  Accepts a VTK class name string or a PyVista
        :class:`~pyvista.DataSet` subclass.  When ``None``, the output
        type matches the input type.

    Returns
    -------
    CallbackFilterAlgorithm
        The callback filter wired to *inp*.

    """
    alg = CallbackFilterAlgorithm(callback=callback, output_type=output_type)
    set_algorithm_input(alg, inp)
    return alg


@_deprecate_positional_args(allowed=['inp'])
def extract_surface_algorithm(  # noqa: PLR0917
    inp,
    pass_pointid: bool = False,  # noqa: FBT001, FBT002
    pass_cellid: bool = False,  # noqa: FBT001, FBT002
    nonlinear_subdivision=1,
):
    """Add :vtk:`vtkDataSetSurfaceFilter` to pipeline.

    Parameters
    ----------
    inp : pyvista.Common
        Input data to be filtered.
    pass_pointid : bool, default: False
        If ``True``, pass point IDs to the output.
    pass_cellid : bool, default: False
        If ``True``, pass cell IDs to the output.
    nonlinear_subdivision : int, default: 1
        Level of nonlinear subdivision.

    Returns
    -------
    :vtk:`vtkDataSetSurfaceFilter`
        Surface filter applied to the input data.

    """
    surf_filter = _vtk.vtkDataSetSurfaceFilter()
    surf_filter.SetPassThroughPointIds(pass_pointid)
    surf_filter.SetPassThroughCellIds(pass_cellid)
    if nonlinear_subdivision != 1:
        surf_filter.SetNonlinearSubdivisionLevel(nonlinear_subdivision)
    set_algorithm_input(surf_filter, inp)
    return surf_filter


def active_scalars_algorithm(inp, name, preference='point'):
    """Add a filter that sets the active scalars.

    Parameters
    ----------
    inp : pyvista.Common
        Input data to be filtered.
    name : str
        Name of the scalars to set as active.
    preference : str, default: 'point'
        Preference for the scalars to be set as active. Options are 'point', 'cell', or 'field'.

    Returns
    -------
    :vtk:`vtkAlgorithm`
        Active scalars filter applied to the input data.

    """
    alg = ActiveScalarsAlgorithm(
        name=name,
        preference=preference,
    )
    set_algorithm_input(alg, inp)
    return alg


@_deprecate_positional_args(allowed=['inp'])
def smooth_shading_algorithm(
    inp,
    split_sharp_edges: bool = False,  # noqa: FBT001, FBT002
    feature_angle: float = 30.0,
):
    """Add a filter that computes point normals for smooth shading.

    Parameters
    ----------
    inp : pyvista.DataSet | :vtk:`vtkAlgorithm`
        Input data or algorithm.

    split_sharp_edges : bool, default: False
        Whether to split sharp edges when computing normals.

    feature_angle : float, default: 30.0
        Angle (in degrees) above which an edge is considered sharp. Only
        used when ``split_sharp_edges`` is ``True``.

    Returns
    -------
    SmoothShadingAlgorithm
        Smooth shading filter applied to the input data.

    """
    alg = SmoothShadingAlgorithm(
        split_sharp_edges=split_sharp_edges,
        feature_angle=feature_angle,
    )
    set_algorithm_input(alg, inp)
    return alg


def pointset_to_polydata_algorithm(inp):
    """Add a filter that casts PointSet to PolyData.

    Parameters
    ----------
    inp : pyvista.PointSet
        Input point set to be cast to PolyData.

    Returns
    -------
    :vtk:`vtkAlgorithm`
        Filter that casts the input PointSet to PolyData.

    """
    alg = PointSetToPolyDataAlgorithm()
    set_algorithm_input(alg, inp)
    return alg


@_deprecate_positional_args(allowed=['inp'])
def add_ids_algorithm(inp, point_ids: bool = True, cell_ids: bool = True):  # noqa: FBT001, FBT002
    """Add a filter that adds point and/or cell IDs.

    Parameters
    ----------
    inp : pyvista.DataSet
        The input data to which the IDs will be added.
    point_ids : bool, default: True
        If ``True``, point IDs will be added to the input data.
    cell_ids : bool, default: True
        If ``True``, cell IDs will be added to the input data.

    Returns
    -------
    AddIDsAlgorithm
        AddIDsAlgorithm filter.

    """
    alg = AddIDsAlgorithm(point_ids=point_ids, cell_ids=cell_ids)
    set_algorithm_input(alg, inp)
    return alg


def crinkle_algorithm(clip, source):
    """Add a filter that crinkles a clip.

    Parameters
    ----------
    clip : pyvista.DataSet
        The input data to be crinkled.
    source : pyvista.DataSet
        The source of the crinkle.

    Returns
    -------
    CrinkleAlgorithm
        CrinkleAlgorithm filter.

    """
    alg = CrinkleAlgorithm()
    set_algorithm_input(alg, clip, 0)
    set_algorithm_input(alg, source, 1)
    return alg


@_deprecate_positional_args(allowed=['inp'])
def cell_data_to_point_data_algorithm(inp, pass_cell_data: bool = False):  # noqa: FBT001, FBT002
    """Add a filter that converts cell data to point data.

    Parameters
    ----------
    inp : pyvista.DataSet
        The input data whose cell data will be converted to point data.
    pass_cell_data : bool, default: False
        If ``True``, the original cell data will be passed to the output.

    Returns
    -------
    :vtk:`vtkCellDataToPointData`
        The :vtk:`vtkCellDataToPointData` filter.

    """
    alg = _vtk.vtkCellDataToPointData()
    alg.SetPassCellData(pass_cell_data)
    set_algorithm_input(alg, inp)
    return alg


@_deprecate_positional_args(allowed=['inp'])
def point_data_to_cell_data_algorithm(inp, pass_point_data: bool = False):  # noqa: FBT001, FBT002
    """Add a filter that converts point data to cell data.

    Parameters
    ----------
    inp : pyvista.DataSet
        The input data whose point data will be converted to cell data.
    pass_point_data : bool, default: False
        If ``True``, the original point data will be passed to the output.

    Returns
    -------
    :vtk:`vtkPointDataToCellData`
        :vtk:`vtkPointDataToCellData` algorithm.

    """
    alg = _vtk.vtkPointDataToCellData()
    alg.SetPassPointData(pass_point_data)
    set_algorithm_input(alg, inp)
    return alg


def triangulate_algorithm(inp):
    """Triangulate the input data.

    Parameters
    ----------
    inp : :vtk:`vtkDataObject`
        The input data to be triangulated.

    Returns
    -------
    :vtk:`vtkTriangleFilter`
        The triangle filter that has been applied to the input data.

    """
    trifilter = _vtk.vtkTriangleFilter()
    trifilter.PassVertsOff()
    trifilter.PassLinesOff()
    set_algorithm_input(trifilter, inp)
    return trifilter


def decimation_algorithm(inp, target_reduction):
    """Decimate the input data to the target reduction.

    Parameters
    ----------
    inp : :vtk:`vtkDataObject`
        The input data to be decimated.
    target_reduction : float
        The target reduction amount, as a fraction of the original data.

    Returns
    -------
    :vtk:`vtkQuadricDecimation`
        The decimation algorithm that has been applied to the input data.

    """
    alg = _vtk.vtkQuadricDecimation()
    alg.SetTargetReduction(target_reduction)
    set_algorithm_input(alg, inp)
    return alg
