"""Fine-grained control of reading data files."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Any
from xml.etree import ElementTree

import pyvista
from pyvista import _vtk
from pyvista.utilities import get_ext, wrap


def get_reader(filename):
    """Get a reader for fine-grained control of reading data files.

    Supported file types and Readers:

    +----------------+---------------------------------------------+
    | File Extension | Class                                       |
    +================+=============================================+
    | ``.case``      | :class:`pyvista.EnSightReader`              |
    +----------------+---------------------------------------------+
    | ``.facet``     | :class:`pyvista.FacetReader`                |
    +----------------+---------------------------------------------+
    | ``.foam``      | :class:`pyvista.OpenFOAMReader`             |
    +----------------+---------------------------------------------+
    | ``.g``         | :class:`pyvista.BYUReader`                  |
    +----------------+---------------------------------------------+
    | ``.obj``       | :class:`pyvista.OBJReader`                  |
    +----------------+---------------------------------------------+
    | ``.p3d``       | :class:`pyvista.Plot3DMetaReader`           |
    +----------------+---------------------------------------------+
    | ``.ply``       | :class:`pyvista.PLYReader`                  |
    +----------------+---------------------------------------------+
    | ``.pvd``       | :class:`pyvista.PVDReader`                  |
    +----------------+---------------------------------------------+
    | ``.pvti``      | :class:`pyvista.XMLPImageDataReader`        |
    +----------------+---------------------------------------------+
    | ``.pvtk``      | :class:`pyvista.VTKPDataSetReader`          |
    +----------------+---------------------------------------------+
    | ``.pvtr``      | :class:`pyvista.XMLPRectilinearGridReader`  |
    +----------------+---------------------------------------------+
    | ``.pvtu``      | :class:`pyvista.XMLPUnstructuredGridReader` |
    +----------------+---------------------------------------------+
    | ``.stl``       | :class:`pyvista.STLReader`                  |
    +----------------+---------------------------------------------+
    | ``.tri``       | :class:`pyvista.BinaryMarchingCubesReader`  |
    +----------------+---------------------------------------------+
    | ``.vti``       | :class:`pyvista.XMLImageDataReader`         |
    +----------------+---------------------------------------------+
    | ``.vtk``       | :class:`pyvista.VTKDataSetReader`           |
    +----------------+---------------------------------------------+
    | ``.vtm``       | :class:`pyvista.XMLMultiBlockDataReader`    |
    +----------------+---------------------------------------------+
    | ``.vtmb``      | :class:`pyvista.XMLMultiBlockDataReader`    |
    +----------------+---------------------------------------------+
    | ``.vtp``       | :class:`pyvista.XMLPolyDataReader`          |
    +----------------+---------------------------------------------+
    | ``.vtr``       | :class:`pyvista.XMLRectilinearGridReader`   |
    +----------------+---------------------------------------------+
    | ``.vts``       | :class:`pyvista.XMLStructuredGridReader`    |
    +----------------+---------------------------------------------+
    | ``.vtu``       | :class:`pyvista.XMLUnstructuredGridReader`  |
    +----------------+---------------------------------------------+

    Parameters
    ----------
    filename : str
        The string path to the file to read.

    Returns
    -------
    pyvista.BaseReader
        A subclass of :class:`pyvista.BaseReader` is returned based on file type.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_human(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'Human.vtp'
    >>> reader = pyvista.get_reader(filename)
    >>> reader  # doctest: +ELLIPSIS
    XMLPolyDataReader('.../Human.vtp')
    >>> mesh = reader.read()
    >>> mesh # doctest: +ELLIPSIS
    PolyData ...
    >>> mesh.plot(color='tan')

    """
    ext = get_ext(filename)

    try:
        Reader = CLASS_READERS[ext]
    except KeyError:
        raise ValueError(f"{ext} not supported")

    return Reader(filename)


class BaseReader:
    """The Base Reader class.

    The base functionality includes reading data from a file,
    and allowing access to the underlying vtk reader. See
    :func:`pyvista.get_reader` for an example using
    a built-in subclass.

    Parameters
    ----------
    filename : str
        Path of the file to read.
    """

    _class_reader: Any = None

    def __init__(self, filename):
        """Initialize Reader by setting filename."""
        self._reader = self._class_reader()
        self.filename = filename
        self._set_filename(filename)

    def __repr__(self):
        """Representation of a Reader object."""
        return f"{self.__class__.__name__}('{self.filename}')"

    @property
    def reader(self):
        """Return the vtk Reader object.

        Returns
        -------
        A vtk Reader object

        """
        if self._reader is None:
            raise NotImplementedError
        return self._reader

    def _set_filename(self, filename):
        """Set filename and update reader."""
        # Private method since changing file type requires a
        # different subclass.
        self.filename = filename
        self.reader.SetFileName(filename)
        self._update_information()

    def read(self):
        """Read data in file.

        Returns
        -------
        pyvista.DataSet
            PyVista Dataset.
        """
        self._update()
        data = wrap(self.reader.GetOutputDataObject(0))
        data._post_file_load_processing()
        return data

    def _update(self):
        """Update reader."""
        self.reader.Update()

    def _update_information(self):
        self.reader.UpdateInformation()


class PointCellDataSelection:
    """Mixin for readers that support data array selections.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_backward_facing_step(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'foam_case_0_0_0_0.case'
    >>> reader = pyvista.get_reader(filename)
    >>> reader  # doctest: +ELLIPSIS
    EnSightReader('.../foam_case_0_0_0_0.case')
    >>> reader.cell_array_names
    ['v2', 'nut', 'k', 'nuTilda', 'p', 'omega', 'f', 'epsilon', 'U']
    >>> reader.point_array_names
    []
    >>> reader.all_cell_arrays_status  # doctest: +NORMALIZE_WHITESPACE
    {'v2': True, 'nut': True, 'k': True, 'nuTilda': True, 'p': True, 'omega': True, 'f': True, 'epsilon': True, 'U': True}
    >>> reader.disable_all_cell_arrays()
    >>> reader.enable_cell_array('U')
    >>> mesh = reader.read()  # MultiBlock mesh
    >>> mesh[0].array_names
    ['U']

    """

    @property
    def number_point_arrays(self):
        """Return the number of point arrays.

        Returns
        -------
        int

        """
        return self.reader.GetNumberOfPointArrays()

    @property
    def point_array_names(self):
        """Return the list of all point array names.

        Returns
        -------
        list[int]

        """
        return [self.reader.GetPointArrayName(i) for i in range(self.number_point_arrays)]

    def enable_point_array(self, name):
        """Enable point array with name.

        Parameters
        ----------
        name : str
            Point array name.

        """
        self.reader.SetPointArrayStatus(name, 1)

    def disable_point_array(self, name):
        """Disable point array with name.

        Parameters
        ----------
        name : str
            Point array name.

        """
        self.reader.SetPointArrayStatus(name, 0)

    def point_array_status(self, name):
        """Get status of point array with name.

        Parameters
        ----------
        name : str
            Point array name.

        Returns
        -------
        bool
            Whether reading the cell array is enabled.

        """
        if self.reader.GetPointArrayStatus(name):
            return True
        return False

    def enable_all_point_arrays(self):
        """Enable all point arrays."""
        for name in self.point_array_names:
            self.enable_point_array(name)

    def disable_all_point_arrays(self):
        """Disable all point arrays."""
        for name in self.point_array_names:
            self.disable_point_array(name)

    @property
    def all_point_arrays_status(self):
        """Return the status of all point arrays.

        Returns
        -------
        dict[str, bool]

        """
        return {name: self.point_array_status(name) for name in self.point_array_names}

    @property
    def number_cell_arrays(self):
        """Return the number of cell arrays.

        Returns
        -------
        int

        """
        return self.reader.GetNumberOfCellArrays()

    @property
    def cell_array_names(self):
        """Return the list of all cell array names.

        Returns
        -------
        list[str]

        """
        return [self.reader.GetCellArrayName(i) for i in range(self.number_cell_arrays)]

    def enable_cell_array(self, name):
        """Enable cell array with name.

        Parameters
        ----------
        name : str
            Cell array name.

        """
        self.reader.SetCellArrayStatus(name, 1)

    def disable_cell_array(self, name):
        """Disable cell array with name.

        Parameters
        ----------
        name : str
            Cell array name.

        """
        self.reader.SetCellArrayStatus(name, 0)

    def cell_array_status(self, name):
        """Get status of cell array with name.

        Parameters
        ----------
        name : str
            Cell array name.

        Returns
        -------
        bool
            Whether reading the cell array is enabled.

        """
        if self.reader.GetCellArrayStatus(name):
            return True
        return False

    def enable_all_cell_arrays(self):
        """Enable all cell arrays."""
        for name in self.cell_array_names:
            self.enable_cell_array(name)

    def disable_all_cell_arrays(self):
        """Disable all cell arrays."""
        for name in self.cell_array_names:
            self.disable_cell_array(name)

    @property
    def all_cell_arrays_status(self):
        """Return the status of all cell arrays.

        Returns
        -------
        dict[str, bool]
        """
        return {name: self.cell_array_status(name) for name in self.cell_array_names}



class TimeReader(ABC):
    """Abstract class for readers supporting time."""

    @property
    @abstractmethod
    def number_time_points(self):
        """Return number of time points or iterations available to read.

        Returns
        -------
        int

        """

    @abstractmethod
    def time_point_value(self, time_point):
        """Value of time point or iteration by index.

        Parameters
        ----------
        time_point: int
            Time point index.

        Returns
        -------
        float

        """

    @property
    def time_values(self):
        """All time or iteration values.

        Returns
        -------
        list[float]

        """
        return [self.time_point_value(idx) for idx in range(self.number_time_points)]

    @property
    @abstractmethod
    def active_time_value(self):
        """Active time or iteration value.

        Returns
        -------
        float

        """

    @abstractmethod
    def set_active_time_value(self, time_value):
        """Set active time or iteration value.

        Parameters
        ----------
        time_value: float
            Time or iteration value to set as active.

        """

    @abstractmethod
    def set_active_time_point(self, time_point):
        """Set active time or iteration by index.

        Parameters
        ----------
        time_point: int
            Time or iteration point index for setting active time.

        """


class XMLImageDataReader(BaseReader, PointCellDataSelection):
    """XML Image Data Reader for .vti files."""

    _class_reader =_vtk.vtkXMLImageDataReader


class XMLPImageDataReader(BaseReader, PointCellDataSelection):
    """Parallel XML Image Data Reader for .pvti files."""

    _class_reader = _vtk.vtkXMLPImageDataReader


class XMLRectilinearGridReader(BaseReader, PointCellDataSelection):
    """XML RectilinearGrid Reader for .vtr files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_rectilinear_grid(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'RectilinearGrid.vtr'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> sliced_mesh = mesh.slice('y')
    >>> sliced_mesh.plot(scalars='Void Volume Fraction', cpos='xz',
    ...                  show_scalar_bar=False)

    """

    _class_reader = _vtk.vtkXMLRectilinearGridReader


class XMLPRectilinearGridReader(BaseReader, PointCellDataSelection):
    """Parallel XML RectilinearGrid Reader for .pvtr files."""

    _class_reader = _vtk.vtkXMLPRectilinearGridReader


class XMLUnstructuredGridReader(BaseReader, PointCellDataSelection):
    """XML UnstructuredGrid Reader for .vtu files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_notch_displacement(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'notch_disp.vtu'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(scalars="Nodal Displacement", component=0,
    ...           cpos='xy', show_scalar_bar=False)

    """

    _class_reader = _vtk.vtkXMLUnstructuredGridReader


class XMLPUnstructuredGridReader(BaseReader, PointCellDataSelection):
    """Parallel XML UnstructuredGrid Reader for .pvtu files."""

    _class_reader = _vtk.vtkXMLPUnstructuredGridReader


class XMLPolyDataReader(BaseReader, PointCellDataSelection):
    """XML PolyData Reader for .vtp files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_cow_head(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'cowHead.vtp'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(
    ...    cpos=((12, 3.5, -4.5), (4.5, 1.6, 0), (0, 1, 0.3)),
    ...    clim=[0, 100], show_scalar_bar=False
    ... )

    """

    _class_reader = _vtk.vtkXMLPolyDataReader


class XMLStructuredGridReader(BaseReader, PointCellDataSelection):
    """XML StructuredGrid Reader for .vts files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_structured_grid(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'StructuredGrid.vts'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(style='wireframe', line_width=4, show_scalar_bar=False)

    """

    _class_reader = _vtk.vtkXMLStructuredGridReader


class XMLMultiBlockDataReader(BaseReader, PointCellDataSelection):
    """XML MultiBlock Data Reader for .vtm or .vtmb files."""

    _class_reader = _vtk.vtkXMLMultiBlockDataReader

# skip pydocstyle D102 check since docstring is taken from TimeReader
class EnSightReader(BaseReader, PointCellDataSelection, TimeReader):
    """EnSight Reader for .case files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_cylinder_crossflow(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'cylinder_Re35.case'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(scalars="velocity", component=1, clim=[-20, 20],
    ...           cpos='xy', cmap='RdBu', show_scalar_bar=False)

    """

    _class_reader = _vtk.vtkGenericEnSightReader

    def _set_filename(self, filename):
        """Set filename and update reader."""
        # Private method since changing file type requires a
        # different subclass.
        self.filename = filename
        self.reader.SetCaseFileName(filename)
        self._update_information()

    @property
    def number_time_points(self):  # noqa: D102
        return self.reader.GetTimeSets().GetItem(0).GetSize()

    def time_point_value(self, time_point):  # noqa: D102
        return self.reader.GetTimeSets().GetItem(0).GetValue(time_point)

    @property
    def active_time_value(self):  # noqa: D102
        return self.reader.GetTimeValue()

    def set_active_time_value(self, time_value):  # noqa: D102
        if time_value not in self.time_values:
            raise ValueError(
                f"Not a valid time {time_value} from available time values: {self.reader_time_values}"
            )
        self.reader.SetTimeValue(time_value)

    def set_active_time_point(self, time_point):  # noqa: D102
        self.reader.SetTimeValue(self.time_point_value(time_point))


# skip pydocstyle D102 check since docstring is taken from TimeReader
class OpenFOAMReader(BaseReader, PointCellDataSelection, TimeReader):
    """OpenFOAM Reader for .foam files."""

    _class_reader = _vtk.vtkOpenFOAMReader

    def __init__(self, filename):
        """Initialize OpenFOAMReader.

        By default, pyvista enables all patch arrays.  This is a deviation
        from the vtk default.

        """
        super().__init__(filename)
        self.enable_all_patch_arrays()

    @property
    def number_time_points(self):  # noqa: D102
        return self.reader.GetTimeValues().GetNumberOfValues()

    def time_point_value(self, time_point):  # noqa: D102
        return self.reader.GetTimeValues().GetValue(time_point)

    @property
    def active_time_value(self):  # noqa: D102
        try:
            value = self.reader.GetTimeValue()
        except AttributeError as err:  # pragma: no cover
            raise AttributeError("Inspecting active time value only supported "
                      "for vtk versions >9.1.0") from err
        return value

    def set_active_time_value(self, time_value):  # noqa: D102
        if time_value not in self.time_values:
            raise ValueError(
                f"Not a valid time {time_value} from available time values: {self.reader_time_values}"
            )
        self.reader.SetTimeValue(time_value)

    def set_active_time_point(self, time_point):  # noqa: D102
        self.reader.SetTimeValue(self.time_point_value(time_point))

    @property
    def cell_to_point_creation(self):
        """Whether cell data is translated to point data when read.

        Returns
        -------
        bool
            If ``True``, translate cell data to point data.

        Warnings
        --------
        When ``True``, cell and point data arrays will have
        duplicate names.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cavity(load=False)
        >>> reader = pyvista.OpenFOAMReader(filename)
        >>> reader.cell_to_point_creation = False
        >>> reader.cell_to_point_creation
        False

        """
        return bool(self.reader.GetCreateCellToPoint())

    @cell_to_point_creation.setter
    def cell_to_point_creation(self, value):
        if value:
            self.reader.CreateCellToPointOn()
        else:
            self.reader.CreateCellToPointOff()

    @property
    def number_patch_arrays(self):
        """Return number of patch arrays in dataset.
        
        Returns
        -------
        int

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cavity(load=False)
        >>> reader = pyvista.OpenFOAMReader(filename)
        >>> reader.number_patch_arrays
        4
        
        """
        return self.reader.GetNumberOfPatchArrays()

    @property
    def patch_array_names(self):
        """Names of patch arrays in a list.
        
        Returns
        -------
        list[str]

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cavity(load=False)
        >>> reader = pyvista.OpenFOAMReader(filename)
        >>> reader.patch_array_names
        ['internalMesh', 'patch/movingWall', 'patch/fixedWalls', 'patch/frontAndBack']

        """
        return [self.reader.GetPatchArrayName(i) for i in range(self.number_patch_arrays)]

    def enable_patch_array(self, name):
        """Enable reading of patch array.

        Parameters
        ----------
        name : str
            Which patch array to enable.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cavity(load=False)
        >>> reader = pyvista.OpenFOAMReader(filename)
        >>> reader.enable_patch_array("patch/movingWall")
        >>> reader.patch_array_status("patch/movingWall")
        True

        """
        self.reader.SetPatchArrayStatus(name, 1)

    def disable_patch_array(self, name):
        """Disable reading of patch array.

        Parameters
        ----------
        name : str
            Which patch array to disable.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cavity(load=False)
        >>> reader = pyvista.OpenFOAMReader(filename)
        >>> reader.disable_patch_array("internalMesh")
        >>> reader.patch_array_status("internalMesh")
        False

        """
        self.reader.SetPatchArrayStatus(name, 0)

    def patch_array_status(self, name):
        """Return status of reading patch array.

        Parameters
        ----------
        name : str
            Which patch array to report status.
        Returns
        -------
        bool
                Whether the patch with the given name is to be read.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cavity(load=False)
        >>> reader = pyvista.OpenFOAMReader(filename)
        >>> reader.enable_patch_array("patch/movingWall")
        >>> reader.patch_array_status("patch/movingWall")
        True

        """
        return bool(self.reader.GetPatchArrayStatus(name))

    def enable_all_patch_arrays(self):
        """Enable reading of all patch arrays.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cavity(load=False)
        >>> reader = pyvista.OpenFOAMReader(filename)
        >>> reader.enable_all_patch_arrays()
        >>> assert reader.patch_array_status("patch/movingWall")
        >>> assert reader.patch_array_status("patch/fixedWalls")

        """
        self.reader.EnableAllPatchArrays()
    
    def disable_all_patch_arrays(self):
        """Disable reading of all patch arrays.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cavity(load=False)
        >>> reader = pyvista.OpenFOAMReader(filename)
        >>> reader.disable_all_patch_arrays()
        >>> assert not reader.patch_array_status("patch.movingWall")
        >>> assert not reader.patch_array_status("internalMesh")

        """
        self.reader.DisableAllPatchArrays()

    @property
    def all_patch_arrays_status(self):
        """Status of reading all patch arrays.

        Returns
        -------
        dict[str, bool]
            dict key is the patch name and the value is whether it will be read.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cavity(load=False)
        >>> reader = pyvista.OpenFOAMReader(filename)
        >>> reader.all_patch_arrays_status  #doctest: +NORMALIZE_WHITESPACE
        {'internalMesh': True, 'patch/movingWall': True, 'patch/fixedWalls': True,
         'patch/frontAndBack': True}

        """
        return {name: self.patch_array_status(name) for name in self.patch_array_names}


class PLYReader(BaseReader):
    """PLY Reader for reading .ply files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_lobster(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'lobster.ply'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot()

    """

    _class_reader = _vtk.vtkPLYReader


class OBJReader(BaseReader):
    """OBJ Reader for reading .obj files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_trumpet(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'trumpet.obj'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(cpos='yz', show_scalar_bar=False)

    """

    _class_reader = _vtk.vtkOBJReader


class STLReader(BaseReader):
    """STL Reader for .stl files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_cad_model(load=False)
    >>> filename.split("/")[-1]  # omit the path
    '42400-IDGH.stl'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot()

    """

    _class_reader = _vtk.vtkSTLReader


class VTKDataSetReader(BaseReader):
    """VTK Data Set Reader for .vtk files.

    Notes
    -----
    This reader calls `ReadAllScalarsOn`, `ReadAllColorScalarsOn`,
    `ReadAllNormalsOn`, `ReadAllTCoordsOn`, `ReadAllVectorsOn`,
    and `ReadAllFieldsOn` on the underlying `vtkDataSetReader`.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_brain(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'brain.vtk'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> sliced_mesh = mesh.slice('x')
    >>> sliced_mesh.plot(cpos='yz', show_scalar_bar=False)

    """

    _class_reader = _vtk.vtkDataSetReader

    def __init__(self, filename):
        """Initialize VTKDataSetReader with filename."""
        super().__init__(filename)
        # Provide consistency with defaults in pyvista.read
        self.reader.ReadAllScalarsOn()
        self.reader.ReadAllColorScalarsOn()
        self.reader.ReadAllNormalsOn()
        self.reader.ReadAllTCoordsOn()
        self.reader.ReadAllVectorsOn()
        self.reader.ReadAllFieldsOn()
        self.reader.ReadAllTensorsOn()


class VTKPDataSetReader(BaseReader):
    """Parallel VTK Data Set Reader for .pvtk files."""

    _class_reader = staticmethod(_vtk.lazy_vtkPDataSetReader)


class BYUReader(BaseReader):
    """BYU Reader for .g files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_teapot(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'teapot.g'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(cpos='xy', show_scalar_bar=False)

    """

    _class_reader = _vtk.vtkBYUReader


class FacetReader(BaseReader):
    """Facet Reader for .facet files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_clown(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'clown.facet'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(color="red")

    """

    _class_reader = staticmethod(_vtk.lazy_vtkFacetReader)


class Plot3DMetaReader(BaseReader):
    """Plot3DMeta Reader for .p3d files."""

    _class_reader = staticmethod(_vtk.lazy_vtkPlot3DMetaReader)


class BinaryMarchingCubesReader(BaseReader):
    """BinaryMarchingCubes Reader for .tri files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_pine_roots(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'pine_root.tri'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(color="brown")

    """

    _class_reader = _vtk.vtkMCubesReader


@dataclass(order=True)
class PVDDataSet:
    """Class for storing dataset info from PVD file."""

    time: float
    part: int
    filename: str
    group: str


# skip pydocstyle D102 check since docstring is taken from TimeReader
class PVDReader(BaseReader, TimeReader):
    """PVD Reader for .pvd files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_wavy(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'wavy.pvd'
    >>> reader = pyvista.get_reader(filename)
    >>> reader.time_values  # doctest: +ELLIPSIS
    [0.0, 1.0, 2.0, 3.0, ... 12.0, 13.0, 14.0]
    >>> reader.set_active_time_point(5)
    >>> reader.active_time_value
    5.0
    >>> mesh = reader.read()[0]  # MultiBlock mesh with only 1 block
    >>> mesh.plot(scalars='z')

    """

    def __init__(self, filename):
        """Initialize PVD file reader."""
        self._reader = None
        self.filename = filename
        self._directory = None
        self._datasets = []
        self._active_datasets = []
        self._active_readers = []
        self._time_values = []

        self._set_filename(filename)

    @property
    def reader(self):
        """Return the PVDReader.

        .. note::
            This Reader does not have an uderlying vtk Reader.

        """
        return self

    @property
    def active_readers(self):
        """Return the active readers.

        Returns
        -------
        list[pyvista.BaseReader]

        """
        return self._active_readers

    @property
    def datasets(self):
        """Return all datasets.

        Returns
        -------
        list[pyvista.PVDDataSet]

        """
        return self._datasets

    @property
    def active_datasets(self):
        """Return all active datasets.

        Returns
        -------
        list[pyvista.PVDDataSet]

        """
        return self._active_datasets

    def _set_filename(self, filename):
        """Set filename and update reader."""
        self.filename = filename
        self._directory = os.path.dirname(filename)
        self._datasets = None
        self._active_datasets = None
        self._update_information()

    def read(self):
        """Read data from PVD timepoint.

        Overrides :func:`pyvista.BaseReader.read`.

        Returns
        -------
        pyvista.MultiBlock

        """
        return pyvista.MultiBlock([reader.read() for reader in self.active_readers])

    def _update_information(self):
        """If dataset information is unavailable, parse file."""
        if self.datasets is None:
            self._parse_file()

    def _update(self):  # pragma: no cover
        """Unused in PVDReader."""
        pass

    def _parse_file(self):
        """Parse PVD file."""
        if self.filename is None:
            raise ValueError("Filename must be set")
        tree = ElementTree.parse(self.filename)
        root = tree.getroot()
        dataset_elements = root[0].findall("DataSet")
        datasets = []
        for element in dataset_elements:
            element_attrib = element.attrib
            datasets.append(
                PVDDataSet(
                    float(element_attrib.get('timestep', 0)),
                    int(element_attrib['part']),
                    element_attrib['file'],
                    element_attrib.get('group'),
                )
            )
        self._datasets = sorted(datasets)
        self._time_values = sorted(list(set([dataset.time for dataset in self.datasets])))

        self.set_active_time_value(self.time_values[0])

    @property
    def time_values(self):  # noqa: D102
        return self._time_values

    @property
    def number_time_points(self):  # noqa: D102
        return len(self.time_values)

    def time_point_value(self, time_point):  # noqa: D102
        return self.time_values[time_point]

    @property
    def active_time_value(self):  # noqa: D102
        # all active datasets have the same time
        return self.active_datasets[0].time

    def set_active_time_value(self, time_value):  # noqa: D102
        self._active_datasets = [
            dataset for dataset in self.datasets if dataset.time == time_value
        ]
        self._active_readers = [
            get_reader(os.path.join(self._directory, dataset.filename))
            for dataset in self.active_datasets
        ]

    def set_active_time_point(self, time_point):  # noqa: D102
        self.set_active_time_value(self.time_values[time_point])


CLASS_READERS = {
    # Standard dataset readers:
    '.vti': XMLImageDataReader,
    '.pvti': XMLPImageDataReader,
    '.vtr': XMLRectilinearGridReader,
    '.pvtr': XMLPRectilinearGridReader,
    '.vtu': XMLUnstructuredGridReader,
    '.pvtu': XMLPUnstructuredGridReader,
    '.vtp': XMLPolyDataReader,
    '.vts': XMLStructuredGridReader,
    '.vtm': XMLMultiBlockDataReader,
    '.vtmb': XMLMultiBlockDataReader,
    '.case': EnSightReader,
    '.foam': OpenFOAMReader,
    '.ply': PLYReader,
    '.obj': OBJReader,
    '.stl': STLReader,
    '.vtk': VTKDataSetReader,
    '.pvtk': VTKPDataSetReader,
    '.g': BYUReader,
    '.facet': FacetReader,
    '.p3d': Plot3DMetaReader,
    '.tri': BinaryMarchingCubesReader,
    '.pvd': PVDReader,
}
