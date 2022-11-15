"""Fine-grained control of reading data files."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import enum
from functools import wraps
import os
import pathlib
from typing import Any, Callable, List, Union
from xml.etree import ElementTree

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.utilities import abstract_class, wrap

from .fileio import _get_ext_force, _process_filename

HDF_HELP = 'https://kitware.github.io/vtk-examples/site/VTKFileFormats/#hdf-file-formats'


def get_reader(filename, force_ext=None):
    """Get a reader for fine-grained control of reading data files.

    Supported file types and Readers:

    +----------------+---------------------------------------------+
    | File Extension | Class                                       |
    +================+=============================================+
    | ``.bmp``       | :class:`pyvista.BMPReader`                  |
    +----------------+---------------------------------------------+
    | ``.cas``       | :class:`pyvista.FluentReader`               |
    +----------------+---------------------------------------------+
    | ``.case``      | :class:`pyvista.EnSightReader`              |
    +----------------+---------------------------------------------+
    | ``.cgns``      | :class:`pyvista.CGNSReader`                 |
    +----------------+---------------------------------------------+
    | ``.dat``       | :class:`pyvista.TecplotReader`              |
    +----------------+---------------------------------------------+
    | ``.dcm``       | :class:`pyvista.DICOMReader`                |
    +----------------+---------------------------------------------+
    | ``.dem``       | :class:`pyvista.DEMReader`                  |
    +----------------+---------------------------------------------+
    | ``.facet``     | :class:`pyvista.FacetReader`                |
    +----------------+---------------------------------------------+
    | ``.foam``      | :class:`pyvista.POpenFOAMReader`            |
    +----------------+---------------------------------------------+
    | ``.g``         | :class:`pyvista.BYUReader`                  |
    +----------------+---------------------------------------------+
    | ``.gif``       | :class:`pyvista.GIFReader`                  |
    +----------------+---------------------------------------------+
    | ``.glb``       | :class:`pyvista.GLTFReader`                 |
    +----------------+---------------------------------------------+
    | ``.gltf``      | :class:`pyvista.GLTFReader`                 |
    +----------------+---------------------------------------------+
    | ``.hdf``       | :class:`pyvista.HDFReader`                  |
    +----------------+---------------------------------------------+
    | ``.img``       | :class:`pyvista.DICOMReader`                |
    +----------------+---------------------------------------------+
    | ``.inp``       | :class:`pyvista.AVSucdReader`               |
    +----------------+---------------------------------------------+
    | ``.jpg``       | :class:`pyvista.JPEGReader`                 |
    +----------------+---------------------------------------------+
    | ``.jpeg``      | :class:`pyvista.JPEGReader`                 |
    +----------------+---------------------------------------------+
    | ``.hdr``       | :class:`pyvista.HDRReader`                  |
    +----------------+---------------------------------------------+
    | ``.mha``       | :class:`pyvista.MetaImageReader`            |
    +----------------+---------------------------------------------+
    | ``.mhd``       | :class:`pyvista.MetaImageReader`            |
    +----------------+---------------------------------------------+
    | ``.nii``       | :class:`pyvista.NIFTIReader`                |
    +----------------+---------------------------------------------+
    | ``.nii.gz``    | :class:`pyvista.NIFTIReader`                |
    +----------------+---------------------------------------------+
    | ``.nhdr``      | :class:`pyvista.NRRDReader`                 |
    +----------------+---------------------------------------------+
    | ``.nrrd``      | :class:`pyvista.NRRDReader`                 |
    +----------------+---------------------------------------------+
    | ``.obj``       | :class:`pyvista.OBJReader`                  |
    +----------------+---------------------------------------------+
    | ``.p3d``       | :class:`pyvista.Plot3DMetaReader`           |
    +----------------+---------------------------------------------+
    | ``.ply``       | :class:`pyvista.PLYReader`                  |
    +----------------+---------------------------------------------+
    | ``.png``       | :class:`pyvista.PNGReader`                  |
    +----------------+---------------------------------------------+
    | ``.pnm``       | :class:`pyvista.PNMReader`                  |
    +----------------+---------------------------------------------+
    | ``.pts``       | :class:`pyvista.PTSReader`                  |
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
    | ``.res``       | :class:`pyvista.MFIXReader`                 |
    +----------------+---------------------------------------------+
    | ``.segy``      | :class:`pyvista.SegYReader`                 |
    +----------------+---------------------------------------------+
    | ``.sgy``       | :class:`pyvista.SegYReader`                 |
    +----------------+---------------------------------------------+
    | ``.slc``       | :class:`pyvista.SLCReader`                  |
    +----------------+---------------------------------------------+
    | ``.stl``       | :class:`pyvista.STLReader`                  |
    +----------------+---------------------------------------------+
    | ``.tif``       | :class:`pyvista.TIFFReader`                 |
    +----------------+---------------------------------------------+
    | ``.tiff``      | :class:`pyvista.TIFFReader`                 |
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

    force_ext : str, optional
        An extension to force a specific reader to be chosen.

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
    ext = _get_ext_force(filename, force_ext)

    try:
        Reader = CLASS_READERS[ext]
    except KeyError:
        raise ValueError(f"`pyvista.get_reader` does not support a file with the {ext} extension")

    return Reader(filename)


@abstract_class
class BaseReader:
    """The Base Reader class.

    The base functionality includes reading data from a file,
    and allowing access to the underlying vtk reader. See
    :func:`pyvista.get_reader` for an example using
    a built-in subclass.

    Parameters
    ----------
    path : str
        Path of the file to read.
    """

    _class_reader: Any = None

    def __init__(self, path):
        """Initialize Reader by setting path."""
        self._reader = self._class_reader()
        self._filename = None
        self._progress_bar = False
        self._progress_msg = None
        self.__directory = None
        self._set_defaults()
        self.path = path
        self._set_defaults_post()

    def __repr__(self):
        """Representation of a Reader object."""
        return f"{self.__class__.__name__}('{self.path}')"

    def show_progress(self, msg=None):
        """Show a progress bar when loading the file.

        Parameters
        ----------
        msg : str, optional
            Progress bar message. Defaults to ``"Reading <file base name>"``.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cavity(load=False)
        >>> reader = pyvista.OpenFOAMReader(filename)
        >>> reader.show_progress()

        """
        self._progress_bar = True
        if msg is None:
            msg = f"Reading {os.path.basename(self.path)}"
        self._progress_msg = msg

    def hide_progress(self):
        """Hide the progress bar when loading the file.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cavity(load=False)
        >>> reader = pyvista.OpenFOAMReader(filename)
        >>> reader.hide_progress()

        """
        self._progress_bar = False

    @property
    def reader(self):
        """Return the vtk Reader object.

        Returns
        -------
        pyvista.BaseReader
            An instance of the Reader object.

        """
        if self._reader is None:  # pragma: no cover
            raise NotImplementedError
        return self._reader

    @property
    def path(self) -> str:
        """Return or set the filename or directory of the reader.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_human(load=False)
        >>> reader = pyvista.XMLPolyDataReader(filename)
        >>> reader.path  # doctest:+SKIP
        '/home/user/.local/share/pyvista/examples/Human.vtp'

        """
        if self._filename is not None:
            return self._filename
        return self.__directory

    @path.setter
    def path(self, path: str):
        if os.path.isdir(path):
            self._set_directory(path)
        elif os.path.isfile(path):
            self._set_filename(path)
        else:
            raise FileNotFoundError(f"Path '{path}' is invalid or does not exist.")

    def _set_directory(self, directory):
        """Set directory and update reader."""
        self._filename = None
        self.__directory = directory
        self.reader.SetDirectoryName(directory)
        self._update_information()

    def _set_filename(self, filename):
        """Set filename and update reader."""
        # Private method since changing file type requires a
        # different subclass.
        self.__directory = None
        self._filename = filename
        self.reader.SetFileName(filename)
        self._update_information()

    def read(self):
        """Read data in file.

        Returns
        -------
        pyvista.DataSet
            PyVista Dataset.
        """
        from pyvista.core.filters import _update_alg  # avoid circular import

        _update_alg(self.reader, progress_bar=self._progress_bar, message=self._progress_msg)
        data = wrap(self.reader.GetOutputDataObject(0))
        if data is None:  # pragma: no cover
            raise RuntimeError("Failed to read file.")
        data._post_file_load_processing()

        # check for any pyvista metadata
        data._restore_metadata()
        return data

    def _update_information(self):
        self.reader.UpdateInformation()

    def _set_defaults(self):
        """Set defaults on reader, if needed."""
        pass

    def _set_defaults_post(self):
        """Set defaults on reader post setting file, if needed."""
        pass


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
        list[str]

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
        return bool(self.reader.GetCellArrayStatus(name))

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

    _class_reader = _vtk.vtkXMLImageDataReader


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
        self._filename = filename
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
                f"Not a valid time {time_value} from available time values: {self.time_values}"
            )
        self.reader.SetTimeValue(time_value)

    def set_active_time_point(self, time_point):  # noqa: D102
        self.reader.SetTimeValue(self.time_point_value(time_point))


# skip pydocstyle D102 check since docstring is taken from TimeReader
class OpenFOAMReader(BaseReader, PointCellDataSelection, TimeReader):
    """OpenFOAM Reader for .foam files.

    By default, pyvista enables all patch arrays.  This is a deviation
    from the vtk default.

    """

    _class_reader = _vtk.vtkOpenFOAMReader

    def _set_defaults_post(self):
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
            raise AttributeError(
                "Inspecting active time value only supported for vtk versions >9.1.0"
            ) from err
        return value

    def set_active_time_value(self, time_value):  # noqa: D102
        if time_value not in self.time_values:
            raise ValueError(
                f"Not a valid time {time_value} from available time values: {self.time_values}"
            )
        self.reader.UpdateTimeStep(time_value)

    def set_active_time_point(self, time_point):  # noqa: D102
        self.reader.UpdateTimeStep(self.time_point_value(time_point))

    @property
    def decompose_polyhedra(self):
        """Whether polyhedra are to be decomposed when read.

        Returns
        -------
        bool
            If ``True``, decompose polyhedra into tetrahedra and pyramids.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cavity(load=False)
        >>> reader = pyvista.OpenFOAMReader(filename)
        >>> reader.decompose_polyhedra = False
        >>> reader.decompose_polyhedra
        False

        """
        return bool(self.reader.GetDecomposePolyhedra())

    @decompose_polyhedra.setter
    def decompose_polyhedra(self, value):
        self.reader.SetDecomposePolyhedra(value)

    @property
    def skip_zero_time(self):
        """Indicate whether or not to ignore the '/0' time directory.

        Returns
        -------
        bool
            If ``True``, ignore the '/0' time directory.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cavity(load=False)
        >>> reader = pyvista.OpenFOAMReader(filename)
        >>> reader.skip_zero_time = False
        >>> reader.skip_zero_time
        False

        """
        return bool(self.reader.GetSkipZeroTime())

    @skip_zero_time.setter
    def skip_zero_time(self, value):
        self.reader.SetSkipZeroTime(value)
        self._update_information()
        self.reader.SetRefresh()

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
        self.reader.SetCreateCellToPoint(value)

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


class POpenFOAMReader(OpenFOAMReader):
    """Parallel OpenFOAM Reader for .foam files.

    Can read parallel-decomposed mesh information and time dependent data.
    This reader can be used for serial generated data,
    parallel reconstructed data, and decomposed data.
    """

    _class_reader = staticmethod(_vtk.lazy_vtkPOpenFOAMReader)

    @property
    def case_type(self):
        """Indicate whether decomposed mesh or reconstructed mesh should be read.

        Returns
        -------
        str
            If ``'reconstructed'``, reconstructed mesh should be read.
            If ``'decomposed'``, decomposed mesh should be read.

        Raises
        ------
        ValueError
            If the value is not in ['reconstructed', 'decomposed']

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cavity(load=False)
        >>> reader = pyvista.POpenFOAMReader(filename)
        >>> reader.case_type = 'reconstructed'
        >>> reader.case_type
        'reconstructed'
        """
        return 'reconstructed' if self.reader.GetCaseType() else 'decomposed'

    @case_type.setter
    def case_type(self, value):
        if value == 'reconstructed':
            self.reader.SetCaseType(1)
        elif value == 'decomposed':
            self.reader.SetCaseType(0)
        else:
            raise ValueError(f"Unknown case type '{value}'.")

        self._update_information()


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


class TecplotReader(BaseReader):
    """Tecplot Reader for ascii .dat files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_tecplot_ascii(load=False)
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh[0].plot()

    """

    _class_reader = _vtk.vtkTecplotReader


class VTKDataSetReader(BaseReader):
    """VTK Data Set Reader for .vtk files.

    Notes
    -----
    This reader calls ``ReadAllScalarsOn``, ``ReadAllColorScalarsOn``,
    ``ReadAllNormalsOn``, ``ReadAllTCoordsOn``, ``ReadAllVectorsOn``,
    and ``ReadAllFieldsOn`` on the underlying ``vtkDataSetReader``.

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

    def _set_defaults_post(self):
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


class Plot3DFunctionEnum(enum.IntEnum):
    """An enumeration for the functions used in :class:`MultiBlockPlot3DReader`."""

    DENSITY = 100
    PRESSURE = 110
    PRESSURE_COEFFICIENT = 111
    MACH = 112
    SPEED_OF_SOUND = 113
    TEMPERATURE = 120
    ENTHALPY = 130
    INTERNAL_ENERGY = 140
    KINETIC_ENERGY = 144
    VELOCITY_MAGNITUDE = 153
    STAGNATION_ENERGY = 163
    ENTROPY = 170
    SWIRL = 184
    VELOCITY = 200
    VORTICITY = 201
    MOMENTUM = 202
    PRESSURE_GRADIENT = 210
    STRAIN_RATE = 212
    VORTICITY_MAGNITUDE = 211


class MultiBlockPlot3DReader(BaseReader):
    """MultiBlock Plot3D Reader.

    The methods :meth:`add_function()` and :meth:`remove_function()` accept values from
    :class:`Plot3DFunctionEnum`. For convenience, the values of that enumeration are available as class variables,
    as shown below.

        - ``MultiBlockPlot3DReader.DENSITY = Plot3DFunctionEnum.DENSITY``
        - ``MultiBlockPlot3DReader.PRESSURE = Plot3DFunctionEnum.PRESSURE``
        - ``MultiBlockPlot3DReader.PRESSURE_COEFFICIENT = Plot3DFunctionEnum.PRESSURE_COEFFICIENT``
        - ``MultiBlockPlot3DReader.MACH = Plot3DFunctionEnum.MACH``
        - ``MultiBlockPlot3DReader.SPEED_OF_SOUND = Plot3DFunctionEnum.SPEED_OF_SOUND``
        - ``MultiBlockPlot3DReader.TEMPERATURE = Plot3DFunctionEnum.TEMPERATURE``
        - ``MultiBlockPlot3DReader.ENTHALPY = Plot3DFunctionEnum.ENTHALPY``
        - ``MultiBlockPlot3DReader.INTERNAL_ENERGY = Plot3DFunctionEnum.INTERNAL_ENERGY``
        - ``MultiBlockPlot3DReader.KINETIC_ENERGY = Plot3DFunctionEnum.KINETIC_ENERGY``
        - ``MultiBlockPlot3DReader.VELOCITY_MAGNITUDE = Plot3DFunctionEnum.VELOCITY_MAGNITUDE``
        - ``MultiBlockPlot3DReader.STAGNATION_ENERGY = Plot3DFunctionEnum.STAGNATION_ENERGY``
        - ``MultiBlockPlot3DReader.ENTROPY = Plot3DFunctionEnum.ENTROPY``
        - ``MultiBlockPlot3DReader.SWIRL = Plot3DFunctionEnum.SWIRL``
        - ``MultiBlockPlot3DReader.VELOCITY = Plot3DFunctionEnum.VELOCITY``
        - ``MultiBlockPlot3DReader.VORTICITY = Plot3DFunctionEnum.VORTICITY``
        - ``MultiBlockPlot3DReader.MOMENTUM = Plot3DFunctionEnum.MOMENTUM``
        - ``MultiBlockPlot3DReader.PRESSURE_GRADIENT = Plot3DFunctionEnum.PRESSURE_GRADIENT``
        - ``MultiBlockPlot3DReader.STRAIN_RATE = Plot3DFunctionEnum.STRAIN_RATE``
        - ``MultiBlockPlot3DReader.VORTICITY_MAGNITUDE = Plot3DFunctionEnum.VORTICITY_MAGNITUDE``
    """

    _class_reader = staticmethod(_vtk.lazy_vtkMultiBlockPLOT3DReader)

    # pull in function name enum values as class constants
    DENSITY = Plot3DFunctionEnum.DENSITY
    PRESSURE = Plot3DFunctionEnum.PRESSURE
    PRESSURE_COEFFICIENT = Plot3DFunctionEnum.PRESSURE_COEFFICIENT
    MACH = Plot3DFunctionEnum.MACH
    SPEED_OF_SOUND = Plot3DFunctionEnum.SPEED_OF_SOUND
    TEMPERATURE = Plot3DFunctionEnum.TEMPERATURE
    ENTHALPY = Plot3DFunctionEnum.ENTHALPY
    INTERNAL_ENERGY = Plot3DFunctionEnum.INTERNAL_ENERGY
    KINETIC_ENERGY = Plot3DFunctionEnum.KINETIC_ENERGY
    VELOCITY_MAGNITUDE = Plot3DFunctionEnum.VELOCITY_MAGNITUDE
    STAGNATION_ENERGY = Plot3DFunctionEnum.STAGNATION_ENERGY
    ENTROPY = Plot3DFunctionEnum.ENTROPY
    SWIRL = Plot3DFunctionEnum.SWIRL
    VELOCITY = Plot3DFunctionEnum.VELOCITY
    VORTICITY = Plot3DFunctionEnum.VORTICITY
    MOMENTUM = Plot3DFunctionEnum.MOMENTUM
    PRESSURE_GRADIENT = Plot3DFunctionEnum.PRESSURE_GRADIENT
    STRAIN_RATE = Plot3DFunctionEnum.STRAIN_RATE
    VORTICITY_MAGNITUDE = Plot3DFunctionEnum.VORTICITY_MAGNITUDE

    def _set_defaults(self):
        self.auto_detect_format = True

    def add_q_files(self, files):
        """Add q file(s).

        Parameters
        ----------
        files : str or Iterable(str)
            Solution file or files to add.

        """
        # files may be a list or a single filename
        if files:
            if isinstance(files, (str, pathlib.Path)):
                files = [files]
        files = [_process_filename(f) for f in files]

        if hasattr(self.reader, 'AddFileName'):
            # AddFileName was added to vtkMultiBlockPLOT3DReader sometime around
            # VTK 8.2. This method supports reading multiple q files.
            for q_filename in files:
                self.reader.AddFileName(q_filename)
        else:
            # SetQFileName is used to add a single q file to be read, and is still
            # supported in VTK9.
            if len(files) > 0:
                if len(files) > 1:
                    raise RuntimeError(
                        'Reading of multiple q files is not supported with this version of VTK.'
                    )
                self.reader.SetQFileName(files[0])

    @property
    def auto_detect_format(self):
        """Whether to try to automatically detect format such as byte order, etc."""
        return bool(self.reader.GetAutoDetectFormat())

    @auto_detect_format.setter
    def auto_detect_format(self, value):
        self.reader.SetAutoDetectFormat(value)

    def add_function(self, value: Union[int, Plot3DFunctionEnum]):
        """Specify additional functions to compute.

        The available functions are enumerated in :class:`Plot3DFunctionEnum`. The members of this enumeration are most
        easily accessed by their aliases as class variables.

        Multiple functions may be requested by calling this method multiple times.

        Parameters
        ----------
        value : int or Plot3DFunctionEnum
            The function to add.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename  = examples.download_file('multi-bin.xyz')
        >>> reader = pyvista.reader.MultiBlockPlot3DReader(filename)
        >>> reader.add_function(112)  # add a function by its integer value
        >>> reader.add_function(reader.PRESSURE_COEFFICIENT)  # add a function by enumeration via class variable alias

        """
        if isinstance(value, enum.Enum):
            value = value.value
        self.reader.AddFunction(value)

    def remove_function(self, value: Union[int, Plot3DFunctionEnum]):
        """Remove one function from list of functions to compute.

        For details on the types of accepted values, see :meth:``add_function``.

        Parameters
        ----------
        value : int or Plot3DFunctionEnum
            The function to remove.
        """
        if isinstance(value, enum.Enum):
            value = value.value
        self.reader.RemoveFunction(value)

    def remove_all_functions(self):
        """Remove all functions from list of functions to compute."""
        self.reader.RemoveAllFunctions()

    @property
    def preserve_intermediate_functions(self):
        """When ``True`` (default), intermediate computed quantities will be preserved.

        For example, if ``VelocityMagnitude`` is enabled, but not ``Velocity``, the reader still needs to compute
        ``Velocity``. If `preserve_intermediate_functions` is ``False``, then the output will not have ``Velocity``
        array, only the requested ``VelocityMagnitude``.

        This is useful to avoid using up memory for arrays that are not relevant for the analysis.
        """
        return self.reader.GetPreserveIntermediateFunctions()

    @preserve_intermediate_functions.setter
    def preserve_intermediate_functions(self, val):
        self.reader.SetPreserveIntermediateFunctions(val)

    @property
    def gamma(self):
        """Ratio of specific heats."""
        return self.reader.GetGamma()

    @gamma.setter
    def gamma(self, val):
        self.reader.SetGamma(val)

    @property
    def r_gas_constant(self):
        """Gas constant."""
        return self.reader.GetR()

    @r_gas_constant.setter
    def r_gas_constant(self, val):
        self.reader.SetR(val)


class CGNSReader(BaseReader, PointCellDataSelection):
    """CGNS Reader for .cgns files.

    Creates a multi-block dataset and reads unstructured grids and structured
    meshes from binary files stored in CGNS file format, with data stored at
    the nodes, cells or faces.

    By default, all point and cell arrays are loaded as well as the boundary
    patch. This varies from VTK's defaults. For more details, see
    `vtkCGNSReader <https://vtk.org/doc/nightly/html/classvtkCGNSReader.html>`_

    Examples
    --------
    Load a CGNS file.  All arrays are loaded by default.

    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_cgns_multi(load=False)
    >>> reader = pyvista.CGNSReader(filename)
    >>> reader.load_boundary_patch = False
    >>> ds = reader.read()
    >>> ds[0][0].cell_data
    pyvista DataSetAttributes
    Association     : CELL
    Active Scalars  : None
    Active Vectors  : Momentum
    Active Texture  : None
    Active Normals  : None
    Contains arrays :
        Density                 float64    (2928,)
        Momentum                float64    (2928, 3)            VECTORS
        EnergyStagnationDensity float64    (2928,)
        ViscosityEddy           float64    (2928,)
        TurbulentDistance       float64    (2928,)
        TurbulentSANuTilde      float64    (2928,)

    """

    _class_reader = staticmethod(_vtk.lazy_vtkCGNSReader)

    def _set_defaults_post(self):
        self.enable_all_point_arrays()
        self.enable_all_cell_arrays()
        self.load_boundary_patch = True

    @property
    def distribute_blocks(self) -> bool:
        """Distribute each block in each zone across ranks.

        To make the reader disregard the piece request and read all blocks in the
        zone, set this to ``False``. The default is ``True``.

        Returns
        -------
        bool
            If ``True``, distribute each block in each zone across ranks.

        Examples
        --------
        Disable distributing blocks.

        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cgns_multi(load=False)
        >>> reader = pyvista.CGNSReader(filename)
        >>> reader.distribute_blocks = False
        >>> reader.distribute_blocks
        False

        """
        return bool(self._reader.GetDistributeBlocks())

    @distribute_blocks.setter
    def distribute_blocks(self, value: str):
        self._reader.SetDistributeBlocks(value)

    def base_array_status(self, name: str) -> bool:
        """Get status of base array with name.

        Parameters
        ----------
        name : str
            Base array name.

        Returns
        -------
        bool
            Whether reading the base array is enabled.

        """
        return bool(self.reader.GetBaseArrayStatus(name))

    @property
    def base_array_names(self):
        """Return the list of all base array names.

        Returns
        -------
        list[int]

        """
        return [self.reader.GetBaseArrayName(i) for i in range(self.number_base_arrays)]

    @property
    def number_base_arrays(self) -> int:
        """Return the number of base arrays.

        Returns
        -------
        int

        """
        return self.reader.GetNumberOfBaseArrays()

    def enable_all_bases(self):
        """Enable reading all bases.

        By default only the 0th base is read.

        Examples
        --------
        Read all bases.

        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cgns_multi(load=False)
        >>> reader = pyvista.CGNSReader(filename)
        >>> reader.enable_all_bases()
        """
        self._reader.EnableAllBases()

    def disable_all_bases(self):
        """Disable reading all bases.

        By default only the 0th base is read.

        Examples
        --------
        Disable reading all bases.

        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cgns_multi(load=False)
        >>> reader = pyvista.CGNSReader(filename)
        >>> reader.disable_all_bases()
        """
        self._reader.DisableAllBases()

    def family_array_status(self, name) -> bool:
        """Get status of family array with name.

        Parameters
        ----------
        name : str
            Family array name.

        Returns
        -------
        bool
            Whether reading the family array is enabled.

        """
        return bool(self.reader.GetFamilyArrayStatus(name))

    @property
    def family_array_names(self) -> List[str]:
        """Return the list of all family array names.

        Returns
        -------
        list[str]

        """
        return [self.reader.GetFamilyArrayName(i) for i in range(self.number_family_arrays)]

    @property
    def number_family_arrays(self) -> int:
        """Return the number of face arrays.

        Returns
        -------
        int

        """
        return self.reader.GetNumberOfFamilyArrays()

    def enable_all_families(self):
        """Enable reading all families.

        By default only the 0th family is read.

        Examples
        --------
        Read all bases.

        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cgns_multi(load=False)
        >>> reader = pyvista.CGNSReader(filename)
        >>> reader.enable_all_families()
        """
        self._reader.EnableAllFamilies()

    def disable_all_families(self):
        """Disable reading all families.

        Examples
        --------
        Disable reading all bases.

        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cgns_multi(load=False)
        >>> reader = pyvista.CGNSReader(filename)
        >>> reader.disable_all_families()
        """
        self._reader.DisableAllFamilies()

    @property
    def unsteady_pattern(self) -> bool:
        """Return or set using an unsteady pattern.

        When set to ``True`` (default is ``False``), the reader will try to
        determine FlowSolution_t nodes to read with a pattern
        matching This can be useful for unsteady solutions when
        FlowSolutionPointers are not reliable.

        Examples
        --------
        Set reading the unsteady pattern to ``True``.

        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cgns_multi(load=False)
        >>> reader = pyvista.CGNSReader(filename)
        >>> reader.unsteady_pattern = True
        >>> reader.unsteady_pattern
        True

        """
        return self._reader.GetUseUnsteadyPattern()

    @unsteady_pattern.setter
    def unsteady_pattern(self, enabled: bool):
        self._reader.SetUseUnsteadyPattern(bool(enabled))

    @property
    def vector_3d(self) -> bool:
        """Return or set adding an empty dimension to vectors in case of 2D solutions.

        Examples
        --------
        Set adding an empty physical dimension to vectors to ``True``.

        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cgns_multi(load=False)
        >>> reader = pyvista.CGNSReader(filename)
        >>> reader.vector_3d = True
        >>> reader.vector_3d
        True

        """
        return self._reader.GetUse3DVector()

    @vector_3d.setter
    def vector_3d(self, enabled: bool):
        self._reader.SetUse3DVector(bool(enabled))

    @property
    def load_boundary_patch(self) -> bool:
        """Return or set loading boundary patches.

        Notes
        -----
        VTK default is ``False``, but PyVista uses ``True``.

        Examples
        --------
        Enable loading boundary patches .

        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = examples.download_cgns_multi(load=False)
        >>> reader = pyvista.CGNSReader(filename)
        >>> reader.load_boundary_patch = True
        >>> reader.load_boundary_patch
        True

        """
        return self._reader.GetLoadBndPatch()

    @load_boundary_patch.setter
    def load_boundary_patch(self, enabled: bool):
        self._reader.SetLoadBndPatch(bool(enabled))


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
    path: str
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
        self.__directory = None
        self._datasets = []
        self._active_datasets = []
        self._active_readers = []
        self._time_values = []

        self._set_filename(filename)

    @property
    def reader(self):
        """Return the PVDReader.

        .. note::
            This Reader does not have an underlying vtk Reader.

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
        self._filename = filename
        self.__directory = os.path.join(os.path.dirname(filename))
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

    def _parse_file(self):
        """Parse PVD file."""
        if self.path is None:
            raise ValueError("Filename must be set")
        tree = ElementTree.parse(self.path)
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
        self._active_datasets = [dataset for dataset in self.datasets if dataset.time == time_value]
        self._active_readers = [
            get_reader(os.path.join(self.__directory, dataset.path))
            for dataset in self.active_datasets
        ]

    def set_active_time_point(self, time_point):  # noqa: D102
        self.set_active_time_value(self.time_values[time_point])


class DICOMReader(BaseReader):
    """DICOM Reader for reading ``.dcm`` files.

    This reader reads a single file or a path containing a several ``.dcm``
    files (DICOM stack).

    Parameters
    ----------
    path : str
        Path to the single DICOM (``.dcm``) file to be opened or the directory
        containing a stack of DICOM files.

    Examples
    --------
    Read a DICOM stack.

    >>> import pyvista
    >>> from pyvista import examples
    >>> path = examples.download_dicom_stack(load=False)
    >>> reader = pyvista.DICOMReader(path)
    >>> dataset = reader.read()
    >>> dataset.plot(volume=True, zoom=3, show_scalar_bar=False)

    """

    _class_reader = _vtk.vtkDICOMImageReader


class BMPReader(BaseReader):
    """BMP Reader for .bmp files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_masonry_texture(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'masonry.bmp'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot()

    """

    _class_reader = _vtk.vtkBMPReader


class DEMReader(BaseReader):
    """DEM Reader for .dem files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_st_helens(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'SainteHelens.dem'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot()

    """

    _class_reader = _vtk.vtkDEMReader


class JPEGReader(BaseReader):
    """JPEG Reader for .jpeg and .jpg files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.planets.download_mars_surface(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'mars.jpg'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot()

    """

    _class_reader = _vtk.vtkJPEGReader


class MetaImageReader(BaseReader):
    """Meta Image Reader for .mha and .mhd files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_chest(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'ChestCT-SHORT.mha'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot()

    """

    _class_reader = _vtk.vtkMetaImageReader


class NIFTIReader(BaseReader):
    """NIFTI Reader for .nii and .nii.gz files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_brain_atlas_with_sides(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'avg152T1_RL_nifti.nii.gz'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot()

    """

    _class_reader = _vtk.vtkNIFTIImageReader


class NRRDReader(BaseReader):
    """NRRDReader for .nrrd and .nhdr files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_beach(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'beach.nrrd'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot()

    """

    _class_reader = _vtk.vtkNrrdReader


class PNGReader(BaseReader):
    """PNGReader for .png files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_vtk_logo(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'vtk.png'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot()

    """

    _class_reader = _vtk.vtkPNGReader


class PNMReader(BaseReader):
    """PNMReader for .pnm files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_gourds_pnm(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'Gourds.pnm'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot()

    """

    _class_reader = _vtk.vtkPNMReader


class SLCReader(BaseReader):
    """SLCReader for .slc files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_knee_full(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'vw_knee.slc'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot()

    """

    _class_reader = _vtk.vtkSLCReader


class TIFFReader(BaseReader):
    """TIFFReader for .tif and .tiff files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_crater_imagery(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'BJ34_GeoTifv1-04_crater_clip.tif'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot()

    """

    _class_reader = _vtk.vtkTIFFReader


class HDRReader(BaseReader):
    """HDRReader for .hdr files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_parched_canal_4k(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'parched_canal_4k.hdr'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot()

    """

    _class_reader = _vtk.vtkHDRReader


class PTSReader(BaseReader):
    """PTSReader for .pts files."""

    _class_reader = _vtk.vtkPTSReader


class AVSucdReader(BaseReader):
    """AVSucdReader for .inp files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_cells_nd(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'cellsnd.ascii.inp'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(cpos="xy")

    """

    _class_reader = _vtk.vtkAVSucdReader


class HDFReader(BaseReader):
    """HDFReader for .hdf files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_can_crushed_hdf(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'can-vtu.hdf'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot()

    """

    _class_reader = staticmethod(_vtk.lazy_vtkHDFReader)

    @wraps(BaseReader.read)
    def read(self):
        """Wrap the base reader to handle the vtk 9.1 --> 9.2 change."""
        try:
            with pyvista.VtkErrorCatcher(raise_errors=True):
                return super().read()
        except RuntimeError as err:  # pragma: no cover
            if "Can't find the `Type` attribute." in str(err):
                raise RuntimeError(
                    f'{self.path} is missing the Type attribute. '
                    'The VTKHDF format has changed as of 9.2.0, '
                    f'see {HDF_HELP} for more details.'
                )
            else:
                raise err


class GLTFReader(BaseReader):
    """GLTFeader for .gltf and .glb files."""

    _class_reader = _vtk.vtkGLTFReader


class FluentReader(BaseReader):
    """FluentReader for .cas files."""

    _class_reader = _vtk.vtkFLUENTReader


class MFIXReader(BaseReader):
    """MFIXReader for .res files."""

    _class_reader = _vtk.vtkMFIXReader


class SegYReader(BaseReader):
    """SegYReader for .sgy and .segy files."""

    _class_reader = staticmethod(_vtk.lazy_vtkSegYReader)


class _GIFReader:
    """Simulate a VTK reader for GIF files."""

    _data_object = None
    _observers: List[Union[int, Callable]] = []
    _n_frames = 0
    _current_frame = 0

    def SetFileName(self, filename):
        """Needed for VTK-like compatibility with BaseReader."""
        self._filename = filename

    def UpdateInformation(self):
        """Needed for VTK-like compatibility with BaseReader."""
        pass

    def AddObserver(self, event_type, callback):
        """Needed for VTK-like compatibility with BaseReader."""
        self._observers.append([event_type, callback])

    def RemoveObservers(self, *args):
        """Needed for VTK-like compatibility with BaseReader."""
        self._observers = []

    def GetProgress(self):
        return self._current_frame / self._n_frames

    def UpdateObservers(self, event_type):
        for event_type_allowed, observer in self._observers:
            if event_type_allowed == event_type:
                observer(self, event_type)

    def Update(self):
        """Read the GIF and store internally to `_data_object`."""
        from PIL import Image, ImageSequence

        img = Image.open(self._filename)
        self._data_object = pyvista.UniformGrid(dimensions=(img.size[0], img.size[1], 1))

        # load each frame to the grid (RGB since gifs do not support transparency
        self._n_frames = img.n_frames
        for i, frame in enumerate(ImageSequence.Iterator(img)):
            self._current_frame = i
            data = np.array(frame.convert('RGB').getdata(), dtype=np.uint8)
            self._data_object.point_data.set_array(data, f'frame{i}')
            self.UpdateObservers(6)

        if 'frame0' in self._data_object.point_data:
            self._data_object.point_data.active_scalars_name = 'frame0'

    def GetOutputDataObject(self, *args):
        """Needed for VTK-like compatibility with BaseReader."""
        return self._data_object


class GIFReader(BaseReader):
    """GIFReader for .gif files.

    Parameters
    ----------
    path : str
        Path of the GIF to read.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_gif_simple(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'sample.gif'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(rgba=True, zoom='tight', border=True, border_width=2)

    """

    _class_reader = _GIFReader


CLASS_READERS = {
    # Standard dataset readers:
    '.bmp': BMPReader,
    '.cas': FluentReader,
    '.case': EnSightReader,
    '.cgns': CGNSReader,
    '.dat': TecplotReader,
    '.dcm': DICOMReader,
    '.dem': DEMReader,
    '.facet': FacetReader,
    '.foam': POpenFOAMReader,
    '.g': BYUReader,
    '.gif': GIFReader,
    '.glb': GLTFReader,
    '.gltf': GLTFReader,
    '.img': DICOMReader,
    '.inp': AVSucdReader,
    '.jpg': JPEGReader,
    '.jpeg': JPEGReader,
    '.hdf': HDFReader,
    '.hdr': HDRReader,
    '.mha': MetaImageReader,
    '.mhd': MetaImageReader,
    '.nii': NIFTIReader,
    '.nii.gz': NIFTIReader,
    '.nhdr': NRRDReader,
    '.nrrd': NRRDReader,
    '.obj': OBJReader,
    '.p3d': Plot3DMetaReader,
    '.ply': PLYReader,
    '.png': PNGReader,
    '.pnm': PNMReader,
    '.pts': PTSReader,
    '.pvd': PVDReader,
    '.pvti': XMLPImageDataReader,
    '.pvtk': VTKPDataSetReader,
    '.pvtr': XMLPRectilinearGridReader,
    '.pvtu': XMLPUnstructuredGridReader,
    '.res': MFIXReader,
    '.segy': SegYReader,
    '.sgy': SegYReader,
    '.slc': SLCReader,
    '.stl': STLReader,
    '.tif': TIFFReader,
    '.tiff': TIFFReader,
    '.tri': BinaryMarchingCubesReader,
    '.vti': XMLImageDataReader,
    '.vtk': VTKDataSetReader,
    '.vtm': XMLMultiBlockDataReader,
    '.vtmb': XMLMultiBlockDataReader,
    '.vtp': XMLPolyDataReader,
    '.vtr': XMLRectilinearGridReader,
    '.vts': XMLStructuredGridReader,
    '.vtu': XMLUnstructuredGridReader,
}
