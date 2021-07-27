"""Fine-grained control of reading data files."""

from abc import ABC, abstractmethod
from pyvista.utilities import wrap, get_ext
from pyvista import _vtk


def get_reader(filename):
    """Get a reader for fine-grained control of reading data files.

    Supported file types and Readers:
    * .vti: :class:`pyvista.XMLImageDataReader`
    * .pvti: :class:`XMLPImageDataReader`
    * .vtr: :class:`XMLRectilinearGridReader`
    * .pvtr: :class:`XMLPRectilinearGridReader`
    * .vtu: :class:`XMLUnstructuredGridReader`
    * .pvtu: :class:`XMLPUnstructuredGridReader`
    * .vtp: :class:`XMLPolyDataReader`
    * .vts: :class:`XMLStructuredGridReader`
    * .vtm: :class:`XMLMultiBlockDataReader`
    * .vtmb: :class:`XMLMultiBlockDataReader`
    * .case: :class:`EnSightReader`
    * .foam: :class:`OpenFOAMReader`

    Parameters
    ----------
    filename : str
        The string path to the file to read.

    Returns
    -------
    :class:`pyvista.BaseReader`
        A subclass of `pyvista.BaseReader` is returned based on file type.

    """
    ext = get_ext(filename)

    try:
        Reader = CLASS_READERS[ext]
    except KeyError:
        raise ValueError(f"{ext} not supported")

    return Reader(filename)

class BaseReader(ABC):
    """The base reader class."""

    def __init__(self, filename):
        """Initialize Reader by setting filename."""
        self.filename = filename
        self._set_filename(filename)

    @property
    @abstractmethod
    def reader(self):
        """Return the vtk Reader object."""
        pass

    def _set_filename(self, filename):
        """Set filename and update reader."""
        # Private method since changing file type requires a
        # different subclass.
        self.filename = filename
        self.reader.SetFileName(filename)
        self.update()

    def read(self):
        """Read data in file.

        Returns
        -------
        :class:`pyvista.DataSet`

        """
        self.update()
        data = wrap(self.reader.GetOutputDataObject(0))
        data._post_file_load_processing()
        return data

    def update(self):
        """Update reader."""
        self.reader.Update()


class DataArraySelection:
    """Mixin for readers that support data array selections."""

    @property
    def number_point_arrays(self):
        """Return the number of point arrays."""
        return self.reader.GetNumberOfPointArrays()

    @property
    def point_array_names(self):
        """Return the list of all point array names."""
        return [self.reader.GetPointArrayName(i) for i in range(self.number_point_arrays)]

    def enable_point_array(self, name):
        """Enable point array with name.

        Parameters
        ----------
        name: str

        """
        self.reader.SetPointArrayStatus(name, 1)

    def disable_point_array(self, name):
        """Disable point array with name.

        Parameters
        ----------
        name: str

        """
        self.reader.SetPointArrayStatus(name, 0)

    def point_array_status(self, name):
        """Get status of point array with name.

        Parameters
        ----------
        name: str

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
        """Return the status of all point arrays as a dict."""
        return {name: self.point_array_status(name) for name in self.point_array_names}

    @property
    def number_cell_arrays(self):
        """Return the number of cell arrays."""
        return self.reader.GetNumberOfCellArrays()

    @property
    def cell_array_names(self):
        """Return the list of all cell array names."""
        return [self.reader.GetCellArrayName(i) for i in range(self.number_cell_arrays)]

    def enable_cell_array(self, name):
        """Enable cell array with name.

        Parameters
        ----------
        name: str

        """
        self.reader.SetCellArrayStatus(name, 1)

    def disable_cell_array(self, name):
        """Disable cell array with name.

        Parameters
        ----------
        name: str

        """
        self.reader.SetCellArrayStatus(name, 0)

    def cell_array_status(self, name):
        """Get status of cell array with name.

        Parameters
        ----------
        name: str

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
        """Return the status of all cell arrays as a dict."""
        return {name: self.cell_array_status(name) for name in self.cell_array_names}


class XMLImageDataReader(BaseReader, DataArraySelection):
    """XML Image Data Reader."""

    def __init__(self, filename):
        """Initialize XMLImageDataReader."""
        self._reader = _vtk.vtkXMLImageDataReader()
        super().__init__(filename)

    @property
    def reader(self):
        """Return vtkXMLImageDataReader object."""
        return self._reader


class XMLPImageDataReader(BaseReader, DataArraySelection):
    """XML P Image Data Reader."""

    def __init__(self, filename):
        """Initialize XMLPImageDataReader."""
        self._reader = _vtk.vtkXMLPImageDataReader()
        super().__init__(filename)

    @property
    def reader(self):
        """Return vtkXMLPImageDataReader object."""
        return self._reader


class XMLRectilinearGridReader(BaseReader, DataArraySelection):
    """XML RectilinearGrid Reader."""

    def __init__(self, filename):
        """Initialize XMLRectilinearGridReader."""
        self._reader = _vtk.vtkXMLRectilinearGridReader()
        super().__init__(filename)

    @property
    def reader(self):
        """Return vtkXMLRectilinearGridReader object."""
        return self._reader


class XMLPRectilinearGridReader(BaseReader, DataArraySelection):
    """XML P RectilinearGrid Reader."""

    def __init__(self, filename):
        """Initialize XMLPRectilinearGridReader."""
        self._reader = _vtk.vtkXMLPRectilinearGridReader()
        super().__init__(filename)

    @property
    def reader(self):
        """Return vtkXMLPRectilinearGridReader object."""
        return self._reader


class XMLUnstructuredGridReader(BaseReader, DataArraySelection):
    """XML UnstructuredGrid Reader."""

    def __init__(self, filename):
        """Initialize XMLUnstructuredGridReader."""
        self._reader = _vtk.vtkXMLUnstructuredGridReader()
        super().__init__(filename)

    @property
    def reader(self):
        """Return vtkXMLUnstructuredGridReader object."""
        return self._reader

class XMLPUnstructuredGridReader(BaseReader, DataArraySelection):
    """XML P UnstructuredGrid Reader."""

    def __init__(self, filename):
        """Initialize XMLPUnstructuredGridReader."""
        self._reader = _vtk.vtkXMLPUnstructuredGridReader()
        super().__init__(filename)

    @property
    def reader(self):
        """Return vtkXMLPUnstructuredGridReader object."""
        return self._reader


class XMLPolyDataReader(BaseReader, DataArraySelection):
    """XML PolyData Reader."""

    def __init__(self, filename):
        """Initialize XMLPolyDataReader."""
        self._reader = _vtk.vtkXMLPolyDataReader()
        super().__init__(filename)

    @property
    def reader(self):
        """Return vtkXMLPolyDataReader object."""
        return self._reader


class XMLStructuredGridReader(BaseReader, DataArraySelection):
    """XML StructuredGrid Reader."""

    def __init__(self, filename):
        """Initialize XMLStructuredGridReader."""
        self._reader = _vtk.vtkXMLStructuredGridReader()
        super().__init__(filename)

    @property
    def reader(self):
        """Return vtkXMLStructuredGridReader object."""
        return self._reader


class XMLMultiBlockDataReader(BaseReader, DataArraySelection):
    """XML MultiBlock Data Reader."""

    def __init__(self, filename):
        """Initialize XMLMultiBlockDataReader."""
        self._reader = _vtk.vtkXMLMultiBlockDataReader()
        super().__init__(filename)

    @property
    def reader(self):
        """Return vtkXMLMultiBlockDataReader object."""
        return self._reader


class EnSightReader(BaseReader, DataArraySelection):
    """EnSight Reader."""

    def __init__(self, filename):
        """Initialize EnSightReader."""
        self._reader = _vtk.vtkGenericEnSightReader()
        super().__init__(filename)

    def _set_filename(self, filename):
        """Set filename and update reader."""
        # Private method since changing file type requires a
        # different subclass.
        self.filename = filename
        self.reader.SetCaseFileName(filename)
        self.update()

    @property
    def reader(self):
        """Return vtkGenericEnSightReader object."""
        return self._reader


class OpenFOAMReader(BaseReader, DataArraySelection):
    """OpenFOAM Reader."""

    def __init__(self, filename):
        """Initialize OpenFOAMReader."""
        self._reader = _vtk.vtkOpenFOAMReader()
        super().__init__(filename)

    @property
    def reader(self):
        """Return vtkOpenFOAMReader object."""
        return self._reader


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
}
