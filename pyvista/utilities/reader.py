from abc import ABC, abstractmethod
from pyvista.utilities import wrap, get_ext
from pyvista import _vtk


class Reader(ABC):
    """Fine-grained control of reading data files.
    
    Parameters
    ----------
    filename : str
        The string path to the file to read.

    """
    def __new__(cls, filename):
        """Create new Reader object from subclass matching filename."""
        ext = get_ext(filename)

        try:
            reader = CLASS_READERS[ext]
        except KeyError:
            raise ValueError(f"{ext} not supported")

        return super().__new__(reader)

    def __init__(self, filename):
        self.filename = filename
        self._set_filename(filename)

    @property
    @abstractmethod
    def reader(self):
        """vtk Reader object."""
        pass

    @property
    def number_point_arrays(self):
        """Number of point arrays."""
        return self.reader.GetNumberOfPointArrays()

    @property
    def point_array_names(self):
        """List of all point array names."""
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
        """The status of all point arrays as a dict."""
        return {name: self.point_array_status(name) for name in self.point_array_names}

    @property
    def number_cell_arrays(self):
        """Number of cell arrays."""
        return self.reader.GetNumberOfCellArrays()

    @property
    def cell_array_names(self):
        """List of all cell array names."""
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
        """The status of all cell arrays as a dict."""
        return {name: self.cell_array_status(name) for name in self.cell_array_names}

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


class XMLPolyReader(Reader):
    """XMLPolyReader class."""
    def __init__(self, filename):
        self._reader = _vtk.vtkXMLPolyDataReader()
        super().__init__(filename)

    @property
    def reader(self):
        """vtkXMLPolyReader object."""
        return self._reader

CLASS_READERS = {
    # Standard dataset readers:
    #'.vti': _vtk.vtkXMLImageDataReader,
    #'.pvti': _vtk.vtkXMLPImageDataReader,
    #'.vtr': _vtk.vtkXMLRectilinearGridReader,
    #'.pvtr': _vtk.vtkXMLPRectilinearGridReader,
    #'.vtu': _vtk.vtkXMLUnstructuredGridReader,
    #'.pvtu': _vtk.vtkXMLPUnstructuredGridReader,
    '.vtp': XMLPolyReader,
    #'.vts': _vtk.vtkXMLStructuredGridReader,
    #'.vtm': _vtk.vtkXMLMultiBlockDataReader,
    #'.vtmb': _vtk.vtkXMLMultiBlockDataReader,
}
