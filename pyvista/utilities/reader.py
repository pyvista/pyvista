"""Fine-grained control of reading data files."""

from typing import Any

from pyvista.utilities import wrap, get_ext
from pyvista import _vtk


def get_reader(filename):
    """Get a reader for fine-grained control of reading data files.

    Supported file types and Readers:
    * .vti: :class:`pyvista.XMLImageDataReader`
    * .pvti: :class:`pyvista.XMLPImageDataReader`
    * .vtr: :class:`pyvista.XMLRectilinearGridReader`
    * .pvtr: :class:`pyvista.XMLPRectilinearGridReader`
    * .vtu: :class:`pyvista.XMLUnstructuredGridReader`
    * .pvtu: :class:`pyvista.XMLPUnstructuredGridReader`
    * .vtp: :class:`pyvista.XMLPolyDataReader`
    * .vts: :class:`pyvista.XMLStructuredGridReader`
    * .vtm: :class:`pyvista.XMLMultiBlockDataReader`
    * .vtmb: :class:`pyvista.XMLMultiBlockDataReader`
    * .case: :class:`pyvista.EnSightReader`
    * .foam: :class:`pyvista.OpenFOAMReader`
    * .ply: :class:`pyvista.PLYReader`
    * .obj: :class:`pyvista.OBJReader`
    * .stl: :class:`pyvista.STLReader`
    * .vtk: :class:`pyvista.VTKDataSetReader`
    * .pvtk: :class:`pyvista.VTKPDataSetReader`
    * .g: :class:`pyvista.BYUReader`
    * .facet: :class:`pyvista.FacetReader`
    * .p3d: :class:`pyvista.Plot3DMetaReader`
    * .tri: :class:`pyvista.BinaryMarchingCubesReader`

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

class BaseReader:
    """The base reader class."""

    _class_reader: Any = None

    def __init__(self, filename):
        """Initialize Reader by setting filename."""
        self._reader = self._class_reader()
        self.filename = filename
        self._set_filename(filename)

    @property
    def reader(self):
        """Return the vtk Reader object."""
        if self._reader is None:
            raise NotImplementedError
        return self._reader

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

    _class_reader =_vtk.vtkXMLImageDataReader


class XMLPImageDataReader(BaseReader, DataArraySelection):
    """XML P Image Data Reader."""

    _class_reader = _vtk.vtkXMLPImageDataReader

    
class XMLRectilinearGridReader(BaseReader, DataArraySelection):
    """XML RectilinearGrid Reader.
    
    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_rectilinear_grid(load=False)
    >>> print(filename.split("/")[-1])  # omit the path
    RectilinearGrid.vtr
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> sliced_mesh = mesh.slice('y')
    >>> sliced_mesh.plot(scalars="Void Volume Fraction",
    ...                  cpos='xz', show_scalar_bar=False)

    """

    _class_reader = _vtk.vtkXMLRectilinearGridReader


class XMLPRectilinearGridReader(BaseReader, DataArraySelection):
    """XML P RectilinearGrid Reader."""

    _class_reader = _vtk.vtkXMLPRectilinearGridReader


class XMLUnstructuredGridReader(BaseReader, DataArraySelection):
    """XML UnstructuredGrid Reader.
    
    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_notch_displacement(load=False)
    >>> print(filename.split("/")[-1])  # omit the path
    notch_disp.vtu
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(scalars="Nodal Displacement", component=0,
    ...           cpos='xy', show_scalar_bar=False)

    """

    _class_reader = _vtk.vtkXMLUnstructuredGridReader


class XMLPUnstructuredGridReader(BaseReader, DataArraySelection):
    """XML P UnstructuredGrid Reader."""

    _class_reader = _vtk.vtkXMLPUnstructuredGridReader


class XMLPolyDataReader(BaseReader, DataArraySelection):
    """XML PolyData Reader.
    
    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_cow_head(load=False)
    >>> print(filename.split("/")[-1])  # omit the path
    cowHead.vtp
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(
    ...    cpos=((12, 3.5, -4.5), (4.5, 1.6, 0), (0, 1, 0.3)),
    ...    clim=[0, 100], show_scalar_bar=False
    ... )

    """

    _class_reader = _vtk.vtkXMLPolyDataReader


class XMLStructuredGridReader(BaseReader, DataArraySelection):
    """XML StructuredGrid Reader.
    
    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_structured_grid(load=False)
    >>> print(filename.split("/")[-1])  # omit the path
    StructuredGrid.vts
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(style='wireframe', line_width=4, show_scalar_bar=False)

    """

    _class_reader = _vtk.vtkXMLStructuredGridReader


class XMLMultiBlockDataReader(BaseReader, DataArraySelection):
    """XML MultiBlock Data Reader."""

    _class_reader = _vtk.vtkXMLMultiBlockDataReader


class EnSightReader(BaseReader, DataArraySelection):
    """EnSight Reader.
    
    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_cylinder_crossflow(load=False)
    >>> print(filename.split("/")[-1])  # omit the path
    cylinder_Re35.case
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
        self.update()


class OpenFOAMReader(BaseReader, DataArraySelection):
    """OpenFOAM Reader."""

    _class_reader = _vtk.vtkOpenFOAMReader


class PLYReader(BaseReader):
    """PLY Reader.
    
    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_lobster(load=False)
    >>> print(filename.split("/")[-1])  # omit the path
    lobster.ply
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot()

    """

    _class_reader = _vtk.vtkPLYReader


class OBJReader(BaseReader):
    """OBJ Reader.
    
    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_trumpet(load=False)
    >>> print(filename.split("/")[-1])  # omit the path
    trumpet.obj
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(cpos='yz', show_scalar_bar=False)

    """

    _class_reader = _vtk.vtkOBJReader


class STLReader(BaseReader):
    """STL Reader.
    
    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_cad_model(load=False)
    >>> print(filename.split("/")[-1])  # omit the path
    42400-IDGH.stl
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot()

    """

    _class_reader = _vtk.vtkSTLReader


class VTKDataSetReader(BaseReader):
    """VTK Data Set Reader.
    
        Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_brain(load=False)
    >>> print(filename.split("/")[-1])  # omit the path
    brain.vtk
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> sliced_mesh = mesh.slice('x')
    >>> sliced_mesh.plot(cpos='yz', show_scalar_bar=False)

    """

    _class_reader = _vtk.vtkDataSetReader


class VTKPDataSetReader(BaseReader):
    """VTK P Data Set Reader."""

    _class_reader = staticmethod(_vtk.lazy_vtkPDataSetReader)


class BYUReader(BaseReader):
    """BYU Reader.
    
    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_teapot(load=False)
    >>> print(filename.split("/")[-1])  # omit the path
    teapot.g
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(cpos='xy', show_scalar_bar=False)
    """

    _class_reader = _vtk.vtkBYUReader


class FacetReader(BaseReader):
    """Facet Reader.
    
    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_clown(load=False)
    >>> print(filename.split("/")[-1])  # omit the path
    clown.facet
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(color="red")
    """

    _class_reader = staticmethod(_vtk.lazy_vtkFacetReader)


class Plot3DMetaReader(BaseReader):
    """Plot3DMeta Reader."""

    _class_reader = staticmethod(_vtk.lazy_vtkPlot3DMetaReader)


class BinaryMarchingCubesReader(BaseReader):
    """BinaryMarchingCubes Reader.
    
    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_pine_roots(load=False)
    >>> print(filename.split("/")[-1])  # omit the path
    pine_root.tri
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(color="brown")

    """

    _class_reader = _vtk.vtkMCubesReader


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
}
