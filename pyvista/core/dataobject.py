"""Attributes common to PolyData and Grid Objects."""

from abc import abstractmethod
import collections.abc
import logging
from pathlib import Path
from typing import Any, DefaultDict, Dict, Type, Union
import warnings

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.utilities import FieldAssociation, abstract_class, fileio
from pyvista.utilities.misc import PyvistaDeprecationWarning

from .datasetattributes import DataSetAttributes

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')

# vector array names
DEFAULT_VECTOR_KEY = '_vectors'


@abstract_class
class DataObject:
    """Methods common to all wrapped data objects."""

    _WRITERS: Dict[str, Union[Type[_vtk.vtkXMLWriter], Type[_vtk.vtkDataWriter]]] = {}

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the data object."""
        super().__init__()
        # Remember which arrays come from numpy.bool arrays, because there is no direct
        # conversion from bool to vtkBitArray, such arrays are stored as vtkCharArray.
        self.association_bitarray_names: DefaultDict = collections.defaultdict(set)

    def __getattr__(self, item: str) -> Any:
        """Get attribute from base class if not found."""
        return super().__getattribute__(item)

    def shallow_copy(self, to_copy: _vtk.vtkDataObject) -> _vtk.vtkDataObject:
        """Shallow copy the given mesh to this mesh.

        Parameters
        ----------
        to_copy : pyvista.DataObject or vtk.vtkDataObject
            Data object to perform a shallow copy from.

        """
        self.ShallowCopy(to_copy)

    def deep_copy(self, to_copy: _vtk.vtkDataObject) -> _vtk.vtkDataObject:
        """Overwrite this data object with another data object as a deep copy.

        Parameters
        ----------
        to_copy : pyvista.DataObject or vtk.vtkDataObject
            Data object to perform a deep copy from.

        """
        self.DeepCopy(to_copy)

    def _from_file(self, filename: Union[str, Path], **kwargs):
        data = pyvista.read(filename, **kwargs)
        if not isinstance(self, type(data)):
            raise ValueError(f'Reading file returned data of `{type(data).__name__}`, '
                             f'but `{type(self).__name__}` was expected.')
        self.shallow_copy(data)
        self._post_file_load_processing()

    def _post_file_load_processing(self):
        """Execute after loading a dataset from file, to be optionally overridden by subclasses."""
        pass

    def save(self, filename: str, binary=True, texture=None):
        """Save this vtk object to file.

        Parameters
        ----------
        filename : str, pathlib.Path
            Filename of output file. Writer type is inferred from
            the extension of the filename.

        binary : bool, optional
            If ``True``, write as binary.  Otherwise, write as ASCII.

        texture : str, np.ndarray, optional
            Write a single texture array to file when using a PLY
            file.  Texture array must be a 3 or 4 component array with
            the datatype ``np.uint8``.  Array may be a cell array or a
            point array, and may also be a string if the array already
            exists in the PolyData.

            If a string is provided, the texture array will be saved
            to disk as that name.  If an array is provided, the
            texture array will be saved as ``'RGBA'``

            .. note::
               This feature is only available when saving PLY files.

        Notes
        -----
        Binary files write much faster than ASCII and have a smaller
        file size.

        """
        if self._WRITERS is None:
            raise NotImplementedError(f'{self.__class__.__name__} writers are not specified,'
                                      ' this should be a dict of (file extension: vtkWriter type)')

        file_path = Path(filename)
        file_path = file_path.expanduser()
        file_path = file_path.resolve()
        file_ext = file_path.suffix
        if file_ext not in self._WRITERS:
            raise ValueError('Invalid file extension for this data type.'
                             f' Must be one of: {self._WRITERS.keys()}')

        writer = self._WRITERS[file_ext]()
        fileio.set_vtkwriter_mode(vtk_writer=writer, use_binary=binary)
        writer.SetFileName(str(file_path))
        writer.SetInputData(self)
        if file_ext == '.ply' and texture is not None:
            if isinstance(texture, str):
                writer.SetArrayName(texture)
                array_name = texture
            elif isinstance(texture, np.ndarray):
                array_name = '_color_array'
                self[array_name] = texture
                writer.SetArrayName(array_name)

            # enable alpha channel if applicable
            if self[array_name].shape[-1] == 4:  # type: ignore
                writer.SetEnableAlpha(True)
        writer.Write()

    @abstractmethod
    def get_data_range(self):  # pragma: no cover
        """Get the non-NaN min and max of a named array."""
        raise NotImplementedError(f'{type(self)} mesh type does not have a `get_data_range` method.')

    def _get_attrs(self):  # pragma: no cover
        """Return the representation methods (internal helper)."""
        raise NotImplementedError('Called only by the inherited class')

    def head(self, display=True, html: bool = None):
        """Return the header stats of this dataset.

        If in IPython, this will be formatted to HTML. Otherwise
        returns a console friendly string.

        Parameters
        ----------
        display : bool, optional
            Display this header in iPython.

        html : bool, optional
            Generate the output as HTML.

        Returns
        -------
        str
            Header statistics.

        """
        # Generate the output
        if html:
            fmt = ""
            # HTML version
            fmt += "\n"
            fmt += "<table>\n"
            fmt += f"<tr><th>{type(self).__name__}</th><th>Information</th></tr>\n"
            row = "<tr><td>{}</td><td>{}</td></tr>\n"
            # now make a call on the object to get its attributes as a list of len 2 tuples
            for attr in self._get_attrs():
                try:
                    fmt += row.format(attr[0], attr[2].format(*attr[1]))
                except:
                    fmt += row.format(attr[0], attr[2].format(attr[1]))
            if hasattr(self, 'n_arrays'):
                fmt += row.format('N Arrays', self.n_arrays)
            fmt += "</table>\n"
            fmt += "\n"
            if display:
                from IPython.display import HTML, display as _display
                _display(HTML(fmt))
                return
            return fmt
        # Otherwise return a string that is Python console friendly
        fmt = f"{type(self).__name__} ({hex(id(self))})\n"
        # now make a call on the object to get its attributes as a list of len 2 tuples
        row = "  {}:\t{}\n"
        for attr in self._get_attrs():
            try:
                fmt += row.format(attr[0], attr[2].format(*attr[1]))
            except:
                fmt += row.format(attr[0], attr[2].format(attr[1]))
        if hasattr(self, 'n_arrays'):
            fmt += row.format('N Arrays', self.n_arrays)
        return fmt

    def _repr_html_(self):  # pragma: no cover
        """Return a pretty representation for Jupyter notebooks.

        This includes header details and information about all arrays.

        """
        raise NotImplementedError('Called only by the inherited class')

    def copy_meta_from(self, ido):  # pragma: no cover
        """Copy pyvista meta data onto this object from another object."""
        pass  # called only by the inherited class

    def copy(self, deep=True):
        """Return a copy of the object.

        Parameters
        ----------
        deep : bool, optional
            When ``True`` makes a full copy of the object.  When
            ``False``, performs a shallow copy where the points, cell,
            and data arrays are references to the original object.

        Returns
        -------
        pyvista.DataSet
            Deep or shallow copy of the input.  Type is identical to
            the input.

        Examples
        --------
        Create and make a deep copy of a PolyData object.

        >>> import pyvista
        >>> mesh_a = pyvista.Sphere()
        >>> mesh_b = mesh_a.copy()
        >>> mesh_a == mesh_b
        True

        """
        thistype = type(self)
        newobject = thistype()

        if deep:
            newobject.deep_copy(self)
        else:
            newobject.shallow_copy(self)
        newobject.copy_meta_from(self)
        return newobject

    def __eq__(self, other):
        """Test equivalency between data objects."""
        if not isinstance(self, type(other)):
            return False

        if self is other:
            return True

        # these attrs use numpy.array_equal
        equal_attrs = ['verts',  # DataObject
                       'points',  # DataObject
                       'lines',  # DataObject
                       'faces',  # DataObject
                       'cells',  # UnstructuredGrid
                       'celltypes']  # UnstructuredGrid
        for attr in equal_attrs:
            if hasattr(self, attr):
                if not np.array_equal(getattr(self, attr), getattr(other, attr)):
                    return False

        # these attrs can be directly compared
        attrs = ['field_data', 'point_data', 'cell_data']
        for attr in attrs:
            if hasattr(self, attr):
                if getattr(self, attr) != getattr(other, attr):
                    return False

        return True

    def add_field_array(self, scalars: np.ndarray, name: str,
                        deep=True):  # pragma: no cover
        """Add field data.

        .. deprecated:: 0.32.0
           Use :func:`DataObject.add_field_data` instead.
        """
        warnings.warn( "Use of `clear_point_arrays` is deprecated. "
            "Use `clear_point_data` instead.",
            PyvistaDeprecationWarning
        )
        return self.clear_point_data()

    def add_field_data(self, array: np.ndarray, name: str, deep=True):
        """Add field data.

        Use field data when size of the data you wish to associate
        with the dataset does not match the number of points or cells
        of the dataset.

        Parameters
        ----------
        array : sequence
            Array of data to add to the dataset as a field array.

        name : str
            Name to assign the field array.

        deep : bool, optional
            Perform a deep copy of the data when adding it to the
            dataset.  Default ``True``.

        Examples
        --------
        Add field data to a PolyData dataset.

        >>> import pyvista
        >>> import numpy as np
        >>> mesh = pyvista.Sphere()
        >>> mesh.add_field_data(np.arange(10), 'my-field-data')
        >>> mesh['my-field-data']
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        Add field data to a UniformGrid dataset.

        >>> mesh = pyvista.UniformGrid(dims=(2, 2, 1))
        >>> mesh.add_field_data(['I could', 'write', 'notes', 'here'],
        ...                      'my-field-data')
        >>> mesh['my-field-data']
        array(['I could', 'write', 'notes', 'here'], dtype='<U7')

        Add field data to a MultiBlock dataset.

        >>> blocks = pyvista.MultiBlock()
        >>> blocks.append(pyvista.Sphere())
        >>> blocks["cube"] = pyvista.Cube(center=(0, 0, -1))
        >>> blocks.add_field_data([1, 2, 3], 'my-field-data')
        >>> blocks.field_data['my-field-data']
        pyvista_ndarray([1, 2, 3])

        """
        self.field_data.set_array(array, name, deep_copy=deep)

    @property
    def field_arrays(self) -> DataSetAttributes:  # pragma: no cover
        """Return vtkFieldData as DataSetAttributes.

        .. deprecated:: 0.32.0
            Use :attr:`DataObject.field_data` to return field data.

        """
        warnings.warn(
            "Use of `field_arrays` is deprecated. "
            "Use `field_data` instead.",
            PyvistaDeprecationWarning
        )
        return self.field_data

    @property
    def field_data(self) -> DataSetAttributes:
        """Return FieldData as DataSetAttributes.

        Use field data when size of the data you wish to associate
        with the dataset does not match the number of points or cells
        of the dataset.

        Examples
        --------
        Add field data to a PolyData dataset and then return it.

        >>> import pyvista
        >>> import numpy as np
        >>> mesh = pyvista.Sphere()
        >>> mesh.field_data['my-field-data'] = np.arange(10)
        >>> mesh.field_data['my-field-data']
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        """
        return DataSetAttributes(self.GetFieldData(), dataset=self, association=FieldAssociation.NONE)

    def clear_field_arrays(self):  # pragma: no cover
        """Remove all field data.

        .. deprecated:: 0.32.0
            Use :func:`DataObject.clear_field_data` instead.

        """
        warnings.warn(
            "Use of `clear_field_arrays` is deprecated. "
            "Use `clear_field_data` instead.",
            PyvistaDeprecationWarning
        )
        self.field_data

    def clear_field_data(self):
        """Remove all field data.

        Examples
        --------
        Add field data to a PolyData dataset and then remove it.

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.field_data['my-field-data'] = range(10)
        >>> len(mesh.field_data)
        1
        >>> mesh.clear_field_data()
        >>> len(mesh.field_data)
        0

        """
        self.field_data.clear()

    @property
    def memory_address(self) -> str:
        """Get address of the underlying VTK C++ object.

        Returns
        -------
        str
            Memory address formatted as ``'Addr=%p'``.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.memory_address
        'Addr=...'

        """
        return self.GetInformation().GetAddressAsString("")

    @property
    def actual_memory_size(self) -> int:
        """Return the actual size of the dataset object.

        Returns
        -------
        int
            The actual size of the dataset object in kibibytes (1024
            bytes).

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> mesh.actual_memory_size  # doctest:+SKIP
        93

        """
        return self.GetActualMemorySize()

    def copy_structure(self, dataset: _vtk.vtkDataSet):
        """Copy the structure (geometry and topology) of the input dataset object.

        Parameters
        ----------
        dataset : vtk.vtkDataSet
            Dataset to copy the geometry and topology from.

        Examples
        --------
        >>> import pyvista as pv
        >>> source = pv.UniformGrid(dims=(10, 10, 5))
        >>> target = pv.UniformGrid()
        >>> target.copy_structure(source)
        >>> target.plot(show_edges=True)

        """
        self.CopyStructure(dataset)

    def copy_attributes(self, dataset: _vtk.vtkDataSet):
        """Copy the data attributes of the input dataset object.

        Parameters
        ----------
        dataset : pyvista.DataSet
            Dataset to copy the data attributes from.

        Examples
        --------
        >>> import pyvista as pv
        >>> source = pv.UniformGrid(dims=(10, 10, 5))
        >>> source = source.compute_cell_sizes()
        >>> target = pv.UniformGrid(dims=(10, 10, 5))
        >>> target.copy_attributes(source)
        >>> target.plot(scalars='Volume', show_edges=True)

        """
        self.CopyAttributes(dataset)

    def __getstate__(self):
        """Support pickle. Serialize the VTK object to ASCII string."""
        state = self.__dict__.copy()
        writer = _vtk.vtkDataSetWriter()
        writer.SetInputDataObject(self)
        writer.SetWriteToOutputString(True)
        writer.SetFileTypeToBinary()
        writer.Write()
        to_serialize = writer.GetOutputStdString()
        state['vtk_serialized'] = to_serialize
        return state

    def __setstate__(self, state):
        """Support unpickle."""
        vtk_serialized = state.pop('vtk_serialized')
        self.__dict__.update(state)
        reader = _vtk.vtkDataSetReader()
        reader.ReadFromInputStringOn()
        if isinstance(vtk_serialized, bytes):
            reader.SetBinaryInputString(vtk_serialized, len(vtk_serialized))
        elif isinstance(vtk_serialized, str):
            reader.SetInputString(vtk_serialized)
        reader.Update()
        mesh = pyvista.wrap(reader.GetOutput())

        # copy data
        self.copy_structure(mesh)
        self.copy_attributes(mesh)
