"""Attributes common to PolyData and Grid Objects."""

from __future__ import annotations

from abc import abstractmethod
from collections import UserDict
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

import numpy as np

import pyvista

from . import _vtk_core as _vtk
from .datasetattributes import DataSetAttributes
from .pyvista_ndarray import pyvista_ndarray
from .utilities.arrays import FieldAssociation
from .utilities.arrays import _JSONValueType
from .utilities.arrays import _SerializedDictArray
from .utilities.fileio import read
from .utilities.fileio import set_vtkwriter_mode
from .utilities.helpers import wrap
from .utilities.misc import abstract_class

if TYPE_CHECKING:  # pragma: no cover
    from ._typing_core import NumpyArray

# vector array names
DEFAULT_VECTOR_KEY = '_vectors'


@abstract_class
class DataObject:
    """Methods common to all wrapped data objects.

    Parameters
    ----------
    *args :
        Any extra args are passed as option to all wrapped data objects.

    **kwargs :
        Any extra keyword args are passed as option to all wrapped data objects.

    """

    _WRITERS: ClassVar[dict[str, type[_vtk.vtkXMLWriter | _vtk.vtkDataWriter]]] = {}

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the data object."""
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            # super() maps to object
            super().__init__()
        # Remember which arrays come from numpy.bool arrays, because there is no direct
        # conversion from bool to vtkBitArray, such arrays are stored as vtkCharArray.
        self._association_bitarray_names: defaultdict[Any, Any] = defaultdict(set)

        # view these arrays as complex128 as VTK doesn't support complex types
        self._association_complex_names: defaultdict[Any, Any] = defaultdict(set)

    def __getattr__(self, item: str) -> Any:
        """Get attribute from base class if not found."""
        return super().__getattribute__(item)

    def shallow_copy(self, to_copy: _vtk.vtkDataObject) -> None:
        """Shallow copy the given mesh to this mesh.

        Parameters
        ----------
        to_copy : pyvista.DataObject or vtk.vtkDataObject
            Data object to perform a shallow copy from.

        """
        self.ShallowCopy(to_copy)

    def deep_copy(self, to_copy: _vtk.vtkDataObject) -> None:
        """Overwrite this data object with another data object as a deep copy.

        Parameters
        ----------
        to_copy : pyvista.DataObject or vtk.vtkDataObject
            Data object to perform a deep copy from.

        """
        self.DeepCopy(to_copy)

    def _from_file(self, filename: str | Path, **kwargs):
        """Read data objects from file."""
        data = read(filename, **kwargs)
        if not isinstance(self, type(data)):
            raise ValueError(
                f'Reading file returned data of `{type(data).__name__}`, '
                f'but `{type(self).__name__}` was expected.',
            )
        self.shallow_copy(data)
        self._post_file_load_processing()

    def _post_file_load_processing(self):
        """Execute after loading a dataset from file, to be optionally overridden by subclasses."""

    def save(
        self,
        filename: Path | str,
        binary: bool = True,
        texture: NumpyArray[np.uint8] | str | None = None,
    ) -> None:
        """Save this vtk object to file.

        Parameters
        ----------
        filename : str, pathlib.Path
            Filename of output file. Writer type is inferred from
            the extension of the filename.

        binary : bool, default: True
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
            raise NotImplementedError(
                f'{self.__class__.__name__} writers are not specified,'
                ' this should be a dict of (file extension: vtkWriter type)',
            )

        file_path = Path(filename)
        file_path = file_path.expanduser()
        file_path = file_path.resolve()
        file_ext = file_path.suffix
        if file_ext not in self._WRITERS:
            raise ValueError(
                'Invalid file extension for this data type.'
                f' Must be one of: {self._WRITERS.keys()}',
            )

        # store complex and bitarray types as field data
        self._store_metadata()

        writer = self._WRITERS[file_ext]()
        set_vtkwriter_mode(vtk_writer=writer, use_binary=binary)
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
            if self[array_name].shape[-1] == 4:  # type: ignore[index]
                writer.SetEnableAlpha(True)
        writer.Write()

    def _store_metadata(self) -> None:
        """Store metadata as field data."""
        fdata = self.field_data
        for assoc_name in ('bitarray', 'complex'):
            for assoc_type in ('POINT', 'CELL'):
                assoc_data = getattr(self, f'_association_{assoc_name}_names')
                array_names = assoc_data.get(assoc_type)
                if array_names:
                    key = f'_PYVISTA_{assoc_name}_{assoc_type}_'.upper()
                    fdata[key] = list(array_names)

    def _restore_metadata(self) -> None:
        """Restore PyVista metadata from field data.

        Metadata is stored using ``_store_metadata`` and contains entries in
        the format of f'_PYVISTA_{assoc_name}_{assoc_type}_'. These entries are
        removed when calling this method.

        """
        fdata = self.field_data
        for assoc_name in ('bitarray', 'complex'):
            for assoc_type in ('POINT', 'CELL'):
                key = f'_PYVISTA_{assoc_name}_{assoc_type}_'.upper()
                if key in fdata:
                    assoc_data = getattr(self, f'_association_{assoc_name}_names')
                    assoc_data[assoc_type] = set(fdata[key])
                    del fdata[key]

    @abstractmethod
    def get_data_range(self):  # pragma: no cover
        """Get the non-NaN min and max of a named array."""
        raise NotImplementedError(
            f'{type(self)} mesh type does not have a `get_data_range` method.',
        )

    def _get_attrs(self):  # pragma: no cover
        """Return the representation methods (internal helper)."""
        raise NotImplementedError('Called only by the inherited class')

    def head(self, display=True, html=None):
        """Return the header stats of this dataset.

        If in IPython, this will be formatted to HTML. Otherwise
        returns a console friendly string.

        Parameters
        ----------
        display : bool, default: True
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
            fmt += "<table style='width: 100%;'>\n"
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
                from IPython.display import HTML
                from IPython.display import display as _display

                _display(HTML(fmt))
                return None
            return fmt
        # Otherwise return a string that is Python console friendly
        fmt = f"{type(self).__name__} ({hex(id(self))})\n"
        # now make a call on the object to get its attributes as a list of len 2 tuples
        # get longest row header
        max_len = max(len(attr[0]) for attr in self._get_attrs()) + 4

        # now make a call on the object to get its attributes as a list of len
        # 2 tuples
        row = "  {:%ds}{}\n" % max_len
        for attr in self._get_attrs():
            try:
                fmt += row.format(attr[0] + ':', attr[2].format(*attr[1]))
            except:
                fmt += row.format(attr[0] + ':', attr[2].format(attr[1]))
        if hasattr(self, 'n_arrays'):
            fmt += row.format('N Arrays:', self.n_arrays)
        return fmt.strip()

    def _repr_html_(self):  # pragma: no cover
        """Return a pretty representation for Jupyter notebooks.

        This includes header details and information about all arrays.

        """
        raise NotImplementedError('Called only by the inherited class')

    def copy_meta_from(self, *args, **kwargs):  # pragma: no cover
        """Copy pyvista meta data onto this object from another object.

        Intended to be overridden by subclasses.

        Parameters
        ----------
        *args : tuple
            Positional arguments.

        **kwargs : dict, optional
            Keyword arguments.

        """
        # called only by the inherited class

    def copy(self, deep=True):
        """Return a copy of the object.

        Parameters
        ----------
        deep : bool, default: True
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

        >>> import pyvista as pv
        >>> mesh_a = pv.Sphere()
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
        newobject.copy_meta_from(self, deep)
        return newobject

    def __eq__(self, other: object) -> bool:
        """Test equivalency between data objects."""
        if not isinstance(self, type(other)):
            return False

        if self is other:
            return True

        # these attrs use numpy.array_equal
        equal_attrs = [
            'verts',  # DataObject
            'points',  # DataObject
            'lines',  # DataObject
            'faces',  # DataObject
            'cells',  # UnstructuredGrid
            'celltypes',
        ]  # UnstructuredGrid
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

    def add_field_data(self, array: NumpyArray[float], name: str, deep: bool = True):
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

        deep : bool, default: True
            Perform a deep copy of the data when adding it to the
            dataset.

        Examples
        --------
        Add field data to a PolyData dataset.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> mesh = pv.Sphere()
        >>> mesh.add_field_data(np.arange(10), 'my-field-data')
        >>> mesh['my-field-data']
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        Add field data to a ImageData dataset.

        >>> mesh = pv.ImageData(dimensions=(2, 2, 1))
        >>> mesh.add_field_data(
        ...     ['I could', 'write', 'notes', 'here'], 'my-field-data'
        ... )
        >>> mesh['my-field-data']
        pyvista_ndarray(['I could', 'write', 'notes', 'here'], dtype='<U7')

        Add field data to a MultiBlock dataset.

        >>> blocks = pv.MultiBlock()
        >>> blocks.append(pv.Sphere())
        >>> blocks["cube"] = pv.Cube(center=(0, 0, -1))
        >>> blocks.add_field_data([1, 2, 3], 'my-field-data')
        >>> blocks.field_data['my-field-data']
        pyvista_ndarray([1, 2, 3])

        """
        if not hasattr(self, 'field_data'):
            raise NotImplementedError(f'`{type(self)}` does not support field data')

        self.field_data.set_array(array, name, deep_copy=deep)

    @property
    def field_data(self) -> DataSetAttributes:
        """Return FieldData as DataSetAttributes.

        Use field data when size of the data you wish to associate
        with the dataset does not match the number of points or cells
        of the dataset.

        Returns
        -------
        DataSetAttributes
            FieldData as DataSetAttributes.

        Examples
        --------
        Add field data to a PolyData dataset and then return it.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> mesh = pv.Sphere()
        >>> mesh.field_data['my-field-data'] = np.arange(10)
        >>> mesh.field_data['my-field-data']
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        """
        return DataSetAttributes(
            self.GetFieldData(),
            dataset=self,
            association=FieldAssociation.NONE,
        )

    def clear_field_data(self) -> None:
        """Remove all field data.

        Examples
        --------
        Add field data to a PolyData dataset and then remove it.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.field_data['my-field-data'] = range(10)
        >>> len(mesh.field_data)
        1
        >>> mesh.clear_field_data()
        >>> len(mesh.field_data)
        0

        """
        if not hasattr(self, 'field_data'):
            raise NotImplementedError(f'`{type(self)}` does not support field data')

        self.field_data.clear()

    @property
    def user_dict(self) -> _SerializedDictArray:
        """Set or return a user-specified data dictionary.

        The dictionary is stored as a JSON-serialized string as part of the mesh's
        field data. Unlike regular field data, which requires values to be stored
        as an array, the user dict provides a mapping for scalar values.

        Since the user dict is stored as field data, it is automatically saved
        with the mesh when it is saved in a compatible file format (e.g. ``'.vtk'``).
        Any saved metadata is automatically de-serialized by PyVista whenever
        the user dict is accessed again. Since the data is stored as JSON, it
        may also be easily retrieved or read by other programs.

        Any JSON-serializable values are permitted by the user dict, i.e. values
        can have type ``dict``, ``list``, ``tuple``, ``str``, ``int``, ``float``,
        ``bool``, or ``None``. Storing NumPy arrays is not directly supported, but
        these may be cast beforehand to a supported type, e.g. by calling ``tolist()``
        on the array.

        To completely remove the user dict string from the dataset's field data,
        set its value to ``None``.

        .. note::

            The user dict is a convenience property and is intended for metadata storage.
            It has an inefficient dictionary implementation and should only be used to
            store a small number of infrequently-accessed keys with relatively small
            values. It should not be used to store frequently accessed array data
            with many entries (a regular field data array should be used instead).

        .. warning::

            Field data is typically passed-through by dataset filters, and therefore
            the user dict's items can generally be expected to persist and remain
            unchanged in the output of filtering methods. However, this behavior is
            not guaranteed, as it's possible that some filters may modify or clear
            field data. Use with caution.

        Returns
        -------
        UserDict
            JSON-serialized dict-like object which is subclassed from :py:class:`collections.UserDict`.

        Examples
        --------
        Load a mesh.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.load_ant()

        Add data to the user dict. The contents are serialized as JSON.

        >>> mesh.user_dict['name'] = 'ant'
        >>> mesh.user_dict
        {"name": "ant"}

        Alternatively, set the user dict from an existing dict.

        >>> mesh.user_dict = dict(name='ant')

        The user dict can be updated like a regular dict.

        >>> mesh.user_dict.update(
        ...     {
        ...         'num_legs': 6,
        ...         'body_parts': ['head', 'thorax', 'abdomen'],
        ...     }
        ... )
        >>> mesh.user_dict
        {"name": "ant", "num_legs": 6, "body_parts": ["head", "thorax", "abdomen"]}

        Data in the user dict is stored as field data.

        >>> mesh.field_data
        pyvista DataSetAttributes
        Association     : NONE
        Contains arrays :
            _PYVISTA_USER_DICT      str        "{"name": "ant",..."

        Since it's field data, the user dict can be saved to file along with the
        mesh and retrieved later.

        >>> mesh.save('ant.vtk')
        >>> mesh_from_file = pv.read('ant.vtk')
        >>> mesh_from_file.user_dict
        {"name": "ant", "num_legs": 6, "body_parts": ["head", "thorax", "abdomen"]}

        """
        self._config_user_dict()
        return self._user_dict

    @user_dict.setter
    def user_dict(
        self,
        dict_: dict[str, _JSONValueType] | UserDict,  # type: ignore[type-arg]
    ):  # numpydoc ignore=GL08
        # Setting None removes the field data array
        if dict_ is None and '_PYVISTA_USER_DICT' in self.field_data.keys():
            del self.field_data['_PYVISTA_USER_DICT']
            return

        self._config_user_dict()
        if isinstance(dict_, dict):
            self._user_dict.data = dict_
        elif isinstance(dict_, UserDict):
            self._user_dict.data = dict_.data
        else:
            raise TypeError(
                f'User dict can only be set with type {dict} or {UserDict}.\nGot {type(dict_)} instead.',
            )

    def _config_user_dict(self):
        """Init serialized dict array and ensure it is added to field_data."""
        field_name = '_PYVISTA_USER_DICT'
        field_data = self.field_data

        if not hasattr(self, '_user_dict'):
            # Init
            self._user_dict = _SerializedDictArray()

        if field_name in field_data.keys():
            if isinstance(array := field_data[field_name], pyvista_ndarray):
                # When loaded from file, field will be cast as pyvista ndarray
                # Convert to string and initialize new user dict object from it
                self._user_dict = _SerializedDictArray(''.join(array))
            elif isinstance(array, str) and repr(self._user_dict) != array:
                # Filters may update the field data block separately, e.g.
                # when copying field data, so we need to capture the new
                # string and re-init
                self._user_dict = _SerializedDictArray(array)
            else:
                # User dict is correctly configured, do nothing
                return

        # Set field data array directly instead of calling 'set_array'
        # This skips the call to '_prepare_array' which will otherwise
        # do all kinds of casting/conversions and mangle this array
        self._user_dict.SetName(field_name)
        field_data.VTKObject.AddArray(self._user_dict)
        field_data.VTKObject.Modified()

    @property
    def memory_address(self) -> str:
        """Get address of the underlying VTK C++ object.

        Returns
        -------
        str
            Memory address formatted as ``'Addr=%p'``.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
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

    def copy_structure(self, dataset: _vtk.vtkDataSet) -> None:
        """Copy the structure (geometry and topology) of the input dataset object.

        Parameters
        ----------
        dataset : vtk.vtkDataSet
            Dataset to copy the geometry and topology from.

        Examples
        --------
        >>> import pyvista as pv
        >>> source = pv.ImageData(dimensions=(10, 10, 5))
        >>> target = pv.ImageData()
        >>> target.copy_structure(source)
        >>> target.plot(show_edges=True)

        """
        self.CopyStructure(dataset)

    def copy_attributes(self, dataset: _vtk.vtkDataSet) -> None:
        """Copy the data attributes of the input dataset object.

        Parameters
        ----------
        dataset : pyvista.DataSet
            Dataset to copy the data attributes from.

        Examples
        --------
        >>> import pyvista as pv
        >>> source = pv.ImageData(dimensions=(10, 10, 5))
        >>> source = source.compute_cell_sizes()
        >>> target = pv.ImageData(dimensions=(10, 10, 5))
        >>> target.copy_attributes(source)
        >>> target.plot(scalars='Volume', show_edges=True)

        """
        self.CopyAttributes(dataset)

    def __getstate__(self):
        """Support pickle by serializing the VTK object data to something which can be pickled natively.

        The format of the serialized VTK object data depends on `pyvista.PICKLE_FORMAT` (case-insensitive).
        - If `pyvista.PICKLE_FORMAT == 'xml'`, the data is serialized as an XML-formatted string.
        - If `pyvista.PICKLE_FORMAT == 'legacy'`, the data is serialized to bytes in VTK's binary format.
        """
        state = self.__dict__.copy()

        if pyvista.PICKLE_FORMAT.lower() == 'xml':
            # the generic VTK XML writer `vtkXMLDataSetWriter` currently has a bug where it does not pass all
            # settings down to the sub-writers. Until this is fixed, use the dataset-specific writers
            # https://gitlab.kitware.com/vtk/vtk/-/issues/18661
            writers = {
                _vtk.vtkImageData: _vtk.vtkXMLImageDataWriter,
                _vtk.vtkStructuredGrid: _vtk.vtkXMLStructuredGridWriter,
                _vtk.vtkRectilinearGrid: _vtk.vtkXMLRectilinearGridWriter,
                _vtk.vtkUnstructuredGrid: _vtk.vtkXMLUnstructuredGridWriter,
                _vtk.vtkPolyData: _vtk.vtkXMLPolyDataWriter,
                _vtk.vtkTable: _vtk.vtkXMLTableWriter,
            }

            for parent_type, writer_type in writers.items():
                if isinstance(self, parent_type):
                    writer = writer_type()
                    break
            else:
                raise TypeError(f'Cannot pickle dataset of type {self.GetDataObjectType()}')

            writer.SetInputDataObject(self)
            writer.SetWriteToOutputString(True)
            writer.SetDataModeToBinary()
            writer.SetCompressorTypeToNone()
            writer.Write()
            to_serialize = writer.GetOutputString()

        elif pyvista.PICKLE_FORMAT.lower() == 'legacy':
            writer = _vtk.vtkDataSetWriter()
            writer.SetInputDataObject(self)
            writer.SetWriteToOutputString(True)
            writer.SetFileTypeToBinary()
            writer.Write()
            to_serialize = writer.GetOutputStdString()

        state['vtk_serialized'] = to_serialize

        # this needs to be here because in multiprocessing situations, `pyvista.PICKLE_FORMAT` is not shared between
        # processes
        state['PICKLE_FORMAT'] = pyvista.PICKLE_FORMAT
        return state

    def __setstate__(self, state):
        """Support unpickle."""
        vtk_serialized = state.pop('vtk_serialized')
        pickle_format = state.pop(
            'PICKLE_FORMAT',
            'legacy',  # backwards compatibility - assume 'legacy'
        )
        self.__dict__.update(state)

        if pickle_format.lower() == 'xml':
            # the generic VTK XML reader `vtkXMLGenericDataObjectReader` currently has a bug where it does not pass all
            # settings down to the sub-readers. Until this is fixed, use the dataset-specific readers
            # https://gitlab.kitware.com/vtk/vtk/-/issues/18661
            readers = {
                _vtk.vtkImageData: _vtk.vtkXMLImageDataReader,
                _vtk.vtkStructuredGrid: _vtk.vtkXMLStructuredGridReader,
                _vtk.vtkRectilinearGrid: _vtk.vtkXMLRectilinearGridReader,
                _vtk.vtkUnstructuredGrid: _vtk.vtkXMLUnstructuredGridReader,
                _vtk.vtkPolyData: _vtk.vtkXMLPolyDataReader,
                _vtk.vtkTable: _vtk.vtkXMLTableReader,
            }

            for parent_type, reader_type in readers.items():
                if isinstance(self, parent_type):
                    reader = reader_type()
                    break
            else:
                raise TypeError(f'Cannot unpickle dataset of type {self.GetDataObjectType()}')

            reader.ReadFromInputStringOn()
            reader.SetInputString(vtk_serialized)
            reader.Update()

        elif pickle_format.lower() == 'legacy':
            reader = _vtk.vtkDataSetReader()
            reader.ReadFromInputStringOn()
            if isinstance(vtk_serialized, bytes):
                reader.SetBinaryInputString(vtk_serialized, len(vtk_serialized))
            elif isinstance(vtk_serialized, str):
                reader.SetInputString(vtk_serialized)
            reader.Update()

        mesh = wrap(reader.GetOutput())

        # copy data
        self.copy_structure(mesh)
        self.copy_attributes(mesh)
