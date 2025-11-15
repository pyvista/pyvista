"""Attributes common to PolyData and Grid Objects."""

from __future__ import annotations

from abc import abstractmethod
from collections import UserDict
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING
import warnings

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.typing.mypy_plugin import promote_type

from . import _vtk_core as _vtk
from .datasetattributes import DataSetAttributes
from .pyvista_ndarray import pyvista_ndarray
from .utilities.arrays import FieldAssociation
from .utilities.arrays import _JSONValueType
from .utilities.arrays import _SerializedDictArray
from .utilities.fileio import PICKLE_EXT
from .utilities.fileio import _CompressionOptions
from .utilities.fileio import get_ext
from .utilities.fileio import read
from .utilities.fileio import save_pickle
from .utilities.helpers import wrap
from .utilities.misc import _NoNewAttrMixin
from .utilities.misc import abstract_class

if TYPE_CHECKING:
    from types import FunctionType
    from typing import Any
    from typing import ClassVar

    from typing_extensions import Self

    from ._typing_core import NumpyArray
    from .utilities.writer import BaseWriter

# vector array names
DEFAULT_VECTOR_KEY = '_vectors'
USER_DICT_KEY = '_PYVISTA_USER_DICT'


@promote_type(_vtk.vtkDataObject)
@abstract_class
class DataObject(_NoNewAttrMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkPyVistaOverride):
    """Methods common to all wrapped data objects.

    Parameters
    ----------
    *args :
        Any extra args are passed as option to all wrapped data objects.

    **kwargs :
        Any extra keyword args are passed as option to all wrapped data objects.

    """

    _WRITERS: ClassVar[dict[str, type[BaseWriter]]] = {}

    def __init__(self: Self, *args, **kwargs) -> None:
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

    def __getattr__(self: Self, item: str) -> Any:
        """Get attribute from base class if not found."""
        return super().__getattribute__(item)

    def shallow_copy(self: Self, to_copy: Self | _vtk.vtkDataObject) -> None:
        """Shallow copy the given mesh to this mesh.

        Parameters
        ----------
        to_copy : DataObject | :vtk:`vtkDataObject`
            Data object to perform a shallow copy from.

        """
        self.ShallowCopy(to_copy)

    def deep_copy(self: Self, to_copy: Self | _vtk.vtkDataObject) -> None:
        """Overwrite this data object with another data object as a deep copy.

        Parameters
        ----------
        to_copy : DataObject | :vtk:`vtkDataObject`
            Data object to perform a deep copy from.

        """
        self.DeepCopy(to_copy)

    def _from_file(self: Self, filename: str | Path, **kwargs) -> None:
        """Read data objects from file."""
        data = read(filename, **kwargs)
        if not isinstance(self, type(data)):
            msg = (
                f'Reading file returned data of `{type(data).__name__}`, '
                f'but `{type(self).__name__}` was expected.'
            )
            raise TypeError(msg)
        self.shallow_copy(data)
        self._post_file_load_processing()

    def _post_file_load_processing(self: Self) -> None:
        """Execute after loading a dataset from file, to be optionally overridden by subclasses."""

    @_deprecate_positional_args(allowed=['filename'])
    def save(  # noqa: PLR0917
        self: Self,
        filename: Path | str,
        binary: bool = True,  # noqa: FBT001, FBT002
        texture: NumpyArray[np.uint8] | str | None = None,
        compression: _CompressionOptions = 'zlib',
    ) -> None:
        """Save this vtk object to file.

        .. versionadded:: 0.45

            Support saving pickled meshes

        See Also
        --------
        pyvista.read

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

        compression : str or None, default: 'zlib'
            The compression type to use when ``binary`` is ``True``
            and VTK writer is of type :vtk:`vtkXMLWriter`. This
            argument has no effect otherwise. Acceptable values are
            ``'zlib'``, ``'lz4'``, ``'lzma'``, and ``None``. ``None``
            indicates no compression.

            .. versionadded:: 0.47

        Notes
        -----
        Binary files write much faster than ASCII and have a smaller
        file size.

        """

        def _warn_multiblock_nested_field_data(mesh: pv.MultiBlock) -> None:
            iterator = mesh.recursive_iterator('all', node_type='parent')
            for index, name, nested_multiblock in iterator:
                if len(nested_multiblock.field_data.keys()) > 0:
                    # Avoid circular import
                    from pyvista.core.filters.composite import _format_nested_index

                    index_fmt = _format_nested_index(index)
                    warnings.warn(
                        f"Nested MultiBlock at index {index_fmt} with name '{name}' "
                        f'has field data which will not be saved.\n'
                        'See https://gitlab.kitware.com/vtk/vtk/-/issues/19414 \n'
                        'Use `move_nested_field_data_to_root` to store the field data '
                        'with the root MultiBlock before saving.',
                        stacklevel=2,
                    )

        def _check_multiblock_hdf_types(mesh: pv.MultiBlock) -> None:
            if (9, 4, 0) <= pv.vtk_version_info < (9, 5, 0):
                if mesh.is_nested:
                    msg = (
                        'Nested MultiBlocks are not supported by the .vtkhdf format in VTK 9.4.'
                        '\nUpgrade to VTK>=9.5 for this functionality.'
                    )
                    raise TypeError(msg)
                if type(None) in mesh.block_types:
                    msg = (
                        'Saving None blocks is not supported by the .vtkhdf format in VTK 9.4.'
                        '\nUpgrade to VTK>=9.5 for this functionality.'
                    )
                    raise TypeError(msg)

            supported_block_types: list[type] = [
                pv.PolyData,
                pv.UnstructuredGrid,
                type(None),
                pv.MultiBlock,
                pv.PartitionedDataSet,
            ]
            for id_, name, block in mesh.recursive_iterator('all'):
                if type(block) not in supported_block_types:
                    from pyvista.core.filters.composite import _format_nested_index

                    index_fmt = _format_nested_index(id_)
                    msg = (
                        f"Block at index {index_fmt} with name '{name}' has type "
                        f'{block.__class__.__name__!r} '
                        f'which cannot be saved to the .vtkhdf format.\n'
                        f'Supported types are: {[typ.__name__ for typ in supported_block_types]}.'
                    )
                    raise TypeError(msg)

        def _warn_imagedata_direction_matrix(mesh: pv.ImageData) -> None:
            if not np.allclose(mesh.direction_matrix, np.eye(3)):
                warnings.warn(
                    'The direction matrix for ImageData will not be saved using the '
                    'legacy `.vtk` format.\n'
                    'See https://gitlab.kitware.com/vtk/vtk/-/issues/19663 \n'
                    'Use the `.vti` extension instead (XML format).',
                    stacklevel=2,
                )

        def _write_vtk(mesh_: DataObject) -> None:
            if file_ext != '.vtk':
                writer = mesh_._WRITERS[file_ext](file_path, mesh_)
                data_mode = 'binary' if binary else 'ascii'
                writer._apply_kwargs_safely(
                    texture=texture, data_mode=data_mode, compression=compression
                )
                writer.write()
                return

            from vtkmodules.vtkIOLegacy import vtkDataSetWriter

            writer = vtkDataSetWriter()
            pv.set_vtkwriter_mode(vtk_writer=writer, use_binary=binary, compression=compression)
            writer.SetFileName(str(file_path))
            writer.SetInputData(mesh_)
            writer.Write()

        if self._WRITERS is None:
            msg = (  # type: ignore[unreachable]
                f'{self.__class__.__name__} writers are not specified,'
                ' this should be a dict of (file extension: vtkWriter type)'
            )
            raise NotImplementedError(msg)

        file_path = Path(filename)
        file_path = file_path.expanduser()
        file_path = file_path.resolve()
        file_ext = get_ext(file_path)

        if file_ext == '.vtkhdf' and binary is False:
            msg = '.vtkhdf files can only be written in binary format.'
            raise ValueError(msg)

        # store complex and bitarray types as field data
        self._store_metadata()

        # warn if data will be lost
        if isinstance(self, pv.MultiBlock):
            _warn_multiblock_nested_field_data(self)
            if file_ext == '.vtkhdf':
                _check_multiblock_hdf_types(self)
        if isinstance(self, pv.ImageData) and file_ext == '.vtk':
            _warn_imagedata_direction_matrix(self)

        writer_exts = self._WRITERS.keys()
        if file_ext in writer_exts:
            _write_vtk(self)
        elif file_ext in PICKLE_EXT:
            save_pickle(filename, self)
        else:
            msg = (
                f'Invalid file extension {file_ext!r} for data type {type(self)}.\n'
                f'Must be one of: {list(writer_exts) + list(PICKLE_EXT)}'
            )
            raise ValueError(msg)

    def _store_metadata(self: Self) -> None:
        """Store metadata as field data."""
        fdata = self.field_data
        for assoc_name in ('bitarray', 'complex'):
            for assoc_type in ('POINT', 'CELL'):
                assoc_data = getattr(self, f'_association_{assoc_name}_names')
                array_names = assoc_data.get(assoc_type)
                if array_names:
                    key = f'_PYVISTA_{assoc_name}_{assoc_type}_'.upper()
                    fdata[key] = list(array_names)

    def _restore_metadata(self: Self) -> None:
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
    def get_data_range(
        self: Self, name: str | None, preference: FieldAssociation | str
    ) -> tuple[float, float]:  # pragma: no cover
        """Get the non-NaN min and max of a named array."""
        msg = f'{type(self)} mesh type does not have a `get_data_range` method.'
        raise NotImplementedError(msg)

    def _get_attrs(self: Self) -> list[tuple[str, Any, str]]:  # pragma: no cover
        """Return the representation methods (internal helper)."""
        msg = 'Called only by the inherited class'
        raise NotImplementedError(msg)

    @_deprecate_positional_args
    def head(self: Self, display: bool = True, html: bool | None = None) -> str:  # noqa: FBT001, FBT002
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
            fmt = ''
            # HTML version
            fmt += '\n'
            fmt += "<table style='width: 100%;'>\n"
            fmt += f'<tr><th>{type(self).__name__}</th><th>Information</th></tr>\n'
            row = '<tr><td>{}</td><td>{}</td></tr>\n'
            # now make a call on the object to get its attributes as a list of len 2 tuples
            for attr in self._get_attrs():
                try:
                    fmt += row.format(attr[0], attr[2].format(*attr[1]))
                except TypeError:
                    fmt += row.format(attr[0], attr[2].format(attr[1]))
            if hasattr(self, 'n_arrays'):
                fmt += row.format('N Arrays', self.n_arrays)
            fmt += '</table>\n'
            fmt += '\n'
            if display:
                from IPython.display import HTML
                from IPython.display import display as _display

                _display(HTML(fmt))
                return ''
            return fmt
        # Otherwise return a string that is Python console friendly
        fmt = f'{type(self).__name__} ({hex(id(self))})\n'
        # now make a call on the object to get its attributes as a list of len 2 tuples
        # get longest row header
        max_len = max(len(attr[0]) for attr in self._get_attrs()) + 4

        # now make a call on the object to get its attributes as a list of len
        # 2 tuples
        row = f'  {{:{max_len}s}}' + '{}\n'
        for attr in self._get_attrs():
            try:
                fmt += row.format(attr[0] + ':', attr[2].format(*attr[1]))
            except TypeError:
                fmt += row.format(attr[0] + ':', attr[2].format(attr[1]))
        if hasattr(self, 'n_arrays'):
            fmt += row.format('N Arrays:', self.n_arrays)
        return fmt.strip()

    def _repr_html_(self: Self) -> str:  # pragma: no cover
        """Return a pretty representation for Jupyter notebooks.

        This includes header details and information about all arrays.

        """
        msg = 'Called only by the inherited class'
        raise NotImplementedError(msg)

    def copy_meta_from(self: Self, *args, **kwargs) -> None:  # pragma: no cover
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

    @_deprecate_positional_args
    def copy(self: Self, deep: bool = True) -> Self:  # noqa: FBT001, FBT002
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
        newobject.copy_meta_from(self, deep=deep)
        return newobject

    def __eq__(self: Self, other: object) -> bool:
        """Test equivalency between data objects."""
        if not isinstance(self, type(other)):
            return False

        if self is other:
            return True

        # these attrs use numpy.array_equal
        if isinstance(self, pv.ImageData):
            equal_attrs = ['extent', 'index_to_physical_matrix']
        else:
            equal_attrs = ['points', 'cells']
            if isinstance(self, pv.PolyData):
                equal_attrs.extend(['verts', 'lines', 'faces', 'strips'])
            elif isinstance(self, pv.UnstructuredGrid):
                equal_attrs.append('celltypes')
                equal_attrs.append('polyhedron_faces')
                equal_attrs.append('polyhedron_face_locations')

        for attr in equal_attrs:
            # Only check equality for attributes defined by PyVista
            # (i.e. ignore any default vtk snake_case attributes)
            if hasattr(self, attr) and not _vtk.is_vtk_attribute(self, attr):
                if not np.array_equal(getattr(self, attr), getattr(other, attr), equal_nan=True):
                    return False

        # these attrs can be directly compared
        attrs = ['field_data', 'point_data', 'cell_data']
        for attr in attrs:
            if hasattr(self, attr):
                if getattr(self, attr) != getattr(other, attr):
                    return False

        return True

    __hash__ = None  # type: ignore[assignment]  # https://github.com/pyvista/pyvista/pull/7671

    @_deprecate_positional_args(allowed=['array', 'name'])
    def add_field_data(self: Self, array: NumpyArray[float], name: str, deep: bool = True) -> None:  # noqa: FBT001, FBT002
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
        >>> mesh.add_field_data(['I could', 'write', 'notes', 'here'], 'my-field-data')
        >>> mesh['my-field-data']
        pyvista_ndarray(['I could', 'write', 'notes', 'here'], dtype='<U7')

        Add field data to a MultiBlock dataset.

        >>> blocks = pv.MultiBlock()
        >>> blocks.append(pv.Sphere())
        >>> blocks['cube'] = pv.Cube(center=(0, 0, -1))
        >>> blocks.add_field_data([1, 2, 3], 'my-field-data')
        >>> blocks.field_data['my-field-data']
        pyvista_ndarray([1, 2, 3])

        """
        if not hasattr(self, 'field_data'):
            msg = f'`{type(self)}` does not support field data'
            raise NotImplementedError(msg)

        self.field_data.set_array(array, name, deep_copy=deep)

    @property
    def field_data(self: Self) -> DataSetAttributes:
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
            dataset=self,  # type: ignore[arg-type]
            association=FieldAssociation.NONE,
        )

    def clear_field_data(self: Self) -> None:
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
            msg = f'`{type(self)}` does not support field data'
            raise NotImplementedError(msg)

        self.field_data.clear()

    @property
    def user_dict(self: Self) -> _SerializedDictArray:
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

        .. versionadded:: 0.44

        Returns
        -------
        UserDict
            JSON-serialized dict-like object which is subclassed from
            :py:class:`collections.UserDict`.

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
        self: Self,
        dict_: dict[str, _JSONValueType] | UserDict[str, _JSONValueType] | None,
    ) -> None:
        # Setting None removes the field data array
        if dict_ is None:
            if hasattr(self, '_user_dict'):
                del self._user_dict
            if USER_DICT_KEY in self.field_data.keys():
                del self.field_data[USER_DICT_KEY]
            return

        self._config_user_dict()
        if isinstance(dict_, dict):
            self._user_dict.data = dict_
        elif isinstance(dict_, UserDict):
            self._user_dict.data = dict_.data
        else:
            msg = (  # type: ignore[unreachable]
                f'User dict can only be set with type {dict} or {UserDict}.\n'
                f'Got {type(dict_)} instead.'
            )
            raise TypeError(msg)

    def _config_user_dict(self: Self) -> None:
        """Init serialized dict array and ensure it is added to field_data."""
        field_data = self.field_data

        if not hasattr(self, '_user_dict'):
            # Init
            object.__setattr__(self, '_user_dict', _SerializedDictArray())

        if USER_DICT_KEY in field_data.keys():
            if isinstance(array := field_data[USER_DICT_KEY], pyvista_ndarray):
                # When loaded from file, field will be cast as pyvista ndarray
                # Convert to string and initialize new user dict object from it
                self._user_dict = _SerializedDictArray(''.join(array))
            elif isinstance(array, str) and repr(self._user_dict) != array:  # type: ignore[unreachable]
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
        self._user_dict.SetName(USER_DICT_KEY)
        field_data.VTKObject.AddArray(self._user_dict)
        field_data.VTKObject.Modified()

    @property
    def memory_address(self: Self) -> str:
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
        return self.GetInformation().GetAddressAsString('')

    @property
    def actual_memory_size(self: Self) -> int:
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

    def copy_structure(self: Self, dataset: Self) -> None:
        """Copy the structure (geometry and topology) of the input dataset object.

        Parameters
        ----------
        dataset : :vtk:`vtkDataSet`
            Dataset to copy the geometry and topology from.

        Examples
        --------
        >>> import pyvista as pv
        >>> source = pv.ImageData(dimensions=(10, 10, 5))
        >>> target = pv.ImageData()
        >>> target.copy_structure(source)
        >>> target.plot(show_edges=True)

        """
        self.CopyStructure(dataset) if dataset is not self else None

    def copy_attributes(self: Self, dataset: Self) -> None:
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

    def __getstate__(  # type: ignore[return]  # noqa: RET503
        self: Self,
    ) -> tuple[FunctionType, tuple[dict[str, Any]]] | dict[str, Any]:
        """Support pickle."""
        pickle_format = pv.PICKLE_FORMAT
        if pickle_format == 'vtk':
            return self._serialize_vtk_pickle_format()
        elif pickle_format in ['xml', 'legacy']:
            return self._serialize_pyvista_pickle_format()
        # Invalid format, use the setter to raise an error
        pv.set_pickle_format(pickle_format)

    def _serialize_vtk_pickle_format(
        self: Self,
    ) -> tuple[FunctionType, tuple[dict[str, Any]]]:
        # Note: The serialized state has format: ( function, (dict,) )
        serialized = _vtk.serialize_VTK_data_object(self)

        # Add this object's data to the state dictionary
        state_dict = serialized[1][0]
        state_dict['_PYVISTA_STATE_DICT'] = self.__dict__.copy()

        # Unlike the PyVista formats, we do not return a dict. Instead, return
        # the same format returned by the vtk serializer.
        return serialized

    def _serialize_pyvista_pickle_format(self: Self) -> dict[str, Any]:
        """Support pickle by serializing the VTK object data.

        The format of the serialized VTK object data depends on `pyvista.PICKLE_FORMAT`
        (case-insensitive).
        - If ``'xml'``, the data is serialized as an XML-formatted string.
        - If ``'legacy'``, the data is serialized to bytes in VTK's binary format.

        .. note::

            These formats are custom PyVista legacy formats. The native 'vtk' format is
            preferred since it supports more objects (e.g. MultiBlock).

        """
        from vtkmodules.vtkIOLegacy import vtkDataSetWriter
        from vtkmodules.vtkIOXML import vtkXMLImageDataWriter
        from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
        from vtkmodules.vtkIOXML import vtkXMLRectilinearGridWriter
        from vtkmodules.vtkIOXML import vtkXMLStructuredGridWriter
        from vtkmodules.vtkIOXML import vtkXMLTableWriter
        from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

        if isinstance(self, pv.MultiBlock):
            msg = (
                "MultiBlock is not supported with 'xml' or 'legacy' pickle formats."
                "\nUse `pyvista.PICKLE_FORMAT='vtk'`."
            )
            raise TypeError(msg)
        state = self.__dict__.copy()

        if pv.PICKLE_FORMAT.lower() == 'xml':
            # the generic VTK XML writer `vtkXMLDataSetWriter` currently has a bug where it does
            # not pass all settings down to the sub-writers. Until this is fixed, use the
            # dataset-specific writers
            # https://gitlab.kitware.com/vtk/vtk/-/issues/18661
            writers = {
                _vtk.vtkImageData: vtkXMLImageDataWriter,
                _vtk.vtkStructuredGrid: vtkXMLStructuredGridWriter,
                _vtk.vtkRectilinearGrid: vtkXMLRectilinearGridWriter,
                _vtk.vtkUnstructuredGrid: vtkXMLUnstructuredGridWriter,
                _vtk.vtkPolyData: vtkXMLPolyDataWriter,
                _vtk.vtkTable: vtkXMLTableWriter,
            }

            for parent_type, writer_type in writers.items():
                if isinstance(self, parent_type):
                    writer = writer_type()  # type: ignore[unreachable]
                    break
            else:
                msg = f'Cannot pickle dataset of type {self.GetDataObjectType()}'
                raise TypeError(msg)

            writer.SetInputDataObject(self)  # type: ignore[unreachable]
            writer.SetWriteToOutputString(True)
            writer.SetDataModeToBinary()
            writer.SetCompressorTypeToNone()
            writer.Write()
            to_serialize = writer.GetOutputString()

        elif pv.PICKLE_FORMAT.lower() == 'legacy':
            writer = vtkDataSetWriter()
            writer.SetInputDataObject(self)
            writer.SetWriteToOutputString(True)
            writer.SetFileTypeToBinary()
            writer.Write()
            to_serialize = writer.GetOutputStdString()

        state['vtk_serialized'] = to_serialize

        # this needs to be here because in multiprocessing situations, `pyvista.PICKLE_FORMAT`
        # is not shared between processes
        state['PICKLE_FORMAT'] = pv.PICKLE_FORMAT
        return state

    def __setstate__(self: Self, state: Any) -> None:
        """Support unpickle."""

        def _is_vtk_format(state_: Any) -> bool:
            # Note: The vtk state has format ( function, (dict,) )
            return (
                isinstance(state_, tuple)
                and len(state_) == 2
                and isinstance(state_[1], tuple)
                and len(state_[1]) == 1
                and isinstance(state_[1][0], dict)
            )

        def _is_pyvista_format(state_: Any) -> bool:
            return isinstance(state_, dict) and 'vtk_serialized' in state_

        if _is_vtk_format(state):
            self._unserialize_vtk_pickle_format(state)
        elif _is_pyvista_format(state):
            self._unserialize_pyvista_pickle_format(state)
        else:
            msg = f"Cannot unpickle '{self.__class__.__name__}'. Invalid pickle format."
            raise RuntimeError(msg)

    def _unserialize_vtk_pickle_format(
        self: Self, state: tuple[FunctionType, tuple[dict[str, Any]]]
    ) -> None:
        """Support unpickle of VTK's format."""
        # The vtk state has format: ( function, (dict,) )
        unserialize_func = state[0]
        state_dict = state[1][0]
        self.__dict__.update(state_dict['_PYVISTA_STATE_DICT'])
        obj = unserialize_func(state_dict)
        self.deep_copy(obj)

    def _unserialize_pyvista_pickle_format(self: Self, state: dict[str, Any]) -> None:
        """Support unpickle of PyVista 'xml' and 'legacy' formats.

        .. note::

            These formats are custom PyVista legacy formats. The native 'vtk' format is
            preferred since it supports more objects (e.g. MultiBlock).

        """
        from vtkmodules.vtkIOLegacy import vtkDataSetReader
        from vtkmodules.vtkIOXML import vtkXMLImageDataReader
        from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
        from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
        from vtkmodules.vtkIOXML import vtkXMLStructuredGridReader
        from vtkmodules.vtkIOXML import vtkXMLTableReader
        from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

        vtk_serialized = state.pop('vtk_serialized')
        pickle_format = state.pop(
            'PICKLE_FORMAT',
            'legacy',  # backwards compatibility - assume 'legacy'
        )
        self.__dict__.update(state)

        if pickle_format.lower() == 'xml':
            # the generic VTK XML reader `vtkXMLGenericDataObjectReader` currently has a
            # bug where it does not pass all settings down to the sub-readers.
            # Until this is fixed, use the dataset-specific readers
            # https://gitlab.kitware.com/vtk/vtk/-/issues/18661
            readers = {
                _vtk.vtkImageData: vtkXMLImageDataReader,
                _vtk.vtkStructuredGrid: vtkXMLStructuredGridReader,
                _vtk.vtkRectilinearGrid: vtkXMLRectilinearGridReader,
                _vtk.vtkUnstructuredGrid: vtkXMLUnstructuredGridReader,
                _vtk.vtkPolyData: vtkXMLPolyDataReader,
                _vtk.vtkTable: vtkXMLTableReader,
            }

            for parent_type, reader_type in readers.items():
                if isinstance(self, parent_type):
                    reader = reader_type()  # type: ignore[unreachable]
                    break
            else:
                msg = f'Cannot unpickle dataset of type {self.GetDataObjectType()}'
                raise TypeError(msg)

            reader.ReadFromInputStringOn()  # type: ignore[unreachable]
            reader.SetInputString(vtk_serialized)
            reader.Update()

        elif pickle_format.lower() == 'legacy':
            reader = vtkDataSetReader()
            reader.ReadFromInputStringOn()
            if isinstance(vtk_serialized, bytes):
                reader.SetBinaryInputString(vtk_serialized, len(vtk_serialized))  # type: ignore[arg-type]
            elif isinstance(vtk_serialized, str):
                reader.SetInputString(vtk_serialized)
            reader.Update()

        mesh = wrap(reader.GetOutput())

        # copy data
        self.copy_structure(mesh)  # type: ignore[arg-type]
        self.copy_attributes(mesh)  # type: ignore[arg-type]

    @property
    @abstractmethod
    def is_empty(self) -> bool:
        """Return ``True`` if the object is empty."""
