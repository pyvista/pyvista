"""Implements DataSetAttributes, which represents and manipulates datasets."""

import warnings
from collections.abc import Iterable

import numpy as np
from typing import Union, Iterator, Optional, List, Tuple, Dict, Sequence, Any

from pyvista import _vtk
import pyvista.utilities.helpers as helpers
from pyvista.utilities.helpers import FieldAssociation
from pyvista.utilities.misc import PyvistaDeprecationWarning
from .pyvista_ndarray import pyvista_ndarray

from .._typing import Number


class DataSetAttributes(_vtk.VTKObjectWrapper):
    """Python friendly wrapper of ``vtk.DataSetAttributes``.

    Implement a ``dict`` like interface for interacting with vtkDataArrays.

    Parameters
    ----------
    vtkobject : vtkFieldData
        The vtk object to wrap as a DataSetAttribute, usually an
        instance of ``vtk.vtkCellData``, ``vtk.vtkPointData``, or
        ``vtk.vtkFieldData``.

    dataset : vtkDataSet
        The vtkDataSet containing the vtkobject.

    association : FieldAssociation
        The array association type of the vtkobject.

    Examples
    --------
    Store data with point association in a DataSet

    >>> import pyvista
    >>> mesh = pyvista.Cube().clean()
    >>> mesh.point_arrays['my_data'] = range(mesh.n_points)
    >>> data = mesh.point_arrays['my_data']
    >>> data
    pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

    Change the data array and show that this is reflected in the DataSet.

    >>> data[:] = 0
    >>> mesh.point_arrays['my_data']
    pyvista_ndarray([0, 0, 0, 0, 0, 0, 0, 0])

    """

    def __init__(self, vtkobject: _vtk.vtkFieldData, dataset: _vtk.vtkDataSet, association: FieldAssociation):
        """Initialize DataSetAttributes."""
        super().__init__(vtkobject=vtkobject)
        self.dataset = dataset
        self.association = association

    def __repr__(self) -> str:
        """Printable representation of DataSetAttributes."""
        array_info = ' None'
        if len(self):
            lines = []
            for name, array in self.items():
                if len(name) > 20:
                    name = f'{name[:20]}...'
                lines.append(f'{name[:23]:<23} {str(array.dtype):<10} {array.shape}')
            array_info = '\n    ' + '\n    '.join(lines)

        return 'pyvista DataSetAttributes\n' \
               f'Association     : {self.association.name}\n' \
               f'Active Scalars  : {self.active_scalars_name}\n' \
               f'Active Vectors  : {self.active_vectors_name}\n' \
               f'Active Texture  : {self.active_texture_name}\n' \
               f'Contains arrays :{array_info}' \

    def get(self, key: str, value: Optional[Any] = None) -> Optional[pyvista_ndarray]:
        """Return the value of the item with the specified key.

        Parameters
        ----------
        key : str
            Name of the array item you want to return the value from.

        value : anything, optional
            A value to return if the key does not exist.  Default
            is ``None``.

        Returns
        -------
        Any
            Array if the ``key`` exists in the dataset, otherwise
            ``value``.

        Examples
        --------
        Show that the default return value for a non-existent key is
        ``None``.

        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.point_arrays['my_data'] = range(mesh.n_points)
        >>> mesh.point_arrays.get('my-other-data')

        """
        if key in self:
            return self[key]
        return value

    def __getitem__(self, key: Union[int, str]) -> pyvista_ndarray:
        """Implement [] operator.

        Accepts an array name or an index.
        """
        return self.get_array(key)

    def __setitem__(self, key: str, value: np.ndarray):
        """Implement setting with the [] operator."""
        self.set_array(value, name=key)

    def __delitem__(self, key: Union[str, int]):
        """Implement del with array name or index."""
        self.remove(key)

    def __contains__(self, name: str) -> bool:
        """Implement 'in' operator."""
        return name in self.keys()

    def __iter__(self) -> Iterator[str]:
        """Implement for loop iteration."""
        for array in self.keys():
            yield array

    def __len__(self) -> int:
        """Return the number of arrays."""
        return self.VTKObject.GetNumberOfArrays()

    @property
    def active_scalars(self) -> Optional[pyvista_ndarray]:
        """Return the active scalar array as pyvista_ndarray.

        Examples
        --------
        Associate point data to a simple cube mesh and show that the
        active scalars in the point array are the most recently added
        array.

        >>> import pyvista
        >>> import numpy as np
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.point_arrays['data0'] = np.zeros(mesh.n_points)
        >>> mesh.point_arrays['data1'] = np.arange(mesh.n_points)
        >>> mesh.point_arrays.active_scalars
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        Set the active scalars of the array to a different array using
        the key of the point array.

        >>> mesh.point_arrays.active_scalars = 'data0'
        >>> mesh.point_arrays.active_scalars
        pyvista_ndarray([0., 0., 0., 0., 0., 0., 0., 0.])

        """
        self._raise_field_data_no_scalars_vectors()
        if self.GetScalars() is not None:
            return pyvista_ndarray(self.GetScalars(), dataset=self.dataset,
                                   association=self.association)
        return None

    @active_scalars.setter
    def active_scalars(self, name: str) -> None:
        self._raise_field_data_no_scalars_vectors()
        dtype = self[name].dtype
        if np.issubdtype(dtype, np.number) or dtype == bool:
            self.SetActiveScalars(name)

    @property
    def active_vectors(self) -> Optional[np.ndarray]:
        """Return the active vectors as a pyvista_ndarray."""
        self._raise_field_data_no_scalars_vectors()
        vectors = self.GetVectors()
        if vectors is not None:
            return pyvista_ndarray(vectors, dataset=self.dataset,
                                   association=self.association)
        return None

    @active_vectors.setter
    def active_vectors(self, name: str):
        """Set the active vectors by name or array."""
        self._raise_field_data_no_scalars_vectors()
        self.SetActiveVectors(name)

    @property
    def valid_array_len(self) -> int:
        """Return the length an ndarray should be when added to the dataset.

        If there are no restrictions, return ``None``
        """
        if self.association == FieldAssociation.POINT:
            return self.dataset.GetNumberOfPoints()
        if self.association == FieldAssociation.CELL:
            return self.dataset.GetNumberOfCells()
        return 0

    @property
    def t_coords(self) -> Optional[pyvista_ndarray]:
        """Return the active texture coordinates."""
        t_coords = self.GetTCoords()
        if t_coords is not None:
            return pyvista_ndarray(t_coords, dataset=self.dataset, association=self.association)
        return None

    @t_coords.setter
    def t_coords(self, t_coords: np.ndarray):
        """Set the active texture coordinates using an np.ndarray."""
        if not isinstance(t_coords, np.ndarray):
            raise TypeError('Texture coordinates must be a numpy array')
        if t_coords.ndim != 2:
            raise ValueError('Texture coordinates must be a 2-dimensional array')
        valid_length = self.valid_array_len
        if t_coords.shape[0] != valid_length:
            raise ValueError(f'Number of texture coordinates ({t_coords.shape[0]}) must match number of points ({valid_length})')
        if t_coords.shape[1] != 2:
            raise ValueError('Texture coordinates must only have 2 components,'
                             f' not ({t_coords.shape[1]})')
        vtkarr = _vtk.numpyTovtkDataArray(t_coords, name='Texture Coordinates')
        self.SetTCoords(vtkarr)
        self.Modified()

    @property
    def active_texture_name(self) -> Optional[str]:
        """Name of the active texture array."""
        try:
            return self.GetTCoords().GetName()
        except:
            return None

    def get_array(self, key: Union[str, int]) -> Union[pyvista_ndarray, _vtk.vtkDataArray, _vtk.vtkAbstractArray]:
        """Get an array in this object.

        Parameters
        ----------
        key : str, index
            The name or index of the array to return.

        Returns
        -------
        array : ``pyvista_ndarray`` or ``vtkDataArray``
            A ``pyvista_ndarray`` if the underlying array is a
            ``vtk.vtkDataArray`` or ``vtk.vtkStringArray``,
            ``vtk.vtkAbstractArray`` if the former does not exist.
            Raises ``KeyError`` if neither exist.
        """
        self._raise_index_out_of_bounds(index=key)
        vtk_arr = self.GetArray(key)
        if vtk_arr is None:
            vtk_arr = self.GetAbstractArray(key)
            if vtk_arr is None:
                raise KeyError(f'{key}')
            if type(vtk_arr) == _vtk.vtkAbstractArray:
                return vtk_arr
        narray = pyvista_ndarray(vtk_arr, dataset=self.dataset, association=self.association)
        if vtk_arr.GetName() in self.dataset.association_bitarray_names[self.association]:
            narray = narray.view(np.bool_)
        return narray

    def set_array(self, data: Union[Sequence[Number], Number, np.ndarray],
                  name: str, deep_copy=False, active_vectors=True,
                  active_scalars=True) -> None:
        """Add an array to this object.

        Parameters
        ----------
        data : sequence
            A ``pyvista_ndarray``, ``numpy.ndarray``, ``list``,
            ``tuple`` or scalar value.

        name : str
            Name of the array to add.

        deep_copy : bool, optional
            When ``True`` makes a full copy of the array.

        active_vectors : bool, optional
            If ``True``, also make this the active vector array.

        active_scalars : bool, optional
            If ``True``, also make this the active scalar array.

        Examples
        --------
        Add a point array to a mesh without making it the active
        scalars.

        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> data = range(mesh.n_points)
        >>> mesh.point_arrays.set_array(data, 'my-data', active_scalars=False)
        >>> mesh.point_arrays['my-data']
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        Add a cell array to a mesh without making it the active
        scalars.

        >>> cell_data = range(mesh.n_cells)
        >>> mesh.cell_arrays.set_array(cell_data, 'my-data',
        ...                            active_scalars=False)
        >>> mesh.cell_arrays['my-data']
        pyvista_ndarray([0, 1, 2, 3, 4, 5])

        Add a field array to a mesh.

        >>> field_data = range(3)
        >>> mesh.field_arrays.set_array(field_data, 'my-data')
        >>> mesh.field_arrays['my-data']
        pyvista_ndarray([0, 1, 2])

        Notes
        -----
        You can simply use the ``[]`` operator to add an array to the
        point, cell, or field data.  Note that by default if this is
        not field data, the array will be made the active scalars, and
        if a ``(N x 3)`` shaped array, then active vector data as
        well.

        """
        vtk_arr, shape = self._prepare_array(data, name, deep_copy)
        self.VTKObject.AddArray(vtk_arr)

        try:
            if active_scalars:
                self.active_scalars = name  # type: ignore
            if active_vectors:
                # verify this is vector data
                if len(shape) == 2 and shape[1] == 3:
                    self.active_vectors = name  # type: ignore
                    self.VTKObject.SetVectors(vtk_arr)
        except TypeError:
            pass
        self.VTKObject.Modified()

    def set_scalars(self, data: Union[Sequence[Number], Number, np.ndarray],
                    name='scalars', deep_copy=False):
        """Set the active scalars of the dataset with an array.

        Parameters
        ----------
        data : sequence
            A ``pyvista_ndarray``, ``numpy.ndarray``, ``list``,
            ``tuple`` or scalar value.

        name : str
            Name of the array to add.

        deep_copy : bool, optional
            When ``True`` makes a full copy of the array.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.clear_arrays()
        >>> scalars = range(mesh.n_points)
        >>> mesh.point_arrays.set_scalars(scalars, 'my-scalars')
        >>> mesh.point_arrays
        pyvista DataSetAttributes
        Association     : POINT
        Active Scalars  : my-scalars
        Active Vectors  : None
        Active Texture  : None
        Contains arrays :
            my-scalars              int64      (8,)

        """
        vtk_arr, shape = self._prepare_array(data, name, deep_copy)
        self.VTKObject.SetScalars(vtk_arr)
        self.VTKObject.Modified()

    def set_vectors(self, vectors: Union[Sequence[Number], Number, np.ndarray],
                    name: str, deep_copy=False):
        """Set the vectors of this data attribute.

        Parameters
        ----------
        data : sequence
            A ``pyvista_ndarray``, ``numpy.ndarray``, ``list``,
            ``tuple`` or scalar value.

        name : str
            Name of the array to add.

        deep_copy : bool, optional
            When ``True`` makes a full copy of the array.

        Examples
        --------
        Add random vectors to a mesh.

        >>> import pyvista
        >>> import numpy as np
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.clear_arrays()
        >>> vectors = np.random.random((mesh.n_points, 3))
        >>> mesh.point_arrays.set_vectors(vectors, 'my-vectors')
        >>> mesh.point_arrays
        pyvista DataSetAttributes
        Association     : POINT
        Active Scalars  : None
        Active Vectors  : my-vectors
        Active Texture  : None
        Contains arrays :
            my-vectors              float64    (8, 3)

        """
        vtk_arr, shape = self._prepare_array(vectors, name, deep_copy)
        self.VTKObject.SetVectors(vtk_arr)
        self.VTKObject.Modified()

    def _prepare_array(self, data: Union[Sequence[Number], Number, np.ndarray],
                       name: str, deep_copy: bool) -> Tuple[_vtk.vtkDataSet, Tuple]:
        """Prepare an array to be added to this dataset."""
        if data is None:
            raise TypeError('``data`` cannot be None.')
        if isinstance(data, Iterable):
            data = pyvista_ndarray(data)

        if self.association == FieldAssociation.POINT:
            array_len = self.dataset.GetNumberOfPoints()
        elif self.association == FieldAssociation.CELL:
            array_len = self.dataset.GetNumberOfCells()
        else:
            array_len = data.shape[0] if isinstance(data, np.ndarray) else 1

        # Fixup input array length for scalar input
        if not isinstance(data, np.ndarray) or np.ndim(data) == 0:
            tmparray = np.empty(array_len)
            tmparray.fill(data)
            data = tmparray
        if data.shape[0] != array_len:
            raise ValueError(f'data length of ({data.shape[0]}) != required length ({array_len})')

        if data.dtype == np.bool_:
            self.dataset.association_bitarray_names[self.association].add(name)
            data = data.view(np.uint8)

        shape = data.shape
        if len(shape) == 3:
            # Array of matrices. We need to make sure the order  in memory is right.
            # If column order (c order), transpose. VTK wants row order (fortran
            # order). The deep copy later will make sure that the array is contiguous.
            # If row order but not contiguous, transpose so that the deep copy below
            # does not happen.
            size = data.dtype.itemsize
            if (data.strides[1] / size == 3 and data.strides[2] / size == 1) or \
                (data.strides[1] / size == 1 and data.strides[2] / size == 3 and \
                 not data.flags.contiguous):
                data = data.transpose(0, 2, 1)

        # If array is not contiguous, make a deep copy that is contiguous
        if not data.flags.contiguous:
            data = np.ascontiguousarray(data)

        # Flatten array of matrices to array of vectors
        if len(shape) == 3:
            data = data.reshape(shape[0], shape[1]*shape[2])

        # Swap bytes from big to little endian.
        if data.dtype.byteorder == '>':
            data = data.byteswap(inplace=True)

        # this handles the case when an input array is directly added to the
        # output. We want to make sure that the array added to the output is not
        # referring to the input dataset.
        copy = pyvista_ndarray(data)

        vtk_arr = helpers.convert_array(copy, name, deep=deep_copy)
        return vtk_arr, shape

    def append(self, narray: Union[Sequence[Number], Number, np.ndarray],
               name: str, deep_copy=False, active_vectors=True,
               active_scalars=True) -> None:
        """Add an array to this object.

        .. deprecated:: 0.32.0
           Use one of the following instead:

           * :func:`DataSetAttributes.set_array`
           * :func:`DataSetAttributes.set_scalars`
           * :func:`DataSetAttributes.set_vectors`
           * The ``[]`` operator

        """
        1/0
        warnings.warn("\n\n`DataSetAttributes.append` is deprecated.\n\n"
                      "Use one of the following instead:\n"
                      "    - `DataSetAttributes.set_array`\n"
                      "    - `DataSetAttributes.set_scalars`\n"
                      "    - `DataSetAttributes.set_vectors`\n"
                      "    - The [] operator",
            PyvistaDeprecationWarning
        )
        self.set_array(narray, name, deep_copy, active_vectors,
                       active_scalars)

    def remove(self, key: Union[int, str]) -> None:
        """Remove an array.

        Parameters
        ----------
        key : int, str
            The name or index of the array to remove.
        """
        self._raise_index_out_of_bounds(index=key)
        name = self.get_array(key).GetName()  # type: ignore
        try:
            self.dataset.association_bitarray_names[self.association].remove(name)
        except KeyError:
            pass
        self.VTKObject.RemoveArray(key)
        self.VTKObject.Modified()

    def pop(self, key: Union[int, str], default=pyvista_ndarray(array=[])) -> pyvista_ndarray:
        """Remove an array and return it.

        Parameters
        ----------
        key : int, str
            The name or index of the array to remove and return.

        default : anything, optional
            If default is not given and key is not in the dictionary,
            a KeyError is raised.

        Returns
        -------
        pyvista_ndarray
            Requested array.

        Examples
        --------
        Add a point data array to a DataSet and then remove it

        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.point_arrays['my_data'] = range(mesh.n_points)
        >>> mesh.point_arrays.pop('my_data')
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        Show that the array no longer exists in ``point_arrays``

        >>> 'my_data' in mesh.point_arrays
        False

        """
        self._raise_index_out_of_bounds(index=key)
        vtk_arr = self.GetArray(key)
        if vtk_arr:
            copy = vtk_arr.NewInstance()
            copy.DeepCopy(vtk_arr)
            vtk_arr = copy
        try:
            self.remove(key)
        except KeyError:
            if default in self.pop.__defaults__:  # type: ignore
                raise
            return default
        return pyvista_ndarray(vtk_arr, dataset=self.dataset, association=self.association)

    def items(self) -> List[Tuple[str, pyvista_ndarray]]:
        """Return a list of (array name, array value).

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.clear_arrays()
        >>> mesh.cell_arrays['data0'] = [0]*mesh.n_cells
        >>> mesh.cell_arrays['data1'] = range(mesh.n_cells)
        >>> mesh.cell_arrays.items()
        [('data0', pyvista_ndarray([0, 0, 0, 0, 0, 0])), ('data1', pyvista_ndarray([0, 1, 2, 3, 4, 5]))]

        """
        return list(zip(self.keys(), self.values()))

    def keys(self) -> List[str]:
        """Return the names of the arrays as a list.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.clear_arrays()
        >>> mesh.point_arrays['data0'] = [0]*mesh.n_points
        >>> mesh.point_arrays['data1'] = range(mesh.n_points)
        >>> mesh.point_arrays.keys()
        ['data0', 'data1']

        """
        keys = []
        for i in range(self.GetNumberOfArrays()):
            name = self.VTKObject.GetAbstractArray(i).GetName()
            if name:
                keys.append(name)
        return keys

    def values(self) -> List[pyvista_ndarray]:
        """Return the arrays as a list.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.clear_arrays()
        >>> mesh.cell_arrays['data0'] = [0]*mesh.n_cells
        >>> mesh.cell_arrays['data1'] = range(mesh.n_cells)
        >>> mesh.cell_arrays.values()
        [pyvista_ndarray([0, 0, 0, 0, 0, 0]), pyvista_ndarray([0, 1, 2, 3, 4, 5])]

        """
        values = []
        for name in self.keys():
            array = self.VTKObject.GetAbstractArray(name)
            arr = pyvista_ndarray(array, dataset=self.dataset, association=self.association)
            values.append(arr)
        return values

    def clear(self):
        """Remove all arrays in this object.

        Examples
        --------
        Add point data to a DataSet and then clear the point_arrays.

        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.clear_arrays()
        >>> mesh.point_arrays['my_data'] = range(mesh.n_points)
        >>> len(mesh.point_arrays)
        1
        >>> mesh.point_arrays.clear()
        >>> len(mesh.point_arrays)
        0

        """
        for array_name in self.keys():
            self.remove(key=array_name)

    def update(self, array_dict: Union[Dict[str, np.ndarray], 'DataSetAttributes']):
        """Update arrays in this object.

        For each key, value given, add the pair, if it already exists,
        update it.

        Parameters
        ----------
        array_dict : dict
            A dictionary of (array name, numpy.ndarray)
        """
        for name, array in array_dict.items():
            self[name] = array.copy()

    def _raise_index_out_of_bounds(self, index: Any):
        max_index = self.VTKObject.GetNumberOfArrays()
        if isinstance(index, int):
            if index < 0 or index >= self.VTKObject.GetNumberOfArrays():
                raise KeyError(f'Array index ({index}) out of range [0, {max_index}]')

    def _raise_field_data_no_scalars_vectors(self):
        if self.association == FieldAssociation.NONE:
            raise TypeError('vtkFieldData does not have active scalars or vectors.')

    @property
    def active_scalars_name(self) -> Optional[str]:
        """Name of the active scalars.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.point_arrays['my_data'] = range(mesh.n_points)
        >>> mesh.point_arrays.active_scalars_name
        'my_data'

        """
        try:
            return self.GetScalars().GetName()
        except:
            return None

    @property
    def active_vectors_name(self) -> Optional[str]:
        """Name of the active scalars.

        Examples
        --------
        >>> import pyvista
        >>> import numpy as np
        >>> mesh = pyvista.Sphere()
        >>> mesh.point_arrays['my_data'] = np.random.random((mesh.n_points, 3))
        >>> mesh.point_arrays.active_scalars_name
        'my_data'

        """
        try:
            return self.GetVectors().GetName()
        except:
            return None

    def __eq__(self, other: Any) -> bool:
        """Test dict-like equivalency."""
        # here we check if other is the same class or a subclass of self.
        if not isinstance(other, type(self)):
            return False

        if set(self.keys()) != set(other.keys()):
            return False

        for key, value in other.items():
            if not np.array_equal(value, self[key]):
                return False

        if self.association != FieldAssociation.NONE:
            if not other.active_scalars_name == self.active_scalars_name:
                return False
            if not other.active_vectors_name == self.active_vectors_name:
                return False

        return True
