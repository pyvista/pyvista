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
        keys = '\n\t'.join(self.keys())
        return 'pyvista DataSetAttributes\n' \
               f'Association    : {self.association.name}\n' \
               f'Active Scalars : {self.active_scalars_name}\n' \
               f'Active Vectors : {self.active_vectors_name}\n' \
               f'Contains keys:\n\t{keys}' \


    def get(self, key: str, value: Optional[Any] = None) -> Optional[pyvista_ndarray]:
        """Returns the value of the item with the specified key.

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
        self._append(narray=value, name=key)

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

    def set_vectors(self, vectors: Union[Sequence[Number], Number, np.ndarray],
                    name: str, deep_copy=False):
        """Set the vectors of this data attribute.

        """
        self._append(vectors, name, deep_copy=deep_copy, vectors_only=True)

    def set_scalars(self, scalars: Union[Sequence[Number], Number, np.ndarray],
                    name: str, deep_copy=False):
        self._append(scalars, name, deep_copy=deep_copy, active_vectors=False)

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

    def get_array(self, key: Union[str, int]) -> Union[pyvista_ndarray, _vtk.vtkDataArray, _vtk.vtkAbstractArray]:
        """Get an array in this object.

        Parameters
        ----------
        key : str, index
            The name or index of the array to return.

        Returns
        ----------
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

    def append(self, narray: Union[Sequence[Number], Number, np.ndarray],
               name: str, deep_copy=False, active_vectors=True,
               active_scalars=True, vectors_only=False) -> None:
        """Add an array to this object.

        .. deprecated:: 0.32.0
           Use one of the following instead:

           * :func:`DataSetAttributes.add_array`
           * :func:`DataSetAttributes.set_scalars`
           * :func:`DataSetAttributes.set_vectors`
           * The ``[]`` operator

        """
        warnings.warn("\n\n`DataSetAttributes.append` is deprecated.\n\n"
                      "Use one of the following instead:\n"
                      "    - `DataSetAttributes.add_array`\n"
                      "    - `DataSetAttributes.set_scalars`\n"
                      "    - `DataSetAttributes.set_vectors`\n"
                      "    - The [] operator",
            PyvistaDeprecationWarning
        )
        self._append(narray, name, deep_copy, active_vectors,
                     active_scalars, vectors_only)

    def _append(self, narray: Union[Sequence[Number], Number, np.ndarray],
               name: str, deep_copy=False, active_vectors=True,
               active_scalars=True, vectors_only=False) -> None:
        """Add an array to this object.

        Parameters
        ----------
        narray : array_like, scalar value
            A ``pyvista_ndarray``, ``numpy.ndarray``, ``list``,
            ``tuple`` or scalar value.

        name : str
            Name of the array to add.

        deep_copy : bool, optional
            When ``True`` makes a full copy of the array.

        active_vectors : bool, optional
            If ``True``, make this the active vector array.

        active_scalars : bool, optional
            If ``True``, make this the active scalar array.

        vectors_only : bool, optional
            If ``True``, does not add array as a scalars array, only
            as a vectors array.

        """
        if narray is None:
            raise TypeError('narray cannot be None.')
        if isinstance(narray, Iterable):
            narray = pyvista_ndarray(narray)

        if vectors_only:
            active_scalars = False

        if self.association == FieldAssociation.POINT:
            array_len = self.dataset.GetNumberOfPoints()
        elif self.association == FieldAssociation.CELL:
            array_len = self.dataset.GetNumberOfCells()
        else:
            array_len = narray.shape[0] if isinstance(narray, np.ndarray) else 1

        # Fixup input array length for scalar input:
        if not isinstance(narray, np.ndarray) or np.ndim(narray) == 0:
            tmparray = np.empty(array_len)
            tmparray.fill(narray)
            narray = tmparray
        if narray.shape[0] != array_len:
            raise ValueError(f'narray length of ({narray.shape[0]}) != required length ({array_len})')

        if narray.dtype == np.bool_:
            self.dataset.association_bitarray_names[self.association].add(name)
            narray = narray.view(np.uint8)

        shape = narray.shape
        if len(shape) == 3:
            # Array of matrices. We need to make sure the order  in memory is right.
            # If column order (c order), transpose. VTK wants row order (fortran
            # order). The deep copy later will make sure that the array is contiguous.
            # If row order but not contiguous, transpose so that the deep copy below
            # does not happen.
            size = narray.dtype.itemsize
            if (narray.strides[1] / size == 3 and narray.strides[2] / size == 1) or \
                (narray.strides[1] / size == 1 and narray.strides[2] / size == 3 and \
                 not narray.flags.contiguous):
                narray = narray.transpose(0, 2, 1)

        # If array is not contiguous, make a deep copy that is contiguous
        if not narray.flags.contiguous:
            narray = np.ascontiguousarray(narray)

        # Flatten array of matrices to array of vectors
        if len(shape) == 3:
            narray = narray.reshape(shape[0], shape[1]*shape[2])

        # Swap bytes from big to little endian.
        if narray.dtype.byteorder == '>':
            narray = narray.byteswap(inplace=True)

        # this handles the case when an input array is directly appended on the
        # output. We want to make sure that the array added to the output is not
        # referring to the input dataset.
        copy = pyvista_ndarray(narray)

        vtk_arr = helpers.convert_array(copy, name, deep=deep_copy)
        if not vectors_only:
            self.VTKObject.AddArray(vtk_arr)

        try:
            if active_scalars or self.active_scalars is None:
                self.active_scalars = name  # type: ignore
            if active_vectors or self.active_vectors is None:
                # verify this is actually vector data
                if len(shape) == 2 and shape[1] == 3:
                    self.active_vectors = name  # type: ignore
                    self.VTKObject.SetVectors(vtk_arr)
        except TypeError:
            pass
        self.VTKObject.Modified()

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
        >>> mesh.point_arrays['data0'] = [0]*mesh.n_points
        >>> mesh.point_arrays['data1'] = range(mesh.n_points)
        >>> mesh.point_arrays.items()
        [('data0', pyvista_ndarray([0, 0, 0, 0, 0, 0, 0, 0])),
         ('data1', pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7]))]

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
        >>> mesh.point_arrays['data0'] = [0]*mesh.n_points
        >>> mesh.point_arrays['data1'] = range(mesh.n_points)
        >>> mesh.point_arrays.values()
        [pyvista_ndarray([0, 0, 0, 0, 0, 0, 0, 0]),
         pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])]

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
