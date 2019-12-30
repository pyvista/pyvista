from vtk.numpy_interface.dataset_adapter import VTKObjectWrapper
import numpy
import pyvista.utilities.helpers as helpers
from .pyvista_ndarray import pyvista_ndarray
from vtk.numpy_interface.dataset_adapter import (VTKObjectWrapper, ArrayAssociation,
                                                 numpyTovtkDataArray)


class DataSetAttributes(VTKObjectWrapper):
    """Python friendly wrapper of vtk.DataSetAttributes.
    Implements a dict like interface for interacting with vtkDataArrays."""
    def __init__(self, vtkobject, dataset, association):
        super().__init__(vtkobject=vtkobject)
        self.dataset = dataset
        self.association = association

    def __getitem__(self, key):
        """Implements the [] operator. Accepts an array name or index."""
        return self.get_array(key)

    def __setitem__(self, key, value):
        self.append(narray=value, name=key)

    def __delitem__(self, key):
        self.remove(key)

    def __contains__(self, item):
        return item in self.keys()

    def __iter__(self):
        for array in self.keys():
            yield array

    def __len__(self):
        return self.VTKObject.GetNumberOfArrays()

    @property
    def valid_array_len(self):
        """Return the length which a numpy array should be when adding to this dataset.
        If there are no restrictions, returns None"""
        if self.association == ArrayAssociation.POINT:
            return self.dataset.GetNumberOfPoints()
        elif self.association == ArrayAssociation.CELL:
            return self.dataset.GetNumberOfCells()
        return None

    @property
    def t_coords(self):
        """Return the active texture coordinates on the points."""
        t_coords = self.GetTCoords()
        if t_coords is not None:
            return pyvista_ndarray.from_vtk_data_array(t_coords)

    @t_coords.setter
    def t_coords(self, t_coords):
        """Set the array to use as the texture coordinates."""
        if not isinstance(t_coords, numpy.ndarray):
            raise TypeError('Texture coordinates must be a numpy array')
        if t_coords.ndim != 2:
            raise AssertionError('Texture coordinates must be a 2-dimensional array')
        valid_length = self.valid_array_len
        if t_coords.shape[0] != valid_length:
            raise AssertionError('Number of texture coordinates ({}) must match number of points ({})'.format(t_coords.shape[0], valid_length))
        if t_coords.shape[1] != 2:
            raise AssertionError('Texture coordinates must only have 2 components, not ({})'.format(t_coords.shape[1]))
        # if np.min(t_coords) < 0.0 or np.max(t_coords) > 1.0:
        #     warnings.warn('Texture coordinates are typically within (0, 1) range. Textures will repeat on this mesh.', RuntimeWarning)
        vtkarr = numpyTovtkDataArray(t_coords, name='Texture Coordinates')
        self.SetTCoords(vtkarr)
        self.Modified()

    def get_array(self, key):
        """Given an index or name, returns a pyvista_ndarray."""
        self._raise_index_out_of_bounds(index=key)
        vtk_arr = self.VTKObject.GetArray(key)
        if not vtk_arr:
            return self.VTKObject.GetAbstractArray(key)
        narray = pyvista_ndarray.from_vtk_data_array(vtk_arr, dataset=self.dataset, association=self.association)
        if vtk_arr.GetName() in self.dataset.association_bitarray_names[self.association]:
            narray = narray.view(numpy.bool)
        return narray

    def append(self, narray, name, deep_copy=False):
        """Add an array to the data set attributes.

        Parameters
        ----------
        narray : array_like, scalar value
            A pyvista_ndarray, numpy.ndarray, list, tuple or scalar value.

        name : str
            Name of the array to add.

        deep_copy : bool
            When True makes a full copy of the array.

        """
        if narray is None:
            raise TypeError('narray cannot be None.')
        if isinstance(narray, (list, tuple)):
            narray = pyvista_ndarray.from_iter(narray)
        if narray.dtype == numpy.bool:
            self.dataset.association_bitarray_names[self.association].add(name)
            narray = narray.view(numpy.uint8)

        if self.association == ArrayAssociation.POINT:
            array_len = self.dataset.GetNumberOfPoints()
        elif self.association == ArrayAssociation.CELL:
            array_len = self.dataset.GetNumberOfCells()
        elif self.association == ArrayAssociation.ROW:
            array_len = narray.shape[0]
        else:
            array_len = narray.shape[0] if isinstance(narray, numpy.ndarray) else 1
        if narray.shape[0] != array_len:
            raise ValueError('narray length of ({}) != required length ({})'.format(
                narray.shape[0], array_len))

        # Fixup input array length:
        if not isinstance(narray, numpy.ndarray) or numpy.ndim(narray) == 0: # Scalar input
            tmparray = numpy.empty(array_len)
            tmparray.fill(narray)
            narray = tmparray
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
            narray = numpy.ascontiguousarray(narray)

        # Flatten array of matrices to array of vectors
        if len(shape) == 3:
            narray = narray.reshape(shape[0], shape[1]*shape[2])

        # this handle the case when an input array is directly appended on the
        # output. We want to make sure that the array added to the output is not
        # referring to the input dataset.
        try:
            copy = pyvista_ndarray(narray, vtk_array=narray.VTKObject)
        except AttributeError:
            copy = pyvista_ndarray(narray)

        vtk_arr = helpers.convert_array(copy, name, deep=deep_copy)
        self.VTKObject.AddArray(vtk_arr)
        self.VTKObject.Modified()

    def remove(self, key):
        self._raise_index_out_of_bounds(index=key)
        try:
            name = self.get_array(key).GetName()
            self.dataset.association_bitarray_names[self.association].remove(name)
        except KeyError:
            pass
        self.VTKObject.RemoveArray(key)
        self.VTKObject.Modified()

    def pop(self, key):
        self._raise_index_out_of_bounds(index=key)
        vtk_arr = self.GetArray(key)
        if vtk_arr:
            copy = vtk_arr.NewInstance()
            copy.DeepCopy(vtk_arr)
            vtk_arr = copy
        self.remove(key)
        return pyvista_ndarray.from_vtk_data_array(vtk_arr)

    def items(self):
        return list(zip(self.keys(), self.values()))

    def keys(self):
        """Returns the names of the arrays as a list."""
        kys = []
        narrays = self.VTKObject.GetNumberOfArrays()
        for i in range(narrays):
            name = self.VTKObject.GetAbstractArray(i).GetName()
            if name:
                kys.append(name)
        return kys

    def values(self):
        """Returns the arrays as a list."""
        vals = []
        narrays = self.VTKObject.GetNumberOfArrays()
        for i in range(narrays):
            a = self.VTKObject.GetAbstractArray(i)
            if a.GetName():
                vals.append(pyvista_ndarray.from_vtk_data_array(a))
        return vals

    def get_scalars(self, name=None):
        if name is not None:
            return self.get_array(key=name)
        if self.association == ArrayAssociation.FIELD:
            raise TypeError(
                'vtkFieldData does not have active scalars, a name must be provided. name={}'.format(name))
        active_scalar = self.GetScalars()
        return pyvista_ndarray.from_vtk_data_array(active_scalar, dataset=self.dataset)

    def clear(self):
        for array in self.values():
            self.remove(key=array.GetName())

    def update(self, array_dict):
        for name, array in array_dict.items():
            self[name] = array

    def _raise_index_out_of_bounds(self, index):
        max_index = self.VTKObject.GetNumberOfArrays()
        if isinstance(index, int) and index >= self.VTKObject.GetNumberOfArrays():
            raise IndexError('Array index ({}) out of range [0, {}]'.format(index, max_index))
