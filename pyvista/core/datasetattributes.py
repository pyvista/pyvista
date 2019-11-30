from vtk.numpy_interface.dataset_adapter import VTKObjectWrapper


class DataSetAttributes(VTKObjectWrapper):
    """Python friendly wrapper of vtk.DataSetAttributes.
    Implements a dict like interface for interacting with vtkDataArrays.
    Loosely based on dataset_adapter.DataSetAttributes."""
    def __init__(self, vtkobject, dataset, association):
        super().__init__(vtkobject=vtkobject)
        self._dataset = dataset
        self._association = association

    def __getitem__(self, key):
        """Implements the [] operator. Accepts an array name or index."""
        return self.get_array(key)

    def __setitem__(self, key, value):
        self.append(narray=value, name=key)

    def __delitem__(self, key):
        self.RemoveArray(key)

    def __contains__(self, item):
        return item in self.keys()

    def __iter__(self):
        for array in self.keys():
            yield array

    def __len__(self):
        return self.VTKObject.GetNumberOfArrays()

    def get_array(self, key):
        """Given an index or name, returns a VTKArray."""
        self._raise_index_out_of_bounds(index=key)
        vtkarray = self.VTKObject.GetArray(key)
        if not vtkarray:
            vtkarray = self.VTKObject.GetAbstractArray(key)
            return vtkarray if vtkarray else None
        array = pyvista_ndarray.from_vtk_data_array(vtkarray, dataset=self._dataset)
        array.Association = self.Association
        return array

    def append(self, narray, name):
        """Add/set an array to the data set attributes.

        Parameters
        ----------
        narray : array_like, scalar value
            A pyvista_ndarray, numpy.ndarray, list, tuple or scalar value.

        name : str
            Name of the array to add.

        """
        if narray is None:
            raise TypeError('narray cannot be None.')
        if isinstance(narray, (list, tuple, vtkDataArray)):
            narray = pyvista_ndarray.from_any(narray)

        if self.Association == ArrayAssociation.POINT:
            array_len = self.DataSet.GetNumberOfPoints()
        elif self.Association == ArrayAssociation.CELL:
            array_len = self.DataSet.GetNumberOfCells()
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
        copy = pyvista_ndarray(narray)
        try:
            copy.VTKObject = narray.VTKObject
        except AttributeError:
            pass
        arr = numpyTovtkDataArray(copy, name)
        self.VTKObject.AddArray(arr)

    def remove(self, key):
        self._raise_index_out_of_bounds(index=key)
        self.RemoveArray(key)

    def pop(self, key):
        self._raise_index_out_of_bounds(index=key)
        vtkarray = self.GetArray(key)
        if vtkarray:
            copy = vtkarray.NewInstance()
            copy.DeepCopy(vtkarray)
            vtkarray = copy
        self.VTKObject.RemoveArray(key)
        return vtkarray

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
        if self._association == ArrayAssociation.FIELD:
            raise TypeError(
                'vtkFieldData does not have active scalars, a name must be provided. name={}'.format(name))
        active_scalar = self.GetScalars()
        return pyvista_ndarray.from_vtk_data_array(active_scalar, dataset=self._dataset)

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

