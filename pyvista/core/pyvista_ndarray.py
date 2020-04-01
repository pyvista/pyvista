"""Contains pyvista_ndarray a numpy ndarray type used in pyvista."""

import numpy as np
from pyvista.utilities.helpers import convert_array, FieldAssociation
from vtk.numpy_interface.dataset_adapter import VTKObjectWrapper, VTKArray
from vtk.vtkCommonCore import vtkWeakReference
from vtk.vtkCommonKitPython import vtkDataArray, vtkAbstractArray


class pyvista_ndarray(VTKArray):
    """
    A numpy.ndarray which can reference a vtk array and it's dataset.

    If a vtk array is given, this array points to the same memory
    location as the given array.
    """

    def __new__(cls, ndarray, vtk_array=None, dataset=None, association=FieldAssociation.NONE):
        """Allocate the array."""
        obj = np.asarray(ndarray).view(cls)
        obj.association = association
        # add the new attributes to the created instance
        obj.VTKObject = vtk_array
        if dataset:
            obj.dataset = vtkWeakReference()
            if isinstance(dataset, VTKObjectWrapper):
                obj.dataset.Set(dataset.VTKObject)
            else:
                obj.dataset.Set(dataset)
        return obj

    def __setitem__(self, key, value):
        """Set item at key index to value."""
        super().__setitem__(key, value)
        if self.VTKObject is not None:
            self.VTKObject.Modified()

    @classmethod
    def from_any(cls, obj, dtype=None, order=None, dataset=None, association=None):
        """Create a `pyvista_ndarray`` instance from an object.

        Parameters
        ----------
        obj : any
            Object to create numpy array from based on type.

        dataset : vtkDataSet, optional
            The vtkDataSet which vtk_data_array belongs to. Required to
            update or access a dataset when this pyvista_ndarray is updated.

        association : FieldAssociation, optional
            The FieldAssociation of the dataset the input array belongs to.

        dtype : data-type, optional
            By default, the data-type is inferred from the input data.

        order : {‘C’, ‘F’}, optional
            Whether to use row-major (C-style) or column-major
            (Fortran-style) memory representation. Defaults to ‘C’.

        """
        if isinstance(obj, (vtkDataArray, vtkAbstractArray)):
            return cls.from_vtk_data_array(vtk_data_array=obj, dataset=dataset, association=association)
        elif isinstance(obj, (list, tuple, np.ndarray)):
            return cls.from_iter(iterable=obj, dtype=dtype, order=order)
        else:
            raise ValueError('Cannot create {} from type: {}'.format(
                type(cls).__name__, type(obj).__name__))

    @classmethod
    def from_vtk_data_array(cls, vtk_data_array, dataset=None, association=None):
        """Create a ``pyvista_ndarray`` instance from a vtk array.

        Parameters
        ----------
        vtk_data_array : vtkDataArray
            Array to copy.

        dataset : vtkDataSet, optional
            The vtkDataSet which vtk_data_array belongs to. Required to
            update or access a dataset when this pyvista_ndarray is updated.

        association : FieldAssociation, optional
            The FieldAssociation of the dataset the input array belongs to.

        """
        narray = convert_array(vtk_data_array)

        # Make arrays of 9 components into matrices. Also transpose
        # as VTK stores matrices in Fortran order
        shape = narray.shape
        if len(shape) == 2 and shape[1] == 9:
            narray = narray.reshape((shape[0], 3, 3)).transpose(0, 2, 1)
        return cls(narray, vtk_data_array, dataset=dataset, association=association)

    @classmethod
    def from_iter(cls, iterable, dtype=None, order=None):
        """Create a ``pyvista_ndarray`` instance from an iterable.

        Parameters
        ----------
        iterable : array_like
            Input data, in any form that can be converted to an array.
            This includes lists, lists of tuples, tuples, tuples of
            tuples, tuples of lists and ndarrays.

        dtype : data-type, optional
            By default, the data-type is inferred from the input data.

        order : {‘C’, ‘F’}, optional
            Whether to use row-major (C-style) or column-major
            (Fortran-style) memory representation. Defaults to ‘C’.

        """
        return cls(np.asarray(iterable, dtype, order))