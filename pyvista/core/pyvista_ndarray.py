import numpy
from vtk.numpy_interface.dataset_adapter import VTKObjectWrapper, ArrayAssociation, VTKArray
from vtk.vtkCommonCore import vtkWeakReference
from vtk.vtkCommonKitPython import vtkDataArray, vtkAbstractArray
import pyvista.utilities.helpers as helpers


# TODO, handle bool values, tests
class pyvista_ndarray(VTKArray):
    """This is a sub-class of numpy ndarray that stores a
    reference to a vtk array as well as the owning dataset.
    The numpy array and vtk array should point to the same
    memory location."""

    def __new__(cls, ndarray, vtk_array=None, dataset=None, association=None):
        # Input array is an already formed ndarray instance
        obj = numpy.asarray(ndarray).view(cls)
        obj.association = association or ArrayAssociation.FIELD
        # add the new attributes to the created instance
        obj.VTKObject = vtk_array
        if dataset:
            obj.dataset = vtkWeakReference()
            # Unless ALL pyvista objects ARE vtk objects, OR, ALL pyvista objects ARE
            # VTKObjectWrappers, which object gets set matters.
            if isinstance(dataset, VTKObjectWrapper):
                obj.dataset.Set(dataset.VTKObject)
            else:
                obj.dataset.Set(dataset)
        return obj

    @classmethod
    def from_any(cls, obj, dtype=None, order=None, dataset=None, association=None):
        """Factory method to create a `pyvista_ndarray`` instance from an object.

        Parameters
        ----------
        obj : any
            Object to create numpy array from based on type.

        dataset : vtkDataSet, optional
            The vtkDataSet which vtk_data_array belongs to. Required to
            update or access a dataset when this pyvista_ndarray is updated.

        dtype : data-type, optional
            By default, the data-type is inferred from the input data.

        order : {‘C’, ‘F’}, optional
            Whether to use row-major (C-style) or column-major (Fortran-style) memory representation. Defaults to ‘C’.

        """
        if isinstance(obj, (vtkDataArray, vtkAbstractArray)):
            return cls.from_vtk_data_array(vtk_data_array=obj, dataset=dataset)
        elif isinstance(obj, (list, tuple, numpy.ndarray)):
            return cls.from_iter(a=obj, dtype=dtype, order=order)
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

        """
        narray = helpers.convert_array(vtk_data_array)

        # Make arrays of 9 components into matrices. Also transpose
        # as VTK stores matrices in Fortran order
        shape = narray.shape
        if len(shape) == 2 and shape[1] == 9:
            narray = narray.reshape((shape[0], 3, 3)).transpose(0, 2, 1)
        return cls(narray, vtk_data_array, dataset=dataset)

    @classmethod
    def from_iter(cls, a, dtype=None, order=None):
        """Create a ``pyvista_ndarray`` instance from an iterable.

        Parameters
        ----------
        a : array_like
            Input data, in any form that can be converted to an array.
            This includes lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.

        dtype : data-type, optional
            By default, the data-type is inferred from the input data.

        order : {‘C’, ‘F’}, optional
            Whether to use row-major (C-style) or column-major (Fortran-style) memory representation. Defaults to ‘C’.

        """
        return cls(numpy.asarray(a, dtype, order))