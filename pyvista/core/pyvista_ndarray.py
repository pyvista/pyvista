"""Contains pyvista_ndarray a numpy ndarray type used in pyvista."""
from collections import Iterable
import numpy as np
from pyvista.utilities.helpers import FieldAssociation, convert_array
from vtk.numpy_interface.dataset_adapter import VTKObjectWrapper, VTKArray

try:
    from vtk.vtkCommonKitPython import buffer_shared, vtkAbstractArray, vtkWeakReference
except ImportError:
    from vtk.vtkCommonCore import buffer_shared, vtkAbstractArray, vtkWeakReference


class pyvista_ndarray(np.ndarray):
    """Link a numpy array with the vtk object the data is attached to."""

    def __new__(cls, array, dataset=None, association=FieldAssociation.NONE):
        """Allocate the array."""
        if isinstance(array, (Iterable, np.ndarray)):
            obj = np.asarray(array).view(cls)
        elif isinstance(array, vtkAbstractArray):
            obj = convert_array(array).view(cls)
            obj.VTKObject = array

        obj.association = association
        obj.dataset = vtkWeakReference()
        if isinstance(dataset, VTKObjectWrapper):
            obj.dataset.Set(dataset.VTKObject)
        else:
            obj.dataset.Set(dataset)
        return obj

    __array_finalize__ = VTKArray.__array_finalize__

    def __setitem__(self, key, value):
        """Set item at key index to value.

        When the array is changed it triggers "Modified()" which updates
        all upstream objects, including any render windows holding the
        object.
        """
        super().__setitem__(key, value)
        if self.VTKObject is not None:
            self.VTKObject.Modified()

    __getattr__ = VTKArray.__getattr__
