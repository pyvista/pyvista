"""Contains pyvista_ndarray a numpy ndarray type used in pyvista."""
from collections.abc import Iterable

from typing import Union
import numpy as np
from vtk.numpy_interface.dataset_adapter import VTKObjectWrapper, VTKArray

from pyvista.utilities.helpers import FieldAssociation, convert_array

try:
    from vtk.vtkCommonKitPython import buffer_shared, vtkAbstractArray, vtkWeakReference
except ImportError:
    from vtk.vtkCommonCore import buffer_shared, vtkAbstractArray, vtkWeakReference


class pyvista_ndarray(np.ndarray):
    """An ndarray which references the owning dataset and the underlying vtkArray."""

    def __new__(cls, array: Union[Iterable, vtkAbstractArray], dataset=None,
                association=FieldAssociation.NONE):
        """Allocate the array."""
        if isinstance(array, Iterable):
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

    def __array_finalize__(self, obj):
        """Finalize array (associate with parent metadata)."""
        # this is necessary to ensure that views/slices of pyvista_ndarray
        # objects stay associated with those of their parents.
        #
        # the VTKArray class uses attributes called `DataSet` and `Assocation`
        # to hold this data. I don't know why this class doesn't use the same
        # convention, but here we just map those over to the appropriate
        # attributes of this class
        VTKArray.__array_finalize__(self, obj)
        if np.shares_memory(self, obj):
            self.dataset = getattr(obj, 'dataset', None)
            self.association = getattr(obj, 'association', None)
            self.VTKObject = getattr(obj, 'VTKObject', None)

    def __setitem__(self, key: int, value):
        """Implement [] set operator.

        When the array is changed it triggers "Modified()" which updates
        all upstream objects, including any render windows holding the
        object.
        """
        super().__setitem__(key, value)
        if self.VTKObject is not None:
            self.VTKObject.Modified()

    __getattr__ = VTKArray.__getattr__
