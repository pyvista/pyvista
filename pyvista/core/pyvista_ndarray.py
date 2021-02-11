"""Contains pyvista_ndarray a numpy ndarray type used in pyvista."""
from collections.abc import Iterable

from typing import Union
import numpy as np

from pyvista.core import vtki
from pyvista.utilities.helpers import FieldAssociation, convert_array


class pyvista_ndarray(np.ndarray):
    """An ndarray which references the owning dataset and the underlying vtkArray."""

    def __new__(cls, array: Union[Iterable, vtki.vtkAbstractArray], dataset=None,
                association=FieldAssociation.NONE):
        """Allocate the array."""
        if isinstance(array, Iterable):
            obj = np.asarray(array).view(cls)
        elif isinstance(array, vtki.vtkAbstractArray):
            obj = convert_array(array).view(cls)
            obj.VTKObject = array

        obj.association = association
        obj.dataset = vtki.vtkWeakReference()
        if isinstance(dataset, vtki.VTKObjectWrapper):
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
        vtki.VTKArray.__array_finalize__(self, obj)
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

    __getattr__ = vtki.VTKArray.__getattr__
