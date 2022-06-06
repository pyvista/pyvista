"""Contains pyvista_ndarray a numpy ndarray type used in pyvista."""
from collections.abc import Iterable
from functools import wraps
from typing import Union

import numpy as np

from pyvista import _vtk
from pyvista.utilities.helpers import FieldAssociation, convert_array


class pyvista_ndarray(np.ndarray):
    """An ndarray which references the owning dataset and the underlying vtkArray."""

    def __new__(
        cls,
        array: Union[Iterable, _vtk.vtkAbstractArray],
        dataset=None,
        association=FieldAssociation.NONE,
    ):
        """Allocate the array."""
        if isinstance(array, Iterable):
            obj = np.asarray(array).view(cls)
        elif isinstance(array, _vtk.vtkAbstractArray):
            obj = convert_array(array).view(cls)
            obj.VTKObject = array
        else:
            raise TypeError(
                f'pyvista_ndarray got an invalid type {type(array)}.  '
                'Expected an Iterable or vtk.vtkAbstractArray'
            )

        obj.association = association
        obj.dataset = _vtk.vtkWeakReference()
        if isinstance(dataset, _vtk.VTKObjectWrapper):
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
        _vtk.VTKArray.__array_finalize__(self, obj)
        if np.shares_memory(self, obj):
            self.dataset = getattr(obj, 'dataset', None)
            self.association = getattr(obj, 'association', FieldAssociation.NONE)
            self.VTKObject = getattr(obj, 'VTKObject', None)
        else:
            self.dataset = None
            self.association = FieldAssociation.NONE
            self.VTKObject = None

    def __setitem__(self, key: Union[int, np.ndarray], value):
        """Implement [] set operator.

        When the array is changed it triggers ``Modified()`` which updates
        all upstream objects, including any render windows holding the
        object.
        """
        super().__setitem__(key, value)
        if self.VTKObject is not None:
            self.VTKObject.Modified()

        # the associated dataset should also be marked as modified
        dataset = self.dataset
        if dataset is not None and dataset.Get():
            dataset.Get().Modified()

    @wraps(np.max)
    def max(self, *args, **kwargs):
        """Wrap numpy.max to return a single value when applicable."""
        output = super().max(*args, **kwargs)
        if output.shape == ():
            return output.item(0)
        return output

    @wraps(np.mean)
    def mean(self, *args, **kwargs):
        """Wrap numpy.mean to return a single value when applicable."""
        output = super().mean(*args, **kwargs)
        if output.shape == ():
            return output.item(0)
        return output

    @wraps(np.sum)
    def sum(self, *args, **kwargs):
        """Wrap numpy.sum to return a single value when applicable."""
        output = super().sum(*args, **kwargs)
        if output.shape == ():
            return output.item(0)
        return output

    @wraps(np.min)
    def min(self, *args, **kwargs):
        """Wrap numpy.min to return a single value when applicable."""
        output = super().min(*args, **kwargs)
        if output.shape == ():
            return output.item(0)
        return output

    @wraps(np.std)
    def std(self, *args, **kwargs):
        """Wrap numpy.std to return a single value when applicable."""
        output = super().std(*args, **kwargs)
        if output.shape == ():
            return output.item(0)
        return output

    @wraps(np.prod)
    def prod(self, *args, **kwargs):
        """Wrap numpy.prod to return a single value when applicable."""
        output = super().prod(*args, **kwargs)
        if output.shape == ():
            return output.item(0)
        return output

    # def __del__(self):
    #     del self.dataset

    __getattr__ = _vtk.VTKArray.__getattr__
