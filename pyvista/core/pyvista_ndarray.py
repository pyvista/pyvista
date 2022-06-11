"""Contains pyvista_ndarray a numpy ndarray type used in pyvista."""
from collections.abc import Iterable
from functools import wraps
from typing import Union

import numpy as np

from pyvista import _vtk
from pyvista.utilities.helpers import FieldAssociation, convert_array


def _to_numpy_dtype(array):
    """Return a NumPy dtype when an array is 0d.

    Otherwise, returns the array.

    Parameters
    ----------
    array : numpy.ndarray
        NumPy array.

    Returns
    -------
    numpy.ndarray or np.dtype
        NumPy dtype when an array is 0d. Otherwise, returns the array.

    """
    if array.shape == ():
        return array.dtype.type(array.item(0))
    return array


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

        When the array is changed it triggers "Modified()" which updates
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

    @wraps(np.min)
    def min(self, *args, **kwargs):
        """Wrap numpy.min to return a single value when applicable."""
        return _to_numpy_dtype(super().min(*args, **kwargs))

    @wraps(np.mean)
    def mean(self, *args, **kwargs):
        """Wrap numpy.mean to return a single value when applicable."""
        return _to_numpy_dtype(super().mean(*args, **kwargs))

    @wraps(np.max)
    def max(self, *args, **kwargs):
        """Wrap numpy.max to return a single value when applicable."""
        return _to_numpy_dtype(super().max(*args, **kwargs))

    @wraps(np.sum)
    def sum(self, *args, **kwargs):
        """Wrap numpy.sum to return a single value when applicable."""
        return _to_numpy_dtype(super().sum(*args, **kwargs))

    @wraps(np.prod)
    def prod(self, *args, **kwargs):
        """Wrap numpy.prod to return a single value when applicable."""
        return _to_numpy_dtype(super().prod(*args, **kwargs))

    @wraps(np.std)
    def std(self, *args, **kwargs):
        """Wrap numpy.std to return a single value when applicable."""
        return _to_numpy_dtype(super().std(*args, **kwargs))

    @wraps(np.ptp)
    def ptp(self, *args, **kwargs):
        """Wrap numpy.ptp to return a single value when applicable."""
        return _to_numpy_dtype(super().ptp(*args, **kwargs))

    __getattr__ = _vtk.VTKArray.__getattr__
