"""Contains pyvista_ndarray a numpy ndarray type used in pyvista."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING
from typing import cast

import numpy as np

from . import _vtk_core as _vtk
from .utilities.arrays import FieldAssociation
from .utilities.arrays import convert_array

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt

    from pyvista import DataSet

    from ._typing_core import ArrayLike
    from ._typing_core import NumpyArray


class pyvista_ndarray(np.ndarray):  # numpydoc ignore=PR02  # noqa: N801
    """A ndarray which references the owning dataset and the underlying vtk array.

    This array can be acted upon just like a :class:`numpy.ndarray`.

    Parameters
    ----------
    array : ArrayLike or :vtk:`vtkAbstractArray`
        Array like.

    dataset : DataSet
        Input dataset.

    association : pyvista.core.utilities.arrays.FieldAssociation
        Field association.

    Examples
    --------
    Return the points of a Sphere as a :class:`pyvista.pyvista_ndarray`.

    >>> import pyvista as pv
    >>> mesh = pv.Sphere()
    >>> mesh.points  # doctest:+SKIP
    pyvista_ndarray([[-5.5511151e-17,  0.0000000e+00, -5.0000000e-01],
                     [ 5.5511151e-17,  0.0000000e+00,  5.0000000e-01],
                     [-5.4059509e-02,  0.0000000e+00, -4.9706897e-01],
                     ...,
                     [-1.5616201e-01, -3.3193260e-02,  4.7382659e-01],
                     [-1.0513641e-01, -2.2347433e-02,  4.8831028e-01],
                     [-5.2878179e-02, -1.1239604e-02,  4.9706897e-01]],
                    dtype=float32)

    """

    def __new__(  # noqa: PYI034
        cls: type[pyvista_ndarray],
        array: ArrayLike[float] | _vtk.vtkAbstractArray,
        dataset: DataSet | _vtk.vtkDataSet | _vtk.VTKObjectWrapper | None = None,
        association: FieldAssociation = FieldAssociation.NONE,
    ) -> pyvista_ndarray:
        """Allocate the array."""
        if isinstance(array, _vtk.vtkAbstractArray):
            obj = convert_array(array).view(cls)
            obj.VTKObject = array
        elif isinstance(array, Iterable):
            obj = np.asarray(array).view(cls)
        else:
            msg = (  # type: ignore[unreachable]
                f'pyvista_ndarray got an invalid type {type(array)}. '
                'Expected an Iterable or vtk.vtkAbstractArray'
            )
            raise TypeError(msg)

        obj.association = association
        obj.dataset = _vtk.vtkWeakReference()
        if isinstance(dataset, _vtk.VTKObjectWrapper):
            obj.dataset.Set(dataset.VTKObject)
        else:
            obj.dataset.Set(cast('_vtk.vtkDataSet', dataset))
        return obj

    def __array_finalize__(self: pyvista_ndarray, obj: npt.NDArray[Any] | None) -> None:
        """Finalize array (associate with parent metadata)."""
        # this is necessary to ensure that views/slices of pyvista_ndarray
        # objects stay associated with those of their parents.
        #
        # the VTKArray class uses attributes called `DataSet` and `Association`
        # to hold this data. I don't know why this class doesn't use the same
        # convention, but here we just map those over to the appropriate
        # attributes of this class
        _vtk.VTKArray.__array_finalize__(self, obj)  # type: ignore[arg-type]
        if np.shares_memory(self, obj):
            self.dataset = getattr(obj, 'dataset', None)
            self.association = getattr(obj, 'association', FieldAssociation.NONE)
            self.VTKObject = getattr(obj, 'VTKObject', None)
        else:
            self.dataset = None
            self.association = FieldAssociation.NONE
            self.VTKObject = None

    def __setitem__(self: pyvista_ndarray, key: int | NumpyArray[int], value: Any) -> None:  # type: ignore[override]
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

    def __array_wrap__(self: pyvista_ndarray, out_arr, context=None, return_scalar: bool = False):  # noqa: ANN001, ANN204, FBT001, FBT002
        """Return a numpy scalar if array is 0d.

        See https://github.com/numpy/numpy/issues/5819

        """
        if out_arr.ndim:
            return super().__array_wrap__(out_arr, context, return_scalar)

        # Match numpy's behavior and return a numpy dtype scalar
        return out_arr[()]

    __getattr__ = _vtk.VTKObjectWrapperCheckSnakeCase.__getattr__
