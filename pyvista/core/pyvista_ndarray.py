"""Contains ``pyvista_ndarray`` a NumPy ``ndarray`` type used in PyVista."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING
from typing import cast

import numpy as np

from pyvista import _vtk
from pyvista.core._vtk_utilities import VTKObjectWrapperCheckSnakeCase

from .utilities.arrays import FieldAssociation
from .utilities.arrays import _vtk_array_to_numpy
from .utilities.misc import _NoNewAttrMixin

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt

    from pyvista import DataSet

    from ._typing_core import ArrayLike
    from ._typing_core import NumpyArray


class pyvista_ndarray(_NoNewAttrMixin, np.ndarray):  # noqa: N801  # numpydoc ignore=PR02
    """A ``ndarray`` which references the owning dataset and the underlying vtk array.

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

    # Metadata of an unassociated array; instances only store what differs
    dataset: _vtk.vtkWeakReference | None = None
    association: FieldAssociation = FieldAssociation.NONE
    VTKObject: _vtk.vtkAbstractArray | None = None

    def __new__(  # noqa: PYI034
        cls: type[pyvista_ndarray],
        array: ArrayLike[float] | _vtk.vtkAbstractArray,
        dataset: DataSet | _vtk.vtkDataSet | _vtk.VTKObjectWrapper | None = None,
        association: FieldAssociation = FieldAssociation.NONE,
    ) -> pyvista_ndarray:
        """Allocate the array."""
        # Write the instance dict directly; attribute assignment goes through _NoNewAttrMixin
        if isinstance(array, _vtk.vtkAbstractArray):
            obj = _vtk_array_to_numpy(array).view(cls)
            obj.__dict__['VTKObject'] = array
        elif isinstance(array, Iterable):
            obj = np.asarray(array).view(cls)
        else:
            msg = (  # type: ignore[unreachable]
                f'pyvista_ndarray got an invalid type {type(array)}. '
                'Expected an Iterable or vtk.vtkAbstractArray'
            )
            raise TypeError(msg)

        if dataset is not None:
            reference = _vtk.vtkWeakReference()
            if isinstance(dataset, _vtk.VTKObjectWrapper):
                reference.Set(dataset.VTKObject)
            else:
                reference.Set(cast('_vtk.vtkDataSet', dataset))
            obj.__dict__['dataset'] = reference
        if association is not FieldAssociation.NONE:
            obj.__dict__['association'] = association
        return obj

    def __array_finalize__(self: pyvista_ndarray, obj: npt.NDArray[Any] | None) -> None:
        """Finalize array (associate with parent metadata)."""
        # Views and slices keep their parent's metadata; copies and ufunc results do not
        if isinstance(obj, pyvista_ndarray):
            dataset = obj.dataset
            vtk_object = obj.VTKObject
            association = obj.association
            if (
                dataset is not None
                or vtk_object is not None
                or association is not FieldAssociation.NONE
            ) and np.may_share_memory(self, obj):
                self.__dict__.update(
                    dataset=dataset, association=association, VTKObject=vtk_object
                )
        elif obj is not None and type(obj) is not np.ndarray and np.may_share_memory(self, obj):
            self.__dict__.update(
                dataset=getattr(obj, 'dataset', None),
                association=getattr(obj, 'association', FieldAssociation.NONE),
                VTKObject=getattr(obj, 'VTKObject', None),
            )

    def __setitem__(self: pyvista_ndarray, key: int | NumpyArray[int], value: Any) -> None:  # type: ignore[override]
        """Implement [] set operator.

        When the array is changed it triggers "Modified()" which updates
        all upstream objects, including any render windows holding the
        object.
        """
        super().__setitem__(key, value)
        vtk_object = self.VTKObject
        if vtk_object is not None:
            vtk_object.Modified()

        # the associated dataset should also be marked as modified
        dataset = self.dataset
        if dataset is not None:
            owner = dataset.Get()
            if owner is not None:
                owner.Modified()

    def __array_wrap__(self: pyvista_ndarray, out_arr, context=None, return_scalar: bool = False):  # noqa: ANN001, ANN204, FBT001, FBT002
        """Return a NumPy scalar if array is 0d.

        See https://github.com/numpy/numpy/issues/5819

        """
        if out_arr.ndim:
            return super().__array_wrap__(out_arr, context, return_scalar)

        # Match numpy's behavior and return a numpy dtype scalar
        return out_arr[()]

    __getattr__ = VTKObjectWrapperCheckSnakeCase.__getattr__
