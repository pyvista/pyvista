"""Contains ``pyvista_ndarray`` a NumPy ``ndarray`` type used in PyVista."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING
from typing import cast

import numpy as np

from pyvista import _vtk
from pyvista.core._vtk_utilities import VTKObjectWrapperCheckSnakeCase

from .utilities.arrays import FieldAssociation
from .utilities.arrays import convert_array
from .utilities.misc import _NoNewAttrMixin

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt

    from pyvista import DataSet

    from ._typing_core import ArrayLike
    from ._typing_core import NumpyArray


class pyvista_ndarray(_NoNewAttrMixin, np.ndarray):  # numpydoc ignore=PR02  # noqa: N801
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

    dataset: _vtk.vtkWeakReference | None
    association: FieldAssociation
    VTKObject: _vtk.vtkAbstractArray | None

    def __new__(  # noqa: PYI034
        cls: type[pyvista_ndarray],
        array: ArrayLike[float] | _vtk.vtkAbstractArray,
        dataset: DataSet | _vtk.vtkDataSet | _vtk.VTKObjectWrapper | None = None,
        association: FieldAssociation = FieldAssociation.NONE,
    ) -> pyvista_ndarray:
        """Allocate the array."""
        vtk_object = None
        if isinstance(array, _vtk.vtkAbstractArray):
            obj = convert_array(array).view(cls)
            vtk_object = array
        elif isinstance(array, Iterable):
            obj = np.asarray(array).view(cls)
        else:
            msg = (  # type: ignore[unreachable]
                f'pyvista_ndarray got an invalid type {type(array)}. '
                'Expected an Iterable or vtk.vtkAbstractArray'
            )
            raise TypeError(msg)

        dataset_ref = None
        if dataset is not None:
            dataset_ref = _vtk.vtkWeakReference()
            if isinstance(dataset, _vtk.VTKObjectWrapper):
                dataset_ref.Set(dataset.VTKObject)
            else:
                dataset_ref.Set(cast('_vtk.vtkDataSet', dataset))
        obj.__dict__.update(dataset=dataset_ref, association=association, VTKObject=vtk_object)
        return obj

    def __array_finalize__(self: pyvista_ndarray, obj: npt.NDArray[Any] | None) -> None:
        """Finalize array (associate with parent metadata)."""
        # Views and slices stay associated with the dataset and VTK array of their parent.
        # This runs for every view and ufunc result, so write the instance dict directly.
        if isinstance(obj, pyvista_ndarray):
            if np.shares_memory(self, obj):
                self.__dict__.update(
                    dataset=obj.dataset, association=obj.association, VTKObject=obj.VTKObject
                )
                return
        elif obj is not None and np.shares_memory(self, obj):
            self.__dict__.update(
                dataset=getattr(obj, 'dataset', None),
                association=getattr(obj, 'association', FieldAssociation.NONE),
                VTKObject=getattr(obj, 'VTKObject', None),
            )
            return
        self.__dict__.update(dataset=None, association=FieldAssociation.NONE, VTKObject=None)

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
        """Return a NumPy scalar if array is 0d.

        See https://github.com/numpy/numpy/issues/5819

        """
        if out_arr.ndim:
            return super().__array_wrap__(out_arr, context, return_scalar)

        # Match numpy's behavior and return a numpy dtype scalar
        return out_arr[()]

    __getattr__ = VTKObjectWrapperCheckSnakeCase.__getattr__
