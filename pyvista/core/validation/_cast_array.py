"""Array casting functions."""

from typing import Optional

import numpy as np
import numpy.typing as npt

from pyvista.core._typing_core import Array, NumpyArray
from pyvista.core._typing_core._array_like import _ArrayLikeOrScalar, _NumberType


def _cast_to_list(arr: _ArrayLikeOrScalar[_NumberType]) -> list:
    """Cast an array to a nested list.

    Parameters
    ----------
    arr : array_like
        Array to cast.

    Returns
    -------
    list
        List or nested list array.
    """
    return _cast_to_numpy(arr).tolist()


def _cast_to_tuple(arr: Array[_NumberType]) -> tuple:
    """Cast an array to a nested tuple.

    Parameters
    ----------
    arr : ArrayLike
        Array to cast.

    Returns
    -------
    tuple
        Tuple or nested tuple array.
    """
    arr = _cast_to_numpy(arr).tolist()

    def _to_tuple(s):
        return tuple(_to_tuple(i) for i in s) if isinstance(s, list) else s

    return _to_tuple(arr)


# from typing import overload


# @overload  # number -> array
# def cast_to_ndarray(  # numpydoc ignore=GL08
#     arr: _NumberType,
#     /,
#     *,
#     as_any: bool = ...,
#     dtype: Optional[npt.DTypeLike] = ...,
#     copy: bool = ...,
# ) -> NumpyArray[_NumberType]:
#     ...
#
#
# @overload  # array -> array
# def cast_to_ndarray(  # numpydoc ignore=GL08
#     arr: Array[_NumberType],
#     /,
#     *,
#     as_any: bool = ...,
#     dtype: Optional[npt.DTypeLike] = ...,
#     copy: bool = ...,
# ) -> NumpyArray[_NumberType]:
#     ...


def _cast_to_numpy(
    arr: _ArrayLikeOrScalar[_NumberType],
    /,
    *,
    as_any: bool = True,
    dtype: Optional[npt.DTypeLike] = None,
    copy: bool = False,
    must_be_real=False,
    name: str = "Array",
) -> NumpyArray[_NumberType]:
    """Cast array to a NumPy ndarray.

    Object arrays are not allowed but the dtype is otherwise unchecked by default.
    String arrays and complex numbers are therefore allowed.

    .. warning::

        Arrays intended for use with vtk should set ``must_be_real=True``
        since ``numpy_to_vtk`` uses the array values directly without
        checking for complex arrays.

    Parameters
    ----------
    arr : ArrayLike
        Array to cast.

    as_any : bool, default: True
        Allow subclasses of ``np.ndarray`` to pass through without
        making a copy.

    dtype : npt.typing.DTypeLike, optional
        The data-type of the returned array.

    copy : bool, default: False
        If ``True``, a copy of the array is returned. A copy is always
        returned if the array:

            * is a nested sequence
            * is a subclass of ``np.ndarray`` and ``as_any`` is ``False``.

    must_be_real : bool, default: True
        Raise a TypeError if the array does not have real numbers, i.e.
        its data type is not integer or floating.

    name : str, default: "Array"
        Variable name to use in the error messages if any of the
        validation checks fail.

    Raises
    ------
    ValueError
        If input cannot be cast as a NumPy ndarray.
    TypeError
        If an object array is created or if the data is not real numbers
        and ``must_be_real`` is ``True``.

    Returns
    -------
    np.ndarray
        NumPy ndarray.

    """
    # needed to support numpy <1.25
    # needed to support vtk 9.0.3
    # check for removal when support for vtk 9.0.3 is removed
    try:
        VisibleDeprecationWarning = np.exceptions.VisibleDeprecationWarning
    except AttributeError:
        VisibleDeprecationWarning = np.VisibleDeprecationWarning

    try:
        if as_any:
            out = np.asanyarray(arr, dtype=dtype)
            if copy and out is arr:
                out = out.copy()
        else:
            out = np.array(arr, dtype=dtype, copy=copy)
    except (ValueError, VisibleDeprecationWarning) as e:
        raise ValueError(f"{name} cannot be cast as {np.ndarray}.") from e
    if must_be_real is True and not issubclass(out.dtype.type, (np.floating, np.integer)):
        raise TypeError(f"{name} must have real numbers. Got dtype {out.dtype.type}.")
    elif out.dtype.name == 'object':
        raise TypeError(f"{name} is an object array. Object arrays are not supported.")
    return out
