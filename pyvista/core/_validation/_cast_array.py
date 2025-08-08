"""Array casting functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pyvista.core._typing_core import ArrayLike
    from pyvista.core._typing_core import NumberType
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import _ArrayLikeOrScalar
    from pyvista.core._typing_core import _FiniteNestedList
    from pyvista.core._typing_core import _FiniteNestedTuple


def _cast_to_list(
    arr: _ArrayLikeOrScalar[NumberType],
) -> NumberType | _FiniteNestedList[NumberType]:
    """Cast an array to a nested list.

    Parameters
    ----------
    arr : float | ArrayLike[float]
        Array to cast.

    Returns
    -------
    list
        List or nested list array.

    """
    return _cast_to_numpy(arr).tolist()


def _cast_to_tuple(
    arr: ArrayLike[NumberType],
) -> NumberType | _FiniteNestedTuple[NumberType]:
    """Cast an array to a nested tuple.

    Parameters
    ----------
    arr : float | ArrayLike[float]
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


def _cast_to_numpy(
    arr: _ArrayLikeOrScalar[NumberType],
    /,
    *,
    as_any: bool = True,
    dtype: npt.DTypeLike | None = None,
    copy: bool = False,
    must_be_real: bool = False,
) -> NumpyArray[NumberType]:
    """Cast array to a NumPy ndarray.

    Object arrays are not allowed but the dtype is otherwise unchecked by default.
    String arrays and complex numbers are therefore allowed.

    .. warning::

        Arrays intended for use with vtk should set ``must_be_real=True``
        since ``numpy_to_vtk`` uses the array values directly without
        checking for complex arrays.

    Parameters
    ----------
    arr : float | ArrayLike[float]
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
        Raise a ``TypeError`` if the array does not have real numbers, i.e.
        its data type is not integer or floating.

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
    try:
        VisibleDeprecationWarning = np.exceptions.VisibleDeprecationWarning
    except AttributeError:
        # we only type for newer numpy, and this branch only touched in older numpy
        if not TYPE_CHECKING:
            VisibleDeprecationWarning = np.VisibleDeprecationWarning

    try:
        out = np.asanyarray(arr, dtype=dtype) if as_any else np.asarray(arr, dtype=dtype)

        if copy and out is arr:
            # we requested a copy but didn't end up with one
            out = out.copy()
    except (ValueError, VisibleDeprecationWarning) as e:
        msg = f'Input cannot be cast as {np.ndarray}.'
        raise ValueError(msg) from e
    if must_be_real and not issubclass(out.dtype.type, (np.floating, np.integer)):
        msg = f'Array must have real numbers. Got dtype {out.dtype.type}'
        raise TypeError(msg)
    elif out.dtype.name == 'object':
        msg = f'Object arrays are not supported. Got {arr} when casting to a NumPy array.'
        raise TypeError(msg)
    return out
