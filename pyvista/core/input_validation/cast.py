import numpy as np

def cast_to_list_array(arr, /, *, name="Input"):
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
    return cast_to_ndarray(arr, name=name).tolist()


def cast_to_tuple_array(arr, /, *, name="Input"):
    """Cast an array to a nested tuple.

    Parameters
    ----------
    arr : array_like
        Array to cast.

    Returns
    -------
    tuple
        Tuple or nested tuple array.
    """
    arr = cast_to_ndarray(arr, name=name).tolist()

    def _to_tuple(s):
        return tuple(_to_tuple(i) for i in s) if isinstance(s, list) else s

    return _to_tuple(arr)


def cast_to_ndarray(arr, /, *, as_any=True, dtype=None, copy=False, name="Input") -> np.ndarray:
    """Cast array to a NumPy ndarray.

    Parameters
    ----------
    arr : array_like
        Array to cast.

    Raises
    ------
    ValueError
        If input cannot be cast as a NumPy ndarray.

    Returns
    -------
    np.ndarray
        NumPy ndarray.

    """
    try:
        if as_any:
            out = np.asanyarray(arr, dtype=dtype)
            if copy:
                out = out.copy()
        else:
            out = np.array(arr, dtype=dtype, copy=copy)
        if out.dtype.name == 'object':
            # NumPy will normally raise ValueError automatically for
            # object arrays, but on some systems it will not, so raise
            # error manually
            raise ValueError
    except (ValueError, np.VisibleDeprecationWarning) as e:
        raise ValueError(f"{name} cannot be cast as {np.ndarray}.") from e
    return out
