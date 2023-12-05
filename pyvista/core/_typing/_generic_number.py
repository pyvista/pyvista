"""Generic number types.

This module supports type casting of builtin types (e.g. `int`) to their
respective numpy type (e.g. `numpy.int_`). Its main purpose is to make
sure mypy does not complain about annotations like `numpy.dtype[int]`
or `numpy.ndarray[Any, numpy.dtype[int]]`.

"""

from typing import Any, Generic, TypeVar, overload

import numpy as np

# NOTE: The TypeVar for numpy's dtype is bound to np.generic, which means builtin types
# are not considered valid by mypy. Instead, we define a strict set of invariant types
# which includes builtin types and their respective numpy type.
_NumberType = TypeVar("_NumberType", bool, int, float, np.uint8, np.bool_, np.int_, np.float_)


class _GenericNumber(Generic[_NumberType], np.generic):
    """Generic number type used to cast builtin types to numpy.generic type."""

    @overload
    def __new__(cls, dtype: type[bool]) -> "_NumpyBool":  # type: ignore[misc, overload-overlap]
        ...

    @overload
    def __new__(cls, dtype: type[np.bool_]) -> "_NumpyBool":
        ...

    @overload
    def __new__(cls, dtype: type[int]) -> "_NumpyInt":
        ...

    @overload
    def __new__(cls, dtype: type[np.int_]) -> "_NumpyInt":
        ...

    @overload
    def __new__(cls, dtype: type[np.uint8]) -> "_NumpyUInt8":
        ...

    @overload
    def __new__(cls, dtype: type[float]) -> "_NumpyFloat":
        ...

    @overload
    def __new__(cls, dtype: type[np.float_]) -> "_NumpyFloat":
        ...

    def __new__(cls, dtype) -> "_GenericNumber":
        if type(dtype) in (bool, np.bool_):
            cls = _NumpyInt
        elif type(dtype) in (int, np.int_):
            cls = _NumpyInt
        elif type(dtype) is np.uint8:
            cls = _NumpyUInt8
        elif type(dtype) in (float, np.float_):
            cls = _NumpyFloat
        else:
            cls = _NumpyNumber
        return object.__new__(cls)


class _NumpyBool(_GenericNumber, np.bool_):
    ...


class _NumpyInt(_GenericNumber, np.int_):
    ...


class _NumpyUInt8(_GenericNumber, np.uint8):
    ...


class _NumpyFloat(_GenericNumber, np.float_):
    ...


class _NumpyNumber(_GenericNumber, np.number):
    ...


# MyPy checks should succeed
# Check bool
_a1: np.bool_ = _GenericNumber(bool)
_a2: np.bool_ = _GenericNumber(np.bool_)
_a3: _GenericNumber[bool] = _GenericNumber(bool)
_a4: _GenericNumber[np.bool_] = _GenericNumber(np.bool_)
_a5: _GenericNumber[np.bool_] = _GenericNumber(bool)
_a6: _GenericNumber[bool] = _GenericNumber(np.bool_)

# Check int
_b1: np.int_ = _GenericNumber(int)
_b2: np.int_ = _GenericNumber(np.int_)
_b3: _GenericNumber[int] = _GenericNumber(int)
_b4: _GenericNumber[np.int_] = _GenericNumber(np.int_)
_b5: _GenericNumber[np.int_] = _GenericNumber(int)
_b6: _GenericNumber[int] = _GenericNumber(np.int_)

# Check uint8
_c1: np.uint8 = _GenericNumber(np.uint8)
_c2: _GenericNumber[np.uint8] = _GenericNumber(np.uint8)

# Check float
_d1: np.float_ = _GenericNumber(float)
_d2: np.float_ = _GenericNumber(np.float_)
_d3: _GenericNumber[float] = _GenericNumber(float)
_d4: _GenericNumber[np.float_] = _GenericNumber(np.float_)
_d5: _GenericNumber[np.float_] = _GenericNumber(float)
_d6: _GenericNumber[float] = _GenericNumber(np.float_)

# Check dtype
_z2: np.dtype[np.bool_] = np.dtype(_GenericNumber(np.bool_))
_z3: np.ndarray[Any, np.dtype[_GenericNumber[bool]]] = np.ndarray((), dtype=bool)
_z4: np.ndarray[Any, np.dtype[np.bool_]] = np.ndarray((), dtype=_GenericNumber(bool))
_z5: np.dtype[_GenericNumber[bool]] = np.dtype(_GenericNumber(bool))


# MyPy checks should fail
_fail1: np.dtype[np.bool_] = np.dtype(_GenericNumber(int))  # type: ignore[arg-type]
_fail2: np.dtype[bool] = np.dtype(bool)  # type: ignore[type-var, assignment]
_fail3: np.dtype[_GenericNumber[bool]] = np.dtype(bool)  # type: ignore[assignment]
