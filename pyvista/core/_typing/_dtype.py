"""Generic number types.

This module supports type casting of builtin types (e.g. `int`) to their
respective numpy type (e.g. `numpy.int_`). Its main purpose is to make
sure mypy does not complain about annotations like `numpy.dtype[int]`
or `numpy.ndarray[Any, numpy.dtype[int]]`.

"""

from abc import ABCMeta
from typing import Generic, TypeVar

import numpy as np

# NOTE:
# The numpy.dtype stub defines overloads for builtin types (e.g. float) but mypy will
# nevertheless complain if you try to annotate dtype[...] for builtin types since the
# numpy.dtype TypeVar is bound to numpy.generic. As a workaround, here we define an
# invariant TypeVar (no bound) for all numpy numeric types from https://numpy.org/doc/stable/user/basics.types.html
# and create a new dtype object using a metaclass (and ignore mypy's typevar error).

_DTypeScalar = TypeVar(
    "_DTypeScalar",
    bool,
    int,
    float,
    np.bool_,
    np.byte,
    np.ubyte,
    np.short,
    np.ushort,
    np.intc,
    np.uintc,
    np.int_,
    np.uint,
    np.longlong,
    np.ulonglong,
    np.half,
    np.double,
    np.longdouble,
    # np.csingle,  # exclude complex types
    # np.cdouble,
    # np.clongdouble,
)


class _DType(np.generic, Generic[_DTypeScalar], metaclass=ABCMeta):
    def __new__(cls, dtype: type[_DTypeScalar]) -> np.dtype[_DTypeScalar]:  # type: ignore[misc, type-var]
        raise NotImplementedError
