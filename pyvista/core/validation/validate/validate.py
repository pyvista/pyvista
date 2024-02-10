"""Functions that validate input and return a standard representation.

.. versionadded:: 0.44.0

An array validator function typically:

* Uses array or type checkers to check the type and/or value of
  input arguments.
* Applies (optional) constraints, e.g. input or output must have a
  specific length, shape, type, data-type, etc.
* Accepts many different input types or values and standardizes the
  output as a single representation with known properties.

"""

from copy import deepcopy
import inspect
from itertools import product
from typing import Any, Dict, List, Literal, NoReturn, Optional, Tuple, Union

import numpy as np

from pyvista.core import _vtk_core as _vtk
from pyvista.core._typing_core import Matrix, NumpyArray, TransformLike, Vector
from pyvista.core._typing_core._array_like import _ArrayLikeOrScalar, _NumberType
from pyvista.core.validation._array_wrapper import _ArrayLikeWrapper, _BuiltinWrapper
from pyvista.core.validation.check import (
    ShapeLike,
    check_contains,
    check_finite,
    check_integer,
    check_length,
    check_nonnegative,
    check_range,
    check_real,
    check_shape,
    check_sorted,
    check_string,
    check_subdtype,
)


def validate_array(
    array: _ArrayLikeOrScalar[_NumberType],
    /,
    *,
    must_have_shape=None,
    must_have_dtype=None,
    must_have_length=None,
    must_have_min_length=None,
    must_have_max_length=None,
    must_be_nonnegative=False,
    must_be_finite=False,
    must_be_real=True,
    must_be_integer=False,
    must_be_sorted=False,
    must_be_in_range=None,
    strict_lower_bound=False,
    strict_upper_bound=False,
    reshape_to=None,
    broadcast_to=None,
    dtype_out=None,
    as_any=True,
    copy=False,
    output_type=None,
    name="Array",
):
    """Check and validate a numeric array meets specific requirements.

    Validate an array to ensure it is numeric, has a specific shape,
    data-type, and/or has values that meet specific requirements such as
    being sorted, having integer values, or is finite. The output type of
    the array can also be explicitly set to standardize the array
    representation.

    By default, the output type of the array is the same as the input
    when the input is a scalar, NumPy array, list, tuple, singly-nested
    list or singly-nested tuple, i.e.:

    * Scalars: ``T`` -> ``T``
    * NumPy arrays: ``NDArray[T]`` -> ``NDArray[T]``
    * Sequences:

        * ``tuple[T]`` -> ``tuple[T]``
        * ``list[T]`` -> ``list[T]``
        * ``tuple[tuple[[T]]`` -> ``tuple[tuple[T]]``
        * ``list[list[[T]]`` -> ``list[list[T]]``

    For the above inputs, this function will validate the array without
    copying its data wherever posibble. Any other array-like inputs
    are first cast to a NumPy array before validation and returned as
    a NumPy array by default.

    The array's output can also be reshaped or broadcast, cast as a
    nested tuple or list array, or cast to a specific data type.

    .. note::

        This function is primarily designed to work with homogeneous
        numeric arrays with a regular shape. Any other array-like
        inputs (e.g. structured arrays, string arrays) are not
        supported.

    See Also
    --------
    validate_number
        Specialized function for single numbers.

    validate_array3
        Specialized function for 3-element arrays.

    validate_arrayN
        Specialized function for one-dimensional arrays.

    validate_arrayN_uintlike
        Specialized function for one-dimensional arrays with unsigned integers.

    validate_arrayNx3
        Specialized function for Nx3 dimensional arrays.

    validate_data_range
        Specialized function for data ranges.

    Parameters
    ----------
    array : Number | Array
        Number or array to be validated, in any form that can be converted to
        a :class:`np.ndarray`. This includes lists, lists of tuples, tuples,
        tuples of tuples, tuples of lists and ndarrays.

    must_have_shape : ShapeLike | list[ShapeLike], optional
        :func:`Check <pyvista.core.validation.check.check_has_shape>`
        if the array has a specific shape. Specify a single shape
        or a ``list`` of any allowable shapes. If an integer, the array must
        be 1-dimensional with that length. Use a value of ``-1`` for any
        dimension where its size is allowed to vary. Use ``()`` to allow
        scalar values (i.e. 0-dimensional). Set to ``None`` if the array
        can have any shape (default).

    must_have_dtype : numpy.typing.DTypeLike | Sequence[numpy.typing.DTypeLike], optional
        :func:`Check <pyvista.core.validation.check.check__subdtype>`
        if the array's data-type has the given dtype. Specify a
        :class:`numpy.dtype` object or dtype-like base class which the
        array's data must be a subtype of. If a sequence, the array's data
        must be a subtype of at least one of the specified dtypes.

    must_have_length : int | Vector[int], optional
        :func:`Check <pyvista.core.validation.check.check_has_length>`
        if the array has the given length. If multiple values are given,
        the array's length must match one of the values.

        .. note ::

            The array's length is determined after reshaping the array
            (if ``reshape_to`` is not ``None``) and after broadcasting (if
            ``broadcast_to`` is not ``None``). Therefore, the specified length
            values should take the array's new shape into consideration if
            applicable.

    must_have_min_length : int, optional
        :func:`Check <pyvista.core.validation.check.check_has_length>`
        if the array's length is this value or greater. See note in
        ``must_have_length`` for details.

    must_have_max_length : int, optional
        :func:`Check <pyvista.core.validation.check.check_has_length>`
        if the array' length is this value or less. See note in
        ``must_have_length`` for details.

    must_be_nonnegative : bool, default: False
        :func:`Check <pyvista.core.validation.check.check_nonnegative>`
        if all elements of the array are nonnegative.

    must_be_finite : bool, default: False
        :func:`Check <pyvista.core.validation.check.check_finite>`
        if all elements of the array are finite, i.e. not ``infinity``
        and not Not a Number (``NaN``).

    must_be_real : bool, default: True
        :func:`Check <pyvista.core.validation.check.check_real>`
        if the array has real numbers, i.e. its data type is integer or
        floating.

        .. warning::

            Setting this parameter to ``False`` can result in unexpected
            behavior and is not recommended. There is limited support
            for complex number and/or string arrays.

    must_be_integer : bool, default: False
        :func:`Check <pyvista.core.validation.check.check_integer>`
        if the array's values are integer-like (i.e. that
        ``np.all(arr, np.floor(arr))``).

        .. note::

            Arrays with a float values may pass this check. Set
            ``dtype_out=int`` if the output must have an integer dtype.

    must_be_sorted : bool | dict, default: False
        :func:`Check <pyvista.core.validation.check.check_sorted>`
        if the array's values are sorted. If ``True``, the check is
        performed with default parameters:

        * ``ascending=True``: the array must be sorted in ascending order
        * ``strict=False``: sequential elements with the same value are allowed
        * ``axis=-1``: the sorting is checked along the array's last axis

        To check for descending order, enforce strict ordering, or to check
        along a different axis, use a ``dict`` with keyword arguments that
        will be passed to :func:`Check <pyvista.core.validation.check.check_sorted>`.

    must_be_in_range : Vector[float], optional
        :func:`Check <pyvista.core.validation.check.check_range>`
        if the array's values are all within a specific range. Range
        must be a vector with two elements specifying the minimum and
        maximum data values allowed, respectively. By default, the range
        endpoints are inclusive, i.e. values must be >= minimum and <=
        maximum. Use ``strict_lower_bound`` and/or ``strict_upper_bound``
        to further restrict the allowable range.

        ..note ::

            Use infinity (``np.inf`` or ``float('inf')``) to check for open
            intervals, e.g.:

            * ``[-np.inf, upper]`` to check if values are less
              than (or equal to) ``upper``
            * ``[lower, np.inf]`` to check if values are greater
              than (or equal to) ``lower``

    strict_lower_bound : bool, default: False
        Enforce a strict lower bound for the range specified by
        ``must_be_in_range``, i.e. array values must be strictly greater
        than the specified minimum.

    strict_upper_bound : bool, default: False
        Enforce a strict upper bound for the range specified by
        ``must_be_in_range``, i.e. array values must be strictly less
        than the specified maximum.

    reshape_to : int | tuple[int, ...], optional
        Reshape the output array to a new shape with :func:`numpy.reshape`.
        The shape should be compatible with the original shape. If an
        integer, then the result will be a 1-D array of that length. One
        shape dimension can be -1.

    broadcast_to : int | tuple[int, ...], optional
        Broadcast the array with :func:`numpy.broadcast_to` to a
        read-only view with the specified shape. Broadcasting is done
        after reshaping (if ``reshape_to`` is not ``None``).

    dtype_out : numpy.typing.DTypeLike, optional
        Set the data-type of the returned array. By default, the
        dtype is inferred from the input data. If ``dtype_out`` differs
        from the array's dtype, a copy of the array is made. The dtype
        of the array is set after any ``must_be_real`` or ``must_have_dtype``
        checks are made.

        .. warning::

            Setting this to a NumPy dtype (e.g. ``np.float64``) will implicitly
            set ``output_type`` to ``numpy``. Set to ``float``, ``int``, or
            ``bool`` to avoid this behavior.

        .. warning::

            Array validation can fail or result in silent integer overflow
            if ``dtype_out`` is integral and the input has infinity values.
            Consider setting ``must_be_finite=True`` for these cases.

    as_any : bool, default: True
        Allow subclasses of ``np.ndarray`` to pass through without
        making a copy.

    copy : bool, default: False
        If ``True``, a copy of the array is returned. A copy is always
        returned if the array:

        * is a nested sequence
        * is a subclass of ``np.ndarray`` and ``as_any`` is ``False``.

        A copy may also be made to satisfy ``dtype_out`` requirements.

    output_type : str | type, optional
        Convert the array to a specific type. The array may be copied.

        * ``"numpy"`` or ``np.ndarray"
        * ``"list"`` or ``list``
        * ``"tuple"`` or ``tuple``

        .. note::

            For scalar inputs, setting the output type to ``list`` or
            ``tuple`` will return a scalar, and not an actual ``list``
            or ``tuple`` object.

    name : str, default: "Array"
        Variable name to use in the error messages if any of the
        validation checks fail.

    Returns
    -------
    float | int | bool | Array[float] | Array[int] | Array[bool]
        Validated array. Returned object is:

        * an instance of ``np.ndarray`` (default), or
        * a nested ``list`` (if ``to_list=True``), or
        * a nested ``tuple`` (if ``to_tuple=True``), or
        * a number (e.g. ``int`` or ``float``) if the input is a scalar.

    Examples
    --------
    Validate a one-dimensional array has at least length two, is
    monotonically increasing (i.e. has strict ascending order), and
    is within some range.

    >>> from pyvista import validation
    >>> array_in = (1, 2, 3, 5, 8, 13)
    >>> rng = (0, 20)
    >>> validation.validate_array(
    ...     array_in,
    ...     must_have_shape=(-1),
    ...     must_have_min_length=2,
    ...     must_be_sorted=dict(strict=True),
    ...     must_be_in_range=rng,
    ... )
    (1, 2, 3, 5, 8, 13)

    """
    wrapped = _ArrayLikeWrapper(array)

    # Check dtype
    if must_be_real:
        check_real(wrapped(), name=name)
    if must_have_dtype is not None:
        check_subdtype(wrapped(), base_dtype=must_have_dtype, name=name)

    if dtype_out not in (None, float, int, bool):
        output_type = "numpy"

    # Check if re-casting is needed in case subclasses are not allowed
    rewrap_numpy = as_any is False and type(wrapped._array) is np.ndarray

    # Check if built-in types need to be cast to numpy in case broadcasting
    # or reshaping is needed (these are only supported for numpy arrays)
    do_reshape = reshape_to is not None and wrapped.shape != reshape_to
    do_broadcast = broadcast_to is not None and wrapped.shape != broadcast_to
    rewrap_builtin = isinstance(wrapped, _BuiltinWrapper) and (do_reshape or do_broadcast)

    if rewrap_numpy or rewrap_builtin or output_type in ("numpy", np.ndarray):
        wrapped = (
            _ArrayLikeWrapper(np.asanyarray(array))
            if as_any
            else _ArrayLikeWrapper(np.asarray(array))
        )

    # Check shape
    if must_have_shape is not None:
        check_shape(wrapped(), shape=must_have_shape, name=name)

    # Do reshape _after_ checking shape to prevent unexpected reshaping
    if do_reshape:
        wrapped._array = np.reshape(wrapped._array, reshape_to)

    if do_broadcast:
        wrapped._array = np.broadcast_to(wrapped._array, broadcast_to, subok=True)

    # Check length _after_ reshaping otherwise length may be wrong
    if (
        must_have_length is not None
        or must_have_min_length is not None
        or must_have_max_length is not None
    ):
        check_length(
            wrapped(),
            exact_length=must_have_length,
            min_length=must_have_min_length,
            max_length=must_have_max_length,
            allow_scalar=True,
            name=name,
        )

    # Check data values
    if must_be_nonnegative:
        check_nonnegative(wrapped(), name=name)
    # Check finite before setting dtype since dtype change can fail with inf
    if must_be_finite:
        check_finite(wrapped(), name=name)
    if must_be_integer:
        check_integer(wrapped(), strict=False, name=name)
    if must_be_in_range is not None:
        check_range(
            wrapped(),
            must_be_in_range,
            strict_lower=strict_lower_bound,
            strict_upper=strict_upper_bound,
            name=name,
        )
    if must_be_sorted:
        if isinstance(must_be_sorted, dict):
            check_sorted(wrapped(), **must_be_sorted, name=name)
        else:
            check_sorted(wrapped(), name=name)

    # Set dtype
    if dtype_out is not None:
        try:
            wrapped.change_dtype(dtype_out)
        except OverflowError as e:
            if 'cannot convert float infinity to integer' in repr(e):
                raise TypeError(
                    f"Cannot change dtype of {name} from {wrapped.dtype} to {dtype_out}.\n"
                    f"Float infinity cannot be converted to integer."
                )
    # reveal_type(wrapped())
    # Cast array to desired output
    if output_type is not None:
        if output_type in ("numpy", np.ndarray):
            return wrapped.to_numpy(array, copy)
        elif output_type in ("list", list):
            return wrapped.to_list(array, copy)
        elif output_type in ("tuple", tuple):
            return wrapped.to_tuple(array, copy)
        else:
            # Invalid type, raise error with check
            check_contains(output_type, ["numpy", "list", "tuple", np.ndarray, list, tuple])

            def _assert_never() -> NoReturn:
                raise AssertionError("Expected code to be unreachable")

            _assert_never()
    elif rewrap_builtin:
        if isinstance(array, tuple):
            return wrapped.to_tuple(array, copy)
        else:
            # to_list should cover lists as well as scalar inputs
            # reveal_type(wrapped._array)
            out = wrapped.to_list(array, copy)
            # reveal_type(out)
            # x: int
            # x = 2.0
            return out
        # reveal_type(array_out)
    else:
        if copy:
            if isinstance(array, np.ndarray):
                return np.ndarray.copy(array)
            else:
                return deepcopy(array)
        return wrapped._array
    # # reveal_type(array_out)
    # # T = TypeVar('T')
    # # def _copy(in:T)
    # if array_out is array and copy:
    #     if isinstance(array_out, np.ndarray):
    #         array_out = np.ndarray.copy(array_out)
    #     else:
    #         array_out = deepcopy(array_out)
    # return array_out


# x: List[int] = [1]
# reveal_type(validate_array(x))


def validate_axes(
    *axes: Union[Matrix[float], Vector[float]],
    normalize: bool = True,
    must_be_orthogonal: bool = True,
    must_have_orientation: Optional[str] = 'right',
    name: str = "Axes",
) -> NumpyArray[float]:
    """Validate 3D axes vectors.

    By default, the axes are normalized and checked to ensure they are orthogonal and
    have a right-handed orientation.

    Parameters
    ----------
    *axes : Matrix[float] | Vector[float]
        Axes to be validated. Axes may be specified as a single array of row vectors
        or as separate arguments for each 3-element axis vector.
        If only two vectors are given and ``must_have_orientation`` is not ``None``,
        the third vector is automatically calculated as the cross-product of the
        two vectors such that the axes have the correct orientation.

    normalize : bool, default: True
        If ``True``, the axes vectors are individually normalized to each have a norm
        of 1.

    must_be_orthogonal : bool, default: True
        Check if the axes are orthogonal. If ``True``, the cross product between any
        two axes vectors must be parallel to the third.

    must_have_orientation : str, default: 'right'
        Check if the axes have a specific orientation. If ``right``, the
        cross-product of the first axis vector with the second must have a positive
        direction. If ``left``, the direction must be negative. If ``None``, the
        orientation is not checked.

    name : str, default: "Axes"
        Variable name to use in the error messages if any of the
        validation checks fail.

    Returns
    -------
    np.ndarray
        Validated 3x3 axes array of row vectors.

    Examples
    --------
    Validate an axes array.

    >>> import numpy as np
    >>> from pyvista import validation
    >>> validation.validate_axes(np.eye(3))
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Validate individual axes vectors as a 3x3 array.

    >>> validation.validate_axes([1, 0, 0], [0, 1, 0], [0, 0, 1])
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Create a validated left-handed axes array from two vectors.

    >>> validation.validate_axes(
    ...     [1, 0, 0], [0, 1, 0], must_have_orientation='left'
    ... )
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0., -1.]])

    """
    if must_have_orientation is not None:
        check_contains(must_have_orientation, ['right', 'left'], name=f"{name} orientation")

    # Validate number of args
    num_args = len(axes)
    if num_args not in (1, 2, 3):
        raise ValueError(
            "Incorrect number of axes arguments. Number of arguments must be either:"
            "    One arg (a single array with two or three vectors),"
            "    Two args (two vectors), or"
            "    Three args (three vectors)."
        )

    # Validate axes array
    vector2: Optional[NumpyArray[float]] = None
    if num_args == 1:
        axes_array = validate_array(
            axes[0], must_have_shape=[(2, 3), (3, 3)], name=name, dtype_out=np.floating
        )
        vector0 = axes_array[0]
        vector1 = axes_array[1]
        if len(axes_array) == 3:
            vector2 = axes_array[2]
    else:
        vector0 = validate_array3(axes[0], name=f"{name} Vector[0]")
        vector1 = validate_array3(axes[1], name=f"{name} Vector[1]")
        if num_args == 3:
            vector2 = validate_array3(axes[2], name=f"{name} Vector[2]")

    if vector2 is None:
        if must_have_orientation is None:
            raise ValueError(
                f"{name} orientation must be specified when only two vectors are given."
            )
        elif must_have_orientation == 'right':
            vector2 = np.cross(vector0, vector1)
        else:
            vector2 = np.cross(vector1, vector0)
    axes_array = np.vstack((vector0, vector1, vector2))
    check_finite(axes_array, name=name)

    if np.isclose(np.dot(axes_array[0], axes_array[1]), 1) or np.isclose(
        np.dot(axes_array[0], axes_array[2]), 1
    ):
        raise ValueError(f"{name} cannot be parallel.")
    if np.any(np.all(np.isclose(axes_array, np.zeros(3)), axis=1)):
        raise ValueError(f"{name} cannot be zeros.")

    # Check orthogonality and orientation using cross products
    # Normalize axes first since norm values are needed for cross product calc
    axes_norm = axes_array / np.linalg.norm(axes_array, axis=1).reshape((3, 1))
    cross_0_1 = np.cross(axes_norm[0], axes_norm[1])
    cross_1_2 = np.cross(axes_norm[1], axes_norm[2])

    if must_be_orthogonal and not (
        (np.allclose(cross_0_1, axes_norm[2]) or np.allclose(cross_0_1, -axes_norm[2]))
        and (np.allclose(cross_1_2, axes_norm[0]) or np.allclose(cross_1_2, -axes_norm[0]))
    ):
        raise ValueError(f"{name} are not orthogonal.")

    if must_have_orientation:
        dot = np.dot(cross_0_1, axes_norm[2])
        if must_have_orientation == 'right' and dot < 0:
            raise ValueError(f"{name} do not have a right-handed orientation.")
        if must_have_orientation == 'left' and dot > 0:
            raise ValueError(f"{name} do not have a left-handed orientation.")

    return axes_norm if normalize else axes_array


def validate_transform4x4(transform: TransformLike, /, *, name="Transform"):
    """Validate transform-like input as a 4x4 ndarray.

    Parameters
    ----------
    transform : Matrix[float] | vtkTransform | vtkMatrix4x4 | vtkMatrix3x3
        Transformation matrix as a 3x3 or 4x4 array, 3x3 or 4x4 vtkMatrix,
        or as a vtkTransform.

    name : str, default: "Transform"
        Variable name to use in the error messages if any of the
        validation checks fail.

    Returns
    -------
    np.ndarray
        Validated 4x4 transformation matrix.

    See Also
    --------
    validate_transform3x3
        Similar function for 3x3 transforms.

    validate_array
        Generic array validation function.

    """
    check_string(name, name="Name")
    array = np.eye(4)  # initialize
    if isinstance(transform, _vtk.vtkMatrix4x4):
        array = _array_from_vtkmatrix(transform, shape=(4, 4))
    elif isinstance(transform, _vtk.vtkMatrix3x3):
        array[:3, :3] = _array_from_vtkmatrix(transform, shape=(3, 3))
    elif isinstance(transform, _vtk.vtkTransform):
        array = _array_from_vtkmatrix(transform.GetMatrix(), shape=(4, 4))
    else:
        try:
            valid_array = validate_array(
                transform,
                must_have_shape=[(3, 3), (4, 4)],
                must_be_finite=True,
                name=name,
                output_type='numpy',
            )
            if valid_array.shape == (3, 3):
                array[:3, :3] = valid_array
            else:
                array = valid_array
        except ValueError:
            raise TypeError(
                'Input transform must be one of:\n'
                '\tvtkMatrix4x4\n'
                '\tvtkMatrix3x3\n'
                '\tvtkTransform\n'
                '\t4x4 np.ndarray\n'
                '\t3x3 np.ndarray\n'
            )

    return array


def validate_transform3x3(
    transform: Union[Matrix[float], _vtk.vtkMatrix3x3], /, *, name="Transform"
):
    """Validate transform-like input as a 3x3 ndarray.

    Parameters
    ----------
    transform : Matrix[float] | vtkMatrix3x3
        Transformation matrix as a 3x3 array or vtkMatrix3x3.

    name : str, default: "Transform"
        Variable name to use in the error messages if any of the
        validation checks fail.

    Returns
    -------
    np.ndarray
        Validated 3x3 transformation matrix.

    See Also
    --------
    validate_transform4x4
        Similar function for 4x4 transforms.

    validate_array
        Generic array validation function.

    """
    check_string(name, name="Name")
    array = np.eye(3)  # initialize
    if isinstance(transform, _vtk.vtkMatrix3x3):
        array[:3, :3] = _array_from_vtkmatrix(transform, shape=(3, 3))
    else:
        try:
            array = validate_array(
                transform, must_have_shape=(3, 3), name=name, output_type="numpy"
            )
        except ValueError:
            raise TypeError(
                'Input transform must be one of:\n' '\tvtkMatrix3x3\n' '\t3x3 np.ndarray\n'
            )
    return array


def _array_from_vtkmatrix(
    matrix: Union[_vtk.vtkMatrix3x3, _vtk.vtkMatrix4x4],
    shape: Union[Tuple[Literal[3], Literal[3]], Tuple[Literal[4], Literal[4]]],
) -> NumpyArray[float]:
    """Convert a vtk matrix to an array."""
    array = np.zeros(shape)
    for i, j in product(range(shape[0]), range(shape[1])):
        array[i, j] = matrix.GetElement(i, j)
    return array


def validate_number(
    num: Union[_NumberType, Vector[_NumberType]], /, *, reshape=True, **kwargs
) -> _NumberType:
    """Validate a real, finite scalar number.

    By default, the number is checked to ensure it:

    * is scalar or is an array with one element
    * is a real number
    * is finite

    Parameters
    ----------
    num : float | int | Vector[float] | Vector[int]
        Number to validate.

    reshape : bool, default: True
        If ``True``, 1D arrays with 1 element are considered valid input
        and are reshaped to be 0-dimensional.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    int | float
        Validated number.

    See Also
    --------
    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a number.

    >>> from pyvista import validation
    >>> validation.validate_number(1)
    1

    1D arrays are automatically reshaped.

    >>> validation.validate_number([42.0])
    42.0

    Additional checks can be added as needed.

    >>> validation.validate_number(
    ...     10, must_be_in_range=[0, 10], must_be_integer=True
    ... )
    10

    """
    kwargs.setdefault('name', 'Number')
    kwargs.setdefault('must_be_finite', True)
    kwargs.setdefault('must_be_real', True)

    shape: Union[ShapeLike, List[ShapeLike]]
    if reshape:
        shape = [(), (1,)]
        _set_default_kwarg_mandatory(kwargs, 'reshape_to', ())
    else:
        shape = ()
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', shape)

    return validate_array(num, **kwargs)


def validate_data_range(rng: Vector[_NumberType], /, **kwargs):
    """Validate a data range.

    By default, the data range is checked to ensure:

    * it has two values
    * it has real numbers
    * the lower bound is not more than the upper bound

    Parameters
    ----------
    rng : Vector[float]
        Range to validate in the form ``(lower_bound, upper_bound)``.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    tuple
        Validated range as ``(lower_bound, upper_bound)``.

    See Also
    --------
    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a data range.

    >>> from pyvista import validation
    >>> validation.validate_data_range([-5, 5.0])
    [-5, 5.0]

    Add additional constraints if needed, e.g. to ensure the output
    only contains floats.

    >>> validation.validate_data_range([-5, 5.0], dtype_out=float)
    [-5.0, 5.0]

    """
    kwargs.setdefault('name', 'Data Range')
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', 2)
    _set_default_kwarg_mandatory(kwargs, 'must_be_sorted', True)
    return validate_array(rng, **kwargs)


def validate_arrayNx3(
    array: Union[Matrix[_NumberType], Vector[_NumberType]], /, *, reshape=True, **kwargs
) -> NumpyArray[_NumberType]:
    """Validate an array is numeric and has shape Nx3.

    The array is checked to ensure its input values:

    * have shape ``(N, 3)`` or can be reshaped to ``(N, 3)``
    * are numeric

    The returned array is formatted so that its values:

    * have shape ``(N, 3)``.

    Parameters
    ----------
    array : Vector[float] | Matrix[float]
        1D or 2D array to validate.

    reshape : bool, default: True
        If ``True``, 1D arrays with 3 elements are considered valid
        input and are reshaped to ``(1, 3)`` to ensure the output is
        two-dimensional.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    np.ndarray
        Validated array with shape ``(N, 3)``.

    See Also
    --------
    validate_arrayN
        Similar function for one-dimensional arrays.

    validate_array
        Generic array validation function.

    Examples
    --------
    Validate an Nx3 array.

    >>> from pyvista import validation
    >>> validation.validate_arrayNx3(((1, 2, 3), (4, 5, 6)))
    ((1, 2, 3), (4, 5, 6))

    One-dimensional 3-element arrays are automatically reshaped to 2D.

    >>> validation.validate_arrayNx3([1, 2, 3])
    [[1, 2, 3]]

    Add additional constraints.

    >>> validation.validate_arrayNx3(
    ...     ((1, 2, 3), (4, 5, 6)), must_be_in_range=[0, 10]
    ... )
    ((1, 2, 3), (4, 5, 6))

    """
    shape: Union[ShapeLike, List[ShapeLike]]
    if reshape:
        shape = [3, (-1, 3)]
        _set_default_kwarg_mandatory(kwargs, 'reshape_to', (-1, 3))
    else:
        shape = (-1, 3)
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', shape)

    return validate_array(array, **kwargs)


def validate_arrayN(
    array: Union[_NumberType, Vector[_NumberType], Matrix[_NumberType]],
    /,
    *,
    reshape=True,
    **kwargs,
):
    """Validate a numeric 1D array.

    The array is checked to ensure its input values:

    * have shape ``(N,)`` or can be reshaped to ``(N,)``
    * are numeric

    The returned array is formatted so that its values:

    * have shape ``(N,)``

    Parameters
    ----------
    array : float | Vector[float] | Matrix[float]
        Array-like input to validate.

    reshape : bool, default: True
        If ``True``, 0-dimensional scalars are reshaped to ``(1,)`` and 2D
        vectors with shape ``(1, N)`` are reshaped to ``(N,)`` to ensure the
        output is consistently one-dimensional. Otherwise, all scalar and
        2D inputs are not considered valid.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    np.ndarray
        Validated 1D array.

    See Also
    --------
    validate_arrayN_uintlike
        Similar function for non-negative integer arrays.

    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a 1D array with four elements.

    >>> from pyvista import validation
    >>> validation.validate_arrayN((1, 2, 3, 4))
    (1, 2, 3, 4)

    Scalar 0-dimensional values are automatically reshaped to be 1D.

    >>> validation.validate_arrayN(42.0)
    [42.0]

    2D arrays where the first dimension is unity are automatically
    reshaped to be 1D.

    >>> validation.validate_arrayN([[1, 2]])
    [1, 2]

    Add additional constraints if needed.

    >>> validation.validate_arrayN((1, 2, 3), must_have_length=3)
    (1, 2, 3)

    """
    shape: Union[ShapeLike, List[ShapeLike]]
    if reshape:
        shape = [(), (-1), (1, -1), (-1, 1)]
        _set_default_kwarg_mandatory(kwargs, 'reshape_to', (-1))
    else:
        shape = -1
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', shape)
    return validate_array(array, **kwargs)


def validate_arrayN_uintlike(
    array: Union[_NumberType, Vector[_NumberType], Matrix[_NumberType]],
    /,
    *,
    reshape=True,
    **kwargs,
):
    """Validate a numeric 1D array of non-negative (unsigned) integers.

    The array is checked to ensure its input values:

    * have shape ``(N,)`` or can be reshaped to ``(N,)``
    * are integer-like
    * are non-negative

    The returned array is formatted so that its values:

    * have shape ``(N,)``
    * have an integer data type

    Parameters
    ----------
    array : float | Vector[float] | Matrix[float]
        0D, 1D, or 2D array to validate.

    reshape : bool, default: True
        If ``True``, 0-dimensional scalars are reshaped to ``(1,)`` and 2D
        vectors with shape ``(1, N)`` are reshaped to ``(N,)`` to ensure the
        output is consistently one-dimensional. Otherwise, all scalar and
        2D inputs are not considered valid.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    np.ndarray
        Validated 1D array with non-negative integers.

    See Also
    --------
    validate_arrayN
        Similar function for numeric one-dimensional arrays.

    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a 1D array with four non-negative integer-like elements.

    >>> import numpy as np
    >>> from pyvista import validation
    >>> array = validation.validate_arrayN_uintlike((1.0, 2.0, 3.0, 4.0))
    >>> array
    (1, 2, 3, 4)

    Verify that the output data type is integral.

    >>> isinstance(array[0], int)
    True

    Scalar 0-dimensional values are automatically reshaped to be 1D.

    >>> validation.validate_arrayN_uintlike(42)
    [42]

    2D arrays where the first dimension is unity are automatically
    reshaped to be 1D.

    >>> validation.validate_arrayN_uintlike([[1, 2]])
    [1, 2]

    Add additional constraints if needed.

    >>> validation.validate_arrayN_uintlike(
    ...     (1, 2, 3), must_be_in_range=[1, 3]
    ... )
    (1, 2, 3)

    """
    # Set default dtype out but allow overriding as long as the dtype
    # is also integral
    kwargs.setdefault('dtype_out', int)
    if kwargs['dtype_out'] is not int:
        check_subdtype(kwargs['dtype_out'], np.integer)

    _set_default_kwarg_mandatory(kwargs, 'must_be_integer', True)
    _set_default_kwarg_mandatory(kwargs, 'must_be_nonnegative', True)

    return validate_arrayN(array, reshape=reshape, **kwargs)


def validate_array3(
    array: Union[_NumberType, Vector[_NumberType], Matrix[_NumberType]],
    /,
    *,
    reshape=True,
    broadcast=False,
    **kwargs,
):
    """Validate a numeric 1D array with 3 elements.

    The array is checked to ensure its input values:

    * have shape ``(3,)`` or can be reshaped to ``(3,)``
    * are numeric and real

    The returned array is formatted so that it has shape ``(3,)``.

    Parameters
    ----------
    array : float | Vector[float] | Matrix[float]
        Array to validate.

    reshape : bool, default: True
        If ``True``, 2D vectors with shape ``(1, 3)`` or ``(3, 1)`` are
        considered valid input, and are reshaped to ``(3,)`` to ensure
        the output is consistently one-dimensional.

    broadcast : bool, default: False
        If ``True``, scalar values or 1D arrays with a single element
        are considered valid input and the single value is broadcast to
        a length 3 array.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    np.ndarray
        Validated 1D array with 3 elements.

    See Also
    --------
    validate_number
        Similar function for a single number.

    validate_arrayN
        Similar function for one-dimensional arrays.

    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a 1D array with three elements.

    >>> from pyvista import validation
    >>> validation.validate_array3((1, 2, 3))
    (1, 2, 3)

    2D 3-element arrays are automatically reshaped to be 1D.

    >>> validation.validate_array3([[1, 2, 3]])
    [1, 2, 3]

    Scalar 0-dimensional values can be automatically broadcast as
    a 3-element 1D array.

    >>> validation.validate_array3(42.0, broadcast=True)
    [42.0, 42.0, 42.0]

    Add additional constraints if needed.

    >>> validation.validate_array3((1, 2, 3), must_be_nonnegative=True)
    (1, 2, 3)

    """
    shape: List[tuple[int, ...]]
    shape = [(3,)]
    if reshape:
        shape.append((1, 3))
        shape.append((3, 1))
        _set_default_kwarg_mandatory(kwargs, 'reshape_to', (-1))
    if broadcast:
        shape.append(())  # allow 0D scalars
        shape.append((1,))  # 1D 1-element vectors
        _set_default_kwarg_mandatory(kwargs, 'broadcast_to', (3,))
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', shape)

    return validate_array(array, **kwargs)


def _set_default_kwarg_mandatory(kwargs: Dict[str, Any], key: str, default: Any):
    """Set a kwarg and raise ValueError if not set to its default value."""
    val = kwargs.pop(key, default)
    if val != default:
        calling_fname = inspect.stack()[1].function
        msg = (
            f"Parameter '{key}' cannot be set for function `{calling_fname}`.\n"
            f"Its value is automatically set to `{default}`."
        )
        raise ValueError(msg)
    kwargs[key] = default
