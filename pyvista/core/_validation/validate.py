"""Functions that validate input and return a standard representation.

.. versionadded:: 0.43.0

A ``validate`` function typically:

* Uses :py:mod:`~pyvista.core._validation.check` functions to
  check the type and/or value of input arguments.
* Applies (optional) constraints, e.g. input or output must have a
  specific length, shape, type, data-type, etc.
* Accepts many different input types or values and standardizes the
  output as a single representation with known properties.

"""

from __future__ import annotations

import inspect
from itertools import product
import reprlib
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import numpy as np

from pyvista.core._validation import check_contains
from pyvista.core._validation import check_finite
from pyvista.core._validation import check_integer
from pyvista.core._validation import check_length
from pyvista.core._validation import check_ndim
from pyvista.core._validation import check_nonnegative
from pyvista.core._validation import check_range
from pyvista.core._validation import check_real
from pyvista.core._validation import check_shape
from pyvista.core._validation import check_sorted
from pyvista.core._validation import check_string
from pyvista.core._validation import check_subdtype
from pyvista.core._validation._cast_array import _cast_to_numpy
from pyvista.core._validation._cast_array import _cast_to_tuple
from pyvista.core._vtk_core import vtkMatrix3x3
from pyvista.core._vtk_core import vtkMatrix4x4
from pyvista.core._vtk_core import vtkTransform

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from pyvista.core._typing_core import ArrayLike
    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import RotationLike
    from pyvista.core._typing_core import TransformLike
    from pyvista.core._typing_core import VectorLike
    from pyvista.core._typing_core._array_like import NumpyArray
    from pyvista.plotting._typing import ColorLike
    from pyvista.plotting.colors import Color

    from .check import _ShapeLike


def validate_array(
    arr: float | ArrayLike[float],
    /,
    *,
    must_have_shape: _ShapeLike | list[_ShapeLike] | None = None,
    must_have_ndim: int | VectorLike[int] | None = None,
    must_have_dtype: npt.DTypeLike | None = None,
    must_have_length: int | VectorLike[int] | None = None,
    must_have_min_length: int | None = None,
    must_have_max_length: int | None = None,
    must_be_nonnegative: bool = False,
    must_be_finite: bool = False,
    must_be_real: bool = True,
    must_be_integer: bool = False,
    must_be_sorted: bool | dict[str, Any] = False,
    must_be_in_range: VectorLike[float] | None = None,
    strict_lower_bound: bool = False,
    strict_upper_bound: bool = False,
    reshape_to: int | tuple[int, ...] | None = None,
    broadcast_to: int | tuple[int, ...] | None = None,
    dtype_out: npt.DTypeLike = None,
    as_any: bool = True,
    copy: bool = False,
    to_list: bool = False,
    to_tuple: bool = False,
    name: str = 'Array',
):
    """Check and validate a numeric array meets specific requirements.

    Validate an array to ensure it is numeric, has a specific shape,
    data-type, and/or has values that meet specific
    requirements such as being sorted, integer-like, or finite.

    The array's output can also be reshaped or broadcast, cast as a
    nested tuple or list array, or cast to a specific data type.

    See Also
    --------
    validate_number
        Specialized function for single numbers.

    validate_array3
        Specialized function for 3-element arrays.

    validate_arrayN
        Specialized function for one-dimensional arrays.

    validate_arrayNx3
        Specialized function for Nx3 dimensional arrays.

    validate_data_range
        Specialized function for data ranges.

    Parameters
    ----------
    arr : array_like
        Array to be validated, in any form that can be converted to
        a :class:`np.ndarray`. This includes lists, lists of tuples, tuples,
        tuples of tuples, tuples of lists and ndarrays.

    must_have_shape : int | tuple[int, ...] | list[int, tuple[int, ...]], optional
        :func:`Check <pyvista.core._validation.check.check_shape>`
        if the array has a specific shape. Specify a single shape
        or a ``list`` of any allowable shapes. If an integer, the array must
        be 1-dimensional with that length. Use a value of ``-1`` for any
        dimension where its size is allowed to vary. Use ``()`` to allow
        scalar values (i.e. 0-dimensional). Set to ``None`` if the array
        can have any shape (default).

    must_have_ndim : int | VectorLike[int], optional
        :func:`Check <pyvista.core._validation.check.check_ndim>` if
        the array has the specified number of dimension(s). Specify a
        single dimension or a sequence of allowable dimensions. If a
        sequence, the array must have at least one of the specified
        number of dimensions.

    must_have_dtype : DTypeLike | list[DTypeLike, ...], optional
        :func:`Check <pyvista.core._validation.check.check_subdtype>`
        if the array's data-type has the given dtype. Specify a
        :class:`np.dtype` object or dtype-like base class which the
        array's data must be a subtype of. If a ``list``, the array's data
        must be a subtype of at least one of the specified dtypes.

    must_have_length : int | VectorLike[int], optional
        :func:`Check <pyvista.core._validation.check.check_length>`
        if the array has the given length. If multiple values are given,
        the array's length must match one of the values.

        .. note ::

            The array's length is determined after reshaping the array
            (if ``reshape`` is not ``None``) and after broadcasting (if
            ``broadcast_to`` is not ``None``). Therefore, the values of
            `length`` should take the array's new shape into
            consideration if applicable.

    must_have_min_length : int, optional
        :func:`Check <pyvista.core._validation.check.check_length>`
        if the array's length is this value or greater.

    must_have_max_length : int, optional
        :func:`Check <pyvista.core._validation.check.check_length>`
        if the array' length is this value or less.

    must_be_nonnegative : bool, default: False
        :func:`Check <pyvista.core._validation.check.check_nonnegative>`
        if all elements of the array are nonnegative.

    must_be_finite : bool, default: False
        :func:`Check <pyvista.core._validation.check.check_finite>`
        if all elements of the array are finite, i.e. not ``infinity``
        and not Not a Number (``NaN``).

    must_be_real : bool, default: True
        :func:`Check <pyvista.core._validation.check.check_real>`
        if the array has real numbers, i.e. its data type is integer or
        floating.

    must_be_integer : bool, default: False
        :func:`Check <pyvista.core._validation.check.check_integer>`
        if the array's values are integer-like (i.e. that
        ``np.all(arr, np.floor(arr))``).

    must_be_sorted : bool | dict, default: False
        :func:`Check <pyvista.core._validation.check.check_sorted>`
        if the array's values are sorted. If ``True``, the check is
        performed with default parameters:

        * ``ascending=True``: the array must be sorted in ascending order
        * ``strict=True``: sequential elements with the same value are allowed
        * ``axis=-1``: the sorting is checked along the array's last axis

        To check for descending order, enforce strict ordering, or to check
        along a different axis, use a ``dict`` with keyword arguments that
        will be passed to ``check_sorted``.

    must_be_in_range : VectorLike[float], optional
        :func:`Check <pyvista.core._validation.check.check_range>`
        if the array's values are all within a specific range. Range
        must be array-like with two elements specifying the minimum and
        maximum data values allowed, respectively. By default, the range
        endpoints are inclusive, i.e. values must be >= minimum and <=
        maximum. Use ``strict_lower_bound`` and/or ``strict_upper_bound``
        to further restrict the allowable range.

        ..note ::

            Use ``np.inf`` to check for open intervals, e.g.:

            * ``[-np.inf, upper_bound]`` to check if values are less
              than (or equal to)  ``upper_bound``
            * ``[lower_bound, np.inf]`` to check if values are greater
              than (or equal to) ``lower_bound``

    strict_lower_bound : bool, default: False
        Enforce a strict lower bound for the range specified by
        ``must_be_in_range``, i.e. array values must be strictly greater
        than the specified minimum.

    strict_upper_bound : bool, default: False
        Enforce a strict upper bound for the range specified by
        ``must_be_in_range``, i.e. array values must be strictly less
        than the specified maximum.

    reshape_to : int | tuple[int, ...], optional
        Reshape the output array to a new shape with :func:`np.reshape`.
        The shape should be compatible with the original shape. If an
        integer, then the result will be a 1-D array of that length. One
        shape dimension can be -1.

    broadcast_to : int | tuple[int, ...], optional
        Broadcast the array with :func:`np.broadcast_to` to a
        read-only view with the specified shape. Broadcasting is done
        after reshaping (if ``reshape_to`` is not ``None``).

    dtype_out : DTypeLike, optional
        Set the data-type of the returned array. By default, the
        dtype is inferred from the input data.

    as_any : bool, default: True
        Allow subclasses of ``np.ndarray`` to pass through without
        making a copy.

    copy : bool, default: False
        If ``True``, a copy of the array is returned. A copy is always
        returned if the array:

        * is a nested sequence
        * is a subclass of ``np.ndarray`` and ``as_any`` is ``False``.

        A copy may also be made to satisfy ``dtype_out`` requirements.

    to_list : bool, default: False
        Return the validated array as a ``list`` or nested ``list``. Scalar
        values are always returned as a ``Number``  (i.e. ``int`` or ``float``).
        Has no effect if ``to_tuple=True``.

    to_tuple : bool, default: False
        Return the validated array as a ``tuple`` or nested ``tuple``. Scalar
        values are always returned as a ``Number``  (i.e. ``int`` or ``float``).

    name : str, default: "Array"
        Variable name to use in the error messages if any of the
        validation checks fail.

    Returns
    -------
    array_like
        Validated array. Returned object is:

        * an instance of ``np.ndarray`` (default), or
        * a nested ``list`` (if ``to_list=True``), or
        * a nested ``tuple`` (if ``to_tuple=True``), or
        * a ``Number`` (i.e. ``int`` or ``float``) if the input is a scalar.

    Examples
    --------
    Validate a one-dimensional array has at least length two, is
    monotonically increasing (i.e. has strict ascending order), and
    is within some range.

    >>> from pyvista import _validation
    >>> array_in = (1, 2, 3, 5, 8, 13)
    >>> rng = (0, 20)
    >>> _validation.validate_array(
    ...     array_in,
    ...     must_have_shape=(-1),
    ...     must_have_min_length=2,
    ...     must_be_sorted=dict(strict=True),
    ...     must_be_in_range=rng,
    ... )
    array([ 1,  2,  3,  5,  8, 13])

    """
    arr_out = _cast_to_numpy(arr, as_any=as_any, copy=copy)

    # Check type
    if must_be_real:
        check_real(arr_out, name=name)

    if must_have_dtype is not None:
        check_subdtype(arr_out, must_have_dtype, name=name)

    # Check shape
    if must_have_shape is not None:
        check_shape(arr_out, must_have_shape, name=name)
    if must_have_ndim is not None:
        check_ndim(arr_out, ndim=must_have_ndim, name=name)

    # Do reshape _after_ checking shape to prevent unexpected reshaping
    if reshape_to is not None and arr_out.shape != reshape_to:
        arr_out = arr_out.reshape(reshape_to)

    if broadcast_to is not None and arr_out.shape != broadcast_to:
        arr_out = np.broadcast_to(arr_out, broadcast_to, subok=True)

    # Check length _after_ reshaping otherwise length may be wrong
    if (
        must_have_length is not None
        or must_have_min_length is not None
        or must_have_max_length is not None
    ):
        check_length(
            arr,
            exact_length=must_have_length,
            min_length=must_have_min_length,
            max_length=must_have_max_length,
            allow_scalar=True,
            name=name,
        )

    # Check data values
    if must_be_nonnegative:
        check_nonnegative(arr_out, name=name)
    if must_be_finite:
        check_finite(arr_out, name=name)
    if must_be_integer:
        check_integer(arr_out, strict=True, name=name)
    if must_be_in_range is not None:
        check_range(
            arr_out,
            must_be_in_range,
            strict_lower=strict_lower_bound,
            strict_upper=strict_upper_bound,
            name=name,
        )
    if must_be_sorted:
        if isinstance(must_be_sorted, dict):
            check_sorted(arr_out, **must_be_sorted, name=name)
        else:
            check_sorted(arr_out, name=name)

    # Process output
    if dtype_out is not None:
        # Copy was done earlier, so don't do it again here
        arr_out = arr_out.astype(dtype_out, copy=False)
    if to_tuple:
        return _cast_to_tuple(arr_out)
    if to_list:
        return arr_out.tolist()
    return arr_out


def validate_axes(
    *axes: VectorLike[float] | MatrixLike[float],
    normalize: bool = True,
    must_be_orthogonal: bool = True,
    must_have_orientation: Literal['right', 'left'] | None = 'right',
    name: str = 'Axes',
):
    """Validate 3D axes vectors.

    By default, the axes are normalized and checked to ensure they are orthogonal and
    have a right-handed orientation.

    Parameters
    ----------
    *axes : VectorLike[float] | MatrixLike[float]
        Axes to be validated. Axes may be specified as a single argument of a 3x3
        array of row vectors or as separate arguments for each 3-element axis vector.
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
    >>> from pyvista import _validation
    >>> _validation.validate_axes(np.eye(3))
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Validate individual axes vectors as a 3x3 array.

    >>> _validation.validate_axes([1, 0, 0], [0, 1, 0], [0, 0, 1])
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Create a validated left-handed axes array from two vectors.

    >>> _validation.validate_axes(
    ...     [1, 0, 0], [0, 1, 0], must_have_orientation='left'
    ... )
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0., -1.]])

    """
    # Validate number of args
    check_length(axes, exact_length=[1, 2, 3], name=f'{name} arguments')
    if must_have_orientation is not None:
        check_contains(
            ['right', 'left'],
            must_contain=must_have_orientation,
            name=f'{name} orientation',
        )
    elif len(axes) == 2:
        msg = f'{name} orientation must be specified when only two vectors are given.'
        raise ValueError(msg)

    # Validate axes array
    if len(axes) == 1:
        axes_array = validate_array(axes[0], must_have_shape=(3, 3), name=name)
    else:
        axes_array = np.zeros((3, 3))
        axes_array[0] = validate_array3(axes[0], name=f'{name} Vector[0]')
        axes_array[1] = validate_array3(axes[1], name=f'{name} Vector[1]')
        if len(axes) == 3:
            axes_array[2] = validate_array3(axes[2], name=f'{name} Vector[2]')
        elif must_have_orientation == 'right':
            axes_array[2] = np.cross(axes_array[0], axes_array[1])
        else:
            axes_array[2] = np.cross(axes_array[1], axes_array[0])
    check_finite(axes_array, name=name)

    if np.isclose(np.dot(axes_array[0], axes_array[1]), 1) or np.isclose(
        np.dot(axes_array[0], axes_array[2]),
        1,
    ):
        msg = f'{name} cannot be parallel.'
        raise ValueError(msg)
    if np.any(np.all(np.isclose(axes_array, np.zeros(3)), axis=1)):
        msg = f'{name} cannot be zeros.'
        raise ValueError(msg)

    # Check orthogonality and orientation using cross products
    # Normalize axes first since norm values are needed for cross product calc
    axes_norm = axes_array / np.linalg.norm(axes_array, axis=1).reshape((3, 1))
    cross_0_1 = np.cross(axes_norm[0], axes_norm[1])
    cross_1_2 = np.cross(axes_norm[1], axes_norm[2])

    if must_be_orthogonal and not (
        (np.allclose(cross_0_1, axes_norm[2]) or np.allclose(cross_0_1, -axes_norm[2]))
        and (np.allclose(cross_1_2, axes_norm[0]) or np.allclose(cross_1_2, -axes_norm[0]))
    ):
        msg = f'{name} are not orthogonal.'
        raise ValueError(msg)

    if must_have_orientation:
        dot = np.dot(cross_0_1, axes_norm[2])
        if must_have_orientation == 'right' and dot < 0:
            msg = f'{name} do not have a right-handed orientation.'
            raise ValueError(msg)
        if must_have_orientation == 'left' and dot > 0:
            msg = f'{name} do not have a left-handed orientation.'
            raise ValueError(msg)

    if normalize:
        return axes_norm
    return axes_array


def validate_rotation(
    rotation: RotationLike,
    must_have_handedness: Literal['right', 'left'] | None = None,
    *,
    tolerance: float = 1e-6,
    name: str = 'Rotation',
):
    """Validate a rotation as a 3x3 matrix.

    The rotation is valid if it is orthogonal and has a determinant
    of ``1`` (right-handed or "proper" rotation) or ``-1`` (left-handed or "improper"
    rotation). By default, right- and left-handed rotations are allowed.
    Use ``must_have_handedness`` to restrict the handedness.

    Parameters
    ----------
    rotation : RotationLike
        3x3 rotation matrix or a SciPy ``Rotation`` object.

    must_have_handedness : 'right' | 'left' | None, default: None
        Check if the rotation has a specific handedness. If ``right``, the
        determinant must be ``1``. If ``left``, the determinant must be ``-1``.
        By default, either handedness is allowed.

    tolerance : float, default: 1e-6
        Tolerance used for checking orthogonality.

    name : str, default: "Rotation"
        Variable name to use in the error messages if any of the
        validation checks fail.

    Returns
    -------
    np.ndarray
        Validated 3x3 rotation matrix.

    Examples
    --------
    Validate a rotation matrix. The identity matrix is used as a toy example.

    >>> import numpy as np
    >>> from pyvista import _validation
    >>> rotation = np.eye(3)
    >>> _validation.validate_rotation(rotation)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    By default, left-handed rotations (which include reflections) are allowed.

    >>> rotation *= -1  # Add reflections
    >>> _validation.validate_rotation(rotation)
    array([[-1., -0., -0.],
           [-0., -1., -0.],
           [-0., -0., -1.]])

    """
    check_contains(
        ['right', 'left', None], must_contain=must_have_handedness, name='must_have_handedness'
    )

    rotation_matrix = validate_transform3x3(rotation, name=name)
    # Check orthogonality
    # A matrix M is orthogonal if the product of M with its transpose is the identity
    norm_diff = np.linalg.norm(rotation_matrix @ rotation_matrix.T - np.eye(3), ord='fro')
    if not norm_diff < tolerance:
        msg = f'{name} is not valid. Rotation must be orthogonal.'
        raise ValueError(msg)

    if must_have_handedness is not None:
        det = np.linalg.det(rotation_matrix)
        if must_have_handedness == 'right' and not det > 0:
            msg = (
                f'{name} has incorrect handedness. Expected a right-handed rotation, but got a '
                f'left-handed rotation instead.'
            )
            raise ValueError(msg)
        elif must_have_handedness == 'left' and not det < 0:
            msg = (
                f'{name} has incorrect handedness. Expected a left-handed rotation, but got a '
                f'right-handed rotation instead.'
            )
            raise ValueError(msg)

    return rotation_matrix


def validate_transform4x4(
    transform: TransformLike, /, *, must_be_finite: bool = True, name: str = 'Transform'
) -> NumpyArray[float]:
    """Validate transform-like input as a 4x4 ndarray.

    Parameters
    ----------
    transform : TransformLike
        Transformation matrix as a 3x3 or 4x4 array, :vtk:`vtkMatrix3x3` or
        :vtk:`vtkMatrix4x4`, :vtk:`vtkTransform`, or a SciPy ``Rotation`` instance.
        If the input is 3x3, the array is padded using a 4x4 identity matrix.

    must_be_finite : bool, default: True
        :func:`Check <pyvista.core._validation.check.check_finite>`
        if all elements of the array are finite, i.e. not ``infinity``
        and not Not a Number (``NaN``).

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
    check_string(name, name='Name')
    try:
        arr = np.eye(4)  # initialize
        arr[:3, :3] = validate_transform3x3(transform, must_be_finite=must_be_finite, name=name)
    except (ValueError, TypeError):
        if isinstance(transform, vtkMatrix4x4):
            arr = _array_from_vtkmatrix(transform, shape=(4, 4))  # type: ignore[assignment]
        elif isinstance(transform, vtkTransform):
            arr = _array_from_vtkmatrix(transform.GetMatrix(), shape=(4, 4))  # type: ignore[assignment]
        else:
            try:
                arr = validate_array(
                    transform,  # type: ignore[arg-type]
                    must_have_shape=[(3, 3), (4, 4)],
                    must_be_finite=must_be_finite,
                    name=name,
                )
            except TypeError:
                msg = (
                    'Input transform must be one of:\n'
                    '\tvtkMatrix4x4\n'
                    '\tvtkMatrix3x3\n'
                    '\tvtkTransform\n'
                    '\t4x4 np.ndarray\n'
                    '\t3x3 np.ndarray\n'
                    '\tscipy.spatial.transform.Rotation\n'
                    f'Got {reprlib.repr(transform)} with type {type(transform)} instead.',
                )
                raise TypeError(msg)

    return arr


def validate_transform3x3(
    transform: TransformLike, /, *, must_be_finite: bool = True, name: str = 'Transform'
):
    """Validate transform-like input as a 3x3 ndarray.

    Parameters
    ----------
    transform : RotationLike
        Transformation matrix as a 3x3 array, vtk matrix, or a SciPy ``Rotation``
        instance.

        .. note::

           Although ``RotationLike`` inputs are accepted, no checks are done
           to verify that the transformation is actually a rotation.
           Therefore, any 3x3 transformation is acceptable.

    must_be_finite : bool, default: True
        :func:`Check <pyvista.core._validation.check.check_finite>`
        if all elements of the array are finite, i.e. not ``infinity``
        and not Not a Number (``NaN``).

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
    check_string(name, name='Name')
    if isinstance(transform, vtkMatrix3x3):
        return _array_from_vtkmatrix(transform, shape=(3, 3))
    else:
        try:
            return validate_array(
                transform,  # type: ignore[arg-type]
                must_have_shape=(3, 3),
                must_be_finite=must_be_finite,
                name=name,
            )
        except ValueError:
            pass
        except TypeError:
            try:
                from scipy.spatial.transform import Rotation  # noqa: PLC0415
            except ModuleNotFoundError:  # pragma: no cover
                pass
            else:
                if isinstance(transform, Rotation):
                    # Get matrix output and try validating again
                    return validate_transform3x3(
                        transform.as_matrix(), must_be_finite=must_be_finite, name=name
                    )

    error_message = (
        f'Input transform must be one of:\n'
        '\tvtkMatrix3x3\n'
        '\t3x3 np.ndarray\n'
        '\tscipy.spatial.transform.Rotation\n'
        f'Got {reprlib.repr(transform)} with type {type(transform)} instead.'
    )
    raise TypeError(error_message)


def _array_from_vtkmatrix(
    matrix: vtkMatrix3x3 | vtkMatrix4x4,
    shape: tuple[Literal[3], Literal[3]] | tuple[Literal[4], Literal[4]],
) -> NumpyArray[float]:
    """Convert a vtk matrix to an array."""
    array = np.zeros(shape)
    for i, j in product(range(shape[0]), range(shape[1])):
        array[i, j] = matrix.GetElement(i, j)
    return array


def validate_number(num: float, /, *, reshape: bool = True, **kwargs):
    """Validate a real, finite number.

    By default, the number is checked to ensure it:

    * is scalar or is an array which can be reshaped as a scalar
    * is a real number
    * is finite

    Parameters
    ----------
    num : float
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

    >>> from pyvista import _validation
    >>> _validation.validate_number(1)
    1

    1D arrays are automatically reshaped.

    >>> _validation.validate_number([42.0])
    42.0

    Additional checks can be added as needed.

    >>> _validation.validate_number(
    ...     10, must_be_in_range=[0, 10], must_be_integer=True
    ... )
    10

    """
    kwargs.setdefault('name', 'Number')
    kwargs.setdefault('to_list', True)
    kwargs.setdefault('must_be_finite', True)

    if reshape:
        shape = [(), (1,)]
        _set_default_kwarg_mandatory(kwargs, 'reshape_to', ())
    else:
        shape = ()  # type: ignore[assignment]
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', shape)

    return validate_array(num, **kwargs)


def validate_data_range(rng: VectorLike[float], /, **kwargs):
    """Validate a data range.

    By default, the data range is checked to ensure:

    * it has two values
    * it has real numbers
    * the lower bound is not more than the upper bound

    Parameters
    ----------
    rng : VectorLike[float]
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

    >>> from pyvista import _validation
    >>> _validation.validate_data_range([-5, 5])
    (-5, 5)

    Add additional constraints if needed.

    >>> _validation.validate_data_range([0, 1.0], must_be_nonnegative=True)
    (0.0, 1.0)

    """
    kwargs.setdefault('name', 'Data Range')
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', 2)
    _set_default_kwarg_mandatory(kwargs, 'must_be_sorted', True)
    if 'to_list' not in kwargs:
        kwargs.setdefault('to_tuple', True)
    return validate_array(rng, **kwargs)


def validate_arrayNx3(  # noqa: N802
    arr: VectorLike[float] | MatrixLike[float], /, *, reshape: bool = True, **kwargs
):
    """Validate an array is numeric and has shape Nx3.

    The array is checked to ensure its input values:

    * have shape ``(N, 3)`` or can be reshaped to ``(N, 3)``
    * are numeric

    The returned array is formatted so that its values:

    * have shape ``(N, 3)``.

    Parameters
    ----------
    arr : VectorLike[float] | MatrixLike[float]
        Array to validate.

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

    >>> from pyvista import _validation
    >>> _validation.validate_arrayNx3(((1, 2, 3), (4, 5, 6)))
    array([[1, 2, 3],
           [4, 5, 6]])

    One-dimensional 3-element arrays are automatically reshaped to 2D.

    >>> _validation.validate_arrayNx3([1, 2, 3])
    array([[1, 2, 3]])

    Add additional constraints.

    >>> _validation.validate_arrayNx3(
    ...     ((1, 2, 3), (4, 5, 6)), must_be_in_range=[0, 10]
    ... )
    array([[1, 2, 3],
           [4, 5, 6]])

    """
    if reshape:
        shape = [3, (-1, 3)]
        _set_default_kwarg_mandatory(kwargs, 'reshape_to', (-1, 3))
    else:
        shape = (-1, 3)  # type: ignore[assignment]
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', shape)

    return validate_array(arr, **kwargs)


def validate_arrayN(arr: float | VectorLike[float], /, *, reshape: bool = True, **kwargs):  # noqa: N802
    """Validate a numeric 1D array.

    The array is checked to ensure its input values:

    * have shape ``(N,)`` or can be reshaped to ``(N,)``
    * are numeric

    The returned array is formatted so that its values:

    * have shape ``(N,)``

    Parameters
    ----------
    arr : VectorLike[float]
        Array to validate.

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
    validate_arrayN_unsigned
        Similar function for non-negative integer arrays.

    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a 1D array with four elements.

    >>> from pyvista import _validation
    >>> _validation.validate_arrayN((1, 2, 3, 4))
    array([1, 2, 3, 4])

    Scalar 0-dimensional values are automatically reshaped to be 1D.

    >>> _validation.validate_arrayN(42.0)
    array([42.0])

    2D arrays where the first dimension is unity are automatically
    reshaped to be 1D.

    >>> _validation.validate_arrayN([[1, 2]])
    array([1, 2])

    Add additional constraints if needed.

    >>> _validation.validate_arrayN((1, 2, 3), must_have_length=3)
    array([1, 2, 3])

    """
    if reshape:
        shape = [(), (-1), (1, -1)]
        _set_default_kwarg_mandatory(kwargs, 'reshape_to', (-1))
    else:
        shape = -1  # type: ignore[assignment]
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', shape)
    return validate_array(arr, **kwargs)


def validate_arrayN_unsigned(  # noqa: N802
    arr: VectorLike[float], /, *, reshape: bool = True, **kwargs
) -> NumpyArray[int]:
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
    arr : VectorLike[float]
        Array to validate.

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
    >>> from pyvista import _validation
    >>> arr = _validation.validate_arrayN_unsigned((1.0, 2.0, 3.0, 4.0))
    >>> arr
    array([1, 2, 3, 4])

    Verify that the output data type is integral.

    >>> np.issubdtype(arr.dtype, int)
    True

    Scalar 0-dimensional values are automatically reshaped to be 1D.

    >>> _validation.validate_arrayN_unsigned(42)
    array([42])

    2D arrays where the first dimension is unity are automatically
    reshaped to be 1D.

    >>> _validation.validate_arrayN_unsigned([[1, 2]])
    array([1, 2])

    Add additional constraints if needed.

    >>> _validation.validate_arrayN_unsigned((1, 2, 3), must_be_in_range=[1, 3])
    array([1, 2, 3])

    """
    # Set default dtype out but allow overriding as long as the dtype
    # is also integral
    kwargs.setdefault('dtype_out', int)
    if kwargs['dtype_out'] is not int:
        check_subdtype(kwargs['dtype_out'], np.integer)

    _set_default_kwarg_mandatory(kwargs, 'must_be_integer', True)
    _set_default_kwarg_mandatory(kwargs, 'must_be_nonnegative', True)

    return validate_arrayN(arr, reshape=reshape, **kwargs)


def validate_array3(
    arr: float | VectorLike[float] | MatrixLike[float],
    /,
    *,
    reshape: bool = True,
    broadcast: bool = False,
    **kwargs,
):
    """Validate a numeric 1D array with 3 elements.

    The array is checked to ensure its input values:

    * have shape ``(3,)`` or can be reshaped to ``(3,)``
    * are numeric and real

    The returned array is formatted so that it has shape ``(3,)``.

    Parameters
    ----------
    arr : float | VectorLike[float] | MatrixLike[float]
        Array to validate.

    reshape : bool, default: True
        If ``True``, 2D vectors with shape ``(1, 3)`` are considered valid
        input, and are reshaped to ``(3,)`` to ensure the output is
        consistently one-dimensional.

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

    >>> from pyvista import _validation
    >>> _validation.validate_array3((1, 2, 3))
    array([1, 2, 3])

    2D 3-element arrays are automatically reshaped to be 1D.

    >>> _validation.validate_array3([[1, 2, 3]])
    array([1, 2, 3])

    Scalar 0-dimensional values can be automatically broadcast as
    a 3-element 1D array.

    >>> _validation.validate_array3(42.0, broadcast=True)
    array([42.0, 42.0, 42.0])

    Add additional constraints if needed.

    >>> _validation.validate_array3((1, 2, 3), must_be_nonnegative=True)
    array([1, 2, 3])

    """
    shape = [(3,)]
    if reshape:
        shape.append((1, 3))  # type: ignore[arg-type]
        shape.append((3, 1))  # type: ignore[arg-type]
        _set_default_kwarg_mandatory(kwargs, 'reshape_to', (-1))
    if broadcast:
        shape.append(())  # type: ignore[arg-type] # allow 0D scalars
        shape.append((1,))  # 1D 1-element vectors
        _set_default_kwarg_mandatory(kwargs, 'broadcast_to', (3,))
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', shape)

    return validate_array(arr, **kwargs)


def _set_default_kwarg_mandatory(kwargs: dict[str, Any], key: str, default: Any) -> None:
    """Set a kwarg and raise ValueError if not set to its default value."""
    val = kwargs.pop(key, default)
    if val != default:
        calling_fname = inspect.stack()[1].function
        msg = (
            f"Parameter '{key}' cannot be set for function `{calling_fname}`.\n"
            f'Its value is automatically set to `{default}`.'
        )
        raise ValueError(msg)
    kwargs[key] = default


def validate_dimensionality(
    dimensionality: Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D'] | VectorLike[int],
    /,
    *,
    reshape: bool = True,
    **kwargs,
) -> int:
    """Validate a dimensionality.

    By default, the dimensionality is checked to ensure it:

    * is scalar or is an array which can be reshaped as a scalar
    * is an integer in the inclusive range ``[0, 3]``
    * or is a valid alias among ``'0D'``, ``'1D'``, ``'2D'``, or ``'3D'``

    Parameters
    ----------
    dimensionality : Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D'] | ArrayLike
        Number to validate.

    reshape : bool, default: True
        If ``True``, 1D arrays with 1 element are considered valid input
        and are reshaped to be 0-dimensional.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    int
        Validated dimensionality.

    Examples
    --------
    Validate a dimensionality.

    >>> from pyvista import _validation
    >>> _validation.validate_dimensionality('1D')
    1

    1D arrays are automatically reshaped.

    >>> _validation.validate_dimensionality([3])
    3

    """
    kwargs.setdefault('name', 'Dimensionality')
    kwargs.setdefault('to_list', True)
    kwargs.setdefault('must_be_finite', True)
    kwargs.setdefault('must_be_in_range', [0, 3])

    dimensionality_as_array = np.asarray(dimensionality)
    if np.issubdtype(dimensionality_as_array.dtype, str):
        dimensionality_as_array = np.char.replace(dimensionality_as_array, 'D', '')

    try:
        dimensionality_as_array = dimensionality_as_array.astype(np.int64)
    except ValueError:
        msg = (
            f'`{dimensionality}` is not a valid dimensionality.'
            ' Use one of [0, 1, 2, 3, "0D", "1D", "2D", "3D"].'
        )
        raise ValueError(msg)

    if reshape:
        shape = [(), (1,)]
        _set_default_kwarg_mandatory(kwargs, 'reshape_to', ())
    else:
        shape = ()  # type: ignore[assignment]
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', shape)

    return validate_array(dimensionality_as_array, **kwargs)


def _validate_color_sequence(
    color: ColorLike | Sequence[ColorLike],
    n_colors: int | None = None,
) -> tuple[Color, ...]:
    """Validate a color sequence.

    If `n_colors` is specified, the output will have `n` colors. For single-color
    inputs, the color is copied and a sequence of `n` identical colors is returned.
    For inputs with multiple colors, the number of colors in the input must
    match `n_colors`.

    If `n_colors` is None, no broadcasting or length-checking is performed.
    """
    from pyvista.plotting.colors import Color  # noqa: PLC0415

    try:
        # Assume we have one color
        color_list = [Color(color)]
        n_colors = 1 if n_colors is None else n_colors
        return tuple(color_list * n_colors)
    except ValueError:
        if isinstance(color, (tuple, list)):
            try:
                color_list = [_validate_color_sequence(c, n_colors=1)[0] for c in color]
                if len(color_list) == 1:
                    n_colors = 1 if n_colors is None else n_colors
                    color_list = color_list * n_colors

                # Only return if we have the correct number of colors
                if n_colors is None or len(color_list) == n_colors:
                    return tuple(color_list)
            except ValueError:
                pass
    n_colors_str = f' {n_colors} ' if n_colors else ' '
    msg = (
        f'Invalid color(s):\n'
        f'\t{color}\n'
        f'Input must be a single ColorLike color '
        f'or a sequence of{n_colors_str}ColorLike colors.'
    )
    raise ValueError(msg)
