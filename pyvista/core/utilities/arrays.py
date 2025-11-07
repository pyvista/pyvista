"""Internal array utilities."""

from __future__ import annotations

from collections import UserDict
from collections.abc import Sequence
import enum
from itertools import product
import json
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast
from typing import overload

import numpy as np
import numpy.typing as npt

import pyvista
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import AmbiguousDataError
from pyvista.core.errors import MissingDataError

if TYPE_CHECKING:
    from pyvista import DataSet
    from pyvista import Table
    from pyvista import pyvista_ndarray
    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import VectorLike
    from pyvista.core.dataset import _ActiveArrayExistsInfoTuple


class FieldAssociation(enum.Enum):
    """Represents which type of vtk field a scalar or vector array is associated with."""

    POINT = int(_vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS)
    CELL = int(_vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)
    NONE = int(_vtk.vtkDataObject.FIELD_ASSOCIATION_NONE)
    ROW = int(_vtk.vtkDataObject.FIELD_ASSOCIATION_ROWS)


PointLiteral = Literal[
    FieldAssociation.POINT,
    'point',
]
CellLiteral = Literal[FieldAssociation.CELL, 'cell']
FieldLiteral = Literal[FieldAssociation.NONE, 'field']
RowLiteral = Literal[FieldAssociation.ROW, 'row']


@overload
def parse_field_choice(
    field: PointLiteral | Literal['p', 'points'],
) -> Literal[FieldAssociation.POINT]: ...
@overload
def parse_field_choice(
    field: CellLiteral | Literal['c', 'cells'],
) -> Literal[FieldAssociation.CELL]: ...
@overload
def parse_field_choice(
    field: FieldLiteral | Literal['f', 'fields'],
) -> Literal[FieldAssociation.NONE]: ...
@overload
def parse_field_choice(field: RowLiteral | Literal['r']) -> Literal[FieldAssociation.ROW]: ...
@overload
def parse_field_choice(field: FieldAssociation) -> FieldAssociation: ...
def parse_field_choice(
    field: FieldAssociation
    | PointLiteral
    | CellLiteral
    | FieldLiteral
    | RowLiteral
    | Literal['p', 'c', 'f', 'r', 'points', 'cells', 'fields'],
) -> FieldAssociation:
    """Return a field association object for a given field type string.

    Parameters
    ----------
    field : str, FieldAssociation
        Name of the field (e.g, ``'cell'``, ``'field'``, ``'point'``,
        ``'row'``).

    Returns
    -------
    pyvista.FieldAssociation
        Field association.

    """
    if isinstance(field, str):
        field_ = field.strip().lower()
        if field_ in ['cell', 'c', 'cells']:
            return FieldAssociation.CELL
        elif field_ in ['point', 'p', 'points']:
            return FieldAssociation.POINT
        elif field_ in ['field', 'f', 'fields']:
            return FieldAssociation.NONE
        elif field_ in ['row', 'r']:
            return FieldAssociation.ROW
        else:
            msg = f'Data field ({field}) not supported.'
            raise ValueError(msg)
    elif isinstance(field, FieldAssociation):
        return field
    else:
        msg = f'Data field ({field}) not supported.'  # type: ignore[unreachable]
        raise TypeError(msg)


def _coerce_pointslike_arg(
    points: MatrixLike[float] | VectorLike[float],
    *,
    copy: bool = False,
) -> tuple[NumpyArray[float], bool]:
    """Check and coerce arg to (n, 3) np.ndarray.

    Parameters
    ----------
    points : MatrixLike[float] | VectorLike[float]
        Argument to coerce into (n, 3) :class:`numpy.ndarray`.

    copy : bool, default: False
        Whether to copy the ``points`` array.  Copying always occurs if ``points``
        is not :class:`numpy.ndarray`.

    Returns
    -------
    numpy.ndarray
        Size ``(n, 3)`` array.
    bool
        Whether the input was a single point in an array-like with shape ``(3,)``.

    """
    if isinstance(points, Sequence):
        points = np.asarray(points)

    if not isinstance(points, np.ndarray):
        msg = 'Given points must be convertible to a numerical array.'  # type: ignore[unreachable]
        raise TypeError(msg)

    if points.ndim > 2:
        msg = 'Array of points must be 1D or 2D'
        raise ValueError(msg)

    if points.ndim == 2:
        if points.shape[1] != 3:
            msg = 'Array of points must have three values per point (shape (n, 3))'
            raise ValueError(msg)
        singular = False

    else:
        if points.size != 3:
            msg = 'Given point must have three values'
            raise ValueError(msg)
        singular = True
        points = np.reshape(points, [1, 3])

    if copy:
        return points.copy(), singular
    return points, singular


@_deprecate_positional_args(allowed=['array'])
def copy_vtk_array(array: _vtk.vtkAbstractArray, deep: bool = True) -> _vtk.vtkAbstractArray:  # noqa: FBT001, FBT002
    """Create a deep or shallow copy of a VTK array.

    Parameters
    ----------
    array : :vtk:`vtkAbstractArray`
        VTK array.

    deep : bool, optional
        When ``True``, create a deep copy of the array. When ``False``, returns
        a shallow copy.

    Returns
    -------
    :vtk:`vtkAbstractArray`
        Copy of the original VTK array.

    Examples
    --------
    Perform a deep copy of a vtk array.

    >>> import vtk
    >>> import pyvista as pv
    >>> arr = vtk.vtkFloatArray()
    >>> _ = arr.SetNumberOfValues(10)
    >>> arr.SetValue(0, 1)
    >>> arr_copy = pv.core.utilities.arrays.copy_vtk_array(arr)
    >>> arr_copy.GetValue(0)
    1.0

    """
    if not isinstance(array, _vtk.vtkAbstractArray):
        msg = f'Invalid type {type(array)}.'  # type: ignore[unreachable]
        raise TypeError(msg)

    new_array = _vtk.vtkAbstractArray.CreateArray(array.GetDataType())

    if deep:
        new_array.DeepCopy(array)
    else:
        new_array.ShallowCopy(array)  # type: ignore[attr-defined]

    return new_array


def has_duplicates(arr: NumpyArray[Any]) -> bool:
    """Return if an array has any duplicates.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be checked for duplicates.

    Returns
    -------
    bool
        ``True`` if the array has any duplicates, otherwise ``False``.

    """
    s = np.sort(arr, axis=None)
    return (s[1:] == s[:-1]).any()


def raise_has_duplicates(arr: NumpyArray[Any]) -> None:
    """Raise a ValueError if an array is not unique.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be checked for duplicates.

    Raises
    ------
    ValueError
        If the array contains duplicate values.

    """
    if has_duplicates(arr):
        msg = 'Array contains duplicate values.'
        raise ValueError(msg)


@overload
def convert_array(
    arr: _vtk.vtkAbstractArray,
    name: str | None = ...,
    deep: bool = ...,  # noqa: FBT001
    array_type: int | None = None,
) -> npt.NDArray[Any]: ...
@overload
def convert_array(
    arr: npt.ArrayLike,
    name: str | None = ...,
    deep: bool = ...,  # noqa: FBT001
    array_type: int | None = None,
) -> _vtk.vtkAbstractArray: ...
@overload
def convert_array(
    arr: None,
    name: str | None = ...,
    deep: bool = ...,  # noqa: FBT001
    array_type: int | None = ...,
) -> None: ...
@_deprecate_positional_args(allowed=['arr', 'name'])
def convert_array(  # noqa: PLR0917
    arr: npt.ArrayLike | _vtk.vtkAbstractArray | None,
    name: str | None = None,
    deep: bool = False,  # noqa: FBT001, FBT002
    array_type: int | None = None,
) -> npt.NDArray[Any] | _vtk.vtkAbstractArray | None:
    """Convert a NumPy array to a :vtk:`vtkDataArray` or vice versa.

    Parameters
    ----------
    arr : np.ndarray | :vtk:`vtkDataArray`
        A numpy array or :vtk:`vtkDataArray` to convert.
    name : str, optional
        The name of the data array for VTK.
    deep : bool, default: False
        If input is numpy array then deep copy values.
    array_type : int, optional
        VTK array type ID as specified in ``vtkType.h``.

    Returns
    -------
    :vtk:`vtkDataArray` | numpy.ndarray
        The converted array.  If input is a :class:`numpy.ndarray` then
        returns :vtk:`vtkDataArray` or if input is :vtk:`vtkDataArray` then
        returns NumPy ``ndarray``.

    """
    if arr is None:
        return None
    if isinstance(arr, (list, tuple, str)):
        arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        if arr.dtype == np.dtype('O'):
            arr = arr.astype('|S')
        if arr.dtype.type in (np.str_, np.bytes_):
            # This handles strings
            if arr.ndim > 0:
                # Do not call ascontiguousarray for scalar strings since this will reshape to 1D
                # and scalars are already contiguous anyway
                arr = np.ascontiguousarray(arr)
            vtk_data = convert_string_array(arr)
        else:
            # This will handle numerical data
            arr = np.ascontiguousarray(arr)
            vtk_data = _vtk.numpy_to_vtk(num_array=arr, deep=deep, array_type=array_type)
        if isinstance(name, str):
            vtk_data.SetName(name)
        return vtk_data
    # Otherwise input must be a vtkDataArray
    if not isinstance(arr, (_vtk.vtkDataArray, _vtk.vtkBitArray, _vtk.vtkStringArray)):
        msg = f'Invalid input array type ({type(arr)}).'
        raise TypeError(msg)
    # Handle booleans
    if isinstance(arr, _vtk.vtkBitArray):
        arr = vtk_bit_array_to_char(arr)
    # Handle string arrays
    if isinstance(arr, _vtk.vtkStringArray):
        return convert_string_array(arr)
    # Convert from vtkDataArry to NumPy
    return _vtk.vtk_to_numpy(arr)


@_deprecate_positional_args(allowed=['mesh', 'name'])
def get_array(  # noqa: PLR0917
    mesh: DataSet | _vtk.vtkDataSet | _vtk.vtkTable,
    name: str,
    preference: PointLiteral | CellLiteral | FieldLiteral | RowLiteral = 'cell',
    err: bool = False,  # noqa: FBT001, FBT002
) -> pyvista_ndarray | None:
    """Search point, cell and field data for an array.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Dataset to get the array from.

    name : str
        The name of the array to get the range.

    preference : str, default: "cell"
        When scalars is specified, this is the preferred array type to
        search for in the dataset.  Must be either ``'point'``,
        ``'cell'``, or ``'field'``.

    err : bool, default: False
        Whether to throw an error if array is not present.

    Returns
    -------
    pyvista.pyvista_ndarray or None
        Requested array.  Return ``None`` if there is no array
        matching the ``name`` and ``err=False``.

    """
    if isinstance(mesh, _vtk.vtkTable):
        arr = row_array(mesh, name)
        if arr is None and err:
            msg = f'Data array ({name}) not present in this dataset.'
            raise KeyError(msg)
        return arr
    else:
        preference_ = parse_field_choice(preference)

        if preference_ not in [
            FieldAssociation.CELL,
            FieldAssociation.POINT,
            FieldAssociation.NONE,
        ]:
            msg = (
                f'`preference` must be either "cell", "point", "field" for a '
                f'{type(mesh)}, not "{preference}".'
            )
            raise ValueError(msg)

        parr = point_array(mesh, name)
        carr = cell_array(mesh, name)
        farr = field_array(mesh, name)
        if sum(array is not None for array in (parr, carr, farr)) > 1:
            if preference_ == FieldAssociation.CELL:
                out = carr
            elif preference_ == FieldAssociation.POINT:
                out = parr
            else:  # must be field
                out = farr
        elif parr is not None:
            out = parr
        elif carr is not None:
            out = carr
        elif farr is not None:
            out = farr
        elif err:
            msg = f'Data array ({name}) not present in this dataset.'
            raise KeyError(msg)
        else:
            out = None
        return out


@_deprecate_positional_args(allowed=['mesh', 'name'])
def get_array_association(  # noqa: PLR0917
    mesh: DataSet | _vtk.vtkDataSet | _vtk.vtkTable,
    name: str,
    preference: PointLiteral | CellLiteral | FieldLiteral | RowLiteral = 'cell',
    err: bool = False,  # noqa: FBT001, FBT002
) -> FieldAssociation:
    """Return the array association.

    Parameters
    ----------
    mesh : Dataset
        Dataset to get the array association from.

    name : str
        The name of the array.

    preference : str, default: "cell"
        When scalars is specified, this is the preferred array type to
        search for in the dataset.  Must be either ``'point'``,
        ``'cell'``, or ``'field'``.

    err : bool, default: False
        Boolean to control whether to throw an error if array is not
        present.

    Returns
    -------
    pyvista.core.utilities.arrays.FieldAssociation
        Association of the array. If array is not present and ``err`` is
        ``False``, ``FieldAssociation.NONE`` is returned.

    """
    if isinstance(mesh, _vtk.vtkTable):
        arr = row_array(mesh, name)
        if arr is None and err:
            msg = f'Data array ({name}) not present in this dataset.'
            raise KeyError(msg)
        return FieldAssociation.ROW

    # with multiple arrays, return the array preference if possible
    parr = point_array(mesh, name)
    carr = cell_array(mesh, name)
    farr = field_array(mesh, name)
    arrays = [parr, carr, farr]
    preferences = [FieldAssociation.POINT, FieldAssociation.CELL, FieldAssociation.NONE]
    preference_field = parse_field_choice(preference)
    if preference_field not in preferences:
        msg = f'Data field ({preference}) not supported.'
        raise ValueError(msg)

    matches = [pref for pref, array in zip(preferences, arrays, strict=True) if array is not None]
    # optionally raise if no match
    if not matches:
        if err:
            msg = f'Data array ({name}) not present in this dataset.'
            raise KeyError(msg)
        return FieldAssociation.NONE
    # use preference if it applies
    if preference_field in matches:
        return preference_field
    # otherwise return first in order of point -> cell -> field
    return matches[0]


def raise_not_matching(scalars: npt.NDArray[Any], dataset: DataSet | Table) -> None:
    """Raise exception about inconsistencies.

    Parameters
    ----------
    scalars : numpy.ndarray
        Array of scalars.

    dataset : pyvista.DataSet | pyvista.Table
        Dataset to check against.

    Raises
    ------
    ValueError
        Raises a ValueError if the size of scalars does not the dataset.

    """
    if isinstance(dataset, _vtk.vtkTable):
        msg = (
            f'Number of scalars ({scalars.shape[0]}) must match number of rows ({dataset.n_rows}).'
        )
        raise ValueError(msg)  # noqa: TRY004
    msg = (
        f'Number of scalars ({scalars.shape[0]}) '
        f'must match either the number of points ({dataset.n_points}) '
        f'or the number of cells ({dataset.n_cells}).'
    )
    raise ValueError(msg)


def _assoc_array(
    obj: DataSet | _vtk.vtkDataSet, name: str, association: str = 'point'
) -> pyvista_ndarray | None:
    """Return a point, cell, or field array from a pyvista.DataSet or VTK object.

    If the array or index doesn't exist, return nothing. This matches VTK's
    behavior when using ``GetAbstractArray`` with an invalid key or index.

    """
    vtk_attr = f'Get{association.title()}Data'
    python_attr = f'{association.lower()}_data'

    if isinstance(obj, pyvista.DataSet):
        try:
            return getattr(obj, python_attr).get_array(name)
        except KeyError:  # pragma: no cover
            return None
    abstract_array = getattr(obj, vtk_attr)().GetAbstractArray(name)
    if abstract_array is not None:
        return pyvista.pyvista_ndarray(abstract_array)
    return None


def point_array(obj: DataSet | _vtk.vtkDataSet, name: str) -> pyvista_ndarray | None:
    """Return point array of a pyvista or vtk object.

    Parameters
    ----------
    obj : DataSet | :vtk:`vtkDataSet`
        PyVista or VTK dataset.

    name : str | int
        Name or index of the array.

    Returns
    -------
    pyvista.pyvista_ndarray or None
        Wrapped array if the index or name is valid. Otherwise, ``None``.

    """
    return _assoc_array(obj, name, 'point')


def field_array(obj: DataSet | _vtk.vtkDataSet, name: str) -> pyvista_ndarray | None:
    """Return field data of a pyvista or vtk object.

    Parameters
    ----------
    obj : DataSet | :vtk:`vtkDataSet`
        PyVista or VTK dataset.

    name : str | int
        Name or index of the array.

    Returns
    -------
    pyvista.pyvista_ndarray or None
        Wrapped array if the index or name is valid. Otherwise, ``None``.

    """
    return _assoc_array(obj, name, 'field')


def cell_array(obj: DataSet | _vtk.vtkDataSet, name: str) -> pyvista_ndarray | None:
    """Return cell array of a pyvista or vtk object.

    Parameters
    ----------
    obj : DataSet | :vtk:`vtkDataSet`
        PyVista or VTK dataset.

    name : str | int
        Name or index of the array.

    Returns
    -------
    pyvista.pyvista_ndarray or None
        Wrapped array if the index or name is valid. Otherwise, ``None``.

    """
    return _assoc_array(obj, name, 'cell')


def row_array(obj: _vtk.vtkTable, name: str) -> pyvista_ndarray | None:
    """Return row array of a vtk object.

    Parameters
    ----------
    obj : :vtk:`vtkTable`
        PyVista or VTK table.

    name : str
        Name of the array.

    Returns
    -------
    numpy.ndarray
        Wrapped array.

    """
    vtkarr = obj.GetRowData().GetAbstractArray(name)
    if vtkarr is not None:
        return pyvista.pyvista_ndarray(convert_array(vtkarr))
    else:
        return None


def get_vtk_type(typ: npt.DTypeLike) -> int:
    """Look up the VTK type for a given numpy data type.

    Corrects for string type mapping issues.

    Parameters
    ----------
    typ : numpy.dtype
        Numpy data type.

    Returns
    -------
    int
        Integer type id specified in ``vtkType.h``.

    """
    typ_ = _vtk.get_vtk_array_type(typ)
    # This handles a silly string type bug
    if typ_ == 3:
        return 13
    return typ_


def vtk_bit_array_to_char(vtkarr_bint: _vtk.vtkBitArray) -> _vtk.vtkCharArray:
    """Cast vtk bit array to a char array.

    Parameters
    ----------
    vtkarr_bint : :vtk:`vtkBitArray`
        VTK binary array.

    Returns
    -------
    :vtk:`vtkCharArray`
        VTK char array.

    Notes
    -----
    This performs a copy.

    """
    vtkarr = _vtk.vtkCharArray()
    vtkarr.DeepCopy(vtkarr_bint)
    return vtkarr


def vtk_id_list_to_array(vtk_id_list: _vtk.vtkIdList) -> NumpyArray[int]:
    """Convert a :vtk:`vtkIdList` to a NumPy array.

    Parameters
    ----------
    vtk_id_list : :vtk:`vtkIdList`
        VTK ID list.

    Returns
    -------
    numpy.ndarray
        Array of IDs.

    """
    return np.array([vtk_id_list.GetId(i) for i in range(vtk_id_list.GetNumberOfIds())])


def _set_string_scalar_object_name(vtkarr: _vtk.vtkStringArray) -> None:
    """Set object name for scalar string arrays."""
    # This is used as a flag so that scalar arrays can be reshaped later.
    try:
        vtkarr.SetObjectName('scalar')
    except AttributeError:
        vtkarr.GetObjectName = lambda: 'scalar'  # type: ignore[method-assign]


@overload
def convert_string_array(
    arr: _vtk.vtkStringArray, name: str | None = ...
) -> npt.NDArray[np.str_]: ...
@overload
def convert_string_array(
    arr: str | npt.NDArray[np.str_], name: str | None = ...
) -> _vtk.vtkStringArray: ...
def convert_string_array(
    arr: str | npt.NDArray[np.str_] | _vtk.vtkStringArray, name: str | None = None
) -> npt.NDArray[np.str_] | _vtk.vtkStringArray:
    """Convert a numpy array of strings to a :vtk:`vtkStringArray` or vice versa.

    If a scalar string is provided, it is converted to a :vtk:`vtkCharArray`

    Parameters
    ----------
    arr : numpy.ndarray | str
        Numpy string array to convert.

    name : str, optional
        Name to set the :vtk:`vtkStringArray` to.

    Returns
    -------
    :vtk:`vtkStringArray`
        VTK string array.

    Notes
    -----
    Note that this is terribly inefficient. If you have ideas on how
    to make this faster, please consider opening a pull request.

    """
    arr = np.array(arr) if isinstance(arr, str) else arr
    if isinstance(arr, np.ndarray):
        # VTK default fonts only support ASCII. See https://gitlab.kitware.com/vtk/vtk/-/issues/16904
        if (
            np.issubdtype(arr.dtype, np.str_) and not ''.join(arr.tolist()).isascii()
        ):  # avoids segfault
            msg = 'String array contains non-ASCII characters that are not supported by VTK.'
            raise ValueError(msg)
        vtkarr = _vtk.vtkStringArray()
        if arr.ndim == 0:
            arr = arr.reshape((1,))
            # distinguish scalar inputs from array inputs by
            # setting the object name
            _set_string_scalar_object_name(vtkarr)

        # OPTIMIZE ###########
        for val in arr:
            vtkarr.InsertNextValue(val)
        ################################
        if isinstance(name, str):
            vtkarr.SetName(name)
        return vtkarr
    # Otherwise it is a vtk array and needs to be converted back to numpy
    # OPTIMIZE ###############
    nvalues = arr.GetNumberOfValues()
    arr_out = np.array([arr.GetValue(i) for i in range(nvalues)], dtype='|U')
    try:
        if arr.GetObjectName() == 'scalar':
            return np.array(''.join(arr_out))
    except AttributeError:
        pass
    return arr_out
    ########################################


def array_from_vtkmatrix(matrix: _vtk.vtkMatrix3x3 | _vtk.vtkMatrix4x4) -> NumpyArray[float]:
    """Convert a vtk matrix to an array.

    Parameters
    ----------
    matrix : :vtk:`vtkMatrix3x3` | :vtk:`vtkMatrix4x4`
        The vtk matrix to be converted to a ``numpy.ndarray``.
        Returned ndarray has shape (3, 3) or (4, 4) as appropriate.

    Returns
    -------
    numpy.ndarray
        Numpy array containing the data from ``matrix``.

    """
    if isinstance(matrix, _vtk.vtkMatrix3x3):
        shape = (3, 3)
    elif isinstance(matrix, _vtk.vtkMatrix4x4):
        shape = (4, 4)
    else:
        msg = (  # type: ignore[unreachable]
            'Expected vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4 input,'
            f' got {type(matrix).__name__} instead.'
        )
        raise TypeError(msg)
    array = np.zeros(shape)
    for i, j in product(range(shape[0]), range(shape[1])):
        array[i, j] = matrix.GetElement(i, j)
    return array


def vtkmatrix_from_array(array: NumpyArray[float]) -> _vtk.vtkMatrix3x3 | _vtk.vtkMatrix4x4:
    """Convert a ``numpy.ndarray`` or array-like to a vtk matrix.

    Parameters
    ----------
    array : array_like[float]
        The array or array-like to be converted to a vtk matrix.
        Shape (3, 3) gets converted to a :vtk:`vtkMatrix3x3`, shape (4, 4)
        gets converted to a :vtk:`vtkMatrix4x4`. No other shapes are valid.

    Returns
    -------
    :vtk:`vtkMatrix3x3` | :vtk:`vtkMatrix4x4`
        VTK matrix.

    """
    array = np.asarray(array)
    if array.shape == (3, 3):
        matrix = _vtk.vtkMatrix3x3()
    elif array.shape == (4, 4):
        matrix = _vtk.vtkMatrix4x4()  # type: ignore[assignment]
    else:
        msg = f'Invalid shape {array.shape}, must be (3, 3) or (4, 4).'
        raise ValueError(msg)
    m, n = array.shape
    for i, j in product(range(m), range(n)):
        matrix.SetElement(i, j, array[i, j])
    return matrix


def set_default_active_vectors(mesh: pyvista.DataSet) -> _ActiveArrayExistsInfoTuple:
    """Set a default vectors array on mesh, if not already set.

    If an active vector already exists, no changes are made.

    If an active vectors does not exist, it checks for possibly cell
    or point arrays with shape ``(n, 3)``.  If only one exists, then
    it is set as the active vectors.  Otherwise, an error is raised.

    .. versionchanged:: 0.45
        The field and name of the active array is now returned.
        Previously, ``None`` was returned.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Dataset to set default active vectors.

    Raises
    ------
    MissingDataError
        If no vector-like arrays exist.

    AmbiguousDataError
        If more than one vector-like arrays exist.

    Returns
    -------
    tuple[FieldAssociation, str]
        The field and name of the active array.

    """
    from pyvista.core.dataset import _ActiveArrayExistsInfoTuple  # noqa: PLC0415

    if mesh.active_vectors_name is None:
        point_data = mesh.point_data
        cell_data = mesh.cell_data

        possible_vectors_point = [
            name for name, value in point_data.items() if value.ndim == 2 and value.shape[1] == 3
        ]
        possible_vectors_cell = [
            name for name, value in cell_data.items() if value.ndim == 2 and value.shape[1] == 3
        ]

        possible_vectors = possible_vectors_point + possible_vectors_cell
        n_possible_vectors = len(possible_vectors)

        if n_possible_vectors == 1:
            preference: Literal['point', 'cell'] = (
                'point' if len(possible_vectors_point) == 1 else 'cell'
            )
            mesh.set_active_vectors(possible_vectors[0], preference=preference)
        elif n_possible_vectors < 1:
            msg = 'No vector-like data available.'
            raise MissingDataError(msg)
        else:  # n_possible_vectors > 1:
            msg = (
                'Multiple vector-like data available\n'
                f'cell data: {possible_vectors_cell}.\n'
                f'point data: {possible_vectors_point}.\n'
                'Set one as active using DataSet.set_active_vectors(name, preference=type)'
            )
            raise AmbiguousDataError(msg)
    field, name = mesh.active_vectors_info
    return _ActiveArrayExistsInfoTuple(field, cast('str', name))


def set_default_active_scalars(mesh: pyvista.DataSet) -> _ActiveArrayExistsInfoTuple:
    """Set a default scalars array on mesh, if not already set.

    If an active scalars already exists, no changes are made.

    If an active scalars does not exist, it checks for point or cell
    arrays.  If only one exists, then it is set as the active scalars.
    Otherwise, an error is raised.

    .. versionchanged:: 0.45
        The field and name of the active array is now returned.
        Previously, ``None`` was returned.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Dataset to set default active scalars.

    Raises
    ------
    MissingDataError
        If no arrays exist.

    AmbiguousDataError
        If more than one array exists.

    Returns
    -------
    tuple[FieldAssociation, str]
        The field and name of the active array.

    """
    from pyvista.core.dataset import _ActiveArrayExistsInfoTuple  # noqa: PLC0415

    if mesh.active_scalars_name is None:
        point_data = mesh.point_data
        cell_data = mesh.cell_data

        possible_scalars_point = point_data.keys()
        possible_scalars_cell = cell_data.keys()

        possible_scalars = possible_scalars_point + possible_scalars_cell
        n_possible_scalars = len(possible_scalars)

        if n_possible_scalars == 1:
            preference: Literal['point', 'cell'] = (
                'point' if len(possible_scalars_point) == 1 else 'cell'
            )
            mesh.set_active_scalars(possible_scalars[0], preference=preference)
        elif n_possible_scalars < 1:
            msg = 'No data available.'
            raise MissingDataError(msg)
        else:  # n_possible_scalars > 1:
            msg = (
                'Multiple data available\n'
                f'cell data: {possible_scalars_cell}.\n'
                f'point data: {possible_scalars_point}.\n'
                'Set one as active using DataSet.set_active_scalars(name, preference=type)'
            )
            raise AmbiguousDataError(msg)
    field, name = mesh.active_scalars_info
    return _ActiveArrayExistsInfoTuple(field, cast('str', name))


_JSONValueType = (
    dict[str, '_JSONValueType']
    | list['_JSONValueType']
    | tuple['_JSONValueType']
    | str
    | int
    | float
    | bool
    | None
)


class _SerializedDictArray(_vtk.DisableVtkSnakeCase, UserDict, _vtk.vtkStringArray):  # type: ignore[type-arg]
    """Dict-like object with a JSON-serialized string array representation.

    This class behaves just like a regular dict, except its contents
    are represented internally as a JSON-formatted :vtk:`vtkStringArray`.
    The string array is updated dynamically any time the dict is
    modified, such that modifying the dict will also implicitly modify
    its JSON string representation.

    Notes
    -----
    This class is intended for use as a dict with a small number of keys and
    relatively small values, e.g. for storing metadata. It should not be
    used to store frequently accessed array data with hundreds of entries.

    """

    @property
    def _string(self: _SerializedDictArray) -> str:
        """Get the :vtk:`vtkStringArray` string."""
        return ''.join([self.GetValue(i) for i in range(self.GetNumberOfValues())])

    @_string.setter
    def _string(self: _SerializedDictArray, str_: str) -> None:
        """Set the :vtk:`vtkStringArray` to a specified string."""
        self.SetNumberOfValues(0)  # Clear string
        for char in str_:  # Populate string
            self.InsertNextValue(char)

    def _update_string(self: _SerializedDictArray) -> None:
        """Format dict data as JSON and update the :vtk:`vtkStringArray`."""
        data_str = json.dumps(self.data)
        if data_str != self._string:
            self._string = data_str

    def __repr__(self: _SerializedDictArray) -> str:
        """Return JSON-formatted dict representation."""
        return self._string

    def __init__(
        self: _SerializedDictArray,
        dict_: str | dict[str, _JSONValueType] | UserDict[str, _JSONValueType] | None = None,
        /,
        **kwargs,
    ) -> None:
        # Init from JSON string
        if isinstance(dict_, str):
            dict_ = json.loads(dict_)

        # Init UserDict
        super().__init__(dict_, **kwargs)  # type: ignore[arg-type]
        self._update_string()

        # Flag self as a scalar string
        # This is only needed so that the Field DatasetAttributes repr
        # shows this array as `str`
        _set_string_scalar_object_name(self)

    def __getstate__(self: _SerializedDictArray) -> None:
        """Support pickling.

        This method does nothing. It only exists to make the pickle library happy.
        Classes that store an instance of this class must pickle this array directly.
        E.g. DataObjects can support this by storing this array as field data
        """

    def __setstate__(self: _SerializedDictArray, state: Any) -> None:
        """Support pickling.

        This method does nothing. It only exists to make the pickle library happy.
        Classes that store an instance of this class must pickle this array directly.
        E.g. DataObjects can support this by storing this array as field data
        """

    # Override any/all `UserDict` or `MutableMapping` methods which mutate
    # the dictionary. This ensures the serialized string is also updated
    # and synced with the dict

    def __setitem__(self: _SerializedDictArray, key: Any, item: Any) -> None:
        super().__setitem__(key, item)
        self._update_string()

    def __delitem__(self: _SerializedDictArray, key: Any) -> None:
        super().__delitem__(key)
        self._update_string()

    def __setattr__(self: _SerializedDictArray, key: Any, value: Any) -> None:
        object.__setattr__(self, key, value)
        self._update_string() if key != '_string' else None

    def update(self: _SerializedDictArray, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._update_string()

    def popitem(self: _SerializedDictArray) -> Any:
        item = super().popitem()
        self._update_string()
        return item

    def pop(self: _SerializedDictArray, __key: Any) -> Any:  # type: ignore[override]  # noqa: PYI063
        item = super().pop(__key)
        self._update_string()
        return item

    def clear(self: _SerializedDictArray) -> None:
        super().clear()
        self._update_string()

    def setdefault(self: _SerializedDictArray, *args, **kwargs) -> None:
        super().setdefault(*args, **kwargs)
        self._update_string()
