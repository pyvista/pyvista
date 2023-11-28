"""Internal array utilities."""
import collections.abc
import enum
from itertools import product
from typing import Optional, Tuple, Union

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core._typing_core import Matrix, NumpyFltArray, TransformLike, Vector
from pyvista.core.errors import AmbiguousDataError, MissingDataError


class FieldAssociation(enum.Enum):
    """Represents which type of vtk field a scalar or vector array is associated with."""

    POINT = _vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
    CELL = _vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
    NONE = _vtk.vtkDataObject.FIELD_ASSOCIATION_NONE
    ROW = _vtk.vtkDataObject.FIELD_ASSOCIATION_ROWS


def parse_field_choice(field):
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
        field = field.strip().lower()
        if field in ['cell', 'c', 'cells']:
            field = FieldAssociation.CELL
        elif field in ['point', 'p', 'points']:
            field = FieldAssociation.POINT
        elif field in ['field', 'f', 'fields']:
            field = FieldAssociation.NONE
        elif field in ['row', 'r']:
            field = FieldAssociation.ROW
        else:
            raise ValueError(f'Data field ({field}) not supported.')
    elif isinstance(field, FieldAssociation):
        pass
    else:
        raise TypeError(f'Data field ({field}) not supported.')
    return field


def _coerce_pointslike_arg(
    points: Union[Matrix, Vector], copy: bool = False
) -> Tuple[np.ndarray, bool]:
    """Check and coerce arg to (n, 3) np.ndarray.

    Parameters
    ----------
    points : Matrix, Vector
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
    if isinstance(points, collections.abc.Sequence):
        points = np.asarray(points)

    if not isinstance(points, np.ndarray):
        raise TypeError("Given points must be convertible to a numerical array.")

    if points.ndim > 2:
        raise ValueError("Array of points must be 1D or 2D")

    if points.ndim == 2:
        if points.shape[1] != 3:
            raise ValueError("Array of points must have three values per point (shape (n, 3))")
        singular = False

    else:
        if points.size != 3:
            raise ValueError("Given point must have three values")
        singular = True
        points = np.reshape(points, [1, 3])

    if copy:
        return points.copy(), singular
    return points, singular


def copy_vtk_array(array, deep=True):
    """Create a deep or shallow copy of a VTK array.

    Parameters
    ----------
    array : vtk.vtkDataArray | vtk.vtkAbstractArray
        VTK array.

    deep : bool, optional
        When ``True``, create a deep copy of the array. When ``False``, returns
        a shallow copy.

    Returns
    -------
    vtk.vtkDataArray or vtk.vtkAbstractArray
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
    if not isinstance(array, (_vtk.vtkDataArray, _vtk.vtkAbstractArray)):
        raise TypeError(f"Invalid type {type(array)}.")

    new_array = type(array)()
    if deep:
        new_array.DeepCopy(array)
    else:
        new_array.ShallowCopy(array)

    return new_array


def has_duplicates(arr):
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


def raise_has_duplicates(arr):
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
        raise ValueError("Array contains duplicate values.")


def convert_array(arr, name=None, deep=False, array_type=None):
    """Convert a NumPy array to a vtkDataArray or vice versa.

    Parameters
    ----------
    arr : np.ndarray | vtkDataArray
        A numpy array or vtkDataArry to convert.
    name : str, optional
        The name of the data array for VTK.
    deep : bool, default: False
        If input is numpy array then deep copy values.
    array_type : int, optional
        VTK array type ID as specified in specified in ``vtkType.h``.

    Returns
    -------
    vtkDataArray or numpy.ndarray
        The converted array.  If input is a :class:`numpy.ndarray` then
        returns ``vtkDataArray`` or is input is ``vtkDataArray`` then
        returns NumPy ``ndarray``.

    """
    if arr is None:
        return
    if isinstance(arr, (list, tuple)):
        arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        if arr.dtype == np.dtype('O'):
            arr = arr.astype('|S')
        arr = np.ascontiguousarray(arr)
        if arr.dtype.type in (np.str_, np.bytes_):
            # This handles strings
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
        raise TypeError(f'Invalid input array type ({type(arr)}).')
    # Handle booleans
    if isinstance(arr, _vtk.vtkBitArray):
        arr = vtk_bit_array_to_char(arr)
    # Handle string arrays
    if isinstance(arr, _vtk.vtkStringArray):
        return convert_string_array(arr)
    # Convert from vtkDataArry to NumPy
    return _vtk.vtk_to_numpy(arr)


def get_array(mesh, name, preference='cell', err=False) -> Optional['pyvista.ndarray']:
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
            raise KeyError(f'Data array ({name}) not present in this dataset.')
        return arr

    if not isinstance(preference, str):
        raise TypeError('`preference` must be a string')
    if preference not in ['cell', 'point', 'field']:
        raise ValueError(
            f'`preference` must be either "cell", "point", "field" for a '
            f'{type(mesh)}, not "{preference}".'
        )

    parr = point_array(mesh, name)
    carr = cell_array(mesh, name)
    farr = field_array(mesh, name)
    preference = parse_field_choice(preference)
    if sum([array is not None for array in (parr, carr, farr)]) > 1:
        if preference == FieldAssociation.CELL:
            return carr
        elif preference == FieldAssociation.POINT:
            return parr
        else:  # must be field
            return farr

    if parr is not None:
        return parr
    elif carr is not None:
        return carr
    elif farr is not None:
        return farr
    elif err:
        raise KeyError(f'Data array ({name}) not present in this dataset.')
    return None


def get_array_association(mesh, name, preference='cell', err=False) -> FieldAssociation:
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
            raise KeyError(f'Data array ({name}) not present in this dataset.')
        return FieldAssociation.ROW

    # with multiple arrays, return the array preference if possible
    parr = point_array(mesh, name)
    carr = cell_array(mesh, name)
    farr = field_array(mesh, name)
    arrays = [parr, carr, farr]
    preferences = [FieldAssociation.POINT, FieldAssociation.CELL, FieldAssociation.NONE]
    preference = parse_field_choice(preference)
    if preference not in preferences:
        raise ValueError(f'Data field ({preference}) not supported.')

    matches = [pref for pref, array in zip(preferences, arrays) if array is not None]
    # optionally raise if no match
    if not matches:
        if err:
            raise KeyError(f'Data array ({name}) not present in this dataset.')
        return FieldAssociation.NONE
    # use preference if it applies
    if preference in matches:
        return preference
    # otherwise return first in order of point -> cell -> field
    return matches[0]


def raise_not_matching(scalars, dataset):
    """Raise exception about inconsistencies.

    Parameters
    ----------
    scalars : numpy.ndarray
        Array of scalars.

    dataset : pyvista.DataSet
        Dataset to check against.

    Raises
    ------
    ValueError
        Raises a ValueError if the size of scalars does not the dataset.
    """
    if isinstance(dataset, _vtk.vtkTable):
        raise ValueError(
            f'Number of scalars ({scalars.shape[0]}) must match number of rows ({dataset.n_rows}).'
        )
    raise ValueError(
        f'Number of scalars ({scalars.shape[0]}) '
        f'must match either the number of points ({dataset.n_points}) '
        f'or the number of cells ({dataset.n_cells}).'
    )


def _assoc_array(obj, name, association='point'):
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


def point_array(obj, name):
    """Return point array of a pyvista or vtk object.

    Parameters
    ----------
    obj : pyvista.DataSet | vtk.vtkDataSet
        PyVista or VTK dataset.

    name : str | int
        Name or index of the array.

    Returns
    -------
    pyvista.pyvista_ndarray or None
        Wrapped array if the index or name is valid. Otherwise, ``None``.

    """
    return _assoc_array(obj, name, 'point')


def field_array(obj, name):
    """Return field data of a pyvista or vtk object.

    Parameters
    ----------
    obj : pyvista.DataSet or vtk.vtkDataSet
        PyVista or VTK dataset.

    name : str | int
        Name or index of the array.

    Returns
    -------
    pyvista.pyvista_ndarray or None
        Wrapped array if the index or name is valid. Otherwise, ``None``.

    """
    return _assoc_array(obj, name, 'field')


def cell_array(obj, name):
    """Return cell array of a pyvista or vtk object.

    Parameters
    ----------
    obj : pyvista.DataSet or vtk.vtkDataSet
        PyVista or VTK dataset.

    name : str | int
        Name or index of the array.

    Returns
    -------
    pyvista.pyvista_ndarray or None
        Wrapped array if the index or name is valid. Otherwise, ``None``.

    """
    return _assoc_array(obj, name, 'cell')


def row_array(obj, name):
    """Return row array of a vtk object.

    Parameters
    ----------
    obj : vtk.vtkDataSet
        PyVista or VTK dataset.

    name : str
        Name of the array.

    Returns
    -------
    numpy.ndarray
        Wrapped array.

    """
    vtkarr = obj.GetRowData().GetAbstractArray(name)
    return convert_array(vtkarr)


def get_vtk_type(typ):
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
    typ = _vtk.get_vtk_array_type(typ)
    # This handles a silly string type bug
    if typ == 3:
        return 13
    return typ


def vtk_bit_array_to_char(vtkarr_bint):
    """Cast vtk bit array to a char array.

    Parameters
    ----------
    vtkarr_bint : vtk.vtkBitArray
        VTK binary array.

    Returns
    -------
    vtk.vtkCharArray
        VTK char array.

    Notes
    -----
    This performs a copy.

    """
    vtkarr = _vtk.vtkCharArray()
    vtkarr.DeepCopy(vtkarr_bint)
    return vtkarr


def vtk_id_list_to_array(vtk_id_list):
    """Convert a vtkIdList to a NumPy array.

    Parameters
    ----------
    vtk_id_list : vtk.vtkIdList
        VTK ID list.

    Returns
    -------
    numpy.ndarray
        Array of IDs.

    """
    return np.array([vtk_id_list.GetId(i) for i in range(vtk_id_list.GetNumberOfIds())])


def convert_string_array(arr, name=None):
    """Convert a numpy array of strings to a vtkStringArray or vice versa.

    Parameters
    ----------
    arr : numpy.ndarray
        Numpy string array to convert.

    name : str, optional
        Name to set the vtkStringArray to.

    Returns
    -------
    vtkStringArray
        VTK string array.

    Notes
    -----
    Note that this is terribly inefficient. If you have ideas on how
    to make this faster, please consider opening a pull request.

    """
    if isinstance(arr, np.ndarray):
        # VTK default fonts only support ASCII. See https://gitlab.kitware.com/vtk/vtk/-/issues/16904
        if np.issubdtype(arr.dtype, np.str_) and not ''.join(arr).isascii():  # avoids segfault
            raise ValueError(
                'String array contains non-ASCII characters that are not supported by VTK.'
            )
        vtkarr = _vtk.vtkStringArray()
        ########### OPTIMIZE ###########
        for val in arr:
            vtkarr.InsertNextValue(val)
        ################################
        if isinstance(name, str):
            vtkarr.SetName(name)
        return vtkarr
    # Otherwise it is a vtk array and needs to be converted back to numpy
    ############### OPTIMIZE ###############
    nvalues = arr.GetNumberOfValues()
    return np.array([arr.GetValue(i) for i in range(nvalues)], dtype='|U')
    ########################################


def array_from_vtkmatrix(matrix) -> NumpyFltArray:
    """Convert a vtk matrix to an array.

    Parameters
    ----------
    matrix : vtk.vtkMatrix3x3 | vtk.vtkMatrix4x4
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
        raise TypeError(
            'Expected vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4 input,'
            f' got {type(matrix).__name__} instead.'
        )
    array = np.zeros(shape)
    for i, j in product(range(shape[0]), range(shape[1])):
        array[i, j] = matrix.GetElement(i, j)
    return array


def vtkmatrix_from_array(array):
    """Convert a ``numpy.ndarray`` or array-like to a vtk matrix.

    Parameters
    ----------
    array : array_like[float]
        The array or array-like to be converted to a vtk matrix.
        Shape (3, 3) gets converted to a ``vtk.vtkMatrix3x3``, shape (4, 4)
        gets converted to a ``vtk.vtkMatrix4x4``. No other shapes are valid.

    Returns
    -------
    vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4
        VTK matrix.

    """
    array = np.asarray(array)
    if array.shape == (3, 3):
        matrix = _vtk.vtkMatrix3x3()
    elif array.shape == (4, 4):
        matrix = _vtk.vtkMatrix4x4()
    else:
        raise ValueError(f'Invalid shape {array.shape}, must be (3, 3) or (4, 4).')
    m, n = array.shape
    for i, j in product(range(m), range(n)):
        matrix.SetElement(i, j, array[i, j])
    return matrix


def set_default_active_vectors(mesh: 'pyvista.DataSet') -> None:
    """Set a default vectors array on mesh, if not already set.

    If an active vector already exists, no changes are made.

    If an active vectors does not exist, it checks for possibly cell
    or point arrays with shape ``(n, 3)``.  If only one exists, then
    it is set as the active vectors.  Otherwise, an error is raised.

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

    """
    if mesh.active_vectors_name is not None:
        return

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
        if len(possible_vectors_point) == 1:
            preference = 'point'
        else:
            preference = 'cell'
        mesh.set_active_vectors(possible_vectors[0], preference=preference)
    elif n_possible_vectors < 1:
        raise MissingDataError("No vector-like data available.")
    elif n_possible_vectors > 1:
        raise AmbiguousDataError(
            "Multiple vector-like data available\n"
            f"cell data: {possible_vectors_cell}.\n"
            f"point data: {possible_vectors_point}.\n"
            "Set one as active using DataSet.set_active_vectors(name, preference=type)"
        )


def set_default_active_scalars(mesh: 'pyvista.DataSet') -> None:
    """Set a default scalars array on mesh, if not already set.

    If an active scalars already exists, no changes are made.

    If an active scalars does not exist, it checks for point or cell
    arrays.  If only one exists, then it is set as the active scalars.
    Otherwise, an error is raised.

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

    """
    if mesh.active_scalars_name is not None:
        return

    point_data = mesh.point_data
    cell_data = mesh.cell_data

    possible_scalars_point = point_data.keys()
    possible_scalars_cell = cell_data.keys()

    possible_scalars = possible_scalars_point + possible_scalars_cell
    n_possible_scalars = len(possible_scalars)

    if n_possible_scalars == 1:
        if len(possible_scalars_point) == 1:
            preference = 'point'
        else:
            preference = 'cell'
        mesh.set_active_scalars(possible_scalars[0], preference=preference)
    elif n_possible_scalars < 1:
        raise MissingDataError("No data available.")
    elif n_possible_scalars > 1:
        raise AmbiguousDataError(
            "Multiple data available\n"
            f"cell data: {possible_scalars_cell}.\n"
            f"point data: {possible_scalars_point}.\n"
            "Set one as active using DataSet.set_active_scalars(name, preference=type)"
        )


def _coerce_transformlike_arg(transform_like: TransformLike) -> NumpyFltArray:
    """Check and coerce transform-like arg to a 4x4 numpy array.

    Parameters
    ----------
    transform_like : np.ndarray | vtkMatrix3x3 | vtkMatrix4x4 | vtkTransform
        Transformation matrix as a 3x3 or 4x4 numpy array, vtkMatrix, or
        from a vtkTransform.

    Returns
    -------
    np.ndarray
        4x4 transformation matrix.

    """
    transform_array: NumpyFltArray = np.eye(4)
    if isinstance(transform_like, _vtk.vtkMatrix4x4):
        transform_array = array_from_vtkmatrix(transform_like)
    elif isinstance(transform_like, _vtk.vtkMatrix3x3):
        transform_array[:3, :3] = array_from_vtkmatrix(transform_like)
    elif isinstance(transform_like, _vtk.vtkTransform):
        transform_array = array_from_vtkmatrix(transform_like.GetMatrix())
    elif isinstance(transform_like, np.ndarray):
        if transform_like.shape == (3, 3):
            transform_array[:3, :3] = transform_like
        elif transform_like.shape == (4, 4):
            transform_array = transform_like
        else:
            raise ValueError('Transformation array must be 3x3 or 4x4.')
    else:
        raise TypeError(
            'Input transform must be one of:\n'
            '\tvtk.vtkMatrix4x4\n'
            '\tvtk.vtkMatrix3x3\n'
            '\tvtk.vtkTransform\n'
            '\t4x4 np.ndarray\n'
            '\t3x3 np.ndarray\n'
        )
    return transform_array


def cast_to_list_array(arr):
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
    return cast_to_ndarray(arr).tolist()


def cast_to_tuple_array(arr):
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
    arr = cast_to_ndarray(arr).tolist()

    def _to_tuple(s):
        return tuple(_to_tuple(i) for i in s) if isinstance(s, list) else s

    return _to_tuple(arr)


def cast_to_ndarray(arr, /, *, as_any=True, dtype=None, copy=False):
    """Cast array to a NumPy ndarray.

    Parameters
    ----------
    arr : array_like
        Array to cast.

    as_any : bool, default: True
        Allow subclasses of ``np.ndarray`` to pass through without
        making a copy.

    dtype : dtype_like
        The data-type of the returned array.

    copy : bool, default: False
        If ``True``, a copy of the array is returned. A copy is always
        returned if the array:

            * is a nested sequence
            * is a subclass of ``np.ndarray`` and ``as_any`` is ``False``.

    Raises
    ------
    ValueError
        If input cannot be cast as a NumPy ndarray.

    Returns
    -------
    np.ndarray
        NumPy ndarray.

    """
    if as_any and not copy and dtype is None and isinstance(arr, np.ndarray):
        return arr
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
        raise ValueError(f"Input cannot be cast as {np.ndarray}.") from e
    return out
