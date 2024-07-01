"""Internal array utilities."""

from __future__ import annotations

from collections import UserDict
import enum
from itertools import product
import json
from typing import TYPE_CHECKING
from typing import Sequence
from typing import Union

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import AmbiguousDataError
from pyvista.core.errors import MissingDataError

if TYPE_CHECKING:  # pragma: no cover
    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import TransformLike
    from pyvista.core._typing_core import VectorLike


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
    points: MatrixLike[float] | VectorLike[float],
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
        VTK array type ID as specified in ``vtkType.h``.

    Returns
    -------
    vtkDataArray or numpy.ndarray
        The converted array.  If input is a :class:`numpy.ndarray` then
        returns ``vtkDataArray`` or is input is ``vtkDataArray`` then
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
        raise TypeError(f'Invalid input array type ({type(arr)}).')
    # Handle booleans
    if isinstance(arr, _vtk.vtkBitArray):
        arr = vtk_bit_array_to_char(arr)
    # Handle string arrays
    if isinstance(arr, _vtk.vtkStringArray):
        return convert_string_array(arr)
    # Convert from vtkDataArry to NumPy
    return _vtk.vtk_to_numpy(arr)


def get_array(mesh, name, preference='cell', err=False) -> pyvista.ndarray | None:
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
            f'{type(mesh)}, not "{preference}".',
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
            f'Number of scalars ({scalars.shape[0]}) must match number of rows ({dataset.n_rows}).',
        )
    raise ValueError(
        f'Number of scalars ({scalars.shape[0]}) '
        f'must match either the number of points ({dataset.n_points}) '
        f'or the number of cells ({dataset.n_cells}).',
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


def _set_string_scalar_object_name(vtkarr: _vtk.vtkStringArray):
    """Set object name for scalar string arrays."""
    # This is used as a flag so that scalar arrays can be reshaped later.
    try:
        vtkarr.SetObjectName('scalar')
    except AttributeError:
        vtkarr.GetObjectName = lambda: 'scalar'


def convert_string_array(arr, name=None):
    """Convert a numpy array of strings to a vtkStringArray or vice versa.

    If a scalar string is provided, it is converted to a vtkCharArray

    Parameters
    ----------
    arr : numpy.ndarray | str
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
    arr = np.array(arr) if isinstance(arr, str) else arr
    if isinstance(arr, np.ndarray):
        # VTK default fonts only support ASCII. See https://gitlab.kitware.com/vtk/vtk/-/issues/16904
        if (
            np.issubdtype(arr.dtype, np.str_) and not ''.join(arr.tolist()).isascii()
        ):  # avoids segfault
            raise ValueError(
                'String array contains non-ASCII characters that are not supported by VTK.',
            )
        vtkarr = _vtk.vtkStringArray()
        if arr.ndim == 0:
            arr = arr.reshape((1,))
            # distinguish scalar inputs from array inputs by
            # setting the object name
            _set_string_scalar_object_name(vtkarr)

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
    arr_out = np.array([arr.GetValue(i) for i in range(nvalues)], dtype='|U')
    try:
        if arr.GetObjectName() == 'scalar':
            return np.array("".join(arr_out))
    except AttributeError:
        pass
    return arr_out
    ########################################


def array_from_vtkmatrix(matrix) -> NumpyArray[float]:
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
            f' got {type(matrix).__name__} instead.',
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


def set_default_active_vectors(mesh: pyvista.DataSet) -> None:
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
        preference = 'point' if len(possible_vectors_point) == 1 else 'cell'
        mesh.set_active_vectors(possible_vectors[0], preference=preference)
    elif n_possible_vectors < 1:
        raise MissingDataError("No vector-like data available.")
    elif n_possible_vectors > 1:
        raise AmbiguousDataError(
            "Multiple vector-like data available\n"
            f"cell data: {possible_vectors_cell}.\n"
            f"point data: {possible_vectors_point}.\n"
            "Set one as active using DataSet.set_active_vectors(name, preference=type)",
        )


def set_default_active_scalars(mesh: pyvista.DataSet) -> None:
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
        preference = 'point' if len(possible_scalars_point) == 1 else 'cell'
        mesh.set_active_scalars(possible_scalars[0], preference=preference)
    elif n_possible_scalars < 1:
        raise MissingDataError("No data available.")
    elif n_possible_scalars > 1:
        raise AmbiguousDataError(
            "Multiple data available\n"
            f"cell data: {possible_scalars_cell}.\n"
            f"point data: {possible_scalars_point}.\n"
            "Set one as active using DataSet.set_active_scalars(name, preference=type)",
        )


def _coerce_transformlike_arg(transform_like: TransformLike) -> NumpyArray[float]:
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
    transform_array: NumpyArray[float] = np.eye(4)
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
            '\t3x3 np.ndarray\n',
        )
    return transform_array


_JSONValueType = Union[
    dict,  # type: ignore[type-arg]
    list,  # type: ignore[type-arg]
    tuple,  # type: ignore[type-arg]
    str,
    int,
    float,
    bool,
    None,
]


# TODO: add generic type annotations 'UserDict[str, _JSONValueType]'
#  once support for Python 3.8 is dropped
class _SerializedDictArray(UserDict, _vtk.vtkStringArray):  # type: ignore[type-arg]
    """Dict-like object with a JSON-serialized string array representation.

    This class behaves just like a regular dict, except its contents
    are represented internally as a JSON-formatted vtkStringArray.
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
    def _string(self) -> str:
        """Get the vtkStringArray string."""
        return ''.join([self.GetValue(i) for i in range(self.GetNumberOfValues())])

    @_string.setter
    def _string(self, str_: str):
        """Set the vtkStringArray to a specified string."""
        self.SetNumberOfValues(0)  # Clear string
        for char in str_:  # Populate string
            self.InsertNextValue(char)

    def _update_string(self):
        """Format dict data as JSON and update the vtkStringArray."""
        data_str = json.dumps(self.data)
        if data_str != self._string:
            self._string = data_str

    def __repr__(self):
        """Return JSON-formatted dict representation."""
        return self._string

    def __init__(self, dict_=None, /, **kwargs):
        # Init from JSON string
        if isinstance(dict_, str):
            dict_ = json.loads(dict_)

        # Init UserDict
        super().__init__(dict_, **kwargs)
        self._update_string()

        # Flag self as a scalar string
        # This is only needed so that the Field DatasetAttributes repr
        # shows this array as `str`
        _set_string_scalar_object_name(self)

    # Override any/all `UserDict` or `MutableMapping` methods which mutate
    # the dictionary. This ensures the serialized string is also updated
    # and synced with the dict

    def __setitem__(self, key, item):
        super().__setitem__(key, item)
        self._update_string()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._update_string()

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        self._update_string() if key != '_string' else None

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._update_string()

    def popitem(self):
        item = super().popitem()
        self._update_string()
        return item

    def pop(self, __key):
        item = super().pop(__key)
        self._update_string()
        return item

    def clear(self):
        super().clear()
        self._update_string()

    def setdefault(self, *args, **kwargs):
        super().setdefault(*args, **kwargs)
        self._update_string()
