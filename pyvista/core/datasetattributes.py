"""Implements DataSetAttributes, which represents and manipulates datasets."""

from __future__ import annotations

import contextlib
import copy
from typing import TYPE_CHECKING
from typing import Any
import warnings

import numpy as np

from . import _vtk_core as _vtk
from .errors import PyVistaDeprecationWarning
from .pyvista_ndarray import pyvista_ndarray
from .utilities.arrays import FieldAssociation
from .utilities.arrays import convert_array
from .utilities.arrays import copy_vtk_array

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator

    from pyvista import DataSet

    from ._typing_core import ArrayLike
    from ._typing_core import MatrixLike
    from ._typing_core import NumpyArray

# from https://vtk.org/doc/nightly/html/vtkDataSetAttributes_8h_source.html
attr_type = [
    'SCALARS',  # 0
    'VECTORS',  # 1
    'NORMALS',  # 2
    'TCOORDS',  # 3
    'TENSORS',  # 4
    'GLOBALIDS',  # 5
    'PEDIGREEIDS',  # 6
    'EDGEFLAG',  # 7
    'TANGENTS',  # 8
    'RATIONALWEIGHTS',  # 9
    'HIGHERORDERDEGREES',  # 10
    '',  # 11  (not an attribute)
]

# used to check if default args have changed in pop
_SENTINEL = pyvista_ndarray([])


class DataSetAttributes(_vtk.VTKObjectWrapper):
    """Python friendly wrapper of ``vtk.DataSetAttributes``.

    This class provides the ability to pick one of the present arrays as the
    currently active array for each attribute type by implementing a
    ``dict`` like interface.

    When adding data arrays but not desiring to set them as active
    scalars or vectors, use :func:`DataSetAttributes.set_array`.

    When adding directional data (such as velocity vectors), use
    :func:`DataSetAttributes.set_vectors`.

    When adding non-directional data (such as temperature values or
    multi-component scalars like RGBA values), use
    :func:`DataSetAttributes.set_scalars`.

    .. versionchanged:: 0.32.0
        The ``[]`` operator no longer allows integers.  Use
        :func:`DataSetAttributes.get_array` to retrieve an array
        using an index.

    Parameters
    ----------
    vtkobject : vtkFieldData
        The vtk object to wrap as a DataSetAttribute, usually an
        instance of ``vtk.vtkCellData``, ``vtk.vtkPointData``, or
        ``vtk.vtkFieldData``.

    dataset : vtkDataSet
        The vtkDataSet containing the vtkobject.

    association : FieldAssociation
        The array association type of the vtkobject.

    Notes
    -----
    When printing out the point arrays, you can see which arrays are
    the active scalars, vectors, normals, and texture coordinates.
    In the arrays list, ``SCALARS`` denotes that these are the active
    scalars, ``VECTORS`` denotes that these arrays are tagged as the
    active vectors data (i.e. data with magnitude and direction) and
    so on.

    Examples
    --------
    Store data with point association in a DataSet.

    >>> import pyvista as pv
    >>> mesh = pv.Cube()
    >>> mesh.point_data['my_data'] = range(mesh.n_points)
    >>> data = mesh.point_data['my_data']
    >>> data
    pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

    Change the data array and show that this is reflected in the DataSet.

    >>> data[:] = 0
    >>> mesh.point_data['my_data']
    pyvista_ndarray([0, 0, 0, 0, 0, 0, 0, 0])

    Remove the array.

    >>> del mesh.point_data['my_data']
    >>> 'my_data' in mesh.point_data
    False

    Print the available arrays from dataset attributes.

    >>> import numpy as np
    >>> mesh = pv.Plane(i_resolution=1, j_resolution=1)
    >>> mesh.point_data.set_array(range(4), 'my-data')
    >>> mesh.point_data.set_array(range(5, 9), 'my-other-data')
    >>> vectors0 = np.random.default_rng().random((4, 3))
    >>> mesh.point_data.set_vectors(vectors0, 'vectors0')
    >>> vectors1 = np.random.default_rng().random((4, 3))
    >>> mesh.point_data.set_vectors(vectors1, 'vectors1')
    >>> mesh.point_data
    pyvista DataSetAttributes
    Association     : POINT
    Active Scalars  : None
    Active Vectors  : vectors1
    Active Texture  : TextureCoordinates
    Active Normals  : Normals
    Contains arrays :
        Normals                 float32    (4, 3)               NORMALS
        TextureCoordinates      float32    (4, 2)               TCOORDS
        my-data                 int64      (4,)
        my-other-data           int64      (4,)
        vectors1                float64    (4, 3)               VECTORS
        vectors0                float64    (4, 3)

    """

    def __init__(
        self,
        vtkobject: _vtk.vtkFieldData,
        dataset: _vtk.vtkDataSet | DataSet,
        association: FieldAssociation,
    ):  # numpydoc ignore=PR01,RT01
        """Initialize DataSetAttributes."""
        super().__init__(vtkobject=vtkobject)
        self.dataset = dataset
        self.association = association

    def __repr__(self) -> str:
        """Printable representation of DataSetAttributes."""
        info = ['pyvista DataSetAttributes']
        array_info = ' None'
        if self:
            lines = []
            for i, (name, array) in enumerate(self.items()):
                if len(name) > 23:
                    name = f'{name[:20]}...'
                try:
                    arr_type = attr_type[self.IsArrayAnAttribute(i)]
                except (IndexError, TypeError, AttributeError):  # pragma: no cover
                    arr_type = ''

                # special treatment for vector data
                if self.association in [FieldAssociation.POINT, FieldAssociation.CELL]:
                    if name == self.active_vectors_name:
                        arr_type = 'VECTORS'
                # special treatment for string field data
                if self.association == FieldAssociation.NONE and isinstance(array, str):
                    dtype = 'str'
                    # Show the string value itself with a max of 20 characters, 18 for string and 2 for quotes
                    val = f'{array[:15]}...' if len(array) > 18 else array
                    line = f'{name[:23]:<24}{dtype!s:<11}"{val}"'
                else:
                    line = (
                        f'{name[:23]:<24}{array.dtype!s:<11}{array.shape!s:<20} {arr_type}'.strip()
                    )
                lines.append(line)
            array_info = '\n    ' + '\n    '.join(lines)

        info.append(f'Association     : {self.association.name}')
        if self.association in [FieldAssociation.POINT, FieldAssociation.CELL]:
            info.append(f'Active Scalars  : {self.active_scalars_name}')
            info.append(f'Active Vectors  : {self.active_vectors_name}')
            info.append(f'Active Texture  : {self.active_texture_coordinates_name}')
            info.append(f'Active Normals  : {self.active_normals_name}')

        info.append(f'Contains arrays :{array_info}')
        return '\n'.join(info)

    def get(self, key: str, value: Any | None = None) -> pyvista_ndarray | None:
        """Return the value of the item with the specified key.

        Parameters
        ----------
        key : str
            Name of the array item you want to return the value from.

        value : Any, optional
            A value to return if the key does not exist.  Default
            is ``None``.

        Returns
        -------
        Any
            Array if the ``key`` exists in the dataset, otherwise
            ``value``.

        Examples
        --------
        Show that the default return value for a non-existent key is
        ``None``.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh.point_data['my_data'] = range(mesh.n_points)
        >>> mesh.point_data.get('my-other-data')

        """
        if key in self:
            return self[key]
        return value

    def __bool__(self) -> bool:
        """Return ``True`` when there are arrays present."""
        return bool(self.GetNumberOfArrays())

    def __getitem__(self, key: str) -> pyvista_ndarray:
        """Implement ``[]`` operator.

        Accepts an array name.
        """
        if not isinstance(key, str):
            raise TypeError('Only strings are valid keys for DataSetAttributes.')
        return self.get_array(key)

    def __setitem__(self, key: str, value: ArrayLike[Any]):  # numpydoc ignore=PR01,RT01
        """Implement setting with the ``[]`` operator."""
        if not isinstance(key, str):
            raise TypeError('Only strings are valid keys for DataSetAttributes.')

        has_arr = key in self
        self.set_array(value, name=key)

        # do not make array active if it already exists.  This covers
        # an inplace update like self.point_data[key] += 1
        if has_arr:
            return

        # make active if not field data and there isn't already an active scalar
        if (
            self.association
            in [
                FieldAssociation.POINT,
                FieldAssociation.CELL,
            ]
            and self.active_scalars_name is None
        ):
            self.active_scalars_name = key

    def __delitem__(self, key: str):
        """Implement del with array name or index."""
        if not isinstance(key, str):
            raise TypeError('Only strings are valid keys for DataSetAttributes.')

        self.remove(key)

    def __contains__(self, name: str) -> bool:
        """Implement the ``in`` operator."""
        return name in self.keys()

    def __iter__(self) -> Iterator[str]:
        """Implement for loop iteration."""
        yield from self.keys()

    def __len__(self) -> int:
        """Return the number of arrays."""
        return self.VTKObject.GetNumberOfArrays()

    @property
    def active_scalars(self) -> pyvista_ndarray | None:
        """Return the active scalars.

        .. versionchanged:: 0.32.0
            Can no longer be used to set the active scalars.  Either use
            :func:`DataSetAttributes.set_scalars` or if the array
            already exists, assign to
            :attr:`pyvista.DataSetAttributes.active_scalars_name`.

        Returns
        -------
        Optional[pyvista_ndarray]
            Active scalars.

        Examples
        --------
        Associate point data to a simple cube mesh and show that the
        active scalars in the point array are the most recently added
        array.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> mesh = pv.Cube()
        >>> mesh.point_data['data0'] = np.arange(mesh.n_points)
        >>> mesh.point_data.active_scalars
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        """
        self._raise_field_data_no_scalars_vectors()
        if self.GetScalars() is not None:
            array = pyvista_ndarray(
                self.GetScalars(),
                dataset=self.dataset,
                association=self.association,
            )
            return self._patch_type(array)
        return None

    @property
    def active_vectors(self) -> NumpyArray[float] | None:
        """Return the active vectors as a pyvista_ndarray.

        .. versionchanged:: 0.32.0
            Can no longer be used to set the active vectors.  Either use
            :func:`DataSetAttributes.set_vectors` or if the array
            already exists, assign to
            :attr:`pyvista.DataSetAttributes.active_vectors_name`.

        Returns
        -------
        Optional[np.ndarray]
            Active vectors as a pyvista_ndarray.

        Examples
        --------
        Associate point data to a simple cube mesh and show that the
        active vectors in the point array are the most recently added
        array.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> mesh = pv.Cube()
        >>> vectors = np.random.default_rng().random((mesh.n_points, 3))
        >>> mesh.point_data.set_vectors(vectors, 'my-vectors')
        >>> vectors_out = mesh.point_data.active_vectors
        >>> vectors_out.shape
        (8, 3)

        """
        self._raise_field_data_no_scalars_vectors()
        vectors = self.GetVectors()
        if vectors is not None:
            return pyvista_ndarray(vectors, dataset=self.dataset, association=self.association)
        return None

    @property
    def valid_array_len(self) -> int | None:
        """Return the length data should be when added to the dataset.

        If there are no restrictions, returns ``None``.

        Returns
        -------
        Optional[int]
            Length data should be when added to the dataset.

        Examples
        --------
        Show that valid array lengths match the number of points and
        cells for point and cell arrays, and there is no length limit
        for field data.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh.n_points, mesh.n_cells
        (8, 6)
        >>> mesh.point_data.valid_array_len
        8
        >>> mesh.cell_data.valid_array_len
        6
        >>> mesh.field_data.valid_array_len is None
        True

        """
        if self.association == FieldAssociation.POINT:
            return self.dataset.GetNumberOfPoints()
        if self.association == FieldAssociation.CELL:
            return self.dataset.GetNumberOfCells()
        return None

    @property
    def active_t_coords(self) -> pyvista_ndarray | None:
        """Return the active texture coordinates array.

        .. deprecated:: 0.43.0
            Use :func:`DataSetAttributes.active_texture_coordinates` instead.

        Returns
        -------
        pyvista.pyvista_ndarray
            Array of the active texture coordinates.

        """
        warnings.warn(
            "Use of `DataSetAttributes.active_t_coords` is deprecated. Use `DataSetAttributes.active_texture_coordinates` instead.",
            PyVistaDeprecationWarning,
        )
        return self.active_texture_coordinates

    @active_t_coords.setter
    def active_t_coords(self, t_coords: NumpyArray[float]):  # numpydoc ignore=GL08
        """Set the active texture coordinates array.

        .. deprecated:: 0.43.0
            Use :func:`DataSetAttributes.active_texture_coordinates` instead.

        Parameters
        ----------
        t_coords : np.ndarray
            Array of the active texture coordinates.

        """
        warnings.warn(
            "Use of `DataSetAttributes.active_t_coords` is deprecated. Use `DataSetAttributes.active_texture_coordinates` instead.",
            PyVistaDeprecationWarning,
        )
        self.active_texture_coordinates = t_coords  # type: ignore[assignment]

    @property
    def active_t_coords_name(self) -> str | None:
        """Return the name of the active texture coordinates array.

        .. deprecated:: 0.43.0
            Use :func:`DataSetAttributes.active_texture_coordinates_name` instead.

        Returns
        -------
        Optional[str]
            Name of the active texture coordinates array.

        """
        warnings.warn(
            "Use of `DataSetAttributes.active_t_coords_name` is deprecated. Use `DataSetAttributes.active_texture_coordinates_name` instead.",
            PyVistaDeprecationWarning,
        )
        return self.active_texture_coordinates_name

    @active_t_coords_name.setter
    def active_t_coords_name(self, name: str) -> None:  # numpydoc ignore=GL08
        """Set the name of the active texture coordinates array.

        .. deprecated:: 0.43.0
            Use :func:`DataSetAttributes.active_texture_coordinates_name` instead.

        Parameters
        ----------
        name : str
            Name of the active texture coordinates array.

        """
        warnings.warn(
            "Use of `DataSetAttributes.active_t_coords_name` is deprecated. Use `DataSetAttributes.active_texture_coordinates_name` instead.",
            PyVistaDeprecationWarning,
        )
        self.active_texture_coordinates_name = name

    def get_array(self, key: str | int) -> pyvista_ndarray:
        """Get an array in this object.

        Parameters
        ----------
        key : str | int
            The name or index of the array to return.  Arrays are
            ordered within VTK DataSetAttributes, and this feature is
            mirrored here.

        Returns
        -------
        pyvista.pyvista_ndarray
            Returns a :class:`pyvista.pyvista_ndarray`.

        Raises
        ------
        KeyError
            If the key does not exist.

        Notes
        -----
        This is provided since arrays are ordered within VTK and can
        be indexed via an int.  When getting an array, you can just
        use the key of the array with the ``[]`` operator with the
        name of the array.

        Examples
        --------
        Store data with point association in a DataSet.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh.clear_data()
        >>> mesh.point_data['my_data'] = range(mesh.n_points)

        Access using an index.

        >>> mesh.point_data.get_array(0)
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        Access using a key.

        >>> mesh.point_data.get_array('my_data')
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        """
        self._raise_index_out_of_bounds(index=key)
        vtk_arr = self.GetArray(key)
        if vtk_arr is None:
            vtk_arr = self.GetAbstractArray(key)
            if vtk_arr is None:
                raise KeyError(f'{key}')
        narray = pyvista_ndarray(vtk_arr, dataset=self.dataset, association=self.association)
        return self._patch_type(narray)

    def _patch_type(self, narray):
        """Check if array needs to be represented as a different type."""
        name = narray.VTKObject.GetName()
        if name in self.dataset._association_bitarray_names[self.association.name]:
            narray = narray.view(np.bool_)
        elif name in self.dataset._association_complex_names[self.association.name]:
            if narray.dtype == np.float32:
                narray = narray.view(np.complex64)
            if narray.dtype == np.float64:
                narray = narray.view(np.complex128)
            # remove singleton dimensions to match the behavior of the rest of 1D
            # VTK arrays
            narray = narray.squeeze()
        elif (
            narray.association == FieldAssociation.NONE
            and np.issubdtype(narray.dtype, np.str_)
            and narray.ndim == 0
        ):
            # For field data with a string scalar, return the string itself instead of a scalar array
            narray = narray.tolist()

        return narray

    def set_array(self, data: ArrayLike[float], name: str, deep_copy=False) -> None:
        """Add an array to this object.

        Use this method when adding arrays to the DataSet.  If
        needed, these arrays can later be assigned to become the
        active scalars, vectors, normals, or texture coordinates with:

        * :attr:`active_scalars_name <DataSetAttributes.active_scalars_name>`
        * :attr:`active_vectors_name <DataSetAttributes.active_vectors_name>`
        * :attr:`active_normals_name <DataSetAttributes.active_normals_name>`
        * :attr:`active_texture_coordinates_name <DataSetAttributes.active_texture_coordinates_name>`

        Parameters
        ----------
        data : ArrayLike[float]
            Array of data.

        name : str
            Name to assign to the data.  If this name already exists,
            it will be overwritten.

        deep_copy : bool, optional
            When ``True`` makes a full copy of the array.

        Notes
        -----
        You can simply use the ``[]`` operator to add an array to the
        dataset.  Note that this will automatically become the active
        scalars.

        Examples
        --------
        Add a point array to a mesh.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> data = range(mesh.n_points)
        >>> mesh.point_data.set_array(data, 'my-data')
        >>> mesh.point_data['my-data']
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        Add a cell array to a mesh.

        >>> cell_data = range(mesh.n_cells)
        >>> mesh.cell_data.set_array(cell_data, 'my-data')
        >>> mesh.cell_data['my-data']
        pyvista_ndarray([0, 1, 2, 3, 4, 5])

        Add field data to a mesh.

        >>> field_data = range(3)
        >>> mesh.field_data.set_array(field_data, 'my-data')
        >>> mesh.field_data['my-data']
        pyvista_ndarray([0, 1, 2])

        """
        if not isinstance(name, str):
            raise TypeError('`name` must be a string')

        vtk_arr = self._prepare_array(data, name, deep_copy)
        self.VTKObject.AddArray(vtk_arr)
        self.VTKObject.Modified()

    def set_scalars(self, scalars: ArrayLike[float], name='scalars', deep_copy=False):
        """Set the active scalars of the dataset with an array.

        In VTK and PyVista, scalars are a quantity that has no
        direction.  This can include data with multiple components
        (such as RGBA values) or just one component (such as
        temperature data).

        See :func:`DataSetAttributes.set_vectors` when adding arrays
        that contain magnitude and direction.

        Parameters
        ----------
        scalars : ArrayLike[float]
            Array of data.

        name : str, default: 'scalars'
            Name to assign the scalars.

        deep_copy : bool, default: False
            When ``True`` makes a full copy of the array.

        Notes
        -----
        When adding directional data (such as velocity vectors), use
        :func:`DataSetAttributes.set_vectors`.

        Complex arrays will be represented internally as a 2 component float64
        array. This is due to limitations of VTK's native datatypes.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh.clear_data()
        >>> scalars = range(mesh.n_points)
        >>> mesh.point_data.set_scalars(scalars, 'my-scalars')
        >>> mesh.point_data
        pyvista DataSetAttributes
        Association     : POINT
        Active Scalars  : my-scalars
        Active Vectors  : None
        Active Texture  : None
        Active Normals  : None
        Contains arrays :
            my-scalars              int64      (8,)                 SCALARS

        """
        vtk_arr = self._prepare_array(scalars, name, deep_copy)
        self.VTKObject.SetScalars(vtk_arr)
        self.VTKObject.Modified()

    def set_vectors(self, vectors: MatrixLike[float], name: str, deep_copy=False):
        """Set the active vectors of this data attribute.

        Vectors are a quantity that has magnitude and direction, such
        as normal vectors or a velocity field.

        The vectors data must contain three components per cell or point.  Use
        :func:`DataSetAttributes.set_scalars` when adding non-directional data.

        Parameters
        ----------
        vectors : MatrixLike
            Data shaped ``(n, 3)`` where n matches the number of points or cells.

        name : str
            Name of the vectors.

        deep_copy : bool, default: False
            When ``True`` makes a full copy of the array.  When ``False``, the
            data references the original array without copying it.

        Notes
        -----
        PyVista and VTK treats vectors and scalars differently when performing
        operations. Vector data, unlike scalar data, is rotated along with the
        geometry when the DataSet is passed through a transformation filter.

        When adding non-directional data (such temperature values or
        multi-component scalars like RGBA values), you can also use
        :func:`DataSetAttributes.set_scalars`.

        Examples
        --------
        Add random vectors to a mesh as point data.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> mesh = pv.Cube()
        >>> mesh.clear_data()
        >>> vectors = np.random.default_rng().random((mesh.n_points, 3))
        >>> mesh.point_data.set_vectors(vectors, 'my-vectors')
        >>> mesh.point_data
        pyvista DataSetAttributes
        Association     : POINT
        Active Scalars  : None
        Active Vectors  : my-vectors
        Active Texture  : None
        Active Normals  : None
        Contains arrays :
            my-vectors              float64    (8, 3)               VECTORS

        """
        # prepare the array and add an attribute so that we can track this as a vector
        vtk_arr = self._prepare_array(vectors, name, deep_copy)

        n_comp = vtk_arr.GetNumberOfComponents()
        if n_comp != 3:
            raise ValueError(f'Vector array should contain 3 components, got {n_comp}')

        # check if there are current vectors, if so, we need to keep
        # this array around since setting active vectors will remove
        # this array.
        current_vectors = self.GetVectors()

        # now we can set the active vectors and add back in the old vectors as an array
        self.VTKObject.SetVectors(vtk_arr)
        if current_vectors is not None:
            self.VTKObject.AddArray(current_vectors)

        self.VTKObject.Modified()

    def _prepare_array(
        self,
        data: ArrayLike[float],
        name: str,
        deep_copy: bool,
    ) -> _vtk.vtkDataArray:  # numpydoc ignore=PR01,RT01
        """Prepare an array to be added to this dataset.

        Notes
        -----
        This method also adds metadata necessary for VTK to support non-VTK
        compatible datatypes like ``numpy.complex128`` or ``numpy.bool_`` to
        the underlying dataset.

        """
        if data is None:
            raise TypeError('``data`` cannot be None.')

        # convert to numpy type if necessary
        data = np.asanyarray(data)

        if self.association == FieldAssociation.POINT:
            array_len = self.dataset.GetNumberOfPoints()
        elif self.association == FieldAssociation.CELL:
            array_len = self.dataset.GetNumberOfCells()
        else:
            array_len = 1 if data.ndim == 0 else data.shape[0]

        if np.issubdtype(data.dtype, np.str_) and data.ndim == 0:
            pass  # Do not reshape string scalars
        else:
            # Fixup input array length for scalar input
            if np.ndim(data) == 0:
                tmparray = np.empty(array_len, dtype=data.dtype)
                tmparray.fill(data)
                data = tmparray
            if data.shape[0] != array_len:
                raise ValueError(
                    f'data length of ({data.shape[0]}) != required length ({array_len})',
                )

        # attempt to reuse the existing pointer to underlying VTK data
        if isinstance(data, pyvista_ndarray):
            # pyvista_ndarray already contains the reference to the vtk object
            # pyvista needs to use the copy of this object rather than wrapping
            # the array (which leaves a C++ pointer uncollected.
            if data.VTKObject is not None:
                # VTK doesn't support strides, therefore we can't directly
                # point to the underlying object
                if data.flags.c_contiguous:
                    # no reason to return a shallow copy if the array and name
                    # are identical, just return the underlying array name
                    if not deep_copy and isinstance(name, str) and data.VTKObject.GetName() == name:
                        return data.VTKObject

                    vtk_arr = copy_vtk_array(data.VTKObject, deep=deep_copy)
                    if isinstance(name, str):
                        vtk_arr.SetName(name)
                    return vtk_arr

        # reset data association
        if name in self.dataset._association_bitarray_names[self.association.name]:
            self.dataset._association_bitarray_names[self.association.name].remove(name)
        if name in self.dataset._association_complex_names[self.association.name]:
            self.dataset._association_complex_names[self.association.name].remove(name)

        if data.dtype == np.bool_:
            self.dataset._association_bitarray_names[self.association.name].add(name)
            data = data.view(np.uint8)
        elif np.issubdtype(data.dtype, np.complexfloating):
            if data.dtype not in (np.complex64, np.complex128):
                raise ValueError(
                    'Only numpy.complex64 or numpy.complex128 is supported when '
                    'setting dataset attributes',
                )

            if data.ndim != 1:
                if data.shape[1] != 1:
                    raise ValueError('Complex data must be single dimensional.')
            self.dataset._association_complex_names[self.association.name].add(name)

            # complex data is stored internally as a contiguous 2 component
            # float arrays
            if data.dtype == np.complex64:
                data = data.view(np.float32).reshape(-1, 2)
            else:
                data = data.view(np.float64).reshape(-1, 2)

        shape = data.shape
        if data.ndim == 3:
            # Array of matrices. We need to make sure the order in
            # memory is right.  If row major (C/C++),
            # transpose. VTK wants column major (Fortran order). The deep
            # copy later will make sure that the array is contiguous.
            # If column order but not contiguous, transpose so that the
            # deep copy below does not happen.
            size = data.dtype.itemsize
            if (data.strides[1] / size == 3 and data.strides[2] / size == 1) or (
                data.strides[1] / size == 1
                and data.strides[2] / size == 3
                and not data.flags.contiguous
            ):
                data = data.transpose(0, 2, 1)

        # If array is not contiguous, make a deep copy that is contiguous
        if not data.flags.contiguous:
            data = np.ascontiguousarray(data)

        # Flatten array of matrices to array of vectors
        if len(shape) == 3:
            data = data.reshape(shape[0], shape[1] * shape[2])

        # Swap bytes from big to little endian.
        if data.dtype.byteorder == '>':
            data = data.byteswap(inplace=True)

        # this handles the case when an input array is directly added to the
        # output. We want to make sure that the array added to the output is not
        # referring to the input dataset.
        copy = pyvista_ndarray(data)

        return convert_array(copy, name, deep=deep_copy)

    def remove(self, key: str) -> None:
        """Remove an array.

        Parameters
        ----------
        key : str
            The name of the array to remove.

        Notes
        -----
        You can also use the ``del`` statement.

        Examples
        --------
        Add a point data array to a DataSet and then remove it.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh.point_data['my_data'] = range(mesh.n_points)
        >>> mesh.point_data.remove('my_data')

        Show that the array no longer exists in ``point_data``.

        >>> 'my_data' in mesh.point_data
        False

        """
        if not isinstance(key, str):
            raise TypeError('Only strings are valid keys for DataSetAttributes.')

        if key not in self:
            raise KeyError(f'{key} not present.')

        with contextlib.suppress(KeyError):
            self.dataset._association_bitarray_names[self.association.name].remove(key)
        self.VTKObject.RemoveArray(key)
        self.VTKObject.Modified()

    def pop(self, key: str, default=_SENTINEL) -> pyvista_ndarray:
        """Remove an array and return it.

        Parameters
        ----------
        key : str
            The name of the array to remove and return.

        default : Any, optional
            If default is not given and key is not in the dictionary,
            a KeyError is raised.

        Returns
        -------
        pyvista_ndarray
            Requested array.

        Examples
        --------
        Add a point data array to a DataSet and then remove it.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh.point_data['my_data'] = range(mesh.n_points)
        >>> mesh.point_data.pop('my_data')
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        Show that the array no longer exists in ``point_data``.

        >>> 'my_data' in mesh.point_data
        False

        """
        if not isinstance(key, str):
            raise TypeError('Only strings are valid keys for DataSetAttributes.')

        if key not in self:
            if default is _SENTINEL:
                raise KeyError(f'{key} not present.')
            return default

        narray = self.get_array(key)

        self.remove(key)
        return narray

    def items(self) -> list[tuple[str, pyvista_ndarray]]:
        """Return a list of (array name, array value) tuples.

        Returns
        -------
        list
            List of keys and values.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh.clear_data()
        >>> mesh.cell_data['data0'] = [0] * mesh.n_cells
        >>> mesh.cell_data['data1'] = range(mesh.n_cells)
        >>> mesh.cell_data.items()
        [('data0', pyvista_ndarray([0, 0, 0, 0, 0, 0])), ('data1', pyvista_ndarray([0, 1, 2, 3, 4, 5]))]

        """
        return list(zip(self.keys(), self.values()))

    def keys(self) -> list[str]:
        """Return the names of the arrays as a list.

        Returns
        -------
        list
            List of keys.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.clear_data()
        >>> mesh.point_data['data0'] = [0] * mesh.n_points
        >>> mesh.point_data['data1'] = range(mesh.n_points)
        >>> mesh.point_data.keys()
        ['data0', 'data1']

        """
        keys = []
        for i in range(self.GetNumberOfArrays()):
            array = self.VTKObject.GetAbstractArray(i)
            name = array.GetName()
            if name:
                keys.append(name)
            else:  # pragma: no cover
                # Assign this array a name
                name = f'Unnamed_{i}'
                array.SetName(name)
                keys.append(name)
        return keys

    def values(self) -> list[pyvista_ndarray]:
        """Return the arrays as a list.

        Returns
        -------
        list
            List of arrays.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh.clear_data()
        >>> mesh.cell_data['data0'] = [0] * mesh.n_cells
        >>> mesh.cell_data['data1'] = range(mesh.n_cells)
        >>> mesh.cell_data.values()
        [pyvista_ndarray([0, 0, 0, 0, 0, 0]), pyvista_ndarray([0, 1, 2, 3, 4, 5])]

        """
        return [self.get_array(name) for name in self.keys()]

    def clear(self):
        """Remove all arrays in this object.

        Examples
        --------
        Add an array to ``point_data`` to a DataSet and then clear the
        point_data.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh.clear_data()
        >>> mesh.point_data['my_data'] = range(mesh.n_points)
        >>> len(mesh.point_data)
        1
        >>> mesh.point_data.clear()
        >>> len(mesh.point_data)
        0

        """
        for array_name in self.keys():
            self.remove(key=array_name)

    def update(self, array_dict: dict[str, NumpyArray[float]] | DataSetAttributes):
        """Update arrays in this object from another dictionary or dataset attributes.

        For each key, value given, add the pair. If it already exists, replace
        it with the new array. These arrays will be copied.

        Parameters
        ----------
        array_dict : dict, DataSetAttributes
            A dictionary of ``(array name, :class:`numpy.ndarray`)`` or a
            :class:`pyvista.DataSetAttributes`.

        Examples
        --------
        Add two arrays to ``point_data`` using ``update``.

        >>> import numpy as np
        >>> from pyvista import examples
        >>> mesh = examples.load_uniform()
        >>> n = len(mesh.point_data)
        >>> arrays = {
        ...     'foo': np.arange(mesh.n_points),
        ...     'rand': np.random.default_rng().random(mesh.n_points),
        ... }
        >>> mesh.point_data.update(arrays)
        >>> mesh.point_data
        pyvista DataSetAttributes
        Association     : POINT
        Active Scalars  : Spatial Point Data
        Active Vectors  : None
        Active Texture  : None
        Active Normals  : None
        Contains arrays :
            Spatial Point Data      float64    (1000,)              SCALARS
            foo                     int64      (1000,)
            rand                    float64    (1000,)

        """
        for name, array in array_dict.items():
            self[name] = array.copy() if hasattr(array, 'copy') else copy.copy(array)

    def _raise_index_out_of_bounds(self, index: Any):
        """Raise a KeyError if array index is out of bounds."""
        if isinstance(index, int):
            max_index = self.VTKObject.GetNumberOfArrays()
            if not 0 <= index < max_index:
                raise KeyError(f'Array index ({index}) out of range [0, {max_index - 1}]')

    def _raise_field_data_no_scalars_vectors(self):
        """Raise a ``TypeError`` if FieldData."""
        if self.association == FieldAssociation.NONE:
            raise TypeError('FieldData does not have active scalars or vectors.')

    @property
    def active_scalars_name(self) -> str | None:
        """Return name of the active scalars.

        Returns
        -------
        Optional[str]
            Name of the active scalars.

        Examples
        --------
        Add two arrays to the mesh point data. Note how the first array becomes
        the active scalars since the ``mesh`` contained no scalars.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.point_data['my_data'] = range(mesh.n_points)
        >>> mesh.point_data['my_other_data'] = range(mesh.n_points)
        >>> mesh.point_data.active_scalars_name
        'my_data'

        Set the name of the active scalars.

        >>> mesh.point_data.active_scalars_name = 'my_other_data'
        >>> mesh.point_data.active_scalars_name
        'my_other_data'

        """
        if self.GetScalars() is not None:
            name = self.GetScalars().GetName()
            if name is None:
                # Getting the keys has the side effect of naming "unnamed" arrays
                self.keys()
                name = self.GetScalars().GetName()
            return str(name)
        return None

    @active_scalars_name.setter
    def active_scalars_name(self, name: str) -> None:  # numpydoc ignore=GL08
        """Set name of the active scalars.

        Parameters
        ----------
        name : str
            Name of the active scalars.

        """
        # permit setting no active scalars
        if name is None:
            self.SetActiveScalars(None)
            return
        self._raise_field_data_no_scalars_vectors()
        dtype = self[name].dtype
        # only vtkDataArray subclasses can be set as active attributes
        if np.issubdtype(dtype, np.number) or np.issubdtype(dtype, bool):
            self.SetActiveScalars(name)

    @property
    def active_vectors_name(self) -> str | None:
        """Return name of the active vectors.

        Returns
        -------
        Optional[str]
            Name of the active vectors.

        Examples
        --------
        >>> import pyvista as pv
        >>> import numpy as np
        >>> mesh = pv.Sphere()
        >>> mesh.point_data.set_vectors(
        ...     np.random.default_rng().random((mesh.n_points, 3)),
        ...     'my-vectors',
        ... )
        >>> mesh.point_data.active_vectors_name
        'my-vectors'

        """
        if self.GetVectors() is not None:
            return str(self.GetVectors().GetName())
        return None

    @active_vectors_name.setter
    def active_vectors_name(self, name: str) -> None:  # numpydoc ignore=GL08
        """Set name of the active vectors.

        Parameters
        ----------
        name : str
            Name of the active vectors.

        """
        # permit setting no active
        if name is None:
            self.SetActiveVectors(None)
            return
        self._raise_field_data_no_scalars_vectors()
        if name not in self:
            raise KeyError(f'DataSetAttribute does not contain "{name}"')
        # verify that the array has the correct number of components
        n_comp = self.GetArray(name).GetNumberOfComponents()
        if n_comp != 3:
            raise ValueError(f'{name} needs 3 components, has ({n_comp})')
        self.SetActiveVectors(name)

    def __eq__(self, other: object) -> bool:
        """Test dict-like equivalency."""
        # here we check if other is the same class or a subclass of self.
        if not isinstance(other, type(self)):
            return False

        if set(self.keys()) != set(other.keys()):
            return False

        # verify the value of the arrays
        for key, value in other.items():
            if not np.array_equal(value, self[key]):
                return False

        # check the name of the active attributes
        if self.association != FieldAssociation.NONE:
            for name in ['scalars', 'vectors', 'texture_coordinates', 'normals']:
                attr = f'active_{name}_name'
                if getattr(other, attr) != getattr(self, attr):
                    return False

        return True

    @property
    def active_normals(self) -> pyvista_ndarray | None:
        """Return the normals.

        Returns
        -------
        pyvista_ndarray
            Normals of this dataset attribute. ``None`` if no normals have been
            set.

        Notes
        -----
        Field data will have no normals.

        Examples
        --------
        First, compute cell normals.

        >>> import pyvista as pv
        >>> mesh = pv.Plane(i_resolution=1, j_resolution=1)
        >>> mesh.point_data
        pyvista DataSetAttributes
        Association     : POINT
        Active Scalars  : None
        Active Vectors  : None
        Active Texture  : TextureCoordinates
        Active Normals  : Normals
        Contains arrays :
            Normals                 float32    (4, 3)               NORMALS
            TextureCoordinates      float32    (4, 2)               TCOORDS

        >>> mesh.point_data.active_normals
        pyvista_ndarray([[0., 0., 1.],
                         [0., 0., 1.],
                         [0., 0., 1.],
                         [0., 0., 1.]], dtype=float32)

        Assign normals to the cell arrays.  An array will be added
        named ``"Normals"``.

        >>> mesh.cell_data.active_normals = [[0.0, 0.0, 1.0]]
        >>> mesh.cell_data
        pyvista DataSetAttributes
        Association     : CELL
        Active Scalars  : None
        Active Vectors  : None
        Active Texture  : None
        Active Normals  : Normals
        Contains arrays :
            Normals                 float64    (1, 3)               NORMALS

        """
        self._raise_no_normals()
        vtk_normals = self.GetNormals()
        if vtk_normals is not None:
            return pyvista_ndarray(vtk_normals, dataset=self.dataset, association=self.association)
        return None

    @active_normals.setter
    def active_normals(self, normals: MatrixLike[float]):  # numpydoc ignore=GL08
        """Set the normals.

        Parameters
        ----------
        normals : MatrixLike
            Normals of this dataset attribute.

        """
        self._raise_no_normals()
        normals = np.asarray(normals)
        if normals.ndim != 2:
            raise ValueError('Normals must be a 2-dimensional array')
        valid_length = self.valid_array_len
        if normals.shape[0] != valid_length:
            raise ValueError(
                f'Number of normals ({normals.shape[0]}) must match number of points ({valid_length})',
            )
        if normals.shape[1] != 3:
            raise ValueError(f'Normals must have exactly 3 components, not ({normals.shape[1]})')

        vtkarr = _vtk.numpyTovtkDataArray(normals, name='Normals')
        self.SetNormals(vtkarr)
        self.Modified()

    @property
    def active_normals_name(self) -> str | None:
        """Return the name of the normals array.

        Returns
        -------
        str
            Name of the active normals array.

        Examples
        --------
        First, compute cell normals.

        >>> import pyvista as pv
        >>> mesh = pv.Plane(i_resolution=1, j_resolution=1)
        >>> mesh_w_normals = mesh.compute_normals()
        >>> mesh_w_normals.point_data.active_normals_name
        'Normals'

        """
        self._raise_no_normals()
        if self.GetNormals() is not None:
            return str(self.GetNormals().GetName())
        return None

    @active_normals_name.setter
    def active_normals_name(self, name: str) -> None:  # numpydoc ignore=GL08
        """Set the name of the normals array.

        Parameters
        ----------
        name : str
            Name of the active normals array.

        """
        # permit setting no active
        if name is None:
            self.SetActiveNormals(None)
            return
        self._raise_no_normals()
        self.SetActiveNormals(name)

    def _raise_no_normals(self):
        """Raise AttributeError when attempting access normals for field data."""
        if self.association == FieldAssociation.NONE:
            raise AttributeError('FieldData does not have active normals.')

    def _raise_no_texture_coordinates(self):
        """Raise AttributeError when attempting access texture_coordinates for field data."""
        if self.association == FieldAssociation.NONE:
            raise AttributeError('FieldData does not have active texture coordinates.')

    @property
    def active_texture_coordinates(self) -> pyvista_ndarray | None:
        """Return the active texture coordinates array.

        Returns
        -------
        pyvista.pyvista_ndarray
            Array of the active texture coordinates.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh.point_data.active_texture_coordinates
        pyvista_ndarray([[ 0.,  0.],
                         [ 1.,  0.],
                         [ 1.,  1.],
                         [ 0.,  1.],
                         [-0.,  0.],
                         [-0.,  1.],
                         [-1.,  1.],
                         [-1.,  0.]], dtype=float32)

        """
        self._raise_no_texture_coordinates()
        texture_coordinates = self.GetTCoords()
        if texture_coordinates is not None:
            return pyvista_ndarray(
                texture_coordinates,
                dataset=self.dataset,
                association=self.association,
            )
        return None

    @active_texture_coordinates.setter
    def active_texture_coordinates(
        self,
        texture_coordinates: NumpyArray[float],
    ):  # numpydoc ignore=GL08
        """Set the active texture coordinates array.

        Parameters
        ----------
        texture_coordinates : np.ndarray
            Array of the active texture coordinates.

        """
        self._raise_no_texture_coordinates()
        if not isinstance(texture_coordinates, np.ndarray):
            raise TypeError('Texture coordinates must be a numpy array')
        if texture_coordinates.ndim != 2:
            raise ValueError('Texture coordinates must be a 2-dimensional array')
        valid_length = self.valid_array_len
        if texture_coordinates.shape[0] != valid_length:
            raise ValueError(
                f'Number of texture coordinates ({texture_coordinates.shape[0]}) must match number of points ({valid_length})',
            )
        if texture_coordinates.shape[1] != 2:
            raise ValueError(
                f'Texture coordinates must only have 2 components, not ({texture_coordinates.shape[1]})',
            )
        vtkarr = _vtk.numpyTovtkDataArray(texture_coordinates, name='Texture Coordinates')
        self.SetTCoords(vtkarr)
        self.Modified()

    @property
    def active_texture_coordinates_name(self) -> str | None:
        """Return the name of the active texture coordinates array.

        Returns
        -------
        Optional[str]
            Name of the active texture coordinates array.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh.point_data.active_texture_coordinates_name
        'TCoords'

        """
        self._raise_no_texture_coordinates()
        if self.GetTCoords() is not None:
            return str(self.GetTCoords().GetName())
        return None

    @active_texture_coordinates_name.setter
    def active_texture_coordinates_name(self, name: str) -> None:  # numpydoc ignore=GL08
        """Set the name of the active texture coordinates array.

        Parameters
        ----------
        name : str
            Name of the active texture coordinates array.

        """
        if name is None:
            self.SetActiveTCoords(None)
            return

        self._raise_no_texture_coordinates()
        dtype = self[name].dtype
        # only vtkDataArray subclasses can be set as active attributes
        if np.issubdtype(dtype, np.number) or np.issubdtype(dtype, bool):
            self.SetActiveTCoords(name)
