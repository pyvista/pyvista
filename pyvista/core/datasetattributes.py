"""Implements DataSetAttributes, which represents and manipulates datasets."""

import weakref
import warnings
from collections.abc import Iterable

import numpy as np
from typing import Union, Iterator, Optional, List, Tuple, Dict, Sequence, Any

from pyvista import _vtk
import pyvista.utilities.helpers as helpers
from pyvista.utilities.helpers import FieldAssociation
from pyvista.utilities.misc import PyvistaDeprecationWarning
from .pyvista_ndarray import pyvista_ndarray

from .._typing import Number

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


class DataSetAttributes(_vtk.VTKObjectWrapper):
    """Python friendly wrapper of ``vtk.DataSetAttributes``.

    Implement a ``dict`` like interface for interacting with
    vtkDataArrays while also including functionality adding vector and
    scalar data.

    When adding data arrays but not desiring to set them as active
    scalars or vectors, use :func:`DataSetAttributes.set_array`.

    When adding directional data (such as velocity vectors), use
    :func:`DataSetAttributes.set_vectors`.

    When adding non-directional data (such temperature values or
    multi-component scalars like RGBA values), use
    :func:`DataSetAttributes.set_scalars`.

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

    Examples
    --------
    Store data with point association in a DataSet

    >>> import pyvista
    >>> mesh = pyvista.Cube().clean()
    >>> mesh.point_arrays['my_data'] = range(mesh.n_points)
    >>> data = mesh.point_arrays['my_data']
    >>> data
    pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

    Change the data array and show that this is reflected in the DataSet.

    >>> data[:] = 0
    >>> mesh.point_arrays['my_data']
    pyvista_ndarray([0, 0, 0, 0, 0, 0, 0, 0])

    Print the available arrays from dataset attributes.

    >>> import numpy as np
    >>> mesh = pyvista.Plane(i_resolution=1, j_resolution=1)
    >>> mesh.point_arrays.set_array(range(4), 'my-data')
    >>> mesh.point_arrays.set_array(range(5, 9), 'my-other-data')
    >>> vectors0 = np.random.random((4, 3))
    >>> mesh.point_arrays.set_vectors(vectors0, 'vectors0')
    >>> vectors1 = np.random.random((4, 3))
    >>> mesh.point_arrays.set_vectors(vectors1, 'vectors1')
    >>> mesh.point_arrays
    pyvista DataSetAttributes
    Association     : POINT
    Active Scalars  : TextureCoordinates
    Active Vectors  : vectors1
    Active Texture  : TextureCoordinates
    Contains arrays :
        Normals                 float32  (4, 3)               NORMALS
        TextureCoordinates      float32  (4, 2)               SCALARS
        my-data                 int64    (4,)
        my-other-data           int64    (4,)
        vectors1                float64  (4, 3)               VECTOR
        vectors0                float64  (4, 3)

    Notes
    -----
    When printing out the point arrays, you can see which arrays are
    the active scalars, vectors, normals, and textures.  In the arrays
    list, ``SCALARS`` denotes that these are the active scalars, and
    vectors denotes that these arrays are tagged as vectors data
    (i.e. data with magnitude and direction).

    """

    def __init__(self, vtkobject: _vtk.vtkFieldData, dataset: _vtk.vtkDataSet,
                 association: FieldAssociation):
        """Initialize DataSetAttributes."""
        super().__init__(vtkobject=vtkobject)
        self.dataset = dataset
        self.association = association

    def __repr__(self) -> str:
        """Printable representation of DataSetAttributes."""
        array_info = ' None'
        if len(self):
            lines = []
            for i, (name, array) in enumerate(self.items()):
                if len(name) > 20:
                    name = f'{name[:20]}...'
                vtk_arr = array.VTKObject
                try:
                    arr_type = attr_type[self.IsArrayAnAttribute(i)]
                except (IndexError, TypeError):  # pragma: no cover
                    arr_type = ''

                # special treatment for vector data
                if name == self.active_vectors_name:
                    arr_type = 'VECTOR'

                line = f'{name[:23]:<24}{str(array.dtype):<9}{str(array.shape):<20} {arr_type}'.strip()
                lines.append(line)
            array_info = '\n    ' + '\n    '.join(lines)

        if self.association in [FieldAssociation.POINT, FieldAssociation.CELL]:
            scalar_info = f'Active Scalars  : {self.active_scalars_name}\n'
            vector_info = f'Active Vectors  : {self.active_vectors_name}\n'
            texture_info = f'Active Texture  : {self.active_texture_name}\n'
        else:
            scalar_info = ''
            vector_info = ''
            texture_info = ''

        return 'pyvista DataSetAttributes\n' \
               f'Association     : {self.association.name}\n' \
               f'{scalar_info}' \
               f'{vector_info}' \
               f'{texture_info}' \
               f'Contains arrays :{array_info}' \

    def get(self, key: str, value: Optional[Any] = None) -> Optional[pyvista_ndarray]:
        """Return the value of the item with the specified key.

        Parameters
        ----------
        key : str
            Name of the array item you want to return the value from.

        value : anything, optional
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

        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.point_arrays['my_data'] = range(mesh.n_points)
        >>> mesh.point_arrays.get('my-other-data')

        """
        if key in self:
            return self[key]
        return value

    def __getitem__(self, key: Union[str]) -> pyvista_ndarray:
        """Implement [] operator.

        Accepts an array name.
        """
        return self.get_array(key)

    def __setitem__(self, key: str, value: np.ndarray):
        """Implement setting with the ``[]`` operator."""
        has_arr = key in self
        self.set_array(value, name=key)

        # do not make array active if it already exists.  This covers
        # an inplace update like self.point_arrays[key] += 1
        if has_arr:
            return

        # make active if not field data
        if self.association in [FieldAssociation.POINT, FieldAssociation.CELL]:
            self.active_scalars_name = key

    def __delitem__(self, key: str):
        """Implement del with array name or index."""
        self.remove(key)

    def __contains__(self, name: str) -> bool:
        """Implement the ``in`` operator."""
        return name in self.keys()

    def __iter__(self) -> Iterator[str]:
        """Implement for loop iteration."""
        for array in self.keys():
            yield array

    def __len__(self) -> int:
        """Return the number of arrays."""
        return self.VTKObject.GetNumberOfArrays()

    @property
    def active_scalars(self) -> Optional[pyvista_ndarray]:
        """Return the active scalars.

        .. versionchanged:: 0.32.0
            Can no longer used to set the active scalars.  Either use
            :func:`DataSetAttributes.set_scalars` or if the array
            already exists, use
            :attr:`DataSetAttribute.active_scalars_name`.

        Examples
        --------
        Associate point data to a simple cube mesh and show that the
        active scalars in the point array are the most recently added
        array.

        >>> import pyvista
        >>> import numpy as np
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.point_arrays['data0'] = np.arange(mesh.n_points)
        >>> mesh.point_arrays.active_scalars
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        """
        self._raise_field_data_no_scalars_vectors()
        if self.GetScalars() is not None:
            return pyvista_ndarray(self.GetScalars(), dataset=self.dataset,
                                   association=self.association)
        return None

    @active_scalars.setter
    def active_scalars(self, name: str):  # pragma: no cover
        warnings.warn("\n\n`Using active_scalars to set the active scalars has been "
                      "deprecated.  Use:\n\n"
                      "  - `DataSetAttributes.set_scalars`\n"
                      "  - `DataSetAttributes.active_scalars_name`\n"
                      "  - The [] operator",
            PyvistaDeprecationWarning
        )
        self.active_scalars_name = name

    @property
    def active_vectors(self) -> Optional[np.ndarray]:
        """Return the active vectors as a pyvista_ndarray.

        .. versionchanged:: 0.32.0
            Can no longer used to set the active scalars.  Either use
            :func:`DataSetAttributes.set_scalars` or if the array
            already exists, use
            :attr:`DataSetAttribute.active_scalars_name`.

        Examples
        --------
        Associate point data to a simple cube mesh and show that the
        active scalars in the point array are the most recently added
        array.

        >>> import pyvista
        >>> import numpy as np
        >>> mesh = pyvista.Cube().clean()
        >>> vectors = np.random.random((mesh.n_points, 3))
        >>> mesh.point_arrays.set_vectors(vectors, 'my-vectors')
        >>> vectors_out = mesh.point_arrays.active_vectors
        >>> vectors_out.shape
        (8, 3)

        """
        self._raise_field_data_no_scalars_vectors()
        vectors = self.GetVectors()
        if vectors is not None:
            return pyvista_ndarray(vectors, dataset=self.dataset,
                                   association=self.association)
        return None

    @active_vectors.setter
    def active_vectors(self, name: str):  # pragma: no cover
        warnings.warn("\n\n`Using active_vectors to set the active vectors has been"
                      "deprecated.  Use:\n\n"
                      "  - `DataSetAttributes.set_vectors`\n"
                      "  - `DataSetAttributes.active_vectors_name`\n",
            PyvistaDeprecationWarning
        )
        self.active_vectors_name = name

    @property
    def valid_array_len(self) -> int:
        """Return the length data should be when added to the dataset.

        If there are no restrictions, returns 0.

        Examples
        --------
        Show the valid array lengths match the number of points and
        cells for point and cell arrays, and there is no length limit
        for field arrays.

        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.n_points, mesh.n_cells
        (8, 6)
        >>> mesh.point_arrays.valid_array_len
        8
        >>> mesh.cell_arrays.valid_array_len
        6
        >>> mesh.field_arrays.valid_array_len
        0

        """
        if self.association == FieldAssociation.POINT:
            return self.dataset.GetNumberOfPoints()
        if self.association == FieldAssociation.CELL:
            return self.dataset.GetNumberOfCells()
        return 0

    @property
    def t_coords(self) -> Optional[pyvista_ndarray]:
        """Return or set the active texture coordinates.

        Returns
        -------
        :class:`pyvista.pyvista_ndarray`
            Array of the active texture coordinates.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.point_arrays.t_coords
        pyvista_ndarray([[ 0.,  0.],
                         [ 1.,  0.],
                         [ 1.,  1.],
                         [ 0.,  1.],
                         [-0.,  0.],
                         [-0.,  1.],
                         [-1.,  1.],
                         [-1.,  0.]], dtype=float32)

        """
        t_coords = self.GetTCoords()
        if t_coords is not None:
            return pyvista_ndarray(t_coords, dataset=self.dataset, association=self.association)
        return None

    @t_coords.setter
    def t_coords(self, t_coords: np.ndarray):
        if not isinstance(t_coords, np.ndarray):
            raise TypeError('Texture coordinates must be a numpy array')
        if t_coords.ndim != 2:
            raise ValueError('Texture coordinates must be a 2-dimensional array')
        valid_length = self.valid_array_len
        if t_coords.shape[0] != valid_length:
            raise ValueError(f'Number of texture coordinates ({t_coords.shape[0]}) must match number of points ({valid_length})')
        if t_coords.shape[1] != 2:
            raise ValueError('Texture coordinates must only have 2 components,'
                             f' not ({t_coords.shape[1]})')
        vtkarr = _vtk.numpyTovtkDataArray(t_coords, name='Texture Coordinates')
        self.SetTCoords(vtkarr)
        self.Modified()

    @property
    def active_texture_name(self) -> Optional[str]:
        """Name of the active texture array.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.point_arrays.active_texture_name
        'TCoords'

        """
        try:
            return self.GetTCoords().GetName()
        except:
            return None

    def get_array(self, key: Union[str, int]) -> Union[pyvista_ndarray, _vtk.vtkDataArray, _vtk.vtkAbstractArray]:
        """Get an array in this object.

        Parameters
        ----------
        key : str, int
            The name or index of the array to return.  Arrays are
            ordered within VTK DataSetAttributes, and this feature is
            mirrored here

        Returns
        -------
        :class:`pyvista.pyvista_ndarray` or ``vtkDataArray``
            A :class:`pyvistapyvista_ndarray` if the underlying array
            is a ``vtk.vtkDataArray`` or ``vtk.vtkStringArray``,
            ``vtk.vtkAbstractArray`` if the former does not exist.
            Raises ``KeyError`` if neither exist.

        Examples
        --------
        Store data with point association in a DataSet.

        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.clear_arrays()
        >>> mesh.point_arrays['my_data'] = range(mesh.n_points)

        Access using an index.

        >>> mesh.point_arrays.get_array(0)
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        Access using a key.

        >>> mesh.point_arrays.get_array('my_data')
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        Notes
        -----
        This is provided since arrays are ordered within VTK and can
        be indexed via an int.  When getting an array, you can just
        use the key of the array with the ``[]`` operator with the
        name of the array.

        """
        self._raise_index_out_of_bounds(index=key)
        vtk_arr = self.GetArray(key)
        if vtk_arr is None:
            vtk_arr = self.GetAbstractArray(key)
            if vtk_arr is None:
                raise KeyError(f'{key}')
            if type(vtk_arr) == _vtk.vtkAbstractArray:
                return vtk_arr
        narray = pyvista_ndarray(vtk_arr, dataset=self.dataset, association=self.association)
        if vtk_arr.GetName() in self.dataset.association_bitarray_names[self.association.name]:
            narray = narray.view(np.bool_)
        return narray

    def set_array(self, data: Union[Sequence[Number], Number, np.ndarray],
                  name: str, deep_copy=False) -> None:
        """Add an array to this object.

        This method is useful when adding arrays to the DataSet when
        you do not wish for them to become the active vectors or
        scalars (which will be displayed within a plot).

        Parameters
        ----------
        data : sequence
            A ``pyvista_ndarray``, ``numpy.ndarray``, ``list``,
            ``tuple`` or scalar value.

        name : str
            Name to assign to the data.  If this name already exists,
            this array will be written.

        deep_copy : bool, optional
            When ``True`` makes a full copy of the array.

        Examples
        --------
        Add a point array to a mesh.

        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> data = range(mesh.n_points)
        >>> mesh.point_arrays.set_array(data, 'my-data')
        >>> mesh.point_arrays['my-data']
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        Add a cell array to a mesh

        >>> cell_data = range(mesh.n_cells)
        >>> mesh.cell_arrays.set_array(cell_data, 'my-data')
        >>> mesh.cell_arrays['my-data']
        pyvista_ndarray([0, 1, 2, 3, 4, 5])

        Add a field array to a mesh.

        >>> field_data = range(3)
        >>> mesh.field_arrays.set_array(field_data, 'my-data')
        >>> mesh.field_arrays['my-data']
        pyvista_ndarray([0, 1, 2])

        Notes
        -----
        You can simply use the ``[]`` operator to add an array to the
        point, cell, or field data.  Note that by default if this is
        not field data, the array will be made the active scalars.

        When adding directional data (such as velocity vectors), use
        :func:`DataSetAttributes.set_vectors`.

        When adding non-directional data (such temperature values or
        multi-component scalars like RGBA values), you can also use
        :func:`DataSetAttributes.set_scalars`.

        """
        vtk_arr = self._prepare_array(data, name, deep_copy)
        self.VTKObject.AddArray(vtk_arr)
        self.VTKObject.Modified()

    def set_scalars(self, scalars: Union[Sequence[Number], Number, np.ndarray],
                    name='scalars', deep_copy=False):
        """Set the active scalars of the dataset with an array.

        In VTK and PyVista, scalars are a quantity that has no
        direction.  This can include data with multiple components
        (such as RGBA values) or just one component (such as
        temperature data).

        See :func:`DataSetAttributes.set_vectors` when adding arrays
        that contain magnitude and direction.

        Parameters
        ----------
        scalars : sequence
            A ``pyvista_ndarray``, ``numpy.ndarray``, ``list``,
            ``tuple`` or scalar value.

        name : str
            Name to assign the scalars.

        deep_copy : bool, optional
            When ``True`` makes a full copy of the array.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.clear_arrays()
        >>> scalars = range(mesh.n_points)
        >>> mesh.point_arrays.set_scalars(scalars, 'my-scalars')
        >>> mesh.point_arrays
        pyvista DataSetAttributes
        Association     : POINT
        Active Scalars  : my-scalars
        Active Vectors  : None
        Active Texture  : None
        Contains arrays :
            my-scalars              int64    (8,)                 SCALARS

        """
        vtk_arr = self._prepare_array(scalars, name, deep_copy)
        self.VTKObject.SetScalars(vtk_arr)
        self.VTKObject.Modified()

    def set_vectors(self, vectors: Union[Sequence[Number], Number, np.ndarray],
                    name: str, deep_copy=False):
        """Set the vectors of this data attribute.

        Vectors are a quantity that has magnitude and direction, such
        as normal vectors or a velocity field.

        The vectors data must contain three components per cell or
        point.  Use :func:`DataSetAttributes.set_scalars` when
        adding non-directional data.

        Parameters
        ----------
        vectors : sequence
            A ``pyvista_ndarray``, ``numpy.ndarray``, ``list``, or
            ``tuple``.  Must match the number of cells or points of
            the dataset.

        name : str
            Name of the vectors.

        deep_copy : bool, optional
            When ``True`` makes a full copy of the array.  When
            ``False``, the data is "pointed" to the original array
            without copying it.

        Examples
        --------
        Add random vectors to a mesh as point data.

        >>> import pyvista
        >>> import numpy as np
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.clear_arrays()
        >>> vectors = np.random.random((mesh.n_points, 3))
        >>> mesh.point_arrays.set_vectors(vectors, 'my-vectors')
        >>> mesh.point_arrays
        pyvista DataSetAttributes
        Association     : POINT
        Active Scalars  : None
        Active Vectors  : my-vectors
        Active Texture  : None
        Contains arrays :
            my-vectors              float64  (8, 3)               VECTOR

        Notes
        -----
        PyVista and VTK treats vectors and scalars differently when
        performing operations. Vector data, unlike scalar data, is
        rotated along with the geometry when the DataSet is passed
        through a transformation filter.

        """
        # prepare the array and add an attribute so that we can track this as a vector
        vtk_arr = self._prepare_array(vectors, name, deep_copy)

        n_comp = vtk_arr.GetNumberOfComponents()
        if n_comp != 3:
            raise ValueError('Vector array should contain 3 components, got '
                             f'{n_comp}')

        # check if there are current vectors, if so, we need to keep
        # this array around since setting active vectors will remove
        # this array.
        current_vectors = self.GetVectors()

        # now we can set the active vectors and add back in the old vectors as an array
        self.VTKObject.SetVectors(vtk_arr)
        if current_vectors is not None:
            self.VTKObject.AddArray(current_vectors)

        self.VTKObject.Modified()

    def _prepare_array(self, data: Union[Sequence[Number], Number, np.ndarray],
                       name: str, deep_copy: bool) -> _vtk.vtkDataSet:
        """Prepare an array to be added to this dataset."""
        if data is None:
            raise TypeError('``data`` cannot be None.')
        if isinstance(data, Iterable):
            data = pyvista_ndarray(data)

        if self.association == FieldAssociation.POINT:
            array_len = self.dataset.GetNumberOfPoints()
        elif self.association == FieldAssociation.CELL:
            array_len = self.dataset.GetNumberOfCells()
        else:
            array_len = data.shape[0] if isinstance(data, np.ndarray) else 1

        # Fixup input array length for scalar input
        if not isinstance(data, np.ndarray) or np.ndim(data) == 0:
            tmparray = np.empty(array_len)
            tmparray.fill(data)
            data = tmparray
        if data.shape[0] != array_len:
            raise ValueError(f'data length of ({data.shape[0]}) != required length ({array_len})')

        if data.dtype == np.bool_:
            self.dataset.association_bitarray_names[self.association.name].add(name)
            data = data.view(np.uint8)

        shape = data.shape
        if len(shape) == 3:
            # Array of matrices. We need to make sure the order  in memory is right.
            # If column order (c order), transpose. VTK wants row order (fortran
            # order). The deep copy later will make sure that the array is contiguous.
            # If row order but not contiguous, transpose so that the deep copy below
            # does not happen.
            size = data.dtype.itemsize
            if (data.strides[1] / size == 3 and data.strides[2] / size == 1) or \
                (data.strides[1] / size == 1 and data.strides[2] / size == 3 and \
                 not data.flags.contiguous):
                data = data.transpose(0, 2, 1)

        # If array is not contiguous, make a deep copy that is contiguous
        if not data.flags.contiguous:
            data = np.ascontiguousarray(data)

        # Flatten array of matrices to array of vectors
        if len(shape) == 3:
            data = data.reshape(shape[0], shape[1]*shape[2])

        # Swap bytes from big to little endian.
        if data.dtype.byteorder == '>':
            data = data.byteswap(inplace=True)

        # this handles the case when an input array is directly added to the
        # output. We want to make sure that the array added to the output is not
        # referring to the input dataset.
        copy = pyvista_ndarray(data)

        return helpers.convert_array(copy, name, deep=deep_copy)

    def append(self, narray: Union[Sequence[Number], Number, np.ndarray],
               name: str, deep_copy=False, active_vectors=False,
               active_scalars=True) -> None:  # pragma: no cover
        """Add an array to this object.

        .. deprecated:: 0.32.0
           Use one of the following instead:

           * :func:`DataSetAttributes.set_array`
           * :func:`DataSetAttributes.set_scalars`
           * :func:`DataSetAttributes.set_vectors`
           * The ``[]`` operator

        """
        warnings.warn("\n\n`DataSetAttributes.append` is deprecated.\n\n"
                      "Use one of the following instead:\n"
                      "  - `DataSetAttributes.set_array`\n"
                      "  - `DataSetAttributes.set_scalars`\n"
                      "  - `DataSetAttributes.set_vectors`\n"
                      "  - The [] operator",
            PyvistaDeprecationWarning
        )
        if active_vectors:  # pragma: no cover
            raise ValueError('Use set_vectors to set vector data')

        self.set_array(narray, name, deep_copy)
        if active_scalars:
            self.active_scalars_name = name

    def remove(self, key: str) -> None:
        """Remove an array.

        Parameters
        ----------
        key : str
            The name of the array to remove.

        Examples
        --------
        Add a point data array to a DataSet and then remove it.

        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.point_arrays['my_data'] = range(mesh.n_points)
        >>> mesh.point_arrays.remove('my_data')

        Show that the array no longer exists in ``point_arrays``

        >>> 'my_data' in mesh.point_arrays
        False

        Notes
        -----
        This is provided as VTK supports indexed arrays in DataSetAttributes.

        """
        name = self.get_array(key).GetName()  # type: ignore
        try:
            self.dataset.association_bitarray_names[self.association.name].remove(name)
        except KeyError:
            pass
        self.VTKObject.RemoveArray(key)
        self.VTKObject.Modified()

    def pop(self, key: str, default=pyvista_ndarray(array=[])) -> pyvista_ndarray:
        """Remove an array and return it.

        Parameters
        ----------
        key : str
            The name of the array to remove and return.

        default : anything, optional
            If default is not given and key is not in the dictionary,
            a KeyError is raised.

        Returns
        -------
        :class:`pyvista_ndarray`
            Requested array.

        Examples
        --------
        Add a point data array to a DataSet and then remove it

        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.point_arrays['my_data'] = range(mesh.n_points)
        >>> mesh.point_arrays.pop('my_data')
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        Show that the array no longer exists in ``point_arrays``

        >>> 'my_data' in mesh.point_arrays
        False

        """
        vtk_arr = self.GetArray(key)
        if vtk_arr:
            copy = vtk_arr.NewInstance()
            copy.DeepCopy(vtk_arr)
            vtk_arr = copy

        try:
            self.remove(key)
        except KeyError:
            if default in self.pop.__defaults__:  # type: ignore
                raise
            return default

        return pyvista_ndarray(vtk_arr, dataset=self.dataset,
                               association=self.association)

    def items(self) -> List[Tuple[str, pyvista_ndarray]]:
        """Return a list of (array name, array value).

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.clear_arrays()
        >>> mesh.cell_arrays['data0'] = [0]*mesh.n_cells
        >>> mesh.cell_arrays['data1'] = range(mesh.n_cells)
        >>> mesh.cell_arrays.items()
        [('data0', pyvista_ndarray([0, 0, 0, 0, 0, 0])), ('data1', pyvista_ndarray([0, 1, 2, 3, 4, 5]))]

        """
        return list(zip(self.keys(), self.values()))

    def keys(self) -> List[str]:
        """Return the names of the arrays as a list.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.clear_arrays()
        >>> mesh.point_arrays['data0'] = [0]*mesh.n_points
        >>> mesh.point_arrays['data1'] = range(mesh.n_points)
        >>> mesh.point_arrays.keys()
        ['data0', 'data1']

        """
        keys = []
        for i in range(self.GetNumberOfArrays()):
            name = self.VTKObject.GetAbstractArray(i).GetName()
            if name:
                keys.append(name)
        return keys

    def values(self) -> List[pyvista_ndarray]:
        """Return the arrays as a list.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.clear_arrays()
        >>> mesh.cell_arrays['data0'] = [0]*mesh.n_cells
        >>> mesh.cell_arrays['data1'] = range(mesh.n_cells)
        >>> mesh.cell_arrays.values()
        [pyvista_ndarray([0, 0, 0, 0, 0, 0]), pyvista_ndarray([0, 1, 2, 3, 4, 5])]

        """
        values = []
        for name in self.keys():
            array = self.VTKObject.GetAbstractArray(name)
            arr = pyvista_ndarray(array, dataset=self.dataset, association=self.association)
            values.append(arr)
        return values

    def clear(self):
        """Remove all arrays in this object.

        Examples
        --------
        Add point data to a DataSet and then clear the point_arrays.

        >>> import pyvista
        >>> mesh = pyvista.Cube().clean()
        >>> mesh.clear_arrays()
        >>> mesh.point_arrays['my_data'] = range(mesh.n_points)
        >>> len(mesh.point_arrays)
        1
        >>> mesh.point_arrays.clear()
        >>> len(mesh.point_arrays)
        0

        """
        for array_name in self.keys():
            self.remove(key=array_name)

    def update(self, array_dict: Union[Dict[str, np.ndarray], 'DataSetAttributes']):
        """Update arrays in this object.

        For each key, value given, add the pair, if it already exists,
        update it.

        Parameters
        ----------
        array_dict : dict
            A dictionary of (array name, numpy.ndarray)
        """
        for name, array in array_dict.items():
            self[name] = array.copy()

    def _raise_index_out_of_bounds(self, index: Any):
        max_index = self.VTKObject.GetNumberOfArrays()
        if isinstance(index, int):
            if index < 0 or index >= self.VTKObject.GetNumberOfArrays():
                raise KeyError(f'Array index ({index}) out of range [0, {max_index}]')

    def _raise_field_data_no_scalars_vectors(self):
        """Raise a TypeError if FieldData."""
        if self.association == FieldAssociation.NONE:
            raise TypeError('FieldData does not have active scalars or vectors.')

    @property
    def active_scalars_name(self) -> Optional[str]:
        """Name of the active scalars.

        Examples
        --------
        Add two arrays to the mesh point data.

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.point_arrays['my_data'] = range(mesh.n_points)
        >>> mesh.point_arrays['my_other_data'] = range(mesh.n_points)
        >>> mesh.point_arrays.active_scalars_name
        'my_other_data'

        Set the name of the active scalars.

        >>> mesh.point_arrays.active_scalars_name = 'my_data'
        >>> mesh.point_arrays.active_scalars_name
        'my_data'

        """
        try:
            return self.GetScalars().GetName()
        except:
            return None

    @active_scalars_name.setter
    def active_scalars_name(self, name: str) -> None:
        self._raise_field_data_no_scalars_vectors()
        dtype = self[name].dtype
        # only vtkDataArray subclasses can be set as active attributes
        if np.issubdtype(dtype, np.number) or dtype == bool:
            self.SetActiveScalars(name)

    @property
    def active_vectors_name(self) -> Optional[str]:
        """Name of the active scalars.

        Examples
        --------
        >>> import pyvista
        >>> import numpy as np
        >>> mesh = pyvista.Sphere()
        >>> mesh.point_arrays['my_data'] = np.random.random((mesh.n_points, 3))
        >>> mesh.point_arrays.active_scalars_name
        'my_data'

        """
        try:
            return self.GetVectors().GetName()
        except:
            return None

    @active_vectors_name.setter
    def active_vectors_name(self, name: str) -> None:
        self._raise_field_data_no_scalars_vectors()
        if name not in self:
            raise KeyError(f'DataSetAttribute does not contain "{name}"')
        # verify that the array has the correct number of components
        n_comp = self.GetArray(name).GetNumberOfComponents()
        if n_comp != 3:
            raise ValueError(f'{name} needs 3 components, has ({n_comp})')
        self.SetActiveVectors(name)

    def __eq__(self, other: Any) -> bool:
        """Test dict-like equivalency."""
        # here we check if other is the same class or a subclass of self.
        if not isinstance(other, type(self)):
            return False

        if set(self.keys()) != set(other.keys()):
            return False

        for key, value in other.items():
            if not np.array_equal(value, self[key]):
                return False

        if self.association != FieldAssociation.NONE:
            if other.active_scalars_name != self.active_scalars_name:
                return False
            if other.active_vectors_name != self.active_vectors_name:
                return False

        return True

    @property
    def normals(self) -> Optional[pyvista_ndarray]:
        """Return or set the normals.

        Returns
        -------
        :class:`pyvista_ndarray`
            Normals of this dataset attributes.  ``None`` if no
            normals have been set.

        Examples
        --------
        First, compute cell normals.

        >>> import pyvista
        >>> mesh = pyvista.Plane(i_resolution=1, j_resolution=1)
        >>> mesh.point_arrays
        pyvista DataSetAttributes
        Association     : POINT
        Active Scalars  : TextureCoordinates
        Active Vectors  : None
        Active Texture  : TextureCoordinates
        Contains arrays :
            Normals                 float32  (4, 3)               NORMALS
            TextureCoordinates      float32  (4, 2)               SCALARS

        >>> mesh.point_arrays.normals
        pyvista_ndarray([[0.000000e+00,  0.000000e+00, -1.000000e+00],
                         [0.000000e+00,  0.000000e+00, -1.000000e+00],
                         [0.000000e+00,  0.000000e+00, -1.000000e+00],
                         [0.000000e+00,  0.000000e+00, -1.000000e+00]],
                        dtype=float32)

        Assign normals to the cell arrays.  An array will be added
        named ``"Normals"``.

        >>> mesh.cell_arrays.normals = [[0.0, 0.0, 1.0]]
        >>> mesh.cell_arrays
        pyvista DataSetAttributes
        Association     : CELL
        Active Scalars  : None
        Active Vectors  : None
        Active Texture  : None
        Contains arrays :
            Normals                 float64  (1, 3)               NORMALS

        Notes
        -----
        Field data will have no normals.

        """
        if self.association == FieldAssociation.NONE:
            raise AttributeError('FieldData does not have normals.')

        vtk_normals = self.GetNormals()
        if vtk_normals is not None:
            return pyvista_ndarray(vtk_normals, dataset=self.dataset,
                                   association=self.association)
        return None

    @normals.setter
    def normals(self, normals: Union[Sequence[Number], np.ndarray]):
        normals = np.asarray(normals)
        if normals.ndim != 2:
            raise ValueError('Normals must be a 2-dimensional array')
        valid_length = self.valid_array_len
        if normals.shape[0] != valid_length:
            raise ValueError(f'Number of normals ({normals.shape[0]}) must match number of points ({valid_length})')
        if normals.shape[1] != 3:
            raise ValueError('Normals must have exactly 3 components,'
                             f' not ({normals.shape[1]})')

        vtkarr = _vtk.numpyTovtkDataArray(normals, name='Normals')
        self.SetNormals(vtkarr)
        self.Modified()
