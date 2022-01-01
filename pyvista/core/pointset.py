"""Sub-classes and wrappers for vtk.vtkPointSet."""
import collections
from functools import wraps
import logging
import numbers
import os
import pathlib
from textwrap import dedent
from typing import Sequence, Tuple, Union
import warnings

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.utilities import PyvistaDeprecationWarning, abstract_class
from pyvista.utilities.cells import (
    CellArray,
    create_mixed_cells,
    generate_cell_offsets,
    get_mixed_cells,
    numpy_to_idarr,
)

from ..utilities.fileio import get_ext
from .dataset import DataSet
from .errors import DeprecationError, VTKVersionError
from .filters import PolyDataFilters, StructuredGridFilters, UnstructuredGridFilters, _get_output

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')
DEFAULT_INPLACE_WARNING = (
    'You did not specify a value for `inplace` and the default value will '
    'be changing to `False` in future versions for point-based meshes (e.g., '
    '`PolyData`). Please make sure you are not assuming this to be an inplace '
    'operation.'
)


class PointSet(DataSet):
    """PyVista's equivalent of vtk.vtkPointSet.

    This holds methods common to PolyData and UnstructuredGrid.
    """

    def center_of_mass(self, scalars_weight=False):
        """Return the coordinates for the center of mass of the mesh.

        Parameters
        ----------
        scalars_weight : bool, optional
            Flag for using the mesh scalars as weights. Defaults to ``False``.

        Returns
        -------
        numpy.ndarray
            Coordinates for the center of mass.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere(center=(1, 1, 1))
        >>> mesh.center_of_mass()
        array([1., 1., 1.])

        """
        alg = _vtk.vtkCenterOfMass()
        alg.SetInputDataObject(self)
        alg.SetUseScalarsAsWeights(scalars_weight)
        alg.Update()
        return np.array(alg.GetCenter())

    def shallow_copy(self, to_copy):
        """Create a shallow copy from a different dataset into this one.

        This method mutates this dataset and returns ``None``.

        Parameters
        ----------
        to_copy : pyvista.DataSet
            Data object to perform the shallow copy from.

        """
        # Set default points if needed
        if not to_copy.GetPoints():
            to_copy.SetPoints(_vtk.vtkPoints())
        DataSet.shallow_copy(self, to_copy)

    def remove_cells(self, ind, inplace=False):
        """Remove cells.

        Parameters
        ----------
        ind : sequence
            Cell indices to be removed.  The array can also be a
            boolean array of the same size as the number of cells.

        inplace : bool, optional
            Whether to update the mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Same type as the input, but with the specified cells
            removed.

        Examples
        --------
        Remove 20 cells from an unstructured grid.

        >>> from pyvista import examples
        >>> import pyvista
        >>> hex_mesh = pyvista.read(examples.hexbeamfile)
        >>> removed = hex_mesh.remove_cells(range(10, 20))
        >>> removed.plot(color='tan', show_edges=True, line_width=3)
        """
        if isinstance(ind, np.ndarray):
            if ind.dtype == np.bool_ and ind.size != self.n_cells:
                raise ValueError('Boolean array size must match the '
                                 f'number of cells ({self.n_cells}')
        ghost_cells = np.zeros(self.n_cells, np.uint8)
        ghost_cells[ind] = _vtk.vtkDataSetAttributes.DUPLICATECELL

        if inplace:
            target = self
        else:
            target = self.copy()

        target.cell_data[_vtk.vtkDataSetAttributes.GhostArrayName()] = ghost_cells
        target.RemoveGhostCells()
        return target

    def points_to_double(self):
        """Convert the points datatype to double precision.

        Returns
        -------
        pyvista.PointSet
            Pointset with points in double precision.

        Notes
        -----
        This operates in place.

        Examples
        --------
        Create a mesh that has points of the type ``float32`` and
        convert the points to ``float64``.

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.points.dtype
        dtype('float32')
        >>> _ = mesh.points_to_double()
        >>> mesh.points.dtype
        dtype('float64')

        """
        if self.points.dtype != np.double:
            self.points = self.points.astype(np.double)
        return self

    # todo: `transform_all_input_vectors` is not handled when modifying inplace
    def translate(self, xyz: Union[list, tuple, np.ndarray], transform_all_input_vectors=False, inplace=None):
        """Translate the mesh.

        Parameters
        ----------
        xyz : list or tuple or np.ndarray
            Length 3 list, tuple or array.

        transform_all_input_vectors : bool, optional
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed. This is only valid when not
            updating in place.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        pyvista.PointSet
            Translated pointset.

        Examples
        --------
        Create a sphere and translate it by ``(2, 1, 2)``.

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.center
        [0.0, 0.0, 0.0]
        >>> trans = mesh.translate((2, 1, 2), inplace=True)
        >>> trans.center
        [2.0, 1.0, 2.0]

        """
        if inplace is None:
            # Deprecated on v0.32.0, estimated removal on v0.35.0
            warnings.warn(DEFAULT_INPLACE_WARNING, PyvistaDeprecationWarning)
            inplace = True
        if inplace:
            self.points += np.asarray(xyz)  # type: ignore
            return self
        return super().translate(xyz, transform_all_input_vectors=transform_all_input_vectors, inplace=inplace)

    def scale(
            self,
            xyz: Union[list, tuple, np.ndarray],
            transform_all_input_vectors=False,
            inplace=None
    ):
        """Scale the mesh.

        Parameters
        ----------
        xyz : scale factor list or tuple or np.ndarray
            Length 3 list, tuple or array.

        transform_all_input_vectors : bool, optional
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed. This is only valid when not
            updating in place.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        pyvista.PointSet
            Scaled pointset.

        Notes
        -----
        ``transform_all_input_vectors`` is not handled when modifying inplace.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> pl = pyvista.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> pl.show_axes()
        >>> _ = pl.show_grid()
        >>> mesh1 = examples.download_teapot()
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.subplot(0, 1)
        >>> pl.show_axes()
        >>> _ = pl.show_grid()
        >>> mesh2 = mesh1.scale([10.0, 10.0, 10.0], inplace=False)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show(cpos="xy")
        """
        if inplace is None:
            # Deprecated on v0.32.0, estimated removal on v0.35.0
            warnings.warn(DEFAULT_INPLACE_WARNING, PyvistaDeprecationWarning)
            inplace = True
        if inplace:
            self.points *= np.asarray(xyz)  # type: ignore
            return self
        return super().scale(xyz, transform_all_input_vectors=transform_all_input_vectors, inplace=inplace)

    @wraps(DataSet.flip_x)
    def flip_x(self, *args, **kwargs):
        """Wrap ``DataSet.flip_x``."""
        if kwargs.get('inplace') is None:
            # Deprecated on v0.32.0, estimated removal on v0.35.0
            warnings.warn(DEFAULT_INPLACE_WARNING, PyvistaDeprecationWarning)
            kwargs['inplace'] = True
        return super().flip_x(*args, **kwargs)

    @wraps(DataSet.flip_y)
    def flip_y(self, *args, **kwargs):
        """Wrap ``DataSet.flip_y``."""
        if kwargs.get('inplace') is None:
            # Deprecated on v0.32.0, estimated removal on v0.35.0
            warnings.warn(DEFAULT_INPLACE_WARNING, PyvistaDeprecationWarning)
            kwargs['inplace'] = True
        return super().flip_y(*args, **kwargs)

    @wraps(DataSet.flip_z)
    def flip_z(self, *args, **kwargs):
        """Wrap ``DataSet.flip_z``."""
        if kwargs.get('inplace') is None:
            # Deprecated on v0.32.0, estimated removal on v0.35.0
            warnings.warn(DEFAULT_INPLACE_WARNING, PyvistaDeprecationWarning)
            kwargs['inplace'] = True
        return super().flip_z(*args, **kwargs)

    @wraps(DataSet.flip_normal)
    def flip_normal(self, *args, **kwargs):
        """Wrap ``DataSet.flip_normal``."""
        if kwargs.get('inplace') is None:
            # Deprecated on v0.32.0, estimated removal on v0.35.0
            warnings.warn(DEFAULT_INPLACE_WARNING, PyvistaDeprecationWarning)
            kwargs['inplace'] = True
        return super().flip_normal(*args, **kwargs)

    @wraps(DataSet.rotate_x)
    def rotate_x(self, *args, **kwargs):
        """Wrap ``DataSet.rotate_x``."""
        if kwargs.get('inplace') is None:
            # Deprecated on v0.32.0, estimated removal on v0.35.0
            warnings.warn(DEFAULT_INPLACE_WARNING, PyvistaDeprecationWarning)
            kwargs['inplace'] = True
        return super().rotate_x(*args, **kwargs)

    @wraps(DataSet.rotate_y)
    def rotate_y(self, *args, **kwargs):
        """Wrap ``DataSet.rotate_y``."""
        if kwargs.get('inplace') is None:
            # Deprecated on v0.32.0, estimated removal on v0.35.0
            warnings.warn(DEFAULT_INPLACE_WARNING, PyvistaDeprecationWarning)
            kwargs['inplace'] = True
        return super().rotate_y(*args, **kwargs)

    @wraps(DataSet.rotate_z)
    def rotate_z(self, *args, **kwargs):
        """Wrap ``DataSet.rotate_z``."""
        if kwargs.get('inplace') is None:
            # Deprecated on v0.32.0, estimated removal on v0.35.0
            warnings.warn(DEFAULT_INPLACE_WARNING, PyvistaDeprecationWarning)
            kwargs['inplace'] = True
        return super().rotate_z(*args, **kwargs)

    @wraps(DataSet.rotate_vector)
    def rotate_vector(self, *args, **kwargs):
        """Wrap ``DataSet.rotate_vector``."""
        if kwargs.get('inplace') is None:
            # Deprecated on v0.32.0, estimated removal on v0.35.0
            warnings.warn(DEFAULT_INPLACE_WARNING, PyvistaDeprecationWarning)
            kwargs['inplace'] = True
        return super().rotate_vector(*args, **kwargs)


class PolyData(_vtk.vtkPolyData, PointSet, PolyDataFilters):
    """Dataset consisting of surface geometry (e.g. vertices, lines, and polygons).

    Can be initialized in several ways:

    - Create an empty mesh
    - Initialize from a vtk.vtkPolyData
    - Using vertices
    - Using vertices and faces
    - From a file

    Parameters
    ----------
    var_inp : vtk.vtkPolyData, str, sequence, optional
        Flexible input type.  Can be a ``vtk.vtkPolyData``, in which case
        this PolyData object will be copied if ``deep=True`` and will
        be a shallow copy if ``deep=False``.

        Also accepts a path, which may be local path as in
        ``'my_mesh.stl'`` or global path like ``'/tmp/my_mesh.ply'``
        or ``'C:/Users/user/my_mesh.ply'``.

        Otherwise, this must be a points array or list containing one
        or more points.  Each point must have 3 dimensions.

    faces : sequence, optional
        Face connectivity array.  Faces must contain padding
        indicating the number of points in the face.  For example, the
        two faces ``[10, 11, 12]`` and ``[20, 21, 22, 23]`` will be
        represented as ``[3, 10, 11, 12, 4, 20, 21, 22, 23]``.  This
        lets you have an arbitrary number of points per face.

        When not including the face connectivity array, each point
        will be assigned to a single vertex.  This is used for point
        clouds that have no connectivity.

    n_faces : int, optional
        Number of faces in the ``faces`` connectivity array.  While
        optional, setting this speeds up the creation of the
        ``PolyData``.

    lines : sequence, optional
        The line connectivity array.  Like ``faces``, this array
        requires padding indicating the number of points in a line
        segment.  For example, the two line segments ``[0, 1]`` and
        ``[1, 2, 3, 4]`` will be represented as
        ``[2, 0, 1, 4, 1, 2, 3, 4]``.

    n_lines : int, optional
        Number of lines in the ``lines`` connectivity array.  While
        optional, setting this speeds up the creation of the
        ``PolyData``.

    deep : bool, optional
        Whether to copy the inputs, or to create a mesh from them
        without copying them.  Setting ``deep=True`` ensures that the
        original arrays can be modified outside the mesh without
        affecting the mesh. Default is ``False``.

    force_ext : str, optional
        If initializing from a file, force the reader to treat the
        file as if it had this extension as opposed to the one in the
        file.

    force_float : bool, optional
        Casts the datatype to ``float32`` if points datatype is
        non-float.  Default ``True``. Set this to ``False`` to allow
        non-float types, though this may lead to truncation of
        intermediate floats when transforming datasets.

    Examples
    --------
    >>> import vtk
    >>> import numpy as np
    >>> from pyvista import examples
    >>> import pyvista

    Create an empty mesh.

    >>> mesh = pyvista.PolyData()

    Initialize from a ``vtk.vtkPolyData`` object.

    >>> vtkobj = vtk.vtkPolyData()
    >>> mesh = pyvista.PolyData(vtkobj)

    Initialize from just vertices.

    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0, 0.5, 0]])
    >>> mesh = pyvista.PolyData(vertices)

    Initialize from vertices and faces.

    >>> faces = np.hstack([[3, 0, 1, 2], [3, 0, 3, 2]])
    >>> mesh = pyvista.PolyData(vertices, faces)

    Initialize from vertices and lines.

    >>> lines = np.hstack([[2, 0, 1], [2, 1, 2]])
    >>> mesh = pyvista.PolyData(vertices, lines=lines)

    Initialize from a filename.

    >>> mesh = pyvista.PolyData(examples.antfile)

    """

    _WRITERS = {'.ply': _vtk.vtkPLYWriter,
                '.vtp': _vtk.vtkXMLPolyDataWriter,
                '.stl': _vtk.vtkSTLWriter,
                '.vtk': _vtk.vtkPolyDataWriter}

    def __init__(self, var_inp=None, faces=None, n_faces=None, lines=None,
                 n_lines=None, deep=False, force_ext=None, force_float=True) -> None:
        """Initialize the polydata."""
        local_parms = locals()
        super().__init__()

        # allow empty input
        if var_inp is None:
            return

        # filename
        opt_kwarg = ['faces', 'n_faces', 'lines', 'n_lines']
        if isinstance(var_inp, (str, pathlib.Path)):
            for kwarg in opt_kwarg:
                if local_parms[kwarg]:
                    raise ValueError('No other arguments should be set when first '
                                     'parameter is a string')
            self._from_file(var_inp, force_ext=force_ext)  # is filename

            return

        # PolyData-like
        if isinstance(var_inp, _vtk.vtkPolyData):
            for kwarg in opt_kwarg:
                if local_parms[kwarg]:
                    raise ValueError('No other arguments should be set when first '
                                     'parameter is a PolyData')
            if deep:
                self.deep_copy(var_inp)
            else:
                self.shallow_copy(var_inp)
            return

        # First parameter is points
        if isinstance(var_inp, (np.ndarray, list, _vtk.vtkDataArray)):
            self.SetPoints(pyvista.vtk_points(
                var_inp, deep=deep, force_float=force_float
            ))

        else:
            msg = f"""
                Invalid Input type:

                Expected first argument to be either a:
                - vtk.PolyData
                - pyvista.PolyData
                - numeric numpy.ndarray (1 or 2 dimensions)
                - List (flat or nested with 3 points per vertex)
                - vtk.vtkDataArray

                Instead got: {type(var_inp)}"""
            raise TypeError(dedent(msg.strip('\n')))

        # At this point, points have been setup, add faces and/or lines
        if faces is None and lines is None:
            # one cell per point (point cloud case)
            verts = self._make_vertex_cells(self.n_points)
            self.verts = CellArray(verts, self.n_points, deep)

        elif faces is not None:
            # here we use CellArray since we must specify deep and n_faces
            self.faces = CellArray(faces, n_faces, deep)

        # can always set lines
        if lines is not None:
            # here we use CellArray since we must specify deep and n_lines
            self.lines = CellArray(lines, n_lines, deep)

    def _post_file_load_processing(self):
        """Execute after loading a PolyData from file."""
        # When loading files with just point arrays, create and
        # set the polydata vertices
        if self.n_points > 0 and self.n_cells == 0:
            verts = self._make_vertex_cells(self.n_points)
            self.verts = CellArray(verts, self.n_points, deep=False)

    def __repr__(self):
        """Return the standard representation."""
        return DataSet.__repr__(self)

    def __str__(self):
        """Return the standard str representation."""
        return DataSet.__str__(self)

    @staticmethod
    def _make_vertex_cells(npoints):
        cells = np.empty((npoints, 2), dtype=pyvista.ID_TYPE)
        cells[:, 0] = 1
        cells[:, 1] = np.arange(npoints, dtype=pyvista.ID_TYPE)
        return cells

    @property
    def verts(self) -> np.ndarray:
        """Get the vertex cells.

        Returns
        -------
        numpy.ndarray
            Array of vertex cell indices.

        Examples
        --------
        Create a point cloud polydata and return the vertex cells.

        >>> import pyvista
        >>> import numpy as np
        >>> points = np.random.random((5, 3))
        >>> pdata = pyvista.PolyData(points)
        >>> pdata.verts
        array([1, 0, 1, 1, 1, 2, 1, 3, 1, 4])

        Set vertex cells.  Note how the mesh plots both the surface
        mesh and the additional vertices in a single plot.

        >>> mesh = pyvista.Plane(i_resolution=3, j_resolution=3)
        >>> mesh.verts = np.vstack((np.ones(mesh.n_points, dtype=np.int64),
        ...                         np.arange(mesh.n_points))).T
        >>> mesh.plot(color='tan', render_points_as_spheres=True, point_size=60)

        """
        return _vtk.vtk_to_numpy(self.GetVerts().GetData())

    @verts.setter
    def verts(self, verts):
        """Set the vertex cells."""
        if isinstance(verts, CellArray):
            self.SetVerts(verts)
        else:
            self.SetVerts(CellArray(verts))

    @property
    def lines(self) -> np.ndarray:
        """Return a pointer to the lines as a numpy array.

        Examples
        --------
        Return the lines from a spline.

        >>> import pyvista
        >>> import numpy as np
        >>> points = np.random.random((3, 3))
        >>> spline = pyvista.Spline(points, 10)
        >>> spline.lines
        array([10,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9])

        """
        return _vtk.vtk_to_numpy(self.GetLines().GetData()).ravel()

    @lines.setter
    def lines(self, lines):
        """Set the lines of the polydata."""
        if isinstance(lines, CellArray):
            self.SetLines(lines)
        else:
            self.SetLines(CellArray(lines))

    @property
    def faces(self) -> np.ndarray:
        """Return a pointer to the faces as a numpy array.

        Returns
        -------
        numpy.ndarray
            Array of face indices.

        Examples
        --------
        >>> import pyvista as pv
        >>> plane = pv.Plane(i_resolution=2, j_resolution=2)
        >>> plane.faces
        array([4, 0, 1, 4, 3, 4, 1, 2, 5, 4, 4, 3, 4, 7, 6, 4, 4, 5, 8, 7])

        Note how the faces contain a "padding" indicating the number
        of points per face:

        >>> plane.faces.reshape(-1, 5)
        array([[4, 0, 1, 4, 3],
               [4, 1, 2, 5, 4],
               [4, 3, 4, 7, 6],
               [4, 4, 5, 8, 7]])
        """
        return _vtk.vtk_to_numpy(self.GetPolys().GetData())

    @faces.setter
    def faces(self, faces):
        """Set the face cells."""
        if isinstance(faces, CellArray):
            self.SetPolys(faces)
        else:
            # TODO: faster to mutate in-place if array is same size?
            self.SetPolys(CellArray(faces))

    @property
    def is_all_triangles(self):
        """Return if all the faces of the :class:`pyvista.PolyData` are triangles.

        .. versionchanged:: 0.32.0
           ``is_all_triangles`` is now a property.  Calling this value
           will warn the user that this should not be called.
           Additionally, the ``is`` operator will not work the return
           value of this property since it is not a ``bool``

        Returns
        -------
        CallableBool
            ``True`` if all the faces of the :class:`pyvista.PolyData`
            are triangles and does not contain any vertices or lines.

        Notes
        -----
        The return value is not a ``bool`` for compatibility
        reasons, though this behavior will change in a future
        release.  Future versions will simply return a ``bool``.

        Examples
        --------
        Show a mesh from :func:`pyvista.Plane` is not composed of all
        triangles.

        >>> import pyvista
        >>> plane = pyvista.Plane()
        >>> plane.is_all_triangles
        False <CallableBool>

        Show that the mesh from :func:`pyvista.Sphere` contains only
        triangles.

        >>> sphere = pyvista.Sphere()
        >>> sphere.is_all_triangles
        True <CallableBool>

        """
        class CallableBool(int):  # pragma: no cover
            """Boolean that can be called.

            Programmer note: We must subclass int and not bool
            https://stackoverflow.com/questions/2172189/why-i-cant-extend-bool-in-python

            Implemented for backwards compatibility as
            ``is_all_triangles`` was changed to be a property in
            ``0.32.0``.

            """

            def __new__(cls, value):
                """Use new instead of __init__.

                See:
                https://jfine-python-classes.readthedocs.io/en/latest/subclass-int.html#emulating-bool-using-new

                """
                return int.__new__(cls, bool(value))

            def __call__(self):
                """Return a ``bool`` of self."""
                warnings.warn('``is_all_triangles`` is now property as of 0.32.0 and '
                              'does not need ()', DeprecationWarning)
                return bool(self)

            def __repr__(self):
                """Return the string of bool."""
                return f'{bool(self)} <CallableBool>'

        # Need to make sure there are only face cells and no lines/verts
        if not self.n_faces or self.n_lines or self.n_verts:
            return CallableBool(False)

        # in VTK9, they use connectivity and offset rather than cell
        # data.  Use the new API as this is faster
        if _vtk.VTK9:
            # early return if not all triangular
            if self._connectivity_array.size % 3:
                return CallableBool(False)

            # next, check if there are three points per face
            return CallableBool((np.diff(self._offset_array) == 3).all())

        else:  # pragma: no cover
            # All we have are faces, check if all faces are indeed triangles
            faces = self.faces  # grab once as this takes time to build
            if faces.size % 4 == 0:
                return CallableBool((faces[::4] == 3).all())
            return CallableBool(False)

    def __sub__(self, cutting_mesh):
        """Compute boolean difference of two meshes."""
        return self.boolean_difference(cutting_mesh)

    @property
    def _offset_array(self):
        """Return the array used to store cell offsets."""
        try:
            return _vtk.vtk_to_numpy(self.GetPolys().GetOffsetsArray())
        except AttributeError:  # pragma: no cover
            raise VTKVersionError('Offset array implemented in VTK 9 or newer.')

    @property
    def _connectivity_array(self):
        """Return the array with the point ids that define the cells connectivity."""
        try:
            return _vtk.vtk_to_numpy(self.GetPolys().GetConnectivityArray())
        except AttributeError:  # pragma: no cover
            raise VTKVersionError('Connectivity array implemented in VTK 9 or newer.')

    @property
    def n_lines(self) -> int:
        """Return the number of lines.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Line()
        >>> mesh.n_lines
        1

        """
        return self.GetNumberOfLines()

    @property
    def n_verts(self) -> int:
        """Return the number of vertices.

        Examples
        --------
        Create a simple mesh containing just two points and return the
        number of vertices.

        >>> import pyvista
        >>> mesh = pyvista.PolyData([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        >>> mesh.n_verts
        2

        """
        return self.GetNumberOfVerts()

    @property
    def n_faces(self) -> int:
        """Return the number of cells.

        Alias for ``n_cells``.

        Examples
        --------
        >>> import pyvista
        >>> plane = pyvista.Plane(i_resolution=2, j_resolution=2)
        >>> plane.n_faces
        4

        """
        return self.n_cells

    @property
    def number_of_faces(self):  # pragma: no cover
        """Return the number of cells."""
        raise DeprecationError('``number_of_faces`` has been depreciated.  '
                               'Please use ``n_faces``')

    def save(self, filename, binary=True, texture=None):
        """Write a surface mesh to disk.

        Written file may be an ASCII or binary ply, stl, or vtk mesh
        file. If ply or stl format is chosen, the face normals are
        computed in place to ensure the mesh is properly saved.

        Parameters
        ----------
        filename : str
            Filename of mesh to be written.  File type is inferred from
            the extension of the filename unless overridden with
            ftype.  Can be one of many of the supported  the following
            types (``'.ply'``, ``'.stl'``, ``'.vtk``).

        binary : bool, optional
            Writes the file as binary when ``True`` and ASCII when ``False``.

        texture : str, np.ndarray, optional
            Write a single texture array to file when using a PLY
            file.  Texture array must be a 3 or 4 component array with
            the datatype ``np.uint8``.  Array may be a cell array or a
            point array, and may also be a string if the array already
            exists in the PolyData.

            If a string is provided, the texture array will be saved
            to disk as that name.  If an array is provided, the
            texture array will be saved as ``'RGBA'`` if the array
            contains an alpha channel (i.e. 4 component array), or
            as ``'RGB'`` if the array is just a 3 component array.

            .. note::
               This feature is only available when saving PLY files.

        Notes
        -----
        Binary files write much faster than ASCII and have a smaller
        file size.

        Examples
        --------
        Save a mesh as a STL.

        >>> import pyvista
        >>> sphere = pyvista.Sphere()
        >>> sphere.save('my_mesh.stl')  # doctest:+SKIP

        Save a mesh as a PLY.

        >>> sphere = pyvista.Sphere()
        >>> sphere.save('my_mesh.ply')  # doctest:+SKIP

        Save a mesh as a PLY with a texture array.  Here we also
        create a simple RGB array representing the texture.

        >>> import numpy as np
        >>> sphere = pyvista.Sphere()
        >>> texture = np.zeros((sphere.n_points, 3), np.uint8)
        >>> texture[:, 1] = np.arange(sphere.n_points)[::-1]  # just blue channel
        >>> sphere.point_data['my_texture'] = texture
        >>> sphere.save('my_mesh.ply', texture='my_texture')  # doctest:+SKIP

        Alternatively, provide just the texture array.  This will be
        written to the file as ``'RGB'`` since it does not contain an
        alpha channel.

        >>> sphere.save('my_mesh.ply', texture=texture)  # doctest:+SKIP

        Save a mesh as a VTK file.

        >>> sphere = pyvista.Sphere()
        >>> sphere.save('my_mesh.vtk')  # doctest:+SKIP

        """
        filename = os.path.abspath(os.path.expanduser(str(filename)))
        ftype = get_ext(filename)
        # Recompute normals prior to save.  Corrects a bug were some
        # triangular meshes are not saved correctly
        if ftype in ['.stl', '.ply']:
            self.compute_normals(inplace=True)

        # validate texture
        if ftype == '.ply' and texture is not None:
            if isinstance(texture, str):
                if self[texture].dtype != np.uint8:
                    raise ValueError(f'Invalid datatype {self[texture].dtype} of '
                                     f'texture array "{texture}"')
            elif isinstance(texture, np.ndarray):
                if texture.dtype != np.uint8:
                    raise ValueError(f'Invalid datatype {texture.dtype} of texture array')
            else:
                raise TypeError(f'Invalid type {type(texture)} for texture.  '
                                'Should be either a string representing a point or '
                                'cell array, or a numpy array.')

        super().save(filename, binary, texture=texture)

    @property
    def area(self) -> float:
        """Return the mesh surface area.

        Returns
        -------
        float
            Total area of the mesh.

        Examples
        --------
        >>> import pyvista
        >>> sphere = pyvista.Sphere()
        >>> sphere.area
        3.126

        """
        areas = self.compute_cell_sizes(length=False, area=True, volume=False,)["Area"]
        return np.sum(areas)

    @property
    def volume(self) -> float:
        """Return the volume of the dataset.

        This will throw a VTK error/warning if not a closed surface.

        Returns
        -------
        float
            Total volume of the mesh.

        Examples
        --------
        >>> import pyvista
        >>> sphere = pyvista.Sphere()
        >>> sphere.volume
        0.5183

        """
        mprop = _vtk.vtkMassProperties()
        mprop.SetInputData(self.triangulate())
        return mprop.GetVolume()

    @property
    def point_normals(self) -> 'pyvista.pyvista_ndarray':
        """Return the point normals.

        Returns
        -------
        pyvista.pyvista_ndarray
            Array of point normals.

        Examples
        --------
        >>> import pyvista
        >>> sphere = pyvista.Sphere()
        >>> sphere.point_normals  # doctest:+SKIP
        pyvista_ndarray([[-2.48721432e-10, -1.08815623e-09, -1.00000000e+00],
                         [-2.48721432e-10, -1.08815623e-09,  1.00000000e+00],
                         [-1.18888125e-01,  3.40539310e-03, -9.92901802e-01],
                         ...,
                         [-3.11940581e-01, -6.81432486e-02,  9.47654784e-01],
                         [-2.09880397e-01, -4.65070531e-02,  9.76620376e-01],
                         [-1.15582108e-01, -2.80492082e-02,  9.92901802e-01]],
                        dtype=float32)

        """
        mesh = self.compute_normals(cell_normals=False, inplace=False)
        return mesh.point_data['Normals']

    @property
    def cell_normals(self) -> 'pyvista.pyvista_ndarray':
        """Return the cell normals.

        Returns
        -------
        pyvista.pyvista_ndarray
            Array of cell normals.

        Examples
        --------
        >>> import pyvista
        >>> sphere = pyvista.Sphere()
        >>> sphere.cell_normals  # doctest:+SKIP
        pyvista_ndarray([[-0.05413816,  0.00569015, -0.9985172 ],
                         [-0.05177207,  0.01682176, -0.9985172 ],
                         [-0.04714328,  0.02721819, -0.9985172 ],
                         ...,
                         [-0.26742265, -0.02810723,  0.96316934],
                         [-0.1617585 , -0.01700151,  0.9866839 ],
                         [-0.1617585 , -0.01700151,  0.9866839 ]], dtype=float32)

        """
        mesh = self.compute_normals(point_normals=False, inplace=False)
        return mesh.cell_data['Normals']

    @property
    def face_normals(self) -> 'pyvista.pyvista_ndarray':
        """Return the cell normals.

        Alias to :func:`PolyData.cell_normals`.

        Returns
        -------
        pyvista.pyvista_ndarray
            Array of face normals.

        Examples
        --------
        >>> import pyvista
        >>> sphere = pyvista.Sphere()
        >>> sphere.face_normals  # doctest:+SKIP
        pyvista_ndarray([[-0.05413816,  0.00569015, -0.9985172 ],
                         [-0.05177207,  0.01682176, -0.9985172 ],
                         [-0.04714328,  0.02721819, -0.9985172 ],
                         ...,
                         [-0.26742265, -0.02810723,  0.96316934],
                         [-0.1617585 , -0.01700151,  0.9866839 ],
                         [-0.1617585 , -0.01700151,  0.9866839 ]], dtype=float32)

        """
        return self.cell_normals

    @property
    def obbTree(self):
        """Return the obbTree of the polydata.

        An obbTree is an object to generate oriented bounding box (OBB)
        trees. An oriented bounding box is a bounding box that does not
        necessarily line up along coordinate axes. The OBB tree is a
        hierarchical tree structure of such boxes, where deeper levels of OBB
        confine smaller regions of space.
        """
        if not hasattr(self, '_obbTree'):
            self._obbTree = _vtk.vtkOBBTree()
            self._obbTree.SetDataSet(self)
            self._obbTree.BuildLocator()

        return self._obbTree

    @property
    def n_open_edges(self) -> int:
        """Return the number of open edges on this mesh.

        Examples
        --------
        Return the number of open edges on a sphere.

        >>> import pyvista
        >>> sphere = pyvista.Sphere()
        >>> sphere.n_open_edges
        0

        Return the number of open edges on a plane.

        >>> plane = pyvista.Plane(i_resolution=1, j_resolution=1)
        >>> plane.n_open_edges
        4

        """
        alg = _vtk.vtkFeatureEdges()
        alg.FeatureEdgesOff()
        alg.BoundaryEdgesOn()
        alg.NonManifoldEdgesOn()
        alg.SetInputDataObject(self)
        alg.Update()
        return alg.GetOutput().GetNumberOfCells()

    @property
    def is_manifold(self) -> bool:
        """Return if the mesh is manifold (no open edges).

        Examples
        --------
        Show a sphere is manifold.

        >>> import pyvista
        >>> pyvista.Sphere().is_manifold
        True

        Show a plane is not manifold.

        >>> pyvista.Plane().is_manifold
        False

        """
        return self.n_open_edges == 0

    def __del__(self):
        """Delete the object."""
        if hasattr(self, '_obbTree'):
            del self._obbTree


@abstract_class
class PointGrid(PointSet):
    """Class in common with structured and unstructured grids."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the point grid."""
        super().__init__()

    def plot_curvature(self, curv_type='mean', **kwargs):
        """Plot the curvature of the external surface of the grid.

        Parameters
        ----------
        curv_type : str, optional
            One of the following strings indicating curvature types.
            - ``'mean'``
            - ``'gaussian'``
            - ``'maximum'``
            - ``'minimum'``

        **kwargs : dict, optional
            Optional keyword arguments.  See :func:`pyvista.plot`.

        Returns
        -------
        list
            Camera position, focal point, and view up.  Returned when
            ``return_cpos`` is ``True``.

        """
        trisurf = self.extract_surface().triangulate()
        return trisurf.plot_curvature(curv_type, **kwargs)

    @property
    def volume(self) -> float:
        """Compute the volume of the point grid.

        This extracts the external surface and computes the interior
        volume.
        """
        surf = self.extract_surface().triangulate()
        return surf.volume


class UnstructuredGrid(_vtk.vtkUnstructuredGrid, PointGrid, UnstructuredGridFilters):
    """Dataset used for arbitrary combinations of all possible cell types.

    Can be initialized by the following:

    - Creating an empty grid
    - From a ``vtk.vtkPolyData`` or ``vtk.vtkStructuredGrid`` object
    - From cell, offset, and node arrays
    - From a file

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> import vtk

    Create an empty grid

    >>> grid = pyvista.UnstructuredGrid()

    Copy a vtkUnstructuredGrid

    >>> vtkgrid = vtk.vtkUnstructuredGrid()
    >>> grid = pyvista.UnstructuredGrid(vtkgrid)  # Initialize from a vtkUnstructuredGrid

    >>> # from arrays (vtk9)
    >>> #grid = pyvista.UnstructuredGrid(cells, celltypes, points)

    >>> # from arrays (vtk<9)
    >>> #grid = pyvista.UnstructuredGrid(offset, cells, celltypes, points)

    From a string filename

    >>> grid = pyvista.UnstructuredGrid(examples.hexbeamfile)

    """

    _WRITERS = {'.vtu': _vtk.vtkXMLUnstructuredGridWriter,
                '.vtk': _vtk.vtkUnstructuredGridWriter}

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the unstructured grid."""
        super().__init__()
        deep = kwargs.pop('deep', False)

        if not len(args):
            return
        if len(args) == 1:
            if isinstance(args[0], _vtk.vtkUnstructuredGrid):
                if deep:
                    self.deep_copy(args[0])
                else:
                    self.shallow_copy(args[0])

            elif isinstance(args[0], (str, pathlib.Path)):
                self._from_file(args[0], **kwargs)

            elif isinstance(args[0], (_vtk.vtkStructuredGrid, _vtk.vtkPolyData)):
                vtkappend = _vtk.vtkAppendFilter()
                vtkappend.AddInputData(args[0])
                vtkappend.Update()
                self.shallow_copy(vtkappend.GetOutput())

            else:
                itype = type(args[0])
                raise TypeError(f'Cannot work with input type {itype}')

        # Cell dictionary creation
        elif len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], np.ndarray):
            self._from_cells_dict(args[0], args[1], deep)
            self._check_for_consistency()

        elif len(args) == 3:  # and VTK9:
            arg0_is_arr = isinstance(args[0], np.ndarray)
            arg1_is_arr = isinstance(args[1], np.ndarray)
            arg2_is_arr = isinstance(args[2], np.ndarray)

            if all([arg0_is_arr, arg1_is_arr, arg2_is_arr]):
                self._from_arrays(None, args[0], args[1], args[2], deep, **kwargs)
                self._check_for_consistency()
            else:
                raise TypeError('All input types must be np.ndarray')

        elif len(args) == 4:
            arg0_is_arr = isinstance(args[0], np.ndarray)
            arg1_is_arr = isinstance(args[1], np.ndarray)
            arg2_is_arr = isinstance(args[2], np.ndarray)
            arg3_is_arr = isinstance(args[3], np.ndarray)

            if all([arg0_is_arr, arg1_is_arr, arg2_is_arr, arg3_is_arr]):
                self._from_arrays(args[0], args[1], args[2], args[3], deep)
                self._check_for_consistency()
            else:
                raise TypeError('All input types must be np.ndarray')

        else:
            err_msg = 'Invalid parameters.  Initialization with arrays ' +\
                      'requires the following arrays:\n'
            if _vtk.VTK9:
                raise TypeError(err_msg + '`cells`, `cell_type`, `points`')
            else:
                raise TypeError(err_msg + '(`offset` optional), `cells`, `cell_type`, `points`')

    def __repr__(self):
        """Return the standard representation."""
        return DataSet.__repr__(self)

    def __str__(self):
        """Return the standard str representation."""
        return DataSet.__str__(self)

    def _from_cells_dict(self, cells_dict, points, deep=True):
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("Points array must be a [M, 3] array")

        nr_points = points.shape[0]
        if _vtk.VTK9:
            cell_types, cells = create_mixed_cells(cells_dict, nr_points)
            self._from_arrays(None, cells, cell_types, points, deep=deep)
        else:
            cell_types, cells, offset = create_mixed_cells(cells_dict, nr_points)
            self._from_arrays(offset, cells, cell_types, points, deep=deep)

    def _from_arrays(
            self, offset, cells, cell_type, points, deep=True, force_float=True,
    ):
        """Create VTK unstructured grid from numpy arrays.

        Parameters
        ----------
        offset : np.ndarray dtype=np.int64
            Array indicating the start location of each cell in the cells
            array.  Set to ``None`` when using VTK 9+.

        cells : np.ndarray dtype=np.int64
            Array of cells.  Each cell contains the number of points in the
            cell and the node numbers of the cell.

        cell_type : np.uint8
            Cell types of each cell.  Each cell type numbers can be found from
            vtk documentation.  See example below.

        points : np.ndarray
            Numpy array containing point locations.

        deep : bool, optional
            When ``True``, makes a copy of the points array.  Default
            ``False``.  Cells and cell types are always copied.

        force_float : bool, optional
            Casts the datatype to ``float32`` if points datatype is
            non-float.  Default ``True``. Set this to ``False`` to allow
            non-float types, though this may lead to truncation of
            intermediate floats when transforming datasets.

        Examples
        --------
        >>> import numpy as np
        >>> import vtk
        >>> import pyvista
        >>> offset = np.array([0, 9])
        >>> cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])
        >>> cell_type = np.array([vtk.VTK_HEXAHEDRON, vtk.VTK_HEXAHEDRON], np.int8)

        >>> cell1 = np.array([[0, 0, 0],
        ...                   [1, 0, 0],
        ...                   [1, 1, 0],
        ...                   [0, 1, 0],
        ...                   [0, 0, 1],
        ...                   [1, 0, 1],
        ...                   [1, 1, 1],
        ...                   [0, 1, 1]], dtype=np.float32)

        >>> cell2 = np.array([[0, 0, 2],
        ...                   [1, 0, 2],
        ...                   [1, 1, 2],
        ...                   [0, 1, 2],
        ...                   [0, 0, 3],
        ...                   [1, 0, 3],
        ...                   [1, 1, 3],
        ...                   [0, 1, 3]], dtype=np.float32)

        >>> points = np.vstack((cell1, cell2))

        >>> grid = pyvista.UnstructuredGrid(offset, cells, cell_type, points)

        """
        # Convert to vtk arrays
        vtkcells = CellArray(cells, cell_type.size, deep)
        if cell_type.dtype != np.uint8:
            cell_type = cell_type.astype(np.uint8)
        cell_type_np = cell_type
        cell_type = _vtk.numpy_to_vtk(cell_type, deep=deep)

        points = pyvista.vtk_points(points, deep, force_float)
        self.SetPoints(points)

        # vtk9 does not require an offset array
        if _vtk.VTK9:
            if offset is not None:
                warnings.warn('VTK 9 no longer accepts an offset array',
                              stacklevel=3)
            self.SetCells(cell_type, vtkcells)
        else:
            if offset is None:
                offset = generate_cell_offsets(cells, cell_type_np)

            self.SetCells(cell_type, numpy_to_idarr(offset), vtkcells)

    def _check_for_consistency(self):
        """Check if size of offsets and celltypes match the number of cells.

        Checks if the number of offsets and celltypes correspond to
        the number of cells.  Called after initialization of the self
        from arrays.
        """
        if self.n_cells != self.celltypes.size:
            raise ValueError(f'Number of cell types ({self.celltypes.size}) '
                             f'must match the number of cells {self.n_cells})')

        if _vtk.VTK9:
            if self.n_cells != self.offset.size - 1:
                raise ValueError(f'Size of the offset ({self.offset.size}) '
                                 'must be one greater than the number of cells '
                                 f'({self.n_cells})')
        else:
            if self.n_cells != self.offset.size:
                raise ValueError(f'Size of the offset ({self.offset.size}) '
                                 f'must match the number of cells ({self.n_cells})')

    @property
    def cells(self) -> np.ndarray:
        """Return a pointer to the cells as a numpy object.

        Examples
        --------
        Return the indices of the first two cells from the example hex
        beam.  Note how the cells have "padding" indicating the number
        of points per cell.

        >>> import pyvista
        >>> from pyvista import examples
        >>> hex_beam = pyvista.read(examples.hexbeamfile)
        >>> hex_beam.cells[:18]  # doctest:+SKIP
        array([ 8,  0,  2,  8,  7, 27, 36, 90, 81,  8,  2,  1,  4,
                8, 36, 18, 54, 90])

        """
        return _vtk.vtk_to_numpy(self.GetCells().GetData())

    @property
    def cells_dict(self) -> dict:
        """Return a dictionary that contains all cells mapped from cell types.

        This function returns a :class:`numpy.ndarray` for each cell
        type in an ordered fashion.  Note that this function only
        works with element types of fixed sizes.

        Returns
        -------
        dict
            A dictionary mapping containing all cells of this unstructured grid.
            Structure: vtk_enum_type (int) -> cells (np.ndarray)

        Examples
        --------
        Return the cells dictionary of the sample hex beam.  Note how
        there is only one key/value pair as the hex beam example is
        composed of only all hexahedral cells, which is
        ``vtk.VTK_HEXAHEDRON``, which evaluates to 12.

        Also note how there is no padding for the cell array.  This
        approach may be more helpful than the ``cells`` property when
        extracting cells.

        >>> import pyvista
        >>> from pyvista import examples
        >>> hex_beam = pyvista.read(examples.hexbeamfile)
        >>> hex_beam.cells_dict  # doctest:+SKIP
        {12: array([[ 0,  2,  8,  7, 27, 36, 90, 81],
                [ 2,  1,  4,  8, 36, 18, 54, 90],
                [ 7,  8,  6,  5, 81, 90, 72, 63],
                ...
                [44, 26, 62, 98, 11, 10, 13, 17],
                [89, 98, 80, 71, 16, 17, 15, 14],
                [98, 62, 53, 80, 17, 13, 12, 15]])}
        """
        return get_mixed_cells(self)

    @property
    def cell_connectivity(self) -> np.ndarray:
        """Return a the vtk cell connectivity as a numpy array.

        This is effecively :attr:`UnstructuredGrid.cells` without the
        padding.

        .. note::
           This is only available in ``vtk>=9.0.0``.

        Returns
        -------
        numpy.ndarray
            Connectivity array.

        Examples
        --------
        Return the cell connectivity for the first two cells.

        >>> import pyvista
        >>> from pyvista import examples
        >>> hex_beam = pyvista.read(examples.hexbeamfile)
        >>> hex_beam.cell_connectivity[:16]
        array([ 0,  2,  8,  7, 27, 36, 90, 81,  2,  1,  4,  8, 36, 18, 54, 90])

        """
        carr = self.GetCells()
        if _vtk.VTK9:
            return _vtk.vtk_to_numpy(carr.GetConnectivityArray())
        raise AttributeError('Install vtk>=9.0.0 for `cell_connectivity`\n'
                             'Otherwise, use the legacy `cells` method')

    def linear_copy(self, deep=False):
        """Return a copy of the unstructured grid containing only linear cells.

        Converts the following cell types to their linear equivalents.

        - ``VTK_QUADRATIC_TETRA      --> VTK_TETRA``
        - ``VTK_QUADRATIC_PYRAMID    --> VTK_PYRAMID``
        - ``VTK_QUADRATIC_WEDGE      --> VTK_WEDGE``
        - ``VTK_QUADRATIC_HEXAHEDRON --> VTK_HEXAHEDRON``

        Parameters
        ----------
        deep : bool
            When ``True``, makes a copy of the points array.  Default
            ``False``.  Cells and cell types are always copied.

        Returns
        -------
        pyvista.UnstructuredGrid
            UnstructuredGrid containing only linear cells when
            ``deep=False``.

        """
        lgrid = self.copy(deep)

        # grab the vtk object
        vtk_cell_type = _vtk.numpy_to_vtk(self.GetCellTypesArray(), deep=True)
        celltype = _vtk.vtk_to_numpy(vtk_cell_type)
        celltype[celltype == _vtk.VTK_QUADRATIC_TETRA] = _vtk.VTK_TETRA
        celltype[celltype == _vtk.VTK_QUADRATIC_PYRAMID] = _vtk.VTK_PYRAMID
        celltype[celltype == _vtk.VTK_QUADRATIC_WEDGE] = _vtk.VTK_WEDGE
        celltype[celltype == _vtk.VTK_QUADRATIC_HEXAHEDRON] = _vtk.VTK_HEXAHEDRON

        # track quad mask for later
        quad_quad_mask = celltype == _vtk.VTK_QUADRATIC_QUAD
        celltype[quad_quad_mask] = _vtk.VTK_QUAD

        quad_tri_mask = celltype == _vtk.VTK_QUADRATIC_TRIANGLE
        celltype[quad_tri_mask] = _vtk.VTK_TRIANGLE

        vtk_offset = self.GetCellLocationsArray()
        cells = _vtk.vtkCellArray()
        cells.DeepCopy(self.GetCells())
        lgrid.SetCells(vtk_cell_type, vtk_offset, cells)

        # fixing bug with display of quad cells
        if np.any(quad_quad_mask):
            if _vtk.VTK9:
                quad_offset = lgrid.offset[:-1][quad_quad_mask]
                base_point = lgrid.cell_connectivity[quad_offset]
                lgrid.cell_connectivity[quad_offset + 4] = base_point
                lgrid.cell_connectivity[quad_offset + 5] = base_point
                lgrid.cell_connectivity[quad_offset + 6] = base_point
                lgrid.cell_connectivity[quad_offset + 7] = base_point
            else:
                quad_offset = lgrid.offset[quad_quad_mask]
                base_point = lgrid.cells[quad_offset + 1]
                lgrid.cells[quad_offset + 5] = base_point
                lgrid.cells[quad_offset + 6] = base_point
                lgrid.cells[quad_offset + 7] = base_point
                lgrid.cells[quad_offset + 8] = base_point

        if np.any(quad_tri_mask):
            if _vtk.VTK9:
                tri_offset = lgrid.offset[:-1][quad_tri_mask]
                base_point = lgrid.cell_connectivity[tri_offset]
                lgrid.cell_connectivity[tri_offset + 3] = base_point
                lgrid.cell_connectivity[tri_offset + 4] = base_point
                lgrid.cell_connectivity[tri_offset + 5] = base_point
            else:
                tri_offset = lgrid.offset[quad_tri_mask]
                base_point = lgrid.cells[tri_offset + 1]
                lgrid.cells[tri_offset + 4] = base_point
                lgrid.cells[tri_offset + 5] = base_point
                lgrid.cells[tri_offset + 6] = base_point

        return lgrid

    @property
    def celltypes(self) -> np.ndarray:
        """Return the cell types array.

        Returns
        -------
        numpy.ndarray
            Array of VTK cell types.  Some of the most popular cell types:

        * ``VTK_EMPTY_CELL = 0``
        * ``VTK_VERTEX = 1``
        * ``VTK_POLY_VERTEX = 2``
        * ``VTK_LINE = 3``
        * ``VTK_POLY_LINE = 4``
        * ``VTK_TRIANGLE = 5``
        * ``VTK_TRIANGLE_STRIP = 6``
        * ``VTK_POLYGON = 7``
        * ``VTK_PIXEL = 8``
        * ``VTK_QUAD = 9``
        * ``VTK_TETRA = 10``
        * ``VTK_VOXEL = 11``
        * ``VTK_HEXAHEDRON = 12``
        * ``VTK_WEDGE = 13``
        * ``VTK_PYRAMID = 14``
        * ``VTK_PENTAGONAL_PRISM = 15``
        * ``VTK_HEXAGONAL_PRISM = 16``
        * ``VTK_QUADRATIC_EDGE = 21``
        * ``VTK_QUADRATIC_TRIANGLE = 22``
        * ``VTK_QUADRATIC_QUAD = 23``
        * ``VTK_QUADRATIC_POLYGON = 36``
        * ``VTK_QUADRATIC_TETRA = 24``
        * ``VTK_QUADRATIC_HEXAHEDRON = 25``
        * ``VTK_QUADRATIC_WEDGE = 26``
        * ``VTK_QUADRATIC_PYRAMID = 27``
        * ``VTK_BIQUADRATIC_QUAD = 28``
        * ``VTK_TRIQUADRATIC_HEXAHEDRON = 29``
        * ``VTK_QUADRATIC_LINEAR_QUAD = 30``
        * ``VTK_QUADRATIC_LINEAR_WEDGE = 31``
        * ``VTK_BIQUADRATIC_QUADRATIC_WEDGE = 32``
        * ``VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33``
        * ``VTK_BIQUADRATIC_TRIANGLE = 34``

        See
        https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
        for all cell types.

        Examples
        --------
        This mesh contains only linear hexahedral cells, type
        ``vtk.VTK_HEXAHEDRON``, which evaluates to 12.

        >>> import pyvista
        >>> from pyvista import examples
        >>> hex_beam = pyvista.read(examples.hexbeamfile)
        >>> hex_beam.celltypes  # doctest:+SKIP
        array([12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
               dtype=uint8)

        """
        return _vtk.vtk_to_numpy(self.GetCellTypesArray())

    @property
    def offset(self) -> np.ndarray:
        """Return the cell locations array.

        In VTK 9, this is the location of the start of each cell in
        :attr:`cell_connectivity`, and in VTK < 9, this is the
        location of the start of each cell in :attr:`cells`.

        Returns
        -------
        numpy.ndarray
            Array of cell offsets indicating the start of each cell.

        Examples
        --------
        Return the cell offset array within ``vtk==9``.  Since this
        mesh is composed of all hexahedral cells, note how each cell
        starts at 8 greater than the prior cell.

        >>> import pyvista
        >>> from pyvista import examples
        >>> hex_beam = pyvista.read(examples.hexbeamfile)
        >>> hex_beam.offset
        array([  0,   8,  16,  24,  32,  40,  48,  56,  64,  72,  80,  88,  96,
               104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200,
               208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304,
               312, 320])

        """
        carr = self.GetCells()
        if _vtk.VTK9:
            # This will be the number of cells + 1.
            return _vtk.vtk_to_numpy(carr.GetOffsetsArray())
        else:  # this is no longer used in >= VTK9
            return _vtk.vtk_to_numpy(self.GetCellLocationsArray())

    def cast_to_explicit_structured_grid(self):
        """Cast to an explicit structured grid.

        .. note::
           This feature is only available in ``vtk>=9.0.0``

        Returns
        -------
        pyvista.ExplicitStructuredGrid
            An explicit structured grid.

        Raises
        ------
        TypeError
            If the unstructured grid doesn't have the ``'BLOCK_I'``,
            ``'BLOCK_J'`` and ``'BLOCK_K'`` cells arrays.

        See Also
        --------
        pyvista.ExplicitStructuredGrid.cast_to_unstructured_grid

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        >>> grid = grid.hide_cells(range(80, 120))
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        >>> grid = grid.cast_to_unstructured_grid()
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        >>> grid = grid.cast_to_explicit_structured_grid()
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        """
        if not _vtk.VTK9:
            raise AttributeError('VTK 9 or higher is required')
        s1 = {'BLOCK_I', 'BLOCK_J', 'BLOCK_K'}
        s2 = self.cell_data.keys()
        if not s1.issubset(s2):
            raise TypeError("'BLOCK_I', 'BLOCK_J' and 'BLOCK_K' cell arrays are required")
        alg = _vtk.vtkUnstructuredGridToExplicitStructuredGrid()
        alg.SetInputData(self)
        alg.SetInputArrayToProcess(0, 0, 0, 1, 'BLOCK_I')
        alg.SetInputArrayToProcess(1, 0, 0, 1, 'BLOCK_J')
        alg.SetInputArrayToProcess(2, 0, 0, 1, 'BLOCK_K')
        alg.Update()
        grid = _get_output(alg)
        grid.cell_data.remove('ConnectivityFlags')  # unrequired
        return grid


class StructuredGrid(_vtk.vtkStructuredGrid, PointGrid, StructuredGridFilters):
    """Dataset used for topologically regular arrays of data.

    Can be initialized in one of the following several ways:

    - Create empty grid
    - Initialize from a vtk.vtkStructuredGrid object
    - Initialize directly from the point arrays

    See _from_arrays in the documentation for more details on initializing
    from point arrays

    Examples
    --------
    >>> import pyvista
    >>> import vtk
    >>> import numpy as np

    Create empty grid

    >>> grid = pyvista.StructuredGrid()

    Initialize from a vtk.vtkStructuredGrid object

    >>> vtkgrid = vtk.vtkStructuredGrid()
    >>> grid = pyvista.StructuredGrid(vtkgrid)

    Create from NumPy arrays

    >>> xrng = np.arange(-10, 10, 2, dtype=np.float32)
    >>> yrng = np.arange(-10, 10, 2, dtype=np.float32)
    >>> zrng = np.arange(-10, 10, 2, dtype=np.float32)
    >>> x, y, z = np.meshgrid(xrng, yrng, zrng)
    >>> grid = pyvista.StructuredGrid(x, y, z)

    """

    _WRITERS = {'.vtk': _vtk.vtkStructuredGridWriter,
                '.vts': _vtk.vtkXMLStructuredGridWriter}

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the structured grid."""
        super().__init__()

        if len(args) == 1:
            if isinstance(args[0], _vtk.vtkStructuredGrid):
                self.deep_copy(args[0])
            elif isinstance(args[0], (str, pathlib.Path)):
                self._from_file(args[0], **kwargs)

        elif len(args) == 3:
            arg0_is_arr = isinstance(args[0], np.ndarray)
            arg1_is_arr = isinstance(args[1], np.ndarray)
            arg2_is_arr = isinstance(args[2], np.ndarray)

            if all([arg0_is_arr, arg1_is_arr, arg2_is_arr]):
                self._from_arrays(args[0], args[1], args[2], **kwargs)

    def __repr__(self):
        """Return the standard representation."""
        return DataSet.__repr__(self)

    def __str__(self):
        """Return the standard str representation."""
        return DataSet.__str__(self)

    def _from_arrays(self, x, y, z, force_float=True):
        """Create VTK structured grid directly from numpy arrays.

        Parameters
        ----------
        x : np.ndarray
            Position of the points in x direction.

        y : np.ndarray
            Position of the points in y direction.

        z : np.ndarray
            Position of the points in z direction.

        force_float : bool, optional
            Casts the datatype to ``float32`` if points datatype is
            non-float.  Default ``True``. Set this to ``False`` to allow
            non-float types, though this may lead to truncation of
            intermediate floats when transforming datasets.

        """
        if not(x.shape == y.shape == z.shape):
            raise ValueError('Input point array shapes must match exactly')

        # make the output points the same precision as the input arrays
        points = np.empty((x.size, 3), x.dtype)
        points[:, 0] = x.ravel('F')
        points[:, 1] = y.ravel('F')
        points[:, 2] = z.ravel('F')

        # ensure that the inputs are 3D
        dim = list(x.shape)
        while len(dim) < 3:
            dim.append(1)

        # Create structured grid
        self.SetDimensions(dim)
        self.SetPoints(pyvista.vtk_points(points, force_float=force_float))

    @property
    def dimensions(self):
        """Return a length 3 tuple of the grid's dimensions.

        Returns
        -------
        tuple
            Grid dimensions.

        Examples
        --------
        >>> import pyvista
        >>> import numpy as np
        >>> xrng = np.arange(-10, 10, 1, dtype=np.float32)
        >>> yrng = np.arange(-10, 10, 2, dtype=np.float32)
        >>> zrng = np.arange(-10, 10, 5, dtype=np.float32)
        >>> x, y, z = np.meshgrid(xrng, yrng, zrng)
        >>> grid = pyvista.StructuredGrid(x, y, z)
        >>> grid.dimensions
        (10, 20, 4)

        """
        return tuple(self.GetDimensions())

    @dimensions.setter
    def dimensions(self, dims):
        """Set the dataset dimensions. Pass a length three tuple of integers."""
        nx, ny, nz = dims[0], dims[1], dims[2]
        self.SetDimensions(nx, ny, nz)
        self.Modified()

    @property
    def x(self):
        """Return the X coordinates of all points.

        Returns
        -------
        numpy.ndarray
            Numpy array of all X coordinates.

        Examples
        --------
        >>> import pyvista
        >>> import numpy as np
        >>> xrng = np.arange(-10, 10, 1, dtype=np.float32)
        >>> yrng = np.arange(-10, 10, 2, dtype=np.float32)
        >>> zrng = np.arange(-10, 10, 5, dtype=np.float32)
        >>> x, y, z = np.meshgrid(xrng, yrng, zrng)
        >>> grid = pyvista.StructuredGrid(x, y, z)
        >>> grid.x.shape
        (10, 20, 4)

        """
        return self._reshape_point_array(self.points[:, 0])

    @property
    def y(self):
        """Return the Y coordinates of all points."""
        return self._reshape_point_array(self.points[:, 1])

    @property
    def z(self):
        """Return the Z coordinates of all points."""
        return self._reshape_point_array(self.points[:, 2])

    @property
    def points_matrix(self):
        """Points as a 4-D matrix, with x/y/z along the last dimension."""
        return self.points.reshape((*self.dimensions, 3), order='F')

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = PointGrid._get_attrs(self)
        attrs.append(("Dimensions", self.dimensions, "{:d}, {:d}, {:d}"))
        return attrs

    def __getitem__(self, key):
        """Slice subsets of the StructuredGrid, or extract an array field."""
        # legacy behavior which looks for a point or cell array
        if not isinstance(key, tuple):
            return super().__getitem__(key)

        # convert slice to VOI specification - only "basic indexing" is supported
        voi = []
        rate = []
        if len(key) != 3:
            raise RuntimeError('Slices must have exactly 3 dimensions.')
        for i, k in enumerate(key):
            if isinstance(k, collections.Iterable):
                raise RuntimeError('Fancy indexing is not supported.')
            if isinstance(k, numbers.Integral):
                start = stop = k
                step = 1
            elif isinstance(k, slice):
                start = k.start if k.start is not None else 0
                stop = k.stop - 1 if k.stop is not None else self.dimensions[i]
                step = k.step if k.step is not None else 1
            voi.extend((start, stop))
            rate.append(step)

        return self.extract_subset(voi, rate, boundary=False)

    def hide_cells(self, ind, inplace=False):
        """Hide cells without deleting them.

        Hides cells by setting the ghost_cells array to ``HIDDEN_CELL``.

        Parameters
        ----------
        ind : sequence
            List or array of cell indices to be hidden.  The array can
            also be a boolean array of the same size as the number of
            cells.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        pyvista.PointSet
            Point set with hidden cells.

        Examples
        --------
        Hide part of the middle of a structured surface.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> x = np.arange(-10, 10, 0.25)
        >>> y = np.arange(-10, 10, 0.25)
        >>> z = 0
        >>> x, y, z = np.meshgrid(x, y, z)
        >>> grid = pv.StructuredGrid(x, y, z)
        >>> grid = grid.hide_cells(range(79*30, 79*50))
        >>> grid.plot(color=True, show_edges=True)
        """
        if not inplace:
            return self.copy().hide_cells(ind, inplace=True)
        if isinstance(ind, np.ndarray):
            if ind.dtype == np.bool_ and ind.size != self.n_cells:
                raise ValueError('Boolean array size must match the '
                                 f'number of cells ({self.n_cells})')
        ghost_cells = np.zeros(self.n_cells, np.uint8)
        ghost_cells[ind] = _vtk.vtkDataSetAttributes.HIDDENCELL

        # NOTE: cells cannot be removed from a structured grid, only
        # hidden setting ghost_cells to a value besides
        # vtk.vtkDataSetAttributes.HIDDENCELL will not hide them
        # properly, additionally, calling self.RemoveGhostCells will
        # have no effect

        # add but do not make active
        self.cell_data.set_array(ghost_cells, _vtk.vtkDataSetAttributes.GhostArrayName())
        return self

    def hide_points(self, ind):
        """Hide points without deleting them.

        Hides points by setting the ghost_points array to ``HIDDEN_CELL``.

        Parameters
        ----------
        ind : sequence
            List or array of point indices to be hidden.  The array
            can also be a boolean array of the same size as the number
            of points.

        Returns
        -------
        pyvista.PointSet
            Point set with hidden points.

        Examples
        --------
        Hide part of the middle of a structured surface.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> x = np.arange(-10, 10, 0.25)
        >>> y = np.arange(-10, 10, 0.25)
        >>> z = 0
        >>> x, y, z = np.meshgrid(x, y, z)
        >>> grid = pv.StructuredGrid(x, y, z)
        >>> grid.hide_points(range(80*30, 80*50))
        >>> grid.plot(color=True, show_edges=True)
        """
        if isinstance(ind, np.ndarray):
            if ind.dtype == np.bool_ and ind.size != self.n_points:
                raise ValueError('Boolean array size must match the '
                                 f'number of points ({self.n_points})')
        ghost_points = np.zeros(self.n_points, np.uint8)
        ghost_points[ind] = _vtk.vtkDataSetAttributes.HIDDENPOINT

        # add but do not make active
        self.point_data.set_array(ghost_points, _vtk.vtkDataSetAttributes.GhostArrayName())

    def _reshape_point_array(self, array):
        """Reshape point data to a 3-D matrix."""
        return array.reshape(self.dimensions, order='F')

    def _reshape_cell_array(self, array):
        """Reshape cell data to a 3-D matrix."""
        cell_dims = np.array(self.dimensions) - 1
        cell_dims[cell_dims == 0] = 1
        return array.reshape(cell_dims, order='F')


class ExplicitStructuredGrid(_vtk.vtkExplicitStructuredGrid, PointGrid):
    """Extend the functionality of the ``vtk.vtkExplicitStructuredGrid`` class.

    Can be initialized by the following:

    - Creating an empty grid
    - From a ``vtk.vtkExplicitStructuredGrid`` or ``vtk.vtkUnstructuredGrid`` object
    - From a VTU or VTK file
    - From ``dims`` and ``corners`` arrays

    Examples
    --------
    >>> import numpy as np
    >>> import pyvista as pv
    >>>
    >>> # grid size: ni*nj*nk cells; si, sj, sk steps
    >>> ni, nj, nk = 4, 5, 6
    >>> si, sj, sk = 20, 10, 1
    >>>
    >>> # create raw coordinate grid
    >>> grid_ijk = np.mgrid[:(ni+1)*si:si, :(nj+1)*sj:sj, :(nk+1)*sk:sk]
    >>>
    >>> # repeat array along each Cartesian axis for connectivity
    >>> for axis in range(1, 4):
    ...     grid_ijk = grid_ijk.repeat(2, axis=axis)
    >>>
    >>> # slice off unnecessarily doubled edge coordinates
    >>> grid_ijk = grid_ijk[:, 1:-1, 1:-1, 1:-1]
    >>>
    >>> # reorder and reshape to VTK order
    >>> corners = grid_ijk.transpose().reshape(-1, 3)
    >>>
    >>> dims = np.array([ni, nj, nk]) + 1
    >>> grid = pv.ExplicitStructuredGrid(dims, corners)
    >>> grid = grid.compute_connectivity()
    >>> grid.plot(show_edges=True)  # doctest:+SKIP

    """

    _WRITERS = {'.vtu': _vtk.vtkXMLUnstructuredGridWriter,
                '.vtk': _vtk.vtkUnstructuredGridWriter}

    def __init__(self, *args, **kwargs):
        """Initialize the explicit structured grid."""
        if not _vtk.VTK9:
            raise AttributeError('VTK 9 or higher is required')
        super().__init__()
        n = len(args)
        if n == 1:
            arg0 = args[0]
            if isinstance(arg0, _vtk.vtkExplicitStructuredGrid):
                self.deep_copy(arg0)
            elif isinstance(arg0, _vtk.vtkUnstructuredGrid):
                grid = arg0.cast_to_explicit_structured_grid()
                self.deep_copy(grid)
            elif isinstance(arg0, (str, pathlib.Path)):
                grid = UnstructuredGrid(arg0)
                grid = grid.cast_to_explicit_structured_grid()
                self.deep_copy(grid)
        elif n == 2:
            arg0, arg1 = args
            if isinstance(arg0, tuple):
                arg0 = np.asarray(arg0)
            if isinstance(arg1, list):
                arg1 = np.asarray(arg1)
            arg0_is_arr = isinstance(arg0, np.ndarray)
            arg1_is_arr = isinstance(arg1, np.ndarray)
            if all([arg0_is_arr, arg1_is_arr]):
                self._from_arrays(arg0, arg1)

    def __repr__(self):
        """Return the standard representation."""
        return DataSet.__repr__(self)

    def __str__(self):
        """Return the standard ``str`` representation."""
        return DataSet.__str__(self)

    def _from_arrays(self, dims: Sequence, corners: Sequence) -> None:
        """Create a VTK explicit structured grid from NumPy arrays.

        Parameters
        ----------
        dims : Sequence
            A sequence of integers with shape (3,) containing the
            topological dimensions of the grid.

        corners : Sequence
            A sequence of floats with shape (number of corners, 3)
            containing the coordinates of the corner points.

        """
        shape0 = np.asanyarray(dims) - 1
        shape1 = 2*shape0
        ncells = np.prod(shape0)
        cells = 8*np.ones((ncells, 9), dtype=int)
        points, indices = np.unique(corners, axis=0, return_inverse=True)
        connectivity = np.asarray([[0, 1, 1, 0, 0, 1, 1, 0],
                                   [0, 0, 1, 1, 0, 0, 1, 1],
                                   [0, 0, 0, 0, 1, 1, 1, 1]])
        for c in range(ncells):
            i, j, k = np.unravel_index(c, shape0, order='F')
            coord = (2*i + connectivity[0],
                     2*j + connectivity[1],
                     2*k + connectivity[2])
            cinds = np.ravel_multi_index(coord, shape1, order='F')
            cells[c, 1:] = indices[cinds]
        cells = cells.flatten()
        points = pyvista.vtk_points(points)
        cells = CellArray(cells, ncells)
        self.SetDimensions(dims)
        self.SetPoints(points)
        self.SetCells(cells)

    def cast_to_unstructured_grid(self) -> 'UnstructuredGrid':
        """Cast to an unstructured grid.

        Returns
        -------
        UnstructuredGrid
            An unstructured grid. VTK adds the ``'BLOCK_I'``,
            ``'BLOCK_J'`` and ``'BLOCK_K'`` cell arrays. These arrays
            are required to restore the explicit structured grid.

        See Also
        --------
        pyvista.DataSetFilters.extract_cells : Extract a subset of a dataset.
        pyvista.UnstructuredGrid.cast_to_explicit_structured_grid : Cast an unstructured grid to an explicit structured grid.

        Notes
        -----
        The ghost cell array is disabled before casting the
        unstructured grid in order to allow the original structure
        and attributes data of the explicit structured grid to be
        restored. If you don't need to restore the explicit
        structured grid later or want to extract an unstructured
        grid from the visible subgrid, use the ``extract_cells``
        filter and the cell indices where the ghost cell array is
        ``0``.

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()  # doctest:+SKIP
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)  # doctest:+SKIP

        >>> grid = grid.hide_cells(range(80, 120))  # doctest:+SKIP
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)  # doctest:+SKIP

        >>> grid = grid.cast_to_unstructured_grid()  # doctest:+SKIP
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)  # doctest:+SKIP

        >>> grid = grid.cast_to_explicit_structured_grid()  # doctest:+SKIP
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)  # doctest:+SKIP

        """
        grid = ExplicitStructuredGrid()
        grid.copy_structure(self)
        alg = _vtk.vtkExplicitStructuredGridToUnstructuredGrid()
        alg.SetInputDataObject(grid)
        alg.Update()
        grid = _get_output(alg)
        grid.cell_data.remove('vtkOriginalCellIds')  # unrequired
        grid.copy_attributes(self)  # copy ghost cell array and other arrays
        return grid

    def save(self, filename, binary=True):
        """Save this VTK object to file.

        Parameters
        ----------
        filename : str
            Output file name. VTU and VTK extensions are supported.
        binary : bool, optional
            If ``True`` (default), write as binary, else ASCII.

        Notes
        -----
        VTK adds the ``'BLOCK_I'``, ``'BLOCK_J'`` and ``'BLOCK_K'``
        cell arrays. These arrays are required to restore the explicit
        structured grid.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()  # doctest:+SKIP
        >>> grid = grid.hide_cells(range(80, 120))  # doctest:+SKIP
        >>> grid.save('grid.vtu')  # doctest:+SKIP

        >>> grid = pv.ExplicitStructuredGrid('grid.vtu')  # doctest:+SKIP
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)  # doctest:+SKIP

        >>> grid.show_cells()  # doctest:+SKIP
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)  # doctest:+SKIP

        """
        grid = self.cast_to_unstructured_grid()
        grid.save(filename, binary)

    def hide_cells(self, ind: Sequence[int], inplace=False) -> 'ExplicitStructuredGrid':
        """Hide specific cells.

        Hides cells by setting the ghost cell array to ``HIDDENCELL``.

        Parameters
        ----------
        ind : sequence(int)
            Cell indices to be hidden. A boolean array of the same
            size as the number of cells also is acceptable.

        inplace : bool, optional
            This method is applied to this grid if ``True`` (default)
            or to a copy otherwise.

        Returns
        -------
        ExplicitStructuredGrid or None
            A deep copy of this grid if ``inplace=False`` with the
            hidden cells, or this grid with the hidden cells if
            otherwise.

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()
        >>> grid = grid.hide_cells(range(80, 120))
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        """
        ind_arr = np.asanyarray(ind)

        if inplace:
            array = np.zeros(self.n_cells, dtype=np.uint8)
            array[ind_arr] = _vtk.vtkDataSetAttributes.HIDDENCELL
            name = _vtk.vtkDataSetAttributes.GhostArrayName()
            self.cell_data[name] = array
            return self

        grid = self.copy()
        grid.hide_cells(ind, inplace=True)
        return grid

    def show_cells(self, inplace=False) -> 'ExplicitStructuredGrid':
        """Show hidden cells.

        Shows hidden cells by setting the ghost cell array to ``0``
        where ``HIDDENCELL``.

        Parameters
        ----------
        inplace : bool, optional
            This method is applied to this grid if ``True`` (default)
            or to a copy otherwise.

        Returns
        -------
        ExplicitStructuredGrid
            A deep copy of this grid if ``inplace=False`` with the
            hidden cells shown.  Otherwise, this dataset with the
            shown cells.

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()
        >>> grid = grid.hide_cells(range(80, 120))
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        >>> grid = grid.show_cells()
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        """
        if inplace:
            name = _vtk.vtkDataSetAttributes.GhostArrayName()
            if name in self.cell_data.keys():
                array = self.cell_data[name]
                ind = np.argwhere(array == _vtk.vtkDataSetAttributes.HIDDENCELL)
                array[ind] = 0
            return self
        else:
            grid = self.copy()
            grid.show_cells(inplace=True)
            return grid

    def _dimensions(self):
        # This method is required to avoid conflict if a developer extends `ExplicitStructuredGrid`
        # and reimplements `dimensions` to return, for example, the number of cells in the I, J and
        # K directions.
        dims = self.extent
        dims = np.reshape(dims, (3, 2))
        dims = np.diff(dims, axis=1)
        dims = dims.flatten() + 1
        return int(dims[0]), int(dims[1]), int(dims[2])

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Return the topological dimensions of the grid.

        Returns
        -------
        tuple(int)
            Number of sampling points in the I, J and Z directions respectively.

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()  # doctest:+SKIP
        >>> grid.dimensions  # doctest:+SKIP
        (5, 6, 7)

        """
        return self._dimensions()

    @property
    def visible_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Return the bounding box of the visible cells.

        Different from `bounds`, which returns the bounding box of the
        complete grid, this method returns the bounding box of the
        visible cells, where the ghost cell array is not
        ``HIDDENCELL``.

        Returns
        -------
        tuple(float)
            The limits of the visible grid in the X, Y and Z
            directions respectively.

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()  # doctest:+SKIP
        >>> grid = grid.hide_cells(range(80, 120))  # doctest:+SKIP
        >>> grid.bounds  # doctest:+SKIP
        [0.0, 80.0, 0.0, 50.0, 0.0, 6.0]

        >>> grid.visible_bounds  # doctest:+SKIP
        [0.0, 80.0, 0.0, 50.0, 0.0, 4.0]

        """
        name = _vtk.vtkDataSetAttributes.GhostArrayName()
        if name in self.cell_data:
            array = self.cell_data[name]
            grid = self.extract_cells(array == 0)
            return grid.bounds
        else:
            return self.bounds

    def cell_id(self, coords) -> Union[int, np.ndarray, None]:
        """Return the cell ID.

        Parameters
        ----------
        coords : tuple(int), list(tuple(int)) or numpy.ndarray
            Cell structured coordinates.

        Returns
        -------
        int, numpy.ndarray, or None
            Cell IDs. ``None`` if ``coords`` is outside the grid extent.

        See Also
        --------
        pyvista.ExplicitStructuredGrid.cell_coords : Return the cell structured coordinates.

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()  # doctest:+SKIP
        >>> grid.cell_id((3, 4, 0))  # doctest:+SKIP
        19

        >>> coords = [(3, 4, 0),
        ...           (3, 2, 1),
        ...           (1, 0, 2),
        ...           (2, 3, 2)]
        >>> grid.cell_id(coords)  # doctest:+SKIP
        array([19, 31, 41, 54])

        """
        # `vtk.vtkExplicitStructuredGrid.ComputeCellId` is not used
        # here because this method returns invalid cell IDs when
        # `coords` is outside the grid extent.
        if isinstance(coords, list):
            coords = np.asarray(coords)
        if isinstance(coords, np.ndarray) and coords.ndim == 2:
            ncol = coords.shape[1]
            coords = [coords[:, c] for c in range(ncol)]
            coords = tuple(coords)
        dims = self._dimensions()
        try:
            ind = np.ravel_multi_index(coords, np.array(dims) - 1, order='F')
        except ValueError:
            return None
        else:
            return ind

    def cell_coords(self, ind):
        """Return the cell structured coordinates.

        Parameters
        ----------
        ind : int or iterable(int)
            Cell IDs.

        Returns
        -------
        tuple(int), numpy.ndarray, or None
            Cell structured coordinates. ``None`` if ``ind`` is
            outside the grid extent.

        See Also
        --------
        pyvista.ExplicitStructuredGrid.cell_id : Return the cell ID.

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()  # doctest:+SKIP
        >>> grid.cell_coords(19)  # doctest:+SKIP
        (3, 4, 0)

        >>> grid.cell_coords((19, 31, 41, 54))  # doctest:+SKIP
        array([[3, 4, 0],
               [3, 2, 1],
               [1, 0, 2],
               [2, 3, 2]])

        """
        dims = self._dimensions()
        try:
            coords = np.unravel_index(ind, np.array(dims) - 1, order='F')
        except ValueError:
            return None
        else:
            if isinstance(coords[0], np.ndarray):
                coords = np.stack(coords, axis=1)
            return coords

    def neighbors(self, ind, rel='connectivity') -> list:
        """Return the indices of neighboring cells.

        Parameters
        ----------
        ind : int or iterable(int)
            Cell IDs.

        rel : str, optional
            Defines the neighborhood relationship. If
            ``'topological'``, returns the ``(i-1, j, k)``, ``(i+1, j,
            k)``, ``(i, j-1, k)``, ``(i, j+1, k)``, ``(i, j, k-1)``
            and ``(i, j, k+1)`` cells. If ``'connectivity'``
            (default), returns only the topological neighbors
            considering faces connectivity. If ``'geometric'``,
            returns the cells in the ``(i-1, j)``, ``(i+1, j)``,
            ``(i,j-1)`` and ``(i, j+1)`` vertical cell groups whose
            faces intersect.

        Returns
        -------
        list(int)
            Indices of neighboring cells.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()  # doctest:+SKIP
        >>> cell = grid.extract_cells(31)  # doctest:+SKIP
        >>> ind = grid.neighbors(31)  # doctest:+SKIP
        >>> neighbors = grid.extract_cells(ind)  # doctest:+SKIP
        >>>
        >>> plotter = pv.Plotter()
        >>> plotter.add_axes()  # doctest:+SKIP
        >>> plotter.add_mesh(cell, color='r', show_edges=True)  # doctest:+SKIP
        >>> plotter.add_mesh(neighbors, color='w', show_edges=True)  # doctest:+SKIP
        >>> plotter.show()  # doctest:+SKIP

        """
        def connectivity(ind):
            indices = []
            cell_coords = self.cell_coords(ind)
            cell_points = self.cell_points(ind)
            if cell_points.shape[0] == 8:
                faces = [[(-1, 0, 0), (0, 4, 7, 3), (1, 5, 6, 2)],
                         [(+1, 0, 0), (1, 2, 6, 5), (0, 3, 7, 4)],
                         [(0, -1, 0), (0, 1, 5, 4), (3, 2, 6, 7)],
                         [(0, +1, 0), (3, 7, 6, 2), (0, 4, 5, 1)],
                         [(0, 0, -1), (0, 3, 2, 1), (4, 7, 6, 5)],
                         [(0, 0, +1), (4, 5, 6, 7), (0, 1, 2, 3)]]
                for f in faces:
                    coords = np.sum([cell_coords, f[0]], axis=0)
                    ind = self.cell_id(coords)
                    if ind:
                        points = self.cell_points(ind)
                        if points.shape[0] == 8:
                            a1 = cell_points[f[1], :]
                            a2 = points[f[2], :]
                            if np.array_equal(a1, a2):
                                indices.append(ind)
            return indices

        def topological(ind):
            indices = []
            cell_coords = self.cell_coords(ind)
            cell_neighbors = [(-1, 0, 0), (1, 0, 0),
                              (0, -1, 0), (0, 1, 0),
                              (0, 0, -1), (0, 0, 1)]
            for n in cell_neighbors:
                coords = np.sum([cell_coords, n], axis=0)
                ind = self.cell_id(coords)
                if ind:
                    indices.append(ind)
            return indices

        def geometric(ind):
            indices = []
            cell_coords = self.cell_coords(ind)
            cell_points = self.cell_points(ind)
            if cell_points.shape[0] == 8:
                for k in [-1, 1]:
                    coords = np.sum([cell_coords, (0, 0, k)], axis=0)
                    ind = self.cell_id(coords)
                    if ind:
                        indices.append(ind)
                faces = [[(-1, 0, 0), (0, 4, 3, 7), (1, 5, 2, 6)],
                         [(+1, 0, 0), (2, 6, 1, 5), (3, 7, 0, 4)],
                         [(0, -1, 0), (1, 5, 0, 4), (2, 6, 3, 7)],
                         [(0, +1, 0), (3, 7, 2, 6), (0, 4, 1, 5)]]
                nk = self.dimensions[2]
                for f in faces:
                    cell_z = cell_points[f[1], 2]
                    cell_z = np.abs(cell_z)
                    cell_z = cell_z.reshape((2, 2))
                    cell_zmin = cell_z.min(axis=1)
                    cell_zmax = cell_z.max(axis=1)
                    coords = np.sum([cell_coords, f[0]], axis=0)
                    for k in range(nk):
                        coords[2] = k
                        ind = self.cell_id(coords)
                        if ind:
                            points = self.cell_points(ind)
                            if points.shape[0] == 8:
                                z = points[f[2], 2]
                                z = np.abs(z)
                                z = z.reshape((2, 2))
                                zmin = z.min(axis=1)
                                zmax = z.max(axis=1)
                                if ((zmax[0] > cell_zmin[0] and zmin[0] < cell_zmax[0]) or
                                    (zmax[1] > cell_zmin[1] and zmin[1] < cell_zmax[1]) or
                                    (zmin[0] > cell_zmax[0] and zmax[1] < cell_zmin[1]) or
                                    (zmin[1] > cell_zmax[1] and zmax[0] < cell_zmin[0])):
                                    indices.append(ind)
            return indices

        if isinstance(ind, int):
            ind = [ind]
        rel = eval(rel)
        indices = set()
        for i in ind:
            indices.update(rel(i))
        return sorted(indices)

    def compute_connectivity(self, inplace=False) -> 'ExplicitStructuredGrid':
        """Compute the faces connectivity flags array.

        This method checks the faces connectivity of the cells with
        their topological neighbors.  The result is stored in the
        array of integers ``'ConnectivityFlags'``. Each value in this
        array must be interpreted as a binary number, where the digits
        shows the faces connectivity of a cell with its topological
        neighbors -Z, +Z, -Y, +Y, -X and +X respectively. For example,
        a cell with ``'ConnectivityFlags'`` equal to ``27``
        (``011011``) indicates that this cell is connected by faces
        with their neighbors ``(0, 0, 1)``, ``(0, -1, 0)``,
        ``(-1, 0, 0)`` and ``(1, 0, 0)``.

        Parameters
        ----------
        inplace : bool, optional
            This method is applied to this grid if ``True`` (default)
            or to a copy otherwise.

        Returns
        -------
        ExplicitStructuredGrid
            A deep copy of this grid if ``inplace=False``, or this
            DataSet if otherwise.

        See Also
        --------
        ExplicitStructuredGrid.compute_connections : Compute an array with the number of connected cell faces.

        Examples
        --------
        >>> from pyvista import examples
        >>>
        >>> grid = examples.load_explicit_structured()  # doctest:+SKIP
        >>> grid = grid.compute_connectivity()  # doctest:+SKIP
        >>> grid.plot(show_edges=True)  # doctest:+SKIP

        """
        if inplace:
            self.ComputeFacesConnectivityFlagsArray()
            return self
        else:
            grid = self.copy()
            grid.compute_connectivity(inplace=True)
            return grid

    def compute_connections(self, inplace=False):
        """Compute an array with the number of connected cell faces.

        This method calculates the number of topological cell
        neighbors connected by faces. The results are stored in the
        ``'number_of_connections'`` cell array.

        Parameters
        ----------
        inplace : bool, optional
            This method is applied to this grid if ``True`` or to a copy
            otherwise.

        Returns
        -------
        ExplicitStructuredGrid
            A deep copy of this grid if ``inplace=False`` or this
            DataSet if otherwise.

        See Also
        --------
        ExplicitStructuredGrid.compute_connectivity : Compute the faces connectivity flags array.

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()  # doctest:+SKIP
        >>> grid = grid.compute_connections()  # doctest:+SKIP
        >>> grid.plot(show_edges=True)  # doctest:+SKIP

        """
        if inplace:
            if 'ConnectivityFlags' in self.cell_data:
                array = self.cell_data['ConnectivityFlags']
            else:
                grid = self.compute_connectivity(inplace=False)
                array = grid.cell_data['ConnectivityFlags']
            array = array.reshape((-1, 1))
            array = array.astype(np.uint8)
            array = np.unpackbits(array, axis=1)
            array = array.sum(axis=1)
            self.cell_data['number_of_connections'] = array
            return self
        else:
            return self.copy().compute_connections(inplace=True)
