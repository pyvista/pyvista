"""Core helper utilities."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING
from typing import Sequence
from typing import cast

if TYPE_CHECKING:  # pragma: no cover
    from meshio import Mesh
    from trimesh import Trimesh

    from pyvista.core._typing_core import NumpyArray

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk

from . import transformations
from .fileio import from_meshio
from .fileio import is_meshio_mesh


def wrap(
    dataset: NumpyArray[float] | _vtk.vtkDataSet | Trimesh | Mesh | None,
) -> pyvista.DataSet | pyvista.pyvista_ndarray | None:
    """Wrap any given VTK data object to its appropriate PyVista data object.

    Other formats that are supported include:

    * 2D :class:`numpy.ndarray` of XYZ vertices
    * 3D :class:`numpy.ndarray` representing a volume. Values will be scalars.
    * 3D :class:`trimesh.Trimesh` mesh.
    * 3D :class:`meshio.Mesh` mesh.

    .. versionchanged:: 0.38.0
        If the passed object is already a wrapped PyVista object, then
        this is no-op and will return that object directly. In previous
        versions of PyVista, this would perform a shallow copy.

    Parameters
    ----------
    dataset : :class:`numpy.ndarray` | :class:`trimesh.Trimesh` | vtk.DataSet
        Dataset to wrap.

    Returns
    -------
    pyvista.DataSet
        The PyVista wrapped dataset.

    Examples
    --------
    Wrap a numpy array representing a random point cloud.

    >>> import numpy as np
    >>> import pyvista as pv
    >>> points = np.random.default_rng().random((10, 3))
    >>> cloud = pv.wrap(points)
    >>> cloud
    PolyData (...)
      N Cells:    10
      N Points:   10
      N Strips:   0
      X Bounds:   ...
      Y Bounds:   ...
      Z Bounds:   ...
      N Arrays:   0

    Wrap a VTK object.

    >>> import pyvista as pv
    >>> import vtk
    >>> points = vtk.vtkPoints()
    >>> p = [1.0, 2.0, 3.0]
    >>> vertices = vtk.vtkCellArray()
    >>> pid = points.InsertNextPoint(p)
    >>> _ = vertices.InsertNextCell(1)
    >>> _ = vertices.InsertCellPoint(pid)
    >>> point = vtk.vtkPolyData()
    >>> _ = point.SetPoints(points)
    >>> _ = point.SetVerts(vertices)
    >>> mesh = pv.wrap(point)
    >>> mesh
    PolyData (...)
      N Cells:    1
      N Points:   1
      N Strips:   0
      X Bounds:   1.000e+00, 1.000e+00
      Y Bounds:   2.000e+00, 2.000e+00
      Z Bounds:   3.000e+00, 3.000e+00
      N Arrays:   0

    Wrap a Trimesh object.

    >>> import trimesh
    >>> import pyvista as pv
    >>> points = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
    >>> faces = [[0, 1, 2]]
    >>> tmesh = trimesh.Trimesh(points, faces=faces, process=False)
    >>> mesh = pv.wrap(tmesh)
    >>> mesh  # doctest:+SKIP
    PolyData (0x7fc55ff27ad0)
      N Cells:  1
      N Points: 3
      X Bounds: 0.000e+00, 0.000e+00
      Y Bounds: 0.000e+00, 1.000e+00
      Z Bounds: 0.000e+00, 1.000e+00
      N Arrays: 0

    """
    # Return if None
    if dataset is None:
        return None

    if isinstance(dataset, tuple(pyvista._wrappers.values())):
        # Return object if it is already wrapped
        return dataset  # type: ignore[return-value]

    # Check if dataset is a numpy array.  We do this first since
    # pyvista_ndarray contains a VTK type that we don't want to
    # directly wrap.
    if isinstance(dataset, (np.ndarray, pyvista.pyvista_ndarray)):
        if dataset.ndim == 1 and dataset.shape[0] == 3:
            return pyvista.PolyData(dataset)
        if dataset.ndim > 1 and dataset.ndim < 3 and dataset.shape[1] == 3:
            return pyvista.PolyData(dataset)
        elif dataset.ndim == 3:
            mesh = pyvista.ImageData(dimensions=dataset.shape)
            if isinstance(dataset, pyvista.pyvista_ndarray):
                # this gets rid of pesky VTK reference since we're raveling this
                dataset = np.asarray(dataset)
            mesh['values'] = dataset.ravel(order='F')
            mesh.active_scalars_name = 'values'
            return mesh
        else:
            raise NotImplementedError('NumPy array could not be wrapped pyvista.')

    # wrap VTK arrays as pyvista_ndarray
    if isinstance(dataset, _vtk.vtkDataArray):
        return pyvista.pyvista_ndarray(dataset)

    # Check if a dataset is a VTK type
    if hasattr(dataset, 'GetClassName'):
        key = dataset.GetClassName()
        try:
            return pyvista._wrappers[key](dataset)
        except KeyError:
            raise TypeError(f'VTK data type ({key}) is not currently supported by pyvista.')
        return None  # pragma: no cover

    # wrap meshio
    if is_meshio_mesh(dataset):
        return from_meshio(dataset)

    # wrap trimesh
    if dataset.__class__.__name__ == 'Trimesh':
        # trimesh doesn't pad faces
        dataset = cast('Trimesh', dataset)
        polydata = pyvista.PolyData.from_regular_faces(
            np.asarray(dataset.vertices),
            faces=dataset.faces,
        )
        # If the Trimesh object has uv, pass them to the PolyData
        if hasattr(dataset.visual, 'uv'):
            polydata.active_texture_coordinates = np.asarray(dataset.visual.uv)
        return polydata

    # otherwise, flag tell the user we can't wrap this object
    raise NotImplementedError(f'Unable to wrap ({type(dataset)}) into a pyvista type.')


def is_pyvista_dataset(obj):
    """Return ``True`` if the object is a PyVista wrapped dataset.

    Parameters
    ----------
    obj : Any
        Any object to test.

    Returns
    -------
    bool
        ``True`` when the object is a :class:`pyvista.DataSet`.

    """
    return isinstance(obj, (pyvista.DataSet, pyvista.MultiBlock))


def generate_plane(normal, origin):
    """Return a _vtk.vtkPlane.

    Parameters
    ----------
    normal : sequence[float]
        Three item sequence representing the normal of the plane.

    origin : sequence[float]
        Three item sequence representing the origin of the plane.

    Returns
    -------
    vtk.vtkPlane
        VTK plane.

    """
    plane = _vtk.vtkPlane()
    # NORMAL MUST HAVE MAGNITUDE OF 1
    normal = normal / np.linalg.norm(normal)
    plane.SetNormal(normal)
    plane.SetOrigin(origin)
    return plane


def axis_rotation(points, angle, inplace=False, deg=True, axis='z'):
    """Rotate points by angle about an axis.

    Parameters
    ----------
    points : numpy.ndarray
        Array of points with shape ``(N, 3)``.

    angle : float
        Rotation angle.

    inplace : bool, default: False
        Updates points in-place while returning nothing.

    deg : bool, default: True
        If ``True``, the angle is interpreted as degrees instead of
        radians.

    axis : str, default: "z"
        Name of axis to rotate about. Valid options are ``'x'``, ``'y'``,
        and ``'z'``.

    Returns
    -------
    numpy.ndarray
        Rotated points.

    Examples
    --------
    Rotate a set of points by 90 degrees about the x-axis in-place.

    >>> import numpy as np
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> points = examples.load_airplane().points
    >>> points_orig = points.copy()
    >>> pv.axis_rotation(points, 90, axis='x', deg=True, inplace=True)
    >>> assert np.all(np.isclose(points[:, 0], points_orig[:, 0]))
    >>> assert np.all(np.isclose(points[:, 1], -points_orig[:, 2]))
    >>> assert np.all(np.isclose(points[:, 2], points_orig[:, 1]))
    """
    axis = axis.lower()
    axis_to_vec = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}

    if axis not in axis_to_vec:
        raise ValueError('Invalid axis. Must be either "x", "y", or "z"')

    rot_mat = transformations.axis_angle_rotation(axis_to_vec[axis], angle, deg=deg)
    return transformations.apply_transformation_to_points(rot_mat, points, inplace=inplace)


def is_inside_bounds(point, bounds):
    """Check if a point is inside a set of bounds.

    This is implemented through recursion so that this is N-dimensional.

    Parameters
    ----------
    point : sequence[float]
        Three item cartesian point (i.e. ``[x, y, z]``).

    bounds : sequence[float]
        Six item bounds in the form of ``(xMin, xMax, yMin, yMax, zMin, zMax)``.

    Returns
    -------
    bool
        ``True`` when ``point`` is inside ``bounds``.

    """
    if isinstance(point, (int, float)):
        point = [point]
    if isinstance(point, (np.ndarray, Sequence)) and not isinstance(
        point,
        deque,
    ):
        if len(bounds) < 2 * len(point) or len(bounds) % 2 != 0:
            raise ValueError('Bounds mismatch point dimensionality')
        point = deque(point)
        bounds = deque(bounds)
        return is_inside_bounds(point, bounds)
    if not isinstance(point, deque):
        raise TypeError(f'Unknown input data type ({type(point)}).')
    if len(point) < 1:
        return True
    p = point.popleft()
    lower, upper = bounds.popleft(), bounds.popleft()
    if lower <= p <= upper:
        return is_inside_bounds(point, bounds)
    return False
