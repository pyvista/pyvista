"""Prop3D module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyvista.core import _validation
from pyvista.core.utilities.arrays import _coerce_transformlike_arg
from pyvista.core.utilities.arrays import array_from_vtkmatrix
from pyvista.core.utilities.arrays import vtkmatrix_from_array
from pyvista.plotting import _vtk

if TYPE_CHECKING:  # pragma: no cover
    from pyvista.core._typing_core import BoundsLike
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import TransformLike
    from pyvista.core._typing_core import VectorLike


class Prop3D(_vtk.vtkProp3D):
    """Prop3D wrapper for vtkProp3D.

    Used to represent an entity in a rendering scene. It provides spatial
    properties and methods relating to an entity's position, orientation
    and scale. It is used as parent class for :class:`pyvista.Actor`,
    :class:`pyvista.AxesActor`, and :class:`pyvista.plotting.volume.Volume`.

    ``Prop3D`` applies transformations in the following order:

        #. Translate entity to its :attr:`~origin`.
        #. Scale entity by the values in :attr:`~scale`.
        #. Rotate entity using the values in :attr:`~orientation`. Internally, rotations are
           applied in the order :func:`~rotate_y`, then :func:`~rotate_x`, then :func:`~rotate_z`.
        #. Translate entity away from its origin and to its :attr:`~position`.
        #. Transform entity with :attr:`~user_matrix`.

    """

    def __init__(self):
        """Initialize Prop3D."""
        super().__init__()

    @property
    def scale(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set entity scale.

        Examples
        --------
        Create an actor using the :class:`pyvista.Plotter` and then change the
        scale of the actor.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> actor.scale = (2.0, 2.0, 2.0)
        >>> actor.scale
        (2.0, 2.0, 2.0)

        """
        return self.GetScale()

    @scale.setter
    def scale(self, value: VectorLike[float]):  # numpydoc ignore=GL08
        self.SetScale(value)

    @property
    def position(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the entity position.

        Examples
        --------
        Change the position of an actor. Note how this does not change the
        position of the underlying dataset, just the relative location of the
        actor in the :class:`pyvista.Plotter`.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color='b')
        >>> actor = pl.add_mesh(mesh, color='r')
        >>> actor.position = (0, 0, 1)  # shifts the red sphere up
        >>> pl.show()

        """
        return self.GetPosition()

    @position.setter
    def position(self, value: VectorLike[float]):  # numpydoc ignore=GL08
        self.SetPosition(value)

    def rotate_x(self, angle: float):
        """Rotate the entity about the x-axis.

        Parameters
        ----------
        angle : float
            Angle to rotate the entity about the x-axis in degrees.

        Examples
        --------
        Rotate the actor about the x-axis 45 degrees. Note how this does not
        change the location of the underlying dataset.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color='b')
        >>> actor = pl.add_mesh(
        ...     mesh,
        ...     color='r',
        ...     style='wireframe',
        ...     line_width=5,
        ...     lighting=False,
        ... )
        >>> actor.rotate_x(45)
        >>> pl.show_axes()
        >>> pl.show()

        """
        self.RotateX(angle)

    def rotate_y(self, angle: float):
        """Rotate the entity about the y-axis.

        Parameters
        ----------
        angle : float
            Angle to rotate the entity about the y-axis in degrees.

        Examples
        --------
        Rotate the actor about the y-axis 45 degrees. Note how this does not
        change the location of the underlying dataset.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color='b')
        >>> actor = pl.add_mesh(
        ...     mesh,
        ...     color='r',
        ...     style='wireframe',
        ...     line_width=5,
        ...     lighting=False,
        ... )
        >>> actor.rotate_y(45)
        >>> pl.show_axes()
        >>> pl.show()

        """
        self.RotateY(angle)

    def rotate_z(self, angle: float):
        """Rotate the entity about the z-axis.

        Parameters
        ----------
        angle : float
            Angle to rotate the entity about the z-axis in degrees.

        Examples
        --------
        Rotate the actor about the z-axis 45 degrees. Note how this does not
        change the location of the underlying dataset.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color='b')
        >>> actor = pl.add_mesh(
        ...     mesh,
        ...     color='r',
        ...     style='wireframe',
        ...     line_width=5,
        ...     lighting=False,
        ... )
        >>> actor.rotate_z(45)
        >>> pl.show_axes()
        >>> pl.show()

        """
        self.RotateZ(angle)

    @property
    def orientation(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the entity orientation angles.

        Orientation angles of the axes which define rotations about the
        world's x-y-z axes. The angles are specified in degrees and in
        x-y-z order. However, the actual rotations are applied in the
        following order: :func:`~rotate_y` first, then :func:`~rotate_x`
        and finally :func:`~rotate_z`.

        Rotations are applied about the specified :attr:`~origin`.

        Examples
        --------
        Reorient just the actor and plot it. Note how the actor is rotated
        about the origin ``(0, 0, 0)`` by default.

        >>> import pyvista as pv
        >>> mesh = pv.Cube(center=(0, 0, 3))
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color='b')
        >>> actor = pl.add_mesh(
        ...     mesh,
        ...     color='r',
        ...     style='wireframe',
        ...     line_width=5,
        ...     lighting=False,
        ... )
        >>> actor.orientation = (45, 0, 0)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        Repeat the last example, but this time reorient the actor about
        its center by specifying its :attr:`~origin`.

        >>> import pyvista as pv
        >>> mesh = pv.Cube(center=(0, 0, 3))
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color='b')
        >>> actor = pl.add_mesh(
        ...     mesh,
        ...     color='r',
        ...     style='wireframe',
        ...     line_width=5,
        ...     lighting=False,
        ... )
        >>> actor.origin = actor.center
        >>> actor.orientation = (45, 0, 0)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        Show that the orientation changes with rotation.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(mesh)
        >>> actor.rotate_x(90)
        >>> actor.orientation  # doctest:+SKIP
        (90, 0, 0)

        Set the orientation directly.

        >>> actor.orientation = (0, 45, 45)
        >>> actor.orientation  # doctest:+SKIP
        (0, 45, 45)

        """
        return self.GetOrientation()

    @orientation.setter
    def orientation(self, value: tuple[float, float, float]):  # numpydoc ignore=GL08
        self.SetOrientation(value)

    @property
    def origin(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the entity origin.

        This is the point about which all rotations take place.

        See :attr:`~orientation` for examples.

        """
        return self.GetOrigin()

    @origin.setter
    def origin(self, value: VectorLike[float]):  # numpydoc ignore=GL08
        self.SetOrigin(value)

    @property
    def bounds(self) -> BoundsLike:  # numpydoc ignore=RT01
        """Return the bounds of the entity.

        Bounds are ``(-X, +X, -Y, +Y, -Z, +Z)``

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> mesh = pv.Cube(x_length=0.1, y_length=0.2, z_length=0.3)
        >>> actor = pl.add_mesh(mesh)
        >>> actor.bounds
        (-0.05, 0.05, -0.1, 0.1, -0.15, 0.15)

        """
        return self.GetBounds()

    @property
    def center(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return the center of the entity.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere(center=(0.5, 0.5, 1)))
        >>> actor.center  # doctest:+SKIP
        (0.5, 0.5, 1)
        """
        return self.GetCenter()

    @property
    def user_matrix(self) -> NumpyArray[float]:  # numpydoc ignore=RT01
        """Return or set the user matrix.

        In addition to the instance variables such as position and orientation, the user
        can add an additional transformation to the actor.

        This matrix is concatenated with the actor's internal transformation that is
        implicitly created when the actor is created. This affects the actor/rendering
        only, not the input data itself.

        The user matrix is the last transformation applied to the actor before
        rendering.

        Returns
        -------
        np.ndarray
            A 4x4 transformation matrix.

        Examples
        --------
        Apply a 4x4 translation to a wireframe actor. This 4x4 transformation
        effectively translates the actor by one unit in the Z direction,
        rotates the actor about the z-axis by approximately 45 degrees, and
        shrinks the actor by a factor of 0.5.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color="b")
        >>> actor = pl.add_mesh(
        ...     mesh,
        ...     color="r",
        ...     style="wireframe",
        ...     line_width=5,
        ...     lighting=False,
        ... )
        >>> arr = np.array(
        ...     [
        ...         [0.707, -0.707, 0, 0],
        ...         [0.707, 0.707, 0, 0],
        ...         [0, 0, 1, 1.500001],
        ...         [0, 0, 0, 2],
        ...     ]
        ... )
        >>> actor.user_matrix = arr
        >>> pl.show_axes()
        >>> pl.show()

        """
        if self.GetUserMatrix() is None:
            self.SetUserMatrix(vtkmatrix_from_array(np.eye(4)))
        return array_from_vtkmatrix(self.GetUserMatrix())

    @user_matrix.setter
    def user_matrix(self, value: TransformLike):  # numpydoc ignore=GL08
        array = np.eye(4) if value is None else _coerce_transformlike_arg(value)
        self.SetUserMatrix(vtkmatrix_from_array(array))

    @property
    def length(self) -> float:  # numpydoc ignore=RT01
        """Return the length of the entity.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> actor.length
        1.7272069317100354
        """
        return self.GetLength()


def _rotation_matrix_as_orientation(
    array: NumpyArray[float] | _vtk.vtkMatrix3x3,
) -> tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to x-y-z orientation angles.

    The orientation angles define rotations about the world's x-y-z axes. The angles
    are specified in degrees and in x-y-z order. However, the rotations should
    be applied in the order: first rotate about the y-axis, then x-axis, then z-axis.

    The rotation angles and rotation matrix can be used interchangeably for
    transformations.

    Parameters
    ----------
    array : NumpyArray[float] | vtkMatrix3x3
        3x3 rotation matrix as a NumPy array or a vtkMatrix.

    Returns
    -------
    tuple
        Tuple with x-y-z axis rotation angles in degrees.

    """
    array_3x3 = _validation.validate_transform3x3(array)
    array_4x4 = np.eye(4)
    array_4x4[:3, :3] = array_3x3
    transform = _vtk.vtkTransform()
    transform.SetMatrix(array_4x4.ravel())
    return transform.GetOrientation()


def _orientation_as_rotation_matrix(orientation: VectorLike[float]) -> NumpyArray[float]:
    """Convert x-y-z orientation angles to a 3x3 matrix.

    The orientation angles define rotations about the world's x-y-z axes. The angles
    are specified in degrees and in x-y-z order. However, the rotations should
    be applied in the order: first rotate about the y-axis, then x-axis, then z-axis.

    The rotation angles and rotation matrix can be used interchangeably for
    transformations.

    Parameters
    ----------
    orientation : VectorLike[float]
        The x-y-z axis orientation angles in degrees.

    Returns
    -------
    numpy.ndarray
        3x3 rotation matrix.

    """
    valid_orientation = _validation.validate_array3(orientation, name='orientation')
    prop = _vtk.vtkActor()
    prop.SetOrientation(valid_orientation)
    matrix = _vtk.vtkMatrix4x4()
    prop.GetMatrix(matrix)
    return array_from_vtkmatrix(matrix)[:3, :3]
