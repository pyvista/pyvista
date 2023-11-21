"""Prop3D module."""
from typing import Tuple, Union

import numpy as np

from pyvista.core._typing_core import BoundsLike, Vector
from pyvista.core.utilities.arrays import array_from_vtkmatrix, vtkmatrix_from_array

from . import _vtk


class Prop3D(_vtk.vtkProp3D):
    """Prop3D wrapper for vtkProp3D.

    Used to represent an entity in a rendering scene. It handles functions
    related to the position, orientation and scaling. Used as parent class
    in Actor and Volume class.
    """

    def __init__(self):
        """Initialize Prop3D."""
        super().__init__()

    @property
    def scale(self) -> Tuple[float, float, float]:  # numpydoc ignore=RT01
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
    def scale(self, value: Vector):  # numpydoc ignore=GL08
        return self.SetScale(value)

    @property
    def position(self) -> Tuple[float, float, float]:  # numpydoc ignore=RT01
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
    def position(self, value: Vector):  # numpydoc ignore=GL08
        self.SetPosition(value)

    def rotate_x(self, angle: float):
        """Rotate the entity about the x axis.

        Parameters
        ----------
        angle : float
            Angle to rotate the entity about the x axis in degrees.

        Examples
        --------
        Rotate the actor about the x axis 45 degrees. Note how this does not
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
        """Rotate the entity about the y axis.

        Parameters
        ----------
        angle : float
            Angle to rotate the entity about the y axis in degrees.

        Examples
        --------
        Rotate the actor about the y axis 45 degrees. Note how this does not
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
        """Rotate the entity about the z axis.

        Parameters
        ----------
        angle : float
            Angle to rotate the entity about the z axis in degrees.

        Examples
        --------
        Rotate the actor about the Z axis 45 degrees. Note how this does not
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
    def orientation(self) -> tuple:  # numpydoc ignore=RT01
        """Return or set the entity orientation.

        Orientation is defined as the rotation from the global axes in degrees
        about the actor's x, y, and z axes.

        Examples
        --------
        Reorient just the actor and plot it. Note how the actor is rotated
        about its own axes as defined by its position.

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
        >>> actor.position = (0, 0, 1)
        >>> actor.orientation = (45, 0, 0)
        >>> pl.show_axes()
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
    def orientation(self, value: tuple):  # numpydoc ignore=GL08
        self.SetOrientation(value)

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
    def center(self) -> tuple:  # numpydoc ignore=RT01
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
    def user_matrix(self) -> np.ndarray:  # numpydoc ignore=RT01
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
        rotates the actor about the Z axis by approximately 45 degrees, and
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
    def user_matrix(self, value: Union[_vtk.vtkMatrix4x4, np.ndarray]):  # numpydoc ignore=GL08
        if isinstance(value, np.ndarray):
            if value.shape != (4, 4):
                raise ValueError('User matrix array must be 4x4.')
            value = vtkmatrix_from_array(value)

        if isinstance(value, _vtk.vtkMatrix4x4):
            self.SetUserMatrix(value)
        else:
            raise TypeError(
                'Input user matrix must be either:\n' '\tvtk.vtkMatrix4x4\n' '\t4x4 np.ndarray\n'
            )
