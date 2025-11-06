"""Prop3D module."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from functools import wraps
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np

from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _validation
from pyvista.core._typing_core import BoundsTuple
from pyvista.core.utilities.arrays import array_from_vtkmatrix
from pyvista.core.utilities.arrays import vtkmatrix_from_array
from pyvista.core.utilities.misc import _BoundsSizeMixin
from pyvista.core.utilities.misc import _NameMixin
from pyvista.core.utilities.misc import _NoNewAttrMixin
from pyvista.core.utilities.transform import Transform
from pyvista.plotting import _vtk

if TYPE_CHECKING:
    from typing_extensions import Self

    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import RotationLike
    from pyvista.core._typing_core import TransformLike
    from pyvista.core._typing_core import VectorLike


class Prop3D(
    _NoNewAttrMixin, _NameMixin, _BoundsSizeMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkProp3D
):
    """Prop3D wrapper for :vtk:`vtkProp3D`.

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

    def __init__(self) -> None:
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
    def scale(self, value: float | VectorLike[float]) -> None:
        self.SetScale(value)  # type: ignore[arg-type]

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
    def position(self, value: VectorLike[float]) -> None:
        self.SetPosition(value)  # type: ignore[call-overload]

    def rotate_x(self, angle: float) -> None:
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

    def rotate_y(self, angle: float) -> None:
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

    def rotate_z(self, angle: float) -> None:
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

        See Also
        --------
        rotation_from
            Alternative method for setting the :attr:`orientation`.

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
    def orientation(self, value: VectorLike[float]) -> None:
        self.SetOrientation(value)  # type: ignore[call-overload]

    @property
    def origin(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the entity origin.

        This is the point about which all rotations take place.

        See :attr:`~orientation` for examples.

        """
        return self.GetOrigin()

    @origin.setter
    def origin(self, value: VectorLike[float]) -> None:
        self.SetOrigin(value)  # type: ignore[arg-type]

    @property
    def bounds(self) -> BoundsTuple:  # numpydoc ignore=RT01
        """Return the bounds of the entity.

        Bounds are ``(x_min, x_max, y_min, y_max, z_min, z_max)``

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> mesh = pv.Cube(x_length=0.1, y_length=0.2, z_length=0.3)
        >>> actor = pl.add_mesh(mesh)
        >>> actor.bounds
        BoundsTuple(x_min = -0.05,
                    x_max =  0.05,
                    y_min = -0.1,
                    y_max =  0.1,
                    z_min = -0.15,
                    z_max =  0.15)

        """
        return BoundsTuple(*self.GetBounds())

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

        See Also
        --------
        transform
            Apply a transformation to the :attr:`user_matrix`.

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
        >>> arr = [
        ...     [0.707, -0.707, 0, 0],
        ...     [0.707, 0.707, 0, 0],
        ...     [0, 0, 1, 1.5],
        ...     [0, 0, 0, 2],
        ... ]
        >>> actor.user_matrix = arr
        >>> pl.show_axes()
        >>> pl.show()

        """
        if self.GetUserMatrix() is None:
            self.SetUserMatrix(vtkmatrix_from_array(np.eye(4)))
        return array_from_vtkmatrix(self.GetUserMatrix())

    @user_matrix.setter
    def user_matrix(self, value: TransformLike) -> None:
        array = np.eye(4) if value is None else _validation.validate_transform4x4(value)
        self.SetUserMatrix(vtkmatrix_from_array(array))

    def transform(
        self,
        trans: TransformLike,
        multiply_mode: Literal['pre', 'post'] = 'post',
        *,
        inplace: bool = False,
    ):
        """Apply a transformation to this object's :attr:`user_matrix`.

        .. note::

            This applies a transformation by modifying the :attr:`user_matrix`. This
            differs from methods like :meth:`rotate_x`, :meth:`rotate_y`, :meth:`rotate_z`,
            and :meth:`rotation_from` which apply a transformation indirectly by modifying
            the :attr:`orientation`. See the :class:`Prop3D` class description for more
            information about how this class is transformed.

        .. versionadded:: 0.45

        Parameters
        ----------
        trans : TransformLike
            Transformation matrix as a 3x3 or 4x4 array, :vtk:`vtkMatrix3x3` or
            :vtk:`vtkMatrix4x4`, :vtk:`vtkTransform`, or a SciPy ``Rotation`` instance.
            If the input is 3x3, the array is padded using a 4x4 identity matrix.

        multiply_mode : 'pre' | 'post', default: 'post'
            Multiplication mode to use.

            - ``'pre'``: pre-multiply ``trans`` with the :attr:`user_matrix`, i.e.
              ``user_matrix @ trans``. The transformation is applied `before` the
              current user-matrix.
            - ``'post'``: post-multiply ``trans`` with the :attr:`user_matrix`, i.e.
              ``trans @ user_matrix``. The transformation is applied `after` the
              current user-matrix.

        inplace : bool, default: False
            When ``True``, modifies the prop inplace. Otherwise, a copy is returned.

        Returns
        -------
        Prop3D
            Transformed prop.

        See Also
        --------
        pyvista.Transform
            Describe linear transformations via a 4x4 matrix.
        pyvista.DataObjectFilters.transform
            Apply a transformation to a mesh.

        """
        # Validate input
        _validation.check_contains(
            ['pre', 'post'], must_contain=multiply_mode, name='multiply_mode'
        )
        matrix = _validation.validate_transform4x4(trans)

        # Update user matrix
        new_matrix = (
            self.user_matrix @ matrix if multiply_mode == 'pre' else matrix @ self.user_matrix
        )
        output = self if inplace else self.copy()
        output.user_matrix = new_matrix
        return output

    @abstractmethod
    @_deprecate_positional_args
    def copy(
        self: Self,
        deep: bool = True,  # noqa: FBT001, FBT002
    ) -> Self:  # numpydoc ignore=RT01
        """Return a copy of this prop."""
        raise NotImplementedError  # pragma: no cover

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

    def rotation_from(self, rotation: RotationLike) -> None:
        """Set the entity's orientation from a rotation.

        Set the rotation of this entity from a 3x3 rotation matrix. This includes
        NumPy arrays, a :vtk:`vtkMatrix3x3`, and SciPy ``Rotation`` objects.

        This method may be used as an alternative for setting the :attr:`orientation`.

        .. versionadded:: 0.45

        Parameters
        ----------
        rotation : RotationLike
            3x3 rotation matrix or a SciPy ``Rotation`` object.

        Examples
        --------
        Create an actor and show its initial orientation.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> actor.orientation
        (0.0, -0.0, 0.0)

        Set the orientation using a 3x3 matrix.

        >>> actor.rotation_from([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        >>> actor.orientation
        (0.0, -180.0, -89.99999999999999)

        """
        self.orientation = _rotation_matrix_as_orientation(rotation)  # type: ignore[arg-type]


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
    array : NumpyArray[float] | :vtk:`vtkMatrix3x3`
        3x3 rotation matrix as a NumPy array or a :vtk:`vtkMatrix3x3`.

    Returns
    -------
    tuple
        Tuple with x-y-z axis rotation angles in degrees.

    """
    return Transform().rotate(array).GetOrientation()


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


class _Prop3DMixin(_BoundsSizeMixin, ABC):
    """Add 3D transformations to props which do not inherit from :class:`pyvista.Prop3D`.

    Derived classes need to implement the :meth:`_post_set_update` method to define
    their behavior, e.g. manually apply a transformation.
    """

    def __init__(self) -> None:
        from pyvista import Actor  # Avoid circular import  # noqa: PLC0415

        self._prop3d = Actor()

    @property
    @wraps(Prop3D.scale.fget)  # type: ignore[attr-defined]
    def scale(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Wrap :class:`pyvista.Prop3D.scale."""
        return self._prop3d.scale

    @scale.setter
    @wraps(Prop3D.scale.fset)  # type: ignore[attr-defined]
    def scale(self, scale: VectorLike[float]) -> None:
        self._prop3d.scale = scale
        self._post_set_update()

    @property
    @wraps(Prop3D.position.fget)  # type: ignore[attr-defined]
    def position(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Wrap :class:`pyvista.Prop3D.position."""
        return self._prop3d.position

    @position.setter
    @wraps(Prop3D.position.fset)  # type: ignore[attr-defined]
    def position(self, position: VectorLike[float]) -> None:
        self._prop3d.position = position
        self._post_set_update()

    @property
    @wraps(Prop3D.orientation.fget)  # type: ignore[attr-defined]
    def orientation(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Wrap :class:`pyvista.Prop3D.orientation."""
        return self._prop3d.orientation

    @orientation.setter
    @wraps(Prop3D.orientation.fset)  # type: ignore[attr-defined]
    def orientation(self, orientation: VectorLike[float]) -> None:
        self._prop3d.orientation = orientation
        self._post_set_update()

    @property
    @wraps(Prop3D.origin.fget)  # type: ignore[attr-defined]
    def origin(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Wrap :class:`pyvista.Prop3D.origin."""
        return self._prop3d.origin

    @origin.setter
    @wraps(Prop3D.origin.fset)  # type: ignore[attr-defined]
    def origin(self, origin: VectorLike[float]) -> None:
        self._prop3d.origin = origin
        self._post_set_update()

    @property
    @wraps(Prop3D.user_matrix.fget)  # type: ignore[attr-defined]
    def user_matrix(self) -> NumpyArray[float]:  # numpydoc ignore=RT01
        """Wrap :class:`pyvista.Prop3D.user_matrix."""
        return self._prop3d.user_matrix

    @user_matrix.setter
    @wraps(Prop3D.user_matrix.fset)  # type: ignore[attr-defined]
    def user_matrix(self, matrix: TransformLike) -> None:
        self._prop3d.user_matrix = matrix
        self._post_set_update()

    @property
    def _transformation_matrix(self):
        """Transformation matrix applied to the actor.

        The transformation is computed from the attributes :attr:`position`
        :attr:`origin`, :attr:`scale`, :attr:`orientation`, and :attr:`user_matrix`.

        It is the actual transformation applied to the actor under-the-hood by vtk.
        """
        return array_from_vtkmatrix(self._prop3d.GetMatrix())

    @abstractmethod
    def _post_set_update(self):
        """Update object after setting Prop3D attributes."""

    @abstractmethod
    def _get_bounds(self) -> BoundsTuple:
        """Return the object's 3D bounds."""

    @property
    @wraps(Prop3D.bounds.fget)  # type: ignore[attr-defined]
    def bounds(self) -> BoundsTuple:  # numpydoc ignore=RT01
        """Wrap :class:`pyvista.Prop3D.bounds`."""
        return BoundsTuple(*self._get_bounds())

    @property
    @wraps(Prop3D.center.fget)  # type: ignore[attr-defined]
    def center(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Wrap :class:`pyvista.Prop3D.center."""
        bnds = self.bounds
        return (
            (bnds.x_min + bnds.x_max) / 2,
            (bnds.y_min + bnds.y_max) / 2,
            (bnds.z_min + bnds.z_max) / 2,
        )

    @property
    @wraps(Prop3D.length.fget)  # type: ignore[attr-defined]
    def length(self) -> float:  # numpydoc ignore=RT01
        """Wrap :class:`pyvista.Prop3D.length."""
        bnds = self.bounds
        return np.linalg.norm(
            (bnds.x_max - bnds.x_min, bnds.y_max - bnds.y_min, bnds.z_max - bnds.z_min)
        ).tolist()
