"""Wrap :vtk:`vtkFollower` module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _vtk
from .actor import Actor

if TYPE_CHECKING:
    from ._property import Property
    from .camera import Camera
    from .mapper import _BaseMapper


class Follower(Actor, _vtk.vtkFollower):
    """Wrap :vtk:`vtkFollower`.

    A Follower is a subclass of Actor that always faces the camera. It
    is useful for screen-aligned labels and billboarding effects.

    The Follower maintains the position and scale of the actor but updates
    its orientation continuously to face the camera.

    Parameters
    ----------
    mapper : pyvista.DataSetMapper, optional
        DataSetMapper.

    prop : pyvista.Property, optional
        Property of the actor.

    name : str, optional
        The name of this actor used when tracking on a plotter.

    camera : pyvista.Camera, optional
        Camera to follow. If not provided, no camera will be set initially.

    See Also
    --------
    :ref:`follower_actor_example` : Example demonstrating the use of follower actors.

    Examples
    --------
    Create a follower actor that always faces the camera.

    >>> import pyvista as pv
    >>> mesh = pv.Sphere()
    >>> mapper = pv.DataSetMapper(mesh)
    >>> follower = pv.Follower(mapper=mapper)
    >>> follower
    Follower (...)
      Center:                     (0.0, 0.0, 0.0)
      Pickable:                   True
      Position:                   (0.0, 0.0, 0.0)
      Scale:                      (1.0, 1.0, 1.0)
      Visible:                    True
      X Bounds                    -4.993E-01, 4.993E-01
      Y Bounds                    -4.965E-01, 4.965E-01
      Z Bounds                    -5.000E-01, 5.000E-01
      User matrix:                Identity
      Has mapper:                 True
    ...

    Set the camera for the follower and render it.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(follower)
    >>> follower.SetCamera(pl.camera)
    >>> pl.show()

    """

    def __init__(  # noqa: PLR0917
        self,
        mapper: _BaseMapper | None = None,
        prop: Property | None = None,
        name: str | None = None,
        camera: Camera | None = None,
    ) -> None:
        """Initialize follower."""
        super().__init__(mapper=mapper, prop=prop, name=name)

        # Set the camera if provided
        if camera is not None:
            self.SetCamera(camera)

    @property
    def camera(self) -> Camera | None:  # numpydoc ignore=RT01
        """Return or set the camera of this follower.

        The follower will continuously update its orientation to face this camera.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> pl = pv.Plotter()
        >>> follower = pv.Follower(pv.DataSetMapper(mesh))
        >>> _ = pl.add_actor(follower)
        >>> follower.camera = pl.camera

        """
        return self.GetCamera()  # type: ignore[return-value]

    @camera.setter
    def camera(self, cam: Camera) -> None:
        self.SetCamera(cam)
