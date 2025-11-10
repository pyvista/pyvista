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

    Examples
    --------
    Create a scene with a Follower text that always faces the camera and a transparent cube.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> pl = pv.Plotter()

    Create the "Hello" text that will follow the camera.

    >>> text_mesh = pv.Text3D('Hello', depth=0.1)
    >>> text_mesh = text_mesh.translate(
    ...     [-text_mesh.center[0], -text_mesh.center[1], 0]
    ... )

    Create mapper and follower actor for the text.

    >>> text_mapper = pv.DataSetMapper(text_mesh)
    >>> follower = pv.Follower(mapper=text_mapper)
    >>> follower.prop.color = 'gold'
    >>> _ = pl.add_actor(follower)

    Create a transparent cube that doesn't follow the camera.

    >>> cube = pv.Cube()
    >>> cube_actor = pl.add_mesh(
    ...     cube, color='MidnightBlue', opacity=0.3, show_edges=False
    ... )

    Set the follower's camera and show the scene.

    >>> follower.camera = pl.camera
    >>> pl.show()

    """

    def __init__(
        self,
        mapper: _BaseMapper | None = None,
        prop: Property | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize follower."""
        super().__init__(mapper=mapper, prop=prop, name=name)

    @property
    def camera(self) -> Camera | None:  # numpydoc ignore=RT01
        """Return or set the camera of this follower.

        The follower will continuously update its orientation to face this camera.

        """
        return self.GetCamera()  # type: ignore[return-value]

    @camera.setter
    def camera(self, cam: Camera) -> None:
        self.SetCamera(cam)
