"""3D text actor that faces the camera."""

from __future__ import annotations

from typing import ClassVar
from typing import List

import pyvista

from . import _vtk
from .actor import Actor


class Text3DFollower(Actor, _vtk.vtkFollower):
    """3D text actor that faces the camera.

    Parameters
    ----------
    string : str
        String of text to display.

    position : vector
        Position of the text in XYZ coordinates. The text is centered at this location.

    prop : pyvista.Property, optional
        Property of the actor.

    name : str, optional
        The name of this actor used when tracking on a plotter.

    """

    _new_attr_exceptions: ClassVar[List[str]] = ['_text_source']

    def __init__(self, string: str = "", position=None, prop=None, name=None):
        super().__init__(prop=prop, name=name)
        text_source = pyvista.Text3DSource(string=string, depth=0.0)
        text_source.update()
        out = text_source.output
        mapper = _vtk.vtkPolyDataMapper()
        mapper.SetInputData(out)

        self._text_source = text_source
        self.mapper = mapper
        self.prop.lighting = False
        self.position = (0, 0, 0) if position is None else position

    def __del__(self):
        del self._text_source

    @property
    def camera(self) -> pyvista.Camera:  # numpydoc ignore=RT01
        """Return or set the depth of the text."""
        return self.GetCamera()

    @camera.setter
    def camera(self, cam: pyvista.Camera):  # numpydoc ignore=GL08
        self.SetCamera(cam)

    @property
    def string(self) -> str:  # numpydoc ignore=RT01
        """Return or set the text string."""
        return self._text_source.string

    @string.setter
    def string(self, string: str):  # numpydoc ignore=GL08
        self._text_source.string = string
        self._text_source.update()

    @property
    def depth(self) -> float:  # numpydoc ignore=RT01
        """Return or set the depth of the text."""
        return self._text_source.depth

    @depth.setter
    def depth(self, depth: float):  # numpydoc ignore=GL08
        self._text_source.depth = depth
        self._text_source.update()

    @property
    def height(self) -> float:  # numpydoc ignore=RT01
        """Return or set the height of the text."""
        return self._text_source.height

    @height.setter
    def height(self, height: float):  # numpydoc ignore=GL08
        self._text_source.height = height
        self._text_source.update()

    @property
    def width(self) -> float:  # numpydoc ignore=RT01
        """Return or set the width of the text."""
        return self._width

    @width.setter
    def width(self, width: float):  # numpydoc ignore=GL08
        self._text_source.width = width
        self._text_source.update()
