"""This module provides a wrapper for vtk.vtkTexture."""

from typing import Union
import warnings

import numpy as np

import pyvista as pv
from pyvista import _vtk
from pyvista.plotting.opts import AnnotatedIntEnum
from pyvista.utilities.misc import PyVistaDeprecationWarning

from .dataset import DataObject


class Texture(_vtk.vtkTexture, DataObject):
    """Wrap vtk.Texture.

    Textures can be used to apply images to surfaces, as in the case of
    :ref:`ref_texture_example`.

    They can also be used for environment textures to affect the lighting of
    the scene, or even as a environment cubemap as in the case of
    :ref:`pbr_example` and :ref:`planets_example`.

    Parameters
    ----------
    uinput : str, vtkImageData, vtkTexture, numpy.ndarray, list, optional
        Filename, ``vtkImagedata``, ``vtkTexture``, :class:`numpy.ndarray` or a
        list of images to create a cubemap. If a list of images, must be of the
        same size and in the following order:

        * +X
        * -X
        * +Y
        * -Y
        * +Z
        * -Z

    **kwargs : dict, optional
        Optional arguments when reading from a file. Generally unused.

    Examples
    --------
    Load a texture from file. Files should be "image" or "image-like" files.

    >>> import os
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> path = examples.download_masonry_texture(load=False)
    >>> os.path.basename(path)
    'masonry.bmp'
    >>> texture = pv.Texture(path)
    >>> texture  # doctest:+SKIP
    Texture (0x7f3131807fa0)
      Components:      3
      Cube Map:        False
      Dimensions:      256, 256
    """

    class WrapType(AnnotatedIntEnum):
        """Types of wrapping a texture can support.

        Wrap mode for the texture coordinates valid values are:

        * CLAMP_TO_EDGE
        * REPEAT (Default in :class:`pyvista.Texture`)
        * MIRRORED_REPEAT
        * CLAMP_TO_BORDER

        See :attr:`Texture.wrap` for usage.

        """

        CLAMP_TO_EDGE = (0, 'Clamp to edge')
        REPEAT = (1, 'Repeat')
        MIRRORED_REPEAT = (2, 'Mirrored repeat')
        CLAMP_TO_BORDER = (3, 'Clamp to border')

    def __init__(self, *args, **kwargs):
        """Initialize the texture."""
        super().__init__(*args, **kwargs)

        if len(args) == 1:
            if isinstance(args[0], _vtk.vtkTexture):
                self._from_texture(args[0])
            elif isinstance(args[0], np.ndarray):
                self._from_array(args[0])
            elif isinstance(args[0], _vtk.vtkImageData):
                self._from_image_data(args[0])
            elif isinstance(args[0], str):
                self._from_file(filename=args[0], **kwargs)
            elif len(args[0]) == 6:
                # Create a cubemap
                self.mipmap = True
                self.interpolate = True
                self.cube_map = True  # Must be set prior to setting images

                # add each image to the cubemap
                for i, image in enumerate(args[0]):
                    if not isinstance(image, pv.UniformGrid):
                        raise ValueError(
                            'If a sequence, the each item in the first argument must be a '
                            'pyvista.UniformGrid'
                        )
                    # must flip y for cubemap to display properly
                    self.SetInputDataObject(i, image._flip_uniform(1))
            else:
                raise TypeError(f'Texture unable to be made from ({type(args[0])})')

    def _from_file(self, filename, **kwargs):
        try:
            image = pv.read(filename, **kwargs)
            if image.GetNumberOfPoints() < 2:
                raise ValueError("Problem reading the image with VTK.")
            self._from_image_data(image)
        except (KeyError, ValueError):
            from imageio import imread

            self._from_array(imread(filename))

    def _from_texture(self, texture):
        image = texture.GetInput()
        self._from_image_data(image)

    @property
    def interpolate(self) -> bool:
        """Return if interpolate is enabled or disabled.

        Examples
        --------
        Show the masonry texture without interpolation. Here, we zoom to show
        the individual pixels.

        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> texture.interpolation = False
        >>> texture.plot(cpos='xy', zoom=3)

        Plot the same texture with interpolation.

        >>> texture.interpolation = True
        >>> texture.plot(cpos='xy', zoom=3)

        """
        return bool(self.GetInterpolate())

    @interpolate.setter
    def interpolate(self, value: bool):
        self.SetInterpolate(value)

    @property
    def mipmap(self) -> bool:
        """Return if mipmap is enabled or disabled."""
        return bool(self.GetMipmap())

    @mipmap.setter
    def mipmap(self, value: bool):
        self.SetMipmap(value)

    def _from_image_data(self, image):
        if not isinstance(image, pv.UniformGrid):
            image = pv.UniformGrid(image)
        self.SetInputDataObject(image)
        self.Update()

    def _from_array(self, image):
        """Create a texture from a np.ndarray."""
        if not 2 <= image.ndim <= 3:
            # we support 2 [single component image] or 3 [e.g. rgb or rgba] dims
            raise ValueError('Input image must be nn by nm by RGB[A]')

        if image.ndim == 3:
            if not 3 <= image.shape[2] <= 4:
                raise ValueError('Third dimension of the array must be of size 3 (RGB) or 4 (RGBA)')
            n_components = image.shape[2]
        elif image.ndim == 2:
            n_components = 1

        grid = pv.UniformGrid(dimensions=(image.shape[1], image.shape[0], 1))
        grid.point_data['Image'] = np.flip(image.swapaxes(0, 1), axis=1).reshape(
            (-1, n_components), order='F'
        )
        grid.set_active_scalars('Image')
        self._from_image_data(grid)

    @property
    def repeat(self) -> bool:
        """Repeat the texture.

        This is provided for convenience and backwards compatibility.

        For new code, use :func:`Texture.wrap`.

        Examples
        --------
        Load the masonry texture and create a simple :class:`pyvista.PolyData`
        with texture coordinates using :func:`pyvista.Plane`. By default the
        texture coordinates are between 0 and 1. Let's raise these values over
        1 by multiplying them in place. This will allow us to wrap the texture.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> plane = pv.Plane()
        >>> plane.active_t_coords *= 2

        This is the texture plotted with repeat set to ``False``.

        >>> texture.repeat = False
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(plane, texture=texture)
        >>> pl.camera.zoom('tight')
        >>> pl.show()

        This is the texture plotted with repeat set to ``True``.

        >>> texture.repeat = True
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(plane, texture=texture)
        >>> pl.camera.zoom('tight')
        >>> pl.show()

        """
        return bool(self.GetRepeat())

    @repeat.setter
    def repeat(self, flag: bool):
        self.SetRepeat(flag)

    def flip(self, axis):
        """Flip this texture inplace along the specified axis.

        0 for X and 1 for Y.

        .. deprecated:: 0.37.0
           ``flip`` is deprecated. Use :func:`Texture.flip_x` or
           :func:`Texture.flip_x` instead.

        """
        warnings.warn(
            '`flip` is deprecated. Use `flip_x` or `flip_y` instead',
            PyVistaDeprecationWarning,
        )

        if not 0 <= axis <= 1:
            raise ValueError(f"Axis {axis} out of bounds")
        array = self.to_array()
        array = np.flip(array, axis=1 - axis)
        self._from_array(array)

    def flip_x(self) -> 'Texture':
        """Flip the texture in the x direction.

        Returns
        -------
        pyvista.Texture
            Flipped texture.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_puppy_texture()
        >>> flipped = texture.flip_x()
        >>> flipped.plot()

        """
        return Texture(self.to_image()._flip_uniform(0))

    def flip_y(self) -> 'Texture':
        """Flip the texture in the y direction.

        Returns
        -------
        pyvista.Texture
            Flipped texture.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_puppy_texture()
        >>> flipped = texture.flip_y()
        >>> flipped.plot()

        """
        return Texture(self.to_image()._flip_uniform(1))

    def to_image(self):
        """Return the texture as an image.

        Returns
        -------
        pyvista.UniformGrid
            Texture represented as a uniform grid.

        """
        return self.GetInput()

    def to_array(self):
        """Return the texture as an array.

        Returns
        -------
        numpy.ndarray
            Texture as a numpy array.

        """
        image = self.to_image()

        if image.active_scalars.ndim > 1:
            shape = (image.dimensions[1], image.dimensions[0], self.n_components)
        else:
            shape = (image.dimensions[1], image.dimensions[0])

        return np.flip(image.active_scalars.reshape(shape, order='F'), axis=1).swapaxes(1, 0)

    @property
    def cube_map(self):
        """Return ``True`` if cube mapping is enabled and ``False`` otherwise.

        Is this texture a cube map? If so it needs 6 inputs, one for
        each side of the cube. You must set this before connecting the
        inputs.  The inputs must all have the same size, data type,
        and depth.
        """
        return self.GetCubeMap()

    @cube_map.setter
    def cube_map(self, flag):
        self.SetCubeMap(flag)

    def copy(self):
        """Make a copy of this texture.

        Returns
        -------
        pyvista.Texture
            Copied texture.
        """
        return Texture(self.to_image().copy())

    def to_skybox(self):
        """Return the texture as a ``vtkSkybox`` if cube mapping is enabled.

        Returns
        -------
        vtk.vtkSkybox
            Skybox if cube mapping is enabled.  Otherwise, ``None``.

        """
        if self.cube_map:
            skybox = _vtk.vtkSkybox()
            skybox.SetTexture(self)
            return skybox

    def __repr__(self):
        """Return the object representation."""
        return pv.DataSet.__repr__(self)

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = []
        attrs.append(("Components", self.n_components, "{:d}"))
        attrs.append(("Cube Map", self.cube_map, "{:}"))
        attrs.append(("Dimensions", self.dimensions, "{:d}, {:d}"))
        return attrs

    @property
    def n_components(self) -> int:
        """Return the number of components in the image.

        In textures, 3 or 4 component are used for representing RGB and RGBA
        images.

        Examples
        --------
        Show the number of components in the example masonry texture.

        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> texture.n_components
        3

        """
        input_data = self.GetInput()
        if input_data is None:
            return 0
        return input_data.GetPointData().GetScalars().GetNumberOfComponents()

    @property
    def dimensions(self) -> tuple:
        """Dimensions of the texture.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> texture.dimensions
        (256, 256)

        """
        input_data = self.GetInput()
        if input_data is None:
            return (0, 0)
        return input_data.GetDimensions()[:2]

    def plot(self, **kwargs):
        """Plot the texture as an image.

        If the texture is a cubemap, it will be displayed as a skybox with a
        sphere in the center reflecting the environment.

        Parameters
        ----------
        **kwargs : dict, optional
            Optional keyworld arguments. See :func:`pyvista.plot`.

        Returns
        -------
        various or None
            See the returns section of :func:`pyvista.plot`.

        Examples
        --------
        Plot a simple texture.

        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> texture.plot()

        Plot a cubemap as a skybox.

        >>> cube_map = examples.download_dikhololo_night()
        >>> cube_map.plot()

        """
        if self.cube_map:
            return self._plot_skybox(**kwargs)
        kwargs.setdefault('zoom', 'tight')
        kwargs.setdefault('lighting', False)
        kwargs.setdefault('show_axes', False)
        kwargs.setdefault('show_scalar_bar', False)
        mesh = pv.Plane(i_size=self.dimensions[0], j_size=self.dimensions[1])
        return mesh.plot(texture=self, **kwargs)

    def _plot_skybox(self, **kwargs):
        """Plot this texture as a skybox."""
        cpos = kwargs.pop('cpos', 'xy')
        zoom = kwargs.pop('zoom', 0.5)
        show_axes = kwargs.pop('show_axes', True)
        lighting = kwargs.pop('lighting', None)
        pl = pv.Plotter(lighting=lighting)
        pl.add_actor(self.to_skybox())
        pl.set_environment_texture(self, True)
        pl.add_mesh(pv.Sphere(), pbr=True, roughness=0.5, metallic=1.0)
        pl.camera_position = cpos
        pl.camera.zoom(zoom)
        if show_axes:
            pl.show_axes()
        pl.show(**kwargs)

    @property
    def wrap(self) -> 'Texture.WrapType':
        """Return or set the Wrap mode for the texture coordinates.

        Wrap mode for the texture coordinates valid values are:

        * ``0`` - CLAMP_TO_EDGE
        * ``1`` - REPEAT
        * ``2`` - MIRRORED_REPEAT
        * ``3`` - CLAMP_TO_BORDER

        Notes
        -----
        CLAMP_TO_BORDER is not supported with OpenGL ES <= 3.2. Wrap will
        default to CLAMP_TO_EDGE if it is set to CLAMP_TO_BORDER in this case.

        Examples
        --------
        Load the masonry texture and create a simple :class:`pyvista.PolyData`
        with texture coordinates using :func:`pyvista.Plane`. By default the
        texture coordinates are between 0 and 1. Let's raise these values over
        1 by multiplying them in place. This will allow us to wrap the texture.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> plane = pv.Plane()
        >>> plane.active_t_coords *= 2

        Let's now set the texture wrap to clamp to edge and visualize it.

        >>> texture.wrap = pv.Texture.WrapType.CLAMP_TO_EDGE
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(plane, texture=texture)
        >>> pl.camera.zoom('tight')
        >>> pl.show()

        Here is the default repeat:

        >>> texture.wrap = pv.Texture.WrapType.REPEAT
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(plane, texture=texture)
        >>> pl.camera.zoom('tight')
        >>> pl.show()

        And here is mirrored repeat:

        >>> texture.wrap = pv.Texture.WrapType.MIRRORED_REPEAT
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(plane, texture=texture)
        >>> pl.camera.zoom('tight')
        >>> pl.show()

        Finally, this is clamp to border:

        >>> texture.wrap = pv.Texture.WrapType.CLAMP_TO_BORDER
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(plane, texture=texture)
        >>> pl.camera.zoom('tight')
        >>> pl.show()

        """
        return Texture.WrapType(self.GetWrap())  # type: ignore

    @wrap.setter
    def wrap(self, value: Union['Texture.WrapType', int]):
        self.SetWrap(value)
