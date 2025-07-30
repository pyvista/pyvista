"""Wrapper for :vtk:`vtkTexture`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import warnings

import numpy as np

import pyvista
from pyvista.core import _validation
from pyvista.core.dataobject import DataObject
from pyvista.core.utilities.fileio import _try_imageio_imread
from pyvista.core.utilities.misc import AnnotatedIntEnum

from . import _vtk

if TYPE_CHECKING:
    from typing import Literal

    from pyvista.core._typing_core import NumpyArray


class Texture(DataObject, _vtk.vtkTexture):
    """Wrap :vtk:`vtkTexture`.

    Textures can be used to apply images to surfaces, as in the case of
    :ref:`texture_example`.

    They can also be used for environment textures to affect the lighting of
    the scene, or even as a environment cubemap as in the case of
    :ref:`pbr_example` and :ref:`planets_example`.

    .. versionchanged:: 0.46

        Multi-component textures (with RGB(A) scalars) now use ``'direct'`` :attr:`color_mode`
        by default.

    Parameters
    ----------
    uinput : str, :vtk:`vtkImageData`, :vtk:`vtkTexture`, sequence[ImageData], optional
        Filename, :vtk:`vtkImageData`, :vtk:`vtkTexture`, :class:`numpy.ndarray` or a
        sequence of images to create a cubemap. If a sequence of images, must
        be of the same size and in the following order:

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
    Load a texture from file. File should be a "image" or "image-like" file.

    >>> from pathlib import Path
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> path = examples.download_masonry_texture(load=False)
    >>> Path(path).name
    'masonry.bmp'
    >>> texture = pv.Texture(path)
    >>> texture
    Texture (...)
      Components:   3
      Cube Map:     False
      Dimensions:   256, 256

    Create a texture from an RGB array. Note how this is colored per "point"
    rather than per "pixel".

    >>> import numpy as np
    >>> arr = np.array(
    ...     [
    ...         [255, 255, 255],
    ...         [255, 0, 0],
    ...         [0, 255, 0],
    ...         [0, 0, 255],
    ...     ],
    ...     dtype=np.uint8,
    ... )
    >>> arr = arr.reshape((2, 2, 3))
    >>> texture = pv.Texture(arr)
    >>> texture.plot()

    Create a cubemap from 6 images.

    >>> px = examples.download_sky(direction='posx')  # doctest:+SKIP
    >>> nx = examples.download_sky(direction='negx')  # doctest:+SKIP
    >>> py = examples.download_sky(direction='posy')  # doctest:+SKIP
    >>> ny = examples.download_sky(direction='negy')  # doctest:+SKIP
    >>> pz = examples.download_sky(direction='posz')  # doctest:+SKIP
    >>> nz = examples.download_sky(direction='negz')  # doctest:+SKIP
    >>> texture = pv.Texture([px, nx, py, ny, pz, nz])  # doctest:+SKIP
    >>> texture.cube_map  # doctest:+SKIP
    True

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

    def __init__(self, uinput=None, **kwargs):
        """Initialize the texture."""
        super().__init__(uinput)

        if isinstance(uinput, _vtk.vtkTexture):
            self._from_texture(uinput)
        elif isinstance(uinput, np.ndarray):
            self._from_array(uinput)
        elif isinstance(uinput, _vtk.vtkImageData):
            self._from_image_data(uinput)
        elif isinstance(uinput, str):
            self._from_file(filename=uinput, **kwargs)
        elif isinstance(uinput, Sequence) and len(uinput) == 6:
            # Create a cubemap
            self.mipmap = True
            self.interpolate = True
            self.cube_map = True  # Must be set prior to setting images

            # add each image to the cubemap
            for i, image in enumerate(uinput):
                if not isinstance(image, pyvista.ImageData):
                    msg = (
                        'If a sequence, the each item in the first argument must be a '
                        'pyvista.ImageData'
                    )
                    raise TypeError(msg)
                # must flip y for cubemap to display properly
                self.SetInputDataObject(i, image._flip_uniform(1))
        elif uinput is None:
            pass
        else:
            msg = f'Cannot create a pyvista.Texture from ({type(uinput)})'
            raise TypeError(msg)

        if self.n_components in (3, 4):  # RGB or RGBA
            self.color_mode = 'direct'

    def _from_file(self, filename, **kwargs):
        try:
            image = pyvista.read(filename, **kwargs)
            if image.n_points < 2:  # pragma: no cover
                msg = 'Problem reading the image with VTK.'
                raise RuntimeError(msg)
            self._from_image_data(image)
        except (KeyError, ValueError, OSError):
            self._from_array(_try_imageio_imread(filename))  # pragma: no cover

    def _from_texture(self, texture):
        image = texture.GetInput()
        self._from_image_data(image)

    @property
    def color_mode(self) -> Literal['map', 'direct']:  # numpydoc ignore=RT01
        """Return or set the color mode.

        Either ``'direct'``, or ``'map'``.

        * ``'direct'`` - All integer types are treated as colors with values in
          the range 0-255 and floating types are treated as colors with values
          in the range 0.0-1.0
        * ``'map'`` - All scalar data will be mapped through the lookup table.

        .. versionadded:: 0.46

        """
        mode = self.GetColorMode()
        return 'direct' if mode == 2 else 'map'

    @color_mode.setter
    def color_mode(self, value: Literal['map', 'direct']):
        _validation.check_contains(['map', 'direct'], must_contain=value, name='color_mode')
        if value == 'direct':
            self.SetColorModeToDirectScalars()
        else:
            self.SetColorModeToMapScalars()

    @property
    def interpolate(self) -> bool:  # numpydoc ignore=RT01
        """Return if interpolate is enabled or disabled.

        Examples
        --------
        Show the masonry texture without interpolation. Here, we zoom to show
        the individual pixels.

        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> texture.interpolate = False
        >>> texture.plot(cpos='xy', zoom=3)

        Plot the same texture with interpolation.

        >>> texture.interpolate = True
        >>> texture.plot(cpos='xy', zoom=3)

        """
        return bool(self.GetInterpolate())

    @interpolate.setter
    def interpolate(self, value: bool):
        self.SetInterpolate(value)

    @property
    def mipmap(self) -> bool:  # numpydoc ignore=RT01
        """Return if mipmap is enabled or disabled."""
        return bool(self.GetMipmap())

    @mipmap.setter
    def mipmap(self, value: bool):
        self.SetMipmap(value)

    def _from_image_data(self, image):
        if not isinstance(image, pyvista.ImageData):
            image = pyvista.ImageData(image)
        self.SetInputDataObject(image)
        self.Update()

    def _from_array(self, image):
        """Create a texture from a np.ndarray."""
        if image.ndim not in [2, 3]:
            # we support 2 [single component image] or 3 [e.g. rgb or rgba] dims
            msg = 'Input image must be nn by nm by RGB[A]'
            raise ValueError(msg)

        if image.ndim == 3:
            if image.shape[2] not in [1, 3, 4]:
                msg = 'Third dimension of the array must be of size 3 (RGB) or 4 (RGBA)'
                raise ValueError(msg)
            n_components = image.shape[2]
        elif image.ndim == 2:
            n_components = 1

        grid = pyvista.ImageData(dimensions=(image.shape[1], image.shape[0], 1))
        grid.point_data['Image'] = np.flip(image.swapaxes(0, 1), axis=1).reshape(
            (-1, n_components),
            order='F',
        )
        grid.set_active_scalars('Image')
        self._from_image_data(grid)

    @property
    def repeat(self) -> bool:  # numpydoc ignore=RT01
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
        >>> plane.active_texture_coordinates *= 2

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

    def flip_x(self) -> Texture:
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
        return Texture(self.to_image()._flip_uniform(0))  # type: ignore[abstract]

    def flip_y(self) -> Texture:
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
        return Texture(self.to_image()._flip_uniform(1))  # type: ignore[abstract]

    def to_image(self):
        """Return the texture as an image.

        Returns
        -------
        pyvista.ImageData
            Texture represented as a uniform grid.

        """
        return self.GetInput()

    def to_array(self) -> NumpyArray[float]:
        """Return the texture as an array.

        Notes
        -----
        The shape of the array's first two dimensions will be swapped. For
        example, a ``(300, 200)`` image will return an array of ``(200, 300)``.

        Returns
        -------
        numpy.ndarray
            Texture as a numpy array.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_puppy_texture()
        >>> texture
        Texture (...)
          Components:   3
          Cube Map:     False
          Dimensions:   1600, 1200
        >>> texture.to_array().shape
        (1200, 1600, 3)
        >>> texture.to_array().dtype
        dtype('uint8')

        """
        return self.to_image().active_scalars.reshape(
            [*list(self.dimensions)[::-1], self.n_components]
        )[::-1]

    def rotate_cw(self) -> Texture:
        """Rotate this texture 90 degrees clockwise.

        Returns
        -------
        pyvista.Texture
            Rotated texture.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_puppy_texture()
        >>> rotated = texture.rotate_cw()
        >>> rotated.plot()

        """
        return Texture(np.rot90(self.to_array()))  # type: ignore[abstract]

    def rotate_ccw(self) -> Texture:
        """Rotate this texture 90 degrees counter-clockwise.

        Returns
        -------
        pyvista.Texture
            Rotated texture.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_puppy_texture()
        >>> rotated = texture.rotate_ccw()
        >>> rotated.plot()

        """
        return Texture(np.rot90(self.to_array(), k=3))  # type: ignore[abstract]

    @property
    def cube_map(self) -> bool:  # numpydoc ignore=RT01
        """Return ``True`` if cube mapping is enabled and ``False`` otherwise."""
        return self.GetCubeMap()

    @cube_map.setter
    def cube_map(self, flag: bool):
        self.SetCubeMap(flag)

    def copy(self):  # type: ignore[override]
        """Make a copy of this texture.

        Returns
        -------
        pyvista.Texture
            Copied texture.

        """
        return Texture(self.to_image().copy())  # type: ignore[abstract]

    def to_skybox(self):
        """Return the texture as a :vtk:`vtkSkybox` if cube mapping is enabled.

        Returns
        -------
        :vtk:`vtkSkybox`
            Skybox if cube mapping is enabled.  Otherwise, ``None``.

        """
        if self.cube_map:
            skybox = _vtk.vtkSkybox()
            skybox.SetTexture(self)
            return skybox
        return None

    def __repr__(self):
        """Return the object representation."""
        return pyvista.DataSet.__repr__(self)  # type: ignore[type-var]

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = []
        attrs.append(('Components', self.n_components, '{:d}'))
        attrs.append(('Cube Map', self.cube_map, '{:}'))
        attrs.append(('Dimensions', self.dimensions, '{:d}, {:d}'))  # type: ignore[arg-type]
        return attrs

    @property
    def n_components(self) -> int:  # numpydoc ignore=RT01
        """Return the number of components in the image.

        In textures, 3 or 4 components are used for representing RGB and RGBA
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
    def dimensions(self) -> tuple[int, int]:  # numpydoc ignore=RT01
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
        pyvista.Actor | None
            See the returns section of :func:`pyvista.plot`.

        Examples
        --------
        Plot a simple texture.

        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> texture.plot()

        Plot a cubemap as a skybox.

        >>> cube_map = examples.download_sky_box_cube_map()
        >>> cube_map.plot()

        """
        if self.cube_map:
            return self._plot_skybox(**kwargs)
        kwargs.setdefault('zoom', 'tight')
        kwargs.setdefault('lighting', False)
        kwargs.setdefault('show_axes', False)
        kwargs.setdefault('show_scalar_bar', False)
        mesh = pyvista.Plane(i_size=self.dimensions[0], j_size=self.dimensions[1])
        self.SetUseSRGBColorSpace(True)
        return mesh.plot(texture=self, **kwargs)

    def _plot_skybox(self, **kwargs):
        """Plot this texture as a skybox."""
        cpos = kwargs.pop('cpos', 'xy')
        zoom = kwargs.pop('zoom', 0.5)
        show_axes = kwargs.pop('show_axes', True)
        lighting = kwargs.pop('lighting', None)
        pl = pyvista.Plotter(lighting=lighting)
        pl.add_actor(self.to_skybox())
        pl.set_environment_texture(self, is_srgb=True)
        pl.add_mesh(pyvista.Sphere(), pbr=True, roughness=0.5, metallic=1.0)
        pl.camera_position = cpos
        pl.camera.zoom(zoom)
        if show_axes:
            pl.show_axes()
        pl.show(**kwargs)

    @property
    def wrap(self) -> Texture.WrapType:  # numpydoc ignore=RT01
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

        Requires ``vtk`` v9.1.0 or newer.

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
        >>> plane.active_texture_coordinates *= 2

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
        if not hasattr(self, 'GetWrap'):  # pragma: no cover
            from pyvista.core.errors import VTKVersionError  # noqa: PLC0415

            msg = '`wrap` requires VTK v9.1.0 or newer.'
            raise VTKVersionError(msg)

        return Texture.WrapType(self.GetWrap())  # type: ignore[call-arg]

    @wrap.setter
    def wrap(self, value: Texture.WrapType | int):
        if not hasattr(self, 'SetWrap'):  # pragma: no cover
            from pyvista.core.errors import VTKVersionError  # noqa: PLC0415

            msg = '`wrap` requires VTK v9.1.0 or newer.'
            raise VTKVersionError(msg)

        self.SetWrap(value)

    def to_grayscale(self) -> Texture:
        """Convert this texture as a single component (grayscale) texture.

        Returns
        -------
        pyvista.Texture
            Texture converted to grayscale. If already grayscale, the original
            texture itself is returned.

        Notes
        -----
        The transparency channel (if available) will be dropped.

        Follows the `CCIR 601 <https://en.wikipedia.org/wiki/Rec._601>`_ luma
        calculation equation of ``Y = 0.299*R + 0.587*G + 0.114*B``.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> bw_texture = texture.to_grayscale()
        >>> bw_texture
        Texture (...)
          Components:   1
          Cube Map:     False
          Dimensions:   256, 256
        >>> bw_texture.plot()

        """
        if self.n_components == 1:
            return self.copy()

        data = self.to_array()
        r, g, b = data[..., 0], data[..., 1], data[..., 2]
        data = (0.299 * r + 0.587 * g + 0.114 * b).round().astype(np.uint8)
        return Texture(data)  # type: ignore[abstract]


def image_to_texture(image):
    """Convert :class:`pyvista.ImageData` to a :class:`pyvista.Texture`.

    Parameters
    ----------
    image : pyvista.ImageData | :vtk:`vtkImageData`
        Image to convert.

    Returns
    -------
    pyvista.Texture
        The texture.

    """
    return Texture(image)  # type: ignore[abstract]


def numpy_to_texture(image):
    """Convert a NumPy image array to a :class:`pyvista.Texture`.

    Parameters
    ----------
    image : numpy.ndarray
        Numpy image array. Texture datatype expected to be ``np.uint8``.

    Returns
    -------
    pyvista.Texture
        PyVista texture.

    Examples
    --------
    Create an all white texture.

    >>> import pyvista as pv
    >>> import numpy as np
    >>> tex_arr = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
    >>> tex = pv.numpy_to_texture(tex_arr)

    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
        warnings.warn(
            'Expected `image` dtype to be ``np.uint8``. `image` has been copied '
            'and converted to np.uint8.',
            UserWarning,
        )

    return Texture(image)  # type: ignore[abstract]
