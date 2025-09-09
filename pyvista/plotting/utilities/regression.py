"""Image regression module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import TypeAlias
from typing import cast

import numpy as np

import pyvista
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core.utilities.arrays import point_array
from pyvista.core.utilities.helpers import wrap
from pyvista.plotting import _vtk

if TYPE_CHECKING:
    from pyvista import ImageData
    from pyvista.core._typing_core import NumpyArray
    from pyvista.plotting import Plotter

    ImageCompareType: TypeAlias = str | Path | np.ndarray | Plotter | _vtk.vtkImageData


def remove_alpha(img: _vtk.vtkImageData) -> ImageData:
    """Remove the alpha channel from a :vtk:`vtkImageData`.

    Parameters
    ----------
    img : :vtk:`vtkImageData`
        The input image data with an alpha channel.

    Returns
    -------
    ImageData
        The output image data with the alpha channel removed.

    """
    ec = _vtk.vtkImageExtractComponents()
    ec.SetComponents(0, 1, 2)
    ec.SetInputData(img)
    ec.Update()
    vtk_image: _vtk.vtkImageData = ec.GetOutput()
    return pyvista.wrap(vtk_image)


def wrap_image_array(arr):
    """Wrap a numpy array as a pyvista.ImageData.

    Parameters
    ----------
    arr : np.ndarray
        A numpy array of shape (X, Y, (3 or 4)) and dtype ``np.uint8``. For
        example, an array of shape ``(768, 1024, 3)``.

    Raises
    ------
    ValueError
        If the input array does not have 3 dimensions, the third dimension of
        the input array is not 3 or 4, or the input array is not of type
        ``np.uint8``.

    Returns
    -------
    pyvista.ImageData
        A PyVista ImageData object with the wrapped array data.

    """
    if arr.ndim != 3:
        msg = 'Expecting a X by Y by (3 or 4) array'
        raise ValueError(msg)
    if arr.shape[2] not in [3, 4]:
        msg = 'Expecting a X by Y by (3 or 4) array'
        raise ValueError(msg)
    if arr.dtype != np.uint8:
        msg = 'Expecting a np.uint8 array'
        raise ValueError(msg)

    img = _vtk.vtkImageData()
    img.SetDimensions(arr.shape[1], arr.shape[0], 1)
    wrap_img = pyvista.wrap(img)
    wrap_img.point_data['PNGImage'] = arr[::-1].reshape(-1, arr.shape[2])
    return wrap_img


def run_image_filter(imfilter: _vtk.vtkWindowToImageFilter) -> NumpyArray[float]:
    """Run a :vtk:`vtkWindowToImageFilter` and get output as array.

    Parameters
    ----------
    imfilter : :vtk:`vtkWindowToImageFilter`
        The :vtk:`vtkWindowToImageFilter` instance to be processed.

    Notes
    -----
    An empty array will be returned if an image cannot be extracted.

    Returns
    -------
    numpy.ndarray
        An array containing the filtered image data. The shape of the array
        is given by (height, width, -1) where height and width are the
        dimensions of the image.

    """
    # Update filter and grab pixels
    imfilter.Modified()
    imfilter.Update()
    image = cast('ImageData | None', wrap(imfilter.GetOutput()))
    if image is None:
        return np.empty((0, 0, 0))
    img_size = image.dimensions
    img_array = cast('NumpyArray[float]', point_array(image, 'ImageScalars'))
    # Reshape and write
    tgt_size = (img_size[1], img_size[0], -1)
    return img_array.reshape(tgt_size)[::-1]


@_deprecate_positional_args(allowed=['render_window'])
def image_from_window(  # noqa: PLR0917
    render_window,
    as_vtk: bool = False,  # noqa: FBT001, FBT002
    ignore_alpha: bool = False,  # noqa: FBT001, FBT002
    scale=1,
):
    """Extract the image from the render window as an array.

    Parameters
    ----------
    render_window : :vtk:`vtkRenderWindow`
        The render window to extract the image from.

    as_vtk : bool, default: False
        If set to True, the image will be returned as a VTK object.

    ignore_alpha : bool, default: False
        If set to True, the image will be returned in RGB format,
        otherwise, it will be returned in RGBA format.

    scale : int, default: 1
        The scaling factor of the extracted image. The default value is 1
        which means that no scaling is applied.

    Returns
    -------
    ndarray | :vtk:`vtkImageData`
        The image as an array or as a VTK object depending on the ``as_vtk`` parameter.

    """
    off = not render_window.GetInteractor().GetEnableRender()
    if off:
        render_window.GetInteractor().EnableRenderOn()
    imfilter = _vtk.vtkWindowToImageFilter()
    imfilter.SetInput(render_window)
    imfilter.SetScale(scale)
    imfilter.FixBoundaryOn()
    imfilter.ReadFrontBufferOff()
    imfilter.ShouldRerenderOff()
    if ignore_alpha:
        imfilter.SetInputBufferTypeToRGB()
    else:
        imfilter.SetInputBufferTypeToRGBA()
    imfilter.ReadFrontBufferOn()
    data = run_image_filter(imfilter)
    if off:
        # Critical for Trame and other offscreen tools
        render_window.GetInteractor().EnableRenderOff()
    if as_vtk:
        return wrap_image_array(data)
    return data


@_deprecate_positional_args(allowed=['im1', 'im2'])
def compare_images(  # noqa: PLR0917
    im1: ImageCompareType,
    im2: ImageCompareType,
    threshold: int = 1,
    use_vtk: bool = True,  # noqa: FBT001, FBT002
) -> float:
    """Compare two different images of the same size.

    Parameters
    ----------
    im1 : str | pathlib.Path | numpy.ndarray | pyvista.Plotter | :vtk:`vtkImageData`
        Path, :class:`pyvista.Plotter`, numpy array representing the output of
        a render window, or :vtk:`vtkImageData`.

    im2 : str | pathlib.Path | numpy.ndarray | pyvista.Plotter | :vtk:`vtkImageData`
        Path, :class:`pyvista.Plotter`, numpy array representing the output of
        a render window, or :vtk:`vtkImageData`.

    threshold : int, default: 1
        Threshold tolerance for pixel differences.  This should be
        greater than 0, otherwise it will always return an error, even
        on identical images.

    use_vtk : bool, default: True
        When disabled, computes the mean pixel error over the entire
        image using numpy.  The difference between pixel is calculated
        for each RGB channel, summed, and then divided by the number
        of pixels.  This is faster than using
        :vtk:`vtkImageDifference` but potentially less accurate.

    Returns
    -------
    float
        Total error between the images if using ``use_vtk=True``, and
        the mean pixel error when ``use_vtk=False``.

    Examples
    --------
    Compare two active plotters.

    >>> import pyvista as pv
    >>> pl1 = pv.Plotter()
    >>> _ = pl1.add_mesh(pv.Sphere(), smooth_shading=True)
    >>> pl2 = pv.Plotter()
    >>> _ = pl2.add_mesh(pv.Sphere(), smooth_shading=False)
    >>> error = pv.compare_images(pl1, pl2)

    Compare images from file.

    >>> import pyvista as pv
    >>> img1 = pv.read('img1.png')  # doctest:+SKIP
    >>> img2 = pv.read('img2.png')  # doctest:+SKIP
    >>> pv.compare_images(img1, img2)  # doctest:+SKIP

    """
    from pyvista import ImageData  # noqa: PLC0415
    from pyvista import Plotter  # noqa: PLC0415
    from pyvista import read  # noqa: PLC0415
    from pyvista import wrap  # noqa: PLC0415

    def to_img(img: ImageCompareType) -> pyvista.ImageData:
        if isinstance(img, ImageData):
            return img
        elif isinstance(img, _vtk.vtkImageData):  # pragma: no cover
            return wrap(img)
        elif isinstance(img, (str, Path)):
            dataset = read(img)
            if not isinstance(dataset, ImageData):
                msg = (
                    f'The file {img} may not be an image. PyVista read it in as a '
                    f'{type(dataset)!r}.'
                )
                raise TypeError(msg)

            return dataset
        elif isinstance(img, np.ndarray):
            return wrap_image_array(img)
        elif isinstance(img, Plotter):
            if img._first_time:  # must be rendered first else segfault
                img._on_first_render_request()
                img.render()
            if img.render_window is None:
                msg = 'Unable to extract image from Plotter as it has already been closed.'
                raise RuntimeError(msg)
            return image_from_window(img.render_window, as_vtk=True, ignore_alpha=True)
        else:
            msg = (
                f'Unsupported data type {type(img)}.  Should be '
                'either a np.ndarray, pyvista.Plotter, or vtk.vtkImageData'
            )
            raise TypeError(msg)

    im1_proc = remove_alpha(to_img(im1))
    im2_proc = remove_alpha(to_img(im2))

    if im1_proc.dimensions != im2_proc.dimensions:
        msg = 'Input images are not the same size.'
        raise RuntimeError(msg)

    if use_vtk:
        img_diff = _vtk.vtkImageDifference()
        img_diff.SetThreshold(threshold)
        img_diff.SetInputData(im1_proc)
        img_diff.SetImageData(im2_proc)
        img_diff.AllowShiftOff()  # vastly increases compute time when enabled
        # img_diff.AveragingOff()  # increases compute time
        img_diff.Update()
        return img_diff.GetThresholdedError()

    # unlikely but possible
    if im1_proc.active_scalars is None:  # pragma: no cover
        msg = 'Missing active scalars in first image'
        raise RuntimeError(msg)
    if im2_proc.active_scalars is None:  # pragma: no cover
        msg = 'Missing active scalars in second image'
        raise RuntimeError(msg)

    # otherwise, simply compute the mean pixel difference
    diff = np.abs(im1_proc.active_scalars - im2_proc.active_scalars)
    return float(np.sum(diff) / im1_proc.active_scalars.shape[0])
