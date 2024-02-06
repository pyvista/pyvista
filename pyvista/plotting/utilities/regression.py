"""Image regression module."""
from typing import Optional, cast

import numpy as np

import pyvista
from pyvista.core.utilities.arrays import point_array
from pyvista.core.utilities.helpers import wrap
from pyvista.plotting import _vtk


def remove_alpha(img):
    """Remove the alpha channel from a ``vtk.vtkImageData``.

    Parameters
    ----------
    img : vtk.vtkImageData
        The input image data with an alpha channel.

    Returns
    -------
    pyvista.ImageData
        The output image data with the alpha channel removed.

    """
    ec = _vtk.vtkImageExtractComponents()
    ec.SetComponents(0, 1, 2)
    ec.SetInputData(img)
    ec.Update()
    return pyvista.wrap(ec.GetOutput())


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
        raise ValueError('Expecting a X by Y by (3 or 4) array')
    if arr.shape[2] not in [3, 4]:
        raise ValueError('Expecting a X by Y by (3 or 4) array')
    if arr.dtype != np.uint8:
        raise ValueError('Expecting a np.uint8 array')

    img = _vtk.vtkImageData()
    img.SetDimensions(arr.shape[1], arr.shape[0], 1)
    wrap_img = pyvista.wrap(img)
    wrap_img.point_data['PNGImage'] = arr[::-1].reshape(-1, arr.shape[2])
    return wrap_img


def run_image_filter(imfilter: _vtk.vtkWindowToImageFilter) -> np.ndarray:
    """Run a ``vtkWindowToImageFilter`` and get output as array.

    Parameters
    ----------
    imfilter : _vtk.vtkWindowToImageFilter
        The ``vtkWindowToImageFilter`` instance to be processed.

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
    image = cast(Optional[pyvista.ImageData], wrap(imfilter.GetOutput()))
    if image is None:
        return np.empty((0, 0, 0))
    img_size = image.dimensions
    img_array = point_array(image, 'ImageScalars')
    # Reshape and write
    tgt_size = (img_size[1], img_size[0], -1)
    return img_array.reshape(tgt_size)[::-1]


def image_from_window(render_window, as_vtk=False, ignore_alpha=False, scale=1):
    """Extract the image from the render window as an array.

    Parameters
    ----------
    render_window : vtk.vtkRenderWindow
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
    ndarray | vtk.vtkImageData
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


def compare_images(im1, im2, threshold=1, use_vtk=True):
    """Compare two different images of the same size.

    Parameters
    ----------
    im1 : str | numpy.ndarray | vtkRenderWindow | vtkImageData
        Render window, numpy array representing the output of a render
        window, or ``vtkImageData``.

    im2 : str | numpy.ndarray | vtkRenderWindow | vtkImageData
        Render window, numpy array representing the output of a render
        window, or ``vtkImageData``.

    threshold : int, default: 1
        Threshold tolerance for pixel differences.  This should be
        greater than 0, otherwise it will always return an error, even
        on identical images.

    use_vtk : bool, default: True
        When disabled, computes the mean pixel error over the entire
        image using numpy.  The difference between pixel is calculated
        for each RGB channel, summed, and then divided by the number
        of pixels.  This is faster than using
        ``vtk.vtkImageDifference`` but potentially less accurate.

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
    from pyvista import ImageData, Plotter, read, wrap

    def to_img(img):  # numpydoc ignore=GL08
        if isinstance(img, ImageData):  # pragma: no cover
            return img
        elif isinstance(img, _vtk.vtkImageData):
            return wrap(img)
        elif isinstance(img, str):
            return read(img)
        elif isinstance(img, np.ndarray):
            return wrap_image_array(img)
        elif isinstance(img, Plotter):
            if img._first_time:  # must be rendered first else segfault
                img._on_first_render_request()
                img.render()
            if img.render_window is None:
                raise RuntimeError(
                    'Unable to extract image from Plotter as it has already been closed.'
                )
            return image_from_window(img.render_window, True, ignore_alpha=True)
        else:
            raise TypeError(
                f'Unsupported data type {type(img)}.  Should be '
                'Either a np.ndarray, vtkRenderWindow, or vtkImageData'
            )

    im1 = remove_alpha(to_img(im1))
    im2 = remove_alpha(to_img(im2))

    if im1.GetDimensions() != im2.GetDimensions():
        raise RuntimeError('Input images are not the same size.')

    if use_vtk:
        img_diff = _vtk.vtkImageDifference()
        img_diff.SetThreshold(threshold)
        img_diff.SetInputData(im1)
        img_diff.SetImageData(im2)
        img_diff.AllowShiftOff()  # vastly increases compute time when enabled
        # img_diff.AveragingOff()  # increases compute time
        img_diff.Update()
        return img_diff.GetThresholdedError()

    # otherwise, simply compute the mean pixel difference
    diff = np.abs(im1.point_data[0] - im2.point_data[0])
    return np.sum(diff) / im1.point_data[0].shape[0]
