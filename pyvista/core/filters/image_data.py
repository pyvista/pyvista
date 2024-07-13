"""Filters with a class to manage filters/algorithms for uniform grid datasets."""

from __future__ import annotations

from collections.abc import Iterable
import operator
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import AmbiguousDataError
from pyvista.core.errors import MissingDataError
from pyvista.core.filters import _get_output
from pyvista.core.filters import _update_alg
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.core.utilities.arrays import FieldAssociation
from pyvista.core.utilities.arrays import set_default_active_scalars
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import abstract_class

if TYPE_CHECKING:  # pragma: no cover
    from pyvista.core._typing_core import VectorLike


@abstract_class
class ImageDataFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for uniform grid datasets."""

    def gaussian_smooth(self, radius_factor=1.5, std_dev=2.0, scalars=None, progress_bar=False):
        """Smooth the data with a Gaussian kernel.

        Parameters
        ----------
        radius_factor : float | sequence[float], default: 1.5
            Unitless factor to limit the extent of the kernel.

        std_dev : float | sequence[float], default: 2.0
            Standard deviation of the kernel in pixel units.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            Uniform grid with smoothed scalars.

        Notes
        -----
        This filter only supports point data. For inputs with cell data, consider
        re-meshing the cell data as point data with :meth:`~pyvista.ImageDataFilters.cells_to_points`
        or resampling the cell data to point data with :func:`~pyvista.DataSetFilters.cell_data_to_point_data`.

        Examples
        --------
        First, create sample data to smooth. Here, we use
        :func:`pyvista.perlin_noise() <pyvista.core.utilities.features.perlin_noise>`
        to create meaningful data.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> noise = pv.perlin_noise(0.1, (2, 5, 8), (0, 0, 0))
        >>> grid = pv.sample_function(
        ...     noise, [0, 1, 0, 1, 0, 1], dim=(20, 20, 20)
        ... )
        >>> grid.plot(show_scalar_bar=False)

        Next, smooth the sample data.

        >>> smoothed = grid.gaussian_smooth()
        >>> smoothed.plot(show_scalar_bar=False)

        See :ref:`gaussian_smoothing_example` for a full example using this filter.

        """
        alg = _vtk.vtkImageGaussianSmooth()
        alg.SetInputDataObject(self)
        if scalars is None:
            set_default_active_scalars(self)
            field, scalars = self.active_scalars_info
            if field.value == 1:
                raise ValueError('If `scalars` not given, active scalars must be point array.')
        else:
            field = self.get_array_association(scalars, preference='point')
            if field.value == 1:
                raise ValueError('Can only process point data, given `scalars` are cell data.')
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)
        if isinstance(radius_factor, Iterable):
            alg.SetRadiusFactors(radius_factor)
        else:
            alg.SetRadiusFactors(radius_factor, radius_factor, radius_factor)
        if isinstance(std_dev, Iterable):
            alg.SetStandardDeviations(std_dev)
        else:
            alg.SetStandardDeviations(std_dev, std_dev, std_dev)
        _update_alg(alg, progress_bar, 'Performing Gaussian Smoothing')
        return _get_output(alg)

    def median_smooth(
        self,
        kernel_size=(3, 3, 3),
        scalars=None,
        preference='point',
        progress_bar=False,
    ):
        """Smooth data using a median filter.

        The Median filter that replaces each pixel with the median value from a
        rectangular neighborhood around that pixel. Neighborhoods can be no
        more than 3 dimensional. Setting one axis of the neighborhood
        kernelSize to 1 changes the filter into a 2D median.

        See `vtkImageMedian3D
        <https://vtk.org/doc/nightly/html/classvtkImageMedian3D.html#details>`_
        for more details.

        Parameters
        ----------
        kernel_size : sequence[int], default: (3, 3, 3)
            Size of the kernel in each dimension (units of voxels), for example
            ``(x_size, y_size, z_size)``. Default is a 3D median filter. If you
            want to do a 2D median filter, set the size to 1 in the dimension
            you don't want to filter over.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        preference : str, default: "point"
            When scalars is specified, this is the preferred array
            type to search for in the dataset.  Must be either
            ``'point'`` or ``'cell'``.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            Uniform grid with smoothed scalars.

        Warnings
        --------
        Applying this filter to cell data will send the output to a new point
        array with the same name, overwriting any existing point data array
        with the same name.

        Examples
        --------
        First, create sample data to smooth. Here, we use
        :func:`pyvista.perlin_noise() <pyvista.core.utilities.features.perlin_noise>`
        to create meaningful data.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> noise = pv.perlin_noise(0.1, (2, 5, 8), (0, 0, 0))
        >>> grid = pv.sample_function(
        ...     noise, [0, 1, 0, 1, 0, 1], dim=(20, 20, 20)
        ... )
        >>> grid.plot(show_scalar_bar=False)

        Next, smooth the sample data.

        >>> smoothed = grid.median_smooth(kernel_size=(10, 10, 10))
        >>> smoothed.plot(show_scalar_bar=False)

        """
        alg = _vtk.vtkImageMedian3D()
        alg.SetInputDataObject(self)
        if scalars is None:
            set_default_active_scalars(self)
            field, scalars = self.active_scalars_info
        else:
            field = self.get_array_association(scalars, preference=preference)
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)
        alg.SetKernelSize(kernel_size[0], kernel_size[1], kernel_size[2])
        _update_alg(alg, progress_bar, 'Performing Median Smoothing')
        return _get_output(alg)

    def extract_subset(self, voi, rate=(1, 1, 1), boundary=False, progress_bar=False):
        """Select piece (e.g., volume of interest).

        To use this filter set the VOI ivar which are i-j-k min/max indices
        that specify a rectangular region in the data. (Note that these are
        0-offset.) You can also specify a sampling rate to subsample the
        data.

        Typical applications of this filter are to extract a slice from a
        volume for image processing, subsampling large volumes to reduce data
        size, or extracting regions of a volume with interesting data.

        Parameters
        ----------
        voi : sequence[int]
            Length 6 iterable of ints: ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
            These bounds specify the volume of interest in i-j-k min/max
            indices.

        rate : sequence[int], default: (1, 1, 1)
            Length 3 iterable of ints: ``(xrate, yrate, zrate)``.

        boundary : bool, default: False
            Control whether to enforce that the "boundary" of the grid
            is output in the subsampling process. This only has effect
            when the rate in any direction is not equal to 1. When
            this is enabled, the subsampling will always include the
            boundary of the grid even though the sample rate is not an
            even multiple of the grid dimensions. By default this is
            disabled.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            ImageData subset.
        """
        alg = _vtk.vtkExtractVOI()
        alg.SetVOI(voi)
        alg.SetInputDataObject(self)
        alg.SetSampleRate(rate)
        alg.SetIncludeBoundary(boundary)
        _update_alg(alg, progress_bar, 'Extracting Subset')
        result = _get_output(alg)
        # Adjust for the confusing issue with the extents
        #   see https://gitlab.kitware.com/vtk/vtk/-/issues/17938
        fixed = pyvista.ImageData()
        fixed.origin = result.bounds[::2]
        fixed.spacing = result.spacing
        fixed.dimensions = result.dimensions
        fixed.point_data.update(result.point_data)
        fixed.cell_data.update(result.cell_data)
        fixed.field_data.update(result.field_data)
        fixed.copy_meta_from(result, deep=True)
        return fixed

    def image_dilate_erode(
        self,
        dilate_value=1.0,
        erode_value=0.0,
        kernel_size=(3, 3, 3),
        scalars=None,
        progress_bar=False,
    ):
        """Dilates one value and erodes another.

        ``image_dilate_erode`` will dilate one value and erode another. It uses
        an elliptical footprint, and only erodes/dilates on the boundary of the
        two values. The filter is restricted to the X, Y, and Z axes for now.
        It can degenerate to a 2 or 1-dimensional filter by setting the kernel
        size to 1 for a specific axis.

        Parameters
        ----------
        dilate_value : float, default: 1.0
            Dilate value in the dataset.

        erode_value : float, default: 0.0
            Erode value in the dataset.

        kernel_size : sequence[int], default: (3, 3, 3)
            Determines the size of the kernel along the three axes.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            Dataset that has been dilated/eroded on the boundary of the specified scalars.

        Notes
        -----
        This filter only supports point data. For inputs with cell data, consider
        re-meshing the cell data as point data with :meth:`~pyvista.ImageDataFilters.cells_to_points`
        or resampling the cell data to point data with :func:`~pyvista.DataSetFilters.cell_data_to_point_data`.

        Examples
        --------
        Demonstrate image dilate/erode on an example dataset. First, plot
        the example dataset with the active scalars.

        >>> from pyvista import examples
        >>> uni = examples.load_uniform()
        >>> uni.plot()

        Now, plot the image threshold with ``threshold=[400, 600]``. Note how
        values within the threshold are 1 and outside are 0.

        >>> ithresh = uni.image_threshold([400, 600])
        >>> ithresh.plot()

        Note how there is a hole in the thresholded image. Apply a dilation/
        erosion filter with a large kernel to fill that hole in.

        >>> idilate = ithresh.image_dilate_erode(kernel_size=[5, 5, 5])
        >>> idilate.plot()

        """
        alg = _vtk.vtkImageDilateErode3D()
        alg.SetInputDataObject(self)
        if scalars is None:
            set_default_active_scalars(self)
            field, scalars = self.active_scalars_info
            if field.value == 1:
                raise ValueError('If `scalars` not given, active scalars must be point array.')
        else:
            field = self.get_array_association(scalars, preference='point')
            if field.value == 1:
                raise ValueError('Can only process point data, given `scalars` are cell data.')
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)
        alg.SetKernelSize(*kernel_size)
        alg.SetDilateValue(dilate_value)
        alg.SetErodeValue(erode_value)
        _update_alg(alg, progress_bar, 'Performing Dilation and Erosion')
        return _get_output(alg)

    def image_threshold(
        self,
        threshold,
        in_value=1.0,
        out_value=0.0,
        scalars=None,
        preference='point',
        progress_bar=False,
    ):
        """Apply a threshold to scalar values in a uniform grid.

        If a single value is given for threshold, scalar values above or equal
        to the threshold are ``'in'`` and scalar values below the threshold are ``'out'``.
        If two values are given for threshold (sequence) then values equal to
        or between the two values are ``'in'`` and values outside the range are ``'out'``.

        If ``None`` is given for ``in_value``, scalars that are ``'in'`` will not be replaced.
        If ``None`` is given for ``out_value``, scalars that are ``'out'`` will not be replaced.

        Warning: applying this filter to cell data will send the output to a
        new point array with the same name, overwriting any existing point data
        array with the same name.

        Parameters
        ----------
        threshold : float or sequence[float]
            Single value or (min, max) to be used for the data threshold.  If
            a sequence, then length must be 2. Threshold(s) for deciding which
            cells/points are ``'in'`` or ``'out'`` based on scalar data.

        in_value : float, default: 1.0
            Scalars that match the threshold criteria for ``'in'`` will be replaced with this.

        out_value : float, default: 0.0
            Scalars that match the threshold criteria for ``'out'`` will be replaced with this.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        preference : str, default: "point"
            When scalars is specified, this is the preferred array
            type to search for in the dataset.  Must be either
            ``'point'`` or ``'cell'``.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            Dataset with the specified scalars thresholded.

        See Also
        --------
        :meth:`~pyvista.DataSetFilters.threshold`

        Examples
        --------
        Demonstrate image threshold on an example dataset. First, plot
        the example dataset with the active scalars.

        >>> from pyvista import examples
        >>> uni = examples.load_uniform()
        >>> uni.plot()

        Now, plot the image threshold with ``threshold=100``. Note how
        values above the threshold are 1 and below are 0.

        >>> ithresh = uni.image_threshold(100)
        >>> ithresh.plot()

        See :ref:`image_representations_example` for more examples using this filter.

        """
        if scalars is None:
            set_default_active_scalars(self)
            field, scalars = self.active_scalars_info
        else:
            field = self.get_array_association(scalars, preference=preference)

        # For some systems integer scalars won't threshold
        # correctly. Cast to float to be robust.
        cast_dtype = np.issubdtype(
            array_dtype := self.active_scalars.dtype,
            int,
        ) and array_dtype != np.dtype(np.uint8)
        if cast_dtype:
            self[scalars] = self[scalars].astype(float, casting='safe')

        alg = _vtk.vtkImageThreshold()
        alg.SetInputDataObject(self)
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)
        # set the threshold(s) and mode
        threshold_val = np.atleast_1d(threshold)
        if (size := threshold_val.size) not in (1, 2):
            raise ValueError(
                f'Threshold must have one or two values, got {size}.',
            )
        if size == 2:
            alg.ThresholdBetween(threshold_val[0], threshold_val[1])
        else:
            alg.ThresholdByUpper(threshold_val[0])
        # set the replacement values / modes
        if in_value is not None:
            alg.SetReplaceIn(True)
            alg.SetInValue(np.array(in_value).astype(array_dtype))
        else:
            alg.SetReplaceIn(False)
        if out_value is not None:
            alg.SetReplaceOut(True)
            alg.SetOutValue(np.array(out_value).astype(array_dtype))
        else:
            alg.SetReplaceOut(False)
        # run the algorithm
        _update_alg(alg, progress_bar, 'Performing Image Thresholding')
        output = _get_output(alg)
        if cast_dtype:
            self[scalars] = self[scalars].astype(array_dtype)
            output[scalars] = output[scalars].astype(array_dtype)
        return output

    def fft(self, output_scalars_name=None, progress_bar=False):
        """Apply a fast Fourier transform (FFT) to the active scalars.

        The input can be real or complex data, but the output is always
        :attr:`numpy.complex128`. The filter is fastest for images that have
        power of two sizes.

        The filter uses a butterfly diagram for each prime factor of the
        dimension. This makes images with prime number dimensions (i.e. 17x17)
        much slower to compute. FFTs of multidimensional meshes (i.e volumes)
        are decomposed so that each axis executes serially.

        The frequencies of the output assume standard order: along each axis
        first positive frequencies are assumed from 0 to the maximum, then
        negative frequencies are listed from the largest absolute value to
        smallest. This implies that the corners of the grid correspond to low
        frequencies, while the center of the grid corresponds to high
        frequencies.

        Parameters
        ----------
        output_scalars_name : str, optional
            The name of the output scalars. By default, this is the same as the
            active scalars of the dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            :class:`pyvista.ImageData` with applied FFT.

        See Also
        --------
        rfft : The reverse transform.
        low_pass : Low-pass filtering of FFT output.
        high_pass : High-pass filtering of FFT output.

        Examples
        --------
        Apply FFT to an example image.

        >>> from pyvista import examples
        >>> image = examples.download_moonlanding_image()
        >>> fft_image = image.fft()
        >>> fft_image.point_data  # doctest:+SKIP
        pyvista DataSetAttributes
        Association     : POINT
        Active Scalars  : PNGImage
        Active Vectors  : None
        Active Texture  : None
        Active Normals  : None
        Contains arrays :
        PNGImage                complex128 (298620,)          SCALARS

        See :ref:`image_fft_example` for a full example using this filter.

        """
        # check for active scalars, otherwise risk of segfault
        if self.point_data.active_scalars_name is None:
            try:
                set_default_active_scalars(self)
            except MissingDataError:
                raise MissingDataError('FFT filter requires point scalars.') from None

            # possible only cell scalars were made active
            if self.point_data.active_scalars_name is None:
                raise MissingDataError('FFT filter requires point scalars.')

        alg = _vtk.vtkImageFFT()
        alg.SetInputDataObject(self)
        _update_alg(alg, progress_bar, 'Performing Fast Fourier Transform')
        output = _get_output(alg)
        self._change_fft_output_scalars(
            output,
            self.point_data.active_scalars_name,
            output_scalars_name,
        )
        return output

    def rfft(self, output_scalars_name=None, progress_bar=False):
        """Apply a reverse fast Fourier transform (RFFT) to the active scalars.

        The input can be real or complex data, but the output is always
        :attr:`numpy.complex128`. The filter is fastest for images that have power
        of two sizes.

        The filter uses a butterfly diagram for each prime factor of the
        dimension. This makes images with prime number dimensions (i.e. 17x17)
        much slower to compute. FFTs of multidimensional meshes (i.e volumes)
        are decomposed so that each axis executes serially.

        The frequencies of the input assume standard order: along each axis
        first positive frequencies are assumed from 0 to the maximum, then
        negative frequencies are listed from the largest absolute value to
        smallest. This implies that the corners of the grid correspond to low
        frequencies, while the center of the grid corresponds to high
        frequencies.

        Parameters
        ----------
        output_scalars_name : str, optional
            The name of the output scalars. By default, this is the same as the
            active scalars of the dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            :class:`pyvista.ImageData` with the applied reverse FFT.

        See Also
        --------
        fft : The direct transform.
        low_pass : Low-pass filtering of FFT output.
        high_pass : High-pass filtering of FFT output.

        Examples
        --------
        Apply reverse FFT to an example image.

        >>> from pyvista import examples
        >>> image = examples.download_moonlanding_image()
        >>> fft_image = image.fft()
        >>> image_again = fft_image.rfft()
        >>> image_again.point_data  # doctest:+SKIP
        pyvista DataSetAttributes
        Association     : POINT
        Active Scalars  : PNGImage
        Active Vectors  : None
        Active Texture  : None
        Active Normals  : None
        Contains arrays :
            PNGImage                complex128 (298620,)            SCALARS

        See :ref:`image_fft_example` for a full example using this filter.

        """
        self._check_fft_scalars()
        alg = _vtk.vtkImageRFFT()
        alg.SetInputDataObject(self)
        _update_alg(alg, progress_bar, 'Performing Reverse Fast Fourier Transform.')
        output = _get_output(alg)
        self._change_fft_output_scalars(
            output,
            self.point_data.active_scalars_name,
            output_scalars_name,
        )
        return output

    def low_pass(
        self,
        x_cutoff,
        y_cutoff,
        z_cutoff,
        order=1,
        output_scalars_name=None,
        progress_bar=False,
    ):
        """Perform a Butterworth low pass filter in the frequency domain.

        This filter requires that the :class:`ImageData` have a complex point
        scalars, usually generated after the :class:`ImageData` has been
        converted to the frequency domain by a :func:`ImageDataFilters.fft`
        filter.

        A :func:`ImageDataFilters.rfft` filter can be used to convert the
        output back into the spatial domain. This filter attenuates high
        frequency components.  Input and output are complex arrays with
        datatype :attr:`numpy.complex128`.

        The frequencies of the input assume standard order: along each axis
        first positive frequencies are assumed from 0 to the maximum, then
        negative frequencies are listed from the largest absolute value to
        smallest. This implies that the corners of the grid correspond to low
        frequencies, while the center of the grid corresponds to high
        frequencies.

        Parameters
        ----------
        x_cutoff : float
            The cutoff frequency for the x-axis.

        y_cutoff : float
            The cutoff frequency for the y-axis.

        z_cutoff : float
            The cutoff frequency for the z-axis.

        order : int, default: 1
            The order of the cutoff curve. Given from the equation
            ``1 + (cutoff/freq(i, j))**(2*order)``.

        output_scalars_name : str, optional
            The name of the output scalars. By default, this is the same as the
            active scalars of the dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            :class:`pyvista.ImageData` with the applied low pass filter.

        See Also
        --------
        fft : Direct fast Fourier transform.
        rfft : Reverse fast Fourier transform.
        high_pass : High-pass filtering of FFT output.

        Examples
        --------
        See :ref:`image_fft_perlin_example` for a full example using this filter.

        """
        self._check_fft_scalars()
        alg = _vtk.vtkImageButterworthLowPass()
        alg.SetInputDataObject(self)
        alg.SetCutOff(x_cutoff, y_cutoff, z_cutoff)
        alg.SetOrder(order)
        _update_alg(alg, progress_bar, 'Performing Low Pass Filter')
        output = _get_output(alg)
        self._change_fft_output_scalars(
            output,
            self.point_data.active_scalars_name,
            output_scalars_name,
        )
        return output

    def high_pass(
        self,
        x_cutoff,
        y_cutoff,
        z_cutoff,
        order=1,
        output_scalars_name=None,
        progress_bar=False,
    ):
        """Perform a Butterworth high pass filter in the frequency domain.

        This filter requires that the :class:`ImageData` have a complex point
        scalars, usually generated after the :class:`ImageData` has been
        converted to the frequency domain by a :func:`ImageDataFilters.fft`
        filter.

        A :func:`ImageDataFilters.rfft` filter can be used to convert the
        output back into the spatial domain. This filter attenuates low
        frequency components.  Input and output are complex arrays with
        datatype :attr:`numpy.complex128`.

        The frequencies of the input assume standard order: along each axis
        first positive frequencies are assumed from 0 to the maximum, then
        negative frequencies are listed from the largest absolute value to
        smallest. This implies that the corners of the grid correspond to low
        frequencies, while the center of the grid corresponds to high
        frequencies.

        Parameters
        ----------
        x_cutoff : float
            The cutoff frequency for the x-axis.

        y_cutoff : float
            The cutoff frequency for the y-axis.

        z_cutoff : float
            The cutoff frequency for the z-axis.

        order : int, default: 1
            The order of the cutoff curve. Given from the equation
            ``1/(1 + (cutoff/freq(i, j))**(2*order))``.

        output_scalars_name : str, optional
            The name of the output scalars. By default, this is the same as the
            active scalars of the dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            :class:`pyvista.ImageData` with the applied high pass filter.

        See Also
        --------
        fft : Direct fast Fourier transform.
        rfft : Reverse fast Fourier transform.
        low_pass : Low-pass filtering of FFT output.

        Examples
        --------
        See :ref:`image_fft_perlin_example` for a full example using this filter.

        """
        self._check_fft_scalars()
        alg = _vtk.vtkImageButterworthHighPass()
        alg.SetInputDataObject(self)
        alg.SetCutOff(x_cutoff, y_cutoff, z_cutoff)
        alg.SetOrder(order)
        _update_alg(alg, progress_bar, 'Performing High Pass Filter')
        output = _get_output(alg)
        self._change_fft_output_scalars(
            output,
            self.point_data.active_scalars_name,
            output_scalars_name,
        )
        return output

    def _change_fft_output_scalars(self, dataset, orig_name, out_name):
        """Modify the name and dtype of the output scalars for an FFT filter."""
        name = orig_name if out_name is None else out_name
        pdata = dataset.point_data
        if pdata.active_scalars_name != name:
            pdata[name] = pdata.pop(pdata.active_scalars_name)

        # always view the datatype of the point_data as complex128
        dataset._association_complex_names['POINT'].add(name)

    def _check_fft_scalars(self):
        """Check for complex active scalars.

        This is necessary for rfft, low_pass, and high_pass filters.

        """
        # check for complex active point scalars, otherwise the risk of segfault
        if self.point_data.active_scalars_name is None:
            possible_scalars = self.point_data.keys()
            if len(possible_scalars) == 1:
                self.set_active_scalars(possible_scalars[0], preference='point')
            elif len(possible_scalars) > 1:
                raise AmbiguousDataError(
                    'There are multiple point scalars available. Set one to be '
                    'active with `point_data.active_scalars_name = `',
                )
            else:
                raise MissingDataError('FFT filters require point scalars.')

        if not np.issubdtype(self.point_data.active_scalars.dtype, np.complexfloating):
            raise ValueError(
                'Active scalars must be complex data for this filter, represented '
                'as an array with a datatype of `numpy.complex64` or '
                '`numpy.complex128`.',
            )

    def _flip_uniform(self, axis) -> pyvista.ImageData:
        """Flip the uniform grid along a specified axis and return a uniform grid.

        This varies from :func:`DataSet.flip_x` because it returns a ImageData.

        """
        alg = _vtk.vtkImageFlip()
        alg.SetInputData(self)
        alg.SetFilteredAxes(axis)
        alg.Update()
        return cast(pyvista.ImageData, wrap(alg.GetOutput()))

    def contour_labeled(
        self,
        n_labels: int | None = None,
        smoothing: bool = False,
        smoothing_num_iterations: int = 50,
        smoothing_relaxation_factor: float = 0.5,
        smoothing_constraint_distance: float = 1,
        output_mesh_type: Literal['quads', 'triangles'] = 'quads',
        output_style: Literal['default', 'boundary'] = 'default',
        scalars: str | None = None,
        progress_bar: bool = False,
    ) -> pyvista.PolyData:
        """Generate labeled contours from 3D label maps.

        SurfaceNets algorithm is used to extract contours preserving sharp
        boundaries for the selected labels from the label maps.
        Optionally, the boundaries can be smoothened to reduce the staircase
        appearance in case of low resolution input label maps.

        This filter requires that the :class:`ImageData` has integer point
        scalars, such as multi-label maps generated from image segmentation.

        .. note::
           Requires ``vtk>=9.3.0``.

        Parameters
        ----------
        n_labels : int, optional
            Number of labels to be extracted (all are extracted if None is given).

        smoothing : bool, default: False
            Apply smoothing to the meshes.

        smoothing_num_iterations : int, default: 50
            Number of smoothing iterations.

        smoothing_relaxation_factor : float, default: 0.5
            Relaxation factor of the smoothing.

        smoothing_constraint_distance : float, default: 1
            Constraint distance of the smoothing.

        output_mesh_type : str, default: 'quads'
            Type of the output mesh. Must be either ``'quads'``, or ``'triangles'``.

        output_style : str, default: 'default'
            Style of the output mesh. Must be either ``'default'`` or ``'boundary'``.
            When ``'default'`` is specified, the filter produces a mesh with both
            interior and exterior polygons. When ``'boundary'`` is selected, only
            polygons on the border with the background are produced (without interior
            polygons). Note that style ``'selected'`` is currently not implemented.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            :class:`pyvista.PolyData` Labeled mesh with the segments labeled.

        References
        ----------
        Sarah F. Frisken, SurfaceNets for Multi-Label Segmentations with Preservation
        of Sharp Boundaries, Journal of Computer Graphics Techniques (JCGT), vol. 11,
        no. 1, 34-54, 2022. Available online http://jcgt.org/published/0011/01/03/

        https://www.kitware.com/really-fast-isocontouring/

        Examples
        --------
        See :ref:`contouring_example` for a full example using this filter.

        See Also
        --------
        pyvista.DataSetFilters.contour
            Generalized contouring method which uses MarchingCubes or FlyingEdges.

        pyvista.DataSetFilters.pack_labels
            Function used internally by SurfaceNets to generate contiguous label data.

        """
        if not hasattr(_vtk, 'vtkSurfaceNets3D'):  # pragma: no cover
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError('Surface nets 3D require VTK 9.3.0 or newer.')

        alg = _vtk.vtkSurfaceNets3D()
        if scalars is None:
            set_default_active_scalars(self)  # type: ignore[arg-type]
            field, scalars = self.active_scalars_info  # type: ignore[attr-defined]
            if field != FieldAssociation.POINT:
                raise ValueError('If `scalars` not given, active scalars must be point array.')
        else:
            field = self.get_array_association(scalars, preference='point')  # type: ignore[attr-defined]
            if field != FieldAssociation.POINT:
                raise ValueError(
                    f'Can only process point data, given `scalars` are {field.name.lower()} data.',
                )
        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)
        alg.SetInputData(self)
        if n_labels is not None:
            alg.GenerateLabels(n_labels, 1, n_labels)
        if output_mesh_type == 'quads':
            alg.SetOutputMeshTypeToQuads()
        elif output_mesh_type == 'triangles':
            alg.SetOutputMeshTypeToTriangles()
        else:
            raise ValueError(
                f'Invalid output mesh type "{output_mesh_type}", use "quads" or "triangles"',
            )
        if output_style == 'default':
            alg.SetOutputStyleToDefault()
        elif output_style == 'boundary':
            alg.SetOutputStyleToBoundary()
        elif output_style == 'selected':
            raise NotImplementedError(f'Output style "{output_style}" is not implemented')
        else:
            raise ValueError(f'Invalid output style "{output_style}", use "default" or "boundary"')
        if smoothing:
            alg.SmoothingOn()
            alg.GetSmoother().SetNumberOfIterations(smoothing_num_iterations)
            alg.GetSmoother().SetRelaxationFactor(smoothing_relaxation_factor)
            alg.GetSmoother().SetConstraintDistance(smoothing_constraint_distance)
        else:
            alg.SmoothingOff()
        # Suppress improperly used INFO for debugging messages in vtkSurfaceNets3D
        verbosity = _vtk.vtkLogger.GetCurrentVerbosityCutoff()
        _vtk.vtkLogger.SetStderrVerbosity(_vtk.vtkLogger.VERBOSITY_OFF)
        _update_alg(alg, progress_bar, 'Performing Labeled Surface Extraction')
        # Restore the original vtkLogger verbosity level
        _vtk.vtkLogger.SetStderrVerbosity(verbosity)
        return cast(pyvista.PolyData, wrap(alg.GetOutput()))

    def points_to_cells(self, scalars: str | None = None, *, copy: bool = True):
        """Re-mesh image data from a point-based to a cell-based representation.

        This filter changes how image data is represented. Data represented as points
        at the input is re-meshed into an alternative representation as cells at the
        output. Only the :class:`~pyvista.ImageData` container is modified so that
        the number of input points equals the number of output cells. The re-meshing is
        otherwise lossless in the sense that point data at the input is passed through
        unmodified and stored as cell data at the output. Any cell data at the input is
        ignored and is not used by this filter.

        To change the image data's representation, the input points are used to
        represent the centers of the output cells. This has the effect of "growing" the
        input image dimensions by one along each axis (i.e. half the cell width on each
        side). For example, an image with 100 points and 99 cells along an axis at the
        input will have 101 points and 100 cells at the output. If the input has 1mm
        spacing, the axis size will also increase from 99mm to 100mm.

        Since filters may be inherently cell-based (e.g. some :class:`~pyvista.DataSetFilters`)
        or may operate on point data exclusively (e.g. most :class:`~pyvista.ImageDataFilters`),
        re-meshing enables the same data to be used with either kind of filter while
        ensuring the input data to those filters has the appropriate representation.
        This filter is also useful when plotting image data to achieve a desired visual
        effect, such as plotting images as voxel cells instead of as points.

        .. versionadded:: 0.44.0

        See Also
        --------
        cells_to_points
            Inverse of this filter to represent cells as points.
        :meth:`~pyvista.DataSetFilters.point_data_to_cell_data`
            Resample point data as cell data without modifying the container.
        :meth:`~pyvista.DataSetFilters.cell_data_to_point_data`
            Resample cell data as point data without modifying the container.

        Parameters
        ----------
        scalars : str, optional
            Name of point data scalars to pass through to the output as cell data. Use
            this parameter to restrict the output to only include the specified array.
            By default, all point data arrays at the input are passed through as cell
            data at the output.

        copy : bool, default: True
            Copy the input point data before associating it with the output cell data.
            If ``False``, the input and output will both refer to the same data array(s).

        Returns
        -------
        pyvista.ImageData
            Image with a cell-based representation.

        Examples
        --------
        Load an image with point data.

        >>> from pyvista import examples
        >>> image = examples.load_uniform()

        Show the current properties and point arrays of the image.

        >>> image
        ImageData (...)
          N Cells:      729
          N Points:     1000
          X Bounds:     0.000e+00, 9.000e+00
          Y Bounds:     0.000e+00, 9.000e+00
          Z Bounds:     0.000e+00, 9.000e+00
          Dimensions:   10, 10, 10
          Spacing:      1.000e+00, 1.000e+00, 1.000e+00
          N Arrays:     2

        >>> image.point_data.keys()
        ['Spatial Point Data']

        Re-mesh the points and point data as cells and cell data.

        >>> cells_image = image.points_to_cells()

        Show the properties and cell arrays of the re-meshed image.

        >>> cells_image
        ImageData (...)
          N Cells:      1000
          N Points:     1331
          X Bounds:     -5.000e-01, 9.500e+00
          Y Bounds:     -5.000e-01, 9.500e+00
          Z Bounds:     -5.000e-01, 9.500e+00
          Dimensions:   11, 11, 11
          Spacing:      1.000e+00, 1.000e+00, 1.000e+00
          N Arrays:     1

        >>> cells_image.cell_data.keys()
        ['Spatial Point Data']

        Observe that:

        - The input point array is now a cell array
        - The output has one less array (the input cell data is ignored)
        - The dimensions have increased by one
        - The bounds have increased by half the spacing
        - The output N Cells equals the input N Points

        See :ref:`image_representations_example` for more examples using this filter.

        """
        if scalars is not None:
            field = self.get_array_association(scalars, preference='point')  # type: ignore[attr-defined]
            if field != FieldAssociation.POINT:
                raise ValueError(
                    f"Scalars '{scalars}' must be associated with point data. Got {field.name.lower()} data instead.",
                )
        return self._remesh_points_cells(points_to_cells=True, scalars=scalars, copy=copy)

    def cells_to_points(self, scalars: str | None = None, *, copy: bool = True):
        """Re-mesh image data from a cell-based to a point-based representation.

        This filter changes how image data is represented. Data represented as cells
        at the input is re-meshed into an alternative representation as points at the
        output. Only the :class:`~pyvista.ImageData` container is modified so that
        the number of input cells equals the number of output points. The re-meshing is
        otherwise lossless in the sense that cell data at the input is passed through
        unmodified and stored as point data at the output. Any point data at the input is
        ignored and is not used by this filter.

        To change the image data's representation, the input cell centers are used to
        represent the output points. This has the effect of "shrinking" the
        input image dimensions by one along each axis (i.e. half the cell width on each
        side). For example, an image with 101 points and 100 cells along an axis at the
        input will have 100 points and 99 cells at the output. If the input has 1mm
        spacing, the axis size will also decrease from 100mm to 99mm.

        Since filters may be inherently cell-based (e.g. some :class:`~pyvista.DataSetFilters`)
        or may operate on point data exclusively (e.g. most :class:`~pyvista.ImageDataFilters`),
        re-meshing enables the same data to be used with either kind of filter while
        ensuring the input data to those filters has the appropriate representation.
        This filter is also useful when plotting image data to achieve a desired visual
        effect, such as plotting images as points instead of as voxel cells.

        .. versionadded:: 0.44.0

        See Also
        --------
        points_to_cells
            Inverse of this filter to represent points as cells.
        :meth:`~pyvista.DataSetFilters.cell_data_to_point_data`
            Resample cell data as point data without modifying the container.
        :meth:`~pyvista.DataSetFilters.point_data_to_cell_data`
            Resample point data as cell data without modifying the container.

        Parameters
        ----------
        scalars : str, optional
            Name of cell data scalars to pass through to the output as point data. Use
            this parameter to restrict the output to only include the specified array.
            By default, all cell data arrays at the input are passed through as point
            data at the output.

        copy : bool, default: True
            Copy the input cell data before associating it with the output point data.
            If ``False``, the input and output will both refer to the same data array(s).

        Returns
        -------
        pyvista.ImageData
            Image with a point-based representation.

        Examples
        --------
        Load an image with cell data.

        >>> from pyvista import examples
        >>> image = examples.load_uniform()

        Show the current properties and cell arrays of the image.

        >>> image
        ImageData (...)
          N Cells:      729
          N Points:     1000
          X Bounds:     0.000e+00, 9.000e+00
          Y Bounds:     0.000e+00, 9.000e+00
          Z Bounds:     0.000e+00, 9.000e+00
          Dimensions:   10, 10, 10
          Spacing:      1.000e+00, 1.000e+00, 1.000e+00
          N Arrays:     2

        >>> image.cell_data.keys()
        ['Spatial Cell Data']

        Re-mesh the cells and cell data as points and point data.

        >>> points_image = image.cells_to_points()

        Show the properties and point arrays of the re-meshed image.

        >>> points_image
        ImageData (...)
          N Cells:      512
          N Points:     729
          X Bounds:     5.000e-01, 8.500e+00
          Y Bounds:     5.000e-01, 8.500e+00
          Z Bounds:     5.000e-01, 8.500e+00
          Dimensions:   9, 9, 9
          Spacing:      1.000e+00, 1.000e+00, 1.000e+00
          N Arrays:     1

        >>> points_image.point_data.keys()
        ['Spatial Cell Data']

        Observe that:

        - The input cell array is now a point array
        - The output has one less array (the input point data is ignored)
        - The dimensions have decreased by one
        - The bounds have decreased by half the spacing
        - The output N Points equals the input N Cells

        See :ref:`image_representations_example` for more examples using this filter.

        """
        if scalars is not None:
            field = self.get_array_association(scalars, preference='cell')  # type: ignore[attr-defined]
            if field != FieldAssociation.CELL:
                raise ValueError(
                    f"Scalars '{scalars}' must be associated with cell data. Got {field.name.lower()} data instead.",
                )
        return self._remesh_points_cells(points_to_cells=False, scalars=scalars, copy=copy)

    def _remesh_points_cells(self, points_to_cells: bool, scalars: str | None, copy: bool):
        """Re-mesh points to cells or vice-versa.

        The active cell or point scalars at the input will be set as active point or
        cell scalars at the output, respectively.

        Parameters
        ----------
        points_to_cells : bool
            Set to ``True`` to re-mesh points to cells.
            Set to ``False`` to re-mesh cells to points.

        scalars : str
            If set, only these scalars are passed through.

        copy : bool
            Copy the input data before associating it with the output data.

        Returns
        -------
        pyvista.ImageData
            Re-meshed image.

        """

        def _get_output_scalars(preference):
            active_scalars = self.active_scalars_name
            if active_scalars:
                field = self.get_array_association(
                    active_scalars,
                    preference=preference,
                )
                active_scalars = active_scalars if field.name.lower() == preference else None
            return active_scalars

        point_data = self.point_data  # type: ignore[attr-defined]
        cell_data = self.cell_data  # type: ignore[attr-defined]

        # Get data to use and operations to perform for the conversion
        new_image = pyvista.ImageData()
        if points_to_cells:
            output_scalars = scalars if scalars else _get_output_scalars('point')
            # Enlarge image so points become cell centers
            origin_operator = operator.sub
            dims_operator = operator.add  # Increase dimensions
            old_data = point_data
            new_data = new_image.cell_data
        else:  # cells_to_points
            output_scalars = scalars if scalars else _get_output_scalars('cell')
            # Shrink image so cell centers become points
            origin_operator = operator.add
            dims_operator = operator.sub  # Decrease dimensions
            old_data = cell_data
            new_data = new_image.point_data

        dims = np.array(self.dimensions)  # type: ignore[attr-defined]
        dims_mask = dims > 1  # Only operate on non-singleton dimensions
        new_image.origin = origin_operator(
            self.origin,  # type: ignore[attr-defined]
            (np.array(self.spacing) / 2) * dims_mask,  # type: ignore[attr-defined]
        )
        new_image.dimensions = dims_operator(
            dims,
            dims_mask,
        )
        new_image.spacing = self.spacing  # type: ignore[attr-defined]
        new_image.SetDirectionMatrix(self.GetDirectionMatrix())  # type: ignore[attr-defined]

        # Copy field data
        new_image.field_data.update(self.field_data)  # type: ignore[attr-defined]

        # Copy old data (point or cell) to new data (cell or point)
        array_names = [scalars] if scalars else old_data.keys()
        for array_name in array_names:
            new_data[array_name] = old_data[array_name].copy() if copy else old_data[array_name]

        new_image.set_active_scalars(output_scalars)
        return new_image

    def pad_image(
        self,
        pad_value: float | VectorLike[float] | Literal['wrap', 'mirror'] = 0.0,
        *,
        pad_size: int | VectorLike[int] = 1,
        pad_singleton_dims: bool = False,
        scalars: str | None = None,
        pad_all_scalars: bool = False,
        progress_bar=False,
    ) -> pyvista.ImageData:
        """Enlarge an image by padding its boundaries with new points.

        .. versionadded:: 0.44.0

        Padded points may be mirrored, wrapped, or filled with a constant value. By
        default, all boundaries of the image are padded with a single constant value.

        This filter is designed to work with 1D, 2D, or 3D image data and will only pad
        non-singleton dimensions unless otherwise specified.

        Parameters
        ----------
        pad_value : float | sequence[float] | 'mirror' | 'wrap', default : 0.0
            Padding value(s) given to new points outside the original image extent.
            Specify:

            - a number: New points are filled with the specified constant value.
            - a vector: New points are filled with the specified multi-component vector.
            - ``'wrap'``: New points are filled by wrapping around the padding axis.
            - ``'mirror'``: New points are filled by mirroring the padding axis.

        pad_size : int | sequence[int], default : 1
            Number of points to add to the image boundaries. Specify:

            - A single value to pad all boundaries equally.
            - Two values, one for each ``(X, Y)`` axis, to apply symmetric padding to
              each axis independently.
            - Three values, one for each ``(X, Y, Z)`` axis, to apply symmetric padding
              to each axis independently.
            - Four values, one for each ``(-X, +X, -Y, +Y)`` boundary, to apply
              padding to each boundary independently.
            - Six values, one for each ``(-X, +X, -Y, +Y, -Z, +Z)`` boundary, to apply
              padding to each boundary independently.

            .. note::
                The pad size for singleton dimensions is set to ``0`` by default, even
                if non-zero pad sizes are specified for these axes with this parameter.
                Set ``pad_singleton_dims`` to ``True`` to override this behavior and
                enable padding any or all dimensions.

        pad_singleton_dims : bool, default : False
            Control whether to pad singleton dimensions. By default, only non-singleton
            dimensions are padded, which means that 1D or 2D inputs will remain 1D or
            2D after padding. Set this to ``True`` to enable padding any or all
            dimensions.

        scalars : str, optional
            Name of scalars to pad. Defaults to currently active scalars. Unless
            ``pad_all_scalars`` is ``True``, only the specified ``scalars`` are included
            in the output.

        pad_all_scalars : bool, default: False
            Pad all point data scalars and include them in the output. This is useful
            for padding images with multiple scalars. If ``False``, only the specified
            ``scalars`` are padded.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.ImageData
            Padded image.

        Examples
        --------
        Pad a grayscale image with a 100-pixel wide border. The padding is black
        (i.e. has a value of ``0``) by default.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>>
        >>> gray_image = examples.download_moonlanding_image()
        >>> gray_image.dimensions
        (630, 474, 1)
        >>> padded = gray_image.pad_image(pad_size=100)
        >>> padded.dimensions
        (830, 674, 1)

        Plot the image. To show grayscale images correctly, we define a custom plotting
        method.

        >>> def grayscale_image_plotter(image):
        ...     import vtk
        ...
        ...     actor = vtk.vtkImageActor()
        ...     actor.GetMapper().SetInputData(image)
        ...     actor.GetProperty().SetInterpolationTypeToNearest()
        ...     plot = pv.Plotter()
        ...     plot.add_actor(actor)
        ...     plot.view_xy()
        ...     plot.camera.tight()
        ...     return plot
        ...
        >>>
        >>> plot = grayscale_image_plotter(padded)
        >>> plot.show()

        Pad only the x-axis with a white border.

        >>> padded = gray_image.pad_image(pad_value=255, pad_size=(200, 0))
        >>> plot = grayscale_image_plotter(padded)
        >>> plot.show()

        Pad with wrapping.

        >>> padded = gray_image.pad_image('wrap', pad_size=100)
        >>> plot = grayscale_image_plotter(padded)
        >>> plot.show()

        Pad with mirroring.

        >>> padded = gray_image.pad_image('mirror', pad_size=100)
        >>> plot = grayscale_image_plotter(padded)
        >>> plot.show()

        Pad a color image using multi-component color vectors. Here, RGBA values are
        used.

        >>> color_image = examples.load_logo()
        >>> red = (255, 0, 0, 255)  # RGBA
        >>> padded = color_image.pad_image(pad_value=red, pad_size=200)
        >>>
        >>> plot_kwargs = dict(
        ...     cpos='xy', zoom='tight', rgb=True, show_axes=False
        ... )
        >>> padded.plot(**plot_kwargs)

        Pad each edge of the image separately with a different color.

        >>> orange = pv.Color('orange').int_rgba
        >>> purple = pv.Color('purple').int_rgba
        >>> blue = pv.Color('blue').int_rgba
        >>> green = pv.Color('green').int_rgba
        >>>
        >>> padded = color_image.pad_image(orange, pad_size=(100, 0, 0, 0))
        >>> padded = padded.pad_image(purple, pad_size=(0, 100, 0, 0))
        >>> padded = padded.pad_image(blue, pad_size=(0, 0, 100, 0))
        >>> padded = padded.pad_image(green, pad_size=(0, 0, 0, 100))
        >>>
        >>> padded.plot(**plot_kwargs)

        """

        def _get_num_components(array_):
            return 1 if array_.ndim == 1 else array_.shape[1]

        # Validate scalars
        if scalars is None:
            set_default_active_scalars(self)  # type: ignore[arg-type]
            field, scalars = self.active_scalars_info  # type: ignore[attr-defined]
        else:
            field = self.get_array_association(scalars, preference='point')  # type: ignore[attr-defined]
        if field != FieldAssociation.POINT:
            raise ValueError(
                f"Scalars '{scalars}' must be associated with point data. Got {field.name.lower()} data instead.",
            )

        # Process pad size to create a length-6 tuple (-X,+X,-Y,+Y,-Z,+Z)
        pad_sz = np.atleast_1d(pad_size)
        if not pad_sz.ndim == 1:
            raise ValueError(f'Pad size must be one dimensional. Got {pad_sz.ndim} dimensions.')
        if not np.issubdtype(pad_sz.dtype, np.integer):
            raise TypeError(f'Pad size must be integers. Got dtype {pad_sz.dtype.name}.')
        if np.any(pad_sz < 0):
            raise ValueError(f'Pad size cannot be negative. Got {pad_size}.')

        length = len(pad_sz)
        if length == 1:
            all_pad_sizes = np.broadcast_to(pad_sz, (6,)).copy()
        elif length == 2:
            all_pad_sizes = np.array(
                (pad_sz[0], pad_sz[0], pad_sz[1], pad_sz[1], 0, 0),
            )
        elif length == 3:
            all_pad_sizes = np.array(
                (pad_sz[0], pad_sz[0], pad_sz[1], pad_sz[1], pad_sz[2], pad_sz[2]),
            )
        elif length == 4:
            all_pad_sizes = np.array(
                (pad_sz[0], pad_sz[1], pad_sz[2], pad_sz[3], 0, 0),
            )
        elif length == 6:
            all_pad_sizes = pad_sz
        else:
            raise ValueError(f"Pad size must have 1, 2, 3, 4, or 6 values, got {length} instead.")

        if not pad_singleton_dims:
            # Set pad size to zero for singleton dimensions (e.g. 2D cases)
            dims = self.dimensions  # type: ignore[attr-defined]
            dim_pairs = (dims[0], dims[0], dims[1], dims[1], dims[2], dims[2])
            is_singleton = np.asarray(dim_pairs) == 1
            all_pad_sizes[is_singleton] = 0

        # Define new extents after padding
        pad_xn, pad_xp, pad_yn, pad_yp, pad_zn, pad_zp = all_pad_sizes
        ext_xn, ext_xp, ext_yn, ext_yp, ext_zn, ext_zp = self.GetExtent()  # type: ignore[attr-defined]

        padded_extents = (
            ext_xn - pad_xn,  # minX
            ext_xp + pad_xp,  # maxX
            ext_yn - pad_yn,  # minY
            ext_yp + pad_yp,  # maxY
            ext_zn - pad_zn,  # minZ
            ext_zp + pad_zp,  # maxZ
        )

        # Validate pad value
        pad_multi_component = None  # Flag for multi-component constants
        error_msg = (
            f"Invalid pad value {pad_value}. Must be 'mirror' or 'wrap', or a "
            f"number/component vector for constant padding."
        )
        if isinstance(pad_value, str):
            if pad_value == 'mirror':
                alg = _vtk.vtkImageMirrorPad()
            elif pad_value == 'wrap':
                alg = _vtk.vtkImageWrapPad()
            else:
                raise ValueError(error_msg)
        else:
            val = np.atleast_1d(pad_value)
            num_input_components = _get_num_components(self.active_scalars)  # type: ignore[attr-defined]
            if not (
                val.ndim == 1
                and (np.issubdtype(val.dtype, np.floating) or np.issubdtype(val.dtype, np.integer))
            ):
                raise ValueError(error_msg)
            if (num_value_components := len(val)) not in [1, num_input_components]:
                raise ValueError(
                    f"Number of components ({num_value_components}) in pad value {pad_value} must "
                    f"match the number components ({num_input_components}) in array '{scalars}'.",
                )
            if num_input_components > 1:
                pad_multi_component = True
                data = self.point_data  # type: ignore[attr-defined]
                array_names = data.keys() if pad_all_scalars else [scalars]
                for array_name in array_names:
                    array = data[array_name]
                    if not np.array_equal(val, val.astype(array.dtype)):
                        raise TypeError(
                            f"Pad value {pad_value} with dtype '{val.dtype.name}' is not compatible with dtype '{array.dtype}' of array {array_name}.",
                        )
                    if (
                        not (n_comp := _get_num_components(data[array_name]))
                        == num_input_components
                    ):
                        raise ValueError(
                            f"Cannot pad array '{array_name}' with value {pad_value}. "
                            f"Number of components ({n_comp}) in '{array_name}' must match "
                            f"the number of components ({num_value_components}) in value."
                            f"\nTry setting `pad_all_scalars=False` or update the array.",
                        )
            else:
                pad_multi_component = False
            alg = _vtk.vtkImageConstantPad()

        alg.SetInputDataObject(self)
        alg.SetOutputWholeExtent(*padded_extents)

        def _get_padded_output(scalars_):
            """Update the active scalars and get the output.

            Includes special handling for padding with multi-component values.
            """

            def _update_and_get_output():
                _update_alg(alg, progress_bar, 'Padding image')
                return _get_output(alg)

            # Set scalars since the filter only operates on the active scalars
            self.set_active_scalars(scalars_, preference='point')
            if pad_multi_component is None:
                return _update_and_get_output()
            else:
                # Constant padding
                alg.SetConstant(val[0])
                output = _update_and_get_output()
                if pad_multi_component is False:
                    # Single component padding
                    return output
                else:  # Mulit-component padding
                    # The constant pad filter only pads with a single value.
                    # We need to apply the filter multiple times for each component.
                    output_scalars = output.active_scalars
                    num_output_components = _get_num_components(output_scalars)
                    for component in range(1, num_output_components):
                        alg.SetConstant(val[component])
                        output_scalars[:, component] = _update_and_get_output()[scalars_][
                            :,
                            component,
                        ]
                    output.point_data[scalars_] = output_scalars
                    return output

        output = _get_padded_output(scalars)

        # This filter pads only the active scalars, other arrays are returned empty.
        # We need to pad those other arrays or remove them from the output.
        for point_array in self.point_data:  # type: ignore[attr-defined]
            if point_array != scalars:
                if pad_all_scalars:
                    output[point_array] = _get_padded_output(point_array)[point_array]
                else:
                    output.point_data.remove(point_array)
        for cell_array in (data := output.cell_data):
            data.remove(cell_array)

        # Restore active scalars
        self.set_active_scalars(scalars, preference='point')  # type: ignore[attr-defined]
        return output
