"""Filters with a class to manage filters/algorithms for uniform grid datasets."""

from __future__ import annotations

import collections.abc
from typing import Literal, Optional, Union, cast

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import AmbiguousDataError, DeprecationError, MissingDataError
from pyvista.core.filters import _get_output, _update_alg
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.core.utilities.arrays import FieldAssociation, set_default_active_scalars
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import abstract_class


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
        This filter only supports point data. Consider converting any cell data
        to point data using the :func:`cell_data_to_point_data()
        <pyvista.DataSetFilters.cell_data_to_point_data>` filter to convert any
        cell data to point data.

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
        if isinstance(radius_factor, collections.abc.Iterable):
            alg.SetRadiusFactors(radius_factor)
        else:
            alg.SetRadiusFactors(radius_factor, radius_factor, radius_factor)
        if isinstance(std_dev, collections.abc.Iterable):
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
        This filter only supports point data. Consider converting any cell data
        to point data using the :func:`cell_data_to_point_data()
        <pyvista.DataSetFilters.cell_data_to_point_data>` filter to convert ny
        cell data to point data.

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

        """
        alg = _vtk.vtkImageThreshold()
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
        # set the threshold(s) and mode
        if isinstance(threshold, (np.ndarray, collections.abc.Sequence)):
            if len(threshold) != 2:
                raise ValueError(
                    f'Threshold must be length one for a float value or two for min/max; not ({threshold}).',
                )
            alg.ThresholdBetween(threshold[0], threshold[1])
        elif isinstance(threshold, collections.abc.Iterable):
            raise TypeError('Threshold must either be a single scalar or a sequence.')
        else:
            alg.ThresholdByUpper(threshold)
        # set the replacement values / modes
        if in_value is not None:
            alg.SetReplaceIn(True)
            alg.SetInValue(in_value)
        else:
            alg.SetReplaceIn(False)
        if out_value is not None:
            alg.SetReplaceOut(True)
            alg.SetOutValue(out_value)
        else:
            alg.SetReplaceOut(False)
        # run the algorithm
        _update_alg(alg, progress_bar, 'Performing Image Thresholding')
        return _get_output(alg)

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
            The cutoff frequency for the x axis.

        y_cutoff : float
            The cutoff frequency for the y axis.

        z_cutoff : float
            The cutoff frequency for the z axis.

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
            The cutoff frequency for the x axis.

        y_cutoff : float
            The cutoff frequency for the y axis.

        z_cutoff : float
            The cutoff frequency for the z axis.

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
        n_labels: Optional[int] = None,
        smoothing: bool = False,
        smoothing_num_iterations: int = 50,
        smoothing_relaxation_factor: float = 0.5,
        smoothing_constraint_distance: float = 1,
        output_mesh_type: Literal['quads', 'triangles'] = 'quads',
        output_style: Literal['default', 'boundary'] = 'default',
        scalars: Optional[str] = None,
        progress_bar: bool = False,
    ) -> pyvista.PolyData:
        """Generate labeled contours from 3D label maps.

        .. warning::
            This filter produces unexpected results and is deprecated.
            Use :meth:`~pyvista.ImageDataFilters.contour_labels` instead.
            See https://github.com/pyvista/pyvista/issues/5981 for details.

            To replicate the default behavior from this filter, call `contour_labels`
            with the following arguments:

            .. code::

                n_labels = range(N)
                contour_labels(
                    select_outputs=n_labels,  #   -replaces old 'n_labels' param
                    internal_boundaries=False,  # -replaces old 'output_style' param
                    smoothing=False,  #           -smoothing is now on by default
                    output_mesh_type='quads',  #  -output type is no longer fixed to 'quads'
                    surface_labels=False,  #      -new default 'SurfaceLabels' array
                    boundary_labels=True,  #      -old 'BoundaryLabels' array is removed by default
                )

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
        raise DeprecationError(
            "This filter produces unexpected results and is deprecated. Use `contour_labels` instead."
            "\nSee https://github.com/pyvista/pyvista/issues/5981 for details.",
        )

        if not hasattr(_vtk, 'vtkSurfaceNets3D'):  # pragma: no cover
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError('Surface nets 3D require VTK 9.3.0 or newer.')

        alg = _vtk.vtkSurfaceNets3D()
        if scalars is None:
            set_default_active_scalars(self)
            field, scalars = self.active_scalars_info
            if field != FieldAssociation.POINT:
                raise ValueError('If `scalars` not given, active scalars must be point array.')
        else:
            field = self.get_array_association(scalars, preference='point')
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

    def contour_labels(
        self,
        *,
        select_inputs: Optional[Union[int, collections.abc.Sequence[int]]] = None,
        select_outputs: Optional[Union[int, collections.abc.Sequence[int]]] = None,
        internal_boundaries: bool = True,
        image_boundaries: bool = False,
        independent_regions: bool = True,
        output_mesh_type: Optional[Literal['quads', 'triangles']] = None,
        smoothing: bool = True,
        smoothing_num_iterations: int = 16,
        smoothing_relaxation_factor: float = 0.5,
        smoothing_constraint_distance: Optional[float] = None,
        smoothing_constraint_scale: float = 1.0,
        surface_labels: bool = True,
        boundary_labels: bool = False,
        scalars: Optional[str] = None,
        progress_bar: bool = False,
    ) -> pyvista.PolyData:
        """Generate surface contours from 3D image labels.

        SurfaceNets algorithm is used to extract contours preserving sharp
        boundaries for the selected labels from the label maps.
        Optionally, the boundaries can be smoothened to reduce the staircase
        appearance in case of low resolution input label maps.

        When smoothing is enabled, a local constraint sphere which is placed
        around each point to restrict its motion. By default, the distance
        (radius of the sphere) is automatically computed using the image
        spacing. This distance can optionally be scaled with
        ``smoothing_constraint_scale``, or alternatively the distance can
        be manually specified with ``smoothing_constraint_distance``.

        This filter requires that the :class:`ImageData` has integer point
        scalars, such as multi-label maps generated from image segmentation.

        .. note::
           Requires ``vtk>=9.3.0``.

        Parameters
        ----------
        select_inputs : int | list[int], default: None
            Specify label ids to include as inputs to the filter. Labels that are
            not selected are removed from the input *before* generating surfaces.
            By default, all label ids are used.

            .. note::
                This parameter has a similar effect as ``select_outputs``.
                ``select_outputs`` should generally be preferred, however, since
                the generated meshes will be exactly the same regardless of the
                selected labels.

        select_outputs : int | list[int], default: None
            Specify label ids to include in the output of the filter. Labels that are
            not selected are removed from the output *after* generating surfaces.
            By default, all label ids are used.

        internal_boundaries : bool, default: True
            Generate polygons which define the boundaries between adjacent
            foreground labels. If ``False``, polygons are only generated at the
            boundaries between foreground labels and the background.

        image_boundaries : bool, default: False
            Generate polygons to "close" the surface at the edges of the image.
            If ``False``, no polygons are generated at the edges, and the generated
            surfaces will therefore be "open" or "clipped at the image boundaries.

        independent_regions : bool, default: True
            Generate duplicate polygons at internal boundaries such that every labeled
            surface at the output has independent boundary cells. This is useful for
            generating complete, independent surface polygons for each labeled region.

            If ``False``, internal boundary polygons between two regions are shared
            by both regions. Has no effect when ``internal_boundaries`` is ``False``,
            since all other boundaries (i.e. between foreground and background, or at
            the edges of the image) do not share cells with other labeled regions.

            Only the cells (quads or triangles) are duplicated, not the points. The
            duplicated cells are inserted next the original cells (i.e. their cell
            ids are off by one).

            .. note::
                In the returned ``'BoundaryLabels'`` array, duplicated cells have a
                descending order. This is in contrast to the default values, where
                foreground labels are sorted in ascending order. E.g. a cell on the
                boundary between regions ``1`` and ``2`` has a default value of
                ``[1, 2]`` in the output, whereas the duplicated cell has a value of
                ``[2, 1]``.

            .. note::
                In the returned ``'SurfaceLabels'`` array, the original cell is
                labeled as the first region, and the duplicated cell is labeled as
                the second region. E.g. a single cell at the boundary between regions
                ``1`` and ``2`` is duplicated such that there are two cells in the
                output, one with a value of ``1`` and the other with value of ``2``.

        output_mesh_type : str, default: None
            Type of the output mesh. Can be either ``'quads'``, or ``'triangles'``.
            By default, if smoothing is off, the output mesh has quadrilateral cells
            (quads). However, if smoothing is enabled, then the output mesh type has
            triangle cells. The mesh type can be forced to be triangles or quads
            whether smoothing is enabled or not.

            .. note::
                If smoothing is enabled and the type is ``'quads'``, the resulting
                quads may not be planar.

        smoothing : bool, default: True
            Apply smoothing to the meshes.

        smoothing_num_iterations : int, default: 16
            Number of smoothing iterations.

        smoothing_relaxation_factor : float, default: 0.5
            Relaxation factor of the smoothing.

        smoothing_constraint_distance : float, default: None
            Constraint distance of the smoothing. Specify the maximum distance
            each point is allowed to move (in any direction) during smoothing.
            This distance may be scaled with ``smoothing_constraint_scale``.
            By default, the constraint distance is computed dynamically from
            the image spacing as

                ``distance = norm(spacing) * scale``.

        smoothing_constraint_scale : float, default: 1.0
            Relative scaling factor applied to ``smoothing_constraint_distance``.
            See that parameter for more details.

        surface_labels : bool, default: True
            Include a single-component cell data array ``'SurfaceLabels'`` in
            the output. The array indicates the labels/regions of the polygons
            composing the output. If ``True``, this array will be set as the
            active scalars of the generated mesh.

            .. note::
                This array is a simplified representation of the ``'BoundaryLabels'``
                array.

        boundary_labels : int, default: False
            Include a two-component cell data array ``'BoundaryLabels'`` in
            the output. The array indicates the labels/regions on either side
            of the polygons composing the output.

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
        BOUNDARY_LABELS = 'BoundaryLabels'
        SURFACE_LABELS = 'SurfaceLabels'
        if not hasattr(_vtk, 'vtkSurfaceNets3D'):  # pragma: no cover
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError('Surface nets 3D require VTK 9.3.0 or newer.')

        # Validate scalars
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

        alg = _vtk.vtkSurfaceNets3D()
        temp_scalars_name = '_PYVISTA_TEMP'
        if select_inputs:
            # Remove non-selected label ids from the input
            # We do this by copying the scalars and setting non-selected ids
            # to the background value to remove them from the input
            temp_scalars = self.active_scalars.copy()  # type: ignore[attr-defined]
            unique_labels = np.unique(temp_scalars)
            background = alg.GetBackgroundLabel()
            for label in unique_labels:
                select_inputs = (
                    [select_inputs]
                    if isinstance(select_inputs, (int, np.integer, float, np.floating))
                    else select_inputs
                )
                if label not in select_inputs:
                    temp_scalars[temp_scalars == label] = background

            self.point_data.set_array(temp_scalars, name=temp_scalars_name)  # type: ignore[attr-defined]
            scalars = temp_scalars_name

        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)
        alg.SetInputData(self)

        if output_mesh_type is None:
            alg.SetOutputMeshTypeToDefault()
        elif output_mesh_type == 'quads':
            alg.SetOutputMeshTypeToQuads()
        elif output_mesh_type == 'triangles':
            alg.SetOutputMeshTypeToTriangles()
        else:
            raise ValueError(
                f'Invalid output mesh type "{output_mesh_type}", use "quads" or "triangles"',
            )

        output_ids: Optional[collections.abc.Sequence[float]] = None
        if internal_boundaries or select_outputs:
            # NOTE: We use 'select_outputs' and 'internal_boundaries' params to implement
            #  similar functionality to `SetOutputStyle`.
            #
            # WARNING: Setting the output style to default or boundary does not really work as
            # expected. Specifically, `SetOutputStyleToDefault` by itself will not actually
            # produce meshes with interior faces at the boundaries between foreground regions
            # (even though this is what is suggested by the docs). Instead, simply calling
            # `SetLabels` below will enable internal boundaries, regardless of the value of
            # `OutputStyle`. This way, we can enable/disable internal boundaries and call
            # `SetOutputStyleToSelected` independently.

            alg.SetOutputStyleToSelected()
            if select_outputs:
                output_ids = (
                    [select_outputs]
                    if isinstance(select_outputs, (int, np.integer, float, np.floating))
                    else select_outputs
                )
            else:
                # Select all inputs as outputs
                # These inputs should already be filtered by `select_inputs`
                output_ids = output_ids if output_ids else np.unique(self.active_scalars).tolist()  # type: ignore[attr-defined]

            # Add selected outputs. Do not add the background value.
            output_ids_list = list(output_ids)
            (
                output_ids_list.remove(bg)
                if (bg := alg.GetBackgroundLabel()) in output_ids_list
                else None
            )
            [alg.AddSelectedLabel(float(label_id)) for label_id in output_ids]
            if internal_boundaries:
                mapping = {
                    val: i + 1
                    for i, val in enumerate(output_ids)
                    if val != alg.GetBackgroundLabel()
                }
                [alg.SetLabel(val, key) for key, val in mapping.items()]

        if smoothing:
            alg.SmoothingOn()
            alg.GetSmoother().SetNumberOfIterations(smoothing_num_iterations)
            alg.GetSmoother().SetRelaxationFactor(smoothing_relaxation_factor)
            # Auto-constraints are On by default which only allows you to scale relative distance
            # (with SetConstraintScale) but not set its value directly.
            # Here, we turn this off so that we can both set its value and/or scale it
            alg.AutomaticSmoothingConstraintsOff()
            # If distance not specified, emulate the auto-constraint calc from vtkSurfaceNets3D
            distance = (
                smoothing_constraint_distance
                if smoothing_constraint_distance
                else np.linalg.norm(self.spacing)  # type: ignore[attr-defined]
            )
            alg.GetSmoother().SetConstraintDistance(distance * smoothing_constraint_scale)
        else:
            alg.SmoothingOff()

        # Get output
        # Suppress improperly used INFO for debugging messages in vtkSurfaceNets3D
        verbosity = _vtk.vtkLogger.GetCurrentVerbosityCutoff()
        _vtk.vtkLogger.SetStderrVerbosity(_vtk.vtkLogger.VERBOSITY_OFF)
        _update_alg(alg, progress_bar, 'Generating label contours')
        # Restore the original vtkLogger verbosity level
        _vtk.vtkLogger.SetStderrVerbosity(verbosity)
        output: pyvista.PolyData = _get_output(alg)

        (  # Clear temp scalars
            self.point_data.remove(temp_scalars_name)  # type: ignore[attr-defined]
            if temp_scalars_name in self.point_data  # type: ignore[attr-defined]
            else None
        )

        if internal_boundaries and independent_regions:

            def duplicate_internal_boundary_cells():
                background_value = alg.GetBackgroundLabel()
                boundary_labels_array = output[BOUNDARY_LABELS]

                # Duplicate if foreground label on both sides of cell
                is_internal_boundary = np.all(boundary_labels_array != background_value, axis=1)
                if np.any(is_internal_boundary):
                    internal_ids = np.nonzero(is_internal_boundary)[0]
                    duplicated_labels = boundary_labels_array[internal_ids]

                    # Insert duplicated scalars. Swap order of 1st and 2nd components
                    insertion_ids = internal_ids + 1
                    duplicated_labels = duplicated_labels[:, ::-1]
                    inserted_array = np.insert(
                        boundary_labels_array,
                        obj=insertion_ids,
                        values=duplicated_labels,
                        axis=0,
                    )

                    # Insert duplicated cells
                    faces = output.regular_faces
                    inserted_faces = np.insert(
                        faces,
                        obj=insertion_ids,
                        values=faces[internal_ids],
                        axis=0,
                    )

                    # Update output
                    output.regular_faces = inserted_faces
                    output.cell_data[BOUNDARY_LABELS] = inserted_array

            duplicate_internal_boundary_cells()

            if surface_labels:
                # Since internal boundaries are duplicated, each surface can be uniquely
                # labeled using the first component of the boundary labels
                output[SURFACE_LABELS] = output[BOUNDARY_LABELS][:, 0]

        if surface_labels:
            if SURFACE_LABELS not in output.cell_data:
                # Replicate what
                output[SURFACE_LABELS] = np.linalg.norm(output[BOUNDARY_LABELS], axis=1)
            output.set_active_scalars(SURFACE_LABELS)

        if not boundary_labels:
            output.cell_data.remove(BOUNDARY_LABELS)

        return output


# def pad_image(
#     image: pyvista.ImageData,
#     pad_width: int | tuple[int, int, int] | tuple[int, int, int, int, int, int] = 1,
#     pad_value: float = 0,
#     pad_empty_dimensions: bool = False,
# ) -> pyvista.ImageData:
#     """Pad an image.
#
#     Parameters
#     ----------
#     pad_width : int | sequence[int], default : 1
#         Specify the amount of padding to add. Specify:
#
#         - A single value to apply constant padding around the entire image.
#         - Three values, one for each ``(X, Y, Z)`` axis, to apply symmetrical
#           padding to each axis independently.
#         - Six values, one for each ``(-X,+X,-Y,+Y,-Z,+Z)`` direction, to apply
#           padding to each direction independently.
#
#     pad_value : int | float, default : 0
#         Value to use for padded elements.
#
#     pad_empty_dimensions : bool, default : False
#         Control if empty dimensions should be padded. E.g. if ``False`` (the default),
#         padding 2D data ensures the output remains 2D. Otherwise, all dimensions are padded,
#         even if they are empty.
#     """
#
#     # parse pad_width to create a length-6 tuple (-X,+X,-Y,+Y,-Z,+Z)
#     pw = (pad_width,) if isinstance(pad_width, int) else pad_width
#     if not isinstance(pw, collections.abc.Sequence) and not all(
#         isinstance(val, (int, np.integer)) for val in pw
#     ):
#         raise TypeError("Pad width must an integer or an iterable of integers.")
#     length = len(pw)
#     if length == 1:
#         all_pad_widths = pw * 6
#     elif length == 3:
#         all_pad_widths = (pw[0], pw[0], pw[1], pw[1], pw[2], pw[2])
#     elif length == 6:
#         all_pad_widths = pw
#     else:
#         raise ValueError(f"Pad width must have 1, 3, or 6 values, got {length} instead.")
#
#     if pad_empty_dimensions is False:
#         # set pad width to zero for dimensions which are empty (e.g. 2D cases)
#         dims = image.dimensions
#         dim_pairs = (dims[0], dims[0], dims[1], dims[1], dims[2], dims[2])
#         is_not_empty = np.asarray(dim_pairs) != 0
#         all_pad_widths = tuple(is_not_empty * np.asarray(all_pad_widths))
#
#     # define new extents after padding
#     pad_xn, pad_xp, pad_yn, pad_yp, pad_zn, pad_zp = all_pad_widths
#     ext_xn, ext_xp, ext_yn, ext_yp, ext_zn, ext_zp = image.GetExtent()
#
#     padded_extents = (
#         ext_xn - pad_xn,  # minX
#         ext_xp + pad_xp,  # maxX
#         ext_yn - pad_yn,  # minY
#         ext_yp + pad_yp,  # maxY
#         ext_zn - pad_zn,  # minZ
#         ext_zp + pad_zp,
#     )  # maxZ
#
#     constant_pad = _vtk.vtkImageConstantPad()
#     constant_pad.SetInputData(image)
#     constant_pad.SetConstant(pad_value)
#     constant_pad.SetOutputWholeExtent(*padded_extents)
#     constant_pad.Update()
#     return constant_pad.GetOutput()
