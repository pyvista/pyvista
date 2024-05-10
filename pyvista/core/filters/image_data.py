"""Filters with a class to manage filters/algorithms for uniform grid datasets."""

from __future__ import annotations

import collections.abc
import operator
from typing import TYPE_CHECKING, Literal, Optional, Union, cast
import warnings

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import AmbiguousDataError, MissingDataError, PyVistaDeprecationWarning
from pyvista.core.filters import _get_output, _update_alg
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.core.utilities.arrays import FieldAssociation, set_default_active_scalars
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

    def contour_labeled(  # pragma: no cover
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

        SurfaceNets algorithm is used to extract contours preserving sharp
        boundaries for the selected labels from the label maps.
        Optionally, the boundaries can be smoothened to reduce the staircase
        appearance in case of low resolution input label maps.

        This filter requires that the :class:`ImageData` has integer point
        scalars, such as multi-label maps generated from image segmentation.

        .. note::
           Requires ``vtk>=9.3.0``.

        .. deprecated:: 0.44
            This filter produces unexpected results and is deprecated.
            Use :meth:`~pyvista.ImageDataFilters.contour_labels` instead.
            See https://github.com/pyvista/pyvista/issues/5981 for details.

            To replicate the default behavior from this filter, call `contour_labels`
            with the following arguments:

            .. code::

                n_labels = range(N)
                contour_labels(
                    select_outputs=n_labels,  # replacement for 'n_labels' param
                    internal_polygons=False,  # replacement for 'output_style' param
                    smoothing=False,  # smoothing is now on by default
                    output_mesh_type='quads',  # mesh type is no longer fixed to 'quads'
                    output_labels='boundary',  # return 'boundary_labels' array
                )

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
        warnings.warn(
            "This filter produces unexpected results and is deprecated. Use `contour_labels` instead."
            "\nRefer to the documentation for `contour_labeled` for details on how to transition to the new filter."
            "\nSee https://github.com/pyvista/pyvista/issues/5981 for details.",
            PyVistaDeprecationWarning,
        )

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

    def contour_labels(
        self,
        *,
        scalars: Optional[str] = None,
        select_inputs: Optional[Union[int, collections.abc.Sequence[int]]] = None,
        select_outputs: Optional[Union[int, collections.abc.Sequence[int]]] = None,
        internal_polygons: bool = True,
        duplicate_polygons: bool = True,
        closed_boundary: bool = False,
        output_mesh_type: Optional[Literal['quads', 'triangles']] = None,
        output_labels: Optional[Union[Literal['surface', 'boundary'], bool]] = 'surface',
        smoothing: bool = True,
        smoothing_iterations: int = 16,
        smoothing_relaxation: float = 0.5,
        smoothing_distance: Optional[float] = None,
        smoothing_scale: float = 1.0,
        progress_bar: bool = False,
    ) -> pyvista.PolyData:
        """Generate surface contours from 3D image label maps.

        This filter uses `vtkSurfaceNets <https://vtk.org/doc/nightly/html/classvtkSurfaceNets3D.html#details>`__
        to extract polygonal surface contours from non-continuous label maps, which
        corresponds to discrete regions in an input 3D image (i.e., volume).

        The generated surface is smoothed using a constrained smoothing filter, which
        may be fine-tuned to control the smoothing process. Optionally, smoothing may
        be disabled to generate a voxelized staircase-like surface.

        This filter is designed to generate surfaces from voxel *points*, i.e.
        points which represent voxels, such as point arrays from 3D medical images.
        If the input is voxel *cells* (i.e. cell scalars are specified), the cell
        centers are used to generate the output.

        .. note::
            By default, the generated surface polygons are labeled using a single-component
            scalar array. This differs from the ``vtkSurfaceNets`` filter, which outputs
            a two-component scalar array. See the documentation for the ``output_labels``
            parameter for details about these arrays.

            If desired, the default output from ``vtkSurfaceNets`` can be obtained with:

            .. code::

                contour_labels(
                    duplicate_polygons=False, output_labels='boundary'
                )


        This filter requires that the :class:`ImageData` has integer point
        scalars, such as multi-label maps generated from image segmentation.

        .. note::
           This filter requires ``vtk>=9.3.0``.

        Parameters
        ----------
        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars. If cell
            scalars are specified, the cell centers are used.

        select_inputs : int | VectorLike[int], default: None
            Specify label ids to include as inputs to the filter. Labels that are not
            selected are removed from the input *before* generating the surface. By
            default, all label ids are used.

            Since the smoothing operation occurs across selected input regions, using
            this option to filter the input can result in smoother and more visually
            pleasing surfaces since non-selected inputs are not considered during
            smoothing. However, this also means that the generated surface will change
            shape depending on which inputs are selected.

        select_outputs : int | VectorLike[int], default: None
            Specify label ids to include in the output of the filter. Labels that are
            not selected are removed from the output *after* generating the surface. By
            default, all label ids are used.

            Since the smoothing operation occurs across all input regions, using this
            option to filter the output means that the selected output regions will have
            the same shape (i.e. smoothed in the same manner), regardless of the outputs
            that are selected. This is useful for generating a surface for specific
            labels while also preserving the effects that non-selected outputs would
            have on the generated surface.

        internal_polygons : bool, default: True
            Generate internal polygons which define the boundaries between adjacent
            foreground labels. If ``False``, only external polygons are generated
            between foreground labels and the background.

        duplicate_polygons : bool, default: True
            Generate duplicate polygons at internal boundaries betweens regions such
            that every labeled surface at the output has independent boundary cells.
            This is useful for generating complete, independent surface polygons for
            each labeled region.

            If ``False``, internal boundary polygons between two regions are shared
            by both regions. Has no effect when ``internal_polygons`` is ``False``
            since only internal polygons are duplicated.

            .. note::
                Only the cells (quads or triangles) are duplicated, not the points.
                Duplicated cells are inserted next the original cells (i.e. their cell
                ids are off by one).

        closed_boundary : bool, default: True
            Generate polygons to "close" the surface at the boundaries of the image.
            Setting this value to ``False`` is useful if processing multiple volumes
            separately so that they fit together without creating surface overlap.

        output_mesh_type : str, default: None
            Type of the output mesh. Can be either ``'quads'``, or ``'triangles'``. By
            default, if smoothing is off, the output mesh has quadrilateral cells
            (quads). However, if smoothing is enabled, then the output mesh type has
            triangle cells. The mesh type can be forced to be triangles or quads
            whether smoothing is enabled or not.

            .. note::
                If smoothing is enabled and the type is ``'quads'``, the resulting
                quads may not be planar.

        output_labels : 'surface' | 'boundary' | bool, default: 'surface'
            Select the labeled cell data array(s) to include with the output. Choose
            either ``'surface'`` or ``'boundary'`` to include a single array, ``True``
            to include both arrays, or ``False`` to not include any data arrays in the
            output.

            The default is ``'surface'``, which is recommended for applications where a
            simple mapping of labeled image data at the input to labeled polygonal at
            the output is required. For more advanced workflows, choose ``'boundary'``.
            See details about the arrays options below.

            - ``'surface'``: Include a single-component cell data array ``'surface_labels'``
                with the output. The array indicates the labels/regions of the polygons
                composing the output.

                This array is a simplified representation of the ``'boundary_labels'``
                array. If ``internal_boundaries=True``, selecting this option also
                requires setting ``independent_regions=True``.

            - ``'boundary'``: Include a two-component cell data array ``'boundary_labels'``
                with the output. The array indicates the labels/regions on either side
                of the polygons composing the output. The array's values are structured
                as follows:

                -   Boundary labels between foreground regions and background are always
                    ordered as ``[foreground, background]``.

                    E.g. ``[1, 0]`` for the boundary between region ``1`` and background ``0``.

                -   If ``internal_boundaries`` is ``True``, boundary labels between two
                    foreground regions are sorted in ascending order.

                    E.g. ``[1, 2]`` for the boundary between regions ``1`` and ``2``.

                -   If ``independent_regions`` is ``True``, internal boundary cells are
                    duplicated, and the labels of the duplicated cells are sorted in
                    descending order.

                    E.g. a cell with the label ``[1, 2]`` is duplicated and labeled
                    as ``[2, 1]`` for the boundary between regions ``1`` and ``2``.

        smoothing : bool, default: True
            Smooth the generated surface using a constrained smoothing filter. Each
            point in surface is smoothed as follows:

            For a point ``pi`` connected to a list of points ``pj`` via an edge, ``pi``
            is moved towards the average position of ``pj`` multiplied by the
            ``smoothing_relaxation`` factor, and limited by the ``smoothing_distance``
            constraint. This process is repeated either until convergence occurs, or
            the maximum number of ``smoothing_iterations`` is reached.

        smoothing_iterations : int, default: 16
            Maximum number of smoothing iterations to use.

        smoothing_relaxation : float, default: 0.5
            Relaxation factor of the smoothing process.

        smoothing_distance : float, default: None
            Maximum distance each point is allowed to move (in any direction) during
            smoothing. This distance may be scaled with ``smoothing_scale``. By default,
            the distance is computed dynamically from the image spacing as:

                ``distance = norm(image_spacing) * smoothing_scale``

        smoothing_scale : float, default: 1.0
            Relative scaling factor applied to ``smoothing_distance``. See that
            parameter for details.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Surface mesh of labeled regions.

        References
        ----------
        S. Frisken, “SurfaceNets for Multi-Label Segmentations with Preservation of
        Sharp Boundaries”, J. Computer Graphics Techniques, 2022. Available online:
        http://jcgt.org/published/0011/01/03/

        W. Schroeder, S. Tsalikis, M. Halle, S. Frisken. A High-Performance SurfaceNets
        Discrete Isocontouring Algorithm. arXiv:2401.14906. 2024. Available online:
        http://arxiv.org/abs/2401.14906

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
        BOUNDARY_LABELS = 'boundary_labels'
        SURFACE_LABELS = 'surface_labels'
        if not hasattr(_vtk, 'vtkSurfaceNets3D'):  # pragma: no cover
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError('Surface nets 3D require VTK 9.3.0 or newer.')

        if output_labels:
            if output_labels is True:
                surface_labels = True
                boundary_labels = True
            elif output_labels == 'surface':
                surface_labels = True
                boundary_labels = False
            elif output_labels == 'boundary':
                surface_labels = False
                boundary_labels = True
            else:
                raise ValueError(
                    f"Output labels must be one of 'surface', 'boundary', or True. Got {output_labels}.",
                )
        else:
            surface_labels = False
            boundary_labels = False

        if surface_labels and internal_polygons and not duplicate_polygons:
            raise ValueError(
                'Parameter duplicate_polygons must be True when generating surface labels with internal polygons.'
                '\nEither set duplicate_polygons to True or set internal_polygons to False.',
            )

        # Validate scalars
        if scalars is None:
            set_default_active_scalars(self)  # type: ignore[arg-type]
            field, scalars = self.active_scalars_info  # type: ignore[attr-defined]
        else:
            field = self.get_array_association(scalars, preference='point')  # type: ignore[attr-defined]

        # Make sure we have points to work with
        if field == FieldAssociation.POINT:
            alg_input = self
        else:
            alg_input = self._cell_voxels_to_point_voxels(cell_scalars=scalars)

        alg = _vtk.vtkSurfaceNets3D()
        background_value = 0
        alg.SetBackgroundLabel(background_value)
        # Pad with background values to create background/foreground polygons at
        # image boundaries
        alg_input = alg_input.pad_image(alg.GetBackgroundLabel()) if closed_boundary else alg_input

        temp_scalars_name = '_PYVISTA_TEMP'
        if select_inputs:
            # Remove non-selected label ids from the input
            # We do this by copying the scalars and setting non-selected ids
            # to the background value to remove them from the input
            temp_scalars = alg_input.active_scalars  # type: ignore[attr-defined]
            temp_scalars = temp_scalars.copy() if alg_input is self else temp_scalars
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

            alg_input.point_data[temp_scalars_name] = temp_scalars  # type: ignore[attr-defined]
            scalars = temp_scalars_name

        alg.SetInputArrayToProcess(
            0,
            0,
            0,
            field.value,
            scalars,
        )  # args: (idx, port, connection, field, name)
        alg.SetInputData(alg_input)

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
        if internal_polygons or select_outputs:
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
                output_ids = output_ids if output_ids else np.unique(alg_input.active_scalars).tolist()  # type: ignore[attr-defined]

            # Add selected outputs. Do not add the background value.
            output_ids_list = list(output_ids)
            (
                output_ids_list.remove(bg)
                if (bg := alg.GetBackgroundLabel()) in output_ids_list
                else None
            )
            [alg.AddSelectedLabel(float(label_id)) for label_id in output_ids_list]
            if internal_polygons:
                [alg.SetLabel(i, float(val)) for i, val in enumerate(output_ids_list)]

        def _is_small_number(num):
            return isinstance(num, (float, int, np.floating, np.integer)) and num < 1e-8

        if (
            smoothing
            and not _is_small_number(smoothing_scale)
            and not _is_small_number(smoothing_distance)
        ):
            # Only enable smoothing if distance is not very small, since a small distance will
            # actually result in large smoothing (suspected division by zero error in vtk code)
            alg.SmoothingOn()
            alg.GetSmoother().SetNumberOfIterations(smoothing_iterations)
            alg.GetSmoother().SetRelaxationFactor(smoothing_relaxation)

            # Auto-constraints are On by default which only allows you to scale relative distance
            # (with SetConstraintScale) but not set its value directly.
            # Here, we turn this off so that we can both set its value and/or scale it
            alg.AutomaticSmoothingConstraintsOff()
            # If distance not specified, emulate the auto-constraint calc from vtkSurfaceNets3D
            distance = (
                smoothing_distance
                if smoothing_distance
                else np.linalg.norm(alg_input.spacing)  # type: ignore[attr-defined]
            )
            alg.GetSmoother().SetConstraintDistance(distance * smoothing_scale)
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
        output.rename_array('BoundaryLabels', BOUNDARY_LABELS)
        (  # Clear temp scalars
            alg_input.point_data.remove(temp_scalars_name)  # type: ignore[attr-defined]
            if temp_scalars_name in alg_input.point_data  # type: ignore[attr-defined]
            else None
        )

        if internal_polygons and duplicate_polygons:

            def duplicate_internal_boundary_cells():
                background_value = alg.GetBackgroundLabel()
                boundary_labels_array = output.cell_data[BOUNDARY_LABELS]

                # Duplicate if foreground label on both sides of cell
                is_internal_boundary = np.all(boundary_labels_array != background_value, axis=1)
                if np.any(is_internal_boundary):
                    internal_ids = np.nonzero(is_internal_boundary)[0]
                    duplicated_labels = boundary_labels_array[internal_ids]
                    if select_outputs:
                        # Only duplicate if both regions on each side
                        # of the boundary are included in the output
                        in_first_component = np.isin(duplicated_labels[:, 0], select_outputs)
                        in_second_component = np.isin(duplicated_labels[:, 1], select_outputs)
                        keep = np.logical_and(in_first_component, in_second_component)
                        internal_ids = internal_ids[keep]
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
            # Safeguard, an error should have been raised earlier
            assert duplicate_polygons if internal_polygons else True

            # Need to simplify the 2-component boundary labels scalars.
            # In general, we cannot simply take the first component and expect correct
            # output, but this method applies constraints to make this possible
            output.cell_data[SURFACE_LABELS] = output.cell_data[BOUNDARY_LABELS][:, 0]
            output.set_active_scalars(SURFACE_LABELS)

        if not boundary_labels:
            output.cell_data.remove(BOUNDARY_LABELS)

        return output

    def _point_voxels_to_cell_voxels(self, point_scalars: Optional[str] = None):
        """Convert point voxel data to cell voxel data."""
        return self._convert_voxels('points_to_cells', scalars=point_scalars)

    def _cell_voxels_to_point_voxels(self, cell_scalars: Optional[str] = None):
        """Convert cell voxel data to point voxel data."""
        return self._convert_voxels('cells_to_points', scalars=cell_scalars)

    def _convert_voxels(
        self,
        method: Optional[Literal['points_to_cells', 'cells_to_points']] = None,
        scalars: Optional[str] = None,
    ):
        point_data = self.point_data  # type: ignore[attr-defined]
        cell_data = self.cell_data  # type: ignore[attr-defined]
        if method is None and scalars is None:
            # Need to determine which method to use
            point_len = len(point_data.keys())
            cell_len = len(cell_data.keys())
            ambiguous_msg = "Dataset contains {} cell data {}} point data. Specify method explicitly with 'points_to_cells' to convert point data, or 'cells_to_points' to convert cell data"
            if point_len > 0 and cell_len > 0:
                raise AmbiguousDataError(ambiguous_msg.format('both', 'and'))
            elif point_len == 0 and cell_len == 0:
                raise AmbiguousDataError(ambiguous_msg.format('no', 'or'))
            elif point_len > 0:
                method = 'points_to_cells'
            elif cell_len > 0:
                method = 'cells_to_points'
            else:
                raise RuntimeError("Error, this code should not be reachable.")

        new_image = pyvista.ImageData()
        if method == 'cells_to_points':
            origin_operator = operator.add
            dims_operator = operator.sub
            old_data = cell_data
            new_data = new_image.point_data
        else:
            origin_operator = operator.sub
            dims_operator = operator.add
            old_data = point_data
            new_data = new_image.cell_data

        # Create new image
        # Point voxels should equal cell voxel centers
        new_image.origin = origin_operator(self.origin, np.array(self.spacing) / 2)  # type: ignore[attr-defined]
        new_image.dimensions = dims_operator(np.array(self.dimensions), 1)  # type: ignore[attr-defined]
        new_image.spacing = self.spacing  # type: ignore[attr-defined]
        new_image.SetDirectionMatrix(self.GetDirectionMatrix())  # type: ignore[attr-defined]

        # Copy old data to new data
        # new_image.field_data = self.field_data.copy()  # type: ignore[attr-defined]

        array_names = old_data.keys() if scalars else [scalars]
        for array_name in array_names:
            new_data[array_name] = old_data[array_name]
        return new_image

    def pad_image(
        self,
        pad_value: Union[float, VectorLike[float], Literal['wrap', 'mirror']] = 0.0,
        *,
        pad_size: int | VectorLike[int] = 1,
        pad_singleton_dims: bool = False,
        scalars: Optional[str] = None,
        pad_all_scalars: bool = False,
        progress_bar=False,
    ) -> pyvista.ImageData:
        """Enlarge an image by padding its boundaries with new points.

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
            - ``'wrap'``': New points are filled by wrapping around the padding axis.
            - ``'mirror'``': New points are filled by mirroring the padding axis.

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
            field = self.get_array_association(  # type: ignore[attr-defined]
                scalars,
                preference='point',
            )
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
            num_input_components = _get_num_components(
                self.active_scalars,  # type: ignore[attr-defined]
            )
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
