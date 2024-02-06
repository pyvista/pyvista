"""Filters with a class to manage filters/algorithms for uniform grid datasets."""
import collections.abc
from typing import Literal, Optional, cast

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import AmbiguousDataError, MissingDataError
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
            0, 0, 0, field.value, scalars
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
        self, kernel_size=(3, 3, 3), scalars=None, preference='point', progress_bar=False
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
            0, 0, 0, field.value, scalars
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
            0, 0, 0, field.value, scalars
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
            0, 0, 0, field.value, scalars
        )  # args: (idx, port, connection, field, name)
        # set the threshold(s) and mode
        if isinstance(threshold, (np.ndarray, collections.abc.Sequence)):
            if len(threshold) != 2:
                raise ValueError(
                    f'Threshold must be length one for a float value or two for min/max; not ({threshold}).'
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
            output, self.point_data.active_scalars_name, output_scalars_name
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
            output, self.point_data.active_scalars_name, output_scalars_name
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
            output, self.point_data.active_scalars_name, output_scalars_name
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
            output, self.point_data.active_scalars_name, output_scalars_name
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
                    'active with `point_data.active_scalars_name = `'
                )
            else:
                raise MissingDataError('FFT filters require point scalars.')

        if not np.issubdtype(self.point_data.active_scalars.dtype, np.complexfloating):
            raise ValueError(
                'Active scalars must be complex data for this filter, represented '
                'as an array with a datatype of `numpy.complex64` or '
                '`numpy.complex128`.'
            )

    def _flip_uniform(self, axis) -> 'pyvista.ImageData':
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
    ) -> 'pyvista.PolyData':
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
            set_default_active_scalars(self)  # type: ignore
            field, scalars = self.active_scalars_info  # type: ignore
            if field != FieldAssociation.POINT:
                raise ValueError('If `scalars` not given, active scalars must be point array.')
        else:
            field = self.get_array_association(scalars, preference='point')  # type: ignore
            if field != FieldAssociation.POINT:
                raise ValueError(
                    f'Can only process point data, given `scalars` are {field.name.lower()} data.'
                )
        alg.SetInputArrayToProcess(
            0, 0, 0, field.value, scalars
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
                f'Invalid output mesh type "{output_mesh_type}", use "quads" or "triangles"'
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
