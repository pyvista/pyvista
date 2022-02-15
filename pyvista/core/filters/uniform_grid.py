"""Filters module with a class to manage filters/algorithms for uniform grid datasets."""
import collections.abc

import numpy as np

import pyvista
from pyvista import _vtk, abstract_class
from pyvista.core.filters import _get_output, _update_alg
from pyvista.core.filters.data_set import DataSetFilters


@abstract_class
class UniformGridFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for uniform grid datasets."""

    def gaussian_smooth(
        self, radius_factor=1.5, std_dev=2.0, scalars=None, preference='points', progress_bar=False
    ):
        """Smooth the data with a Gaussian kernel.

        Parameters
        ----------
        radius_factor : float or iterable, optional
            Unitless factor to limit the extent of the kernel.

        std_dev : float or iterable, optional
            Standard deviation of the kernel in pixel units.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        preference : str, optional
            When scalars is specified, this is the preferred array
            type to search for in the dataset.  Must be either
            ``'point'`` or ``'cell'``.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.UniformGrid
            Uniform grid with smoothed scalars.

        """
        alg = _vtk.vtkImageGaussianSmooth()
        alg.SetInputDataObject(self)
        if scalars is None:
            field, scalars = self.active_scalars_info
        else:
            field = self.get_array_association(scalars, preference=preference)
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
        self, kernel_size=(3, 3, 3), scalars=None, preference='points', progress_bar=False
    ):
        """Smooth data using a median filter.

        Parameters
        ----------
        kernel_size : list(int) or tuple(int), optional
            Length 3 list or tuple of ints : ``(x_size, y_size, z_size)``
            Size of the kernel in each dimension (units of voxels). Default is
            a 3D median filter. If you want to do a 2D median filter, set the
            size to 1 in the dimension you don't want to filter over.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        preference : str, optional
            When scalars is specified, this is the preferred array
            type to search for in the dataset.  Must be either
            ``'point'`` or ``'cell'``.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.UniformGrid
            Uniform grid with smoothed scalars.

        """
        alg = _vtk.vtkImageMedian3D()
        alg.SetInputDataObject(self)
        if scalars is None:
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
        voi : tuple(int)
            Length 6 iterable of ints: ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
            These bounds specify the volume of interest in i-j-k min/max
            indices.

        rate : tuple(int), optional
            Length 3 iterable of ints: ``(xrate, yrate, zrate)``.
            Default: ``(1, 1, 1)``.

        boundary : bool, optional
            Control whether to enforce that the "boundary" of the grid
            is output in the subsampling process. This only has effect
            when the rate in any direction is not equal to 1. When
            this is enabled, the subsampling will always include the
            boundary of the grid even though the sample rate is not an
            even multiple of the grid dimensions. By default this is
            disabled.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.UniformGrid
            UniformGrid subset.
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
        fixed = pyvista.UniformGrid()
        fixed.origin = result.bounds[::2]
        fixed.spacing = result.spacing
        fixed.dimensions = result.dimensions
        fixed.point_data.update(result.point_data)
        fixed.cell_data.update(result.cell_data)
        fixed.field_data.update(result.field_data)
        fixed.copy_meta_from(result)
        return fixed

    def image_dilate_erode(
        self,
        dilate_value=1,
        erode_value=0,
        kernel_size=(3, 3, 3),
        scalars=None,
        preference='points',
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
        dilate_value : int or float, optional
            Dilate value in the dataset. Default: ``1``.

        erode_value : int or float, optional
            Erode value in the dataset. Default: ``0``.

        kernel_size : list(int) or tuple(int), optional
            Length 3 iterable of ints: ``(xsize, ysize, zsize)``.
            Determines the size (and center) of the kernel.
            Default: ``(3, 3, 3)``.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        preference : str, optional
            When scalars are specified, this is the preferred array
            type to search for in the dataset.  Must be either
            ``'point'`` or ``'cell'``.

        progress_bar : bool, optional
            Display a progress bar to indicate progress. Default ``False``.

        Returns
        -------
        pyvista.UniformGrid
            Dataset that has been dilated/eroded on the boundary of the specified scalars.

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
            field, scalars = self.active_scalars_info
        else:
            field = self.get_array_association(scalars, preference=preference)
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
        in_value=1,
        out_value=0,
        scalars=None,
        preference='points',
        progress_bar=False,
    ):
        """Apply a threshold to scalar values in a uniform grid.

        If a single value is given for threshold, scalar values above or equal
        to the threshold are ``'in'`` and scalar values below the threshold are ``'out'``.
        If two values are given for threshold (sequence) then values equal to
        or between the two values are ``'in'`` and values outside the range are ``'out'``.

        If ``None`` is given for ``in_value``, scalars that are ``'in'`` will not be replaced.
        If ``None`` is given for ``out_value``, scalars that are ``'out'`` will not be replaced.

        Parameters
        ----------
        threshold : float or sequence
            Single value or (min, max) to be used for the data threshold.  If
            a sequence, then length must be 2. Threshold(s) for deciding which
            cells/points are ``'in'`` or ``'out'`` based on scalar data.

        in_value : float or int or None, optional
            Scalars that match the threshold criteria for ``'in'`` will be replaced with this.
            Default is 1.

        out_value : float or int or None, optional
            Scalars that match the threshold criteria for ``'out'`` will be replaced with this.
            Default is 0.

        scalars : str, optional
            Name of scalars to process. Defaults to currently active scalars.

        preference : str, optional
            When scalars is specified, this is the preferred array
            type to search for in the dataset.  Must be either
            ``'point'`` or ``'cell'``.

        progress_bar : bool, optional
            Display a progress bar to indicate progress. Default ``False``.

        Returns
        -------
        pyvista.UniformGrid
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
            alg.ReplaceInOn()
            alg.SetInValue(in_value)
        else:
            alg.ReplaceInOff()
        if out_value is not None:
            alg.ReplaceOutOn()
            alg.SetOutValue(out_value)
        else:
            alg.ReplaceOutOff()
        # run the algorithm
        _update_alg(alg, progress_bar, 'Performing Image Thresholding')
        return _get_output(alg)
