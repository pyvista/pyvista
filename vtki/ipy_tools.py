"""
A set of useful plotting tools and widgets that can be used in a Jupyter
notebook
"""
ipy_available = False
try:
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets
    ipy_available = True
except:
    pass

import collections
import numpy as np

import vtk

import vtki
from vtki.utilities import is_vtki_obj, wrap
from vtki.plotting import run_from_ipython


class InteractiveTool(object):
    """A backend helper for various interactive ipython tools.
    This tool can be added to an active plotter in the background if passed as
    the ``plotter`` argument.
    """

    def __init__(self, dataset, plotter=None, scalars=None, preference='cell',
                 show_bounds=False, reset_camera=True, outline=None,
                 displayParams={}, defaultParams={}, **kwargs):
        if not run_from_ipython() or not ipy_available:
            raise RuntimeError('Interactive plotting tools require IPython and the ``ipywidgets`` package.')
        # Check the input dataset to make sure its compatible
        if not is_vtki_obj(dataset):
            dataset = wrap(dataset)
            if not is_vtki_obj(dataset):
                raise RuntimeError('Object not supported for plotting in vtki.')

        # Make the input/output of this tool available
        self.input_dataset = dataset
        self.output_dataset = None

        if plotter is None:
            plotter = vtki.BackgroundPlotter()
            plotter.setWindowTitle(type(self).__name__)
        self.plotter = plotter

        # This is the actor that will be removed and re-added to the plotter
        self._data_to_update = None

        # Intialize plotting parameters
        self.valid_range = self.input_dataset.get_data_range(arr=scalars, preference=preference)
        displayParams.setdefault('rng', self.valid_range)
        displayParams.setdefault('scalars', scalars)
        displayParams.setdefault('preference', preference)
        self.displayParams = displayParams

        # Set the tool status
        self._need_to_update = True

        # Add some intital plotting stuff to the scene
        self._initialize(show_bounds, reset_camera, outline)

        # Run the tool
        self.tool(defaultParams=defaultParams, **kwargs)


    def tool(**kwargs):
        """This method is implemented for each tool to perfrom the data
        filtering and setting up the widgets"""
        raise NotImplementedError('This method has not been implemented')


    def _initialize(self, show_bounds, reset_camera, outline):
        """Outlines the input dataset and sets up the scene"""
        if outline is None:
            self.plotter.add_mesh(self.input_dataset.outline_corners())
        elif outline:
            self.plotter.add_mesh(self.input_dataset.outline())
        # add the axis labels
        if show_bounds:
            self.plotter.add_bounds_axes()
        if reset_camera:
            cpos = self.plotter.get_default_cam_pos()
            self.plotter.camera_position = cpos
            self.plotter.reset_camera()
            self.plotter.camera_set = False


    def _update_plotting_params(self, **kwargs):
        """Some plotting parameters can be changed through the tool; this
        updataes those plotting parameters.
        """
        scalars = kwargs.get('scalars', None)
        if scalars is not None:
            old = self.displayParams['scalars']
            self.displayParams['scalars'] = scalars
            if old != scalars:
                self.plotter.remove_actor(self._data_to_update, resetcam=False)
                self._need_to_update = True
        if hasattr(self, 'valid_range'):
            self.displayParams['rng'] = self.valid_range
        cmap = kwargs.get('cmap', None)
        if cmap is not None:
            self.displayParams['cmap'] = cmap



class OrthogonalSlicer(InteractiveTool):
    """Within ipython enviornments like Jupyter notebooks, this will create
    an interactive render window with slider bars in te ipython enviornment to
    move orthogonal slices through the scene.

    Parameters
    ----------
    dataset : vtki.Common
        The datset to orthogonalally slice

    plotter : vtki.BasePlotter
        The active plotter (rendering window) to use

    clean : bool, optional
        This will apply a threshold on the input dataset to remove any NaN
        values. Default is True if active scalar present.

    step : float or tuple(float)
        The increments for the XYZ locations on each of the slider bars

    scalars : str
        The name of the scalars to plot

    preference : str, optional
        The preference for data choice when search for the scalar array

    displayParams : dict
        Any plotting keyword parameters to use

    """

    def tool(self, clean=True, step=None, defaultParams={}):
        if clean and self.input_dataset.active_scalar is not None:
            # This will clean out the nan values
            self.input_dataset = self.input_dataset.threshold()

        x, y, z = self.input_dataset.center

        self._data_to_update = [None, None, None]
        self.output_dataset = vtki.MultiBlock()
        self._old = [None, None, None]

        axes = ['x', 'y', 'z']

        def _update_slice(index, x, y, z):
            self.plotter.remove_actor(self._data_to_update[index], resetcam=False)
            self.output_dataset[index] = self.input_dataset.slice(normal=axes[index], origin=[x,y,z])
            self._data_to_update[index] = self.plotter.add_mesh(self.output_dataset[index],
                    show_edges=False, resetcam=False, **self.displayParams)
            self._old[index] = [x,y,z][index]

        def update(x, y, z, **kwargs):
            self._update_plotting_params(**kwargs)
            if x != self._old[0] or self._need_to_update:
                _update_slice(0, x, y, z)
            if y != self._old[1] or self._need_to_update:
                _update_slice(1, x, y, z)
            if z != self._old[2] or self._need_to_update:
                _update_slice(2, x, y, z)
            self._need_to_update = False

        # Set up the step sizes for the sliders
        if step is None:
            stepx = 0.05 * (self.input_dataset.bounds[1] - self.input_dataset.bounds[0])
            stepy = 0.05 * (self.input_dataset.bounds[3] - self.input_dataset.bounds[2])
            stepz = 0.05 * (self.input_dataset.bounds[5] - self.input_dataset.bounds[4])
        elif isinstance(step, collections.Iterable):
            stepx = step[0]
            stepy = step[1]
            stepz = step[2]
        else:
            stepx = step
            stepy = step
            stepz = step

        # Now set up the widgets
        xsl = widgets.FloatSlider(min=self.input_dataset.bounds[0]+stepx,
                            max=self.input_dataset.bounds[1]-stepx,
                            step=stepx,
                            value=self.input_dataset.center[0],
                            continuous_update=False)
        ysl = widgets.FloatSlider(min=self.input_dataset.bounds[2]+stepy,
                            max=self.input_dataset.bounds[3]-stepy,
                            step=stepy,
                            value=self.input_dataset.center[1],
                            continuous_update=False)
        zsl = widgets.FloatSlider(min=self.input_dataset.bounds[4]+stepz,
                            max=self.input_dataset.bounds[5]-stepz,
                            step=stepz,
                            value=self.input_dataset.center[2],
                            continuous_update=False)

        # Create/display the widgets
        interact(update, x=xsl, y=ysl, z=zsl,
                 scalars=self.input_dataset.scalar_names)


class ManySlicesAlongAxis(InteractiveTool):
    """Within ipython enviornments like Jupyter notebooks, this will create
    an interactive render window with slider bars in te ipython enviornment to
    create many slices along a specified axis.

    Parameters
    ----------
    dataset : vtki.Common
        The datset to orthogonalally slice

    plotter : vtki.BasePlotter
        The active plotter (rendering window) to use

    clean : bool, optional
        This will apply a threshold on the input dataset to remove any NaN
        values. Default is True if active scalar present.

    tol : float, optional
        The tolerance to the edge of the dataset bounds to create the slices

    scalars : str
        The name of the scalars to plot

    preference : str, optional
        The preference for data choice when search for the scalar array

    displayParams : dict
        Any plotting keyword parameters to use

    """

    def tool(self, clean=True, tol=None, defaultParams={}):
        if clean and self.input_dataset.active_scalar is not None:
            # This will clean out the nan values
            self.input_dataset = self.input_dataset.threshold()

        nsl = widgets.IntSlider(min=1, max=10, step=1, value=5,
                                continuous_update=False)

        def update(n, axis, **kwargs):
            if n >= nsl.max:
                nsl.max *= 2
            self._update_plotting_params(**kwargs)
            self.plotter.remove_actor(self._data_to_update, resetcam=False)
            self.output_dataset = self.input_dataset.slice_along_axis(n=n, axis=axis, tol=tol)
            self._data_to_update = self.plotter.add_mesh(self.output_dataset,
                show_edges=False, resetcam=False, **self.displayParams)
            self._need_to_update = False

        # Create/display the widgets
        interact(update, n=nsl, axis=['x', 'y', 'z'],
                 scalars=self.input_dataset.scalar_names)


class Threshold(InteractiveTool):
    """Yields slider bars for user to control the threshold range in an
    interactive plot. The threshold will default at 25 and 75 percent of the
    range.

    Parameters
    ----------
    dataset : vtki.Common
        The datset to orthogonalally slice

    plotter : vtki.BasePlotter
        The active plotter (rendering window) to use

    scalars : str
        The name of the scalars to plot

    preference : str, optional
        The preference for data choice when search for the scalar array

    displayParams : dict
        Any plotting keyword parameters to use

    """

    def tool(self, defaultParams={}):
        preference = self.displayParams['preference']
        lowstart = ((self.valid_range[1] - self.valid_range[0]) * 0.25) + self.valid_range[0]
        highstart = ((self.valid_range[1] - self.valid_range[0]) * 0.75) + self.valid_range[0]

        # Now set up the widgets
        minsl = widgets.FloatSlider(min=self.valid_range[0],
                            max=self.valid_range[1],
                            value=lowstart,
                            continuous_update=False)
        maxsl = widgets.FloatSlider(min=self.valid_range[0],
                            max=self.valid_range[1],
                            value=highstart,
                            continuous_update=False)

        def update(dmin, dmax, invert, continuous, **kwargs):
            if dmax < dmin:
                # If user chooses a min that is more than max, correct them:
                # Set max threshold as 1 percent of the range more than min
                dmax = dmin + (self.valid_range[1] - self.valid_range[0]) * 0.01
                maxsl.value = dmax

            scalars = kwargs.get('scalars')

            # Update the sliders if scalar is changed
            self.valid_range = self.input_dataset.get_data_range(arr=scalars, preference=preference)
            if self._last_scalars != scalars:
                self._last_scalars = scalars
                kwargs['scalars'] = scalars
                # Update to the new range
                minsl.min = self.valid_range[0]
                minsl.max = self.valid_range[1]
                maxsl.min = self.valid_range[0]
                maxsl.max = self.valid_range[1]
                return update(dmin, dmax, invert, continuous, **kwargs)

            # Run the threshold
            self.output_dataset = self.input_dataset.threshold([dmin, dmax],
                    scalars=scalars, continuous=continuous, preference=preference,
                    invert=invert)

            # Update the plotter
            self._update_plotting_params(**kwargs)
            self.plotter.remove_actor(self._data_to_update, resetcam=False)
            if self.output_dataset.n_points == 0 and self.output_dataset.n_cells == 0:
                pass
            else:
                self._data_to_update = self.plotter.add_mesh(self.output_dataset,
                    resetcam=False, **self.displayParams)

            self._need_to_update = False


        # Only give scalar options that have a varying range
        names = []
        for name in self.input_dataset.scalar_names:
            arr = self.input_dataset.get_scalar(name)
            rng = self.input_dataset.get_data_range(name)
            if arr is not None and arr.size > 0 and (rng[1]-rng[0] > 0.0):
                names.append(name)

        self._last_scalars = names[0]

        # Create/display the widgets
        interact(update, dmin=minsl, dmax=maxsl,
                 scalars=names,
                 invert=defaultParams.get('invert', False),
                 continuous=False)
