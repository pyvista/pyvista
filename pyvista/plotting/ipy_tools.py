"""
A set of useful plotting tools and widgets that can be used in a
Jupyter notebook.
"""
IPY_AVAILABLE = False
try:
    from ipywidgets import interact, interactive
    import ipywidgets as widgets
    IPY_AVAILABLE = True
except ImportError:
    pass

import collections
import logging

import numpy as np

import pyvista
from pyvista.plotting import run_from_ipython
from pyvista.utilities import is_pyvista_obj, wrap


class ScaledPlotter(pyvista.BackgroundPlotter):
    """
    An extension of the ``pyvista.BackgroundPlotter`` that has
    interactive widgets for scaling the axes in the rendering scene.
    """
    def __init__(self, xscale=1.0, yscale=1.0, zscale=1.0, show=True, app=None,
                 continuous_update=False, **kwargs):
        if not run_from_ipython() or not IPY_AVAILABLE:
            logging.warning('Interactive plotting tools require IPython and the ``ipywidgets`` package.')
        pyvista.BackgroundPlotter.__init__(self, show=show, app=app, **kwargs)
        # Now set up the IPython scaling widgets
        self.continuous_update = bool(continuous_update)
        self.xslider = widgets.FloatSlider(min=0, max=xscale*2, value=xscale,
                                continuous_update=self.continuous_update)
        self.yslider = widgets.FloatSlider(min=0, max=yscale*2, value=yscale,
                                continuous_update=self.continuous_update)
        self.zslider = widgets.FloatSlider(min=0, max=zscale*2, value=zscale,
                                continuous_update=self.continuous_update)

        def update(xscale, yscale, zscale):
            """Update the scales"""
            # Update max range if needed
            if xscale >= self.xslider.max:
                self.xslider.max *= 2
            if yscale >= self.yslider.max:
                self.yslider.max *= 2
            if zscale >= self.zslider.max:
                self.zslider.max *= 2
            # reset max range if needed
            if xscale < self.xslider.max * 0.10 and xscale > 1.0:
                self.xslider.max /= 2
            if yscale < self.yslider.max * 0.10 and yscale > 1.0:
                self.yslider.max /= 2
            if zscale < self.zslider.max * 0.10 and zscale > 1.0:
                self.zslider.max /= 2
            self.set_scale(xscale, yscale, zscale)

        # Create/display the widgets
        interact(update, xscale=self.xslider, yscale=self.yslider,
                        zscale=self.zslider, **kwargs)


class InteractiveTool(object):
    """A backend helper for various interactive ipython tools.
    This tool can be added to an active plotter in the background if passed as
    the ``plotter`` argument.
    """

    def __new__(cls, *args, **kwargs):
        if cls is InteractiveTool:
            raise TypeError("pyvista.InteractiveTool is an abstract class and may not be instantiated.")
        return object.__new__(cls)

    def __init__(self, dataset, plotter=None, scalars=None, preference='cell',
                 show_bounds=False, reset_camera=True, outline=None,
                 display_params=None, default_params=None,
                 continuous_update=False, clean=True, **kwargs):
        if not run_from_ipython() or not IPY_AVAILABLE:
            logging.warning('Interactive plotting tools require IPython and the ``ipywidgets`` package.')
        # Check the input dataset to make sure its compatible
        if not is_pyvista_obj(dataset):
            dataset = wrap(dataset)
            if not is_pyvista_obj(dataset):
                raise RuntimeError('Object not supported for plotting in pyvista.')

        # Make the input/output of this tool available
        self.input_dataset = dataset
        self.output_dataset = None

        self.continuous_update = continuous_update

        if plotter is None:
            plotter = pyvista.BackgroundPlotter(**kwargs)
            plotter.setWindowTitle(type(self).__name__)
        self._plotter = plotter
        self._loc = plotter.index_to_loc(plotter._active_renderer_index)

        # This is the actor that will be removed and re-added to the plotter
        self._data_to_update = None

        self._last_scalars = None

        # Intialize plotting parameters
        self.valid_range = self.input_dataset.get_data_range(arr=scalars, preference=preference)
        if display_params is None:
            display_params = {}
        if default_params is None:
            default_params = {}
        display_params.setdefault('rng', self.valid_range)
        display_params.setdefault('scalars', scalars)
        display_params.setdefault('preference', preference)
        display_params.setdefault('show_edges', False)
        # Make sure to remove the reset_camera parameter if present
        display_params.pop('reset_camera', None)
        # Make a name
        display_params.setdefault('name', '{}({})'.format(type(self).__name__, str(hex(id(self)))))
        self.display_params = display_params

        self.clean = clean

        # Set the tool status
        self._need_to_update = True

        # Add some intital plotting stuff to the scene
        self._initialize(show_bounds, reset_camera, outline)

        # Run the tool
        self.tool(default_params=default_params, **kwargs)


    def _get_scalar_names(self, limit=None):
        """Only give scalar options that have a varying range"""
        names = []
        if limit == 'point':
            inpnames = list(self.input_dataset.point_arrays.keys())
        elif limit == 'cell':
            inpnames = list(self.input_dataset.cell_arrays.keys())
        else:
            inpnames = self.input_dataset.scalar_names
        for name in inpnames:
            arr = self.input_dataset.get_scalar(name)
            rng = self.input_dataset.get_data_range(name)
            if arr is not None and arr.size > 0 and (rng[1]-rng[0] > 0.0):
                names.append(name)
        try:
            self._last_scalars = names[0]
        except IndexError:
            pass
        return names


    def tool(**kwargs):
        """This method is implemented for each tool to perfrom the data
        filtering and setting up the widgets"""
        raise NotImplementedError('This method has not been implemented')


    def _initialize(self, show_bounds, reset_camera, outline):
        """Outlines the input dataset and sets up the scene"""
        self.plotter.subplot(*self.loc)
        if outline is None:
            self.plotter.add_mesh(self.input_dataset.outline_corners(),
                    reset_camera=False, color=pyvista.rcParams['outline_color'],
                    loc=self.loc)
        elif outline:
            self.plotter.add_mesh(self.input_dataset.outline(),
                    reset_camera=False, color=pyvista.rcParams['outline_color'],
                    loc=self.loc)
        # add the axis labels
        if show_bounds:
            self.plotter.show_bounds(reset_camera=False, loc=loc)
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
            old = self.display_params['scalars']
            self.display_params['scalars'] = scalars
            if old != scalars:
                self.plotter.subplot(*self.loc)
                self.plotter.remove_actor(self._data_to_update, reset_camera=False)
                self._need_to_update = True
                self.valid_range = self.input_dataset.get_data_range(scalars)
                # self.display_params['rng'] = self.valid_range
        cmap = kwargs.get('cmap', None)
        if cmap is not None:
            self.display_params['cmap'] = cmap


    @property
    def plotter(self):
        """The active plotter that this tool uses. This must be set upon
        instantiation
        """
        return self._plotter

    @property
    def loc(self):
        """The location in the plotter window."""
        return self._loc

    @property
    def renderer(self):
        return self.plotter.renderers[self.loc]



class OrthogonalSlicer(InteractiveTool):
    """Within ipython enviornments like Jupyter notebooks, this will create
    an interactive render window with slider bars in te ipython enviornment to
    move orthogonal slices through the scene.

    Parameters
    ----------
    dataset : pyvista.Common
        The datset to orthogonalally slice

    plotter : pyvista.BasePlotter
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

    generate_triangles: bool, optional
        If this is enabled (``False`` by default), the output will be
        triangles otherwise, the output will be the intersection polygons.

    display_params : dict
        Any plotting keyword parameters to use

    contour : bool, optional
        If True, apply a ``contour`` filter after slicing

    """

    def tool(self, step=None, generate_triangles=False, contour=False,
             default_params=None, **kwargs):
        if default_params is None:
            default_params = {}
        if self.clean and self.input_dataset.active_scalar is not None:
            # This will clean out the nan values
            self.input_dataset = self.input_dataset.threshold()

        self.contour = contour

        x, y, z = self.input_dataset.center
        x = default_params.get("x", x)
        y = default_params.get("y", y)
        z = default_params.get("z", z)

        self._data_to_update = [None, None, None]
        self.output_dataset = pyvista.MultiBlock()
        self._old = [None, None, None]

        axes = ['x', 'y', 'z']

        def _update_slice(index, x, y, z):
            name = self.display_params.pop('name')
            self.plotter.subplot(*self.loc)
            self.plotter.remove_actor(self._data_to_update[index], reset_camera=False)
            self.output_dataset[index] = self.input_dataset.slice(normal=axes[index],
                    origin=[x,y,z], generate_triangles=generate_triangles, contour=self.contour)
            self._data_to_update[index] = self.plotter.add_mesh(self.output_dataset[index],
                    reset_camera=False, name='{}-{}'.format(name, index), **self.display_params)
            self._old[index] = [x,y,z][index]
            self.display_params['name'] = name

        def update(x, y, z, **kwargs):
            """Update the slices"""
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
                            value=x,
                            continuous_update=self.continuous_update)
        ysl = widgets.FloatSlider(min=self.input_dataset.bounds[2]+stepy,
                            max=self.input_dataset.bounds[3]-stepy,
                            step=stepy,
                            value=y,
                            continuous_update=self.continuous_update)
        zsl = widgets.FloatSlider(min=self.input_dataset.bounds[4]+stepz,
                            max=self.input_dataset.bounds[5]-stepz,
                            step=stepz,
                            value=z,
                            continuous_update=self.continuous_update)

        # Create/display the widgets
        return interact(update, x=xsl, y=ysl, z=zsl,
                        scalars=self._get_scalar_names())


class ManySlicesAlongAxis(InteractiveTool):
    """Within ipython enviornments like Jupyter notebooks, this will create
    an interactive render window with slider bars in te ipython enviornment to
    create many slices along a specified axis.

    Parameters
    ----------
    dataset : pyvista.Common
        The datset to orthogonalally slice

    plotter : pyvista.BasePlotter
        The active plotter (rendering window) to use

    clean : bool, optional
        This will apply a threshold on the input dataset to remove any NaN
        values. Default is True if active scalar present.

    tolerance : float, optional
        The tolerance to the edge of the dataset bounds to create the slices

    scalars : str
        The name of the scalars to plot

    preference : str, optional
        The preference for data choice when search for the scalar array

    generate_triangles: bool, optional
        If this is enabled (``False`` by default), the output will be
        triangles otherwise, the output will be the intersection polygons.

    display_params : dict
        Any plotting keyword parameters to use

    contour : bool, optional
        If True, apply a ``contour`` filter after slicing

    """

    def tool(self, tolerance=None, generate_triangles=False, contour=False,
             default_params=None, **kwargs):
        if default_params is None:
            default_params = {}
        if self.clean and self.input_dataset.active_scalar is not None:
            # This will clean out the nan values
            self.input_dataset = self.input_dataset.threshold()

        self.contour = contour

        n = default_params.get("n", 5)
        axis = default_params.get("axis", "x")
        nsl = widgets.IntSlider(min=1, max=n*2, step=1, value=n,
                                continuous_update=self.continuous_update)

        def update(n, axis, **kwargs):
            """Update the slices"""
            if n >= nsl.max:
                nsl.max *= 2
            self.plotter.subplot(*self.loc)
            self._update_plotting_params(**kwargs)
            self.plotter.remove_actor(self._data_to_update, reset_camera=False)
            self.output_dataset = self.input_dataset.slice_along_axis(n=n, axis=axis, tolerance=tolerance, generate_triangles=generate_triangles, contour=self.contour)
            self._data_to_update = self.plotter.add_mesh(self.output_dataset,
                reset_camera=False, loc=self.loc, **self.display_params)
            self._need_to_update = False

        # Create/display the widgets
        axes = ['x', 'y', 'z']
        idx = axes.index(axis)
        del axes[idx]
        axes.insert(0, axis)
        return interact(update, n=nsl, axis=axes,
                        scalars=self._get_scalar_names())


class Threshold(InteractiveTool):
    """Yields slider bars for user to control the threshold range in an
    interactive plot. The threshold will default at 25 and 75 percent of the
    range.

    Parameters
    ----------
    dataset : pyvista.Common
        The datset to orthogonalally slice

    plotter : pyvista.BasePlotter
        The active plotter (rendering window) to use

    scalars : str
        The name of the scalars to plot

    preference : str, optional
        The preference for data choice when search for the scalar array

    display_params : dict
        Any plotting keyword parameters to use

    """

    def tool(self, default_params=None, **kwargs):
        if default_params is None:
            default_params = {}
        preference = self.display_params['preference']

        def _calc_start_values(rng):
            """Get starting values for sliders use a data range"""
            lowstart = ((rng[1] - rng[0]) * 0.25) + rng[0]
            highstart = ((rng[1] - rng[0]) * 0.75) + rng[0]
            return lowstart, highstart

        # Now set up the widgets
        lowstart, highstart = _calc_start_values(self.valid_range)
        lowstart = default_params.get("dmin", lowstart)
        highstart = default_params.get("dmax", highstart)
        minsl = widgets.FloatSlider(min=self.valid_range[0],
                            max=self.valid_range[1],
                            value=lowstart,
                            continuous_update=self.continuous_update)
        maxsl = widgets.FloatSlider(min=self.valid_range[0],
                            max=self.valid_range[1],
                            value=highstart,
                            continuous_update=self.continuous_update)

        def _update_slider_ranges(new_rng):
            """Updates the slider ranges when switching scalars"""
            vmin, vmax = np.nanmin([new_rng[0], minsl.min]), np.nanmax([new_rng[1], minsl.max])
            # Update to the total range
            minsl.min = vmin
            minsl.max = vmax
            maxsl.min = vmin
            maxsl.max = vmax
            lowstart, highstart = _calc_start_values(new_rng)
            minsl.value = lowstart
            maxsl.value = highstart
            minsl.min = new_rng[0]
            minsl.max = new_rng[1]
            maxsl.min = new_rng[0]
            maxsl.max = new_rng[1]
            return lowstart, highstart


        def update(dmin, dmax, invert, continuous, **kwargs):
            """Update the threshold"""
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
                # Update to the new range
                dmin, dmax = _update_slider_ranges(self.input_dataset.get_data_range(scalars))

            # Run the threshold
            self.output_dataset = self.input_dataset.threshold([dmin, dmax],
                    scalars=scalars, continuous=continuous, preference=preference,
                    invert=invert)

            # Update the plotter
            self.plotter.subplot(*self.loc)
            self._update_plotting_params(**kwargs)
            self.plotter.remove_actor(self._data_to_update, reset_camera=False)
            if self.output_dataset.n_points == 0 and self.output_dataset.n_cells == 0:
                pass
            else:
                self._data_to_update = self.plotter.add_mesh(self.output_dataset,
                    reset_camera=False, loc=self.loc, **self.display_params)

            self._need_to_update = False


        # Create/display the widgets
        scalars = self._get_scalar_names()
        name = default_params.get("scalars", scalars[0])
        idx = scalars.index(name)
        del scalars[idx]
        scalars.insert(0, name)
        return interact(update, dmin=minsl, dmax=maxsl,
                        scalars=scalars,
                        invert=default_params.get('invert', False),
                        continuous=False)


class Clip(InteractiveTool):
    """Clips a dataset along an axis.

    Within ipython enviornments like Jupyter notebooks, this will create
    an interactive render window with slider bars in te ipython enviornment to
    create clip a dataset.

    Parameters
    ----------
    dataset : pyvista.Common
        The datset to orthogonalally slice

    plotter : pyvista.BasePlotter
        The active plotter (rendering window) to use

    clean : bool, optional
        This will apply a threshold on the input dataset to remove any NaN
        values. Default is True if active scalar present.

    scalars : str
        The name of the scalars to plot

    preference : str, optional
        The preference for data choice when search for the scalar array

    display_params : dict
        Any plotting keyword parameters to use

    """

    def tool(self, default_params=None, **kwargs):
        if default_params is None:
            default_params = {}
        if self.clean and self.input_dataset.active_scalar is not None:
            # This will clean out the nan values
            self.input_dataset = self.input_dataset.threshold()

        bnds = self.input_dataset.bounds
        locataion = default_params.get("location", self.input_dataset.center[0])
        axchoices = ['x', 'y', 'z']

        locsl = widgets.FloatSlider(min=bnds[0],
                            max=bnds[1],
                            value=locataion,
                            continuous_update=self.continuous_update)

        def _update_slider_ranges(normal):
            """Update the sliders ranges"""
            ax = axchoices.index(normal)
            new_rng = bnds[2*ax:2*ax+2]
            vmin, vmax = np.nanmin([new_rng[0], locsl.min]), np.nanmax([new_rng[1], locsl.max])
            # Update to the total range
            locsl.min = vmin
            locsl.max = vmax
            locsl.value = self.input_dataset.center[ax]
            locsl.min = new_rng[0]
            locsl.max = new_rng[1]
            return

        def update(location, normal, invert, **kwargs):
            """Update the clip location"""
            if self._last_normal != normal:
                self._last_normal = normal
                _update_slider_ranges(normal)
                return update(locsl.value, normal, invert, **kwargs)
            self.plotter.subplot(*self.loc)
            self._update_plotting_params(**kwargs)
            self.plotter.remove_actor(self._data_to_update, reset_camera=False)
            origin = list(self.input_dataset.center)
            origin[axchoices.index(normal)] = location
            self.output_dataset = self.input_dataset.clip(normal=normal, origin=origin, invert=invert)
            self._data_to_update = self.plotter.add_mesh(self.output_dataset,
                reset_camera=False, loc=self.loc, **self.display_params)
            self._need_to_update = False

        # Create/display the widgets
        self._last_normal = 'x'
        return interact(update, location=locsl, normal=axchoices, invert=True,
                        scalars=self._get_scalar_names())




class Isocontour(InteractiveTool):
    """Create a sinlge iso-value contour of a dataset. Contouring only supports
    point data attributes.
    Within ipython enviornments like Jupyter notebooks, this will create
    an interactive render window with slider bars in the ipython enviornment to
    contour a volumetric dataset.

    Parameters
    ----------
    dataset : pyvista.Common
        The datset to orthogonalally slice

    plotter : pyvista.BasePlotter
        The active plotter (rendering window) to use

    clean : bool, optional
        This will apply a threshold on the input dataset to remove any NaN
        values. Default is True if active scalar present.

    scalars : str
        The name of the scalars to plot

    preference : str, optional
        The preference for data choice when search for the scalar array

    display_params : dict
        Any plotting keyword parameters to use

    """

    def tool(self, default_params=None, **kwargs):
        if default_params is None:
            default_params = {}
        preference = self.display_params['preference']

        def _calc_start_value(rng):
            """Get starting value for slider using a data range"""
            return ((rng[1] - rng[0]) * 0.5) + rng[0]

        # Now set up the widget
        start = _calc_start_value(self.valid_range)
        start = default_params.get("value", start)
        valsl = widgets.FloatSlider(min=self.valid_range[0],
                            max=self.valid_range[1],
                            value=start,
                            continuous_update=self.continuous_update)

        def _update_slider_range(new_rng):
            """Updates the slider ranges when switching scalars"""
            vmin, vmax = np.nanmin([new_rng[0], valsl.min]), np.nanmax([new_rng[1], valsl.max])
            # Update to the total range
            valsl.min = vmin
            valsl.max = vmax
            start = _calc_start_value(new_rng)
            valsl.value = start
            valsl.min = new_rng[0]
            valsl.max = new_rng[1]
            return start


        def update(value, **kwargs):
            """Update the contour"""
            scalars = kwargs.get('scalars')

            # Update the sliders if scalar is changed
            self.valid_range = self.input_dataset.get_data_range(arr=scalars, preference=preference)
            if self._last_scalars != scalars:
                self._last_scalars = scalars
                # Update to the new range
                value = _update_slider_range(self.input_dataset.get_data_range(scalars))

            # Run the threshold
            self.output_dataset = self.input_dataset.contour([value],
                    scalars=scalars, preference=preference)

            # Update the plotter
            self.plotter.subplot(*self.loc)
            self._update_plotting_params(**kwargs)
            self.plotter.remove_actor(self._data_to_update, reset_camera=False)
            if self.output_dataset.n_points == 0 and self.output_dataset.n_cells == 0:
                pass
            else:
                self._data_to_update = self.plotter.add_mesh(self.output_dataset,
                    reset_camera=False, loc=self.loc, **self.display_params)

            self._need_to_update = False


        # Create/display the widgets
        # NOTE: Contour filter can only contour by point data
        scalars = self._get_scalar_names(limit='point')
        name = default_params.get("scalars", scalars[0])
        idx = scalars.index(name)
        del scalars[idx]
        scalars.insert(0, name)
        return interact(update, value=valsl, scalars=scalars)
