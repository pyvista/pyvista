"""
A set of useful plotting tools and widgets that can be used in a Jupyter
notebook
"""

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import collections

import vtk

import vtki
from vtki.utilities import is_vtki_obj, wrap


class InteractiveTool(object):
    """A backend helper for various interactive ipython tools"""

    def __init__(self, dataset, plotter=None, scalars=None, preference='cell', plotParams={}, **kwargs):
        if not is_vtki_obj(dataset):
            dataset = wrap(dataset)
            if not is_vtki_obj(dataset):
                raise RuntimeError('Object not supported for plotting in vtki.')
        self.input_dataset = dataset
        self.output_dataset = None

        if plotter is None:
            plotter = vtki.BackgroundPlotter()
            plotter.setWindowTitle(type(self).__name__)
        self.plotter = plotter

        self._data_to_update = None

        plotParams.setdefault('rng', self.input_dataset.get_data_range(scalars, preference=preference))
        plotParams.setdefault('scalars', scalars)
        plotParams.setdefault('preference', preference)
        self.plotParams = plotParams

        self._initialize()

        self.tool(**kwargs)

    def tool():
        raise NotImplementedError('This method has not been implemented')

    def _initialize(self):
        outline = self.plotter.add_mesh(self.input_dataset.outline_corners())
        # add the axis labels
        self.plotter.add_bounds_axes()
        cpos = self.plotter.get_default_cam_pos()
        self.plotter.camera_position = cpos
        self.plotter.reset_camera()
        self.plotter.camera_set = False



class OthogonalSlicer(InteractiveTool):
    """Within ipython enviornments like Jupyter notebooks, this will create
    an interactive render window with slider bars in te ipython enviornment to
    move orthogonal slices through the scene.

    Parameters
    ----------
    dataset : vtki.Common
        The datset to orthogonalally slice

    plotter : vtki.BasePlotter
        The active plotter (rendering window) to use

    threshold : bool, optional
        This will apply a threshold on the input dataset to remove any NaN
        values. Default is True.

    step : float or tuple(float)
        The increments for the XYZ locations on each of the slider bars

    scalars : str
        The name of the scalars to plot

    preference : str, optional
        The preference for data choice when search for the scalar array

    plotParams : dict
        Any plotting keyword parameters to use

    """

    def tool(self, threshold=True, step=None):
        if threshold:
            # This will clean out the nan values
            self.input_dataset = self.input_dataset.threshold()

        x, y, z = self.input_dataset.center

        self._data_to_update = [None, None, None]

        self._old_x = None
        self._old_y = None
        self._old_z = None

        def update(x, y, z):
            if x != self._old_x:
                self.plotter.remove_actor(self._data_to_update[0])
                self.slice_yz = self.input_dataset.slice(normal='x', origin=[x,y,z])
                self._data_to_update[0] = self.plotter.add_mesh(self.slice_yz,
                        showedges=False, resetcam=False, **self.plotParams)
                self._old_x = x
            if y != self._old_y:
                self.plotter.remove_actor(self._data_to_update[1])
                self.slice_xz = self.input_dataset.slice(normal='y', origin=[x,y,z])
                self._data_to_update[1] = self.plotter.add_mesh(self.slice_xz,
                        showedges=False, resetcam=False, **self.plotParams)
                self._old_y = y
            if z != self._old_z:
                self.plotter.remove_actor(self._data_to_update[2])
                self.slice_xy = self.input_dataset.slice(normal='z', origin=[x,y,z])
                self._data_to_update[2] = self.plotter.add_mesh(self.slice_xy,
                        showedges=False, resetcam=False, **self.plotParams)
                self._old_z = z

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

        interact(update, x=xsl, y=ysl, z=zsl)


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

    threshold : bool, optional
        This will apply a threshold on the input dataset to remove any NaN
        values. Default is True.

    tol : float, optional
        The tolerance to the edge of the dataset bounds to create the slices

    scalars : str
        The name of the scalars to plot

    preference : str, optional
        The preference for data choice when search for the scalar array

    plotParams : dict
        Any plotting keyword parameters to use

    """

    def tool(self, threshold=True, tol=1e-3):
        if threshold:
            # This will clean out the nan values
            self.input_dataset = self.input_dataset.threshold()

        def update(n, axis):
            self.plotter.remove_actor(self._data_to_update)
            self.output_dataset = self.input_dataset.slice_along_axis(n=n, axis=axis, tol=tol)
            self._data_to_update = self.plotter.add_mesh(self.output_dataset, showedges=False, resetcam=False, **self.plotParams)

        #continuous_update=False
        nsl = widgets.IntSlider(min=1, max=25, step=1, value=5,
                                continuous_update=False)

        interact(update, n=nsl, axis=['x', 'y', 'z'])


class Threshold(InteractiveTool):
    """Allows user to use slider bars to control the threshold range.

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

    plotParams : dict
        Any plotting keyword parameters to use

    """

    def tool(self):
        preference = self.plotParams['preference']
        # Basically set up two threshold filters

        self.valid_range = self.input_dataset.get_data_range(arr=self.plotParams['scalars'], preference=preference)
        # Now set up the widgets
        minsl = widgets.FloatSlider(min=self.valid_range[0],
                            max=self.valid_range[1],
                            value=self.valid_range[0],
                            continuous_update=False)
        maxsl = widgets.FloatSlider(min=self.valid_range[0],
                            max=self.valid_range[1],
                            value=self.valid_range[1],
                            continuous_update=False)

        def update(dmin, dmax, scalars, invert, continuous):
            if dmax < dmin:
                foo = dmax
                dmax = dmin
                dmin = foo
                invert = True

            # Update the sliders if scalar is changed
            self.valid_range = self.input_dataset.get_data_range(arr=scalars, preference=preference)
            minsl.min = self.valid_range[0]
            minsl.max = self.valid_range[1]
            maxsl.min = self.valid_range[0]
            maxsl.max = self.valid_range[1]

            if invert:
                # raise RuntimeError('Not supported')
                # Create two thresholds and merge result
                t1 = self.input_dataset.threshold([self.valid_range[0], dmin], scalars=scalars, continuous=continuous, preference=preference)
                t2 = self.input_dataset.threshold([dmax, self.valid_range[1]], scalars=scalars, continuous=continuous, preference=preference)
                appender = vtk.vtkAppendFilter()
                appender.AddInputData(t1)
                appender.AddInputData(t2)
                appender.Update()
                self.output_dataset =  appender.GetOutputDataObject(0)
            else:
                self.output_dataset = self.input_dataset.threshold([dmin, dmax], scalars=scalars, continuous=continuous, preference=preference)
            # Update the plot
            self.plotParams['scalars'] = scalars
            self.plotter.remove_actor(self._data_to_update)
            self._data_to_update = self.plotter.add_mesh(self.output_dataset, **self.plotParams)


        interact(update, dmin=minsl, dmax=maxsl,
                 scalars=self.input_dataset.scalar_names, invert=False,
                 continuous=False)
