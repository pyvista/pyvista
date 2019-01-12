"""
A set of useful plotting tools and widgets that can be used in a Jupyter
notebook
"""

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import collections

import vtki
from vtki.utilities import is_vtki_obj, wrap


class InteractiveTool(object):
    """A backend helper for various interactive ipython tools"""

    def __init__(self, dataset, plotter=None, **kwargs):
        if not is_vtki_obj(dataset):
            dataset = wrap(dataset)
            if not is_vtki_obj(dataset):
                raise RuntimeError('Object not supported for plotting in vtki.')
        self.input_dataset = dataset

        if plotter is None:
            plotter = vtki.BackgroundPlotter()
            plotter.setWindowTitle(type(self).__name__)
        self.plotter = plotter

        self._data_to_update = None

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

    def tool(self, threshold=True, step=None, scalars=None, preference='cell', plotParams={}):
        if threshold:
            # This will clean out the nan values
            self.input_dataset = self.input_dataset.threshold()

        plotParams.setdefault('rng', self.input_dataset.get_data_range(scalars, preference=preference))
        plotParams.setdefault('scalars', scalars)
        plotParams.setdefault('preference', preference)

        self._initialize()


        # Now set up the widgets
        def update(x, y, z):
            self.plotter.remove_actor(self._data_to_update)
            self._data_to_update = self.plotter.add_mesh(self.input_dataset.slice_orthogonal(x,y,z), showedges=False, resetcam=False, **plotParams)

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

        xsl = widgets.FloatSlider(min=self.input_dataset.bounds[0]+stepx,
                            max=self.input_dataset.bounds[1]-stepx,
                            step=stepx,
                            value=self.input_dataset.center[0])
        ysl = widgets.FloatSlider(min=self.input_dataset.bounds[2]+stepy,
                            max=self.input_dataset.bounds[3]-stepy,
                            step=stepy,
                            value=self.input_dataset.center[1])
        zsl = widgets.FloatSlider(min=self.input_dataset.bounds[4]+stepz,
                            max=self.input_dataset.bounds[5]-stepz,
                            step=stepz,
                            value=self.input_dataset.center[2])

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

    def tool(self, threshold=True, tol=1e-3, scalars=None, preference='cell', plotParams={}):
        if threshold:
            # This will clean out the nan values
            self.input_dataset = self.input_dataset.threshold()

        plotParams.setdefault('rng', self.input_dataset.get_data_range(scalars, preference=preference))
        plotParams.setdefault('scalars', scalars)
        plotParams.setdefault('preference', preference)

        self._initialize()

        # Now set up the widgets
        def update(n, axis):
            self.plotter.remove_actor(self._data_to_update)
            self._data_to_update = self.plotter.add_mesh(self.input_dataset.slice_along_axis(n=n, axis=axis, tol=tol), showedges=False, resetcam=False, **plotParams)

        nsl = widgets.IntSlider(min=1, max=25, step=1, value=5)

        interact(update, n=nsl, axis=['x', 'y', 'z'])
