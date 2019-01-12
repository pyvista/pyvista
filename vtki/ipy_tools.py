"""
A set of useful plotting tools and widgets that can be used in a Jupyter
notebook
"""

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import collections

import vtki
from vtki.utilities import is_vtki_obj, wrap


def orthogonal_slicer(dataset, plotter=None, threshold=True, step=None,
                        scalars=None, preference='cell', plotParams={}):
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
    if not is_vtki_obj(dataset):
        dataset = wrap(dataset)
        if not is_vtki_obj(dataset):
            raise RuntimeError('Object not supported for plotting in vtki.')
    if threshold:
        # This will clean out the nan values
        dataset = dataset.threshold()
    if plotter is None:
        plotter = vtki.BackgroundPlotter()

    plotParams.setdefault('rng', dataset.get_data_range(scalars, preference=preference))
    plotParams.setdefault('scalars', scalars)
    plotParams.setdefault('preference', preference)

    # Add dataset to the rendering window
    global slices
    slices = plotter.add_mesh(dataset.slice_orthogonal(), showedges=False, **plotParams)
    outline = plotter.add_mesh(dataset.outline_corners())
    # add the axis labels
    plotter.add_bounds_axes()
    cpos = plotter.get_default_cam_pos()
    plotter.camera_position = cpos
    plotter.reset_camera()

    # Now set up the widgets
    def update_slices(x, y, z):
        global slices
        plotter.remove_actor(slices)
        slices = plotter.add_mesh(dataset.slice_orthogonal(x,y,z), showedges=False, **plotParams)

    if step is None:
        stepx = 0.05 * (dataset.bounds[1] - dataset.bounds[0])
        stepy = 0.05 * (dataset.bounds[3] - dataset.bounds[2])
        stepz = 0.05 * (dataset.bounds[5] - dataset.bounds[4])
    elif isinstance(step, collections.Iterable):
        stepx = step[0]
        stepy = step[1]
        stepz = step[2]
    else:
        stepx = step
        stepy = step
        stepz = step

    xsl = widgets.FloatSlider(min=dataset.bounds[0]+stepx,
                        max=dataset.bounds[1]-stepx,
                        step=stepx,
                        value=dataset.center[0])
    ysl = widgets.FloatSlider(min=dataset.bounds[2]+stepy,
                        max=dataset.bounds[3]-stepy,
                        step=stepy,
                        value=dataset.center[1])
    zsl = widgets.FloatSlider(min=dataset.bounds[4]+stepz,
                        max=dataset.bounds[5]-stepz,
                        step=stepz,
                        value=dataset.center[2])

    interact(update_slices, x=xsl, y=ysl, z=zsl);
