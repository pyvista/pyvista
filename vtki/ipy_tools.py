"""
A set of useful plotting tools and widgets that can be used in a Jupyter
notebook
"""

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import collections

import vtki


def orthographic_slicer(dataset, plotter=None, threshold=True, step=None,
                        scalars=None, preference='cell', plotParams={}):
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
    slices = plotter.add_mesh(dataset.slice_orthographic(), showedges=False, **plotParams)
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
        slices = plotter.add_mesh(dataset.slice_orthographic(x,y,z), showedges=False, **plotParams)

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

    return interact(update_slices, x=xsl, y=ysl, z=zsl);
