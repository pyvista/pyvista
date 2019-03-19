Plot Time Series Data
---------------------

This example outlines how to plot data where the spatial reference and data
values change through time:


.. code-block:: python

    from threading import Thread
    import time
    import numpy as np
    import vtki
    from vtki import examples


    globe = examples.load_globe()
    globe.point_arrays['scalars'] = np.random.rand(globe.n_points)
    globe.set_active_scalar('scalars')


    plotter = vtki.BackgroundPlotter()
    plotter.add_mesh(globe, lighting=False, show_edges=True, texture=True, scalars='scalars')
    plotter.view_isometric()

    # shrink globe in the background
    def shrink():
        for i in range(50):
            globe.points *= 0.95
            # Update scalars
            globe.point_arrays['scalars'] = np.random.rand(globe.n_points)
            time.sleep(0.5)

    thread = Thread(target=shrink)
    thread.start()


.. figure:: ../../../images/gifs/shrink-globe.gif
