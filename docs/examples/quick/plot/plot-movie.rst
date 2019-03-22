Creating a Movie
----------------


.. testcode:: python

    import vtki
    import numpy as np

    filename = './images/sphere-shrinking.mp4'

    mesh = vtki.Sphere()
    mesh.cell_arrays['data'] = np.random.random(mesh.n_cells)

    plotter = vtki.Plotter()
    # Open a moview file
    plotter.open_movie(filename)

    # Add intial mesh
    plotter.add_mesh(mesh, scalars='data', clim=[0, 1])
    # Add outline for shrinking reference
    plotter.add_mesh(mesh.outline_corners())

    # Render and do NOT close
    plotter.plot(auto_close=False)

    # Run through each frame
    plotter.write_frame() # write intial data

    # Update scalars on each frame
    for i in range(100):
        random_points = np.random.random(mesh.points.shape)
        mesh.points = random_points*0.01 + mesh.points*0.99
        mesh.points -= mesh.points.mean(0)
        mesh.cell_arrays['data'] = np.random.random(mesh.n_cells)
        plotter.write_frame() # Write this frame

    plotter.close()


.. figure:: ../../../images/sphere-shrinking.mp4
