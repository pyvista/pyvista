

Clearing a Mesh or the Entire Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Removing a single actor:

.. code:: python

    import vtki
    plotter = vtki.Plotter(notebook=True)
    actor = plotter.add_mesh(vtki.Sphere())
    plotter.remove_actor(actor)
    plotter.show()


Clearing the entire plotting window:

.. code:: python

    import vtki
    plotter = vtki.Plotter(notebook=True)
    plotter.add_mesh(vtki.Sphere())
    plotter.add_mesh(vtki.Plane())
    plotter.clear()  # clears all actors
    plotter.show()


Or you can give any actor a ``name`` when adding it and if an actor is added
with that same name at a later time, it will replace the previous actor:

.. testcode:: python

    import vtki
    plotter = vtki.Plotter(notebook=True)
    plotter.add_mesh(vtki.Sphere(), name='mydata')
    plotter.add_mesh(vtki.Plane(), name='mydata')
    # Only the Plane is shown!
