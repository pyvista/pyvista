.. _userguide_themes:

Plotting Themes
===============

PyVista plotting parameters can be controlled on a plot by plot basis
or through a global theme, making it possible to control mesh colors
and styles through one global configuration.

The default theme parameters in PyVista can be accessed and displayed with:

.. code:: python

   >>> import pyvista
   >>> pyvista.global_theme

Default plotting parameters can be accessed individually by their
attribute names:

.. code:: python

   >>> pyvista.global_theme.color = 'tan'

Here's an example plot of the Stanford Dragon using default plotting
parameters:

.. pyvista-plot::
   :context:

   >>> import pyvista
   >>> from pyvista import examples
   >>> dragon = examples.download_dragon()
   >>> dragon.plot(cpos='xy')

These parameters can then be modified globally with:

.. pyvista-plot::
   :context:

   >>> pyvista.global_theme.color = 'red'
   >>> pyvista.global_theme.background = 'white'
   >>> pyvista.global_theme.axes.show = False

Now, the mesh will be plotted with the new global parameters:

.. pyvista-plot::
   :context:

   >>> dragon.plot(cpos='xy')

This is identical to plotting the mesh with the following parameters:

.. pyvista-plot::
   :context:

   >>> dragon.plot(cpos='xy', color='red', background='white', show_axes=False)


Creating A Custom Theme
-----------------------
You can customize a theme based on one of the built-in themes and then
apply it globally with:

.. code:: python

   Create a theme based off the DocumentTheme

   >>> my_theme = pyvista.themes.DocumentTheme()
   >>> my_theme.cmap = 'jet'
   >>> my_theme.show_edges = True

   Apply it globally

   >>> pyvista.global_theme.load_theme(my_theme)

Alternatively, you can save the theme to disk to be used later with:

.. code:: python

   >>> my_theme.save('my_theme.json')

And then subsequently loaded in a new session of pyvista with:

.. code:: python

   >>> pyvista.global_theme.load_theme('my_theme.json')


Theme API
---------
See :ref:`theme_api` for the full API definition.
