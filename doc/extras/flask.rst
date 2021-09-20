.. _ref_flask:

Flask Application
=================
You can use ``pyvista`` in to make a flask application to display
static plots.  See the following example as well as the demo at `Flask
Example <https://github.com/pyvista/pyvista/tree/main/examples_flask>`__.

For dynamic examples, it's recommended to use `Jupyter Notebooks <https://jupyter.org/>`__.  See our documentation regarding this at :ref:`jupyter_plotting`.


.. figure:: ../images/user-generated/flask_example.png
    :width: 500pt

    Example Flask Webpage


Python Application ``app.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples_flask/app.py


Ajax Template ``index.html``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This template should be within the ``templates`` directory in the same
path as ``app.py``.

This template returns the ``meshtype`` parameter back to the
``get_img`` method in the flask app, which is used to select the type
of mesh to be plotted.

.. literalinclude:: ../../examples_flask/templates/index.html
