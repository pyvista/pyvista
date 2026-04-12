Getting Started
***************

This guide is here to help you start creating interactive 3D plots with PyVista
with the help of our examples and tutorials.


.. tab-set::

   .. tab-item:: JupyterLab

      Here's a quick demo of PyVista running within `JupyterLab
      <https://jupyter.org/>`_.

      .. raw:: html

         <video width="100%" height="auto" controls autoplay muted> <source
           src="https://tutorial.pyvista.org/_static/pyvista_jupyterlab_demo.mp4"
           type="video/mp4" style="margin-left: -220px; margin-right: -10.5%">
           Your browser does not support the video tag.  </video>

   .. tab-item:: IPython

      Here's a quick demo of PyVista running within a terminal using `IPython
      <https://ipython.org/>`_.

      .. raw:: html

         <video width="100%" height="auto" controls autoplay muted> <source
           src="https://tutorial.pyvista.org/_static/pyvista_ipython_demo.mp4"
           type="video/mp4"> Your browser does not support the video tag.
           </video>


.. toctree::
   :hidden:

   why
   authors
   installation
   connections
   external_examples


Installation
============
The only prerequisite for installing PyVista is Python itself. If you don’t
have Python yet and want the simplest way to get started, we recommend you use
the `Anaconda Distribution <https://www.anaconda.com/>`_.

.. grid:: 2

    .. grid-item-card:: Working with conda?
       :class-title: pyvista-card-title

       PyVista is available on `conda-forge
       <https://anaconda.org/conda-forge/pyvista>`_.

       .. code-block:: bash

          conda install -c conda-forge pyvista


    .. grid-item-card:: Prefer pip?
       :columns: auto
       :class-title: pyvista-card-title

       PyVista can be installed via pip from `PyPI
       <https://pypi.org/project/pyvista>`__.

       .. code-block:: bash

          pip install pyvista


.. grid::

   .. grid-item-card:: In-depth instructions?
      :link: install
      :link-type: ref
      :class-title: pyvista-card-title

      Installing a specific version? Installing from source? Check the
      :ref:`install` page.


First Steps
===========
We've provided a variety of resources for you to get used to PyVista's API
through a range of examples and tutorials.


.. grid::

   .. grid-item-card:: Tutorial
      :link: https://tutorial.pyvista.org/tutorial.html
      :class-title: pyvista-card-title

      Probably the best way for you to get used to PyVista is to visit our
      dedicated `tutorial <https://tutorial.pyvista.org/tutorial.html>`_.

..
   This code is used in the plot in the card.

.. pyvista-plot::
   :include-source: False
   :context:

   >>> bunny_cpos = [
   ...     (0.14826, 0.275729, 0.4215911),
   ...     (-0.01684, 0.110154, -0.0015369),
   ...     (-0.15446, 0.939031, -0.3071841),
   ... ]


.. grid:: 2

   .. grid-item-card:: Why PyVista?
      :link: why_pyvista
      :link-type: ref
      :class-title: pyvista-card-title

      Learn more about why we created PyVista as an interface to the
      Visualization Toolkit (VTK).

      .. code-block:: python

          import pyvista as pv

          mesh = pv.read('bunny.stl')
          mesh.plot()

      .. pyvista-plot::
         :include-source: False
         :context:

         from pyvista import examples
         mesh = examples.download_bunny()
         mesh.plot(cpos=bunny_cpos)


   .. grid-item-card:: Authors & Citation
      :link: authors
      :link-type: ref
      :class-title: pyvista-card-title

      Using PyVista in your research? Please consider citing or acknowledging
      us.  We have a `JOSS Publication`_!

      .. image:: ../images/user-generated/joss.png
         :target: https://joss.theoj.org/papers/10.21105/joss.01450

.. grid::

   .. grid-item-card:: See PyVista in External Efforts
      :link: external_examples
      :link-type: ref
      :class-title: pyvista-card-title

      Take a look at third party projects using PyVista.


Support
=======

For general questions about the project, its applications, or about software
usage, please create a discussion in `pyvista/discussions`_
where the community can collectively address your questions. You are also
welcome to join us on Slack_.

.. _pyvista/discussions: https://github.com/pyvista/pyvista/discussions
.. _Slack: https://communityinviter.com/apps/pyvista/pyvista
.. _info@pyvista.org: mailto:info@pyvista.org


Citing PyVista
==============

There is a `paper about PyVista <https://doi.org/10.21105/joss.01450>`_.

If you are using PyVista in your scientific research, please help our scientific
visibility by citing our work. Head over to :ref:`citation` to learn more
about citing PyVista.

.. _JOSS Publication: https://joss.theoj.org/papers/10.21105/joss.01450
