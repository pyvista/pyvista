Getting Started
***************

.. toctree::
   :hidden:

   why
   authors
   installation
   connections
   external_examples

..
   used in the plot in the card

.. pyvista-plot::
   :include-source: False
   :context:

   >>> bunny_cpos = [( 0.14826, 0.275729,  0.4215911),
   ...               (-0.01684, 0.110154, -0.0015369),
   ...               (-0.15446, 0.939031, -0.3071841)]


.. grid:: 2

   .. grid-item-card:: Why PyVista?
      :link: why_pyvista
      :link-type: ref

      Learn more about why we created PyVista as an interface to the
      Visualization Toolkit (VTK).

      .. code:: python

         import pyvista
         mesh = pyvista.read('bunny.stl')
         mesh.plot()

      .. pyvista-plot::
         :include-source: False
         :context:

         from pyvista import examples
         mesh = examples.download_bunny()
         mesh.plot(cpos=bunny_cpos, anti_aliasing='ssao')


   .. grid-item-card:: Authors & Citation
      :link: authors_ref
      :link-type: ref

      Using PyVista in your research? Please consider citing or acknowledging
      us.  We have a `JOSS Publication`_!

      .. image:: ../images/user-generated/joss.png
         :target: https://joss.theoj.org/papers/10.21105/joss.01450


.. grid::

   .. grid-item-card:: See PyVista in External Efforts
      :link: external_examples
      :link-type: ref

      Take a look at third party projects using PyVista.


Installation
============


.. grid:: 2

    .. grid-item-card:: Working with conda?

       PyVista is available on `conda-forge
       <https://anaconda.org/conda-forge/pyvista>`_

       .. code-block:: bash

          conda install -c conda-forge pyvista


    .. grid-item-card:: Prefer pip?
       :columns: auto

       PyVista can be installed via pip from `PyPI
       <https://pypi.org/project/pyvista>`__.

       .. code-block:: bash

          pip install pyvista


.. grid::

   .. grid-item-card:: In-depth instructions?
      :link: install_ref
      :link-type: ref

      Installing a specific version? Installing from source? Check the
      :ref:`install_ref` page.


Support
=======

For general questions about the project, its applications, or about software
usage, please create a discussion in `pyvista/discussions`_
where the community can collectively address your questions. You are also
welcome to join us on Slack_ or send one of the developers an email.
The project support team can be reached at `info@pyvista.org`_.

.. _pyvista/discussions: https://github.com/pyvista/pyvista/discussions
.. _Slack: http://slack.pyvista.org
.. _info@pyvista.org: mailto:info@pyvista.org


Citing PyVista
==============

There is a `paper about PyVista <https://doi.org/10.21105/joss.01450>`_!

If you are using PyVista in your scientific research, please help our scientific
visibility by citing our work! Head over to :ref:`citation_ref` to learn more
about citing PyVista.

.. _JOSS Publication: https://joss.theoj.org/papers/10.21105/joss.01450
