.. _install_ref:

Installation
============

PyPI
~~~~

.. image:: https://img.shields.io/pypi/v/vtki.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/vtki/

Installing ``vtki`` itself is quite straightforward as it can be installed
from `PyPi <http://pypi.python.org/pypi/vtki>`_ using ``pip``::

    pip install vtki

``vtki`` requires ``numpy``, ``imageio``, and ``VTK`` version 7.0 or newer
which should be installed by pip automatically.


Anaconda
~~~~~~~~

.. image:: https://img.shields.io/conda/vn/conda-forge/vtki.svg
   :target: https://anaconda.org/conda-forge/vtki

To install this package with conda run::

    conda install -c conda-forge vtki


Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

The following are a list of optional dependencies and their purpose:

+-----------------------------------+-----------------------------------------+
| Package                           | Purpose                                 |
+===================================+=========================================+
| ``matplotlib``                    | Using Colormaps                         |
+-----------------------------------+-----------------------------------------+
| ``PyQt5==5.11.3``                 | Background plotting                     |
+-----------------------------------+-----------------------------------------+
| ``ipywidgets``                    | IPython interactive tools               |
+-----------------------------------+-----------------------------------------+
| ``panel``                         | Interactive notebook rendering          |
+-----------------------------------+-----------------------------------------+
| ``sphinx_gallery``                | Capturing ``vtki`` output for docs      |
+-----------------------------------+-----------------------------------------+


Source / Developers
~~~~~~~~~~~~~~~~~~~

Alternatively, you can install the latest version from GitHub by visiting
`vtki <https://github.com/vtkiorg/vtki>`_, downloading the source
(or cloning), and running::

    git clone https://github.com/vtkiorg/vtki.git
    cd vtki
    pip install -e .


Test Installation
~~~~~~~~~~~~~~~~~

You can test your installation by running an example:

.. testcode:: python

    from vtki import examples
    examples.plot_wave()

See other examples:

.. code:: python

    from vtki import examples

    # list all examples
    print(dir(examples))


.. warning:: Developers, please see :ref:`testing_ref` for details on development testing


Running on CI Services
~~~~~~~~~~~~~~~~~~~~~~

This section is for advanced users that would like to install and use ``vtki``
with headless displays and Docker containers. The steps here will work for
using ``vtki`` on Linux based continuous integration services like Travis CI
and on notebook hosting services like MyBinder_.

Please see `this project`_ for a convenient Cookiecutter_ to get started using
``vtki`` on the notebook hosting service MyBinder_.

.. _this project: https://github.com/vtkiorg/cookiecutter-vtki-binder
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _MyBinder: https://mybinder.org

To get started, the Docker container will need to have ``libgl1-mesa-dev`` and
``xvfb`` installed through and ``apt-get``. In a Travis scripts, simply include
the following in your ``.travis.yml`` file:

.. code-block:: yaml

    addons:
      apt:
        packages:
          - xvfb

and for MyBinder, include the following in a file called ``apt.txt``::

    libgl1-mesa-dev
    xvfb

Then, you need to configure the headless display, on Travis, add this to the
``.travis.yml`` file:

.. code-block:: yaml

    before_script: # configure a headless display to test plot generation
      - export DISPLAY=:99.0
      - export VTKI_OFF_SCREEN=True
      - which Xvfb
      - Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
      - sleep 3 # give xvfb some time to start


Likewise for MyBinder, create a file called ``start`` and include the following
set up script that will run everytime your Docker container is launched:

.. code-block:: bash

    #!/bin/bash
    export DISPLAY=:99.0
    export VTKI_OFF_SCREEN=True
    which Xvfb
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    sleep 3
    exec "$@"


And that's it! Include ``vtki`` in your Python requirements and get to
visualizing your data! If you need more help than this on setting up ``vtki``
for CI-like services, hop on Slack and chat with the developers or take a look
at `this repository`_ that is currently using ``vtki`` on MyBinder.

.. _this repository: https://github.com/OpenGeoVis/PVGeo-Examples

.. warning:: Offscreen rendering is required for headless displays

    Note that ``vtki`` will have to be used in offscreen mode. This can be forced on import with:

    .. code-block:: python

        import vtki
        vtki.OFF_SCREEN = True
