.. _install:

Installation
============

PyVista is supported on Python versions 3.9+.

For the best experience, please considering using Anaconda as a virtual
environment and package manager for Python and following the instructions to
install PyVista with Anaconda.


Dependencies
~~~~~~~~~~~~

PyVista is built on top of the Visualization Toolkit (VTK) and NumPy - as such,
the following projects are required dependencies of PyVista:

* `vtk <https://pypi.org/project/vtk/>`_ - PyVista directly inherits types from the VTK library.
* `NumPy <https://pypi.org/project/numpy/>`_ - NumPy arrays provide a core foundation for PyVista's data array access.
* `matplotlib <https://pypi.org/project/matplotlib/>`_ - Used for colormaps and 2D plotting with :class:`pyvista.ChartMPL`.
* `pillow <https://pypi.org/project/Pillow/>`_ - PIL fork used for saving screenshots.
* `imageio <https://pypi.org/project/imageio/>`_ - This library is used for reading images and writing animations.
* `pooch <https://pypi.org/project/pooch/>`_ - Data management for our example datasets so users can download tutorials on the fly.
* `scooby <https://github.com/banesullivan/scooby>`_ - Reporting and debugging tools.


Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~
PyVista includes several optional dependencies for visualization and reading a variety of additional file formats, including:

* `cmocean <https://pypi.org/project/cmocean/>`_ - Colormaps for Oceanography.
* `colorcet <https://colorcet.holoviz.org/>`_ - Perceptually accurate 256-color colormaps for use with Python.
* `trame <https://github.com/Kitware/trame>`_ - Used for client and server-side rendering in Jupyter.
* `meshio <https://pypi.org/project/meshio/>`_ - Input/Output for many mesh formats.


PyPI
~~~~

.. image:: https://img.shields.io/pypi/v/pyvista.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/pyvista/

PyVista can be installed from `PyPI <https://pypi.org/project/pyvista/>`_
using ``pip``::

    pip install pyvista

To install all the additional packages that extend PyVista, install using
``pip`` with::

    pip install pyvista[all]

.. asciinema:: 507562

Anaconda
~~~~~~~~

.. image:: https://img.shields.io/conda/vn/conda-forge/pyvista.svg?logo=conda-forge&logoColor=white
   :target: https://anaconda.org/conda-forge/pyvista

To install this package with ``conda`` run::

    conda install -c conda-forge pyvista

.. asciinema:: 507565


Pixi
~~~~

`Pixi <https://pixi.sh/>`_ is a modern package management tool that provides fast, reliable, and reproducible software environments. It can be used as an alternative to conda/mamba for managing PyVista installations.

To install PyVista using pixi, first ensure you have pixi installed (see `pixi installation instructions <https://pixi.sh/latest/#installation>`_), then run::

    pixi add pyvista

This will add PyVista to your ``pixi.toml`` file and install it in your project environment. To use PyVista in your pixi environment::

    pixi shell
    python -c "import pyvista; print(pyvista.__version__)"

For global installation, you can use::

    pixi global install pyvista

Pixi automatically handles all dependencies and ensures compatibility across different platforms.


Installing the Current Development Branch from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There may be features or bug-fixes that have been implemented in PyVista but
have not made their way into a release. To install ``pyvista`` from the latest
up-to-date development branch from github, use one of the following

.. code-block:: bash

   pip install -U git+https://github.com/pyvista/pyvista.git@main

Alternatively, you can clone the repository with git and install it with pip.

.. code-block:: bash

   git clone https://github.com/pyvista/pyvista.git
   cd pyvista
   pip install -e .

Note the development flag ``-e``. This allows you to change pyvista
in-place without having to reinstall it for each change.


Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

The following are a list of optional dependencies and their purpose:

+-----------------------------------+-----------------------------------------+
| Package                           | Purpose                                 |
+===================================+=========================================+
| ``trame``                         | Interactive notebook rendering          |
+-----------------------------------+-----------------------------------------+
| ``sphinx_gallery``                | Capturing PyVista output for docs       |
+-----------------------------------+-----------------------------------------+
| ``colorcet``                      | Perceptually uniform colormaps          |
+-----------------------------------+-----------------------------------------+
| ``cmocean``                       | Oceanographic colormaps                 |
+-----------------------------------+-----------------------------------------+
| ``imageio-ffmpeg``                | Saving movie files                      |
+-----------------------------------+-----------------------------------------+
| ``tqdm``                          | Status bars for monitoring filters      |
+-----------------------------------+-----------------------------------------+
| ``trimesh``                       |                                         |
| ``rtree``                         | Vectorised ray tracing                  |
| ``pyembree``                      |                                         |
+-----------------------------------+-----------------------------------------+


Source / Developers
~~~~~~~~~~~~~~~~~~~

Alternatively, you can install the latest version from GitHub by visiting
`PyVista <https://github.com/pyvista/pyvista>`_, and downloading the source
(cloning) by running::

    git clone https://github.com/pyvista/pyvista.git
    cd pyvista
    python -m pip install -e .


The latest documentation for the ``main`` branch of PyVista can be found at
`dev.pyvista.org <https://dev.pyvista.org>`_.


Test Installation
~~~~~~~~~~~~~~~~~

You can test your installation by running an example:

.. code-block:: python

    >>> from pyvista import demos
    >>> demos.plot_wave()

See other examples and demos:

.. code-block:: python

    >>> from pyvista import examples
    >>> from pyvista import demos

    List all available examples.

    >>> print(dir(examples))

    List all available demos.


.. note::

    A more comprehensive testing suite is available after cloning the source
    repository. For details on how to clone and test the PyVista source, please
    see our `Contributing Guide`_ and specifically, the `Testing`_ section.

.. _Contributing Guide: https://github.com/pyvista/pyvista/blob/main/CONTRIBUTING.rst
.. _Testing: https://github.com/pyvista/pyvista/blob/main/CONTRIBUTING.rst#user-content-testing


Running on CI Services
~~~~~~~~~~~~~~~~~~~~~~
Please head over to `pyvista/setup-headless-display-action`_ for details on
setting up a headless display in GitHub Actions Workflows using PyVista.

In the past, we have also supported `pyvista/gl-ci-helpers`_ for setting up CI
services like Travis and Azure Pipelines to run PyVista.

.. _pyvista/setup-headless-display-action: https://github.com/pyvista/setup-headless-display-action
.. _pyvista/gl-ci-helpers: https://github.com/pyvista/gl-ci-helpers


Running on MyBinder
~~~~~~~~~~~~~~~~~~~
This section is for advanced users that would like to install and use PyVista
with headless displays on notebook hosting services like MyBinder_.

Please see `this project`_ for a convenient Cookiecutter_ to get started using
PyVista on the notebook hosting service MyBinder_.

.. _this project: https://github.com/pyvista/cookiecutter-pyvista-binder
.. _Cookiecutter: https://github.com/cookiecutter/cookiecutter
.. _MyBinder: https://mybinder.org

To get started, the Docker container will need to have ``libgl1-mesa-dev`` and
``xvfb`` installed through ``apt-get``. For MyBinder, include the following in
a file called ``apt.txt``::

    libgl1-mesa-dev
    xvfb

Then, you need to configure the headless display, for MyBinder, create a file
called ``start`` and include the following set up script that will run every
time your Docker container is launched:

.. code-block:: bash

    #!/bin/bash
    set -x
    export DISPLAY=:99.0
    export PYVISTA_OFF_SCREEN=true
    which Xvfb
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    sleep 3
    set +x
    exec "$@"


All you have to do next is include PyVista in your Python requirements and you
can get to visualizing your data. If you need more help than this on setting up
PyVista for these types of services, hop on Discussions page and chat with the developers
or take a look at `this repository`_ that is currently using PyVista on
MyBinder.

.. _this repository: https://github.com/OpenGeoVis/PVGeo-Examples

Running on Remote Servers
~~~~~~~~~~~~~~~~~~~~~~~~~
Using PyVista on remote servers requires similar setup steps as in the above
Docker case. As an example, here are the complete steps to use PyVista on AWS
EC2 Ubuntu 18.04 LTS (``ami-0a313d6098716f372`` in ``us-east-1``).
Other servers would work similarly.

After logging into the remote server, install Miniconda and related packages:

.. code-block:: bash

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p miniconda
    echo '. $HOME/miniconda/etc/profile.d/conda.sh' >> ~/.bashrc && source ~/.bashrc
    conda create --name vtk_env python=3.9
    conda activate vtk_env
    conda install nodejs  # required when importing pyvista in Jupyter
    pip install jupyter pyvista trame

    # To avoid "ModuleNotFoundError: No module named 'vtkOpenGLKitPython' " when importing vtk
    # https://stackoverflow.com/q/32389599
    # https://askubuntu.com/q/629692
    sudo apt update && sudo apt install python-qt4 libgl1-mesa-glx

Then, configure the headless display:

.. code-block:: bash

    sudo apt-get install xvfb
    export DISPLAY=:99.0
    export PYVISTA_OFF_SCREEN=true
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    sleep 3

Reconnect to the server with port-forwarding, and start Jupyter:

.. code-block:: bash

    ssh -i "your-ssh-key" your-user-name@your-server-ip -L 8888:localhost:8888
    conda activate vtk_env
    jupyter lab --NotebookApp.token='' --no-browser --port=8888

Visit ``localhost:8888`` in the web browser.

Running on WSL
~~~~~~~~~~~~~~
Similar to the example of the remote server above, the windows subsystem for Linux does
not provide an x-server for visualization. Instead, the fastest way to get up and
running on WSL is through `JupyterLab <https://jupyter.org/>`_.

First, make sure you have installed the correct environment through Miniconda and
related packages:

.. code-block:: bash

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p miniconda
    echo '. $HOME/miniconda/etc/profile.d/conda.sh' >> ~/.bashrc && source ~/.bashrc
    conda create --name vtk_env python=3.9
    conda activate vtk_env
    conda install nodejs  # required when importing pyvista in Jupyter
    pip install jupyter pyvista[jupyter] trame

    # To avoid "ModuleNotFoundError: No module named 'vtkOpenGLKitPython' " when importing vtk
    # https://stackoverflow.com/q/32389599
    # https://askubuntu.com/q/629692
    sudo apt update && sudo apt install python-qt4 libgl1-mesa-glx

VTK Link to Jupyter
^^^^^^^^^^^^^^^^^^^
There are two ways to get vtk rendering 3D objects in JupyterLab. First you
can follow the example above for remote servers, skipping over the ``ssh``
instructions.

Configure the headless display:

.. code-block:: bash

    sudo apt-get install xvfb
    export DISPLAY=:99.0
    export PYVISTA_OFF_SCREEN=true
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    sleep 3

Start Jupyter:

.. code-block:: bash

    jupyter lab --NotebookApp.token='' --no-browser --port=8888

Visit ``localhost:8888`` in the web browser.

Finally add this example code and your interactive visualizations
should be displayed in JupyterLab.

.. code-block:: python

    import pyvista

    pl = pyvista.Plotter(shape=(1, 2))
    actor = pl.add_mesh(pyvista.Cube())
    pl.subplot(0, 1)
    actor = pl.add_mesh(pyvista.Sphere())
    pl.set_background('orange', all_renderers=False)
    pl.show()

Your visualizations should now be showing directly in the Jupyter frontend.

Running with Sphinx-Gallery
~~~~~~~~~~~~~~~~~~~~~~~~~~~
In your ``conf.py``, add the following:


.. code-block:: python

    import pyvista

    # necessary when building the sphinx gallery
    pyvista.BUILDING_GALLERY = True
    pyvista.OFF_SCREEN = True

    # Optional - set parameters like theme or window size
    pyvista.set_plot_theme('document')
    pyvista.global_theme.window_size = np.array([1024, 768]) * 2

    extensions = [
        ...,
        "sphinx_gallery.gen_gallery",
    ]

    # Add the PyVista image scraper to SG
    sphinx_gallery_conf = {
        ...: ...,
        "image_scrapers": ('pyvista', ...),
        ...: ...,
    }

We also have a Sphinx-Gallery scraper for embedding dynamic 3D scenes
instead of static screenshots. This scraper can be enabled by passing
an instance directly to the ``image_scrapers`` parameter instead of
the string ``'pyvista'`` above and by registering the
``pyvista.ext.viewer_directive`` extension

.. code-block:: python

    import pyvista
    from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper

    # necessary when building the sphinx gallery
    pyvista.BUILDING_GALLERY = True
    pyvista.OFF_SCREEN = True

    # Optional - set parameters like theme or window size
    pyvista.set_plot_theme('document')
    pyvista.global_theme.window_size = np.array([1024, 768]) * 2

    extensions = [
        ...,
        "sphinx_gallery.gen_gallery",
        "pyvista.ext.viewer_directive",
    ]

    # Add the PyVista image scraper to SG
    sphinx_gallery_conf = {
        ...: ...,
        "image_scrapers": (DynamicScraper(), ...),
        ...: ...,
    }
