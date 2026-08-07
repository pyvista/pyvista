PyVista within a Docker Container
=================================
You can use ``pyvista`` from within a docker container with
jupyterlab. To launch a local docker container, install ``docker``, then
pull and run the image with:

.. code-block:: bash

  docker run -p 8888:8888 ghcr.io/pyvista/pyvista:latest

Finally, open the link that shows up from the terminal output and
start playing around with pyvista in jupyterlab. For example:

.. code-block:: bash

    To access the notebook, open this file in a browser:
        file:///home/jovyan/.local/share/jupyter/runtime/nbserver-6-open.html
    Or copy and paste one of these URLs:
        http://861c873f6352:8888/?token=b3ac1f6397188944fb21e1f58b673b5b4e6f1ede1a84787b
     or http://127.0.0.1:8888/?token=b3ac1f6397188944fb21e1f58b673b5b4e6f1ede1a84787b


.. note::

    You can see the latest tags of `our Docker containers here <https://github.com/pyvista/pyvista/pkgs/container/pyvista>`_. ``ghcr.io/pyvista/pyvista:latest`` has
    JupyterLab set up while ``ghcr.io/pyvista/pyvista:latest-slim`` is a
    lightweight Python environment without Jupyter


.. note::

    You may need to log into the GitHub container registry by following the directions at
    `Working with the Docker registry <https://docs.github.com/en/enterprise-server@3.0/packages/working-with-a-github-packages-registry/working-with-the-docker-registry>`_)


Create your own Docker Container with PyVista
---------------------------------------------
Both the ``latest`` and ``latest-slim`` images are produced from a single
multi-stage Dockerfile at ``docker/Dockerfile``. The PyVista wheel is built
*inside* the Docker build, so no pre-build steps are required. Clone the
repository and run ``docker build`` from the project root:

.. code-block:: bash

  git clone https://github.com/pyvista/pyvista
  cd pyvista

  # JupyterLab image (equivalent to ghcr.io/pyvista/pyvista:latest)
  docker build -f docker/Dockerfile --target jupyter -t my-pyvista-jupyter .

  # Slim off-screen image (equivalent to ghcr.io/pyvista/pyvista:latest-slim)
  docker build -f docker/Dockerfile --target slim -t my-pyvista-slim .

The ``jupyter`` target installs the ``jupyter``, ``colormaps``, and ``io``
optional dependency groups directly from ``pyproject.toml``, so the package
set always matches the project's pins. There is no separate
``requirements.txt`` to keep in sync.

Override the Python version (must match a supported VTK wheel) via a build
argument:

.. code-block:: bash

  docker build --build-arg PY_VERSION=3.12 \
    -f docker/Dockerfile --target jupyter -t my-pyvista-jupyter .

GPU Rendering with the NVIDIA Container Runtime
-----------------------------------------------
Both published images ship with ``libegl1`` and set
``NVIDIA_VISIBLE_DEVICES=all`` plus
``NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute``. VTK 9.5+ picks
the ``vtkEGLRenderWindow`` automatically, so rendering uses the host's
NVIDIA driver when the container is started with the NVIDIA container
runtime:

.. code-block:: bash

  docker run --rm --gpus all -p 8888:8888 ghcr.io/pyvista/pyvista:latest

The only host-side requirement is the `NVIDIA Container Toolkit
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_.
No in-container driver install, no kernel modules, no version matching
against the host driver. On CPU-only hosts (no ``--gpus``) the same
image transparently falls back to Mesa's ``llvmpipe`` software renderer
via EGL.

To verify you are rendering on a GPU, inspect
``pv.Report()`` from inside the container:

.. code-block:: python

    import pyvista as pv

    print(pv.Report())

The ``GPU Vendor`` and ``GPU Renderer`` fields should report the NVIDIA
driver and the GPU model. ``Render Window`` should read
``vtkEGLRenderWindow``.
