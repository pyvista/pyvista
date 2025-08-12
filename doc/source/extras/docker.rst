!

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
Clone PyVista and build it to create your own customized docker image.

.. code-block:: bash

  git clone https://github.com/pyvista/pyvista
  cd pyvista
  pip install build
  python -m build --sdist
  IMAGE=my-pyvista-jupyterlab
  docker build -t $IMAGE -f docker/jupyter.Dockerfile .

If you wish to have off-screen GPU support when rending on jupyterlab,
see the notes about building with EGL at :ref:`building_vtk`,
or use the custom, pre-built wheels at
`Release 0.27.0 <https://github.com/pyvista/pyvista/releases/tag/0.27.0>`_.
Install that customized vtk wheel onto your docker image by modifying
the docker image at ``pyvista/docker/jupyter.Dockerfile`` with:

.. code-block:: docker

  COPY vtk-9.0.20201105-cp38-cp38-linux_x86_64.whl /tmp/
  RUN pip install /tmp/vtk-9.0.20201105-cp38-cp38-linux_x86_64.whl

Additionally, you must install GPU drivers on the docker image of the
same version running on the host machine. For example, if you are
running on Azure Kubernetes Service and the GPU nodes on the
kubernetes cluster are running ``450.51.06``, you must install the same
version on your image. Since you will be using the underlying kernel
module, there's no reason to build it on the container (and trying
will only result in an error).

.. code-block:: docker

  COPY NVIDIA-Linux-x86_64-450.51.06.run nvidia_drivers.run
  RUN sudo apt-get install kmod libglvnd-dev pkg-config -yq
  RUN ./NVIDIA-Linux-x86_64-450.51.06.run -s --no-kernel-module

To verify that you're rendering on a GPU, first check the output of
``nvidia-smi``. You should get something like:

.. code-block:: bash

  $ nvidia-smi
  Sun Nov  8 05:48:46 2020
  +-----------------------------------------------------------------------------+
  | NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
  |-------------------------------+----------------------+----------------------+
  | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
  |                               |                      |               MIG M. |
  |===============================+======================+======================|
  |   0  Tesla K80           Off  | 00000001:00:00.0 Off |                    0 |
  | N/A   34C    P8    32W / 149W |   1297MiB / 11441MiB |      0%      Default |
  |                               |                      |                  N/A |
  +-------------------------------+----------------------+----------------------+

Note the driver version (which is actually the kernel driver version),
and verify it matches the version you installed on your docker image.

Finally, check that your render window is using NVIDIA by running
``ReportCapabilities``:

.. code-block:: python

  >>> import pyvista
  >>> pl = pyvista.Plotter()
  >>> print(pl.render_window.ReportCapabilities())

  OpenGL vendor string:  NVIDIA Corporation
  OpenGL renderer string:  Tesla K80/PCIe/SSE2
  OpenGL version string:  4.6.0 NVIDIA 450.51.06
  OpenGL extensions:
    GL_AMD_multi_draw_indirect
    GL_AMD_seamless_cubemap_per_texture
    GL_ARB_arrays_of_arrays
    GL_ARB_base_instance
    GL_ARB_bindless_texture

If you get ``display id not set``, then your environment is likely not
set up correctly.
