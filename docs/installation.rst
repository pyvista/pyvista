.. _install_ref:

Installation
============
Installing vtkInterface itself is quite straightforward as it can be installed using ``pip``.

``vtkInterface`` requires ``numpy``, ``imageio``, and ``VTK`` version 7.0 or newer.


Install VTK
~~~~~~~~~~~
VTK can be installed using pip for most versions of Python::

  $ pip install vtk

If this command fails, install VTK by installing from a distribution like `Anaconda <https://www.continuum.io/downloads>`_ and then installing VTK for Python 2.7, 3.4, 3.5, and 3.6 by running the following::

    conda install -c conda-forge vtk


Install vtkInterface
~~~~~~~~~~~~~~~~~~~~
Install vtkInterface from `PyPi <http://pypi.python.org/pypi/vtkInterface>`_ by running::

    pip install vtkInterface

Alternatively, you can install the latest version from GitHub by visiting `vtkInterface <https://github.com/akaszynski/vtkInterface>`_, downloading the source, and running::

    cd C:\Where\You\Downloaded\vtkInterface
    pip install .


Test Installation
-----------------
You can test your installation by running an example:

.. code:: python

    from vtkInterface import examples
    examples.ShowWave()

See other examples:

.. code:: python

    from vtkInterface import examples

    # list all examples
    print(dir(examples))
