.. _install_ref:

Installation
============
Installing vtkInterface itself is quite straightforward as it can be installed using ``pip``.  ``VTK`` itself can also be installed using pip or from a a distribution such as `Anaconda <https://www.continuum.io/downloads>`_. The installation directions are different depending on your OS; see the directions below.

``vtkInterface`` requires ``numpy``, ``imageio``, and ``VTK`` version 7.0 or newer.


Windows Installation
--------------------

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
Regardless of your OS, you can test your installation by running an example:

.. code:: python

    from vtkInterface import examples
    examples.ShowWave()

See other examples:

.. code:: python

    from vtkInterface import examples

    # list all examples
    print(dir(examples))
