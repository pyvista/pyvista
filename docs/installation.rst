.. _install_ref:

Installation
============
Installing vtki itself is quite straightforward as it can be installed using ``pip``.

``vtki`` requires ``numpy``, ``imageio``, and ``VTK`` version 7.0 or newer.


Install VTK
~~~~~~~~~~~
VTK can be installed using pip for most versions of Python::

  $ pip install vtk

If this command fails, install VTK by installing from a distribution like `Anaconda <https://www.continuum.io/downloads>`_ and then installing VTK for Python 2.7, 3.4, 3.5, and 3.6 by running the following::

    conda install -c conda-forge vtk


Install vtki
~~~~~~~~~~~~
Install vtki from `PyPi <http://pypi.python.org/pypi/vtki>`_ by running::

    pip install vtki

Alternatively, you can install the latest version from GitHub by visiting `vtki <https://github.com/akaszynski/vtki>`_, downloading the source, and running::

    cd C:\Where\You\Downloaded\vtki
    pip install .


Test Installation
-----------------
You can test your installation by running an example:

.. code:: python

    from vtki import examples
    examples.plot_wave()

See other examples:

.. code:: python

    from vtki import examples

    # list all examples
    print(dir(examples))
