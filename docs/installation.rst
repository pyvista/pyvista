.. _install_ref:

Installation
============

Installing vtkInterface itself is quite straightforward, but you will need VTK.  
VTK can be easily compiled on a UNIX or Linux machine, but with a Windows 
environment it's easier to use a distribution such as 
`Anaconda <https://www.continuum.io/downloads>`_. The installation directions are different depending on your OS.  See below.

vtkInterface requires numpy and VTK.  Optional modules include moviepy and imageio for saving movies or moving gifs.


Windows Installation
--------------------

Install VTK
~~~~~~~~~~~
Install VTK by installing from a distribution like `Anaconda <https://www.continuum.io/downloads>`_ and then installing VTK for Python 3.6 by running the following from a command prompt::

    conda install -c clinicalgraphics vtk=7.1.0

If you're using Python 3.5::

    conda install -c menpo vtk=7.0.0
    
Or, if you're using Python 2.7, install using::

    conda install -c anaconda vtk=6.3.0

You can also install VTK from the source by following these `Directions <http://www.vtk.org/Wiki/VTK/Building/Windows>`_.  This is generally quite difficult in Windows.


Install vtkInterface
~~~~~~~~~~~~~~~~~~~~
Install vtkInterface from `PyPi <http://pypi.python.org/pypi/vtkInterface>`_ by running::

    pip install vtkInterface

Alternatively, you can install the latest version from GitHub by visiting `vtkInterface <https://github.com/akaszynski/vtkInterface>`_, downloading the source, and running::

    cd C:\Where\You\Downloaded\vtkInterface
    pip install .
    

Linux Installation
------------------

Install VTK
~~~~~~~~~~~
If you have Ubuntu 14.04 or newer, you can also install VTK using the apt
package manager::

    sudo apt-get install python-vtk

This will be an earlier version of VTK and while it’s the easiest approach, it’s probably not the best approach given how easy it is to build VTK.

Building VTK from the source under Linux is generally straightforward.  See `Building VTK for Linux <http://www.vtk.org/Wiki/VTK/Building/Linux>`_ and make sure to enable building with Python.  See `Python Environment Setup <http://www.vtk.org/Wiki/VTK/Tutorials/PythonEnvironmentSetup>`_ if you have problems loading VTK from Python.


Install vtkInterface
~~~~~~~~~~~~~~~~~~~~
Install vtkInterface from `PyPi <http://pypi.python.org/pypi/vtkInterface>`_ by running::

    pip install vtkInterface

You can also install the latest source from 
`GitHub <https://github.com/akaszynski/vtkInterface>`_ with::

    git clone https://github.com/akaszynski/vtkInterface
    cd vtkInterface
    pip install .

Test Installation
-----------------
Regardless of your OS, you can test your installation by running an example 
from tests:

.. code:: python

    from vtkInterface import tests
    tests.ShowWave()

You can also run examples from:

.. code:: python

    from vtkInterface import examples
    print dir(examples) # look for functions

See the examples for other examples.
