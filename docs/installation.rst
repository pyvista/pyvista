.. _install_ref:

Installation
============
Installing vtkInterface itself is quite straightforward as it can be installed using ``pip``.  ``VTK`` itself can also be installed using pip or from a a distribution such as `Anaconda <https://www.continuum.io/downloads>`_. The installation directions are different depending on your OS; see the directions below.

``vtkInterface`` requires ``numpy`` and ``VTK``.  Optional modules include ``moviepy`` and ``imageio`` for saving movies or moving gifs.


Windows Installation
--------------------

Install VTK
~~~~~~~~~~~
VTK can be installed using pip for Python 3.6::

  $ pip install vtk


Install VTK by installing from a distribution like `Anaconda <https://www.continuum.io/downloads>`_ and then installing VTK for Python 3.6 by running the following from a command prompt::

    conda install -c clinicalgraphics vtk=7.1.0

If you're using Python 3.5::

    conda install -c menpo vtk=7.0.0
    
Or, if you're using Python 2.7, install using::

    conda install -c anaconda vtk=6.3.0

You can also install VTK from the source by following these `Directions <http://www.vtk.org/Wiki/VTK/Building/Windows>`_.  This is quite difficult.


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
If using Python 3.4 or greater, VTK can be installed from pip with::

    $ pip install vtk --user

Please note that as of the time of this writing, python will not be able to find the dynamic libraries of the vtk install.  This can be fixed by appending the LD_LIBRARY_PATH::

    $ touch pythonvtk.conf
    $ echo '/home/user/.local/lib/python3.5/site-packages/vtk' >> pythonvtk.conf
    $ sudo mv pythonvtk.conf /etc/ld.so.conf.d/pythonvtk.conf

This path will vary depending on the user name and if the vtk package has been installed using the ``user`` flag or if it has been installed as root.

Install vtkInterface
~~~~~~~~~~~~~~~~~~~~
Install vtkInterface from `PyPi <http://pypi.python.org/pypi/vtkInterface>`_ by running::

    $ pip install vtkInterface --user

You can also install the latest source from 
`GitHub <https://github.com/akaszynski/vtkInterface>`_ with::

    $ git clone https://github.com/akaszynski/vtkInterface
    $ cd vtkInterface
    $ pip install . --user


Test Installation
-----------------
Regardless of your OS, you can test your installation by running an example from tests:

.. code:: python

    from vtkInterface import tests
    tests.ShowWave()

You can also run examples from:

.. code:: python

    from vtkInterface import examples
    print(dir(examples))
