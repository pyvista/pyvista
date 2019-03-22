.. _install_ref:

Installation
============

Installing ``vtki`` itself is quite straightforward as it can be installed
from `PyPi <http://pypi.python.org/pypi/vtki>`_ using ``pip``::

    pip install vtki

``vtki`` requires ``numpy``, ``imageio``, and ``VTK`` version 7.0 or newer
which should be installed by pip automatically.


Install from Anaconda
~~~~~~~~~~~~~~~~~~~~~

To install this package with conda run::

    conda install -c conda-forge vtki


Install from Source
~~~~~~~~~~~~~~~~~~~

Alternatively, you can install the latest version from GitHub by visiting
`vtki <https://github.com/vtkiorg/vtki>`_, downloading the source
(or cloning), and running::

    git clone https://github.com/vtkiorg/vtki.git
    cd vtki
    pip install .


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
