.. _testing_ref:

Test Framework
==============

After making changes, please test changes locally before creating a pull
request. The following tests will be executed after any commit or pull request,
so we ask that you perform the following sequence locally to track down any new
issues from your changes.
To run our comprehensive suite of unit tests, install all the dependencies
listed in ``requirements.txt``, ``requirements_docs.txt``:


.. code:: bash

    pip install -r requirements.txt
    pip install -r requirements_docs.txt


Then, if you have everything installed, you can run the various test suites:


Run the primary test suite and generate coverage report:

.. code:: bash

    python -m pytest -v --cov pyvista


Run all code examples in the docstrings:

.. code:: bash

    python -m pytest -v --doctest-modules pyvista


Now make sure notebooks are running

.. code:: bash

    python -m pytest -v --nbval-lax --current-env --disable-warnings notebooks/*.ipynb
    python -m pytest -v --nbval-lax --current-env --disable-warnings tests/*.ipynb

And finally, test the documentation examples:

.. code:: bash

    cd ./docs/
    make doctest
    make html

The finished documentation can be found in the `docs/_build/html` directory.
