.. _testing_ref:

Test Framework
==============

After making changes, please test changes locally before creating a pull
request. The following tests will be executed after any commit or pull request,
so we ask that you perform the following sequence locally to track down any new
issues from your changes.
To run our comprehensive suite of unit tests, install all the dependencies
listed in ``requirements.txt``, ``requirements_docs.txt``, and the following
list:


.. code:: bash

    pip install -r requirements.txt
    pip install -r requirements_docs.txt
    pip install pytest-cov
    pip install codecov
    pip install PyQt5==5.11.3
    pip install pytest-qt
    pip install nbval
    pip install ipywidgets


Then, if you have everything installed, you can run the various test suites:


Run the primary test suite and generate coverage report:

.. code:: bash

    pytest -v --cov vtki


Run all code examples in the docstrings:

.. code:: bash

    pytest -v --doctest-modules vtki


Now make sure notebooks are running

.. code:: bash

    pytest -v --nbval-lax --current-env --disable-warnings notebooks/*.ipynb;


And finally, test the documentation examples:

.. code:: bash

    cd ./docs/
    make doctest
    make html

The finished documentation can be found in the `docs/_build/html` directory.
