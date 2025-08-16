Contributing
============

.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg
   :target: CODE_OF_CONDUCT.md

.. |codetriage| image:: https://www.codetriage.com/pyvista/pyvista/badges/users.svg
   :target: https://www.codetriage.com/pyvista/pyvista
   :alt: Code Triage

|Contributor Covenant|
|codetriage|

We absolutely welcome contributions and we hope that this guide will
facilitate an understanding of the PyVista code repository. It is
important to note that the PyVista software package is maintained on a
volunteer basis and thus we need to foster a community that can support
user questions and develop new features to make this software a useful
tool for all users.

This page is dedicated to outline where you should start with your
question, concern, feature request, or desire to contribute.

Being Respectful
----------------

Please demonstrate empathy and kindness toward other people, other software,
and the communities who have worked diligently to build (un)related tools.

Please do not talk down in Pull Requests, Issues, or otherwise in a way that
portrays other people or their works in a negative light.

Cloning the Source Repository
-----------------------------

You can clone the source repository from
`<https://github.com/pyvista/pyvista>`_ and install the latest version by
running:

.. code-block:: bash

   git clone https://github.com/pyvista/pyvista.git
   cd pyvista
   python -m pip install -e .

.. note::

   Use ``python -m pip install -e . --group dev`` to also install all of the
   packages required for development.

Quick Start Development with Codespaces
---------------------------------------

.. |Open in GitHub Codespaces| image:: https://github.com/codespaces/badge.svg
   :target: https://codespaces.new/pyvista/pyvista
   :alt: Open in GitHub Codespaces

|Open in GitHub Codespaces|

A dev container is provided to quickly get started. The default container
comes with the repository code checked out on a branch of your choice
and all pyvista dependencies including test dependencies pre-installed.
In addition, it uses the
`desktop-lite feature <https://github.com/devcontainers/features/tree/main/src/desktop-lite>`_
to provide live interaction windows.  Follow directions
`Connecting to the desktop <https://github.com/devcontainers/features/tree/main/src/desktop-lite#connecting-to-the-desktop>`_
to use the live interaction.

Alternatively, an offscreen version using OSMesa libraries and ``vtk-osmesa`` is available.

Questions
---------

For general questions about the project, its applications, or about
software usage, please create a discussion in the
`Discussions <https://github.com/pyvista/pyvista/discussions>`_
repository where the community can collectively address your questions.

You are also welcome to join us on `Slack <https://communityinviter.com/apps/pyvista/pyvista>`_,
but Slack should be reserved for ad hoc conversations and community engagement
rather than technical discussions.

For critical, high-level project support and engagement, please email
info@pyvista.org - but please do not use this email for technical support.

For all technical conversations, you are welcome to create an issue on the
`Discussions page <https://github.com/pyvista/pyvista/discussions>`_
which we will address promptly. Through posting on the Discussions page,
your question can be addressed by community members with the needed
expertise and the information gained will remain available for other
users to find.

Reporting Bugs
--------------

If you stumble across any bugs, crashes, or concerning quirks while
using code distributed here, please report it on the `issues
page <https://github.com/pyvista/pyvista/issues>`_ with an appropriate
label so we can promptly address it. When reporting an issue, please be
overly descriptive so that we may reproduce it. Whenever possible,
please provide tracebacks, screenshots, and sample files to help us
address the issue.

Feature Requests
----------------

We encourage users to submit ideas for improvements to PyVista code
base. Please create an issue on the `issues
page <https://github.com/pyvista/pyvista/issues>`_ with a *Feature
Request* label to suggest an improvement. Please use a descriptive title
and provide ample background information to help the community implement
that functionality. For example, if you would like a reader for a
specific file format, please provide a link to documentation of that
file format and possibly provide some sample files with screenshots to
work with. We will use the issue thread as a place to discuss and
provide feedback.

Contributing New Code
---------------------

If you have an idea for how to improve PyVista, please first create an
issue as a feature request which we can use as a discussion thread to
work through how to implement the contribution.

Once you are ready to start coding and develop for PyVista, please see
the `Development Practices <#development-practices>`_ section for more
details.

Licensing
---------

All contributed code will be licensed under The MIT License found in the
repository. If you did not write the code yourself, it is your
responsibility to ensure that the existing license is compatible and
included in the contributed files or you can obtain permission from the
original author to relicense the code.

--------------

Development Practices
---------------------

This section provides a guide to how we conduct development in the
PyVista repository. Please follow the practices outlined here when
contributing directly to this repository.

Guidelines
~~~~~~~~~~

Through direct access to the Visualization Toolkit (VTK) via direct
array access and intuitive Python properties, we hope to make the entire
VTK library easily accessible to researchers of all disciplines. To
further PyVista towards being a valuable Python interface to VTK, we
need your help to make it even better.

If you want to add one or two interesting analysis algorithms as
filters, implement a new plotting routine, or just fix 1-2 typos - your
efforts are welcome.

There are three general coding paradigms that we believe in:

#. **Make it intuitive**. PyVista’s goal is to create an intuitive and
   easy to use interface back to the VTK library. Any new features
   should have intuitive naming conventions and explicit keyword
   arguments for users to make the bulk of the library accessible to
   novice users.

#. **Document everything**. At the least, include a docstring for any
   method or class added. Do not describe what you are doing but why you
   are doing it and provide a simple example for the new features.

#. **Keep it tested**. We aim for a high test coverage. See testing for
   more details.

There are two important copyright guidelines:

#. Please do not include any data sets for which a license is not
   available or commercial use is prohibited. Those can undermine the
   license of the whole projects.

#. Do not use code snippets for which a license is not available
   (for example from Stack Overflow) or commercial use is prohibited. Those can
   undermine the license of the whole projects.

Please also take a look at our `Code of
Conduct <https://github.com/pyvista/pyvista/blob/main/CODE_OF_CONDUCT.md>`_.

Contributing to PyVista through GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To submit new code to pyvista, first fork the `pyvista GitHub
Repository <https://github.com/pyvista/pyvista>`_ and then clone the forked
repository to your computer. Then, create a new branch based on the
`Branch Naming Conventions Section <#branch-naming-conventions>`_ in
your local repository.

Next, add your new feature and commit it locally. Be sure to commit
frequently as it is often helpful to revert to past commits, especially
if your change is complex. Also, be sure to test often. See the `Testing
Section <#testing>`_ below for automating testing.

When you are ready to submit your code, create a pull request by
following the steps in the `Creating a New Pull Request
section <#creating-a-new-pull-request>`_.

Coding Style
^^^^^^^^^^^^

We adhere to `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_
wherever possible, except that line widths are permitted to go beyond 79
characters to a max of 99 characters for code. This should tend to be
the exception rather than the norm. A uniform code style is enforced
by `ruff format <https://docs.astral.sh/ruff/formatter/#the-ruff-formatter>`_ to prevent energy wasted on
style disagreements.

Keyword-only arguments are generally preferred over positional keywords
in function signatures (see `PEP 3102 <https://peps.python.org/pep-3102/>`_),
and positional arguments should be limited to just one or two where possible.
Boolean-type arguments should always be keyword-only. This is also
enforced by ``ruff``.

As for docstrings, PyVista follows the ``numpydoc`` style for its docstrings.
Please also take a look at `Docstrings <#docstrings>`_.

Outside of PEP 8, when coding please consider `PEP 20 - The Zen of
Python <https://www.python.org/dev/peps/pep-0020/>`_. When in doubt:

.. code-block:: python

    import this

PyVista uses `pre-commit`_ to enforce PEP8 and other styles
automatically. Please see the `Style Checking section <#style-checking>`_ for
further details.

Documentation Style
^^^^^^^^^^^^^^^^^^^

PyVista follows the `Google Developer Documentation Style
<https://developers.google.com/style>`_ with the following exceptions:

- Allow first person pronouns. These pronouns (for example, "We") refer to
  "PyVista Developers", which can be anyone who contributes to PyVista.
- Future tense is permitted.

These rules are enforced for all text files (for example, ``*.md``, ``*.rst``)
and partially enforced for Python source files.

These rules are enforced through the use of `Vale <https://vale.sh/>`_ via our
GitHub Actions, and you can run Vale locally with:

.. code-block:: bash

   pip install vale
   vale --config doc/.vale.ini doc pyvista examples ./*.rst --glob='!*{_build,AUTHORS.rst}*'

If you are on Linux or macOS, you can run:

.. code-block:: bash

   make docstyle


Docstrings
^^^^^^^^^^

PyVista uses Python docstrings to create reference documentation for our Python
APIs. Docstrings are read by developers, interactive Python users, and readers
of our online documentation. This section describes how to write these docstrings
for PyVista.

PyVista follows the ``numpydoc`` style for its docstrings. Please follow the
`numpydoc Style Guide`_ in all ways except for the following:

* Be sure to describe all ``Parameters`` and ``Returns`` for all public
  methods.
* We strongly encourage you to add an example section. PyVista is a visual
  library, so adding examples that show a plot will really help users figure
  out what individual methods do.
* With optional parameters, use ``default: <value>`` instead of ``optional``
  when the parameter has a default value instead of ``None``.

Sample docstring follows:

.. code-block:: python

    def slice_x(self, x=None, generate_triangles=False):
        """Create an orthogonal slice through the dataset in the X direction.

        Parameters
        ----------
        x : float, optional
            The X location of the YZ slice. By default this will be the X center
            of the dataset.

        generate_triangles : bool, default: False
            If this is enabled, the output will be all triangles. Otherwise the
            output will consist of the intersection polygons.

        Returns
        -------
        pyvista.PolyData
            Sliced dataset.

        Examples
        --------
        Slice the random hills dataset with one orthogonal plane.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> slices = hills.slice_x(5, generate_triangles=False)
        >>> slices.plot(line_width=5)

        See :ref:`slice_example` for more examples using this filter.

        """

        pass  # implementation goes here

Note the following:

* The parameter definition of ``generate_triangles`` uses ``default: False``,
  and does not include the default in the docstring's "description" section.
* There is a newline between each parameter. This is different than
  ``numpydoc``'s documentation where there are no empty lines between parameter
  docstrings.
* This docstring also contains a returns section and an examples section.
* The returns section does not include the parameter name if the function has
  a single return value. Multiple return values (not shown) should have
  descriptive parameter names for each returned value, in the same format as
  the input parameters.
* The examples section references the "full example" in the gallery if it
  exists.

In addition, docstring examples which make use of randomly-generated data
should be reproducible. See `Generating Random Data`_ for details.

These standards will be enforced using ``pre-commit`` using
``numpydoc-validate``, with errors being reported as:

.. code-block:: text

   +-----------------+--------------------------+---------+-------------------------------------------------+
   | file            | item                     | check   | description                                     |
   +=================+==========================+=========+=================================================+
   | cells.py:85     | cells.create_mixed_cells | RT05    | Return value description should finish with "." |
   +-----------------+--------------------------+---------+-------------------------------------------------+
   | cells.py:85     | cells.create_mixed_cells | RT05    | Return value description should finish with "." |
   +-----------------+--------------------------+---------+-------------------------------------------------+
   | features.py:250 | features.merge           | PR09    | Parameter "datasets" description should finish  |
   |                 |                          |         | with "."                                        |
   +-----------------+--------------------------+---------+-------------------------------------------------+

If for whatever reason you feel that your function should have an exception to
any of the rules, add an exception to the function either in the
``[tool.numpydoc_validation]`` section in ``pyproject.toml`` or add an inline
comment to exclude a certain check. For example, we can omit the ``Return``
section from docstrings and skip the RT01 check for magic methods like ``__init__``.

.. code-block:: python

    def __init__(self, foo):  # numpydoc ignore=RT01
        """Initialize A Class."""
        super().__init__()
        self.foo = foo

See the available validation checks in `numpydoc Validation
<https://numpydoc.readthedocs.io/en/latest/validation.html>`_.


Deprecating Features or other Backwards-Breaking Changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When implementing backwards-breaking changes within PyVista, care must be taken
to give users the chance to adjust to any new changes. Any non-backwards
compatible modifications should proceed through the following steps:

#. Retain the old behavior and issue a ``PyVistaDeprecationWarning`` indicating
   the new interface you should use.
#. Retain the old behavior but raise a ``pyvista.core.errors.DeprecationError``
   indicating the new interface you must use.
#. Remove the old behavior.

Whenever possible, PyVista developers should seek to have at least three minor
versions of backwards compatibility to give users the ability to update their
software and scripts.

Here's an example of a soft deprecation of a function. Note the usage of both
the ``PyVistaDeprecationWarning`` warning and the ``.. deprecated`` Sphinx
directive.

.. code-block:: python

    import warnings
    from pyvista.core.errors import PyVistaDeprecationWarning


    def addition(a, b):
        """Add two numbers.

        .. deprecated:: 0.37.0
           Since PyVista 0.37.0, you can use :func:`pyvista.add` instead.

        Parameters
        ----------
        a : float
            First term to add.

        b : float
            Second term to add.

        Returns
        -------
        float
            Sum of the two inputs.

        """
        # deprecated 0.37.0, convert to error in 0.40.0, remove 0.41.0
        warnings.warn(
            '`addition` has been deprecated. Use pyvista.add instead',
            PyVistaDeprecationWarning,
        )
        add(a, b)


    def add(a, b):
        """Add two numbers."""

        pass  # implementation goes here

In the above code example, note how a comment is made to convert to an error in
three minor releases and completely remove in the following minor release. For
significant changes, this can be made longer, and for trivial ones this can be
kept short.

Here's an example of adding error test codes that raise deprecation warning messages.

.. code-block:: python

    with pytest.warns(PyVistaDeprecationWarning):
        addition(a, b)
        if pv._version.version_info[:2] > (0, 40):
            raise RuntimeError("Convert error this function")
        if pv._version.version_info[:2] > (0, 41):
            raise RuntimeError("Remove this function")

In the above code example, the old test code raises an error in v0.40 and v0.41.
This will prevent us from forgetting to remove deprecations on version upgrades.

.. note::

    When releasing a new version, we need to update the version number to the next
    development version. For example, if we are releasing version 0.37.0, the next
    development version should be 0.38.0.dev0 which is greater than 0.37.0. This is
    why we need to check if the version is greater than 0.40.0 and 0.41.0 in the
    test code.

When adding an additional parameter to an existing method or function, you are
encouraged to use the ``.. versionadded`` sphinx directive. For example:

.. code-block:: python

    def Cube(clean=True):
        """Create a cube.

        Parameters
        ----------
        clean : bool, default: True
            Whether to clean the raw points of the mesh.

            .. versionadded:: 0.33.0
        """


Branch Naming Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^

To streamline development, we have the following requirements for naming
branches. These requirements help the core developers know what kind of
changes any given branch is introducing before looking at the code.

-  ``fix/``, ``patch/`` and ``bug/``: any bug fixes, patches, or experimental changes that are
   minor
-  ``feat/``: any changes that introduce a new feature or significant
   addition
-  ``junk/``: for any experimental changes that can be deleted if gone
   stale
-  ``maint/`` and ``ci/``: for general maintenance of the repository or CI routines
-  ``doc/``: for any changes only pertaining to documentation
-  ``no-ci/``: for low impact activity that should NOT trigger the CI
   routines
-  ``testing/``: improvements or changes to testing
-  ``release/``: releases (see below)
-  ``breaking-change/``: Changes that break backward compatibility

Testing
^^^^^^^

After making changes, please test changes locally before creating a pull
request. The following tests will be executed after any commit or pull
request, so we ask that you perform the following sequence locally to
track down any new issues from your changes.

To run our comprehensive suite of unit tests, install PyVista with all
test dependencies:

.. code-block:: bash

   pip install -e . --group test

Then, if you have everything installed, you can run the various test
suites.

Unit Testing
~~~~~~~~~~~~
Run the primary test suite and generate coverage report:

.. code-block:: bash

   python -m pytest -v --cov pyvista

Unit testing can take some time, if you wish to speed it up, set the
number of processors with the ``-n`` flag. This uses ``pytest-xdist`` to
leverage multiple processes. Example usage:

.. code-block:: bash

   python -m pytest -n <NUMCORE> --cov pyvista

When submitting a PR, it is highly recommended that all modifications are thoroughly tested.
This is further enforced in the CI by the `codecov GitHub action <https://app.codecov.io/gh/pyvista/pyvista>`_
which has a 90% target, ie. it ensures that 90% of the code modified in the PR is tested.
It should be mentioned that branch coverage is measured on the CI, meaning for examples that both
values of an ``if`` clause must be tested to ensure full coverage. For more details on branch
coverage, please refer to the `coverage documentation <https://coverage.readthedocs.io/en/latest/branch.html>`_.

If needed, code coverage can be deactivated for specific lines by adding the ``# pragma: no cover`` or
``# pragma: no branch`` comments. See the documentation `excluding code <https://coverage.readthedocs.io/en/latest/branch.html#excluding-code>`__
for more details.
However, code coverage exclusion should rarely be used and has to be carefully justified in the PR thread
if no simple alternative solution has been found.

The CI is configured to test multiple vtk versions to ensure sufficient compatibility with vtk.
If needed, the minimum and/or maximum vtk version needed by a specific test can be controlled with a
custom pytest marker ``needs_vtk_version``, enabling the following usage (note the inclusive and exclusive signs):

.. code-block:: python

    @pytest.mark.needs_vtk_version(9, 1)
    def test():
        """Test is skipped if pv.vtk_version_info < (9,1)"""


    @pytest.mark.needs_vtk_version((9, 1))
    def test():
        """Test is skipped if pv.vtk_version_info < (9,1)"""


    @pytest.mark.needs_vtk_version(less_than=(9, 1))
    def test():
        """Test is skipped if pv.vtk_version_info >= (9,1)"""


    @pytest.mark.needs_vtk_version(at_least=(8, 2), less_than=(9, 1))
    def test():
        """Test is skipped if pv.vtk_version_info >= (9,1) or pv.vtk_version_info < (8,2,0)"""


    @pytest.mark.needs_vtk_version(less_than=(9, 1))
    @pytest.mark.needs_vtk_version(8, 2)
    def test():
        """Test is skipped if pv.vtk_version_info >= (9,1) or pv.vtk_version_info < (8,2,0)"""


    @pytest.mark.needs_vtk_version(9, 1, reason='custom reason')
    def test():
        """Test is skipped with a custom message"""

VTK Dev Wheel Testing
^^^^^^^^^^^^^^^^^^^^^
Most unit testing is run with stable VTK releases. However, it is sometimes useful to
run tests with the latest VTK dev wheels. To install these locally, run

.. code-block:: shell

    pip install vtk --upgrade --pre --extra-index-url https://wheels.vtk.org

For CI on GitHub, the ``vtk-dev-testing`` label can be used to enable unit testing with
the VTK dev wheels. The tests only run when the label is applied.

.. note::

    The PR either needs a new commit, e.g. updating the branch from ``main``, or to be
    closed/re-opened to rerun the CI with the label applied.

Docstring Testing
~~~~~~~~~~~~~~~~~
Run all code examples in the docstrings with:

.. code-block:: bash

   python -m pytest -v --doctest-modules pyvista

.. note::

    Additional testing is also performed on any images generated
    by the docstring. See `Documentation Image Regression Testing`_.


Style Checking
~~~~~~~~~~~~~~
PyVista follows PEP8 standard as outlined in the `Coding Style section
<#coding-style>`_ and implements style checking using `pre-commit`_.

To ensure your code meets minimum code styling standards, run::

  pip install pre-commit
  pre-commit run --all-files

If you have issues related to ``setuptools`` when installing ``pre-commit``, see
`pre-commit Issue #2178 comment <https://github.com/pre-commit/pre-commit/issues/2178#issuecomment-1002163763>`_
for a potential resolution.

You can also install this as a pre-commit hook by running::

  pre-commit install

This way, it's not possible for you to push code that fails the style
checks. For example, each commit automatically checks that you meet the style
requirements::

  $ pre-commit install
  $ git commit -m "added my cool feature"
  codespell................................................................Passed
  ruff.....................................................................Passed

The actual installation of the environment happens before the first commit
following ``pre-commit install``. This will take a bit longer, but subsequent
commits will only trigger the actual style checks.

Even if you are not in a situation where you are not performing or able to
perform the above tasks, you can comment ``pre-commit.ci autofix`` on a pull
request to manually trigger auto-fixing.

Notes Regarding Image Regression Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since PyVista is primarily a plotting module, it’s imperative we
actually check the images that we generate in some sort of regression
testing. In practice, this ends up being quite a bit of work because:

-  OpenGL software vs. hardware rending causes slightly different images
   to be rendered.
-  We want our CI (which uses a virtual frame buffer) to match our
   desktop images (uses hardware acceleration).
-  Different OSes render different images.

As each platform and environment renders different slightly images
relative to Linux (which these images were built from), so running these
tests across all OSes isn’t optimal. We need to know if
something fundamental changed with our plotting without actually looking
at the plots (like the docs at dev.pyvista.com)

Based on these points, image regression testing only occurs on Linux CI,
and multi-sampling is disabled as that seems to be one of the biggest
difference between software and hardware based rendering.

Image cache is stored here as ``./tests/plotting/image_cache``.

Image resolution is kept low at 400x400 as we don’t want to pollute git
with large images. Small variations between versions and environments
are to be expected, so error < ``IMAGE_REGRESSION_ERROR`` is allowable
(and will be logged as a warning) while values over that amount will
trigger an error.

There are two mechanisms within ``pytest`` to control image regression
testing, ``--reset_image_cache`` and ``--ignore_image_cache``. For
example:

.. code-block:: bash

       pytest tests/plotting --reset_image_cache

Running ``--reset_image_cache`` creates a new image for each test in
``tests/plotting/test_plotting.py`` and is not recommended except for
testing or for potentially a major or minor release. You can use
``--ignore_image_cache`` if you’re running on Linux and want to
temporarily ignore regression testing. Realize that regression testing
will still occur on our CI testing.

Images are currently only cached from tests in
``tests/plotting/test_plotting.py``. By default, any test that uses
``Plotter.show`` will cache images automatically. To skip image caching,
the ``verify_image_cache`` fixture can be utilized:

.. code-block:: python

    def test_add_background_image_not_global(verify_image_cache):
        verify_image_cache.skip = True  # Turn off caching
        plotter = pyvista.Plotter()
        plotter.add_mesh(sphere)
        plotter.show()
        # Turn on caching for further plotting
        verify_image_cache.skip = False
        ...

This ensures that immediately before the plotter is closed, the current
render window will be verified against the image in CI. If no image
exists, be sure to add the resulting image with

.. code-block:: bash

    git add tests/plotting/image_cache/*

During unit testing, if you get image regression failures and would like to
compare the images generated locally to the regression test suite, allow
`pytest-pyvista`_ to write all new
generated images to a local directory using the ``--generated_image_dir`` flag.

.. _pytest-pyvista: https://pytest.pyvista.org/

For example, the following writes all images generated by ``pytest`` to
``debug_images/`` for any tests in ``tests/plotting`` whose function name has
``volume`` in it.

.. code-block:: bash

   pytest tests/plotting/ -k volume --generated_image_dir debug_images

See `pytest-pyvista`_ for more details.

.. note::

    Additional regression testing is also performed on the documentation
    images. See `Documentation Image Regression Testing`_.

Notes Regarding Input Validation Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``pyvista.core.validation`` package has two distinct test suites which
are executed with ``pytest``:

#. Regular unit tests in ``tests/core/test_validation.py``
#. Customized unit tests in ``tests/core/typing`` for testing type hints

The custom unit tests check that the type hints for the validation package are
correct both statically and dynamically. This is mainly used to check complex and
overloaded function signatures, such as the type hints for ``validate_array``
or related functions.

Individual test cases are written as a single line of Python code with the format:

.. code-block:: python

    reveal_type(arg)  # EXPECTED_TYPE: "<T>"

where ``arg`` is any argument you want mypy to analyze, and ``"<T>"`` is the
expected revealed type returned by ``Mypy``.

For example, the ``validate_array`` function, by default, returns a list of floats
when a list of floats is provided at the input. The type hint should reflect this.
To test this, we can write a test case for the function call ``validate_array([1.0])``
as follows:

.. code-block:: python

    reveal_type(validate_array([1.0]))  # EXPECTED_TYPE: "list[float]"

The actual revealed type returned by ``Mypy`` for this test can be generated with
the following command. Note that ``grep`` is needed to only return the output
from the input string. Otherwise, all ``Mypy`` errors for the ``pyvista`` package
are reported.

.. code-block:: bash

    mypy -c "from pyvista.core._validation import validate_array; reveal_type(validate_array([1.0]))" | grep \<string\>

For this test case, the revealed type by ``Mypy`` is:

.. code-block:: python

    "builtins.list[builtins.float]"

Notice that the revealed type is fully qualified, i.e. includes ``builtins``. For
brevity, the custom test suite omits this and requires that only ``list`` be
included in the expected type. Therefore, for this test case, the ``EXPECTED_TYPE``
type is ``"list[float]"``, not ``"builtins.list[builtins.float]"``. (Similarly, the
package name ``numpy`` should also be omitted for tests where a ``numpy.ndarray`` is
expected.)

Any number of related test cases (one test case per line) may be written and
included in a single ``.py`` file. The test cases are all stored in
``tests/core/typing/validation_cases``.

The tests can be executed with:

.. code-block:: bash

    pytest tests/core/typing

When executed, a single instance of ``Mypy`` will statically analyze all the
test cases. The actual revealed types by ``Mypy`` are compared against the
``EXPECTED_TYPE`` is defined by each test case.

In addition, the ``pyanalyze`` package tests the actual returned
type at runtime to match the statically-revealed type. The
`pyanalyze.runtime.get_compatibility_error <https://pyanalyze.readthedocs.io/en/latest/reference/runtime.html#pyanalyze.runtime.get_compatibility_error>`_
method is used for this. If new typing test cases are added for a new
validation function, the new function must be added to the list of
imports in ``tests/core/typing/test_validation_typing.py`` so that the
runtime test can call the function.

Building the Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~
Install documentation dependencies with:

.. code-block:: shell

   python -m pip install -e . --group docs

Build the documentation on Linux or Mac OS with:

.. code-block:: bash

   make -C doc html

Build the documentation on Windows with:

.. code-block:: winbatch

   cd doc
   python -msphinx -M html source _build
   python -msphinx -M html . _build

The generated documentation can be found in the ``doc/_build/html``
directory.

The first time you build the documentation locally will take a while as all the
examples need to be built. After the first build, the documentation should take
a fraction of the time.

To test this locally you need to run a http server in the html directory with:

.. code-block:: bash

   make serve-html

Clearing the Local Build
^^^^^^^^^^^^^^^^^^^^^^^^

If you need to clear the locally built documentation, run:

.. code-block:: bash

   make -C doc clean

This will clear out everything, including the examples gallery. If you only
want to clear everything except the gallery examples, run:

.. code-block:: bash

   make -C doc clean-except-examples

This will clear out the cache without forcing you to rebuild all the examples.


Parallel Documentation Build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can improve your documentation build time on Linux and Mac OS with:

.. code-block:: bash

   make -C doc phtml

This effectively invokes ``SPHINXOPTS=-j`` and can be especially useful for
multi-core computers.

Documentation Image Regression Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Image regression testing is performed on all published documentation images.
When the documentation is built, all generated images are automatically
saved to

    Build Image Directory: ``./doc/_build/html/_images``

The regression testing compares these generated images to those stored in

    Doc Image Cache: ``./tests/doc/doc_image_cache``

To test all the images, run ``pytest`` with:

.. code-block:: bash

   pytest tests/doc/tst_doc_build.py::test_static_images

The tests must be executed explicitly with this command. The name of the test
file is prefixed with ``tst``, and not ``test`` specifically to avoid being
automatically executed by ``pytest`` (``pytest`` collects all tests prefixed
with ``test`` by default.) This is done since the tests require building the
documentation, and are not a primary form of testing.

When executed, the test will first pre-process the build images. The images are:

#. Collected from the ``Build Image Directory``.

#. Resized to a maximum of 400x400 pixels.

#. Saved to a flat directory as JPEG images in ``./_doc_debug_images``.

Next, the pre-processed images in ``./_doc_debug_images`` are compared to the
cached images in the ``Doc Image Cache`` using :func:`pyvista.compare_images`.

The tests can fail in three ways. To make it easy to review images for failed tests,
copies of the images are made as follows:

#. If the comparison between the two images fails:

    - The cache image is copied to ``./_doc_debug_images_failed/errors/from_cache``
    - The build image is copied to ``./_doc_debug_images_failed/errors/from_build``

#.  If an image is in the cache but missing from the build:

    - The cache image is copied to  ``./_doc_debug_images_failed/errors/from_cache``

#.  If an image is in the build but missing from the cache:

    - The build image is copied to  ``./_doc_debug_images_failed/errors/from_build``

If a warning is generated instead of an error, images are saved to the
``warnings`` sub-directory instead of ``errors``.

To resolve failed tests, any images in ``from_build`` or ``from_cache``
may be copied to or removed from the ``Doc Image Cache``. For example,
if adding new docstring examples or plots, the test will initially fail,
and the images in ``from_build`` may be added to the ``Doc Image Cache``.
Similarly, if removing examples, the images in ``from_cache`` may be removed
from the ``Doc Image Cache``.

If a test is flaky, e.g. the build sometimes generates different images
for the same plot, the multiple versions of the image may be saved to the
flaky test directory ``./tests/doc/flaky_tests``. A folder with the same
name as the test image should be created, and all versions of the image
should be stored in this directory. The test will first compare the
build image to the cached image in ``Doc Image Cache`` as normal. If that
comparison fails, the build image is then compared to all images in the
flaky test directory. The test is successful if one of the comparisons
is successful, but a warning will still be issued. If a warning is
emitted by a flaky test, images are saved to the ``flaky`` sub-directory
instead of ``warnings``.

.. note::

    It is not necessary to build the documentation images locally in order
    to add to or update the doc image cache. The documentation is automatically
    built as part of CI testing, and an artifact is generated for (1) all
    pre-processed build images and (2) failed test cases. These artifacts may
    simply be downloaded from GitHub for review.

    The debug images saved with the artifact can also be used to "simulate"
    building the documentation images locally. If the images are copied to the
    local ``Build Image Directory``, the tests can then be executed locally for
    debugging as though the documentation has already been built.

.. note::

   These tests are intended to provide *additional* test coverage to ensure the
   plots generated by ``pyvista`` are correct, and should not be used as the
   primary source of testing. See `Docstring Testing`_ and
   `Notes Regarding Image Regression Testing`_ for testing methods which should
   be considered first.

Interactive Plot Testing
^^^^^^^^^^^^^^^^^^^^^^^^

PyVista's documentation uses a custom ``pyvista-plot`` directive to generate
static images as well as interactive plot files. The interactive files have a
``.vtksz`` extension and can be relatively large when plotting high-resolution
datasets.

To ensure that the interactive plots do not unnecessarily inflate the size
of the documentation build, a limit is placed on the size of ``.vtksz`` files.
To test that interactive plots do not exceed this limit, run:

.. code:: bash

   pytest tests/doc/tst_doc_build.py::test_interactive_plot_file_size

If any of these tests fail, the example(s) which generated the plot should be
modified, e.g.:

#. Simplify any dataset(s) used, e.g. crop, clip, down-sample, decimate, or
   otherwise reduce the complexity of the plot.

#. Force the plot to be static only.
   In docstrings, use the plot directive with the ``force_static`` option, e.g.:

    .. code:: rst

        .. pyvista-plot::
           :force_static:

           >>> import pyvista as pv
           >>> # Your example code here
           >>> # ...
           >>> mesh = pv.sphere()
           >>> mesh.plot()

   In sphinx gallery examples use:

   .. code:: python

       # sphinx_gallery_start_ignore
       PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
       # sphinx_gallery_end_ignore

   to disable all plots in the example or use ``PYVISTA_GALLERY_FORCE_STATIC``
   before the call to ``plot()`` or ``show()`` to force static for a single
   plot. See :ref:`add_example_example` for more information.

.. note::

    Reducing the complexity of the plot is preferred as this will also
    also likely reduce the processing times.

.. seealso::

    See `Documentation Image Regression Testing`_. for testing performed on
    the static images generated by the plot directive.

Controlling Cache for CI Documentation Build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To reduce build times of the documentation for PRs, cached sphinx gallery, example data, and sphinx build directories
are used in the CI on GitHub.  In some cases, the caching action can cause problems for a specific
PR.  To invalidate a cache for a specific PR, one of the following labels can be applied to the PR.

- ``no-example-data-cache``
- ``no-gallery-cache``
- ``no-sphinx-build-cache``

The PR either needs a new commit, e.g. updating the branch from ``main``, or to be closed/re-opened to
rerun the CI with the labels applied.


Contributing to the Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Documentation for PyVista is generated from three sources:

- Docstrings from the classes, functions, and modules of ``pyvista`` using
  `sphinx.ext.autodoc
  <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_.
- Restructured test from ``doc/``
- Gallery examples from ``examples/``

General usage and API descriptions should be placed within ``doc/api`` and
the docstrings. Full gallery examples should be placed in ``examples``.


Generating Random Data
^^^^^^^^^^^^^^^^^^^^^^
All documentation should be reproducible. In particular, any documentation
or examples which use random data should be properly seeded so that the
same random data is generated every time. This enables users to copy code
in the documentation and generate the same results and plots locally.

When using NumPy's random number generator (RNG) you should create an RNG at
the beginning of your script and use this RNG in the rest of the script. Be
sure to include a seed value. For example:

.. code-block:: python

    import numpy as np

    rng = np.random.default_rng(seed=0)
    rng.random()  # generate a floating point number between 0 and 1

See Scientific Python's `Best Practices for Using NumPy's Random Number Generators
<https://blog.scientific-python.org/numpy/numpy-rng/>`_ for details.

Adding a New Example
^^^^^^^^^^^^^^^^^^^^
PyVista's examples come in two formats: basic code snippets demonstrating the
functionality of an individual method or a full gallery example displaying one
or more concepts. Small code samples and snippets are contained in the
``doc/api`` directory or within our documentation strings, while the full
gallery examples, meant to be run as individual downloadable scripts, are
contained in the ``examples`` directory at the root of this repository.

To add a fully fledged, standalone example, add your example to the
``examples`` directory in the root directory of the `PyVista Repository
<https://github.com/pyvista/pyvista/>`_ within one of the applicable
subdirectories. Should none of the existing directories match the category of
your example, create a new directory with a ``README.txt`` describing the new
category. Additionally, as these examples are built using the sphinx gallery
extension, follow coding guidelines as established by `Sphinx-Gallery
<https://sphinx-gallery.github.io/stable/index.html>`_.

For more details see :ref:`add_example_example`.


Adding a New Dataset
^^^^^^^^^^^^^^^^^^^^
If you have a dataset that you want to feature or want to include as part
of a full gallery example, add it to `pyvista/vtk-data <https://github.com/pyvista/vtk-data/>`_
and follow the directions there. You will then need to add a new function to
download the dataset in ``pyvista/examples/downloads.py``. This might be as easy as:

.. code-block:: python

    def download_my_new_mesh(load=True):
        """Download my new mesh."""
        return _download_dataset(_dataset_my_new_mesh, load=load)


    _dataset_my_new_mesh = _SingleFileDownloadableDatasetLoader(
        'mydata/my_new_mesh.vtk'
    )

Note that a separate dataset loading object, ``_dataset_my_new_mesh``, should
first be defined outside of the function (with module scope), and the new
``download_my_new_mesh`` function should then use this object to facilitate
downloading and loading the dataset. The dataset loader variable should start
with ``_dataset_``.

This will enable:

.. code-block::

   >>> from pyvista import examples
   >>> dataset = examples.download_my_new_mesh()

For loading complex datasets with multiple files or special processing
requirements, see the private ``pyvista/examples/_dataset_loader.py``
module for more details on how to create a suitable dataset loader.

Using a dataset loader in this way will enable metadata to be collected
for the new dataset. A new dataset card titled ``My New Mesh Dataset``
will automatically be generated and included in the :ref:`dataset_gallery`.

In the docstring of the new ``download_my_new_mesh`` function, be sure
to also include:

#. A sample plot of the dataset in the examples section

#. A reference link to the dataset's new (auto-generated) gallery card
   in the see also section

For example:

.. code-block:: python

    def download_my_new_mesh(load=True):
        """Download my new mesh.

        Examples
        --------
        >>> from pyvista import examples
        >>> dataset = examples.download_my_new_mesh()
        >>> dataset.plot()

        .. seealso::

           :ref:`My New Mesh Dataset <my_new_mesh_dataset>`
               See this dataset in the Dataset Gallery for more info.

        """

.. note::

   The rst ``seealso`` directive must be used instead of the
   ``See Also`` heading due to limitations with how ``numpydoc`` parses
   explicit references.

Extending the Dataset Gallery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you have multiple related datasets to contribute, or would like to
group any existing datasets together that share similar properties,
the :ref:`dataset_gallery` can easily be extended to feature these
datasets in a new `card carousel <https://sphinx-design.readthedocs.io/en/latest/cards.html#card-carousels>`_.

For example, to add a new ``Instrument`` dataset category to :ref:`dataset_gallery_category`
featuring two datasets of musical instruments, e.g.

#.  :func:`pyvista.examples.downloads.download_guitar`
#.  :func:`pyvista.examples.downloads.download_trumpet`

complete the following steps:

#. Define a new carousel in ``doc/source/make_tables.py``, e.g.:

    .. code-block:: python

        class InstrumentCarousel(DatasetGalleryCarousel):
            """Class to generate a carousel of instrument dataset cards."""

            name = 'instrument_carousel'
            doc = 'Instrument datasets.'
            badge = CategoryBadge('Instrument', ref='instrument_gallery')

            @classmethod
            def fetch_dataset_names(cls):
                return sorted(
                    (
                        'guitar',
                        'trumpet',
                    )
                )

   where

   -  ``name`` is used internally to define the name of the generated
      ``.rst`` file for the carousel.

   -  ``doc`` is a short text description of the carousel which will
      appear in the documentation in the header above the carousel.

   -  ``badge`` is used to give all datasets in the carousel a reference
      tag. The ``ref`` argument for the badge should be a new reference
      target (details below).

   -  ``fetch_dataset_names`` should return a list of any/all dataset names
      to be included in the carousel. The dataset names should not include
      any ``load_``, ``download_``, or ``dataset_`` prefix.

#. Add the new carousel class to the ``CAROUSEL_LIST`` variable defined
   in ``doc/source/make_tables.py``. This will enable the rst to be
   auto-generated for the carousel.

#. Update the ``doc/source/api/examples/dataset_gallery.rst`` file to
   include the new generated ``<name>_carousel.rst`` file. E.g. to add the
   carousel as a new drop-down item, add the following:

   .. code-block:: rst

      .. dropdown:: Instrument Datasets
         :name: instrument_gallery

         .. include:: /api/examples/dataset-gallery/instrument_carousel.rst

   where:

   -  The dropdown name ``:name: <reference>`` should be the badge's ``ref``
      variable defined earlier. This will make it so that clicking on the new
      badge will link to the new dropdown menu.

   -  The name of the included ``.rst`` file should match the ``name``
      variable defined in the new ``Carousel`` class.

After building the documentation, the carousel should now be part
of the gallery.

Creating a New Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have tested your branch locally, create a pull request on
`pyvista GitHub <https://github.com/pyvista/pyvista>`_ while merging to
main. This will automatically run continuous integration (CI) testing
and verify your changes will work across several platforms.

To ensure someone else reviews your code, at least one other member of
the pyvista contributors group must review and verify your code meets
our community’s standards. Once approved, if you have write permission
you may merge the branch. If you don’t have write permission, the
reviewer or someone else with write permission will merge the branch and
delete the PR branch.

Since it may be necessary to merge your branch with the current release
branch (see below), please do not delete your branch if it is a ``fix/``
branch.

Preview the Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~
For PRs of branches coming from the main pyvista repository, the documentation
is automatically deployed using `Netifly GitHub actions <https://github.com/nwtgck/actions-netlify>`_.
However, new contributors that submit PRs from a fork can download a light-weight documentation CI artifact
that contains a non-interactive subset of the documentation build. It typically weights
500 Mb and is available from the ``Upload non-interactive HTML documentation`` step of the
``Build Documentation`` CI job.

Branching Model
~~~~~~~~~~~~~~~

This project has a branching model that enables rapid development of
features without sacrificing stability, and closely follows the `Trunk
Based Development <https://trunkbaseddevelopment.com/>`_ approach.

The main features of our branching model are:

-  The ``main`` branch is the main development branch. All features,
   patches, and other branches should be merged here. While all PRs
   should pass all applicable CI checks, this branch may be functionally
   unstable as changes might have introduced unintended side-effects or
   bugs that were not caught through unit testing.
-  There will be one or many ``release/`` branches based on minor
   releases (for example ``release/0.24``) which contain a stable
   version of the code base that is also reflected on PyPI/. Hotfixes
   from ``fix/`` branches should be merged both to main and to these
   branches. When necessary to create a new patch release these release
   branches will have their ``pyvista/_version.py`` updated and be tagged
   with a semantic version (for example ``v0.24.1``). This triggers CI
   to push to PyPI, and allow us to rapidly push hotfixes for past
   versions of ``pyvista`` without having to worry about untested
   features.
-  When a minor release candidate is ready, a new ``release`` branch
   will be created from ``main`` with the next incremented minor version
   (for example ``release/0.25``), which will be thoroughly tested. When deemed
   stable, the release branch will be tagged with the version
   (``v0.25.0`` in this case), and if necessary merged with main if any
   changes were pushed to it. Feature development then continues on
   ``main`` and any hotfixes will now be merged with this release. Older
   release branches should not be deleted so they can be patched as
   needed.

Minor Release Steps
^^^^^^^^^^^^^^^^^^^

Minor releases are feature and bug releases that improve the
functionality and stability of ``pyvista``. Before a minor release is
created the following will occur:

#.  Create a new branch from the ``main`` branch with name
    ``release/MAJOR.MINOR`` (for example ``release/0.25``).

#.  Update the development version numbers in ``pyvista/_version.py``
    and commit it (for example ``0, 26, 'dev0'``). Push the branch to GitHub
    and create a new PR for this release that merges it to main.
    Development to main should be limited at this point while effort
    is focused on the release.

#.  Locally run all tests as outlined in the `Testing
    Section <#testing>`_ and ensure all are passing.

#.  Locally test and build the documentation with link checking to make
    sure no links are outdated. Be sure to run ``make clean`` to ensure
    no results are cached.

    .. code-block:: bash

       cd doc
       make clean  # deletes the sphinx-gallery cache
       make doctest-modules
       make html -b linkcheck

#.  After building the documentation, open the local build and examine
    the examples gallery for any obvious issues.

#.  It is now the responsibility of the ``pyvista`` community to
    functionally test the new release. It is best to locally install
    this branch and use it in production. Any bugs identified should
    have their hotfixes pushed to this release branch.

#.  When the branch is deemed as stable for public release, the PR will
    be merged to main. After update the version number in
    ``release/MAJOR.MINOR`` branch, the ``release/MAJOR.MINOR`` branch
    will be tagged with a ``vMAJOR.MINOR.0`` release. The release branch
    will not be deleted. Tag the release with:

    .. code-block:: bash

       git tag v$(python -c "import pyvista as pv; print(pv.__version__)")

#.  Please check again that the tag has been created correctly and push the branch and tag.

    .. code-block:: bash

       git push origin HEAD
       git push origin v$(python -c "import pyvista as pv; print(pv.__version__)")

#.  Create a list of all changes for the release. It is often helpful to
    leverage `GitHub’s compare
    feature <https://github.com/pyvista/pyvista/compare>`_ to see the
    differences from the last tag and the ``main`` branch. Be sure to
    acknowledge new contributors by their GitHub username and place
    mentions where appropriate if a specific contributor is to thank for
    a new feature.

#.  Place your release notes from previous step in the description for `the new
    release on
    GitHub <https://github.com/pyvista/pyvista/releases/new>`_.

#.  Go grab a beer/coffee/water and wait for
    `@regro-cf-autotick-bot <https://github.com/regro/cf-scripts>`_
    to open a pull request on the conda-forge `PyVista
    feedstock <https://github.com/conda-forge/pyvista-feedstock>`_.
    Merge that pull request.

#.  Announce the new release in the Discussions page and
    celebrate.

Patch Release Steps
^^^^^^^^^^^^^^^^^^^

Patch releases are for critical and important bugfixes that can not or
should not wait until a minor release. The steps for a patch release

#. Push the necessary bugfix(es) to the applicable release branch. This
   will generally be the latest release branch (for example ``release/0.25``).

#. Update ``pyvista/_version.py`` with the next patch increment (for example
   ``v0.25.1``), commit it, and open a PR that merge with the release
   branch. This gives the ``pyvista`` community a chance to validate and
   approve the bugfix release. Any additional hotfixes should be outside
   of this PR.

#. When approved, merge with the release branch, but not ``main`` as
   there is no reason to increment the version of the ``main`` branch.
   Then create a tag from the release branch with the applicable version
   number (see above for the correct steps).

#. If deemed necessary, create a release notes page. Also, open the PR
   from conda and follow the directions in step 10 in the minor release
   section.

Dependency version policy
-------------------------

Python and VTK dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We support all supported `Python versions`_ and `VTK versions`_ that
support those Python versions. As much as we would prefer to follow
`SPEC 0`_, we follow VTK versions as an interface library of VTK.

.. _pre-commit: https://pre-commit.com/
.. _numpydoc Style Guide: https://numpydoc.readthedocs.io/en/latest/format.html
.. _Python versions: https://endoflife.date/python
.. _VTK versions: https://pypi.org/project/vtk/
.. _SPEC 0: https://scientific-python.org/specs/spec-0000/


Self-hosted runners
-------------------
GitHub hosted runners are the preferred way of running PyVista's CI. However
given the volume of development, the number of workflows, and the need to test
across several operating systems, it may be necessary to use self-hosted
runners due to GitHub's concurrency limits.

Any PyVista self-hosted runner must:

- Be as compatible as possible with a GitHub hosted runner.
- Use labels to denote hardware and software and match GitHub's labels whenever
  possible (e.g. ``GPU``, ``ubuntu-22.04``, ``macos-15``)
- Be secure against intrusion and follow best cybersecurity practices (e.g. no
  ``sudo`` permissions, dedicated and isolated VLAN)
- Require a compatible CI/CD workflow.
- Provide runner documentation here.
- Be on a host with a battery backup.

Setting up a runner on bare metal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visit PyVista's `Create self-hosted runner
<https://github.com/organizations/pyvista/settings/actions/runners/new>`_.

Follow the directions to download, run and install. If the runner is intended
to run public workflows, add the runner to the ``pyvista-self-hosted`` group.

Follow your OSes instructions to enable a service for the runner (if
applicable) to ensure the runner restarts should it be interrupted.

PyVista Hosts and Runners
~~~~~~~~~~~~~~~~~~~~~~~~~

Apple 2024 Mac mini M4
^^^^^^^^^^^^^^^^^^^^^^
- CPU: 10-core CPU ARM64 (Apple Silicon)
- GPU: 10-core GPU
- Storage: 256 GB SSD
- Memory: 16 GB Unified Memory
- OS: MacOS 15

With the following runners
- macos-arm-runner-0
- macos-arm-runner-1
- macos-arm-runner-2
- macos-arm-runner-3
- macos-arm-runner-4

**Notes**
Testing showed peak memory usage of ~2GB per runner for the
``testing-and-deployment.yml`` workflow. With 16GB of memory and ~4 GB used by
the OS, there's room to spare. Should we encounter memory issues we can disable
runners.
