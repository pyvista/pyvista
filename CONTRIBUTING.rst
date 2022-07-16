Contributing
============

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
and the communities who have worked diligently to build (un-)related tools.

Please do not talk down in Pull Requests, Issues, or otherwise in a way that
portrays other people or their works in a negative light.

Cloning the Source Repository
-----------------------------

You can clone the source repository from
`<https://github.com/pyvista/pyvista>`_ and install the latest version by
running:

.. code:: bash

   git clone https://github.com/pyvista/pyvista.git
   cd pyvista
   python -m pip install -e .

Questions
---------

For general questions about the project, its applications, or about
software usage, please create a discussion in the
`Discussions <https://github.com/pyvista/pyvista/discussions>`_
repository where the community can collectively address your questions.
You are also welcome to join us on `Slack <http://slack.pyvista.org>`_
or send one of the developers an email. The project support team can be
reached at info@pyvista.org

For more technical questions, you are welcome to create an issue on the
`issues page <https://github.com/pyvista/pyvista/issues>`_ which we
will address promptly. Through posting on the issues page, your question
can be addressed by community members with the needed expertise and the
information gained will remain available on the issues page for other
users.

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
base! Please create an issue on the `issues
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
further PyVista towards being the de facto Python interface to VTK, we
need your help to make it even better!

If you want to add one or two interesting analysis algorithms as
filters, implement a new plotting routine, or just fix 1-2 typos - your
efforts are welcome!

There are three general coding paradigms that we believe in:

1. **Make it intuitive**. PyVista’s goal is to create an intuitive and
   easy to use interface back to the VTK library. Any new features
   should have intuitive naming conventions and explicit keyword
   arguments for users to make the bulk of the library accessible to
   novice users.

2. **Document everything!** At the least, include a docstring for any
   method or class added. Do not describe what you are doing but why you
   are doing it and provide a simple example for the new features.

3. **Keep it tested**. We aim for a high test coverage. See testing for
   more details.

There are two important copyright guidelines:

4. Please do not include any data sets for which a license is not
   available or commercial use is prohibited. Those can undermine the
   license of the whole projects.

5. Do not use code snippets for which a license is not available
   (e.g. from stackoverflow) or commercial use is prohibited. Those can
   undermine the license of the whole projects.

Please also take a look at our `Code of
Conduct <https://github.com/pyvista/pyvista/blob/main/CODE_OF_CONDUCT.md>`_.

Contributing to pyvista through GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To submit new code to pyvista, first fork the `pyvista GitHub
Repo <https://github.com/pyvista/pyvista>`_ and then clone the forked
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
by `black <https://github.com/psf/black>`_ to prevent energy wasted on
style disagreements.

As for docstrings, follow the guidelines specified in `PEP 8 Maximum
Line
Length <https://www.python.org/dev/peps/pep-0008/#maximum-line-length>`_
of limiting docstrings to 72 characters per line. This follows the
directive:

   Some teams strongly prefer a longer line length. For code maintained
   exclusively or primarily by a team that can reach agreement on this
   issue, it is okay to increase the line length limit up to 99
   characters, provided that comments and docstrings are still wrapped
   at 72 characters.

Outside of PEP 8, when coding please consider `PEP 20 – The Zen of
Python <https://www.python.org/dev/peps/pep-0020/>`_. When in doubt:

Docstrings
^^^^^^^^^^^^

When using Sphinx in combination with the NumPy conventions, you should 
use the numpydoc extension so that your docstrings will be handled 
correctly. For example, Sphinx will extract the Parameters section from 
your docstring and convert it into a field list. Using numpydoc will also 
avoid the reStructuredText errors produced by plain Sphinx when it 
encounters NumPy docstring conventions like section headers 
(e.g. -------------) that sphinx does not expect to find in docstrings.

It is available from:

-  numpydoc on PyPI

-  numpydoc on GitHub

Note that for documentation within NumPy, it is not necessary to do import numpy as np at the beginning of an example.

Please use the numpydoc formatting standard as shown in their example.
.. code:: python

   import this

PyVista uses `pre-commit`_ to enforce PEP8 and other styles
automatically. Please see the `Style Checking section <#style-checking>`_ for
further details.


Branch Naming Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^

To streamline development, we have the following requirements for naming
branches. These requirements help the core developers know what kind of
changes any given branch is introducing before looking at the code.

-  ``fix/``: any bug fixes, patches, or experimental changes that are
   minor
-  ``feat/``: any changes that introduce a new feature or significant
   addition
-  ``junk/``: for any experimental changes that can be deleted if gone
   stale
-  ``maint/``: for general maintenance of the repository or CI routines
-  ``doc/``: for any changes only pertaining to documentation
-  ``no-ci/``: for low impact activity that should NOT trigger the CI
   routines
-  ``testing/``: improvements or changes to testing
-  ``release/``: releases (see below)

Testing
^^^^^^^

After making changes, please test changes locally before creating a pull
request. The following tests will be executed after any commit or pull
request, so we ask that you perform the following sequence locally to
track down any new issues from your changes.

To run our comprehensive suite of unit tests, install all the
dependencies listed in ``requirements_test.txt``,
``requirements_docs.txt``

.. code:: bash

   pip install -r requirements_test.txt
   pip install -r requirements_docs.txt

Then, if you have everything installed, you can run the various test
suites.

Unit Testing
~~~~~~~~~~~~
Run the primary test suite and generate coverage report:

.. code:: bash

   python -m pytest -v --cov pyvista

Unit testing can take some time, if you wish to speed it up, set the
number of processors with the ``-n`` flag. This uses ``pytest-xdist`` to
leverage multiple processes. Example usage:

.. code:: bash

   python -m pytest -n <NUMCORE> --cov pyvista

Documentation Testing
~~~~~~~~~~~~~~~~~~~~~
Run all code examples in the docstrings with:

.. code:: bash

   python -m pytest -v --doctest-modules pyvista

Build the documentation on Linux or Mac OS with:

.. code:: bash

   make -C doc html

Build the documentation on Windows with:

   cd doc
   python -msphinx -M html . _build

The generated documentation can be found in the ``doc/_build/html``
directory.

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
  black....................................................................Passed
  isort....................................................................Passed
  flake8...................................................................Passed
  codespell................................................................Passed

The actual installation of the environment happens before the first commit 
following ``pre-commit install``. This will take a bit longer, but subsequent
commits will only trigger the actual style checks.

Notes Regarding Image Regression Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since PyVista is primarily a plotting module, it’s imperative we
actually check the images that we generate in some sort of regression
testing. In practice, this ends up being quite a bit of work because:

-  OpenGL software vs. hardware rending causes slightly different images
   to be rendered.
-  We want our CI (which uses a virtual frame buffer) to match our
   desktop images (uses hardware acceleration).
-  Different OSes render different images.

As each platform and environment renders different slightly images
relative to Linux (which these images were built from), so running these
tests across all OSes isn’t optimal. We could generate different images
for each OS, but it’s overkill in my opinion; we need to know if
something fundamental changed with our plotting without actually looking
at the plots (like the docs at dev.pyvista.com)

Based on these points, image regression testing only occurs on Linux CI,
and multi-sampling is disabled as that seems to be one of the biggest
difference between software and hardware based rendering.

Image cache is stored here as ``./image_cache``.

Image resolution is kept low at 400x400 as we don’t want to pollute git
with large images. Small variations between versions and environments
are to be expected, so error < ``IMAGE_REGRESSION_ERROR`` is allowable
(and will be logged as a warning) while values over that amount will
trigger an error.

There are two mechanisms within ``pytest`` to control image regression
testing, ``--reset_image_cache`` and ``--ignore_image_cache``. For
example:

.. code:: bash

       pytest tests/plotting --reset_image_cache

Running ``--reset_image_cache`` creates a new image for each test in
``tests/plotting/test_plotting.py`` and is not recommended except for
testing or for potentially a major or minor release. You can use
``--ignore_image_cache`` if you’re running on Linux and want to
temporarily ignore regression testing. Realize that regression testing
will still occur on our CI testing.

If you need to add a new test to ``tests/plotting/test_plotting.py`` and
wish to include image regression testing, be sure to add
``verify_cache_image`` to ``show``. For example:

.. code:: python


       @skip_no_plotting
       def test_add_background_image_not_global():
           plotter = pyvista.Plotter()
           plotter.add_mesh(sphere)
           plotter.show(before_close_callback=verify_cache_image)

This ensures that immediately before the plotter is closed, the current
render window will be verified against the image in CI. If no image
exists, be sure to add the resulting image with

.. code:: bash

    git add tests/plotting/image_cache/*

Creating a New Pull Request
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
   with a semantic version (e.g. ``v0.24.1``). This triggers CI
   to push to PyPI, and allow us to rapidly push hotfixes for past
   versions of ``pyvista`` without having to worry about untested
   features.
-  When a minor release candidate is ready, a new ``release`` branch
   will be created from ``main`` with the next incremented minor version
   (e.g. ``release/0.25``), which will be thoroughly tested. When deemed
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

1.  Create a new branch from the ``main`` branch with name
    ``release/MAJOR.MINOR`` (e.g. ``release/0.25``).

2.  Locally run all tests as outlined in the `Testing
    Section <#testing>`_ and ensure all are passing.

3.  Locally test and build the documentation with link checking to make
    sure no links are outdated. Be sure to run ``make clean`` to ensure
    no results are cached.

    .. code:: bash

       cd doc
       make clean  # deletes the sphinx-gallery cache
       make doctest
       make html -b linkcheck

4.  After building the documentation, open the local build and examine
    the examples gallery for any obvious issues.

5.  Update the development version numbers in ``pyvista/_version.py``
    and commit it (e.g. ``0, 26, 'dev0'``). Push the branch to GitHub
    and create a new PR for this release that merges it to main.
    Development to main should be limited at this point while effort
    is focused on the release.

6.  It is now the responsibility of the ``pyvista`` community to
    functionally test the new release. It is best to locally install
    this branch and use it in production. Any bugs identified should
    have their hotfixes pushed to this release branch.

7.  When the branch is deemed as stable for public release, the PR will
    be merged to main. After update the version number in
    ``release/MAJOR.MINOR`` branch, the ``release/MAJOR.MINOR`` branch
    will be tagged with a ``vMAJOR.MINOR.0`` release. The release branch
    will not be deleted. Tag the release with:

    .. code:: bash

       git tag v$(python -c "import pyvista as pv; print(pv.__version__)")
       git push origin --tags

8.  Create a list of all changes for the release. It is often helpful to
    leverage `GitHub’s compare
    feature <https://github.com/pyvista/pyvista/compare>`_ to see the
    differences from the last tag and the ``main`` branch. Be sure to
    acknowledge new contributors by their GitHub username and place
    mentions where appropriate if a specific contributor is to thank for
    a new feature.

9.  Place your release notes from step 8 in the description for `the new
    release on
    GitHub <https://github.com/pyvista/pyvista/releases/new>`_.

10. Go grab a beer/coffee/water and wait for
    `@regro-cf-autotick-bot <https://github.com/regro-cf-autotick-bot>`_
    to open a pull request on the conda-forge `PyVista
    feedstock <https://github.com/conda-forge/pyvista-feedstock>`_.
    Merge that pull request.

11. Announce the new release in the PyVista Slack workspace and
    celebrate!

Patch Release Steps
^^^^^^^^^^^^^^^^^^^

Patch releases are for critical and important bugfixes that can not or
should not wait until a minor release. The steps for a patch release

1. Push the necessary bugfix(es) to the applicable release branch. This
   will generally be the latest release branch (e.g. ``release/0.25``).

2. Update ``pyvista/_version.py`` with the next patch increment (e.g.
   ``v0.25.1``), commit it, and open a PR that merge with the release
   branch. This gives the ``pyvista`` community a chance to validate and
   approve the bugfix release. Any additional hotfixes should be outside
   of this PR.

3. When approved, merge with the release branch, but not ``main`` as
   there is no reason to increment the version of the ``main`` branch.
   Then create a tag from the release branch with the applicable version
   number (see above for the correct steps).

4. If deemed necessary, create a release notes page. Also, open the PR
   from conda and follow the directions in step 10 in the minor release
   section.


.. _pre-commit: https://pre-commit.com/
