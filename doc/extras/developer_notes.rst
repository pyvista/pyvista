.. _ref_developer_notes:

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

Cloning the Source Repository
-----------------------------

You can clone the source repository from
``https://github.com/pyvista/pyvista`` and install the latest version by
running:

.. code:: bash

    git clone https://github.com/pyvista/pyvista.git
    cd pyvista
    python -m pip install -e .


Questions
---------

For general questions about the project, its applications, or about
software usage, please create a discussion in
`pyvista/discussions <https://github.com/pyvista/pyvista/discussions>`__
where the community can collectively address your questions.
You are also welcome to join us on `Slack <http://slack.pyvista.org>`__
or send one of the developers an email. The project support team can be
reached at info@pyvista.org

For more technical questions, you are welcome to create an issue on the
`issues page <https://github.com/pyvista/pyvista/issues>`__ which we
will address promptly. Through posting on the issues page, your question
can be addressed by community members with the needed expertise and the
information gained will remain available on the issues page for other
users.

Reporting Bugs
--------------

If you stumble across any bugs, crashes, or concerning quirks while
using code distributed here, please report it on the `issues
page <https://github.com/pyvista/pyvista/issues>`__ with an appropriate
label so we can promptly address it. When reporting an issue, please be
overly descriptive so that we may reproduce it. Whenever possible,
please provide tracebacks, screenshots, and sample files to help us
address the issue.

Feature Requests
----------------

We encourage users to submit ideas for improvements to PyVista code
base! Please create an issue on the `issues
page <https://github.com/pyvista/pyvista/issues>`__ with a *Feature
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
the `Development Practices <#development-practices>`__ section for more
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

1. **Make it intuitive**. PyVista's goal is to create an intuitive and
   easy to use interface back to the VTK library. Any new features
   should have intuitive naming conventions and explicit keyword
   arguments for users to make the bulk of the library accessible to
   novice users.

2. **Document everything!** At the least, include a docstring for any
   method or class added. Do not describe what you are doing but why you
   are doing it and provide simple use cases for the new features.

3. **Keep it tested**. We aim for a high test coverage. See testing for
   more details.

There are two important copyright guidelines:

4. Please do not include any data sets for which a license is not
   available or commercial use is prohibited. Those can undermine the
   license of the whole projects.

5. Do not use code snippets for which a license is not available (e.g.
   from stackoverflow) or commercial use is prohibited. Those can
   undermine the license of the whole projects.

Please also take a look at our `Code of
Conduct <https://github.com/pyvista/pyvista/blob/main/CODE_OF_CONDUCT.md>`__

Contributing to pyvista through GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To submit new code to pyvista, first fork the `pyvista GitHub
Repo <https://github.com/pyvista/pyvista>`__ and then clone the forked
repository to your computer. Then, create a new branch based on the
`Branch Naming Conventions Section <#branch-naming-conventions>`__ in
your local repository.

Next, add your new feature and commit it locally. Be sure to commit
often as it is often helpful to revert to past commits, especially if
your change is complex. Also, be sure to test often. See the `Testing
Section <#testing>`__ below for automating testing.

When you are ready to submit your code, create a pull request by
following the steps in the `Creating a New Pull Request
section <#creating-a-new-pull-request>`__.

Coding Style
^^^^^^^^^^^^

We adhere to `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`__
wherever possible, except that line widths are permitted to go beyond 79
characters to a max of 90 to 100 characters.

Outside of PEP 8, when coding please consider `PEP 20 -- The Zen of
Python <https://www.python.org/dev/peps/pep-0020/>`__. When in doubt:

.. code:: python

    import this

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
``requirements_docs.txt``, ``requirements_style.txt``:

.. code:: bash

    pip install -r requirements_test.txt
    pip install -r requirements_docs.txt
    pip install -r requirements_style.txt

Then, if you have everything installed, you can run the various test
suites.

Run the primary test suite and generate coverage report:

.. code:: bash

    python -m pytest -v --cov pyvista

Run all code examples in the docstrings:

.. code:: bash

    python -m pytest -v --doctest-modules pyvista

Run documentation testing by running

.. code:: bash

    make

If you are running windows and ``make`` is unavailable, then run:

::

    pydocstyle pyvista

    codespell pyvista/ examples/ tests/ -S "*.pyc,*.txt,*.gif,*.png,*.jpg,*.ply,*.vtk,*.vti,*.js,*.html,*.doctree,*.ttf,*.woff,*.woff2,*.eot,*.mp4,*.inv,*.pickle,*.ipynb,flycheck*" -I "ignore_words.txt"

And finally, test the documentation examples:

.. code:: bash

    cd doc
    make clean
    make doctest
    make html -b linkcheck

The finished documentation can be found in the ``doc/_build/html``
directory.

Notes Regarding Image Regression Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since ``pyvista`` is primarily a plotting module, it's imperative we
actually check the images that we generate in some sort of regression
testing.  In practice, this ends up being quite a bit of work because:

- OpenGL software vs. hardware rending causes slightly different
  images to be rendered.
- We want our CI (which uses a virtual frame buffer) to match our
  desktop images (uses hardware acceleration).
- Different OSes render different images.

As each platform and environment renders different slightly images
relative to Linux (which these images were built from), so running
these tests across all OSes isn't optimal.  We could generate
different images for each OS, but it's overkill in my opinion; we need
to know if something fundamental changed with our plotting without
actually looking at the plots (like the docs at dev.pyvista.com)

Based on these points, image regression testing only occurs on Linux
CI, and multi-sampling is disabled as that seems to be one of the
biggest difference between software and hardware based rendering.

Image cache is stored here as ./image_cache

Image resolution is kept low at 400x400 as we don't want to pollute
git with large images.  Small variations between versions and
environments are to be expected, so error < ``IMAGE_REGRESSION_ERROR``
is allowable (and will be logged as a warning) while values over that
amount will trigger an error.

There are two mechanisms within ``pytest`` to control image regression
testing, ``--reset_image_cache`` and ``--ignore_image_cache``.  For
example:

.. code:: bash

    pytest tests/plotting --reset_image_cache

Running ``--reset_image_cache`` creates a new image for each test in
``tests/plotting/test_plotting.py`` and is not recommended except for
testing or for potentially a major or minor release.  You can use
``--ignore_image_cache`` if you're running on Linux and want to
temporarily ignore regression testing.  Realize that regression
testing will still occur on our CI testing.

If you need to add a new test to ``tests/plotting/test_plotting.py``
and wish to include image regression testing, be sure to add
``verify_cache_image`` to ``show``.  For example:

.. code:: python

    @skip_no_plotting
    def test_add_background_image_not_global():
        plotter = pyvista.Plotter()
        plotter.add_mesh(sphere)
        plotter.show(before_close_callback=verify_cache_image)

This ensures that immediately before the plotter is closed, the
current render window will be verified against the image in CI.  If no
image exists, be sure to add the resulting image with ``git add
tests/plotting/image_cache/*``.


Creating a New Pull Request
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have tested your branch locally, create a pull request on
`pyvista GitHub <https://github.com/pyvista/pyvista>`__ while merging to
main. This will automatically run continuous integration (CI) testing
and verify your changes will work across several platforms.

To ensure someone else reviews your code, at least one other member of
the pyvista contributors group must review and verify your code meets
our community's standards. Once approved, if you have write permission
you may merge the branch. If you don't have write permission, the
reviewer or someone else with write permission will merge the branch and
delete the PR branch.

Since it may be necessary to merge your branch with the current release
branch (see below), please do not delete your branch if it is a ``fix/``
branch.

Branching Model
~~~~~~~~~~~~~~~

This project has a branching model that enables rapid development of
features without sacrificing stability, and closely follows the `Trunk
Based Development <https://trunkbaseddevelopment.com/>`__ approach.

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
   branches will have their ``__version__.py`` updated and be tagged
   with a patched semantic version (e.g. ``0.24.1``). This triggers CI
   to push to PyPI, and allow us to rapidly push hotfixes for past
   versions of ``pyvista`` without having to worry about untested
   features.
-  When a minor release candidate is ready, a new ``release`` branch
   will be created from ``main`` with the next incremented minor
   version (e.g. ``release/0.25``), which will be thoroughly tested.
   When deemed stable, the release branch will be tagged with the
   version (``0.25.0`` in this case), and if necessary merged with
   main if any changes were pushed to it. Feature development then
   continues on ``main`` and any hotfixes will now be merged with this
   release. Older release branches should not be deleted so they can be
   patched as needed.

Minor Release Steps
^^^^^^^^^^^^^^^^^^^

Minor releases are feature and bug releases that improve the
functionality and stability of ``pyvista``. Before a minor release is
created the following will occur:

1.  Create a new branch from the ``main`` branch with name
    ``release/MAJOR.MINOR`` (e.g. ``release/0.25``).

2.  Locally run all tests as outlined in the `Testing
    Section <#testing>`__ and ensure all are passing.

3.  Locally test and build the documentation with link checking to make
    sure no links are outdated. Be sure to run ``make clean`` to ensure
    no results are cached.
    ``bash     cd doc     make clean  # deletes the sphinx-gallery cache     make doctest     make html -b linkcheck``

4.  After building the documentation, open the local build and examine
    the examples gallery for any obvious issues.

5.  Update the version numbers in ``pyvista/_version.py`` and commit it.
    Push the branch to GitHub and create a new PR for this release that
    merges it to main. Development to main should be limited at this
    point while effort is focused on the release.

6.  It is now the responsibility of the ``pyvista`` community to
    functionally test the new release. It is best to locally install
    this branch and use it in production. Any bugs identified should
    have their hotfixes pushed to this release branch.

7.  When the branch is deemed as stable for public release, the PR will
    be merged to main and the ``main`` branch will be tagged with a
    ``MAJOR.MINOR.0`` release. The release branch will not be deleted.
    Tag the release with:

    .. code:: bash

        git tag MAJOR.MINOR.0
        git push origin --tags

8.  Create a list of all changes for the release. It is often helpful to
    leverage `GitHub's *compare*
    feature <https://github.com/pyvista/pyvista/compare>`__ to see the
    differences from the last tag and the ``main`` branch. Be sure to
    acknowledge new contributors by their GitHub username and place
    mentions where appropriate if a specific contributor is to thank for
    a new feature.

9.  Place your release notes from step 8 in the description for `the new
    release on
    GitHub <https://github.com/pyvista/pyvista/releases/new>`__

10. Go grab a beer/coffee/water and wait for
    [@regro-cf-autotick-bot](https://github.com/regro-cf-autotick-bot)
    to open a pull request on the conda-forge `PyVista
    feedstock <https://github.com/conda-forge/pyvista-feedstock>`__.
    Merge that pull request.

11. Announce the new release in the PyVista Slack workspace and
    celebrate!

Patch Release Steps
^^^^^^^^^^^^^^^^^^^

Patch releases are for critical and important bugfixes that can not or
should not wait until a minor release. The steps for a patch release

1. Push the necessary bugfix(es) to the applicable release branch. This
   will generally be the latest release branch (e.g. ``release/0.25``).

2. Update ``__version__.py`` with the next patch increment (e.g.
   ``0.25.1``), commit it, and open a PR that merge with the release
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

