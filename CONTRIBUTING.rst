Contributing
============

.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-3.0-4baaaa.svg
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

Quick Start Development With Codespaces
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

Alternatively, an offscreen version using OSMesa libraries with VTK 9.5+ is available.

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

Generative AI
-------------

We follow the Python Developer's Guide on `AI tools <https://devguide.python.org/getting-started/ai-tools/>`_,
with one difference: disclosure is required here, where the guide only appreciates it.
The resulting contribution is the responsibility of the contributor, and we value good code,
concise accurate documentation, and avoiding unneeded code churn.

If an AI tool wrote any part of a pull request -- code, tests, documentation, or the
description itself -- say so in the description. Write that sentence yourself: it states
that you reviewed the change and can explain it, which no tool can attest to on your
behalf. One clause naming the tool and what it did is enough, in the form merged
descriptions already use, for example ``Changes drafted by Claude Opus 5 but fully understood by me``.

That responsibility covers what the contribution costs us to review and to test.
If you work with a coding agent, point it at ``AGENTS.md`` in the repository root,
which routes to the task guides and repeats the rules agents get wrong most often,
and read `Continuous Integration Etiquette`_ before it pushes anything.

--------------

Development Practices
---------------------

This section provides a guide to how we conduct development in the
PyVista repository. Please follow the practices outlined here when
contributing directly to this repository.

Quick Development Commands
~~~~~~~~~~~~~~~~~~~~~~~~~~

For convenience, the most common developer tasks are wrapped as ``make``
targets in the repository's top-level ``Makefile``. These are the
recommended entry points for day-to-day development.

Most targets delegate to ``uv``, so ``uv`` must be installed on your
system first (see https://docs.astral.sh/uv/getting-started/installation/).
``make`` itself must also be available on your ``PATH``; on Windows it
can be installed via package managers like ``scoop`` or ``chocolatey``.

.. code-block:: bash

    make sync-deps         # install dev dependencies via uv (includes tox + tox-uv)
    make lint              # run pre-commit on all files
    make docstyle          # run Vale (matches CI)
    make typecheck         # run mypy via tox (matches CI)
    make test              # run the full test suite via tox (matches CI flags)
    make test-core         # run the core test suite via tox (matches CI)
    make test-plotting     # run the plotting test suite via tox (matches CI)
    make doctest           # run all docstring tests via tox (matches CI)
    make docs              # build the full documentation via tox (matches CI)
    make docs-test-build   # sanity-check the built documentation via tox (matches CI)
    make docs-test-images  # compare documentation images against cached baselines via tox (matches CI)
    make integration PROJECT=<name>  # run integration tests for trame/geovista/mne/pyvistaqt/playwright/cvista

``make test``, ``make test-core``, and ``make test-plotting`` all
invoke tox environments defined in ``tox.ini`` so they run with the
exact same pytest filters and flags as the corresponding CI jobs. The
filter definitions live in ``tox.ini`` so they only need to be
maintained in one place.

Running ``make`` with no target is equivalent to ``make test``.

Additional arguments can be forwarded to ``pytest`` via the ``ARGS``
variable, for example:

.. code-block:: bash

    make test ARGS="-n 10"               # run tests in parallel with 10 workers
    make test ARGS="-k filters"          # only run tests matching "filters"
    make test-core ARGS="-n auto -x"     # core tests, auto parallelism, stop on first failure

These targets are thin wrappers around ``uv``, ``pre-commit``, ``tox``,
and ``pytest``. If you need more control (for example, running against a
specific ``vtk`` or ``numpy`` version, or building documentation), see
the `Unit Testing`_, `Docstring Testing`_, `Type Checking`_, `Style
Checking`_, and `Building the Documentation`_ sections below, which
document the underlying tools directly.

Continuous Integration Etiquette
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Opening a pull request, and every push to it afterwards, starts the full
continuous integration suite: unit tests on Linux, macOS, and Windows across
every supported Python version, a separate VTK version matrix, the
documentation build, the integration tests, type checking, and the style and
docstring jobs. Every one of those runs costs the project paid runner time.
Push when the change is ready, and use the local gates rather than CI to find
out whether it works.

Before you push:

#. Run ``make lint``. It runs ``pre-commit`` on all files and catches most of
   what the style jobs would report.
#. Run the tests covering what you changed, for example
   ``make test-core ARGS="-k threshold"`` or ``make test-plotting``. These
   targets use the same ``tox`` environments and ``pytest`` flags as CI, so a
   green local run means the same run is green in CI.
#. Run ``make typecheck`` when you change type annotations, and ``make
   doctest`` when you change a docstring example.

While you iterate, keep the pull request in draft and amend or squash locally,
so a single push carries the finished change instead of five pushes carrying a
debugging session. A new push cancels the runs still in progress for the same
branch, but it cannot refund a run that already finished.

When a job fails:

-  Read the log and find the failing assertion before changing anything. The
   job name alone does not identify the cause.
-  Reproduce it locally with the matching ``make`` target or ``tox``
   environment. When the failure needs a Python or VTK version you cannot run,
   say so in the pull request instead of pushing speculative fixes.
-  Check whether the same failure occurs on ``main``. If it does, the failure
   is not yours to chase in this pull request.
-  Diagnose a flaky test rather than re-running the job until it passes. A
   re-run costs what the first run cost, and a test that only passes on the
   second attempt is worth reporting.
-  For image regression failures, download the failed image artifact from the
   job and compare it against the committed baseline (see `Notes Regarding
   Image Regression Testing`_) instead of pushing baseline updates to see
   which ones stick.

This applies with particular force to contributions made with coding agents,
which can run every gate above locally in a shell. Configure yours to do that
and review its work before it reaches CI. Whoever opens the pull request is
responsible for what each push costs the project, in the same way they are
responsible for the content of the contribution.

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

Contributing to PyVista Through GitHub
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

We adhere to `PEP 8 <https://peps.python.org/pep-0008/>`_
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
Python <https://peps.python.org/pep-0020/>`_. When in doubt:

.. code-block:: python

    import this

PyVista uses `pre-commit`_ to enforce PEP8 and other styles
automatically. Please see the `Style Checking section <#style-checking>`_ for
further details.

Import Conventions
^^^^^^^^^^^^^^^^^^

Standard library imports follow one rule: **import the name that carries its
own meaning at the call site.**

Modules that export *types* are imported by member. "Type" here means a name
that appears in a type position -- an annotation, a base class, or a
class-defining decorator -- where the module prefix is pure noise:

.. code-block:: python

    from pathlib import Path
    from collections.abc import Sequence
    from dataclasses import dataclass


    @dataclass
    class Config:
        path: Path
        names: Sequence[str]

Everything else keeps the namespace prefix, because the module name supplies
the domain that makes the call readable:

.. code-block:: python

    import functools
    import re

    pattern = re.escape(text)  # not `escape` -- shell? HTML? regex?


    @functools.wraps(func)  # not `wraps` -- wraps what?
    def wrapper(*args, **kwargs): ...


Some member imports also shadow their own module (``from time import time``,
``from glob import glob``), which the namespace form avoids.

The unit is the module, not the name. ``argparse`` exports ``ArgumentParser``
but is namespace-imported, because one type does not make a type module;
``argparse.ArgumentParser`` reads fine. The member list is closed and short:
``__future__``, ``abc``, ``collections``, ``collections.abc``, ``concurrent.futures``,
``dataclasses``, ``enum``, ``http.server``, ``importlib.metadata``, ``io``, ``pathlib``,
``types``, ``typing``, ``typing_extensions``, ``unittest.mock``.

How This Is Enforced
""""""""""""""""""""

Two lists, because ``ruff`` can only express one direction:

* ``banned-from`` under ``[tool.ruff.lint.flake8-import-conventions]`` in
  ``pyproject.toml`` rejects ``from re import escape`` (``ICN003``).
* the ``namespace-stdlib-imports`` pygrep hook in ``.pre-commit-config.yaml``
  rejects ``import pathlib``. Ruff cannot do this direction --
  ``flake8-tidy-imports``' ``banned-api`` matches the resolved symbol, so
  banning ``pathlib`` would reject ``from pathlib import Path`` too.

``tests/test_import_conventions.py`` asserts the lists stay disjoint and
jointly govern every standard library module the repository imports, so a
module governed by neither fails CI instead of settling into whichever form its
first author picked. When it fails, add the module to the matching list.

Two details:

* ``banned-from`` does not match submodules -- banning ``importlib`` still
  permits ``from importlib.metadata import entry_points``. Govern the submodule
  explicitly when it is used directly.
* Aliased imports (``import xml.dom.minidom as md``) are an intentional escape
  hatch and neither list matches them.

Prefer fixing the code over adding a waiver: a local variable shadowing a
module is a reason to rename the variable. The sole exception in the tree is
``contextlib.AbstractContextManager``, a type in a base-class position inside
an otherwise action-shaped module, carrying a ``# noqa: ICN003``.

Documentation Style
^^^^^^^^^^^^^^^^^^^

PyVista follows the `Google Developer Documentation Style
<https://developers.google.com/style>`_ with the following exceptions:

- Allow `first person pronouns
  <https://developers.google.com/style/pronouns#personal-pronouns>`_. These
  pronouns (for example, "We") refer to "PyVista Developers", which can be
  anyone who contributes to PyVista.
- Future tense is permitted.
- Always place commas and periods outside closing `quotation marks
  <https://developers.google.com/style/quotation-marks>`_, rather than
  Google's prose-vs-literal-string distinction, which a linter cannot
  reliably apply.

These rules are enforced for all text files (for example, ``*.md``, ``*.rst``)
and partially enforced for Python source files.

Every rule in ``doc/styles/Google/`` links to the specific Google style page it
enforces (each file's ``link:`` field). There is no warning-only tier: CI fails
on anything Vale reports, at any level, so a rule that only warns is still a
required fix, not a suggestion to skip. Four that come up often in review, with
their Google pages:

- `Capitalization in titles and headings
  <https://developers.google.com/style/capitalization#capitalization-in-titles-and-headings>`_
- `Commas <https://developers.google.com/style/commas>`_ (the Oxford comma)
- `Abbreviations <https://developers.google.com/style/abbreviations>`_
  (``e.g.``/``i.e.`` -> "for example"/"that is")
- `Plurals in parentheses
  <https://developers.google.com/style/plurals-parentheses>`_ (``word(s)`` ->
  "words")

These rules are enforced through the use of `Vale <https://vale.sh/>`_ via our
GitHub Actions, and you can run Vale locally with:

.. code-block:: bash

   pip install vale 'docutils<0.22' 'sphinx-gallery<0.22.0'
   python3 doc/run_vale.py

If you are on Linux or macOS, ``make docstyle`` runs the same script.

``doc/run_vale.py`` extracts the ``.rst`` files described below, runs Vale over
every path CI checks, and then confirms that the rule still rejects the
headings in ``tests/doc/vale/headings_invalid.rst``. The path list lives in
that script alone; the workflow reads it with ``--print-files``.

Vale cannot parse prose written inside a Python file directly (for example,
the ``# %%`` cell headings in a gallery example, or a docstring's
``Parameters`` section), so ``doc/extract_rst_from_py_for_vale.py`` first
converts the relevant ``.py`` files into ``.rst`` files that mirror them line
for line, padding out everything else with blank lines. Vale only ever sees
those generated files, so it reports errors against them instead of the
original source -- look for the same path and line number under
``.vale/examples/`` or ``.vale/pyvista/`` instead of ``examples/`` or
``pyvista/``, with the ``.py`` extension swapped for ``.rst``. For example:

- A gallery heading error reported as
  ``.vale/examples/02-plot/point_picking.rst:31:1`` refers to
  ``examples/02-plot/point_picking.py:31:1``.
- A docstring error reported as
  ``.vale/pyvista/core/pointset.rst:109:1`` refers to
  ``pyvista/core/pointset.py:109:1`` (the ``Flag for using the mesh scalars as
  weights.`` line of ``PointSet.center_of_mass``'s docstring).

``doc/styles/config/vocabularies/pyvista/accept.txt`` is a spelling-vocabulary
waiver list, not a style waiver list: a word belongs there only if it is a
legitimate technical term, proper noun, or acronym that Vale's dictionary
does not know, and there is no fix that would make the waiver unnecessary.
Prefer, in order:

1. **Reword.** A Latin abbreviation, an awkward compound, or a non-Oxford
   list is a text problem, not a vocabulary problem -- fix the sentence.
2. **Hyphenate or split.** ``down-sample``, ``in-place``, ``de-registration``,
   ``file path`` are two recognizable words, not one unrecognized one.
   Check for an existing convention first (``grep`` the word without the
   hyphen); a prior commit may have already settled it, and re-litigating it
   by re-accepting the joined form is itself the mistake to avoid.
3. **Backtick it as code** if it actually is: a parameter, attribute, class,
   or module name. If the value is a string literal a parameter accepts
   (``mode='cell_tree'``), keep the quotes inside the backticks when writing
   it up -- ``'cell_tree'`` (quotes and all), not just ``cell_tree`` -- or
   the rendered text stops looking like a string. Suffixing a plain letter
   directly onto a backtick-wrapped term (writing the plural of ``int`` as
   code, immediately followed by an ``s``) needs an escaped space between
   the closing backticks and the suffix, and the docstring needs an ``r``
   prefix for that escape to survive -- otherwise ``docutils`` raises "Inline
   literal start-string without end-string", since its inline-markup rules
   require whitespace or punctuation immediately after a closing pair of
   backticks, and a bare backslash in a non-raw Python string is itself an
   invalid escape sequence.
4. **Only then accept it**: ``colormap``, ``cubemap``, ``framerate`` are
   established one-word technical terms with no better spelling. A
   dual-cased pair (``PyVista``/``pyvista``, ``VTK``/``vtk``, ``NumPy``/
   ``numpy``) is not a vocabulary problem either -- both spellings are
   already known words. Use the capitalized form when naming the project or
   library in prose, and the lowercase form only where it is literally code
   (an import, a module path, a parameter default); this split is not
   machine-checked (``Vale.Terms`` is disabled -- see the comment in
   ``doc/.vale.ini`` for why), so it needs a human read.

A docstring should not describe a parameter or property by repeating its own
name in backticks, for example
``"""Return or set the \`\`tube_width\`\`."""``. Describe it in plain English
(``"""Return or set the tube width."""``); backticks are for naming a
*different* real identifier, not the thing whose docstring this is.


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
* The returns section structure depends on the number of return values and types:
    * for a single return value with a single return type, the parameter name
      can be omitted (as shown above),
    * for a single return value with multiple types (that is, ``str | int``), the parameter
      must be specified (not shown),
    * for multiple return values (not shown), descriptive parameter names for each returned value
      must be specified in the same format as the input parameters.
* The examples section references the "full example" in the gallery if it
  exists.

In addition, docstring examples which make use of randomly generated data
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


Deprecating Features or Other Backwards-Breaking Changes
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
the ``PyVistaDeprecationWarning`` warning, the ``.. deprecated`` Sphinx
directive and the ``warn_external`` helper function.

.. code-block:: python

    from pyvista.core.errors import PyVistaDeprecationWarning
    from pyvista._warn_external import warn_external  # available from 0.47


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
        warn_external(
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

A prefix is unusable on a remote that already has a branch named exactly that, since
Git cannot store a ref as both a file and a directory: pushing ``doc/my-change`` to a
fork that still has an old ``doc`` branch is rejected with ``directory file conflict``.
Check the remote you push to with ``git ls-remote --heads origin refs/heads/doc``, then
either use another prefix or delete the stale branch.

Points dtype
^^^^^^^^^^^^

The ``points`` dtype of a filter's output is decided globally, by
``pyvista.global_config.points_dtype``, and enforced in ``_update_alg`` and
``_get_output``. A filter gets this for free by calling those two, and must not
add a keyword of its own for precision.

- Call ``_update_alg`` rather than ``alg.Update()``. It asks the algorithm for the
  configured precision before updating, so algorithms that support
  ``SetOutputPointsPrecision`` compute in that precision rather than being cast after
  the fact.
- Call ``_get_output`` rather than wrapping ``alg.GetOutput()``. It casts the output
  points for the algorithms that ignore the request.
- Sources have no input to preserve, so they subclass ``_Source``, which requests the
  precision in ``Update`` and casts in ``_update_and_wrap_output``. Return
  ``self._update_and_wrap_output()`` from a source's ``output`` property rather than
  wrapping ``GetOutput()``, which is uncast.
- Geometry that PyVista builds without a VTK algorithm passes through
  ``_apply_points_dtype``.
- Neither helper needs to know whether the algorithm supports double precision. The
  ones that do not are cast, and warn with ``PyVistaPrecisionWarning`` when the user
  asked for ``'float64'`` -- so no filter needs a keyword to opt out of the setting.
- Under ``'preserve'`` only the meshes that store their points constrain the output.
  ``ImageData`` and ``RectilinearGrid`` generate theirs, so a filter reading one, or
  building one as an intermediate, leaves the precision to VTK.

Testing
^^^^^^^

After making changes, please test changes locally before creating a pull
request. The following tests will be executed after any commit or pull
request, so we ask that you perform the following sequence locally to
track down any new issues from your changes.

To run our comprehensive suite of unit tests, please refer to the `Unit Testing`_
section.

Unit Testing
~~~~~~~~~~~~
Unit testing can be run either directly using `pytest <https://docs.pytest.org/en/stable/>`_
or `tox <https://tox.wiki/en/stable/>`_ to ensure environment isolation and reproducibility with CI.
The top-level ``Makefile`` also wraps the most common invocations—see
`Quick Development Commands`_.

.. tab-set::
    :sync-group: category

    .. tab-item:: pytest
        :sync: pytest

        .. code-block:: bash

            pip install -e . --group=test # installing testing dependencies
            pytest # alternatively: python -m pytest


    .. tab-item:: tox
        :sync: tox

        .. code-block:: bash

            pip install tox
            tox run -e py3.11 # change to the python version targeted

        .. admonition:: tox usage
            :class: hint dropdown

            When using ``tox``, specific test environments can be used to test against various
            dependencies versions (mostly ``numpy`` and ``vtk``). The full list is available by running:

            .. code-block:: bash

                tox list

            For example, to run tests on ``python 3.11`` against the wheels produced by the ``vtk`` CI
            on the main branch, simply run:

            .. code-block:: bash

                tox run -e py3.11-vtk_dev

            Note that several dependencies versions are already predefined in the ``tox.ini`` configuration
            and can be specified with ``tox`` factors such that:

            .. code-block:: bash

                tox run -e py3.11-vtk_9.4.2 # run tests for vtk==9.4.2
                tox run -e py3.11-vtk_9.4.2-numpy_nightly # run tests for vtk==9.4.2 with nightly numpy

            If you need to tests dependencies that are not predefined in the configuration, you can always override them such
            that:

            .. code-block:: bash

                tox run -e py3.11 --override testenv.deps+=vtk==9.4.2 # run tests for vtk==9.4.2
                tox run -e py3.11 --override testenv.deps+=vtk==9.4.2 --override testenv.deps+=numpy==2.0 # run tests for vtk==9.4.2 and numpy==2.0

            By default, all tests (that is, plotting and core modules) are executed if nothing is specified.
            To only run core or plotting tests, add ``core`` or ``plotting`` factors to the environment name such that:

            .. code-block:: bash

                tox run -e py3.11-core # run core tests (no need for graphics library)
                tox run -e py3.11-plotting # run plotting tests (requires graphics library)
                tox run -e py3.11-core-plotting # equivalent to 'tox run -e py3.11'

            To specify supplementary arguments to the ``pytest`` command line, use ``--`` to separate
            ``tox`` arguments from ``pytest`` ones such that:

            .. code-block:: bash

                tox run -e py3.11 -- -k "filters" # run all tests whose name match `filters`
                tox run -e py3.11 -- -n 4 # run all tests in parallel with 4 processes

            For a more detailed description of ``tox`` usage, please refer to the following `cheat sheet <https://tox.wiki/en/stable/user_guide.html#cheat-sheet>`_.

    .. tab-item:: make
        :sync: make

        .. code-block:: bash

            make sync-deps # install dev dependencies via uv
            make test      # run the full test suite (equivalent to `tox -e test`)

Unit testing can take some time, if you wish to speed it up, set the
number of processors with the ``-n`` flag. This uses ``pytest-xdist`` to
leverage multiple processes. Example usage:

.. tab-set::
    :sync-group: category

    .. tab-item:: pytest
        :sync: pytest

        .. code-block:: bash

            pytest -n <NUMCORE>

    .. tab-item:: tox
        :sync: tox

        .. code-block:: bash

            tox run -e py3.11 -- -n <NUMCORE>

    .. tab-item:: make
        :sync: make

        .. code-block:: bash

            make test ARGS="-n <NUMCORE>"

Code coverage (that is, the amount of tested code in the codebase) can be measured by modifying the previous commands
such that:

.. tab-set::
    :sync-group: category

    .. tab-item:: pytest
        :sync: pytest

        .. code-block:: bash

            pytest --cov pyvista

    .. tab-item:: tox
        :sync: tox

        .. code-block:: bash

            tox run -e py3.11-cov

        .. note::

            The ``-cov`` factor can be added to any existing environment to enable test coverage, such that:

            .. code-block:: bash

                tox run -e py3.10-numpy_1.23-vtk_9.4.2-cov
                tox run -e py3.13-vtk_dev-cov # to test with coverage against the wheels produced by the VTK CI on the main branch

    .. tab-item:: make
        :sync: make

        .. code-block:: bash

            make coverage # pytest -v --cov pyvista
            make coverage-html # same, with an HTML report at ./htmlcov

When submitting a PR, it is highly recommended that all modifications are thoroughly tested.
This is further enforced in the CI by the `codecov GitHub action <https://app.codecov.io/gh/pyvista/pyvista>`_
which has a 90% target, that is, it ensures that 90% of the code modified in the PR is tested.
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

Testing Against VTK Master
^^^^^^^^^^^^^^^^^^^^^^^^^^
Most unit testing is run against stable VTK releases. However, when developing features that depend on upstream VTK
changes or when investigating regressions, it can be useful to test against the latest VTK development code.
VTK publishes development wheels to the VTK wheels index, which are snapshots of recent development builds.
To install them locally, run:

.. code-block:: shell

    pip install vtk --upgrade --pre --extra-index-url https://wheels.vtk.org

For pull requests, applying the ``vtk-dev-testing`` label enables an additional CI job that installs these development
wheels and runs the unit test suite against them. Although these wheels are official VTK builds, they are only published
periodically (typically once per week) and may not include the latest commits from the VTK repository. As a result,
passing ``vtk-dev-testing`` does not guarantee compatibility with the current VTK master branch.

To test against the very latest upstream VTK source, apply the ``vtk-master-testing`` label instead. This enables a CI
job that clones the VTK repository, builds VTK directly from the current master branch, and runs the unit tests against
that build. This provides the most up-to-date compatibility testing and is recommended when changes depend on recent VTK
development.

The ``vtk-dev-testing`` and ``vtk-master-testing`` labels are independent and may be applied separately or together.

.. note::

    The PR either needs a new commit, for example updating the branch from ``main``, or to be
    closed/re-opened to rerun the CI with the label applied.

Testing Against the ``cvista`` Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyVista also runs against `cvista <https://github.com/pyvista/cvista>`_, a community fork of VTK. Stock VTK is the
default and is tested on every PR; cvista is tested at **integration cadence**—nightly, and on PRs carrying the
``integration-testing`` label:

.. code-block:: shell

    make integration PROJECT=cvista

It is not on the per-PR fast path because it is a full extra run of the suite against a second VTK build, and because a
failure there is rarely a reason to block an unrelated PR.

**When the suites disagree, fix cvista.** PyVista is not held back by the fork: if a change here is correct against
stock VTK but fails on cvista, the fix belongs upstream in cvista, not in a marker or an ignore list here. Open an issue
on `pyvista/cvista <https://github.com/pyvista/cvista/issues>`_ and keep going.

The ``skip_vtk_backend`` marker is only for **permanent, by-design** divergence—a module the fork does not build, or
behaviour that differs deliberately. Attach it to the test with a reason naming the specific cause:

.. code-block:: python

    @pytest.mark.skip_vtk_backend(
        'cvista',
        reason='cvista does not ship vtkIOParallel (vtkPOpenFOAMReader)',
    )
    def test_openfoam_patch_arrays(): ...

It is not a way to park a real regression. In library code, use :func:`pyvista.vtk_backend` to raise a clear error for a
build that cannot support a feature.

Garbage Collection Checks
^^^^^^^^^^^^^^^^^^^^^^^^^
Every test is checked for reference leaks: no plotter or VTK object created by a test
may outlive it. The autouse ``check_gc`` fixture in ``tests/conftest.py`` covers the
whole repository, and ``tests/plotting/conftest.py`` overrides it with a version that
also watches plotters. A leaking test fails at teardown with a rendered chain of what
still holds a reference; see the
`refleak <https://github.com/mne-tools/refleak>`_ documentation for how to read it.

The check freezes the heap rather than scanning it, so it costs no measurable time and
every CI job runs it. ``--no_check_gc`` turns it off for a whole run, for local
iteration where the report is in the way:

.. code-block:: bash

    tox run -e test-plotting-no_check_gc

The cause of a leak is usually a reference cycle, and fixing it (for example, with
:mod:`weakref`) is preferred over silencing the check with either of these markers:

.. code-block:: python

    @pytest.mark.skip_check_gc
    def test():
        """Do not check this test for leaks.

        Use sparingly, with a comment saying why the leak is not fixable here,
        for example an upstream VTK issue or a module-level cache pinning the object.
        """


    @pytest.mark.expect_check_gc_fail
    def test():
        """This test is expected to leak; fail if it does *not*."""

``expect_check_gc_fail`` outranks ``--no_check_gc``, so the tests that exercise the
check itself (``tests/core/test_gc.py``, ``tests/plotting/test_gc.py``) cannot pass
while nothing is running.

Docstring Testing
~~~~~~~~~~~~~~~~~
Run all code examples in the docstrings with:

.. tab-set::
    :sync-group: category

    .. tab-item:: pytest
        :sync: pytest

        .. code-block:: bash

            pytest -v --doctest-modules pyvista

    .. tab-item:: tox
        :sync: tox

        .. code-block:: bash

            tox run -e doctest-modules

    .. tab-item:: make
        :sync: make

        .. code-block:: bash

            make doctest

        .. note::

            ``make doctest`` runs ``tox run -f doctest``, which matches CI
            (``.github/workflows/style-docstring.yml``) by running both the
            ``doctest-modules`` environment above and the ``doctest-local``
            environment. The latter has no ``pytest``/``tox -e`` equivalent
            shown above since it doesn't run through ``pytest``: it statically
            checks that names used in docstring examples are actually defined
            (see ``tests/check_doctest_names.py``).

.. note::

    Additional testing is also performed on any images generated
    by the docstring. See `Documentation Image Regression Testing`_.


Type Checking
~~~~~~~~~~~~~
PyVista uses `mypy <https://mypy.readthedocs.io/>`_ for static type checking. Configuration
lives in the ``[tool.mypy]`` section of ``pyproject.toml``, so no additional command-line
flags are required to run it.

.. tab-set::
    :sync-group: category

    .. tab-item:: mypy
        :sync: pytest

        .. code-block:: bash

            pip install -e . --group typing
            mypy

    .. tab-item:: tox
        :sync: tox

        .. code-block:: bash

            tox run -e mypy

    .. tab-item:: make
        :sync: make

        .. code-block:: bash

            make typecheck

.. seealso::

    `Notes Regarding Input Validation Testing`_ describes a related but separate
    ``pytest``-based suite that checks the type hints of ``pyvista.core._validation``
    using ``mypy`` and ``pyanalyze`` at both static-analysis and runtime.


Style Checking
~~~~~~~~~~~~~~
PyVista follows PEP8 standard as outlined in the `Coding Style section
<#coding-style>`_ and implements style checking using `pre-commit`_.

To ensure your code meets minimum code styling standards, run::

  pip install pre-commit
  pre-commit run --all-files

Alternatively, the top-level ``Makefile`` wraps this invocation::

  make lint

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

.. tab-set::
    :sync-group: category

    .. tab-item:: pytest
        :sync: pytest

        .. code-block:: bash

            pytest tests/plotting --reset_image_cache

    .. tab-item:: tox
        :sync: tox

        .. code-block:: bash

            tox run -e py3.11 -- tests/plotting --reset_image_cache

    .. tab-item:: make
        :sync: make

        .. code-block:: bash

            make test ARGS="tests/plotting --reset_image_cache"

Running ``--reset_image_cache`` regenerates the baseline of every test the
run collected, including tests that were failing for reasons unrelated to
your change. Reserve it for a deliberate regeneration of the whole cache at
a major or minor release, and use the scoped flags below for everything
else. ``--ignore_image_cache`` skips the comparison locally while you
iterate; regression testing still runs in our CI.

Two scoped flags cover day-to-day work. Give both of them a test node id so
they cannot touch a baseline you did not mean to change:

.. code-block:: bash

    # a new test that has no baseline yet
    make test-plotting ARGS="tests/plotting/test_plotting.py::test_new_render --add_missing_images"

    # a render that legitimately changed: overwrite only the failed baselines
    make test-plotting ARGS="tests/plotting/test_plotting.py::test_my_render --reset_only_failed"

Arguments passed through ``ARGS`` replace the environment's whole-suite defaults
rather than adding to them, so the node id above is the entire run and neither
the ``tests/plotting`` collection root nor ``--disallow_unused_cache`` comes
along with it. Plain ``pytest`` with the same arguments works too.

Look at every image before committing it. ``failed_image_dir`` and
``generated_image_dir`` are already configured in ``pyproject.toml``, so a
failing run writes ``_failed_test_images/`` with no extra flag: the committed
baseline under ``from_cache/`` and the new render under ``from_test/``. For a
run with many failures, build the HTML report:

.. code-block:: bash

    tox run -e image-report -- _failed_test_images _image_report

Since the baselines are the Linux CI renders, the most reliable way to accept
a change is to download the ``failed_test_images-*`` artifact from the failing
job and copy its ``from_test`` images over the cache, instead of re-rendering
on your own hardware.

Any test that requests the ``verify_image_cache`` fixture and calls
``Plotter.show`` (or ``mesh.plot``) caches and compares an image. The
comparison runs from a callback that ``show`` registers, so a test that builds
a plotter and only calls ``close`` compares nothing while still passing. To
skip image caching within a test, the ``verify_image_cache`` fixture can be
utilized:

.. code-block:: python

    def test_add_background_image_not_global(verify_image_cache):
        verify_image_cache.skip = True  # Turn off caching
        pl = pv.Plotter()
        pl.add_mesh(sphere)
        pl.show()
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

.. tab-set::
    :sync-group: category

    .. tab-item:: pytest
        :sync: pytest

        .. code-block:: bash

            pytest tests/plotting/ -k volume --generated_image_dir debug_images

    .. tab-item:: tox
        :sync: tox

        .. code-block:: bash

            tox run -e py3.11 -- tests/plotting/ -k volume --generated_image_dir debug_images

    .. tab-item:: make
        :sync: make

        .. code-block:: bash

            make test ARGS="tests/plotting/ -k volume --generated_image_dir debug_images"

See `pytest-pyvista`_ for more details.

.. note::

    Additional regression testing is also performed on the documentation
    images. See `Documentation Image Regression Testing`_.

Notes Regarding Input Validation Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``pyvista.core._validation`` package has two distinct test suites which
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

Notice that the revealed type is fully qualified, that is, it includes ``builtins``. For
brevity, the custom test suite omits this and requires that only ``list`` be
included in the expected type. Therefore, for this test case, the ``EXPECTED_TYPE``
type is ``"list[float]"``, not ``"builtins.list[builtins.float]"``. (Similarly, the
package name ``numpy`` should also be omitted for tests where a ``numpy.ndarray`` is
expected.)

Any number of related test cases (one test case per line) may be written and
included in a single ``.py`` file. The test cases are all stored in
``tests/core/typing/validation_cases``.

The tests can be executed with:

.. tab-set::
    :sync-group: category

    .. tab-item:: pytest
        :sync: pytest

        .. code-block:: bash

            pytest tests/core/typing

    .. tab-item:: tox
        :sync: tox

        .. code-block:: bash

            tox run -e py3.11 -- tests/core/typing

    .. tab-item:: make
        :sync: make

        .. code-block:: bash

            make test ARGS="tests/core/typing"


When executed, a single instance of ``Mypy`` will statically analyze all the
test cases. The actual revealed types by ``Mypy`` are compared against the
``EXPECTED_TYPE`` is defined by each test case.

In addition, the ``pyanalyze`` package tests the actual returned
type at runtime to match the statically revealed type. The
`pyanalyze.runtime.get_compatibility_error <https://pyanalyze.readthedocs.io/en/latest/reference/runtime.html#pyanalyze.runtime.get_compatibility_error>`_
method is used for this. If new typing test cases are added for a new
validation function, the new function must be added to the list of
imports in ``tests/core/typing/test_validation_typing.py`` so that the
runtime test can call the function.

Building the Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~
Documentation can be build either directly (that is, using Python commands) or with `tox <https://tox.wiki/en/stable/>`_ such that:

.. tab-set::
    :sync-group: category

    .. tab-item:: python
        :sync: pytest

        .. code-block:: bash

            python -m pip install -e . --group docs

        .. tab-set::

            .. tab-item:: Mac OS / Linux

                .. code-block:: bash

                    make -C doc html

            .. tab-item:: Windows

                .. code-block:: bash

                    cd doc
                    python -msphinx -M html source _build
                    python -msphinx -M html . _build

    .. tab-item:: tox
        :sync: tox

        .. code-block:: bash

            tox run -e docs-build

        .. note::
            By default, the ``html`` builder of sphinx is specified when running the ``docs-build``
            environment.
            You can customize it as a separate positional argument such that:

            .. code-block:: bash

                tox run -e docs-build -- mini18n-html # for translated languages

    .. tab-item:: make
        :sync: make

        .. code-block:: bash

            make sync-deps # install dev dependencies via uv
            make docs      # matches CI

The generated documentation can be found in the ``doc/_build/html``
directory.

The first time you build the documentation locally will take a while as all the
examples need to be built. After the first build, the documentation should take
a fraction of the time.

To test this locally you need to run a http server in the html directory with:

.. code-block:: bash

   make -C doc serve-html

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

To test all the images, run tests using either ``pytest`` or ``tox`` such that:

.. tab-set::
    :sync-group: category

    .. tab-item:: pytest
        :sync: pytest

        .. code-block:: bash

            pytest --doc_mode

    .. tab-item:: tox
        :sync: tox

        .. code-block:: bash

            tox run -e docs-test-images

    .. tab-item:: make
        :sync: make

        .. code-block:: bash

            make docs-test-images

Note that above commands use the ``doc-mode`` feature implemented in `pytest-pyvista`_.
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
    - The build image is copied to ``./_doc_debug_images_failed/errors/from_test``

#.  If an image is in the cache but missing from the build:

    - The cache image is copied to  ``./_doc_debug_images_failed/errors/from_cache``

#.  If an image is in the build but missing from the cache:

    - The build image is copied to  ``./_doc_debug_images_failed/errors/from_test``

If a warning is generated instead of an error, images are saved to the
``warnings`` sub-directory instead of ``errors``.

To resolve failed tests, any images in ``from_test`` or ``from_cache``
may be copied to or removed from the ``Doc Image Cache``. For example,
if adding new docstring examples or plots, the test will initially fail,
and the images in ``from_test`` may be added to the ``Doc Image Cache``.
Similarly, if removing examples, the images in ``from_cache`` may be removed
from the ``Doc Image Cache``.

If a test is flaky, for example the build sometimes generates different images
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

.. tab-set::
    :sync-group: category

    .. tab-item:: pytest
        :sync: pytest

        .. code-block:: bash

            pytest --doc_mode

    .. tab-item:: tox
        :sync: tox

        .. code-block:: bash

            tox run -e docs-test-images

    .. tab-item:: make
        :sync: make

        .. code-block:: bash

            make docs-test-images


Note that above commands use the ``doc-mode`` feature implemented in `pytest-pyvista`_
with the limit being specified by ``max_vtksz_file_size`` in the ``pyproject.toml`` file.

If any of these tests fail, the examples which generated the plot should be
modified, e.g.:

#. Simplify any datasets used, for example crop, clip, down-sample, decimate, or
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
of a full gallery example, add it to `pyvista/data <https://github.com/pyvista/data/>`_
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

Creating a New Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have tested your branch locally, create a pull request on
`pyvista GitHub <https://github.com/pyvista/pyvista>`_ while merging to
main. This will automatically run continuous integration (CI) testing
and verify your changes will work across several platforms. See
`Continuous Integration Etiquette`_ for what that run costs and how to
keep it to one.

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
is automatically deployed using `Netlify GitHub actions <https://github.com/nwtgck/actions-netlify>`_.
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

#.  Locally test and build the documentation. Be sure to run ``make -C doc clean``
    to ensure no results are cached. Run these commands from the repository
    root, using the ``make`` targets from `Quick Development Commands`_ so
    they match CI:

    .. code-block:: bash

       make -C doc clean  # deletes the sphinx-gallery cache
       make doctest       # matches CI
       make docs          # matches CI

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
    `@regro-cf-autotick-bot <https://github.com/conda-forge/conda-forge-bot>`_
    to open a pull request on the conda-forge `PyVista
    feedstock <https://github.com/conda-forge/pyvista-feedstock>`_.
    Merge that pull request.

#.  Announce the new release in the Discussions page and
    celebrate.

Patch Release Steps
^^^^^^^^^^^^^^^^^^^

Patch releases are for critical and important bug fixes that can not or
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

Dependency Version Policy
-------------------------

Python and VTK Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We support all supported `Python versions`_ and `VTK versions`_ that
support those Python versions. As much as we would prefer to follow
`SPEC 0`_, we follow VTK versions as an interface library of VTK.

.. _pre-commit: https://pre-commit.com/
.. _numpydoc Style Guide: https://numpydoc.readthedocs.io/en/latest/format.html
.. _Python versions: https://endoflife.date/python
.. _VTK versions: https://pypi.org/project/vtk/
.. _SPEC 0: https://scientific-python.org/specs/spec-0000/


Self-Hosted Runners
-------------------
GitHub hosted runners are the preferred way of running PyVista's CI. However
given the volume of development, the number of workflows, and the need to test
across several operating systems, it may be necessary to use self-hosted
runners due to GitHub's concurrency limits.

Any PyVista self-hosted runner must:

- Be as compatible as possible with a GitHub hosted runner.
- Use labels to denote the OS of a runner that are the same as GitHub's labels
  appended with ``self-hosted`` to ensure that there isn't overlap with GitHub
  labels.  For example, ``macos-15-self-hosted``. Additional labels may be
  specified (e.g. ``GPU``), but there must always be an OS label. Do not use a
  label that overlaps with GitHub's labels.
- Be secure against intrusion and follow best cybersecurity practices (for example, no
  ``sudo`` permissions, dedicated and isolated VLAN)
- Require a compatible CI/CD workflow.
- Provide runner documentation here.
- Be on a host with a battery backup.

GitHub Runner Workflow Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When setting up the GitHub workflow and using a ``matrix``, ensure that the
name of each job in the matrix is fixed rather than dependent on labels. This
way the `Branch Protection Rules
<https://github.com/pyvista/pyvista/settings/branches>`_ can use the same
status check label regardless of if it is self hosted.

.. code-block:: yaml

  macOS:
    name: ${{ matrix.job-name }}
    needs: cache-pyvista-data
    strategy:
      fail-fast: false
      matrix:
        include:
          # GitHub-hosted runner configuration
          - job-name: MacOS Unit Testing (Python 3.10)
            python-version: "3.10"
            runner-labels: "macos-15"
          # Self-hosted runner configurations
          - job-name: MacOS Unit Testing (Python 3.11)
            python-version: "3.11"
            runner-labels: "macos-15-self-hosted"

With this approach, a job can be configured to use GitHub's hosted runners simply
by changing ``"macos-15-self-hosted"`` to ``"macos-15"``.


Setting Up a Runner on Bare Metal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visit PyVista's `Create self-hosted runner
<https://github.com/organizations/pyvista/settings/actions/runners/new>`_.

Follow the directions to download, run, and install. If the runner is intended
to run public workflows, add the runner to the ``pyvista-self-hosted`` group.

Follow your OSes instructions to enable a service for the runner (if
applicable) to ensure the runner restarts should it be interrupted.

PyVista Hosts and Runners
~~~~~~~~~~~~~~~~~~~~~~~~~

Apple Silicon - 2024 Mac mini M4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
Testing showed peak memory usage of ~2 GB per runner for the
``testing-and-deployment.yml`` workflow. With 16GB of memory and ~4 GB used by
the OS, there's room to spare. Should we encounter memory issues we can disable
runners.


Linux Runners
^^^^^^^^^^^^^
PyVista uses a high availability Linux cluster running [k3s](https://k3s.io/) and deployed
using [Ansible](https://docs.ansible.com/). See
[pyvista/arc-runners](https://github.com/pyvista/arc-runners) for more details.

GPU enabled runs should use the ``ubuntu-24.04-self-hosted-gpu`` labels. Runners
using this label will receive a minimum of 2 CPUs and at maximum 8 CPUs along
with access to either an NVIDIA Quadro P2000 or a NVIDIA T400 (4GB VRAM).
