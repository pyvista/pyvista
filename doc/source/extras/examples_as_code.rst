.. _examples_as_code_docs:

Sphinx Examples-as-Code
=======================

.. versionadded:: 0.49

PyVista's documentation uses the Sphinx extension
https://github.com/pyvista/sphinx-examples-as-code to turn every docstring's
"Examples" section into a downloadable, runnable ``.py`` and ``.ipynb`` file.

This extension automatically inserts a download link at each "Examples"
section it finds, converting the section's content into standalone,
runnable source -- no role or directive needed in the docstring itself.

See `sphinx-examples-as-code's repository
<https://github.com/pyvista/sphinx-examples-as-code>`_ for installation and
usage details for adding it to your own project.
