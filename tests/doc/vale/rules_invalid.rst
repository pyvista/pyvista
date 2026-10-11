Every Case Here Must Fail
=========================

This file is a negative fixture for the rules ``doc/.vale.ini`` raises to
``error``. It lives outside the paths Vale scans -- see ``PATHS`` in
``doc/run_vale.py`` -- for the same reason as ``headings_invalid.rst``.
``check_expected_failures.py``, beside it, runs Vale over this file and fails
if the line after each ``expect`` comment is not flagged by the rule it names.

.. expect: Google.OxfordComma

Clip, slice and plot the mesh.

.. expect: Google.Ellipses

Wait for the render to finish...

.. expect: Google.FirstPerson

Here I plot the mesh.

.. expect: Google.Ranges

Pick from 1-10 points.

.. expect: Google.Spacing

Plot the mesh.  Then close it.

.. expect: Google.Spacing

Plot the mesh.Then close it.

.. expect: Google.Spacing

Plot the mesh (once).  Then close it.

.. expect: Google.HeadingPunctuation

Plot the mesh.
--------------
