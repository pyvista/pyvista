Every Heading Here Must Fail
============================

This file is the negative half of the ``Google.Headings`` fixture; the positive
half is ``tests/doc/vale/headings.rst``. It lives outside the paths Vale scans -- see
``PATHS`` in ``doc/run_vale.py`` -- because a heading in a scanned file has to
pass, which makes it impossible to assert that the rule catches anything. ``check_expected_failures.py``, beside it, runs Vale over this
file and fails if any heading below is *not* flagged.

Clip with Plane
---------------

Textures from Files
-------------------

Distance between Two Surfaces
-----------------------------

What is a Mesh?
---------------

Modify Which Actors are Pickable
--------------------------------

Confirm that the Capped Result is Watertight
--------------------------------------------

List-like Features
------------------

Create Your Own Docker Container with PyVista
---------------------------------------------

the End of an Era
-----------------

a calibration of the camera
---------------------------

calibration camera isotope tofu
-------------------------------

pytesting the Waters
--------------------
