Every Case Here Must Fail
=========================

This file is the negative half of the ``PyVista.Repetition`` fixture; the
positive half is ``tests/doc/vale/repetition.rst``. It lives outside the paths
Vale scans -- see ``PATHS`` in ``doc/run_vale.py`` -- because prose in a scanned
file has to pass, which makes it impossible to assert that the rule catches
anything. ``check_expected_failures.py``, beside it, runs Vale over this file
and fails if any bullet below is *not* flagged.

The rule is case-sensitive, so both halves of each doubled word are written
with the same casing. A word in ``accept.txt`` is exempt from the rule
altogether, so none of these are vocabulary terms.

- A doubled word mid mid sentence.
- A doubled word ending a sentence here here.
- A doubled capitalized name Sphere Sphere in one sentence.
