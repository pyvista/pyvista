Repetition Test Cases
=====================

Every line in this file must pass ``PyVista.Repetition``. The file is not part
of the documentation. It is passed to Vale explicitly -- see ``PATHS`` in
``doc/run_vale.py`` -- so a change to the rule that starts flagging one of these
shapes fails CI here rather than somewhere in the real documentation. Its
negative half is ``tests/doc/vale/repetition_invalid.rst``.

The rule's ``tokens`` pattern decides all of it; see
``doc/styles/PyVista/Repetition.yml``.

A dotted identifier is one token, so a module path may repeat a segment:
pyvista.plotting.plotting is a real module and reader.reader.read() is a real
call.

Punctuation stays attached to the token before it, so an enumeration may repeat
a word across commas: this includes lists, lists of tuples, tuples, tuples of
lists and arrays.

Vale segments sentences, so the same word may end one sentence and begin the
next. Consider a mesh. Mesh cells follow.
