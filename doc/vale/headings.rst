Heading Style Test Cases
========================

Every heading in this file must pass ``Google.Headings``. The file is not part
of the documentation -- Sphinx builds from ``doc/source`` -- but Vale scans it
along with the rest of ``doc``, so a change to the rule that breaks one of
these cases fails CI here rather than somewhere in the real documentation.

The style is AP title case with one deviation, ``to``. See
``doc/styles/Google/Headings.yml`` for the rule and the reasoning.

Reading a File for Fun
----------------------

Short words stay lower-case in the middle of a heading: ``a an the and but or
nor for of in on at by as if per via en vs``.

The End of an Era
-----------------

A short word is still capitalized as the first or last word.

Clip With Plane
---------------

Prepositions of four letters or more are capitalized, which is what separates
AP from Chicago. ``with`` is the most common one here.

Textures From Files
-------------------

Same for ``from``.

Distance Between Two Surfaces
-----------------------------

And for the rarer ones: between, over, along, before, about, within, across.

Terrain Following Mesh
----------------------

``following`` is a participle here, not a preposition. Chicago would lower-case
it and be wrong.

What Is a Mesh?
---------------

Verbs are capitalized, however short.

Modify Which Actors Are Pickable
--------------------------------

So is ``are``.

Confirm That the Capped Result Is Watertight
--------------------------------------------

And ``that``. This heading is also long enough to catch a threshold that lets
one wrong word through.

List-Like Features
------------------

Both halves of a hyphenated compound are capitalized.

Built to Extend
---------------

``to`` stays lower-case throughout, unlike AP proper.

Ways To Extend PyVista
----------------------

A capital ``To`` also passes: Vale decides AP's infinitive rule by asking
whether the next word is a noun, which is wrong in both directions, so the rule
does not enforce either form.

Working With glTF Files
-----------------------

Terms keep the casing they are written with.

Testing With pytest
-------------------

Including ones that start lower-case.

Building on a Mac mini (64-bit)
-------------------------------

``mini`` is Apple's spelling and ``64-bit`` is the ordinary one.

pyvista.ArrayLike
-----------------

An API name is a heading in ``doc/source/api/core/typing.rst``, so the
``pyvista.`` namespace is exempt.
