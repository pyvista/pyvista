.. _named_colormaps:

Named Colormaps
===============

Named colormaps supported by :class:`~pyvista.LookupTable`, :class:`~pyvista.Plotter`,
and other plotting-related methods.

The colormaps on this page are from multiple packages:

- :bdg-secondary:`mpl` -
  `matplotlib colormaps <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_.
- :bdg-success:`cc` -
  ``colorcet`` `continuous <https://colorcet.holoviz.org/user_guide/Continuous.html#named-colormaps>`_
  and `categorical <https://colorcet.holoviz.org/user_guide/Categorical.html#categorical>`_
  colormaps.
- :bdg-primary:`cmo` - `cmocean colormaps <https://matplotlib.org/cmocean/>`_.

The type of the colormap is indicated as:

- :bdg-muted:`LSC` if it is a :class:`~matplotlib.colors.LinearSegmentedColormap`
- :bdg-muted:`LC` if it is a :class:`~matplotlib.colors.ListedColormap`

Each colormap is also labeled as:

- :bdg-muted:`PU` if it is perceptually uniform
- :bdg-danger:`NPU` if it is not perceptually uniform

.. warning::

    Many of the ``matplotlib`` colormaps such as ``jet`` are not perceptually
    uniform and should be avoided where possible, since these colormaps
    can generate misleading visualizations. Colormaps from ``colorcet`` and
    ``cmocean`` are therefore generally recommended over those from ``matplotlib``.
    See `the misuse of colour in science communication <https://doi.org/10.1038/s41467-020-19160-7>`_
    and `testing perceptual uniformity <https://colorcet.holoviz.org/user_guide/Continuous.html#testing-perceptual-uniformity>`_
    for more information.

.. note::

    Some colormap names such as ``gray`` and ``rainbow`` are duplicated across
    packages, but have slight variations in the mapped colors. Colormaps from
    ``colorcet`` have priority and are used first if available, followed
    by ``cmocean``, followed by the stock colormaps from ``matplotlib``.

.. note::

    Install PyVista with ``pyvista[colormaps]`` or ``pyvista[all]`` to also
    install the ``colorcet`` and ``cmocean`` packages.

.. seealso::

    :ref:`colormap_example`
        Example using colormaps from different sources.

    :ref:`named_colors`
        Similar reference for named colors.

Linear (Sequential)
-------------------

Colormaps that progress in a single direction, ideal for representing ordered
data such as intensities or magnitudes.

.. dropdown::
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_LINEAR.rst

Diverging
---------

Colormaps with two contrasting colors diverging from a central midpoint, useful
for highlighting deviation from a reference value.

.. dropdown::
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_DIVERGING.rst

Cyclic
------

Colormaps designed to wrap around smoothly, best for data that is inherently
circular such as angles and phase.

.. dropdown::
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_CYCLIC.rst

Categorical (Qualitative)
-------------------------

Colormaps using distinct colors for individual categories, suitable for labeling
discrete classes or groups.

.. dropdown::
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_CATEGORICAL.rst

Miscellaneous
-------------

Colormaps that don’t fit neatly into other categories, often used for artistic
or specialized purposes.

.. dropdown::
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_MISC.rst

CET Colormaps
=============

This section includes all ``colorcet`` colormaps that have a named ``CET``
alias (originally from the Center for Exploration Targeting).

Colormap names have the form::

    CET_[prefix]{type}{number}[suffix]

Where the prefix and suffix are optional (``[]``) and the type and number are
required (``{}``).

Prefix :
    Indicates additional information about color perception.

    - ``CB``: Colorblind—safe for red—green color vision deficiency (protanopia/deuteranopia)
    - ``CBT``: Colorblind—safe for blue—yellow color vision deficiency (tritanopia)

Type :
    The type of colormap.

    - ``C``: Cyclic
    - ``D``: Diverging
    - ``L``: Linear
    - ``R``: Rainbow
    - ``I``: Isoluminant

Number :
    A version number (starting at ``1``) for each unique ``[prefix]{type}`` combination.

Suffix :
    Indicates a minor variation of the base colormap.

    - ``A``: Alternate version (such as adjusted contrast or hue emphasis)
    - ``s``: Shifted version (phase-shifted)

.. note::

    Most of the ``colorcet`` colormaps presented above in :ref:`named_colors`
    are duplicated here (using their ``CET`` aliases).

Linear (Sequential)
-------------------

Colormaps that progress in a single direction, ideal for representing ordered
data such as intensities or magnitudes.

.. dropdown::
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_CET_LINEAR.rst

Diverging
---------

Colormaps with two contrasting colors diverging from a central midpoint, useful
for highlighting deviation from a reference value.

.. dropdown::
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_CET_DIVERGING.rst

Cyclic
------

Colormaps designed to wrap around smoothly, best for data that is inherently
circular such as angles and phase.

.. dropdown::
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_CET_CYCLIC.rst

Rainbow
-------

Multi-hue colormaps that span the full visible spectrum, useful for highlighting
local differences in sequential data.

.. warning::

    Some of the “rainbow” colormaps have a perceptual discontinuity around the colors red and yellow.

.. dropdown::
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_CET_RAINBOW.rst

Isoluminant
-----------

Colormaps with constant perceived brightness, useful for emphasizing shape and
structure without introducing false intensity cues.

.. dropdown::
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_CET_ISOLUMINANT.rst
