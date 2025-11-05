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
- :bdg-danger:`cmc` - `cmcrameri colormaps <https://github.com/callumrollo/cmcrameri?tab=readme-ov-file#cmcrameri>`_.

The type of the colormap is indicated as:

- :bdg-muted:`LSC` if it is a :class:`~matplotlib.colors.LinearSegmentedColormap`
- :bdg-muted:`LC` if it is a :class:`~matplotlib.colors.ListedColormap`

Each colormap is also tagged as:

- :material-regular:`visibility;2em;sd-text-info` - if it is perceptually uniform
- :material-regular:`visibility_off;2em;sd-text-warning` - if it is not perceptually uniform

A colormap is considered *perceptually uniform* if its color gradients are
evenly spaced, meaning equal steps in data produce equal perceptual changes.

Some colormap libraries assess perceptual uniformity using only the lightness
component (:math:`L^*` in the Lab color space), while others also consider
chromatic components (:math:`a^*` and :math:`b^*`). For the colormaps on this
reference page, both lightness *and* color differences must meet the uniformity
criteria to earn the :material-regular:`visibility;2em;sd-text-info` label.

Specifically, a colormap is labeled as perceptually uniform if:

#. The cumulative change in lightness (:math:`L^*` from CAM02-UCS) progresses
   linearly. See `lightness of matplotlib colormaps <https://matplotlib.org/stable/users/explain/colors/colormaps.html#lightness-of-matplotlib-colormaps>`_.
#. The cumulative color difference (:math:`\Delta E`, using the CIEDE2000
   metric) progresses linearly. See `Fig. 3c: cumulative color lightness difference <https://www.nature.com/articles/s41467-020-19160-7/figures/3>`_.

Linearity is defined as having a coefficient of determination (:math:`R^2`)
greater than 0.99 when fitted with linear regression.
When choosing a colormap, those with the :material-regular:`visibility;2em;sd-text-info`
tag should be preferred over those with the :material-regular:`visibility_off;2em;sd-text-warning`
tag.

.. warning::

    Many of the ``matplotlib`` colormaps such as ``jet`` are not perceptually
    uniform and should be avoided where possible, since these colormaps
    can generate misleading visualizations. Colormaps from ``colorcet`` and
    ``cmocean``, and ``cmcrameri`` are therefore generally recommended over
    those from ``matplotlib``.
    See `the misuse of colour in science communication <https://doi.org/10.1038/s41467-020-19160-7>`_
    and `testing perceptual uniformity <https://colorcet.holoviz.org/user_guide/Continuous.html#testing-perceptual-uniformity>`_
    for more information.

Refer to the flowchart in the following dropdown for guidance on how
to choose a colormap.

.. dropdown:: Guideline for choosing the right scientific colormap

    .. image:: https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-020-19160-7/MediaObjects/41467_2020_19160_Fig6_HTML.png

    Figure from Crameri, F., Shephard, G.E., & Heron, P.J. (2020). The misuse of
    colour in science communication. Nature Communications, 11, 5444.
    https://doi.org/10.1038/s41467-020-19160-7. Licensed under CC BY 4.0.

.. note::

    Some colormap names such as ``gray`` and ``rainbow`` are duplicated across
    packages, but have slight variations in the mapped colors. Colormaps from
    ``colorcet`` have priority and are used first if available, followed
    by ``cmocean``, followed by the stock colormaps from ``matplotlib``.

.. note::

    Install PyVista with ``pyvista[colormaps]`` or ``pyvista[all]`` to also
    install the ``colorcet``, ``cmocean``, and ``cmcrameri`` packages.

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

Multi-Sequential
----------------

Colormaps with multiple sequential gradients diverging from a midpoint, ideal
for highlighting deviations on both sides of a reference value.

.. dropdown::
    :open:

    .. include:: /api/utilities/colormap_table/colormap_table_MULTI_SEQUENTIAL.rst

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
