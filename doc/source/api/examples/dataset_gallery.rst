.. _dataset_gallery:

Dataset Gallery
===============
Browse PyVista datasets and their metadata.

The gallery includes datasets from the following modules:

- :bdg-primary:`Built-in` - datasets from :mod:`pyvista.examples.examples`
- :bdg-secondary:`Downloads` - datasets from :mod:`pyvista.examples.downloads`
- :bdg-success:`Planets` - datasets from :mod:`pyvista.examples.planets`

Use the search box and filters below to narrow the results by module, data
type, cell type, reader, or file size. Each filter accepts multiple values
and combines with the others.

Badge legend
------------

Every card carries an :guilabel:`Origin & License` dropdown whose rows are named
after the properties of :class:`~pyvista.examples.Example`, so anything shown here
can be read back at runtime. Its badges come in two families:

- **Solid** badges are obligations the licence places on you.
- **Outlined** badges are how far the origin of the data could be established,
  which is a statement about this project's confidence rather than about the licence.

.. list-table::
   :header-rows: 1
   :widths: 26 48 26

   * - Badge
     - Meaning
     - ``Example`` property
   * - :bdg-primary:`CC-BY-4.0`
     - The licence. On a card it is a link to the full text, and a dataset may
       carry more than one.
     - :attr:`~pyvista.examples.Example.license`
   * - :bdg-success:`Commercial use`
     - The licence permits commercial use.
     - :attr:`~pyvista.examples.Example.commercial_use`
   * - :bdg-danger:`Not for commercial use`
     - It does not, or the terms could not be established. Treat as unlicensed.
     - :attr:`~pyvista.examples.Example.commercial_use`
   * - :bdg-warning:`ShareAlike`
     - Work derived from this data must be shared under the same licence.
     - :attr:`~pyvista.examples.Example.share_alike`
   * - :bdg-info:`Attribution required`
     - The licence requires credit; the wording to use is in the ``Attribution`` row.
     - :attr:`~pyvista.examples.Example.attribution_required`
   * - :bdg-success-line:`verified`
     - The origin is stated at the source, or proved by comparing bytes.
     - :attr:`~pyvista.examples.Example.provenance`
   * - :bdg-warning-line:`inferred`
     - The origin is a reasoned conclusion; the ``Notes`` row gives the reasoning.
     - :attr:`~pyvista.examples.Example.provenance`
   * - :bdg-danger-line:`unknown`
     - The origin could not be established at all.
     - :attr:`~pyvista.examples.Example.provenance`

Provenance and licence are separate: a dataset can have a certain origin and
undetermined terms, so a :bdg-success-line:`verified` badge beside a
:bdg-danger:`Not for commercial use` badge is not a contradiction.

The module badges at the top of this page are a separate set and say nothing
about licensing. Licence badges are the only ones that are links.

.. note::
    Everything on this page can also be queried at runtime with
    :func:`~pyvista.examples.get_example`. Looking up an example by name gives its
    file paths, their sizes and download URLs, the readers used to read them, and
    every field of the record above::

        >>> from pyvista import examples
        >>> frog = examples.get_example('frog')
        >>> frog.paths  # doctest:+SKIP
        ('.../froggy/frog.mhd', '.../froggy/frog.zraw')
        >>> frog.license  # doctest:+SKIP
        'Apache-2.0'

    See :class:`~pyvista.examples.Example` for every field.

.. include:: /api/examples/dataset-gallery/dataset_carousel.rst
