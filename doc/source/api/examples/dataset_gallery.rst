.. _dataset_gallery:

Dataset Gallery
===============
Browse PyVista datasets and their metadata.

The gallery includes datasets from the following modules:

- :bdg-primary:`Built-in` - datasets from :mod:`pyvista.examples.examples`
- :bdg-secondary:`Downloads` - datasets from :mod:`pyvista.examples.downloads`
- :bdg-success:`Planets` - datasets from :mod:`pyvista.examples.planets`

Use the search box and filters below to narrow the results by module, data
type, cell type, reader, file size, license, or usage. Each filter accepts
multiple values and combines with the others.

.. _dataset_gallery_usage:

Usage Badges
------------

The second badge on every card answers that in a few words. It is the most
restrictive term of the dataset's license, and clicking it brings you back here.
Read it as: what you must do, or must not do, when you use the data outside of
learning PyVista.

.. list-table::
   :header-rows: 1
   :widths: 22 56 22

   * - Badge
     - What it means for you
     - :attr:`Example.usage <pyvista.examples.Example.usage>`
   * - :bdg-success:`No restrictions`
     - Use it for anything, including a product for sale. Nothing is required of you.
     - ``'unrestricted'``
   * - :bdg-info:`Credit required`
     - Use it for anything, including a product for sale, but credit the source.
       The wording to use is the ``Attribution`` row of the card, and
       :attr:`Example.attribution <pyvista.examples.Example.attribution>` at runtime.
     - ``'attribution'``
   * - :bdg-warning:`Share alike`
     - Use it and credit it, and anything you derive from it must be shared under
       the same license.
     - ``'share_alike'``
   * - :bdg-danger:`Not for commercial use`
     - Fine for learning, teaching, research, and demos; not for a product for sale.
       Credit the source where the ``Attribution`` row asks you to.
     - ``'non_commercial'``
   * - :bdg-muted:`Terms undetermined`
     - Nobody could establish the terms. Treat it as all rights reserved: run the
       examples, but do not redistribute it or build on it.
     - ``'undetermined'``
   * - :bdg-muted-line:`Not recorded`
     - The file is not catalogued in `pyvista/data <https://github.com/pyvista/data>`_
       yet, so nothing has been checked. Treat it like ``Terms undetermined``.
     - ``None``

A dataset generated in code carries no badge, because there is no data file to
license. A license can attach more than one term, and the badge shows only the
strictest: a share-alike license also requires credit, and a non-commercial one
may too. The dropdown below lists every term.

Origin & License
----------------

Every card ends in an :guilabel:`Origin & License` dropdown with the details a
compliance review needs. Its rows are named after the properties of
:class:`~pyvista.examples.Example` and :class:`~pyvista.examples.ExampleMetadata`,
so anything shown there can be read back at runtime.

.. list-table::
   :header-rows: 1
   :widths: 24 50 26

   * - Row
     - Contents
     - Property
   * - Usage
     - The badge from the table above.
     - :attr:`Example.usage <pyvista.examples.Example.usage>`
   * - License
     - One :bdg-link-primary:`badge` per license, linking the exact text redistributed
       with the data, followed by the license name linking the issuing organization's
       own page.
     - :attr:`Example.license <pyvista.examples.Example.license>`,
       :attr:`ExampleMetadata.licenses <pyvista.examples.ExampleMetadata.licenses>`
   * - Commercial use, Attribution required, Share alike
     - ``Yes`` or ``No`` for each, combined over every license named, so a second
       term the badge does not show is still visible.
     - :attr:`~pyvista.examples.ExampleMetadata.commercial_use`,
       :attr:`~pyvista.examples.ExampleMetadata.attribution_required`,
       :attr:`~pyvista.examples.ExampleMetadata.share_alike`
   * - Attribution
     - The credit line to use.
     - :attr:`Example.attribution <pyvista.examples.Example.attribution>`
   * - Origin, Collection, Redistributed from, Authors, Copyright
     - Where the data came from, and through whom it reached PyVista.
     - :attr:`~pyvista.examples.ExampleMetadata.origin_url`,
       :attr:`~pyvista.examples.ExampleMetadata.collection`,
       :attr:`~pyvista.examples.ExampleMetadata.redistributed_from`,
       :attr:`~pyvista.examples.ExampleMetadata.authors`,
       :attr:`~pyvista.examples.ExampleMetadata.copyright`
   * - Provenance
     - How far the origin could be established: :bdg-success-line:`verified` is
       stated at the source or proved by comparing bytes,
       :bdg-warning-line:`inferred` is a reasoned conclusion explained in the
       notes, and :bdg-danger-line:`unknown` could not be established at all. This
       is a statement about the origin, not the license, so a verified origin with
       undetermined terms is not a contradiction.
     - :attr:`~pyvista.examples.ExampleMetadata.provenance`
   * - Files
     - Each file the example downloads, linking where it is fetched from.
     - :attr:`Example.paths <pyvista.examples.Example.paths>`,
       :attr:`Example.source_urls <pyvista.examples.Example.source_urls>`
   * - Modification, References, Provenance notes
     - What was changed after the data left its source, what to cite, and what was
       and was not established.
     - :attr:`~pyvista.examples.ExampleMetadata.modification`,
       :attr:`~pyvista.examples.ExampleMetadata.references`,
       :attr:`~pyvista.examples.ExampleMetadata.notes`

The module badges at the top of this page are a separate set and say nothing
about licensing.

.. note::
    Everything on this page can also be queried at runtime with
    :func:`~pyvista.examples.get_example`. Looking up an example by name gives its
    file paths, their sizes and download URLs, the readers used to read them, the
    usage badge as a string, and the full record above::

        >>> from pyvista import examples
        >>> bunny = examples.get_example('bunny')
        >>> bunny.usage
        'non_commercial'
        >>> bunny.attribution
        'Stanford Computer Graphics Laboratory.'
        >>> bunny.metadata.provenance
        'verified'

.. include:: /api/examples/dataset-gallery/dataset_carousel.rst
