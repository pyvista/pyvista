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

The second badge on a card says in a few words what using the data outside of
learning PyVista obliges you to do. It is the most restrictive term of the
dataset's license, and :attr:`Example.usage <pyvista.examples.Example.usage>`
is the same value as a string.

- :bdg-success:`No restrictions`: use it for anything.
- :bdg-info:`Credit required`: use it for anything, but credit the source.
- :bdg-warning:`Share alike`: credit the source, and share derived work under
  the same license.
- :bdg-danger:`Not for commercial use`: fine for learning, research, and demos,
  not for a product for sale.
- :bdg-muted:`Terms undetermined`: nobody could establish the terms, so treat
  it as all rights reserved.
- :bdg-muted-line:`Not recorded`: the file has no record yet, so treat it like
  ``Terms undetermined``.

A dataset generated in code carries no badge. The :guilabel:`License` dropdown
under each card names the license, the credit line to use, and the origin; the
full record, with the authors, the route the file took, and the notes behind
the license, is :attr:`Example.metadata <pyvista.examples.Example.metadata>`.

.. note::
    Everything on this page can also be queried at runtime with
    :func:`~pyvista.examples.get_example`::

        >>> from pyvista import examples
        >>> bunny = examples.get_example('bunny')
        >>> bunny.usage
        'non_commercial'
        >>> bunny.attribution
        'Stanford Computer Graphics Laboratory.'

.. include:: /api/examples/dataset-gallery/dataset_carousel.rst
