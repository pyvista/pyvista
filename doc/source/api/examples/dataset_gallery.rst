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

.. note::
    Much of the metadata shown on this page can also be queried at runtime with
    :func:`~pyvista.examples.get_example`. Looking up an example by name gives its
    file paths, their sizes and source URLs, and the readers used to read them::

        >>> from pyvista import examples
        >>> frog = examples.get_example('frog')
        >>> frog.paths  # doctest:+SKIP
        ('.../froggy/frog.mhd', '.../froggy/frog.zraw')
        >>> [type(reader).__name__ for reader in frog.readers]
        ['MetaImageReader']

    See :class:`~pyvista.examples.Example` for every field.

.. include:: /api/examples/dataset-gallery/dataset_carousel.rst
