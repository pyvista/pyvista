.. tip::

   A native ``.pv`` binary format with ``zstd`` compression is
   available via the `pyvista-zstd
   <https://github.com/pyvista/pyvista-zstd>`_ companion package,
   included in the ``io`` extra (``pip install pyvista[io]``). It is
   a compact, multi-threaded alternative to the built-in VTK formats
   below when file size or I/O latency matters.  Third-party
   packages can register additional custom writers via
   :func:`pyvista.register_writer`.

.. dropdown:: :class:`~pyvista.ImageData` File Formats

    .. include:: /api/utilities/io_table/ImageData_io_table.rst

.. dropdown:: :class:`~pyvista.RectilinearGrid` File Formats

    .. include:: /api/utilities/io_table/RectilinearGrid_io_table.rst

.. dropdown:: :class:`~pyvista.StructuredGrid` File Formats

    .. include:: /api/utilities/io_table/StructuredGrid_io_table.rst

.. dropdown:: :class:`~pyvista.PolyData` File Formats

    .. include:: /api/utilities/io_table/PolyData_io_table.rst

.. dropdown:: :class:`~pyvista.UnstructuredGrid` File Formats

    .. include:: /api/utilities/io_table/UnstructuredGrid_io_table.rst

.. dropdown:: :class:`~pyvista.MultiBlock` File Formats

    .. include:: /api/utilities/io_table/MultiBlock_io_table.rst

.. dropdown:: :class:`~pyvista.PartitionedDataSet` File Formats

    .. include:: /api/utilities/io_table/PartitionedDataSet_io_table.rst
