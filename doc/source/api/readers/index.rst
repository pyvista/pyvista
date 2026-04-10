.. _reader_api:

Readers
=======
PyVista provides class based readers to have more control over reading
data files. These classes allows for more fine-grained control over
reading datasets from files. See :func:`pyvista.get_reader` for a
list of file types supported.

Also, see :ref:`reader_example` for a full example using reader classes.

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   get_reader

Reader Classes
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

    AVSucdReader
    BMPReader
    BYUReader
    BinaryMarchingCubesReader
    CGNSReader
    DEMReader
    DICOMReader
    EnSightReader
    ExodusIIBlockSet
    ExodusIIReader
    FacetReader
    FLUENTCFFReader
    FluentReader
    FRDReader
    GambitReader
    GaussianCubeReader
    GESignaReader
    GIFReader
    GLTFReader
    HDFReader
    HDRReader
    JPEGReader
    MFIXReader
    MetaImageReader
    MINCImageReader
    MultiBlockPlot3DReader
    Nek5000Reader
    NIFTIReader
    NRRDReader
    OBJReader
    OpenFOAMReader
    ParticleReader
    PDBReader
    PLYReader
    PNGReader
    PNMReader
    POpenFOAMReader
    PTSReader
    PVDReader
    Plot3DMetaReader
    ProStarReader
    SLCReader
    STLReader
    SegYReader
    SeriesReader
    TIFFReader
    TecplotReader
    VTKDataSetReader
    VTKPDataSetReader
    XMLImageDataReader
    XMLMultiBlockDataReader
    XMLPImageDataReader
    XMLPRectilinearGridReader
    XMLPUnstructuredGridReader
    XMLPartitionedDataSetReader
    XMLPolyDataReader
    XMLRectilinearGridReader
    XMLStructuredGridReader
    XMLUnstructuredGridReader
    XdmfReader

Custom Readers
~~~~~~~~~~~~~~

Third-party packages can register custom readers so that
:func:`pyvista.read` handles additional file formats automatically.
Registration can be done programmatically or via Python entry points
for zero-config discovery at install time.

.. autofunction:: pyvista.register_reader

**Entry points**

Packages can also register readers in ``pyproject.toml`` so they are
discovered automatically when installed:

.. code-block:: toml

   [project.entry-points."pyvista.readers"]
   ".myformat" = "my_package:read_my_format"

**Remote URI support**

When :func:`pyvista.read` is given a remote URI (``https://``,
``s3://``, etc.) and a custom reader is registered for the file
extension, the URI is passed directly to the reader.  If the reader
raises :class:`~pyvista.LocalFileRequiredError`, PyVista downloads
the file to a temporary local path and retries.  For built-in
formats with no custom reader, the download happens automatically.
This uses ``fsspec`` when available (install with
``pip install pyvista[io]``), falling back to ``pooch`` for HTTP(S)
URIs.

.. autoclass:: pyvista.LocalFileRequiredError
.. autofunction:: pyvista.has_scheme


Custom Writers
~~~~~~~~~~~~~~

Third-party packages can register custom writers so that
:meth:`pyvista.DataObject.save` handles additional file formats
automatically.  Registration mirrors :func:`pyvista.register_reader`
and supports programmatic calls, decorators, and Python entry points
for zero-config discovery at install time.

.. autofunction:: pyvista.register_writer

**Handler signature**

A writer handler is a callable ``handler(dataset, path)`` that writes
*dataset* to *path*.  Writer-specific options (compression level,
threading, etc.) are configured through the handler's own API, not
through :meth:`pyvista.DataObject.save`.

**Entry points**

Packages can register writers in ``pyproject.toml`` so they are
discovered automatically when installed:

.. code-block:: toml

   [project.entry-points."pyvista.writers"]
   ".myformat" = "my_package:write_my_format"

**Dispatch order**

When :meth:`~pyvista.DataObject.save` is called, custom writers
registered via :func:`pyvista.register_writer` are dispatched *before*
built-in VTK writers for the same extension — mirroring the dispatch
order used by :func:`pyvista.read`.  By default, registering a
handler for an extension that collides with a built-in PyVista writer
raises :class:`ValueError`; pass ``override=True`` to replace the
built-in writer.


The ``.pv`` format — PyVista's native binary format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyVista has a native Zstd-compressed binary format with the ``.pv``
extension, implemented by the
`pyvista-zstd <https://github.com/pyvista/pyvista-zstd>`_ companion
package.  It is a compact, multi-threaded format for fast dataset
I/O and is included in the ``io`` extra::

   pip install pyvista[io]

Once installed, ``.pv`` round-trips "just work" via the
``pyvista.readers`` and ``pyvista.writers`` entry-point hooks — no
manual registration is needed::

   import pyvista as pv

   mesh = pv.Sphere()
   mesh.save('sphere.pv')
   pv.read('sphere.pv')

Supported dataset types include :class:`~pyvista.ImageData`,
:class:`~pyvista.PolyData`, :class:`~pyvista.StructuredGrid`,
:class:`~pyvista.RectilinearGrid`, :class:`~pyvista.UnstructuredGrid`,
:class:`~pyvista.MultiBlock`, and
:class:`~pyvista.ExplicitStructuredGrid`.  The format uses Zstd
compression with multi-threaded encode/decode and is a good choice
over ``.vtu`` / ``.vtp`` / ``.vtm`` when file size or I/O latency
matters.


Inherited Classes
~~~~~~~~~~~~~~~~~

The :class:`pyvista.BaseReader` is inherited by all sub-readers. It
has the basic functionality of all readers to set filename and read
the data.

The :class:`PointCellDataSelection` is inherited by readers that
support inspecting and setting data related to point and cell arrays.

The :class:`TimeReader` is inherited by readers that support inspecting
and setting time or iterations for reading.

.. autosummary::
   :toctree: _autosummary

   BaseReader
   PointCellDataSelection
   TimeReader


Enumerations
~~~~~~~~~~~~

Enumerations are available to simplify inputs to certain readers.

.. toctree::
    :maxdepth: 2

    enums
