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
    GRDECLReader
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
    PExodusIIReader
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
    ThreeDSReader
    TIFFReader
    TecplotReader
    VRMLReader
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
.. autofunction:: pyvista.registered_readers
.. autoclass:: pyvista.ReaderRegistration

**Two forms of reader**

A registration is either a plain callable or a
:class:`pyvista.BaseReader` subclass, and the choice decides how much
of PyVista's reader machinery the format gets:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Capability
     - Callable
     - ``BaseReader`` subclass
   * - :func:`pyvista.read`
     - yes
     - yes
   * - :func:`pyvista.get_reader`
     - no
     - yes
   * - Keyword arguments to :func:`pyvista.read`
     - dropped
     - set as reader attributes
   * - ``progress_bar=True``, ``validate=``
     - ignored
     - honored
   * - :class:`pyvista.TimeReader`,
       :class:`pyvista.PointCellDataSelection`
     - unavailable
     - available

A callable is the lighter option and is the right choice for a format
with no reader-level state to expose. Register a
:class:`pyvista.BaseReader` subclass for anything a user will want to
configure, step through in time, or select arrays from.

To write a reader class for a format VTK has no reader for, subclass
:class:`pyvista.BaseVTKReader` for the parsing and point a
:class:`pyvista.BaseReader` subclass at it::

   import pyvista as pv


   class _MyVTKReader(pv.BaseVTKReader):
       def UpdateInformation(self):
           pass

       def Update(self):
           self._data_object = _parse(self._filename)


   @pv.register_reader('.myformat')
   class MyReader(pv.BaseReader):
       _class_reader = _MyVTKReader

**Entry points**

Packages can also register readers in ``pyproject.toml`` so they are
discovered automatically when installed. The entry-point value may name
either a callable or a :class:`pyvista.BaseReader` subclass:

.. code-block:: toml

   [project.entry-points."pyvista.readers"]
   ".myformat" = "my_package:read_my_format"
   ".myotherformat" = "my_package:MyOtherReader"

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
.. autofunction:: pyvista.registered_writers
.. autoclass:: pyvista.WriterRegistration

**Handler signature**

A writer handler is a callable ``handler(dataset, path, **kwargs)``
that writes *dataset* to *path*.  Any extra keyword arguments passed
to :meth:`pyvista.DataObject.save` beyond its documented parameters
are forwarded verbatim to the handler as ``**kwargs``. Use them to
expose format-specific options such as compression level, thread
count, or chunking.  When no custom writer is registered for the
target extension, passing extra keyword arguments to
:meth:`~pyvista.DataObject.save` raises :class:`TypeError`; PyVista
never silently drops writer options.

**Entry points**

Packages can register writers in ``pyproject.toml`` so they are
discovered automatically when installed:

.. code-block:: toml

   [project.entry-points."pyvista.writers"]
   ".myformat" = "my_package:write_my_format"

**Dispatch order**

When :meth:`~pyvista.DataObject.save` is called, custom writers
registered via :func:`pyvista.register_writer` are dispatched *before*
built-in VTK writers for the same extension, mirroring the dispatch
order used by :func:`pyvista.read`.  By default, registering a
handler for an extension that collides with a built-in PyVista writer
raises :class:`ValueError`; pass ``override=True`` to replace the
built-in writer.


The ``.pv`` Format: PyVista's Native Binary Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyVista has a native ``zstd``-compressed binary format with the
``.pv`` extension, implemented by the
`pyvista-zstd <https://github.com/pyvista/pyvista-zstd>`_ companion
package.  It is a compact, multi-threaded format for fast dataset
I/O and is included in the ``io`` extra::

   pip install pyvista[io]

Once installed, ``.pv`` round-trips "just work" via the
``pyvista.readers`` and ``pyvista.writers`` entry-point hooks
without any manual registration::

   import pyvista as pv

   mesh = pv.Sphere()
   mesh.save('sphere.pv')
   pv.read('sphere.pv')

Supported dataset types include :class:`~pyvista.ImageData`,
:class:`~pyvista.PolyData`, :class:`~pyvista.StructuredGrid`,
:class:`~pyvista.RectilinearGrid`, :class:`~pyvista.UnstructuredGrid`,
:class:`~pyvista.MultiBlock`, and
:class:`~pyvista.ExplicitStructuredGrid`.  The format uses ``zstd``
compression with multi-threaded encode/decode and is a good choice
over ``.vtu`` / ``.vtp`` / ``.vtm`` when file size or I/O latency
matters.


Writer Classes
~~~~~~~~~~~~~~

PyVista provides built-in writer classes for saving datasets to various file
formats. These are used internally by :meth:`pyvista.DataObject.save`.

.. autosummary::
   :toctree: _autosummary

   BaseWriter
   BMPWriter
   DataSetWriter
   EnSightWriter
   HDFWriter
   HoudiniPolyDataWriter
   IVWriter
   JPEGWriter
   NIFTIImageWriter
   OBJWriter
   PLYWriter
   PNGWriter
   PNMWriter
   PolyDataWriter
   RectilinearGridWriter
   STLWriter
   SimplePointsWriter
   StructuredGridWriter
   TIFFWriter
   UnstructuredGridWriter
   XMLImageDataWriter
   XMLMultiBlockDataWriter
   XMLPartitionedDataSetWriter
   XMLPolyDataWriter
   XMLRectilinearGridWriter
   XMLStructuredGridWriter
   XMLUnstructuredGridWriter


Inherited Classes
~~~~~~~~~~~~~~~~~

The :class:`pyvista.BaseReader` is inherited by all sub-readers. It
has the basic functionality of all readers to set filename and read
the data.

The :class:`PointCellDataSelection` is inherited by readers that
support inspecting and setting data related to point and cell arrays.

The :class:`TimeReader` is inherited by readers that support inspecting
and setting time or iterations for reading.

The :class:`BaseVTKReader` is the base for a reader implemented in pure
Python rather than by a VTK reader class. Subclass it, implement
``UpdateInformation`` and ``Update``, and point a
:class:`pyvista.BaseReader` subclass at it through ``_class_reader``.
This is how :class:`pyvista.PVDReader` and :class:`pyvista.FRDReader`
are built, and it is the supported base for a third-party reader
registered with :func:`pyvista.register_reader`.

.. autosummary::
   :toctree: _autosummary

   BaseReader
   BaseVTKReader
   PointCellDataSelection
   PVDDataSet
   SeriesDataSet
   TimeReader


Enumerations
~~~~~~~~~~~~

Enumerations are available to simplify inputs to certain readers.

.. toctree::
    :maxdepth: 2

    enums
