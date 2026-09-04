.. _reader_api:

Readers and Writers
===================
PyVista provides class based readers to have more control over reading
data files. These classes allow for more fine-grained control over
reading datasets from files. See :func:`pyvista.get_reader` for a
list of file types supported. The writer classes used by
:meth:`pyvista.DataObject.save` are listed further down this page.

Also, see :ref:`reader_example` for a full example using reader classes.

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   get_reader

Reading Functions
~~~~~~~~~~~~~~~~~

These functions read a file in a single call, selecting the reader from
the file extension. To write a file, see :meth:`~pyvista.DataObject.save`.

.. autosummary::
   :toctree: _autosummary

   get_ext
   read
   read_exodus
   read_grdecl
   read_texture

.. seealso::

   :ref:`read_file_example`
      Load and plot a mesh from a file.

   :ref:`conversions_api`
      Read and write files with ``meshio``.

Reader Classes
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

    AVSucdReader
    BinaryMarchingCubesReader
    BMPReader
    BYUReader
    CGNSReader
    DEMReader
    DICOMReader
    EnSightReader
    ExodusIIBlockSet
    ExodusIIReader
    FacetReader
    FLUENTCFFReader
    FluentReader
    GambitReader
    GaussianCubeReader
    GESignaReader
    GIFReader
    GLTFReader
    GRDECLReader
    HDFReader
    HDRReader
    JPEGReader
    MetaImageReader
    MFIXReader
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
    Plot3DMetaReader
    PLYReader
    PNGReader
    PNMReader
    POpenFOAMReader
    ProStarReader
    PTSReader
    PVDReader
    SegYReader
    SeriesReader
    SLCReader
    STLReader
    TecplotReader
    ThreeDSReader
    TIFFReader
    VRMLReader
    VTKDataSetReader
    VTKPDataSetReader
    XdmfReader
    XMLImageDataReader
    XMLMultiBlockDataReader
    XMLPartitionedDataSetReader
    XMLPImageDataReader
    XMLPolyDataReader
    XMLPRectilinearGridReader
    XMLPUnstructuredGridReader
    XMLRectilinearGridReader
    XMLStructuredGridReader
    XMLUnstructuredGridReader

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

**Replacing a built-in reader**

An entry point in the ``pyvista.readers`` group may only claim an
extension PyVista does not already read.  Claiming one it does
(``.vtp``, ``.stl``, ``.ply``) would silently change what every
:func:`pyvista.read` call in the environment returns, so PyVista
refuses and raises :class:`ValueError` naming the package, the built-in
reader, and this section.

To replace a built-in reader on purpose, declare the entry point in the
``pyvista.readers.override`` group instead.  The two groups are
identical except that the override group is permitted to take an
extension PyVista ships a reader for, and does so silently:

.. code-block:: toml

   [project.entry-points."pyvista.readers.override"]
   ".vtp" = "my_package:MyPolyDataReader"

This is the entry-point equivalent of ``override=True`` on
:func:`pyvista.register_reader`.  Both groups accept both forms, a
callable or a :class:`pyvista.BaseReader` subclass.

Declaring an override for an extension PyVista does *not* currently
read is allowed and silent.  It costs nothing and keeps the package
working if a later PyVista release adds a reader for that extension.

Because an override changes the meaning of a format the user did not
choose, it is visible from :func:`pyvista.registered_readers`: the
record for the extension reports ``override=True`` along with the
``source`` that claimed it.  That is the first call to make when a
built-in format reads differently than expected.

.. code-block:: python

    import pyvista as pv

    taken = [
        (r.extension, r.source)
        for r in pv.registered_readers()
        if r.override
    ]
    # [('.vtp', 'my_package:MyPolyDataReader')]

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
that writes ``dataset`` to ``path``.  Any extra keyword arguments passed
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

Without it, both :func:`pyvista.read` and
:meth:`pyvista.DataObject.save` raise :class:`ImportError` naming the
package and the install command; see :ref:`optional_formats` below.

Supported dataset types include :class:`~pyvista.ImageData`,
:class:`~pyvista.PolyData`, :class:`~pyvista.StructuredGrid`,
:class:`~pyvista.RectilinearGrid`, :class:`~pyvista.UnstructuredGrid`,
:class:`~pyvista.MultiBlock`, and
:class:`~pyvista.ExplicitStructuredGrid`.  The format uses ``zstd``
compression with multi-threaded encode/decode and is a good choice
over ``.vtu`` / ``.vtp`` / ``.vtm`` when file size or I/O latency
matters.


.. _optional_formats:

Optional Formats
~~~~~~~~~~~~~~~~

A few formats are served by companion packages rather than by PyVista
itself, so that a heavyweight or narrowly used codec is not carried by
every install.  PyVista still knows the extension: :func:`pyvista.read`
and :meth:`pyvista.DataObject.save` dispatch to the companion package
when it is installed, and raise :class:`ImportError` naming the package
and the install command when it is not.

.. list-table::
   :header-rows: 1
   :widths: 10 30 15 25

   * - Extension
     - Format
     - Direction
     - Package
   * - ``.frd``
     - CalculiX FRD result files
     - read
     - `pyvista-frd-reader <https://github.com/pyvista/pyvista-frd-reader>`_
   * - ``.pv``
     - PyVista's native ``zstd``-compressed format
     - read, write
     - `pyvista-zstd <https://github.com/pyvista/pyvista-zstd>`_

All of them are included in the ``io`` extra::

   pip install pyvista[io]

Reading and saving are then transparent::

   import pyvista as pv

   mesh = pv.read('mesh.frd')
   mesh.save('mesh.pv')

Without the package, the extension is still recognized: the
:class:`ImportError` names the format, the missing package, and both
the ``pip install pyvista[io]`` command and the command for that one
package on its own.  When the package is present but fails to import,
the error reports the import failure instead, with no install command.

These packages provide the reader object themselves, so
:func:`pyvista.get_reader` does not resolve their extensions.  Use the
package's own reader class when reader-level control such as time-step
selection is needed::

   import pyvista_frd

   reader = pyvista_frd.FRDReader('mesh.frd')
   reader.set_active_time_value(reader.time_values[-1])
   mesh = reader.read()

The error :func:`pyvista.get_reader` raises names that class, so it
says where to go: ``pyvista_frd.FRDReader`` for ``.frd`` and
``pyvista_zstd.Reader`` for ``.pv``.

Keyword arguments beyond those :meth:`~pyvista.DataObject.save`
documents are forwarded to the package's writer, so format-specific
options are reachable without a separate import::

   mesh.save('mesh.pv', level=19, n_threads=4)

.. versionchanged:: 0.49.0
   ``.frd`` moved from a built-in reader to ``pyvista-frd-reader``.
   ``pyvista.FRDReader`` was removed; use ``pyvista_frd.FRDReader``.

.. versionchanged:: 0.49.0
   Reading or saving ``.pv`` without ``pyvista-zstd`` raises
   :class:`ImportError` rather than :class:`OSError` / :class:`ValueError`.


Faster Readers for Built-in Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two further companion packages read a format PyVista already supports,
faster than the VTK reader does.  They declare the
``pyvista.readers.override`` entry point described above, so installing
one is all it takes for :func:`pyvista.read` to use it.

.. list-table::
   :header-rows: 1
   :widths: 10 35 25

   * - Extension
     - Format
     - Package
   * - ``.ply``
     - Polygon File Format
     - `pyvista-miniply <https://github.com/pyvista/pyvista-miniply>`_
   * - ``.stl``
     - Stereolithography
     - `pyvista-stl <https://github.com/pyvista/pyvista-stl>`_

Both ship in the ``io-override`` extra, which is intentionally separate
from ``io`` because installing it changes the readers used for existing
formats::

   pip install pyvista[io-override]

The packages aim to match the stock VTK readers, including point normals,
texture coordinates, and colors, but their behavior and output are not
guaranteed to be identical.  Neither is required: without them
:func:`pyvista.read` falls back to :class:`pyvista.PLYReader` and
:class:`pyvista.STLReader`, which also remain what
:func:`pyvista.get_reader` hands back.

Because an override changes a format the user did not choose,
:func:`pyvista.registered_readers` reports it::

   import pyvista as pv

   [(r.extension, r.source) for r in pv.registered_readers() if r.override]
   # [('.ply', 'pyvista_miniply:read_as_mesh'), ('.stl', 'pyvista_stl:read_as_mesh')]

To read a file with the VTK reader while a package is installed, use
the reader class directly::

   mesh = pv.STLReader('mesh.stl').read()

.. versionadded:: 0.49.0

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
   SimplePointsWriter
   STLWriter
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
This is how :class:`pyvista.PVDReader` and :class:`pyvista.GIFReader`
are built, and it is the supported base for a third-party reader
registered with :func:`pyvista.register_reader`.

The remaining classes are not used directly. They are documented because they
define members shared by several readers and writers, so that each of those
members is documented once and linked from every class that inherits it.

.. autosummary::
   :toctree: _autosummary

   BaseReader
   BaseVTKReader
   core.utilities.writer._DataFormatMixin
   core.utilities.fileio._FileIOBase
   PointCellDataSelection
   PVDDataSet
   SeriesDataSet
   TimeReader
   core.utilities.writer._XMLWriter


Enumerations
~~~~~~~~~~~~

Enumerations are available to simplify inputs to certain readers.

.. toctree::
    :maxdepth: 2

    enums
