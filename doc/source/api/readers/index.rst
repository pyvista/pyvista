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
``s3://``, etc.), it downloads the file to a local temporary path
before reading. This uses ``fsspec`` when available (install with
``pip install pyvista[io]``), falling back to ``pooch`` for HTTP(S)
URIs. Readers registered with ``cloud=True`` receive the raw URI
and are expected to handle remote access themselves.


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
