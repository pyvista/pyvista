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
