.. _reader_api:

Readers
=======
PyVista provides class based readers to have more control over reading
data files.  These classes allows for more fine-grained control over
reading datasets from files.  See :func:`pyvista.get_reader` for a
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
   :template: custom-class-template.rst

    XMLImageDataReader
    XMLPImageDataReader
    XMLRectilinearGridReader
    XMLPRectilinearGridReader
    XMLUnstructuredGridReader
    XMLPUnstructuredGridReader
    XMLPolyDataReader
    XMLStructuredGridReader
    XMLMultiBlockDataReader
    PVDReader
    EnSightReader
    OpenFOAMReader
    PLYReader
    OBJReader
    STLReader
    VTKDataSetReader
    VTKPDataSetReader
    BYUReader
    FacetReader
    Plot3DMetaReader
    BinaryMarchingCubesReader


Inherited Classes
~~~~~~~~~~~~~~~~~

The :class:`pyvista.BaseReader` is inherited by all subreaders.  It
has the basic functionality of all readers to set filename and read
the data.

The :class:`PointCellDataSelection` is inherited by readers that
support inspecting and setting data related to point and cell arrays.

The :class:`TimeReader` is inherited by readers that support inspecting
and setting time or iterations for reading.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   BaseReader
   PointCellDataSelection
   TimeReader