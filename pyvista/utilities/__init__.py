"""Utilities routines."""
from . import transformations
from .algorithms import algorithm_to_mesh_handler, set_algorithm_input
from .common import perlin_noise, sample_function
from .errors import (
    GPUInfo,
    Observer,
    Report,
    VtkErrorCatcher,
    assert_empty_kwargs,
    check_valid_vector,
    get_gpu_info,
    send_errors_to_logging,
    set_error_output_file,
)
from .features import *
from .fileio import *
from .geometric_objects import *
from .helpers import *
from .parametric_objects import *
from .reader import (
    AVSucdReader,
    BaseReader,
    BinaryMarchingCubesReader,
    BMPReader,
    BYUReader,
    CGNSReader,
    DEMReader,
    DICOMReader,
    EnSightReader,
    FacetReader,
    FluentReader,
    GIFReader,
    GLTFReader,
    HDFReader,
    HDRReader,
    JPEGReader,
    MetaImageReader,
    MFIXReader,
    MultiBlockPlot3DReader,
    NIFTIReader,
    NRRDReader,
    OBJReader,
    OpenFOAMReader,
    Plot3DFunctionEnum,
    Plot3DMetaReader,
    PLYReader,
    PNGReader,
    PNMReader,
    PointCellDataSelection,
    POpenFOAMReader,
    PTSReader,
    PVDDataSet,
    PVDReader,
    SegYReader,
    SLCReader,
    STLReader,
    TecplotReader,
    TIFFReader,
    TimeReader,
    VTKDataSetReader,
    VTKPDataSetReader,
    XdmfReader,
    XMLImageDataReader,
    XMLMultiBlockDataReader,
    XMLPImageDataReader,
    XMLPolyDataReader,
    XMLPRectilinearGridReader,
    XMLPUnstructuredGridReader,
    XMLRectilinearGridReader,
    XMLStructuredGridReader,
    XMLUnstructuredGridReader,
    get_reader,
)
from .regression import compare_images
from .sphinx_gallery import Scraper, _get_sg_image_scraper
from .xvfb import start_xvfb
