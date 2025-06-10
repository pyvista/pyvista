"""Utilities routines."""

from __future__ import annotations

import contextlib

from .arrays import FieldAssociation as FieldAssociation
from .arrays import array_from_vtkmatrix as array_from_vtkmatrix
from .arrays import cell_array as cell_array
from .arrays import convert_array as convert_array
from .arrays import convert_string_array as convert_string_array
from .arrays import field_array as field_array
from .arrays import get_array as get_array
from .arrays import get_array_association as get_array_association
from .arrays import get_vtk_type as get_vtk_type
from .arrays import parse_field_choice as parse_field_choice
from .arrays import point_array as point_array
from .arrays import raise_has_duplicates as raise_has_duplicates
from .arrays import raise_not_matching as raise_not_matching
from .arrays import row_array as row_array
from .arrays import set_default_active_scalars as set_default_active_scalars
from .arrays import set_default_active_vectors as set_default_active_vectors
from .arrays import vtk_bit_array_to_char as vtk_bit_array_to_char
from .arrays import vtk_id_list_to_array as vtk_id_list_to_array
from .arrays import vtkmatrix_from_array as vtkmatrix_from_array
from .cells import create_mixed_cells as create_mixed_cells
from .cells import get_mixed_cells as get_mixed_cells
from .cells import ncells_from_cells as ncells_from_cells
from .cells import numpy_to_idarr as numpy_to_idarr
from .features import cartesian_to_spherical as cartesian_to_spherical
from .features import create_grid as create_grid
from .features import grid_from_sph_coords as grid_from_sph_coords
from .features import merge as merge
from .features import perlin_noise as perlin_noise
from .features import sample_function as sample_function
from .features import spherical_to_cartesian as spherical_to_cartesian
from .features import transform_vectors_sph_to_cart as transform_vectors_sph_to_cart
from .features import voxelize as voxelize
from .features import voxelize_volume as voxelize_volume
from .fileio import from_meshio as from_meshio
from .fileio import get_ext as get_ext
from .fileio import is_meshio_mesh as is_meshio_mesh
from .fileio import read as read
from .fileio import read_exodus as read_exodus
from .fileio import read_grdecl as read_grdecl
from .fileio import read_meshio as read_meshio
from .fileio import read_pickle as read_pickle
from .fileio import read_texture as read_texture
from .fileio import save_meshio as save_meshio
from .fileio import save_pickle as save_pickle
from .fileio import set_pickle_format as set_pickle_format
from .fileio import set_vtkwriter_mode as set_vtkwriter_mode
from .fileio import to_meshio as to_meshio
from .geometric_objects import NORMALS as NORMALS
from .geometric_objects import Arrow as Arrow
from .geometric_objects import Box as Box
from .geometric_objects import Capsule as Capsule
from .geometric_objects import Circle as Circle
from .geometric_objects import CircularArc as CircularArc
from .geometric_objects import CircularArcFromNormal as CircularArcFromNormal
from .geometric_objects import Cone as Cone
from .geometric_objects import Cube as Cube
from .geometric_objects import Cylinder as Cylinder
from .geometric_objects import CylinderStructured as CylinderStructured
from .geometric_objects import Disc as Disc
from .geometric_objects import Dodecahedron as Dodecahedron
from .geometric_objects import Ellipse as Ellipse
from .geometric_objects import Icosahedron as Icosahedron
from .geometric_objects import Icosphere as Icosphere
from .geometric_objects import Line as Line
from .geometric_objects import MultipleLines as MultipleLines
from .geometric_objects import Octahedron as Octahedron
from .geometric_objects import Plane as Plane
from .geometric_objects import PlatonicSolid as PlatonicSolid
from .geometric_objects import Polygon as Polygon
from .geometric_objects import Pyramid as Pyramid
from .geometric_objects import Quadrilateral as Quadrilateral
from .geometric_objects import Rectangle as Rectangle
from .geometric_objects import SolidSphere as SolidSphere
from .geometric_objects import SolidSphereGeneric as SolidSphereGeneric
from .geometric_objects import Sphere as Sphere
from .geometric_objects import Superquadric as Superquadric
from .geometric_objects import Tetrahedron as Tetrahedron
from .geometric_objects import Text3D as Text3D
from .geometric_objects import Triangle as Triangle
from .geometric_objects import Tube as Tube
from .geometric_objects import Wavelet as Wavelet
from .geometric_sources import ArrowSource as ArrowSource
from .geometric_sources import AxesGeometrySource as AxesGeometrySource
from .geometric_sources import BoxSource as BoxSource
from .geometric_sources import ConeSource as ConeSource
from .geometric_sources import CubeFacesSource as CubeFacesSource
from .geometric_sources import CubeSource as CubeSource
from .geometric_sources import CylinderSource as CylinderSource
from .geometric_sources import DiscSource as DiscSource
from .geometric_sources import LineSource as LineSource
from .geometric_sources import MultipleLinesSource as MultipleLinesSource
from .geometric_sources import OrthogonalPlanesSource as OrthogonalPlanesSource
from .geometric_sources import PlaneSource as PlaneSource
from .geometric_sources import PlatonicSolidSource as PlatonicSolidSource
from .geometric_sources import PolygonSource as PolygonSource
from .geometric_sources import SphereSource as SphereSource
from .geometric_sources import SuperquadricSource as SuperquadricSource
from .geometric_sources import Text3DSource as Text3DSource
from .geometric_sources import translate as translate
from .image_sources import ImageEllipsoidSource as ImageEllipsoidSource
from .image_sources import ImageGaussianSource as ImageGaussianSource
from .image_sources import ImageGridSource as ImageGridSource
from .image_sources import ImageMandelbrotSource as ImageMandelbrotSource
from .image_sources import ImageNoiseSource as ImageNoiseSource
from .image_sources import ImageSinusoidSource as ImageSinusoidSource

with contextlib.suppress(ImportError):
    from .geometric_sources import CapsuleSource as CapsuleSource

from .cell_quality import cell_quality_info as cell_quality_info
from .helpers import axis_rotation
from .helpers import generate_plane
from .helpers import is_inside_bounds
from .helpers import is_pyvista_dataset
from .helpers import wrap
from .misc import AnnotatedIntEnum
from .misc import abstract_class
from .misc import assert_empty_kwargs
from .misc import check_valid_vector
from .misc import conditional_decorator
from .misc import has_module
from .misc import threaded
from .misc import try_callback
from .observers import Observer
from .observers import ProgressMonitor
from .observers import VtkErrorCatcher
from .observers import send_errors_to_logging
from .observers import set_error_output_file
from .parametric_objects import KochanekSpline
from .parametric_objects import ParametricBohemianDome
from .parametric_objects import ParametricBour
from .parametric_objects import ParametricBoy
from .parametric_objects import ParametricCatalanMinimal
from .parametric_objects import ParametricConicSpiral
from .parametric_objects import ParametricCrossCap
from .parametric_objects import ParametricDini
from .parametric_objects import ParametricEllipsoid
from .parametric_objects import ParametricEnneper
from .parametric_objects import ParametricFigure8Klein
from .parametric_objects import ParametricHenneberg
from .parametric_objects import ParametricKlein
from .parametric_objects import ParametricKuen
from .parametric_objects import ParametricMobius
from .parametric_objects import ParametricPluckerConoid
from .parametric_objects import ParametricPseudosphere
from .parametric_objects import ParametricRandomHills
from .parametric_objects import ParametricRoman
from .parametric_objects import ParametricSuperEllipsoid
from .parametric_objects import ParametricSuperToroid
from .parametric_objects import ParametricTorus
from .parametric_objects import Spline
from .parametric_objects import parametric_keywords
from .parametric_objects import surface_from_para
from .points import fit_line_to_points
from .points import fit_plane_to_points
from .points import line_segments_from_points
from .points import lines_from_points
from .points import make_tri_mesh
from .points import principal_axes
from .points import vector_poly_data
from .points import vtk_points
from .reader import AVSucdReader
from .reader import BaseReader
from .reader import BinaryMarchingCubesReader
from .reader import BMPReader
from .reader import BYUReader
from .reader import CGNSReader
from .reader import DEMReader
from .reader import DICOMReader
from .reader import EnSightReader
from .reader import ExodusIIBlockSet as ExodusIIBlockSet
from .reader import ExodusIIReader as ExodusIIReader
from .reader import FacetReader
from .reader import FLUENTCFFReader
from .reader import FluentReader
from .reader import GambitReader
from .reader import GaussianCubeReader
from .reader import GESignaReader
from .reader import GIFReader
from .reader import GLTFReader
from .reader import HDFReader
from .reader import HDRReader
from .reader import JPEGReader
from .reader import MetaImageReader
from .reader import MFIXReader
from .reader import MINCImageReader
from .reader import MultiBlockPlot3DReader
from .reader import Nek5000Reader as Nek5000Reader
from .reader import NIFTIReader
from .reader import NRRDReader
from .reader import OBJReader
from .reader import OpenFOAMReader
from .reader import ParticleReader
from .reader import PDBReader
from .reader import Plot3DFunctionEnum
from .reader import Plot3DMetaReader
from .reader import PLYReader
from .reader import PNGReader
from .reader import PNMReader
from .reader import PointCellDataSelection
from .reader import POpenFOAMReader
from .reader import ProStarReader
from .reader import PTSReader
from .reader import PVDDataSet
from .reader import PVDReader
from .reader import SegYReader
from .reader import SLCReader
from .reader import STLReader
from .reader import TecplotReader
from .reader import TIFFReader
from .reader import TimeReader
from .reader import VTKDataSetReader
from .reader import VTKPDataSetReader
from .reader import XdmfReader
from .reader import XGMLReader
from .reader import XMLImageDataReader
from .reader import XMLMultiBlockDataReader
from .reader import XMLPartitionedDataSetReader
from .reader import XMLPImageDataReader
from .reader import XMLPolyDataReader
from .reader import XMLPRectilinearGridReader
from .reader import XMLPUnstructuredGridReader
from .reader import XMLRectilinearGridReader
from .reader import XMLStructuredGridReader
from .reader import XMLUnstructuredGridReader
from .reader import get_reader
from .state_manager import vtk_snake_case as vtk_snake_case
from .state_manager import vtk_verbosity as vtk_verbosity
from .transform import Transform
