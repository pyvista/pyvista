"""File formats PyVista reads and writes, each with its name and file extensions."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class _FileFormat:
    """A file format's human-readable description and the file extensions it uses."""

    description: str
    extensions: tuple[str, ...] = ()
    # Display name and filename pattern, for formats whose extension encodes partitioning
    patterns: tuple[tuple[str, re.Pattern[str]], ...] = ()


AVS_UCD = _FileFormat('AVS UCD', ('.inp',))
BINARY_MARCHING_CUBES = _FileFormat('Binary Marching Cubes', ('.tri',))
BMP = _FileFormat('BMP Image', ('.bmp',))
BYU = _FileFormat('MOVIE.BYU', ('.g',))
CGNS = _FileFormat('CGNS', ('.cgns',))
DEM = _FileFormat('Digital Elevation Model', ('.dem',))
DICOM = _FileFormat('DICOM', ('.dcm', '.img'))
ECLIPSE_GRDECL = _FileFormat('Eclipse GRDECL', ('.grdecl',))
ENSIGHT = _FileFormat('EnSight', ('.case',))
EXODUS = _FileFormat('Exodus II', ('.e', '.ex2', '.exii', '.exo'))
EXODUS_PARTITIONED = _FileFormat(
    'Exodus II (partitioned)',
    # ``.e.N.p`` and ``.n.N.p``, for ``N`` partitions and partition index ``p``
    patterns=(
        ('.e.N.p', re.compile(r'\.e\.\d+\.\d+$', re.IGNORECASE)),
        ('.n.N.p', re.compile(r'\.n\.\d+\.\d+$', re.IGNORECASE)),
    ),
)
FACET = _FileFormat('Facet Polygonal Data', ('.facet',))
FLUENT = _FileFormat('Fluent Case', ('.cas',))
FLUENT_CFF = _FileFormat('Fluent CFF Case', ('.h5',))
GAMBIT = _FileFormat('GAMBIT Neutral', ('.neu',))
GAUSSIAN_CUBE = _FileFormat('Gaussian Cube', ('.cube',))
GE_SIGNA = _FileFormat('GE Signa MR', ('.mr',))
GIF = _FileFormat('GIF Image', ('.gif',))
GLTF = _FileFormat('glTF 2.0', ('.glb', '.gltf'))
HOUDINI = _FileFormat('Houdini Geometry', ('.geo',))
JPEG = _FileFormat('JPEG Image', ('.jpeg', '.jpg'))
LEGACY_VTK = _FileFormat('Legacy VTK', ('.vtk',))
LEGACY_VTK_PARTITIONED = _FileFormat('Legacy VTK (partitioned)', ('.pvtk',))
META_IMAGE = _FileFormat('MetaImage', ('.mha', '.mhd'))
MFIX = _FileFormat('MFIX', ('.res',))
MINC = _FileFormat('MINC Image', ('.mnc',))
NEK5000 = _FileFormat('Nek5000', ('.nek5000',))
NIFTI = _FileFormat('NIfTI', ('.nii', '.nii.gz'))
NRRD = _FileFormat('NRRD', ('.nhdr', '.nrrd'))
OBJ = _FileFormat('Wavefront OBJ', ('.obj',))
OPEN_INVENTOR = _FileFormat('Open Inventor', ('.iv',))
OPENFOAM = _FileFormat('OpenFOAM', ('.foam',))
PARAVIEW_DATA = _FileFormat('ParaView Data', ('.pvd',))
PARTICLE = _FileFormat('VTK Particle', ('.raw',))
PDB = _FileFormat('Protein Data Bank', ('.pdb',))
PLOT3D = _FileFormat('PLOT3D')
PLOT3D_META = _FileFormat('PLOT3D Meta', ('.p3d',))
PLY = _FileFormat('Stanford PLY', ('.ply',))
PNG = _FileFormat('PNG Image', ('.png',))
PNM = _FileFormat('PNM Image', ('.pnm',))
PROSTAR = _FileFormat('proSTAR', ('.vrt',))
PTS = _FileFormat('PTS Point Cloud', ('.pts',))
RADIANCE_HDR = _FileFormat('Radiance HDR', ('.hdr',))
SEGY = _FileFormat('SEG-Y', ('.segy', '.sgy'))
SLC = _FileFormat('SLC Volume', ('.slc',))
STL = _FileFormat('Stereolithography', ('.stl',))
TECPLOT = _FileFormat('Tecplot', ('.dat',))
THREE_DS = _FileFormat('3D Studio', ('.3ds',))
TIFF = _FileFormat('TIFF Image', ('.tif', '.tiff'))
VRML = _FileFormat('VRML 2.0', ('.vrml', '.wrl'))
VTK_IMAGE_DATA = _FileFormat('VTK Image Data', ('.vti',))
VTK_IMAGE_DATA_PARTITIONED = _FileFormat('VTK Image Data (partitioned)', ('.pvti',))
VTK_FILE_SERIES = _FileFormat('VTK File Series', ('.series',))
VTK_MULTIBLOCK = _FileFormat('VTK MultiBlock', ('.vtm', '.vtmb'))
VTK_PARTITIONED_DATASET = _FileFormat('VTK Partitioned Dataset', ('.vtpd',))
VTK_POLYDATA = _FileFormat('VTK PolyData', ('.vtp',))
VTK_RECTILINEAR_GRID = _FileFormat('VTK Rectilinear Grid', ('.vtr',))
VTK_RECTILINEAR_GRID_PARTITIONED = _FileFormat('VTK Rectilinear Grid (partitioned)', ('.pvtr',))
VTK_STRUCTURED_GRID = _FileFormat('VTK Structured Grid', ('.vts',))
VTK_UNSTRUCTURED_GRID = _FileFormat('VTK Unstructured Grid', ('.vtu',))
VTK_UNSTRUCTURED_GRID_PARTITIONED = _FileFormat('VTK Unstructured Grid (partitioned)', ('.pvtu',))
VTKHDF = _FileFormat('VTKHDF', ('.hdf', '.vtkhdf'))
XDMF = _FileFormat('XDMF', ('.xdmf',))
XYZ_POINTS = _FileFormat('XYZ Point Cloud', ('.xyz',))
