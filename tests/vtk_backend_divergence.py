"""Reasons a test diverges on an alternative VTK build.

Defined once and imported at each site, so when the fork's module set changes a
single edit covers every test rather than a grep for repeated string literals.
Each is used with the ``skip_vtk_backend`` marker::

    @pytest.mark.skip_vtk_backend('cvista', reason=NO_POPENFOAM_READER)
    def test_openfoam_patch_arrays(): ...
"""

from __future__ import annotations

# Modules the cvista build does not ship. Each names the VTK module and the class
# the test needs from it, so the reason is actionable without opening the test.
NO_POPENFOAM_READER = 'cvista does not ship vtkIOParallel (vtkPOpenFOAMReader)'
NO_PEXODUS_READER = 'cvista does not ship vtkIOParallelExodus (vtkPExodusIIReader)'
NO_NEK5000_READER = 'cvista does not ship vtkIOParallel (vtkNek5000Reader)'
NO_PLOT3D_READER = 'cvista does not ship vtkIOParallel (vtkMultiBlockPLOT3DReader)'
NO_PLOT3D_META_READER = 'cvista does not ship vtkIOParallel (vtkPlot3DMetaReader)'
NO_CGNS_READER = 'cvista does not ship vtkIOCGNSReader (vtkCGNSReader)'
NO_DELIMITED_TEXT_READER = 'cvista does not ship vtkIOInfovis (vtkDelimitedTextReader)'
NO_XDMF2 = 'cvista does not ship vtkIOXdmf2'
NO_ENSIGHT_WRITER = 'cvista does not ship vtkEnSightWriter'
NO_XML_PARTITIONED_WRITER = (
    'cvista does not ship vtkIOParallelXML (vtkXMLPartitionedDataSetWriter)'
)

# Behaviour that differs by design rather than by omission.
NO_SNAKE_CASE = 'cvista omits the VTK snake_case wrapper API'
TRIMMED_CLASS_SET = 'cvista wraps a trimmed VTK class set'
TRIMMED_MODULE_SET = 'cvista wraps a trimmed VTK module set'
CVISTA_NAMESPACE = 'module loads under the cvista namespace'
INT32_CELL_STORAGE = 'cvista stores connectivity as int32 (no zero-copy share)'
INT32_COMPRESSION = 'cvista stores indices as int32 (smaller, less compressible)'
CELL_STATUS_ENUM = 'cvista diverges on vtkCellStatus enum exposure'
FIXED_SIZE_CELL_STORAGE = (
    'cvista uses fixed-size cell storage for uniform cells where stock VTK does not'
)
