"""Contains a dictionary that maps file extensions to VTK readers."""

import os

import numpy as np
import vtk

import pyvista


READERS = {
    # Standard dataset readers:
    '.vtk': vtk.vtkDataSetReader,
    '.pvtk': vtk.vtkPDataSetReader,
    '.vti': vtk.vtkXMLImageDataReader,
    '.pvti': vtk.vtkXMLPImageDataReader,
    '.vtr': vtk.vtkXMLRectilinearGridReader,
    '.pvtr': vtk.vtkXMLPRectilinearGridReader,
    '.vtu': vtk.vtkXMLUnstructuredGridReader,
    '.pvtu': vtk.vtkXMLPUnstructuredGridReader,
    '.ply': vtk.vtkPLYReader,
    '.obj': vtk.vtkOBJReader,
    '.stl': vtk.vtkSTLReader,
    '.vtp': vtk.vtkXMLPolyDataReader,
    '.vts': vtk.vtkXMLStructuredGridReader,
    '.vtm': vtk.vtkXMLMultiBlockDataReader,
    '.vtmb': vtk.vtkXMLMultiBlockDataReader,
    # Image formats:
    '.bmp': vtk.vtkBMPReader,
    '.dem': vtk.vtkDEMReader,
    '.dcm': vtk.vtkDICOMImageReader,
    '.img': vtk.vtkDICOMImageReader,
    '.jpeg': vtk.vtkJPEGReader,
    '.jpg': vtk.vtkJPEGReader,
    '.mhd': vtk.vtkMetaImageReader,
    '.png': vtk.vtkPNGReader,
    '.pnm': vtk.vtkPNMReader, # TODO: not tested
    '.slc': vtk.vtkSLCReader,
    '.tiff': vtk.vtkTIFFReader,
    '.tif': vtk.vtkTIFFReader,
    # Other formats:
    '.byu': vtk.vtkBYUReader, # TODO: not tested with this extension
    '.g': vtk.vtkBYUReader,
    # '.chemml': vtk.vtkCMLMoleculeReader, # TODO: not tested
    # '.cml': vtk.vtkCMLMoleculeReader, # vtkMolecule is not supported by pyvista
    # TODO: '.csv': vtk.vtkCSVReader, # vtkTables are currently not supported
    '.facet': vtk.vtkFacetReader,
    '.cas': vtk.vtkFLUENTReader, # TODO: not tested
    # '.dat': vtk.vtkFLUENTReader, # TODO: not working
    # '.cube': vtk.vtkGaussianCubeReader, # Contains `atom_types` which are note supported?
    '.res': vtk.vtkMFIXReader, # TODO: not tested
    '.foam': vtk.vtkOpenFOAMReader,
    # '.pdb': vtk.vtkPDBReader, # Contains `atom_types` which are note supported?
    '.p3d': vtk.vtkPlot3DMetaReader,
    '.pts': vtk.vtkPTSReader,
    # '.particles': vtk.vtkParticleReader, # TODO: not tested
    #TODO: '.pht': vtk.vtkPhasta??????,
    #TODO: '.vpc': vtk.vtkVPIC?????,
    # '.bin': vtk.vtkMultiBlockPLOT3DReader,# TODO: non-default routine
    '.tri': vtk.vtkMCubesReader,
    '.inp': vtk.vtkAVSucdReader,
}

VTK_MAJOR = vtk.vtkVersion().GetVTKMajorVersion()
VTK_MINOR = vtk.vtkVersion().GetVTKMinorVersion()

if (VTK_MAJOR >= 8 and VTK_MINOR >= 2):
    try:
        READERS['.sgy'] = vtk.vtkSegYReader
        READERS['.segy'] = vtk.vtkSegYReader
    except AttributeError:
        pass


def get_ext(filename):
    """Extract the extension of the filename."""
    ext = os.path.splitext(filename)[1].lower()
    return ext


def get_reader(filename):
    """Get the corresponding reader based on file extension and instantiates it."""
    ext = get_ext(filename)
    return READERS[ext]() # Get and instantiate the reader


def standard_reader_routine(reader, filename, attrs=None):
    """Use a given reader in the common VTK reading pipeline routine.

    The reader must come from the ``READERS`` mapping.

    Parameters
    ----------
    reader : vtkReader
        Any instantiated VTK reader class

    filename : str
        The string filename to the data file to read.

    attrs : dict, optional
        A dictionary of attributes to call on the reader. Keys of dictionary are
        the attribute/method names and values are the arguments passed to those
        calls. If you do not have any attributes to call, pass ``None`` as the
        value.

    """
    if attrs is None:
        attrs = {}
    if not isinstance(attrs, dict):
        raise TypeError('Attributes must be a dictionary of name and arguments.')
    reader.SetFileName(filename)
    # Apply any attributes listed
    for name, args in attrs.items():
        attr = getattr(reader, name)
        if args is not None:
            if not isinstance(args, (list, tuple)):
                args = [args]
            attr(*args)
        else:
            attr()
    # Perform the read
    reader.Update()
    return pyvista.wrap(reader.GetOutputDataObject(0))


def read_legacy(filename):
    """Use VTK's legacy reader to read a file."""
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    # Ensure all data is fetched with poorly formatted legacy files
    reader.ReadAllScalarsOn()
    reader.ReadAllColorScalarsOn()
    reader.ReadAllNormalsOn()
    reader.ReadAllTCoordsOn()
    reader.ReadAllVectorsOn()
    # Perform the read
    reader.Update()
    output = reader.GetOutputDataObject(0)
    if output is None:
        raise AssertionError('No output when using VTKs legacy reader')
    return pyvista.wrap(output)


def read(filename, attrs=None, file_format=None):
    """Read any VTK file.

    It will figure out what reader to use then wrap the VTK object for
    use in PyVista.

    Parameters
    ----------
    filename : str
        The string path to the file to read. If a list of files is given,
        a :class:`pyvista.MultiBlock` dataset is returned with each file being
        a seperate block in the dataset.
    attrs : dict, optional
        A dictionary of attributes to call on the reader. Keys of dictionary are
        the attribute/method names and values are the arguments passed to those
        calls. If you do not have any attributes to call, pass ``None`` as the
        value.
    file_format : str, optional
        Format of file to read with meshio.

    """
    if isinstance(filename, (list, tuple)):
        multi = pyvista.MultiBlock()
        for each in filename:
            if isinstance(each, str):
                name = os.path.basename(each)
            else:
                name = None
            multi[-1, name] = read(each)
        return multi
    filename = os.path.abspath(os.path.expanduser(filename))
    if not os.path.isfile(filename):
        raise FileNotFoundError('File ({}) not found'.format(filename))
    ext = get_ext(filename)

    # Read file using meshio.read if file_format is present
    if file_format:
        return read_meshio(filename, file_format)

    # From the extension, decide which reader to use
    if attrs is not None:
        reader = get_reader(filename)
        return standard_reader_routine(reader, filename, attrs=attrs)
    elif ext in '.vti': # ImageData
        return pyvista.UniformGrid(filename)
    elif ext in '.vtr': # RectilinearGrid
        return pyvista.RectilinearGrid(filename)
    elif ext in '.vtu': # UnstructuredGrid
        return pyvista.UnstructuredGrid(filename)
    elif ext in ['.ply', '.obj', '.stl']: # PolyData
        return pyvista.PolyData(filename)
    elif ext in '.vts': # StructuredGrid
        return pyvista.StructuredGrid(filename)
    elif ext in ['.vtm', '.vtmb']:
        return pyvista.MultiBlock(filename)
    elif ext in ['.e', '.exo']:
        return read_exodus(filename)
    elif ext in ['.vtk']:
        # Attempt to use the legacy reader...
        return read_legacy(filename)
    else:
        # Attempt find a reader in the readers mapping
        try:
            reader = get_reader(filename)
            return standard_reader_routine(reader, filename)
        except KeyError:
            # Attempt read with meshio
            try:
                from meshio._exceptions import ReadError
                try:
                    return read_meshio(filename)
                except ReadError:
                    pass
            except SyntaxError:
                # https://github.com/pyvista/pyvista/pull/495
                pass

    raise IOError("This file was not able to be automatically read by pyvista.")


def read_texture(filename, attrs=None):
    """Load a ``vtkTexture`` from an image file."""
    filename = os.path.abspath(os.path.expanduser(filename))
    try:
        # initialize the reader using the extension to find it
        reader = get_reader(filename)
        image = standard_reader_routine(reader, filename, attrs=attrs)
        if image.n_points < 2:
            raise AssertionError("Problem reading the image with VTK.")
        return pyvista.image_to_texture(image)
    except (KeyError, AssertionError):
        # Otherwise, use the imageio reader
        pass
    import imageio
    return pyvista.numpy_to_texture(imageio.imread(filename))


def read_exodus(filename,
                animate_mode_shapes=True,
                apply_displacements=True,
                displacement_magnitude=1.0,
                enabled_sidesets=None):
    """Read an ExodusII file (``'.e'`` or ``'.exo'``)."""
    reader = vtk.vtkExodusIIReader()
    reader.SetFileName(filename)
    reader.UpdateInformation()
    reader.SetAnimateModeShapes(animate_mode_shapes)
    reader.SetApplyDisplacements(apply_displacements)
    reader.SetDisplacementMagnitude(displacement_magnitude)

    if enabled_sidesets is None:
        enabled_sidesets = list(range(reader.GetNumberOfSideSetArrays()))

    for sideset in enabled_sidesets:
        if isinstance(sideset, int):
            name = reader.GetSideSetArrayName(sideset)
        elif isinstance(sideset, str):
            name = sideset
        else:
            raise ValueError('Could not parse sideset ID/name: {}'.format(sideset))

        reader.SetSideSetArrayStatus(name, 1)

    reader.Update()
    return pyvista.wrap(reader.GetOutput())


def read_meshio(filename, file_format = None):
    """Read any mesh file using meshio."""
    import meshio
    from meshio.vtk._vtk import (
        meshio_to_vtk_type,
        vtk_type_to_numnodes,
    )

    # Make sure relative paths will work
    filename = os.path.abspath(os.path.expanduser(str(filename)))

    # Read mesh file
    mesh = meshio.read(filename, file_format)

    # Extract cells from meshio.Mesh object
    offset = []
    cells = []
    cell_type = []
    cell_data = {}
    next_offset = 0
    for k, v in mesh.cells.items():
        vtk_type = meshio_to_vtk_type[k]
        numnodes = vtk_type_to_numnodes[vtk_type]
        offset += [next_offset+i*(numnodes+1) for i in range(len(v))]
        cells.append(np.hstack((np.full((len(v), 1), numnodes), v)).ravel())
        cell_type += [vtk_type] * len(v)
        next_offset = offset[-1] + numnodes + 1

        # Extract cell data
        if k in mesh.cell_data.keys():
            for kk, vv in mesh.cell_data[k].items():
                if kk in cell_data:
                    cell_data[kk] = np.concatenate((cell_data[kk], np.array(vv, np.float64)))
                else:
                    cell_data[kk] = np.array(vv, np.float64)

    # Create pyvista.UnstructuredGrid object
    points = mesh.points
    if points.shape[1] == 2:
        points = np.hstack((points, np.zeros((len(points),1))))

    grid = pyvista.UnstructuredGrid(
        np.array(offset),
        np.concatenate(cells),
        np.array(cell_type),
        np.array(points, np.float64),
    )

    # Set point data
    grid.point_arrays.update({k: np.array(v, np.float64) for k, v in mesh.point_data.items()})
    # Set cell data
    grid.cell_arrays.update(cell_data)

    return grid


def save_meshio(filename, mesh, file_format = None, **kwargs):
    """Save mesh to file using meshio.

    Parameters
    ----------
    mesh : pyvista.Common
        Any PyVista mesh/spatial data type.
    file_format : str
        File type for meshio to save.

    """
    import meshio
    from meshio.vtk._vtk import vtk_to_meshio_type

    # Make sure relative paths will work
    filename = os.path.abspath(os.path.expanduser(str(filename)))

    # Cast to pyvista.UnstructuredGrid
    if not isinstance(mesh, pyvista.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()

    # Copy useful arrays to avoid repeated calls to properties
    vtk_offset = mesh.offset
    vtk_cells = mesh.cells
    vtk_cell_type = mesh.celltypes

    # Get cells
    cells = {k: [] for k in np.unique(vtk_cell_type)}
    if 8 in cells.keys():
        cells[9] = cells.pop(8)                 # Handle pixels
    if 11 in cells.keys():
        cells[12] = cells.pop(11)               # Handle voxels
    mapper = {k: [] for k in cells.keys()}      # For cell data
    for i, (offset, cell_type) in enumerate(zip(vtk_offset, vtk_cell_type)):
        numnodes = vtk_cells[offset]
        cell = vtk_cells[offset+1:offset+1+numnodes]
        cell = cell if cell_type not in {8, 11} \
            else cell[[ 0, 1, 3, 2 ]] if cell_type == 8 \
            else cell[[ 0, 1, 3, 2, 4, 5, 7, 6 ]]
        cell_type = cell_type if cell_type not in {8, 11} else cell_type+1
        cells[cell_type].append(cell)
        mapper[cell_type].append(i)
    cells = {vtk_to_meshio_type[k]: np.vstack(v) for k, v in cells.items()}
    mapper = {vtk_to_meshio_type[k]: v for k, v in mapper.items()}

    # Get point data
    point_data = {k.replace(" ", "_"): v for k, v in mesh.point_arrays.items()}

    # Get cell data
    vtk_cell_data = mesh.cell_arrays
    cell_data = {
        k: {kk.replace(" ", "_"): vv[v] for kk, vv in vtk_cell_data.items()}
        for k, v in mapper.items()
    } if vtk_cell_data else {}

    # Save using meshio
    meshio.write_points_cells(
        filename = filename,
        points = np.array(mesh.points),
        cells = cells,
        point_data = point_data,
        cell_data = cell_data,
        file_format = file_format,
        **kwargs
    )
