"""Contains a dictionary that maps file extensions to VTK readers."""

import pathlib
import os
import warnings

import numpy as np

import pyvista
from pyvista import _vtk

READERS = {
    # Standard dataset readers:
    '.vtk': _vtk.vtkDataSetReader,
    '.pvtk': _vtk.lazy_vtkPDataSetReader,
    '.vti': _vtk.vtkXMLImageDataReader,
    '.pvti': _vtk.vtkXMLPImageDataReader,
    '.vtr': _vtk.vtkXMLRectilinearGridReader,
    '.pvtr': _vtk.vtkXMLPRectilinearGridReader,
    '.vtu': _vtk.vtkXMLUnstructuredGridReader,
    '.pvtu': _vtk.vtkXMLPUnstructuredGridReader,
    '.ply': _vtk.vtkPLYReader,
    '.obj': _vtk.vtkOBJReader,
    '.stl': _vtk.vtkSTLReader,
    '.vtp': _vtk.vtkXMLPolyDataReader,
    '.vts': _vtk.vtkXMLStructuredGridReader,
    '.vtm': _vtk.vtkXMLMultiBlockDataReader,
    '.vtmb': _vtk.vtkXMLMultiBlockDataReader,
    '.case': _vtk.vtkGenericEnSightReader,
    # Image formats:
    '.bmp': _vtk.vtkBMPReader,
    '.dem': _vtk.vtkDEMReader,
    '.dcm': _vtk.vtkDICOMImageReader,
    '.img': _vtk.vtkDICOMImageReader,
    '.jpeg': _vtk.vtkJPEGReader,
    '.jpg': _vtk.vtkJPEGReader,
    '.mhd': _vtk.vtkMetaImageReader,
    '.nrrd': _vtk.vtkNrrdReader,
    '.nhdr': _vtk.vtkNrrdReader,
    '.png': _vtk.vtkPNGReader,
    '.pnm': _vtk.vtkPNMReader, # TODO: not tested
    '.slc': _vtk.vtkSLCReader,
    '.tiff': _vtk.vtkTIFFReader,
    '.tif': _vtk.vtkTIFFReader,
    # Other formats:
    '.byu': _vtk.vtkBYUReader, # TODO: not tested with this extension
    '.g': _vtk.vtkBYUReader,
    # '.chemml': _vtk.vtkCMLMoleculeReader, # TODO: not tested
    # '.cml': _vtk.vtkCMLMoleculeReader, # vtkMolecule is not supported by pyvista
    # TODO: '.csv': _vtk.vtkCSVReader, # vtkTables are currently not supported
    '.facet': _vtk.lazy_vtkFacetReader,
    '.cas': _vtk.vtkFLUENTReader, # TODO: not tested
    # '.dat': _vtk.vtkFLUENTReader, # TODO: not working
    # '.cube': _vtk.vtkGaussianCubeReader, # Contains `atom_types` which are note supported?
    '.res': _vtk.vtkMFIXReader, # TODO: not tested
    '.foam': _vtk.vtkOpenFOAMReader,
    # '.pdb': _vtk.vtkPDBReader, # Contains `atom_types` which are note supported?
    '.p3d': _vtk.lazy_vtkPlot3DMetaReader,
    '.pts': _vtk.vtkPTSReader,
    # '.particles': _vtk.vtkParticleReader, # TODO: not tested
    #TODO: '.pht': _vtk.vtkPhasta??????,
    #TODO: '.vpc': _vtk.vtkVPIC?????,
    # '.bin': _vtk.lazy_vtkMultiBlockPLOT3DReader,# TODO: non-default routine
    '.tri': _vtk.vtkMCubesReader,
    '.inp': _vtk.vtkAVSucdReader,
}

VTK_MAJOR = _vtk.vtkVersion().GetVTKMajorVersion()
VTK_MINOR = _vtk.vtkVersion().GetVTKMinorVersion()

if (VTK_MAJOR >= 8 and VTK_MINOR >= 2):
    try:
        READERS['.sgy'] = _vtk.lazy_vtkSegYReader
        READERS['.segy'] = _vtk.lazy_vtkSegYReader
    except AttributeError:
        pass


def _get_ext_force(filename, force_ext=None):
    if force_ext:
        return str(force_ext).lower()
    else:
        return get_ext(filename)


def get_ext(filename):
    """Extract the extension of the filename."""
    ext = os.path.splitext(filename)[1].lower()
    return ext


def get_reader(filename, force_ext=None):
    """Get the corresponding reader based on file extension and instantiates it."""
    ext = _get_ext_force(filename, force_ext=force_ext)
    return READERS[ext]()  # Get and instantiate the reader


def set_vtkwriter_mode(vtk_writer, use_binary=True):
    """Set any vtk writer to write as binary or ascii."""
    if isinstance(vtk_writer, (_vtk.vtkDataWriter, _vtk.vtkPLYWriter, _vtk.vtkSTLWriter)):
        if use_binary:
            vtk_writer.SetFileTypeToBinary()
        else:
            vtk_writer.SetFileTypeToASCII()
    elif isinstance(vtk_writer, _vtk.vtkXMLWriter):
        if use_binary:
            vtk_writer.SetDataModeToBinary()
        else:
            vtk_writer.SetDataModeToAscii()
    return vtk_writer


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
        A dictionary of attributes to call on the reader. Keys of
        dictionary are the attribute/method names and values are the
        arguments passed to those calls. If you do not have any
        attributes to call, pass ``None`` as the value.

    """
    observer = pyvista.utilities.errors.Observer()
    observer.observe(reader)

    if attrs is None:
        attrs = {}
    if not isinstance(attrs, dict):
        raise TypeError('Attributes must be a dictionary of name and arguments.')
    if filename is not None:
        try:
            reader.SetCaseFileName(filename)
        except AttributeError:
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

    # Check reader for errors
    if observer.has_event_occurred():
        warnings.warn(f'The VTK reader `{reader.GetClassName()}` raised an error while reading the file.\n'
                      f'\t"{observer.get_message()}"')

    data = pyvista.wrap(reader.GetOutputDataObject(0))
    data._post_file_load_processing()
    return data


def read_legacy(filename):
    """Use VTK's legacy reader to read a file."""
    reader = _vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    # Ensure all data is fetched with poorly formatted legacy files
    reader.ReadAllScalarsOn()
    reader.ReadAllColorScalarsOn()
    reader.ReadAllNormalsOn()
    reader.ReadAllTCoordsOn()
    reader.ReadAllVectorsOn()
    # Perform the read
    output = standard_reader_routine(reader, None)
    if output is None:
        raise RuntimeError('No output when using VTKs legacy reader')
    return output


def read(filename, attrs=None, force_ext=None, file_format=None):
    """Read any VTK file.

    It will figure out what reader to use then wrap the VTK object for
    use in PyVista.

    Parameters
    ----------
    filename : str
        The string path to the file to read. If a list of files is
        given, a :class:`pyvista.MultiBlock` dataset is returned with
        each file being a separate block in the dataset.

    attrs : dict, optional
        A dictionary of attributes to call on the reader. Keys of
        dictionary are the attribute/method names and values are the
        arguments passed to those calls. If you do not have any
        attributes to call, pass ``None`` as the value.

    force_ext: str, optional
        If specified, the reader will be chosen by an extension which
        is different to its actual extension. For example, ``'.vts'``,
        ``'.vtu'``.

    file_format : str, optional
        Format of file to read with meshio.

    Examples
    --------
    Load an example mesh

    >>> import pyvista
    >>> from pyvista import examples
    >>> mesh = pyvista.read(examples.antfile)

    Load a vtk file

    >>> mesh = pyvista.read('my_mesh.vtk')  # doctest:+SKIP

    Load a meshio file

    >>> mesh = pyvista.read("mesh.obj")  # doctest:+SKIP
    """
    if file_format is not None and force_ext is not None:
        raise ValueError('Only one of `file_format` and `force_ext` may be specified.')

    if isinstance(filename, (list, tuple)):
        multi = pyvista.MultiBlock()
        for each in filename:
            if isinstance(each, (str, pathlib.Path)):
                name = os.path.basename(str(each))
            else:
                name = None
            multi[-1, name] = read(each, attrs=attrs,
                                   file_format=file_format)
        return multi
    filename = os.path.abspath(os.path.expanduser(str(filename)))
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'File ({filename}) not found')

    ext = _get_ext_force(filename, force_ext)

    # Read file using meshio.read if file_format is present
    if file_format:
        return read_meshio(filename, file_format)

    # From the extension, decide which reader to use
    if attrs is not None:
        reader = get_reader(filename, force_ext=ext)
        return standard_reader_routine(reader, filename, attrs=attrs)
    elif ext in ['.e', '.exo']:
        return read_exodus(filename)
    elif ext in ['.vtk']:
        # Attempt to use the legacy reader...
        return read_legacy(filename)
    else:
        # Attempt find a reader in the readers mapping
        try:
            reader = get_reader(filename, force_ext=ext)
            return standard_reader_routine(reader, filename)
        except KeyError:
            # Don't fall back to meshio if using `force_ext`, which is really
            # just intended to be used with the native PyVista readers
            if force_ext is not None:
                from meshio._exceptions import ReadError
                raise ReadError
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
            raise ValueError("Problem reading the image with VTK.")
        return pyvista.Texture(image)
    except (KeyError, ValueError):
        # Otherwise, use the imageio reader
        pass
    import imageio
    return pyvista.Texture(imageio.imread(filename))


def read_exodus(filename,
                animate_mode_shapes=True,
                apply_displacements=True,
                displacement_magnitude=1.0,
                read_point_data=True,
                read_cell_data=True,
                enabled_sidesets=None):
    """Read an ExodusII file (``'.e'`` or ``'.exo'``).

    Parameters
    ----------
    filename : str
        The path to the exodus file to read.

    animate_mode_shapes : bool, optional
        When ``True`` then this reader will report a continuous time
        range [0,1] and animate the displacements in a periodic
        sinusoid.

    apply_displacements : bool, optional
        Geometric locations can include displacements. When ``True``,
        the nodal positions are 'displaced' by the standard exodus
        displacement vector. If displacements are turned off, the user
        can explicitly add them by applying a warp filter.

    displacement_magnitude : bool, optional
        This is a number between 0 and 1 that is used to scale the
        ``DisplacementMagnitude`` in a sinusoidal pattern.

    read_point_data : bool, optional
        Read in data associated with points.  Default ``True``.

    read_cell_data : bool, optional
        Read in data associated with cells.  Default ``True``.

    enabled_sidesets : str or int, optional
        The name of the array that store the mapping from side set
        cells back to the global id of the elements they bound.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> data = read_exodus('mymesh.exo')  # doctest:+SKIP

    """
    # lazy import here to avoid loading module on import pyvista
    try:
        from vtkmodules.vtkIOExodus import vtkExodusIIReader
    except ImportError:
        from vtk import vtkExodusIIReader

    reader = vtkExodusIIReader()
    reader.SetFileName(filename)
    reader.UpdateInformation()
    reader.SetAnimateModeShapes(animate_mode_shapes)
    reader.SetApplyDisplacements(apply_displacements)
    reader.SetDisplacementMagnitude(displacement_magnitude)

    if read_point_data:  # read in all point data variables
        reader.SetAllArrayStatus(vtkExodusIIReader.NODAL, 1)

    if read_cell_data:  # read in all cell data variables
        reader.SetAllArrayStatus(vtkExodusIIReader.ELEM_BLOCK, 1)

    if enabled_sidesets is None:
        enabled_sidesets = list(range(reader.GetNumberOfSideSetArrays()))

    for sideset in enabled_sidesets:
        if isinstance(sideset, int):
            name = reader.GetSideSetArrayName(sideset)
        elif isinstance(sideset, str):
            name = sideset
        else:
            raise ValueError(f'Could not parse sideset ID/name: {sideset}')

        reader.SetSideSetArrayStatus(name, 1)

    reader.Update()
    return pyvista.wrap(reader.GetOutput())


def read_plot3d(filename, q_filenames=(), auto_detect=True, attrs=None):
    """Read a Plot3D grid file (e.g., grid.in) and optional q file(s).

    Parameters
    ----------
    filename : str
        The string filename to the data file to read.

    q_filenames : str or tuple(str), optional
        The string filename of the q-file, or iterable of such
        filenames.

    auto_detect : bool, optional
        When this option is turned on, the reader will try to figure
        out the values of various options such as byte order, byte
        count etc. Default is ``True``.

    attrs : dict, optional
        A dictionary of attributes to call on the reader. Keys of
        dictionary are the attribute/method names and values are the
        arguments passed to those calls. If you do not have any
        attributes to call, pass ``None`` as the value.

    Returns
    -------
    mesh : pyvista.MultiBlock
        Data read from the file.

    """
    filename = _process_filename(filename)

    reader = _vtk.lazy_vtkMultiBlockPLOT3DReader()
    reader.SetFileName(filename)

    # q_filenames may be a list or a single filename
    if q_filenames:
        if isinstance(q_filenames, (str, pathlib.Path)):
            q_filenames = [q_filenames]
    q_filenames = [_process_filename(f) for f in q_filenames]

    if hasattr(reader, 'AddFileName'):
        # AddFileName was added to vtkMultiBlockPLOT3DReader sometime around
        # VTK 8.2. This method supports reading multiple q files.
        for q_filename in q_filenames:
            reader.AddFileName(q_filename)
    else:
        # SetQFileName is used to add a single q file to be read, and is still
        # supported in VTK9.
        if len(q_filenames) > 0:
            if len(q_filenames) > 1:
                raise RuntimeError('Reading of multiple q files is not supported '
                                   'with this version of VTK.')
            reader.SetQFileName(q_filenames[0])

    attrs = {} if not attrs else attrs
    attrs['SetAutoDetectFormat'] = auto_detect

    return standard_reader_routine(reader, filename=None, attrs=attrs)


def from_meshio(mesh):
    """Convert a ``meshio`` mesh instance to a PyVista mesh."""
    from meshio.vtk._vtk import (
        meshio_to_vtk_type,
        vtk_type_to_numnodes,
    )

    # Extract cells from meshio.Mesh object
    offset = []
    cells = []
    cell_type = []
    next_offset = 0
    for c in mesh.cells:
        vtk_type = meshio_to_vtk_type[c.type]
        numnodes = vtk_type_to_numnodes[vtk_type]
        cells.append(
            np.hstack((np.full((len(c.data), 1), numnodes), c.data)).ravel()
        )
        cell_type += [vtk_type] * len(c.data)
        if not _vtk.VTK9:
            offset += [next_offset + i * (numnodes + 1) for i in range(len(c.data))]
            next_offset = offset[-1] + numnodes + 1

    # Extract cell data from meshio.Mesh object
    cell_data = {k: np.concatenate(v) for k, v in mesh.cell_data.items()}

    # Create pyvista.UnstructuredGrid object
    points = mesh.points
    if points.shape[1] == 2:
        points = np.hstack((points, np.zeros((len(points), 1))))

    if _vtk.VTK9:
        grid = pyvista.UnstructuredGrid(
            np.concatenate(cells),
            np.array(cell_type),
            np.array(points, np.float64),
        )
    else:
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

    # Call datatype-specific post-load processing
    grid._post_file_load_processing()

    return grid


def read_meshio(filename, file_format=None):
    """Read any mesh file using meshio."""
    import meshio
    # Make sure relative paths will work
    filename = os.path.abspath(os.path.expanduser(str(filename)))
    # Read mesh file
    mesh = meshio.read(filename, file_format)
    return from_meshio(mesh)


def save_meshio(filename, mesh, file_format=None, **kwargs):
    """Save mesh to file using meshio.

    Parameters
    ----------
    mesh : pyvista.DataSet
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

    # Check that meshio supports all cell types in input mesh
    pixel_voxel = {8, 11}       # Handle pixels and voxels
    for cell_type in np.unique(vtk_cell_type):
        if cell_type not in vtk_to_meshio_type.keys() and cell_type not in pixel_voxel:
            raise TypeError(f"meshio does not support VTK type {cell_type}.")

    # Get cells
    cells = []
    c = 0
    for offset, cell_type in zip(vtk_offset, vtk_cell_type):
        numnodes = vtk_cells[offset+c]
        if _vtk.VTK9:  # must offset by cell count
            cell = vtk_cells[offset+1+c:offset+1+c+numnodes]
            c += 1
        else:
            cell = vtk_cells[offset+1:offset+1+numnodes]
        cell = (
            cell if cell_type not in pixel_voxel
            else cell[[0, 1, 3, 2]] if cell_type == 8
            else cell[[0, 1, 3, 2, 4, 5, 7, 6]]
        )
        cell_type = cell_type if cell_type not in pixel_voxel else cell_type+1
        cell_type = (
            vtk_to_meshio_type[cell_type] if cell_type != 7
            else f"polygon{numnodes}"
        )

        if len(cells) > 0 and cells[-1][0] == cell_type:
            cells[-1][1].append(cell)
        else:
            cells.append((cell_type, [cell]))

    for k, c in enumerate(cells):
        cells[k] = (c[0], np.array(c[1]))

    # Get point data
    point_data = {k.replace(" ", "_"): v for k, v in mesh.point_arrays.items()}

    # Get cell data
    vtk_cell_data = mesh.cell_arrays
    n_cells = np.cumsum([len(c[1]) for c in cells[:-1]])
    cell_data = (
        {k.replace(" ", "_"): np.split(v, n_cells) for k, v in vtk_cell_data.items()}
        if vtk_cell_data
        else {}
    )

    # Save using meshio
    meshio.write_points_cells(
        filename=filename,
        points=np.array(mesh.points),
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        file_format=file_format,
        **kwargs
    )

def _process_filename(filename):
    return os.path.abspath(os.path.expanduser(str(filename)))
