"""
Contains a dictionary that maps file extensions to VTK readers
"""
import os

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
    # '.tri': vtk.vtkMCubesReader, # TODO: Breaks pyvista wrapping on point arrays
}


if (vtk.vtkVersion().GetVTKMajorVersion() >= 8 and
    vtk.vtkVersion().GetVTKMinorVersion() >= 2):
    try:
        READERS['.sgy'] = vtk.vtkSegYReader
        READERS['.segy'] = vtk.vtkSegYReader
    except AttributeError:
        pass


def get_ext(filename):
    """Extract the extension of the filename"""
    ext = os.path.splitext(filename)[1].lower()
    return ext


def get_reader(filename):
    """Gets the corresponding reader based on file extension and instantiates it
    """
    ext = get_ext(filename)
    return READERS[ext]() # Get and instantiate the reader


def standard_reader_routine(reader, filename, attrs=None):
    """Use a given reader from the ``READERS`` mapping in the common VTK reading
    pipeline routine.

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
    """Use VTK's legacy reader to read a file"""
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    # Ensure all data is fetched with poorly formated legacy files
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


def read(filename, attrs=None):
    """This will read any VTK file! It will figure out what reader to use
    then wrap the VTK object for use in PyVista.

    Parameters
    ----------
    attrs : dict, optional
        A dictionary of attributes to call on the reader. Keys of dictionary are
        the attribute/method names and values are the arguments passed to those
        calls. If you do not have any attributes to call, pass ``None`` as the
        value.
    """
    filename = os.path.abspath(os.path.expanduser(filename))
    ext = get_ext(filename)

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
            pass
    raise IOError("This file was not able to be automatically read by pyvista.")


def read_texture(filename, attrs=None):
    """Loads a ``vtkTexture`` from an image file."""
    filename = os.path.abspath(os.path.expanduser(filename))
    try:
        # intitialize the reader using the extnesion to find it
        reader = get_reader(filename)
        image = standard_reader_routine(reader, filename, attrs=attrs)
        return pyvista.image_to_texture(image)
    except KeyError:
        # Otherwise, use the imageio reader
        pass
    return pyvista.numpy_to_texture(imageio.imread(filename))


def read_exodus(filename,
                animate_mode_shapes=True,
                apply_displacements=True,
                displacement_magnitude=1.0,
                enabled_sidesets=None):
    """Read an ExodusII file (``'.e'`` or ``'.exo'``)"""
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
