"""
Contains a dictionary that maps file extensions to VTK readers
"""
import os

import vtk
import vtki


READERS = {
    # Standard dataset readers:
    '.vtk': vtk.vtkDataSetReader,
    '.vti': vtk.vtkXMLImageDataReader,
    '.vtr': vtk.vtkXMLRectilinearGridReader,
    '.vtu': vtk.vtkXMLUnstructuredGridReader,
    '.ply': vtk.vtkPLYReader,
    '.obj': vtk.vtkOBJReader,
    '.stl': vtk.vtkSTLReader,
    '.vts': vtk.vtkXMLStructuredGridReader,
    '.vtm': vtk.vtkXMLMultiBlockDataReader,
    '.vtmb': vtk.vtkXMLMultiBlockDataReader,
    # Image formats:
    '.bmp': vtk.vtkBMPReader,
    '.dem': vtk.vtkDEMReader,
    '.dcm': vtk.vtkDICOMImageReader,
    '.jpeg': vtk.vtkJPEGReader,
    '.jpg': vtk.vtkJPEGReader,
    '.png': vtk.vtkPNGReader,
    '.pnm': vtk.vtkPNMReader,
    '.slc': vtk.vtkSLCReader,
    '.tiff': vtk.vtkTIFFReader,
    '.tif': vtk.vtkTIFFReader,
}


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
    return vtki.wrap(reader.GetOutputDataObject(0))



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
    return reader.GetOutputDataObject(0)


def read(filename, attrs=None):
    """This will read any VTK file! It will figure out what reader to use
    then wrap the VTK object for use in ``vtki``.

    Parameters
    ----------
    attrs :
        A string list of reader attributes to call. If specified, the standard
        reading routine will be used with each attribute specified called.
    """
    filename = os.path.abspath(os.path.expanduser(filename))
    ext = get_ext(filename)

    # From the extension, decide which reader to use
    if attrs is not None:
        reader = get_reader(filename)
        return standard_reader_routine(reader, filename, attrs=attrs)
    elif ext in '.vti': # ImageData
        return vtki.UniformGrid(filename)
    elif ext in '.vtr': # RectilinearGrid
        return vtki.RectilinearGrid(filename)
    elif ext in '.vtu': # UnstructuredGrid
        return vtki.UnstructuredGrid(filename)
    elif ext in ['.ply', '.obj', '.stl']: # PolyData
        return vtki.PolyData(filename)
    elif ext in '.vts': # StructuredGrid
        return vtki.StructuredGrid(filename)
    elif ext in ['.vtm', '.vtmb']:
        return vtki.MultiBlock(filename)
    elif ext in ['.vtk']:
        # Attempt to use the legacy reader...
        output = vtki.wrap(read_legacy(filename))
        if output is None:
            raise AssertionError('No output when using VTKs legacy reader')
        return output
    else:
        # Attempt find a reader in the readers mapping
        try:
            reader = get_reader(filename)
            return standard_reader_routine(reader, filename)
        except KeyError:
            pass
    raise IOError("This file was not able to be automatically read by vtki.")


def load_texture(filename):
    """Loads a ``vtkTexture`` from an image file."""
    filename = os.path.abspath(os.path.expanduser(filename))
    try:
        # intitialize the reader using the extnesion to find it
        reader = get_reader(filename)
    except KeyError:
        # Otherwise, use the imageio reader
        return vtki.numpy_to_texture(imageio.imread(filename))
    reader.SetFileName(filename)
    reader.Update()
    return vtki.image_to_texture(reader.GetOutputDataObject(0))
