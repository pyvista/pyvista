"""Contains a dictionary that maps file extensions to VTK readers."""

import os
import pathlib
import warnings

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import PyVistaDeprecationWarning

from .observers import Observer


def set_pickle_format(format: str):
    """Set the format used to serialize :class:`pyvista.DataObject` when pickled.

    Parameters
    ----------
    format : str
        The format for serialization. Acceptable values are "xml" or "legacy".

    Raises
    ------
    ValueError
        If the provided format is not supported.

    """
    supported = {'xml', 'legacy'}
    format = format.lower()
    if format not in supported:
        raise ValueError(
            f'Unsupported pickle format `{format}`. Valid options are `{"`, `".join(supported)}`.'
        )
    pyvista.PICKLE_FORMAT = format


def _get_ext_force(filename, force_ext=None):
    if force_ext:
        return str(force_ext).lower()
    else:
        return get_ext(filename)


def get_ext(filename):
    """Extract the extension of the filename.

    For files with the .gz suffix, the previous extension is returned as well.
    This is needed e.g. for the compressed NIFTI format (.nii.gz).

    Parameters
    ----------
    filename : str
        The filename from which to extract the extension.

    Returns
    -------
    str
        The extracted extension. For files with the .gz suffix, the previous
        extension is returned as well.

    """
    base, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext == ".gz":
        ext_pre = os.path.splitext(base)[1].lower()
        ext = f"{ext_pre}{ext}"
    return ext


def set_vtkwriter_mode(vtk_writer, use_binary=True):
    """Set any vtk writer to write as binary or ascii.

    Parameters
    ----------
    vtk_writer : vtkDataWriter, vtkPLYWriter, vtkSTLWriter, or _vtk.vtkXMLWriter
        The vtk writer instance to be configured.
    use_binary : bool, default: True
        If ``True``, the writer is set to write files in binary format. If
        ``False``, the writer is set to write files in ASCII format.

    Returns
    -------
    vtkDataWriter, vtkPLYWriter, vtkSTLWriter, or _vtk.vtkXMLWriter
        The configured vtk writer instance.

    """
    from vtkmodules.vtkIOGeometry import vtkSTLWriter
    from vtkmodules.vtkIOLegacy import vtkDataWriter
    from vtkmodules.vtkIOPLY import vtkPLYWriter

    if isinstance(vtk_writer, (vtkDataWriter, vtkPLYWriter, vtkSTLWriter)):
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


def read(filename, force_ext=None, file_format=None, progress_bar=False):
    """Read any file type supported by ``vtk`` or ``meshio``.

    Automatically determines the correct reader to use then wraps the
    corresponding mesh as a pyvista object.  Attempts native ``vtk``
    readers first then tries to use ``meshio``.

    See :func:`pyvista.get_reader` for list of formats supported.

    .. note::
       See https://github.com/nschloe/meshio for formats supported by
       ``meshio``. Be sure to install ``meshio`` with ``pip install
       meshio`` if you wish to use it.

    Parameters
    ----------
    filename : str
        The string path to the file to read. If a list of files is
        given, a :class:`pyvista.MultiBlock` dataset is returned with
        each file being a separate block in the dataset.

    force_ext : str, optional
        If specified, the reader will be chosen by an extension which
        is different to its actual extension. For example, ``'.vts'``,
        ``'.vtu'``.

    file_format : str, optional
        Format of file to read with meshio.

    progress_bar : bool, default: False
        Optionally show a progress bar. Ignored when using ``meshio``.

    Returns
    -------
    pyvista.DataSet
        Wrapped PyVista dataset.

    Examples
    --------
    Load an example mesh.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = pv.read(examples.antfile)
    >>> mesh.plot(cpos='xz')

    Load a vtk file.

    >>> mesh = pv.read('my_mesh.vtk')  # doctest:+SKIP

    Load a meshio file.

    >>> mesh = pv.read("mesh.obj")  # doctest:+SKIP
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
            multi.append(read(each, file_format=file_format), name)
        return multi
    filename = os.path.abspath(os.path.expanduser(str(filename)))
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'File ({filename}) not found')

    # Read file using meshio.read if file_format is present
    if file_format:
        return read_meshio(filename, file_format)

    ext = _get_ext_force(filename, force_ext)
    if ext in ['.e', '.exo']:
        return read_exodus(filename)

    try:
        reader = pyvista.get_reader(filename, force_ext)
    except ValueError:
        # if using force_ext, we are explicitly only using vtk readers
        if force_ext is not None:
            raise OSError("This file was not able to be automatically read by pvista.")
        from meshio._exceptions import ReadError

        try:
            return read_meshio(filename)
        except ReadError:
            raise OSError("This file was not able to be automatically read by pyvista.")
    else:
        observer = Observer()
        observer.observe(reader.reader)
        if progress_bar:
            reader.show_progress()
        mesh = reader.read()
        if observer.has_event_occurred():
            warnings.warn(
                f"The VTK reader `{reader.reader.GetClassName()}` in pyvista reader `{reader}` raised an error"
                "while reading the file.\n"
                f'\t"{observer.get_message()}"'
            )
        return mesh


def _apply_attrs_to_reader(reader, attrs):
    """For a given pyvista reader, call methods according to attrs.

    Parameters
    ----------
    reader : pyvista.BaseReader
        Reader to call methods on.

    attrs : dict
        Mapping of methods to call on reader.

    """
    warnings.warn(
        "attrs use is deprecated.  Use a Reader class for more flexible control",
        PyVistaDeprecationWarning,
    )
    for name, args in attrs.items():
        attr = getattr(reader.reader, name)
        if args is not None:
            if not isinstance(args, (list, tuple)):
                args = [args]
            attr(*args)
        else:
            attr()


def read_texture(filename, progress_bar=False):
    """Load a texture from an image file.

    Will attempt to read any file type supported by ``vtk``, however
    if it fails, it will attempt to use ``imageio`` to read the file.

    Parameters
    ----------
    filename : str
        The path of the texture file to read.

    progress_bar : bool, default: False
        Optionally show a progress bar.

    Returns
    -------
    pyvista.Texture
        PyVista texture object.

    Examples
    --------
    Read in an example jpg map file as a texture.

    >>> import os
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> os.path.basename(examples.mapfile)
    '2k_earth_daymap.jpg'
    >>> texture = pv.read_texture(examples.mapfile)
    >>> type(texture)
    <class 'pyvista.plotting.texture.Texture'>

    """
    filename = os.path.abspath(os.path.expanduser(filename))
    try:
        # initialize the reader using the extension to find it

        image = read(filename, progress_bar=progress_bar)
        if image.n_points < 2:
            raise ValueError("Problem reading the image with VTK.")
        return pyvista.Texture(image)
    except (KeyError, ValueError):
        # Otherwise, use the imageio reader
        pass

    return pyvista.Texture(_try_imageio_imread(filename))  # pragma: no cover


def read_exodus(
    filename,
    animate_mode_shapes=True,
    apply_displacements=True,
    displacement_magnitude=1.0,
    read_point_data=True,
    read_cell_data=True,
    enabled_sidesets=None,
):
    """Read an ExodusII file (``'.e'`` or ``'.exo'``).

    Parameters
    ----------
    filename : str
        The path to the exodus file to read.

    animate_mode_shapes : bool, default: True
        When ``True`` then this reader will report a continuous time
        range [0,1] and animate the displacements in a periodic
        sinusoid.

    apply_displacements : bool, default: True
        Geometric locations can include displacements. When ``True``,
        the nodal positions are 'displaced' by the standard exodus
        displacement vector. If displacements are turned off, the user
        can explicitly add them by applying a warp filter.

    displacement_magnitude : bool, default: 1.0
        This is a number between 0 and 1 that is used to scale the
        ``DisplacementMagnitude`` in a sinusoidal pattern.

    read_point_data : bool, default: True
        Read in data associated with points.

    read_cell_data : bool, default: True
        Read in data associated with cells.

    enabled_sidesets : str | int, optional
        The name of the array that store the mapping from side set
        cells back to the global id of the elements they bound.

    Returns
    -------
    pyvista.DataSet
        Wrapped PyVista dataset.

    Examples
    --------
    >>> import pyvista as pv
    >>> data = pv.read_exodus('mymesh.exo')  # doctest:+SKIP

    """
    from .helpers import wrap

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
    return wrap(reader.GetOutput())


def is_meshio_mesh(obj):
    """Test if passed object is instance of ``meshio.Mesh``.

    Parameters
    ----------
    obj : object
        Any object.

    Returns
    -------
    bool
        ``True`` if ``obj`` is a ``meshio.Mesh``.

    """
    try:
        import meshio

        return isinstance(obj, meshio.Mesh)
    except ImportError:
        return False


def from_meshio(mesh):
    """Convert a ``meshio`` mesh instance to a PyVista mesh.

    Parameters
    ----------
    mesh : meshio.Mesh
        A mesh instance from the ``meshio`` library.

    Returns
    -------
    pyvista.UnstructuredGrid
        A PyVista unstructured grid representation of the input ``meshio`` mesh.

    Raises
    ------
    ImportError
        If the appropriate version of ``meshio`` library is not found.

    """
    try:  # meshio<5.0 compatibility
        from meshio.vtk._vtk import meshio_to_vtk_type, vtk_type_to_numnodes
    except ImportError:  # pragma: no cover
        from meshio._vtk_common import meshio_to_vtk_type
        from meshio.vtk._vtk_42 import vtk_type_to_numnodes

    # Extract cells from meshio.Mesh object
    cells = []
    cell_type = []
    for c in mesh.cells:
        vtk_type = meshio_to_vtk_type[c.type]
        numnodes = vtk_type_to_numnodes[vtk_type]
        fill_values = np.full((len(c.data), 1), numnodes, dtype=c.data.dtype)
        cells.append(np.hstack((fill_values, c.data)).ravel())
        cell_type += [vtk_type] * len(c.data)

    # Extract cell data from meshio.Mesh object
    cell_data = {k: np.concatenate(v) for k, v in mesh.cell_data.items()}

    # Create pyvista.UnstructuredGrid object
    points = mesh.points

    # convert to 3D if points are 2D
    if points.shape[1] == 2:
        zero_points = np.zeros((len(points), 1), dtype=points.dtype)
        points = np.hstack((points, zero_points))

    grid = pyvista.UnstructuredGrid(
        np.concatenate(cells).astype(np.int64, copy=False),
        np.array(cell_type),
        np.array(points, np.float64),
    )

    # Set point data
    grid.point_data.update({k: np.array(v, np.float64) for k, v in mesh.point_data.items()})

    # Set cell data
    grid.cell_data.update(cell_data)

    # Call datatype-specific post-load processing
    grid._post_file_load_processing()

    return grid


def read_meshio(filename, file_format=None):
    """Read any mesh file using meshio.

    Parameters
    ----------
    filename : str
        The name of the file to read. It should include the file extension.
    file_format : str, optional
        The format of the file to read. If not provided, the file format will
        be inferred from the file extension.

    Returns
    -------
    pyvista.Dataset
        The mesh read from the file.

    Raises
    ------
    ImportError
        If the meshio package is not installed.

    """
    try:
        import meshio
    except ImportError:  # pragma: no cover
        raise ImportError("To use this feature install meshio with:\n\npip install meshio")

    # Make sure relative paths will work
    filename = os.path.abspath(os.path.expanduser(str(filename)))
    # Read mesh file
    mesh = meshio.read(filename, file_format)
    return from_meshio(mesh)


def save_meshio(filename, mesh, file_format=None, **kwargs):
    """Save mesh to file using meshio.

    Parameters
    ----------
    filename : str
        Filename to save the mesh to.

    mesh : pyvista.DataSet
        Any PyVista mesh/spatial data type.

    file_format : str, optional
        File type for meshio to save.  For example ``'.bdf'``.  This
        is normally inferred from the extension but this can be
        overridden.

    **kwargs : dict, optional
        Additional keyword arguments.  See
        ``meshio.write_points_cells`` for more details.

    Examples
    --------
    Save a pyvista sphere to a Abaqus data file.

    >>> import pyvista as pv
    >>> sphere = pv.Sphere()
    >>> pv.save_meshio('mymesh.inp', sphere)  # doctest:+SKIP

    """
    try:
        import meshio
    except ImportError:  # pragma: no cover
        raise ImportError("To use this feature install meshio with:\n\npip install meshio")

    try:  # for meshio<5.0 compatibility
        from meshio.vtk._vtk import vtk_to_meshio_type
    except:  # noqa: E722 pragma: no cover
        from meshio._vtk_common import vtk_to_meshio_type

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
    pixel_voxel = {8, 11}  # Handle pixels and voxels
    for cell_type in np.unique(vtk_cell_type):
        if cell_type not in vtk_to_meshio_type.keys() and cell_type not in pixel_voxel:
            raise TypeError(f"meshio does not support VTK type {cell_type}.")

    # Get cells
    cells = []
    c = 0
    for offset, cell_type in zip(vtk_offset, vtk_cell_type):
        numnodes = vtk_cells[offset + c]
        cell = vtk_cells[offset + 1 + c : offset + 1 + c + numnodes]
        c += 1
        cell = (
            cell
            if cell_type not in pixel_voxel
            else cell[[0, 1, 3, 2]]
            if cell_type == 8
            else cell[[0, 1, 3, 2, 4, 5, 7, 6]]
        )
        cell_type = cell_type if cell_type not in pixel_voxel else cell_type + 1
        cell_type = vtk_to_meshio_type[cell_type] if cell_type != 7 else f"polygon{numnodes}"

        if len(cells) > 0 and cells[-1][0] == cell_type:
            cells[-1][1].append(cell)
        else:
            cells.append((cell_type, [cell]))

    for k, c in enumerate(cells):
        cells[k] = (c[0], np.array(c[1]))

    # Get point data
    point_data = {k.replace(" ", "_"): v for k, v in mesh.point_data.items()}

    # Get cell data
    vtk_cell_data = mesh.cell_data
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
        **kwargs,
    )


def _process_filename(filename):
    return os.path.abspath(os.path.expanduser(str(filename)))


def _try_imageio_imread(filename):
    """Attempt to read a file using ``imageio.imread``.

    Parameters
    ----------
    filename : str
        Name of the file to read using ``imageio``.

    Returns
    -------
    imageio.core.util.Array
        Image read from ``imageio``.

    Raises
    ------
    ModuleNotFoundError
        Raised when ``imageio`` is not installed when attempting to read
        ``filename``.

    """
    try:
        from imageio import imread
    except ModuleNotFoundError:  # pragma: no cover
        raise ModuleNotFoundError(
            'Problem reading the image with VTK. Install imageio to try to read the '
            'file using imageio with:\n\n'
            '   pip install imageio'
        ) from None

    return imread(filename)
