"""Contains a dictionary that maps file extensions to VTK readers."""

from __future__ import annotations

from collections.abc import Sequence
import itertools
from pathlib import Path
import pickle
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import TextIO
from typing import TypeVar
from typing import cast
from typing import overload
import warnings

import numpy as np

import pyvista
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import PyVistaDeprecationWarning

from .observers import Observer

if TYPE_CHECKING:
    from collections.abc import Iterable

    import imageio
    import meshio

    from pyvista.core._typing_core import VectorLike
    from pyvista.core.composite import MultiBlock
    from pyvista.core.dataobject import DataObject
    from pyvista.core.dataset import DataSet
    from pyvista.core.pointset import ExplicitStructuredGrid
    from pyvista.core.pointset import UnstructuredGrid
    from pyvista.core.utilities.reader import BaseReader
    from pyvista.plotting.texture import Texture

PathStrSeq = str | Path | Sequence['PathStrSeq']

if TYPE_CHECKING:
    _VTKWriterAlias = (
        _vtk.vtkXMLPartitionedDataSetWriter
        | _vtk.vtkXMLWriter
        | _vtk.vtkDataWriter
        | _vtk.vtkHDFWriter
    )
    _VTKWriterType = TypeVar('_VTKWriterType', bound=_VTKWriterAlias)

PICKLE_EXT = ('.pkl', '.pickle')


def set_pickle_format(format: Literal['vtk', 'xml', 'legacy']) -> None:  # noqa: A002
    """Set the format used to serialize :class:`pyvista.DataObject` when pickled.

    Parameters
    ----------
    format : str
        The format for serialization. Acceptable values are:

        - ``'vtk'`` (default) : objects are serialized using VTK's official
            marshalling methods.
        - ``'xml'``: objects are serialized as an XML-formatted string.
        - ``'legacy'`` objects are serialized to bytes in VTK's binary format.

        .. note::

            The ``'vtk'`` format requires VTK 9.3 or greater.

        .. warning::

            ``'xml'`` and ``'legacy'`` are not recommended. These formats are not
            officially supported by VTK and have limitations. For example, these
            formats cannot be used to pickle :class:`pyvista.MultiBlock`.

    Raises
    ------
    ValueError
        If the provided format is not supported.

    """
    supported = {'vtk', 'xml', 'legacy'}
    format_ = cast('Literal["vtk", "xml", "legacy"]', format.lower())
    if format_ not in supported:
        msg = (
            f'Unsupported pickle format `{format_}`. Valid options are `{"`, `".join(supported)}`.'
        )
        raise ValueError(msg)
    if format_ == 'vtk' and pyvista.vtk_version_info < (9, 3):
        msg = "'vtk' pickle format requires VTK >= 9.3"
        raise ValueError(msg)

    pyvista.PICKLE_FORMAT = format_


def _get_ext_force(filename: str | Path, force_ext: str | None = None) -> str:
    if force_ext:
        return str(force_ext).lower()
    else:
        return get_ext(filename)


def get_ext(filename: str | Path) -> str:
    """Extract the extension of the filename.

    For files with the .gz suffix, the previous extension is returned as well.
    This is needed e.g. for the compressed NIFTI format (.nii.gz).

    Parameters
    ----------
    filename : str, Path
        The filename from which to extract the extension.

    Returns
    -------
    str
        The extracted extension. For files with the .gz suffix, the previous
        extension is returned as well.

    """
    path = Path(filename)
    base = str(path.parent / path.stem)
    ext = path.suffix
    ext = ext.lower()
    if ext == '.gz':
        path = Path(base)
        ext_pre = path.suffix.lower()
        ext = f'{ext_pre}{ext}'
    return ext


@_deprecate_positional_args(allowed=['vtk_writer'])
def set_vtkwriter_mode(
    vtk_writer: _VTKWriterType,
    use_binary: bool = True,  # noqa: FBT001, FBT002
    compression: Literal['zlib', 'lz4', 'lzma', None] = 'zlib',
) -> _VTKWriterType:
    """Set any vtk writer to write as binary or ascii.

    Parameters
    ----------
    vtk_writer
        The vtk writer instance to be configured. Must be one of :vtk:`vtkDataWriter`,
        :vtk:`vtkPLYWriter`, :vtk:`vtkSTLWriter`, :vtk:`vtkXMLWriter`.
    use_binary : bool, default: True
        If ``True``, the writer is set to write files in binary format. If
        ``False``, the writer is set to write files in ASCII format.
    compression : str or None, default: 'zlib'
        The compression type to use when ``use_binary`` is ``True`` and ``vtk_writer``
        is of type :vtk:`vtkXMLWriter`. This argument has no effect otherwise.
        Acceptable values are ``'zlib'``, ``'lz4'``, ``'lzma'``, and ``None``.
        ``None`` indicates no compression.

    Returns
    -------
    :vtk:`vtkDataWriter` | :vtk:`vtkPLYWriter` | :vtk:`vtkSTLWriter` | :vtk:`vtkXMLWriter`
        The configured vtk writer instance.

    """
    from vtkmodules.vtkIOGeometry import vtkSTLWriter  # noqa: PLC0415
    from vtkmodules.vtkIOLegacy import vtkDataWriter  # noqa: PLC0415
    from vtkmodules.vtkIOPLY import vtkPLYWriter  # noqa: PLC0415

    if isinstance(vtk_writer, (vtkDataWriter, vtkPLYWriter, vtkSTLWriter)):
        if use_binary:
            vtk_writer.SetFileTypeToBinary()
        else:
            vtk_writer.SetFileTypeToASCII()
    elif isinstance(vtk_writer, _vtk.vtkXMLWriter):
        if use_binary:
            vtk_writer.SetDataModeToBinary()
            if compression is None:
                vtk_writer.SetCompressorTypeToNone()
            else:
                supported = {'zlib', 'lz4', 'lzma'}
                compression_ = cast('Literal["zlib", "lz4", "lzma"]', compression.lower())
                if compression_ not in supported:
                    supported_str = "', '".join(supported)
                    msg = (
                        f"Unsupported compression format '{compression_}'. "
                        f"Valid options are '{supported_str}', and `None`."
                    )
                    raise ValueError(msg)
                if compression_ == 'zlib':
                    vtk_writer.SetCompressorTypeToZLib()
                elif compression_ == 'lz4':
                    vtk_writer.SetCompressorTypeToLZ4()
                elif compression_ == 'lzma':
                    vtk_writer.SetCompressorTypeToLZMA()
        else:
            vtk_writer.SetDataModeToAscii()
    return vtk_writer


@_deprecate_positional_args(allowed=['filename'])
def read(  # noqa: PLR0911, PLR0917
    filename: PathStrSeq,
    force_ext: str | None = None,
    file_format: str | None = None,
    progress_bar: bool = False,  # noqa: FBT001, FBT002
) -> DataObject:
    """Read any file type supported by ``vtk`` or ``meshio``.

    Automatically determines the correct reader to use then wraps the
    corresponding mesh as a pyvista object.  Attempts native ``vtk``
    readers first then tries to use ``meshio``. :py:mod:`Pickled<pickle>`
    meshes (``'.pkl'`` or ``'.pickle'``) are also supported.

    See :func:`pyvista.get_reader` for list of vtk formats supported.

    .. note::
       See https://github.com/nschloe/meshio for formats supported by
       ``meshio``. Be sure to install ``meshio`` with ``pip install
       meshio`` if you wish to use it.

    .. versionadded:: 0.45

        Support reading pickled meshes.

    .. warning::

        The pickle module is not secure. Only read pickled mesh files
        (``'.pkl'`` or ``'.pickle'``) you trust. See :py:mod:`pickle`
        for details.

    See Also
    --------
    pyvista.DataObject.save
        Save a mesh to file.

    Parameters
    ----------
    filename : str, Path, Sequence[str | Path]
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

    >>> mesh = pv.read('mesh.obj')  # doctest:+SKIP

    Load a pickled mesh file.

    >>> mesh = pv.read('mesh.pkl')  # doctest:+SKIP

    """
    if file_format is not None and force_ext is not None:
        msg = 'Only one of `file_format` and `force_ext` may be specified.'
        raise ValueError(msg)

    if isinstance(filename, Sequence) and not isinstance(filename, str):
        multi = pyvista.MultiBlock()
        for each in filename:
            name = Path(each).name if isinstance(each, (str, Path)) else None
            multi.append(read(each, file_format=file_format), name)  # type: ignore[arg-type]
        return multi

    filename = Path(filename).expanduser().resolve()
    if not filename.is_file() and not filename.is_dir():
        msg = f'File ({filename}) not found'
        raise FileNotFoundError(msg)

    # Read file using meshio.read if file_format is present
    if file_format:
        return read_meshio(filename, file_format)

    ext = _get_ext_force(filename, force_ext)
    if ext in ['.e', '.exo']:
        return read_exodus(filename)
    if ext.lower() == '.grdecl':
        return read_grdecl(filename)
    if ext in ['.wrl', '.vrml']:
        msg = (
            'VRML files must be imported directly into a Plotter. '
            'See `pyvista.Plotter.import_vrml` for details.'
        )
        raise ValueError(msg)
    if ext in PICKLE_EXT:
        return read_pickle(filename)

    try:
        reader = pyvista.get_reader(filename, force_ext)
    except ValueError:
        # if using force_ext, we are explicitly only using vtk readers
        if force_ext is not None:
            msg = 'This file was not able to be automatically read by pyvista.'
            raise OSError(msg)
        from meshio._exceptions import ReadError  # noqa: PLC0415

        try:
            return read_meshio(filename)
        except ReadError:
            msg = 'This file was not able to be automatically read by pyvista.'
            raise OSError(msg)
    else:
        observer = Observer()
        observer.observe(reader.reader)
        if progress_bar:
            reader.show_progress()
        mesh = reader.read()
        if observer.has_event_occurred():
            warnings.warn(
                f'The VTK reader `{reader.reader.GetClassName()}` in pyvista reader `{reader}` '
                'raised an error while reading the file.\n'
                f'\t"{observer.get_message()}"',
            )
        return mesh


def _apply_attrs_to_reader(
    reader: BaseReader, attrs: dict[str, object | Sequence[object]]
) -> None:
    """For a given pyvista reader, call methods according to attrs.

    Parameters
    ----------
    reader : pyvista.BaseReader
        Reader to call methods on.

    attrs : dict
        Mapping of methods to call on reader.

    """
    warnings.warn(
        'attrs use is deprecated.  Use a Reader class for more flexible control',
        PyVistaDeprecationWarning,
    )
    for name, args in attrs.items():
        attr = getattr(reader.reader, name)
        if args is not None:
            if not isinstance(args, (list, tuple)):
                args = [args]  # noqa: PLW2901
            attr(*args)
        else:
            attr()


@_deprecate_positional_args(allowed=['filename'])
def read_texture(filename: str | Path, progress_bar: bool = False) -> Texture:  # noqa: FBT001, FBT002
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

    >>> from pathlib import Path
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> Path(examples.mapfile).name
    '2k_earth_daymap.jpg'
    >>> texture = pv.read_texture(examples.mapfile)
    >>> type(texture)
    <class 'pyvista.plotting.texture.Texture'>

    """
    filename = Path(filename).expanduser().resolve()
    try:
        # initialize the reader using the extension to find it

        image = read(filename, progress_bar=progress_bar)
        if image.n_points < 2:
            msg = 'Problem reading the image with VTK.'
            raise ValueError(msg)
        return pyvista.Texture(image)  # type: ignore[abstract]
    except (KeyError, ValueError):
        # Otherwise, use the imageio reader
        pass

    return pyvista.Texture(_try_imageio_imread(filename))  # type: ignore[abstract] # pragma: no cover


@_deprecate_positional_args(allowed=['filename'])
def read_exodus(  # noqa: PLR0917
    filename: str | Path,
    animate_mode_shapes: bool = True,  # noqa: FBT001, FBT002
    apply_displacements: bool = True,  # noqa: FBT001, FBT002
    displacement_magnitude: float = 1.0,
    read_point_data: bool = True,  # noqa: FBT001, FBT002
    read_cell_data: bool = True,  # noqa: FBT001, FBT002
    enabled_sidesets: Iterable[str | int] | None = None,
) -> DataSet | MultiBlock:
    """Read an ExodusII file (``'.e'`` or ``'.exo'``).

    Parameters
    ----------
    filename : str, Path
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

    displacement_magnitude : float, default: 1.0
        This is a number between 0 and 1 that is used to scale the
        ``DisplacementMagnitude`` in a sinusoidal pattern.

    read_point_data : bool, default: True
        Read in data associated with points.

    read_cell_data : bool, default: True
        Read in data associated with cells.

    enabled_sidesets : Iterable[str | int], optional
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
    from .helpers import wrap  # noqa: PLC0415

    # lazy import here to avoid loading module on import pyvista
    try:
        from vtkmodules.vtkIOExodus import vtkExodusIIReader  # noqa: PLC0415
    except ImportError:
        from vtk import vtkExodusIIReader  # type: ignore[no-redef]  # noqa: PLC0415

    reader = vtkExodusIIReader()
    reader.SetFileName(str(filename))
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
            msg = f'Could not parse sideset ID/name: {sideset} with type {type(sideset)}'  # type: ignore[unreachable]
            raise TypeError(msg)

        reader.SetSideSetArrayStatus(name, 1)

    reader.Update()
    return cast('pyvista.DataSet', wrap(reader.GetOutput()))


@_deprecate_positional_args(allowed=['filename'])
def read_grdecl(
    filename: str | Path,
    elevation: bool = True,  # noqa: FBT001, FBT002
    other_keywords: Sequence[str] | None = None,
) -> ExplicitStructuredGrid:
    """Read a GRDECL file (``'.GRDECL'``).

    Parameters
    ----------
    filename : str | Path
        The path to the GRDECL file to read.

    elevation : bool, default: True
        If True, convert depths to elevations and flip grid along Z axis.

    other_keywords : sequence[str], optional
        Additional keywords to read that are ignored by default.

    Returns
    -------
    pyvista.ExplicitStructuredGrid
        Output explicit structured grid.

    Examples
    --------
    Read a ``'.GRDECL'`` file.

    >>> import pyvista as pv
    >>> grid = pv.read('file.GRDECL')  # doctest:+SKIP

    Unused keywords contained in the file are stored in :attr:`pyvista.DataObject.user_dict`:

    >>> grid.user_dict  # doctest:+SKIP
    {"MAPUNITS": ..., "GRIDUNIT": ..., ...}

    """
    property_keywords = (
        'ACTNUM',
        'COORD',
        'ZCORN',
        'PERMEABILITY',
        'PERMX',
        'PERMY',
        'PERMZ',
        'POROSITY',
        'PORO',
        'LAYERS',
        'ZONES',
    )

    @overload
    def read_keyword(
        f: TextIO,
        split: Literal[True] = True,  # noqa: FBT002
        converter: type = ...,
    ) -> list[str]: ...
    @overload
    def read_keyword(f: TextIO, split: Literal[False] = False, converter: type = ...) -> str: ...  # noqa: FBT002
    @overload
    def read_keyword(f: TextIO, split: bool = ..., converter: type = ...) -> list[str]: ...  # noqa: FBT001
    @_deprecate_positional_args(allowed=['f'])
    def read_keyword(
        f: TextIO,
        split: bool = True,  # noqa: FBT001, FBT002
        converter: type | None = None,
    ) -> str | list[str]:
        """Read a keyword.

        Parameters
        ----------
        f : TextIO
            File buffer.

        split : bool, default: True
            If True, split strings.

        converter : callable, optional
            Function to apply to split strings.

        Returns
        -------
        list | str
            A list or a string.

        """
        out: list[str] = []

        while True:
            line = f.readline().strip()

            if line.endswith('/'):
                line = line[:-1].strip()

                if line:
                    end = True

                else:
                    break

            elif line.startswith('--') or not line:
                continue

            else:
                end = False

            if split:
                line_ = line.split()
                out += [converter(x) for x in line_] if converter is not None else line_

            else:
                out.append(line)

            if end:
                break

        if not split:
            return ' '.join(out)

        return out

    def read_buffer(
        f: TextIO, other_keywords: Sequence[str]
    ) -> tuple[dict[str, Any], Sequence[str]]:
        """Read a file buffer.

        Parameters
        ----------
        f : TextIO
            File buffer.

        other_keywords : sequence[str], optional
            Additional keywords to read that are ignored by default.

        Returns
        -------
        dict
            Dictionary of read keywords.

        sequence[str]
            Included file names.

        """
        keys: Sequence[str] = list(property_keywords) + list(other_keywords)
        keys = tuple(keys)

        keywords: dict[str, str | list[Any]] = {}
        includes = []

        for line in f:
            line_ = line.strip()

            if line_.startswith('MAPUNITS'):
                keywords['MAPUNITS'] = read_keyword(f, split=False).replace("'", '').strip()

            elif line_.startswith('MAPAXES'):
                keywords['MAPAXES'] = read_keyword(f, converter=float)

            elif line_.startswith('GRIDUNIT'):
                keywords['GRIDUNIT'] = read_keyword(f, split=False).replace("'", '').strip()

            elif line_.startswith('SPECGRID'):
                data = read_keyword(f)
                keywords['SPECGRID'] = [
                    int(data[0]),
                    int(data[1]),
                    int(data[2]),
                    int(data[3]),
                    data[4].strip(),
                ]

            elif line_.startswith('INCLUDE'):
                filename = read_keyword(f, split=False)
                includes.append(filename.replace("'", ''))

            elif line_.startswith(keys):
                key = line.split()[0]
                data = read_keyword(f)

                if key in property_keywords:
                    keywords[key] = []

                    for x in data:
                        if '*' in x:
                            size, new_x = x.split('*')
                            keywords[key] += int(size) * [float(new_x)]  # type: ignore[operator]

                        else:
                            keywords[key].append(float(x))  # type: ignore[union-attr]

                else:
                    keywords[key] = data

        return keywords, includes

    def read_keywords(filename: str | Path, other_keywords: Sequence[str]) -> dict[str, Any]:
        """Read a GRDECL file and return its keywords.

        Parameters
        ----------
        filename : str | Path
            The path to the GRDECL file to read.

        other_keywords : sequence[str], optional
            Additional keywords to read that are ignored by default.

        Returns
        -------
        dict
            Dictionary of read keywords.

        """
        with Path.open(Path(filename)) as f:
            keywords, includes = read_buffer(f, other_keywords)

        if includes:
            path = Path(filename).parent

            for include in includes:
                with Path.open(path / include) as f:
                    keywords_, _ = read_buffer(f, other_keywords)

                keywords.update(keywords_)

        return keywords

    # Read keywords
    other_keywords = other_keywords or []
    keywords = read_keywords(filename, other_keywords)

    try:
        ni, nj, nk = keywords['SPECGRID'][:3]
        cylindric = keywords['SPECGRID'][4] == 'T'

        if cylindric:
            msg = 'Cylindric grids are not supported.'
            raise TypeError(msg)

    except KeyError:
        msg = "Unable to generate grid without keyword 'SPECGRID'."
        raise ValueError(msg)

    relative = False

    if 'GRIDUNIT' in keywords:
        grid_unit = keywords['GRIDUNIT'].lower()

        if not grid_unit.endswith('map'):
            try:
                cond1 = grid_unit.startswith(keywords['MAPUNITS'].lower())

                if not cond1:
                    warnings.warn(
                        'Unable to convert relative coordinates with different '
                        'grid and map units. Skipping conversion.'
                    )

            except KeyError:
                warnings.warn(
                    "Unable to convert relative coordinates without keyword 'MAPUNITS'. "
                    'Skipping conversion.'
                )
                cond1 = False

            try:
                origin = keywords['MAPAXES'][2:4]

            except KeyError:
                warnings.warn(
                    "Unable to convert relative coordinates without keyword 'MAPAXES'. "
                    'Skipping conversion.'
                )
                origin = None

            relative = cond1 and origin is not None

    # Pillars and Z corner points
    pillars = np.reshape(keywords['COORD'], ((ni + 1) * (nj + 1), 6), order='C')
    zcorners = np.reshape(keywords['ZCORN'], (2 * ni, 2 * nj, 2 * nk), order='F')

    # Convert depth to elevation
    if elevation:
        zcorners = -zcorners[..., ::-1]
        pillars[:, [2, 5]] *= -1.0
        pillars[:] = pillars[:, [3, 4, 5, 0, 1, 2]]

    # Shift relative to absolute units
    if relative:
        pillars[:, [0, 3]] += origin[0]
        pillars[:, [1, 4]] += origin[1]

    # Interpolate X and Y corner points
    xcorners = np.empty_like(zcorners)
    ycorners = np.empty_like(zcorners)

    for i, j in itertools.product(range(2 * ni), range(2 * nj)):
        ip = np.ravel_multi_index(((i + 1) // 2, (j + 1) // 2), (ni + 1, nj + 1), order='F')
        z = pillars[ip, [2, 5]]
        xcorners[i, j] = np.interp(zcorners[i, j], z, pillars[ip, [0, 3]])
        ycorners[i, j] = np.interp(zcorners[i, j], z, pillars[ip, [1, 4]])

    # Generate explicit structured grid
    dims = ni + 1, nj + 1, nk + 1
    corners = np.column_stack(
        (
            xcorners.ravel(order='F'),
            ycorners.ravel(order='F'),
            zcorners.ravel(order='F'),
        )
    )
    grid = pyvista.ExplicitStructuredGrid(dims, corners)

    # Add property data
    for key in property_keywords:
        if key in {'ACTNUM', 'COORD', 'ZCORN'}:
            continue

        if key in keywords:
            v = keywords[key]

            if elevation:
                v = np.reshape(v, (ni, nj, nk), order='F')
                v = v[..., ::-1].ravel(order='F')

            grid[key] = v

    # Active cells
    if 'ACTNUM' in keywords:
        active = np.array(keywords['ACTNUM']) > 0.0
        grid.hide_cells(~active, inplace=True)  # type: ignore[arg-type]

    # Store unused keywords in user dict
    grid.user_dict = {k: v for k, v in keywords.items() if k not in property_keywords}

    return grid


def read_pickle(filename: str | Path) -> DataObject:
    """Load a pickled mesh from file.

    Parameters
    ----------
    filename : str
        The path of the pickled mesh to read.

    Returns
    -------
    pyvista.DataObject
        Unpickled mesh.

    Examples
    --------
    Save a pickled mesh and read it.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.load_ant()
    >>> pv.save_pickle('ant.pkl', mesh)
    >>> new_mesh = pv.read_pickle('ant.pkl')
    >>> new_mesh
    PolyData (...)
      N Cells:    912
      N Points:   486
      N Strips:   0
      X Bounds:   -1.601e+01, 1.601e+01
      Y Bounds:   -9.385e+00, 9.385e+00
      Z Bounds:   -1.678e+01, 1.678e+01
      N Arrays:   0

    Unlike other file formats, custom attributes are saved with pickled meshes.

    >>> pv.set_new_attribute(mesh, 'custom_attribute', 42)
    >>> pv.save_pickle('ant.pkl', mesh)
    >>> new_mesh = pv.read_pickle('ant.pkl')
    >>> new_mesh.custom_attribute
    42

    """
    filename_str = str(filename)
    if filename_str.endswith(PICKLE_EXT):
        with open(filename_str, 'rb') as f:  # noqa: PTH123
            mesh = pickle.load(f)

        if not isinstance(mesh, pyvista.DataObject):
            msg = (
                f'Pickled object must be an instance of {pyvista.DataObject}. '
                f'Got {mesh.__class__} instead.'
            )
            raise TypeError(msg)
        return mesh
    msg = f'Filename must be a file path with extension {PICKLE_EXT}. Got {filename} instead.'
    raise ValueError(msg)


def save_pickle(filename: str | Path, mesh: DataObject) -> None:
    """Pickle a mesh and save it to file.

    Parameters
    ----------
    filename : str
        The path of the pickled mesh to save, including the extension ``'.pkl'``
        or ``'.pickle'``.

    mesh : pyvista.DataObject
        Any PyVista mesh.

    Examples
    --------
    Save a pickled mesh and read it.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.load_ant()
    >>> pv.save_pickle('ant.pkl', mesh)
    >>> new_mesh = pv.read_pickle('ant.pkl')
    >>> new_mesh
    PolyData (...)
      N Cells:    912
      N Points:   486
      N Strips:   0
      X Bounds:   -1.601e+01, 1.601e+01
      Y Bounds:   -9.385e+00, 9.385e+00
      Z Bounds:   -1.678e+01, 1.678e+01
      N Arrays:   0

    Unlike other file formats, custom attributes are saved with pickled meshes.

    >>> pv.set_new_attribute(mesh, 'custom_attribute', 42)
    >>> pv.save_pickle('ant.pkl', mesh)
    >>> new_mesh = pv.read_pickle('ant.pkl')
    >>> new_mesh.custom_attribute
    42

    """
    filename_str = str(filename)
    if not filename_str.endswith(PICKLE_EXT):
        filename_str += '.pkl'
    if not isinstance(mesh, pyvista.DataObject):
        msg = (  # type: ignore[unreachable]
            f'Only {pyvista.DataObject} are supported for pickling. Got {mesh.__class__} instead.'
        )
        raise TypeError(msg)

    with open(filename_str, 'wb') as f:  # noqa: PTH123
        pickle.dump(mesh, f)


def is_meshio_mesh(obj: object) -> bool:
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
        import meshio  # noqa: PLC0415

        return isinstance(obj, meshio.Mesh)
    except ImportError:
        return False


def from_meshio(mesh: meshio.Mesh) -> UnstructuredGrid:
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
        from meshio.vtk._vtk import meshio_to_vtk_type  # noqa: PLC0415
        from meshio.vtk._vtk import vtk_type_to_numnodes  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from meshio._vtk_common import meshio_to_vtk_type  # noqa: PLC0415
        from meshio.vtk._vtk_42 import vtk_type_to_numnodes  # noqa: PLC0415

    if len(mesh.cells) == 0:
        # Empty mesh
        grid = pyvista.UnstructuredGrid()
        if mesh.points.size > 0:
            grid.points = mesh.points
        return grid

    # Extract cells from meshio.Mesh object
    cells = []
    cell_type = []
    for c in mesh.cells:
        if c.type.startswith('polyhedron'):
            vtk_type = meshio_to_vtk_type['polyhedron']

            for cell in c.data:
                connectivity = [len(cell)]
                for face in cell:
                    connectivity += [len(face), *face]

                connectivity.insert(0, len(connectivity))
                cells.append(connectivity)

        else:
            vtk_type = meshio_to_vtk_type[c.type]
            numnodes = vtk_type_to_numnodes[vtk_type]
            if numnodes == -1:
                # Count nodes in each cell
                fill_values = np.array([[len(data)] for data in c.data], dtype=c.data.dtype)
            else:
                fill_values = np.full((len(c.data), 1), numnodes, dtype=c.data.dtype)
            cells.append(np.hstack((fill_values, c.data)).ravel())  # type: ignore[arg-type]

        cell_type += [vtk_type] * len(c.data)

    # Convert cell sets to cell data
    if mesh.cell_sets:
        mesh.cell_sets_to_data()

    # Extract cell data from meshio.Mesh object
    cell_data = {k: np.concatenate(v) for k, v in mesh.cell_data.items()}

    # Create pyvista.UnstructuredGrid object
    points = mesh.points

    # Convert to 3D if points are 2D
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


def to_meshio(mesh: DataSet) -> meshio.Mesh:
    """Convert a PyVista mesh to a ``meshio`` mesh instance.

    .. versionadded:: 0.45

    Parameters
    ----------
    mesh : pyvista.DataSet
        Any PyVista mesh/spatial data type.

    Returns
    -------
    meshio.Mesh
        A mesh instance from the ``meshio`` library.

    Raises
    ------
    ImportError
        If the meshio package is not installed.

    See Also
    --------
    from_meshio, read_meshio, save_meshio, :func:`~pyvista.wrap`

    Examples
    --------
    Convert a pyvista sphere to a ``meshio`` mesh instance.

    >>> import pyvista as pv
    >>> sphere = pv.Sphere()
    >>> mesh = pv.to_meshio(sphere)

    """
    try:
        import meshio  # noqa: PLC0415

    except ImportError:  # pragma: no cover
        msg = 'To use this feature install meshio with:\n\npip install meshio'
        raise ImportError(msg)

    try:  # for meshio<5.0 compatibility
        from meshio.vtk._vtk import vtk_to_meshio_type  # noqa: PLC0415

    except (ImportError, AttributeError):  # pragma: no cover
        from meshio._vtk_common import vtk_to_meshio_type  # noqa: PLC0415

    # Cast to unstructured grid
    mesh = mesh.cast_to_unstructured_grid()
    mesh = (
        mesh.extract_cells(mesh.cell_data['vtkGhostType'] == 0)
        if 'vtkGhostType' in mesh.cell_data
        else mesh
    )
    if mesh.is_empty:
        return meshio.Mesh(mesh.points, [])

    vtk_celltypes = mesh.celltypes
    connectivity = mesh.cell_connectivity

    # Generate polyhedral cell faces if any
    def split(arr: VectorLike[int]) -> list[VectorLike[int]]:
        i = 0
        offsets: list[int] = [0]

        while i < len(arr):
            offsets.append(int(arr[i]) + 1)
            i += offsets[-1]

        offsets_ = np.cumsum(offsets)

        return [arr[i1 + 1 : i2] for i1, i2 in itertools.pairwise(offsets_)]

    polyhedron_faces = split(mesh.polyhedron_faces)

    if polyhedron_faces:
        polyhedron_locations = split(mesh.polyhedron_face_locations)
        polyhedral_cell_faces: list[list[VectorLike[int]]] = [
            [polyhedron_faces[face] for face in cell] for cell in polyhedron_locations
        ]

    # Single cell type (except POLYGON and POLYHEDRON)
    if vtk_celltypes.min() == vtk_celltypes.max() and vtk_celltypes[0] not in {
        pyvista.CellType.POLYGON,
        pyvista.CellType.POLYHEDRON,
    }:
        vtk_celltype = vtk_celltypes[0]
        cells = connectivity.reshape((mesh.n_cells, connectivity.size // mesh.n_cells))

        if vtk_celltype == pyvista.CellType.PIXEL:
            cells = cells[:, [0, 1, 3, 2]]
            celltype = 'quad'

        elif vtk_celltype == pyvista.CellType.VOXEL:
            cells = cells[:, [0, 1, 3, 2, 4, 5, 7, 6]]
            celltype = 'hexahedron'

        else:
            celltype = vtk_to_meshio_type[vtk_celltype]

        cells = [(celltype, cells)]

    # Mixed cell types
    else:
        cells = []
        offset = mesh.offset

        for i, (i1, i2, vtk_celltype) in enumerate(zip(offset[:-1], offset[1:], vtk_celltypes)):
            cell = connectivity[i1:i2]

            if vtk_celltype == pyvista.CellType.POLYHEDRON:
                celltype = f'polyhedron{len(cell)}'
                cell = polyhedral_cell_faces[i]

            # Handle the missing voxel key (11) in vtk_to_meshio_type
            elif vtk_celltype == pyvista.CellType.VOXEL:
                celltype = 'hexahedron'
                cell = cell[[0, 1, 3, 2, 4, 5, 7, 6]]

            # Handle the missing "pixel" key in meshio._mesh.topological_dimension
            elif vtk_celltype == pyvista.CellType.PIXEL:
                celltype = 'quad'
                cell = cell[[0, 1, 3, 2]]

            else:
                celltype = (
                    f'polygon{len(cell)}'
                    if vtk_celltype == pyvista.CellType.POLYGON
                    else vtk_to_meshio_type[vtk_celltype]
                )

            if len(cells) > 0 and cells[-1][0] == celltype:
                cells[-1][1].append(cell)

            else:
                cells.append((celltype, [cell]))

        cells = [
            (celltype if not celltype.startswith('polygon') else 'polygon', celldata)
            for celltype, celldata in cells
        ]

    # Point data
    point_data = {k.replace(' ', '_'): v for k, v in mesh.point_data.items()}

    # Cell data
    vtk_cell_data = mesh.cell_data
    indices = np.insert(np.cumsum([len(c[1]) for c in cells]), 0, 0)
    cell_data = {
        k.replace(' ', '_'): [v[i1:i2] for i1, i2 in itertools.pairwise(indices)]
        for k, v in vtk_cell_data.items()
    }

    return meshio.Mesh(mesh.points, cells, point_data=point_data, cell_data=cell_data)


def read_meshio(filename: str | Path, file_format: str | None = None) -> meshio.Mesh:
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
    pyvista.DataSet
        The mesh read from the file.

    Raises
    ------
    ImportError
        If the meshio package is not installed.

    """
    try:
        import meshio  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        msg = 'To use this feature install meshio with:\n\npip install meshio'
        raise ImportError(msg)

    # Make sure relative paths will work
    filename = Path(filename).expanduser().resolve()
    # Read mesh file
    mesh = meshio.read(filename, file_format)
    return from_meshio(mesh)


def save_meshio(
    filename: str | Path, mesh: DataSet, file_format: str | None = None, **kwargs
) -> None:
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
        ``meshio.Mesh.write`` for more details.

    Examples
    --------
    Save a pyvista sphere to a Abaqus data file.

    >>> import pyvista as pv
    >>> sphere = pv.Sphere()
    >>> pv.save_meshio('mymesh.inp', sphere)  # doctest:+SKIP

    """
    # Make sure relative paths will work
    filename = Path(filename).expanduser().resolve()

    # Save using meshio
    to_meshio(mesh).write(filename, file_format=file_format, **kwargs)


def _process_filename(filename: str | Path) -> Path:
    return Path(filename).expanduser().resolve()


def _try_imageio_imread(filename: str | Path) -> imageio.core.util.Array:
    """Attempt to read a file using ``imageio.imread``.

    Parameters
    ----------
    filename : str, Path
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
        from imageio.v2 import imread  # noqa: PLC0415
    except ModuleNotFoundError:  # pragma: no cover
        msg = (
            'Problem reading the image with VTK. Install imageio to try to read the '
            'file using imageio with:\n\n'
            '   pip install imageio'
        )
        raise ModuleNotFoundError(msg) from None

    return imread(filename)
