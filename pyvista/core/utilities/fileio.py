"""Contains a dictionary that maps file extensions to VTK readers."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
import importlib
import itertools
import json
from pathlib import Path
import pickle
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import TextIO
from typing import cast
from typing import overload

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista._warn_external import warn_external
from pyvista.core import _validation
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.misc import _classproperty
from pyvista.core.utilities.misc import _NoNewAttrMixin

from .observers import Observer

if TYPE_CHECKING:
    from collections.abc import Iterable

    import imageio
    import meshio
    import trimesh
    from vtk import vtkWriter

    from pyvista import BaseReader
    from pyvista import DataObject
    from pyvista import DataSet
    from pyvista import ExplicitStructuredGrid
    from pyvista import MultiBlock
    from pyvista import NumpyArray
    from pyvista import PolyData
    from pyvista import Texture
    from pyvista import UnstructuredGrid
    from pyvista import VectorLike

_CompressionOptions = Literal['zlib', 'lz4', 'lzma', None]  # noqa: PYI061
PathStrSeq = str | Path | Sequence['PathStrSeq']
PICKLE_EXT = ('.pkl', '.pickle')
_PointCellField = Literal['point', 'cell', 'field']
_PassDataOptions = bool | _PointCellField | Sequence[_PointCellField]


def _lazy_vtk_import(module_name: str, class_name: str) -> type:
    """Lazy import of a class from vtkmodules."""
    module = importlib.import_module(f'vtkmodules.{module_name}')
    return getattr(module, class_name)


class _FileIOBase(ABC, _NoNewAttrMixin):
    _vtk_module_name: str = ''
    _vtk_class_name: str = ''

    def __repr__(self) -> str:
        """Representation of a FileIO object."""
        return f'{self.__class__.__name__}({self.path!r})'

    @property
    @abstractmethod
    def path(self) -> str:
        """Get the path."""

    @path.setter
    @abstractmethod
    def path(self, path: str | Path) -> None:
        """Set the path."""

    @_classproperty
    def _vtk_class(cls) -> vtkWriter | None:  # noqa: N805
        if cls._vtk_module_name and cls._vtk_class_name:
            return _lazy_vtk_import(cls._vtk_module_name, cls._vtk_class_name)  # type: ignore[return-value]
        return None

    @classmethod
    @abstractmethod
    def _get_extension_mappings(cls) -> list[dict[str, type]]: ...

    @_classproperty
    def extensions(cls) -> tuple[str, ...]:  # noqa: N805
        """Return the file extension(s) associated with this class.

        These extensions are used by :func:`~pyvista.read` and :class:`~pyvista.DataObject.save`
        to determine which reader and/or writer is used for reading and/or saving files.

        """
        extensions = set()
        for mapping in cls._get_extension_mappings():
            for ext, typ in mapping.items():
                if typ is cls:  # type: ignore[comparison-overlap]
                    extensions.add(ext)
        return tuple(sorted(extensions))


def _warn_multiblock_nested_field_data(mesh: pv.DataObject) -> None:
    if not isinstance(mesh, pv.MultiBlock):
        return
    iterator = mesh.recursive_iterator('all', node_type='parent')
    for index, name, nested_multiblock in iterator:
        if len(nested_multiblock.field_data.keys()) > 0:
            # Avoid circular import
            from pyvista.core.filters.composite import _format_nested_index  # noqa: PLC0415

            index_fmt = _format_nested_index(index)
            msg = (
                f"Nested MultiBlock at index {index_fmt} with name '{name}' "
                f'has field data which will not be saved.\n'
                'See https://gitlab.kitware.com/vtk/vtk/-/issues/19414 \n'
                'Use `move_nested_field_data_to_root` to store the field data '
                'with the root MultiBlock before saving.'
            )
            warn_external(msg)


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
    if format_ == 'vtk' and pv.vtk_version_info < (9, 3):
        msg = "'vtk' pickle format requires VTK >= 9.3"
        raise ValueError(msg)

    pv.PICKLE_FORMAT = format_


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

    .. include:: /api/utilities/mesh_io.rst

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
        multi = pv.MultiBlock()
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
        reader = pv.get_reader(filename, force_ext)
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
            warn_external(
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
    warn_external(
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
        return pv.Texture(image)  # type: ignore[abstract]
    except (KeyError, ValueError):
        # Otherwise, use the imageio reader
        pass

    return pv.Texture(_try_imageio_imread(filename))  # type: ignore[abstract] # pragma: no cover


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
        from vtk import vtkExodusIIReader  # noqa: PLC0415

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
    return cast('pv.DataSet', wrap(reader.GetOutput()))


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
        output : list | str
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
                    warn_external(
                        'Unable to convert relative coordinates with different '
                        'grid and map units. Skipping conversion.',
                    )

            except KeyError:
                warn_external(
                    "Unable to convert relative coordinates without keyword 'MAPUNITS'. "
                    'Skipping conversion.',
                )
                cond1 = False

            try:
                origin = keywords['MAPAXES'][2:4]

            except KeyError:
                warn_external(
                    "Unable to convert relative coordinates without keyword 'MAPAXES'. "
                    'Skipping conversion.',
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
    grid = pv.ExplicitStructuredGrid(dims, corners)

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

        if not isinstance(mesh, pv.DataObject):
            msg = (
                f'Pickled object must be an instance of {pv.DataObject}. '
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
    if not isinstance(mesh, pv.DataObject):
        msg = (  # type: ignore[unreachable]
            f'Only {pv.DataObject} are supported for pickling. Got {mesh.__class__} instead.'
        )
        raise TypeError(msg)
    _warn_multiblock_nested_field_data(mesh)
    with open(filename_str, 'wb') as f:  # noqa: PTH123
        pickle.dump(mesh, f)


def is_meshio_mesh(obj: object) -> bool:
    """Test if passed object is an instance of :class:`meshio.Mesh`.

    Parameters
    ----------
    obj : object
        Any object.

    Returns
    -------
    bool
        ``True`` if ``obj`` is a :class:`meshio.Mesh`.

    """
    try:
        import meshio  # noqa: PLC0415

        return isinstance(obj, meshio.Mesh)
    except ImportError:
        return False


def is_trimesh_mesh(obj: object) -> bool:
    """Test if passed object is an instance of :class:`trimesh.Trimesh`.

    .. versionadded:: 0.47

    Parameters
    ----------
    obj : object
        Any object.

    Returns
    -------
    bool
        ``True`` if ``obj`` is a :class:`trimesh.Trimesh`.

    """
    try:
        import trimesh  # noqa: PLC0415

        return isinstance(obj, trimesh.Trimesh)
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
        grid = pv.UnstructuredGrid()
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
                fill_values = np.array([[len(data)] for data in c.data], dtype=c.data.dtype)  # type: ignore[union-attr]
            else:
                fill_values = np.full((len(c.data), 1), numnodes, dtype=c.data.dtype)  # type: ignore[union-attr]
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

    grid = pv.UnstructuredGrid(
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
    mesh = mesh if isinstance(mesh, pv.UnstructuredGrid) else mesh.cast_to_unstructured_grid()
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
        pv.CellType.POLYGON,
        pv.CellType.POLYHEDRON,
    }:
        vtk_celltype = vtk_celltypes[0]
        cells = connectivity.reshape((mesh.n_cells, connectivity.size // mesh.n_cells))

        if vtk_celltype == pv.CellType.PIXEL:
            cells = cells[:, [0, 1, 3, 2]]
            celltype = 'quad'

        elif vtk_celltype == pv.CellType.VOXEL:
            cells = cells[:, [0, 1, 3, 2, 4, 5, 7, 6]]
            celltype = 'hexahedron'

        else:
            celltype = vtk_to_meshio_type[vtk_celltype]

        cells = [(celltype, cells)]

    # Mixed cell types
    else:
        cells = []
        offset = mesh.offset

        for i, (i1, i2, vtk_celltype) in enumerate(
            zip(offset[:-1], offset[1:], vtk_celltypes, strict=False)
        ):
            cell = connectivity[i1:i2]

            if vtk_celltype == pv.CellType.POLYHEDRON:
                celltype = f'polyhedron{len(cell)}'
                cell = polyhedral_cell_faces[i]

            # Handle the missing voxel key (11) in vtk_to_meshio_type
            elif vtk_celltype == pv.CellType.VOXEL:
                celltype = 'hexahedron'
                cell = cell[[0, 1, 3, 2, 4, 5, 7, 6]]

            # Handle the missing "pixel" key in meshio._mesh.topological_dimension
            elif vtk_celltype == pv.CellType.PIXEL:
                celltype = 'quad'
                cell = cell[[0, 1, 3, 2]]

            else:
                celltype = (
                    f'polygon{len(cell)}'
                    if vtk_celltype == pv.CellType.POLYGON
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


def read_meshio(filename: str | Path, file_format: str | None = None) -> UnstructuredGrid:
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
    UnstructuredGrid
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


def _validate_pass_data(pass_data: _PassDataOptions) -> tuple[bool, bool, bool]:
    pass_point_data = pass_cell_data = pass_field_data = False
    if pass_data is True:
        pass_point_data = pass_cell_data = pass_field_data = True
    elif pass_data:
        if isinstance(pass_data, str):
            if pass_data == 'point':
                pass_point_data = True
            elif pass_data == 'cell':
                pass_cell_data = True
            elif pass_data == 'field':
                pass_field_data = True
        elif isinstance(pass_data, Sequence):
            if 'point' in pass_data:
                pass_point_data = True
            if 'cell' in pass_data:
                pass_cell_data = True
            if 'field' in pass_data:
                pass_field_data = True

    if not (pass_point_data or pass_cell_data or pass_field_data) and pass_data is not False:
        # Input is not valid
        allowed = [True, False, 'point', 'cell', 'field']
        _validation.check_contains(allowed, must_contain=pass_data, name='pass_data')

    return pass_point_data, pass_cell_data, pass_field_data


def from_trimesh(
    mesh: trimesh.Trimesh, *, pass_data: _PassDataOptions = True
) -> PolyData:  # numpydoc ignore=RT01
    """Convert a Trimesh mesh to a PyVista mesh.

    - ``vertex_attributes`` are stored as point data.
    - ``face_attributes`` are stored as cell data.
    - ``metadata`` is stored as field data: NumPy arrays are stored directly as field data
      arrays, and any other metadata (e.g. strings or lists) is stored in the
      :attr:`~pyvista.DataObject.user_dict`.

    .. note::

        No copies of point, cell, or data arrays are made. Use :meth:`~pyvista.DataObject.copy`
        after converting to avoid any side effects.

    .. versionadded:: 0.47

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Trimesh object to convert.

    pass_data : bool | str | sequence[str], default: True
        Pass point, cell, and/or field data from the Trimesh object. All data is passed by default.
        Set this to ``'point'``, ``'cell'``, ``'field'`` or any combination thereof to only pass
        specific fields.

    See Also
    --------
    to_trimesh, from_meshio, :func:`~pyvista.wrap`

    Examples
    --------
    See :ref:`wrap_trimesh_example` for examples.

    """
    try:
        import trimesh  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        msg = 'To use this feature install trimesh with:\n\npip install trimesh'
        raise ImportError(msg)
    else:
        _validation.check_instance(mesh, trimesh.Trimesh, name='mesh')

    # Handle case with no faces
    faces: NumpyArray[int] = mesh.faces
    if faces.size == 0:
        faces = faces.reshape((0, 3))
    # Trimesh doesn't pad faces
    polydata = pv.PolyData.from_regular_faces(mesh.vertices, faces=faces, deep=False)

    pass_point_data, pass_cell_data, pass_field_data = _validate_pass_data(pass_data)

    if pass_point_data:
        # Set texture coordinates
        if (
            hasattr(visual := mesh.visual, 'uv')
            and visual is not None
            and (uv := visual.uv) is not None
        ):
            polydata.active_texture_coordinates = uv
        polydata.point_data.update(mesh.vertex_attributes, copy=False)

    if pass_cell_data:
        polydata.cell_data.update(mesh.face_attributes, copy=False)

    if pass_field_data:
        for key, val in mesh.metadata.items():
            if isinstance(val, np.ndarray):
                polydata.field_data[key] = val
            else:
                try:
                    json.dumps(val)
                except TypeError:
                    msg = (
                        f'Unable to store metadata key {key!r} with value type {type(val)}.\n'
                        f'Only NumPy arrays or JSON-serializable values are supported.'
                    )
                    warn_external(msg)
                else:
                    polydata.user_dict[key] = val

    return polydata


def to_trimesh(  # numpydoc ignore=RT01
    mesh: DataSet,
    *,
    triangulate: bool = False,
    pass_data: _PassDataOptions = True,
) -> trimesh.Trimesh:
    """Convert a PyVista mesh to a Trimesh mesh.

    - Point data is stored as ``vertex_attributes``.
    - Cell data is stored as ``face_attributes``.
    - Field data is stored as ``metadata``. Any :attr:`~pyvista.DataObject.user_dict` keys
      are stored directly as metadata.

    .. note::

        No copies of point, cell, or data arrays are made. Use :meth:`~pyvista.DataObject.copy`
        before converting to avoid any side effects.

    .. versionadded:: 0.47

    Parameters
    ----------
    mesh : DataSet
        Dataset to convert.

    triangulate : bool, default: False
        Triangulate the mesh before conversion. If the mesh has 3D cells, the mesh's surface
        is extracted. All 2D polygonal cells are triangulated as required, and all 0D and 1D
        cells or any unused points are ignored.

    pass_data : bool | str | sequence[str], default: True
        Pass point, cell, and/or field data to the Trimesh object. All data is passed by default.
        Set this to ``'point'``, ``'cell'``, ``'field'`` or any combination thereof to only pass
        specific fields.

    See Also
    --------
    from_trimesh, to_meshio, :func:`~pyvista.wrap`

    Examples
    --------
    See :ref:`wrap_trimesh_example` for examples.

    """
    try:
        import trimesh  # noqa: PLC0415
        from trimesh.visual import TextureVisuals  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        msg = 'To use this feature install trimesh with:\n\npip install trimesh'
        raise ImportError(msg)

    # Avoid circular import
    from pyvista.core.dataobject import USER_DICT_KEY  # noqa: PLC0415

    _validation.check_instance(mesh, pv.DataSet, name='mesh')

    if isinstance(mesh, pv.PolyData):
        is_all_triangles = mesh.is_all_triangles
    elif isinstance(mesh, pv.UnstructuredGrid):
        is_all_triangles = mesh.distinct_cell_types == {pv.CellType.TRIANGLE}
    elif mesh.is_empty:
        is_all_triangles = True
    else:
        is_all_triangles = False

    if not is_all_triangles and not triangulate:
        msg = (
            'Mesh must be all triangles to convert to Trimesh object.\n'
            'Use `triangulate=True` to automatically convert to a triangle surface mesh.'
        )
        raise pv.NotAllTrianglesError(msg)

    surf = mesh if isinstance(mesh, pv.PolyData) else mesh.extract_surface(algorithm='geometry')
    surf = surf if is_all_triangles else surf.triangulate()

    pass_point_data, pass_cell_data, pass_field_data = _validate_pass_data(pass_data)

    if pass_point_data:
        vertex_attributes = dict((point_data := mesh.point_data).items())
        vertex_normals = vertex_attributes.pop(point_data.active_normals_name, None)  # type: ignore[arg-type]
        # Store texture coordinates
        texture_coordinates = vertex_attributes.pop(
            point_data.active_texture_coordinates_name,  # type: ignore[arg-type]
            None,
        )
        visual = (
            TextureVisuals(uv=texture_coordinates) if texture_coordinates is not None else None
        )
    else:
        vertex_attributes = None
        vertex_normals = None
        visual = None

    if pass_cell_data:
        face_attributes = dict((cell_data := mesh.cell_data).items())
        face_normals = face_attributes.pop(cell_data.active_normals_name, None)  # type: ignore[arg-type]
    else:
        face_attributes = None
        face_normals = None

    if pass_field_data:
        metadata = dict(mesh.field_data.items())
        if USER_DICT_KEY in metadata.keys():
            metadata.pop(USER_DICT_KEY)
            metadata.update(mesh.user_dict)
    else:
        metadata = None

    return trimesh.Trimesh(
        vertices=surf.points,
        faces=surf.regular_faces,
        vertex_normals=vertex_normals,
        face_normals=face_normals,
        face_attributes=face_attributes,  # type: ignore[arg-type]
        vertex_attributes=vertex_attributes,  # type: ignore[arg-type]
        metadata=metadata,
        visual=visual,
        process=False,
    )
