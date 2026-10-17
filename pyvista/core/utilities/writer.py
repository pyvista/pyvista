"""Writers for writing VTK objects to file."""

from __future__ import annotations

from abc import abstractmethod
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal
from typing import get_args

import numpy as np
import pyvista_validation as _validation

import pyvista as pv
from pyvista.core._vtk_utilities import vtk_version_info
from pyvista.core.filters.composite import _format_nested_index
from pyvista.core.utilities import _file_formats as _formats
from pyvista.core.utilities.fileio import _CompressionOptions
from pyvista.core.utilities.fileio import _FileIOBase
from pyvista.core.utilities.fileio import _warn_multiblock_nested_field_data
from pyvista.core.utilities.misc import abstract_class

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pyvista import DataObject
    from pyvista import _vtk

_DataFormatOptions = Literal['binary', 'ascii']


@abstract_class
class _DataFormatMixin:
    # Different writers use different values to indicate the current format
    """Add a ``data_format`` property to writers which support ASCII and binary output.

    .. note::
        This class is a private internal implementation detail. It is documented
        solely so that its public members, which are inherited by public classes,
        are visible in the documentation.

    Parameters
    ----------
    *args : tuple, optional
        Positional arguments passed to the parent class.

    **kwargs : dict, optional
        Keyword arguments passed to the parent class.


    """

    _ascii0_binary1: ClassVar[dict[int, _DataFormatOptions]] = {0: 'ascii', 1: 'binary'}
    _ascii1_binary2: ClassVar[dict[int, _DataFormatOptions]] = {1: 'ascii', 2: 'binary'}
    _format_mapping: ClassVar[dict[int, _DataFormatOptions]] = _ascii1_binary2

    @property
    @abstractmethod
    def writer(self) -> _vtk.vtkWriter:  # numpydoc ignore=RT01
        """Return the VTK writer this class configures."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data_format = 'binary'

    @property
    def data_format(self) -> _DataFormatOptions:  # numpydoc ignore=RT01
        """Return or set whether the file is written as ``'ascii'`` or ``'binary'``."""
        try:
            mode = self.writer.GetDataMode()  # type: ignore[attr-defined]
        except AttributeError:
            mode = self.writer.GetFileType()  # type: ignore[attr-defined]
        return self._format_mapping[mode]

    @data_format.setter
    def data_format(self, data_format: _DataFormatOptions) -> None:
        _validation.check_contains(get_args(_DataFormatOptions), data_format, name='data format')
        if data_format == 'binary':
            try:
                # DataWriter, PLYWriter, STLWriter
                self.writer.SetDataModeToBinary()  # type: ignore[attr-defined]
            except AttributeError:
                # XMLWriter
                self.writer.SetFileTypeToBinary()  # type: ignore[attr-defined]
        else:
            try:
                # DataWriter, PLYWriter, STLWriter
                self.writer.SetDataModeToAscii()  # type: ignore[attr-defined]
            except AttributeError:
                # XMLWriter
                self.writer.SetFileTypeToASCII()  # type: ignore[attr-defined]


@abstract_class
class BaseWriter(_FileIOBase):
    """The base writer class.

    The base functionality includes writing data to a file,
    and allowing access to the underlying vtk writer.

    Parameters
    ----------
    path : str, Path
        Path of the file to write to.

    data_object : DataObject
        Data object to write.

    """

    def __init__(self, path: str | Path, data_object: DataObject) -> None:
        """Initialize writer."""
        self._writer = self._vtk_class()
        super().__init__()
        self.path = path
        self.data_object = data_object
        self.written_path = path

    @classmethod
    def _get_extension_mappings(cls) -> list[dict[str, type]]:
        all_mesh_types = (
            pv.ImageData,
            pv.RectilinearGrid,
            pv.StructuredGrid,
            pv.PointSet,
            pv.PolyData,
            pv.UnstructuredGrid,
            pv.ExplicitStructuredGrid,
            pv.MultiBlock,
            pv.PartitionedDataSet,
        )
        return [mesh_type._WRITERS for mesh_type in all_mesh_types]

    @property
    def writer(self) -> _vtk.vtkWriter:
        """Return the vtk writer object.

        Returns
        -------
        vtkWriter
            An instance of the vtk writer.

        """
        return self._writer

    @property
    def path(self) -> str:  # numpydoc ignore=RT01
        """Return or set the filename or directory of the writer.

        Notes
        -----
        This is the path that will be passed to the underlying VTK writer.
        For most writers, this is the actual path of the written file.
        For writers that write multiple files (for example, EnSightWriter),
        this path can be renamed. See :attr:`written_path` for the actual path of the written file.

        Returns
        -------
        str
            The path of the file to write to.

        """
        return self.writer.GetFileName()  # type: ignore[attr-defined]

    @path.setter
    def path(self, path: str | Path) -> None:
        self.writer.SetFileName(str(path))  # type: ignore[attr-defined]

    @property
    def data_object(self) -> DataObject:  # numpydoc ignore=RT01
        """Get or set the dataset to write."""
        return self._data_object

    @data_object.setter
    def data_object(self, data_object: DataObject) -> None:
        self._data_object = data_object
        self.writer.SetInputData(data_object)

    @property
    def written_path(self) -> Path:
        """Return the formatted path of the written files.

        .. versionadded:: 0.49.0

        Notes
        -----
        Unlike :attr:`path`, ``written_path`` is the actual path of the written file.
        For most readers, ``path`` and ``written_path`` are identical. In cases where
        multiple files are written (for example, :class:`vtkEnSightWriter`), this path corresponds
        to the "main" output file that would be used for reading the mesh again.

        Returns
        -------
        pathlib.Path
            The path of the written file.

        """
        return self._written_path

    @written_path.setter
    def written_path(self, path: str | Path) -> None:
        self._written_path = Path(path)

    def _execute_before_write(self) -> None:
        """Execute code before calling ``write()``.

        Subclasses may optionally define this, for example, to issue warnings.
        """

    def write(self) -> None:
        """Write data to path."""
        self._execute_before_write()
        self.writer.Write()

    def _apply_kwargs_safely(self, **kwargs) -> None:
        """Try to set property keyword values and ignore attribute errors."""
        for name, value in kwargs.items():
            with contextlib.suppress(AttributeError):
                setattr(self, name, value)


class BMPWriter(BaseWriter):
    """Writer for the BMP Image format (``.bmp``).

    Wraps :vtk:`vtkBMPWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkBMPWriter'
    _format = _formats.BMP


class DataSetWriter(BaseWriter, _DataFormatMixin):
    """Writer for the Legacy VTK format (``.vtk``), for any dataset.

    Wraps :vtk:`vtkDataSetWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkDataSetWriter'
    _format = _formats.LEGACY_VTK

    def _execute_before_write(self) -> None:
        # Warn if data will be lost
        if isinstance(mesh := self.data_object, pv.ImageData) and not np.allclose(
            mesh.direction_matrix, np.eye(3)
        ):
            msg = (
                'The direction matrix for ImageData cannot be saved using the '
                'Legacy VTK `.vtk` format.\n'
                'See https://gitlab.kitware.com/vtk/vtk/-/issues/19663 \n'
                'Use the `.vti` extension instead (VTK ImageData format).'
            )
            raise ValueError(msg)


class HDFWriter(BaseWriter):
    """Writer for the VTKHDF format (``.hdf``, ``.vtkhdf``).

    Wraps :vtk:`vtkHDFWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkHDFWriter'
    _format = _formats.VTKHDF

    def _execute_before_write(self) -> None:
        if not isinstance(mesh := self.data_object, pv.MultiBlock):
            return
        # Check multiblock block types
        if pv.vtk_version_info < (9, 5, 0):
            if mesh.is_nested:
                msg = (
                    'Nested MultiBlocks are not supported by the VTKHDF format in VTK 9.4.'
                    '\nUpgrade to VTK>=9.5 for this functionality.'
                )
                raise TypeError(msg)
            if type(None) in mesh.block_types:
                msg = (
                    'Saving None blocks is not supported by the VTKHDF format in VTK 9.4.'
                    '\nUpgrade to VTK>=9.5 for this functionality.'
                )
                raise TypeError(msg)

        supported_block_types: list[type] = [
            pv.PolyData,
            pv.UnstructuredGrid,
            type(None),
            pv.MultiBlock,
            pv.PartitionedDataSet,
        ]
        for id_, name, block in mesh.recursive_iterator('all'):
            if type(block) not in supported_block_types:
                index_fmt = _format_nested_index(id_)
                msg = (
                    f"Block at index {index_fmt} with name '{name}' has type "
                    f'{block.__class__.__name__!r} '
                    f'which cannot be saved to the VTKHDF format.\n'
                    f'Supported types are: {[typ.__name__ for typ in supported_block_types]}.'
                )
                raise TypeError(msg)


# HDFWriter where the installed VTK can write VTKHDF, for splatting into a _WRITERS table
_HDF_WRITERS: tuple[type[HDFWriter], ...] = (HDFWriter,) if vtk_version_info >= (9, 4) else ()


class HoudiniPolyDataWriter(BaseWriter):
    """Writer for the Houdini Geometry format (``.geo``).

    Wraps :vtk:`vtkHoudiniPolyDataWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkHoudiniPolyDataWriter'
    _format = _formats.HOUDINI


class IVWriter(BaseWriter):
    """Writer for the Open Inventor format (``.iv``).

    Wraps :vtk:`vtkIVWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkIVWriter'
    _format = _formats.OPEN_INVENTOR


class JPEGWriter(BaseWriter):
    """Writer for the JPEG Image format (``.jpeg``, ``.jpg``).

    Wraps :vtk:`vtkJPEGWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkJPEGWriter'
    _format = _formats.JPEG


class NIFTIImageWriter(BaseWriter):
    """Writer for the NIfTI format (``.nii``, ``.nii.gz``).

    Wraps :vtk:`vtkNIFTIImageWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkNIFTIImageWriter'
    _format = _formats.NIFTI


class OBJWriter(BaseWriter):
    """Writer for the Wavefront OBJ format (``.obj``).

    Wraps :vtk:`vtkOBJWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkOBJWriter'
    _format = _formats.OBJ


class PLYWriter(BaseWriter, _DataFormatMixin):
    """Writer for the Stanford PLY format (``.ply``).

    Wraps :vtk:`vtkPLYWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkPLYWriter'
    _format = _formats.PLY

    @property
    def texture(self) -> str | None:  # numpydoc ignore=RT01
        """Get or set a texture array to be written."""
        return self.writer.GetArrayName()  # type: ignore[attr-defined]

    @texture.setter
    def texture(self, texture: str | NDArray[np.uint8] | None) -> None:
        if texture is None:
            self.writer.SetArrayName(None)  # type: ignore[attr-defined]
            return
        _validation.check_instance(texture, (str, np.ndarray), name='texture')
        if isinstance(texture, str):
            array_name = texture
            array = self.data_object[array_name]  # type: ignore[index]
        else:
            array_name = '_color_array'
            array = texture
            self.data_object[array_name] = texture

        _validation.check_subdtype(array, 'uint8', name='texture')
        self.writer.SetArrayName(array_name)  # type: ignore[attr-defined]

        # enable alpha channel if applicable
        enable_alpha = array.shape[-1] == 4
        self.writer.SetEnableAlpha(enable_alpha)  # type: ignore[attr-defined]


class PNGWriter(BaseWriter):
    """Writer for the PNG Image format (``.png``).

    Wraps :vtk:`vtkPNGWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkPNGWriter'
    _format = _formats.PNG


class PNMWriter(BaseWriter):
    """Writer for the PNM Image format (``.pnm``).

    Wraps :vtk:`vtkPNMWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkPNMWriter'
    _format = _formats.PNM


class PolyDataWriter(BaseWriter, _DataFormatMixin):
    """Writer for the Legacy VTK format (``.vtk``), for :class:`~pyvista.PolyData`.

    Wraps :vtk:`vtkPolyDataWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkPolyDataWriter'
    _format = _formats.LEGACY_VTK


class RectilinearGridWriter(BaseWriter, _DataFormatMixin):
    """Writer for the Legacy VTK format (``.vtk``), for :class:`~pyvista.RectilinearGrid`.

    Wraps :vtk:`vtkRectilinearGridWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkRectilinearGridWriter'
    _format = _formats.LEGACY_VTK


class STLWriter(BaseWriter, _DataFormatMixin):
    """Writer for the Stereolithography format (``.stl``).

    Wraps :vtk:`vtkSTLWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkSTLWriter'
    _format = _formats.STL


class SimplePointsWriter(BaseWriter, _DataFormatMixin):
    """Writer for the XYZ Point Cloud format (``.txt``, ``.xyz``).

    Wraps :vtk:`vtkSimplePointsWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkSimplePointsWriter'
    _format = _formats.XYZ_POINTS


class StructuredGridWriter(BaseWriter, _DataFormatMixin):
    """Writer for the Legacy VTK format (``.vtk``), for :class:`~pyvista.StructuredGrid`.

    Wraps :vtk:`vtkStructuredGridWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkStructuredGridWriter'
    _format = _formats.LEGACY_VTK


class TIFFWriter(BaseWriter):
    """Writer for the TIFF Image format (``.tif``, ``.tiff``).

    Wraps :vtk:`vtkTIFFWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkTIFFWriter'
    _format = _formats.TIFF


class UnstructuredGridWriter(BaseWriter, _DataFormatMixin):
    """Writer for the Legacy VTK format (``.vtk``), for :class:`~pyvista.UnstructuredGrid`.

    Wraps :vtk:`vtkUnstructuredGridWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkUnstructuredGridWriter'
    _format = _formats.LEGACY_VTK


class EnSightWriter(BaseWriter):
    """Writer for the EnSight format (``.case``).

    Wraps :vtk:`vtkEnSightWriter`.

    .. note::
        This is a parallel writer that prepends a process number to the ``.case`` extension,
        for example, ``<filename>.0.case``. Use :attr:`~pyvista.BaseWriter.written_path`
        to get the saved file after calling :meth:`~pyvista.BaseWriter.write`.

    .. note::
        This writer saves the mesh as a multi-block dataset.
        Even if the input mesh is a single block, the output will be a multi-block dataset
        and the default block name will be ``VTK Part``.

    .. versionadded:: 0.49.0

    """

    _vtk_class_name = 'vtkEnSightWriter'
    _format = _formats.ENSIGHT

    @property
    def path(self) -> str:  # numpydoc ignore=RT01
        """Return or set the filename or directory of the writer."""
        # vtkEnSightWriter has no single FileName concept, only Path/BaseName.
        path = Path(self.writer.GetPath())  # type: ignore[attr-defined]
        basename = self.writer.GetBaseName()  # type: ignore[attr-defined]
        return str(path / basename) if basename else str(path)

    @path.setter
    def path(self, path: str | Path) -> None:
        # Set Path/BaseName directly to avoid vtkEnSightWriter's ComputeNames(),
        # which splits FileName on the last '/' and breaks on Windows backslash
        # paths (falls back to Path="./" and mangles the rest into BaseName).
        raw_path = Path(path)
        self.writer.SetPath(str(raw_path.parent))  # type: ignore[attr-defined]
        self.writer.SetBaseName(raw_path.stem)  # type: ignore[attr-defined]

    def _execute_before_write(self) -> None:
        # ProcessNumber can still change after path is set (e.g. via writer_kwargs),
        # so the final filename must be resolved here, not in the path setter.
        raw_path = Path(self.path)
        process_number = self.writer.GetProcessNumber()  # type: ignore[attr-defined]
        self.written_path = raw_path.parent / f'{raw_path.stem}.{process_number}.case'


@abstract_class
class _XMLWriter(BaseWriter, _DataFormatMixin):
    """Base class for the XML writers, which also support compression.

    .. note::
        This class is a private internal implementation detail. It is documented
        solely so that its public members, which are inherited by public classes,
        are visible in the documentation.

    Parameters
    ----------
    *args : tuple, optional
        Positional arguments passed to the parent class.

    **kwargs : dict, optional
        Keyword arguments passed to the parent class.


    """

    _format_mapping = _DataFormatMixin._ascii0_binary1

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compression = 'zlib'

    @property
    def compression(self) -> _CompressionOptions:  # numpydoc ignore=RT01
        """Return or set the compressor the XML writer uses."""
        return self._compression

    @compression.setter
    def compression(self, compression: _CompressionOptions) -> None:
        supported = get_args(_CompressionOptions)
        _validation.check_contains(supported, must_contain=compression, name='compression')
        self._compression = compression
        if compression is None:
            self.writer.SetCompressorTypeToNone()  # type: ignore[attr-defined]
        elif compression == 'zlib':
            self.writer.SetCompressorTypeToZLib()  # type: ignore[attr-defined]
        elif compression == 'lz4':
            self.writer.SetCompressorTypeToLZ4()  # type: ignore[attr-defined]
        else:
            self.writer.SetCompressorTypeToLZMA()  # type: ignore[attr-defined]


class XMLImageDataWriter(_XMLWriter):
    """Writer for the VTK ImageData format (``.vti``).

    Wraps :vtk:`vtkXMLImageDataWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkXMLImageDataWriter'
    _format = _formats.VTK_IMAGE_DATA


class XMLMultiBlockDataWriter(_XMLWriter):
    """Writer for the VTK MultiBlock format (``.vtm``, ``.vtmb``).

    Wraps :vtk:`vtkXMLMultiBlockDataWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkXMLMultiBlockDataWriter'
    _format = _formats.VTK_MULTIBLOCK

    def _execute_before_write(self) -> None:
        _warn_multiblock_nested_field_data(self.data_object)


class XMLPartitionedDataSetWriter(_XMLWriter):
    """Writer for the VTK PartitionedDataSet format (``.vtpd``).

    Wraps :vtk:`vtkXMLPartitionedDataSetWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkXMLPartitionedDataSetWriter'
    _format = _formats.VTK_PARTITIONED_DATASET


class XMLPolyDataWriter(_XMLWriter):
    """Writer for the VTK PolyData format (``.vtp``).

    Wraps :vtk:`vtkXMLPolyDataWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkXMLPolyDataWriter'
    _format = _formats.VTK_POLYDATA


class XMLRectilinearGridWriter(_XMLWriter):
    """Writer for the VTK RectilinearGrid format (``.vtr``).

    Wraps :vtk:`vtkXMLRectilinearGridWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkXMLRectilinearGridWriter'
    _format = _formats.VTK_RECTILINEAR_GRID


class XMLStructuredGridWriter(_XMLWriter):
    """Writer for the VTK StructuredGrid format (``.vts``).

    Wraps :vtk:`vtkXMLStructuredGridWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkXMLStructuredGridWriter'
    _format = _formats.VTK_STRUCTURED_GRID


class XMLUnstructuredGridWriter(_XMLWriter):
    """Writer for the VTK UnstructuredGrid format (``.vtu``).

    Wraps :vtk:`vtkXMLUnstructuredGridWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_class_name = 'vtkXMLUnstructuredGridWriter'
    _format = _formats.VTK_UNSTRUCTURED_GRID
