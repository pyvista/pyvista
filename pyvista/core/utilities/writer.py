"""Writers for writing vtk objects to file."""

from __future__ import annotations

from abc import abstractmethod
import contextlib
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal
from typing import get_args

import numpy as np

from pyvista._warn_external import warn_external
from pyvista.core import _validation
from pyvista.core.utilities.fileio import _CompressionOptions
from pyvista.core.utilities.fileio import _FileIOBase
from pyvista.core.utilities.fileio import _warn_multiblock_nested_field_data
from pyvista.core.utilities.misc import abstract_class

if TYPE_CHECKING:
    from pathlib import Path

    from vtk import vtkWriter

    from pyvista import DataObject
    from pyvista import NumpyArray

_DataFormatOptions = Literal['binary', 'ascii']


@abstract_class
class _DataFormatMixin:
    # Different writers use different values to indicate the current format
    _ascii0_binary1: ClassVar[dict[int, _DataFormatOptions]] = {0: 'ascii', 1: 'binary'}
    _ascii1_binary2: ClassVar[dict[int, _DataFormatOptions]] = {1: 'ascii', 2: 'binary'}
    _format_mapping: ClassVar[dict[int, _DataFormatOptions]] = _ascii1_binary2

    @property
    @abstractmethod
    def writer(self) -> vtkWriter: ...

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data_format = 'binary'

    @property
    def data_format(self) -> _DataFormatOptions:
        try:
            mode = self.writer.GetDataMode()
        except AttributeError:
            mode = self.writer.GetFileType()
        return self._format_mapping[mode]

    @data_format.setter
    def data_format(self, data_format: _DataFormatOptions) -> None:
        _validation.check_contains(get_args(_DataFormatOptions), data_format, name='data format')
        if data_format == 'binary':
            try:
                self.writer.SetDataModeToBinary()  # DataWriter, PLYWriter, STLWriter
            except AttributeError:
                self.writer.SetFileTypeToBinary()  # XMLWriter
        else:
            try:
                self.writer.SetDataModeToAscii()  # DataWriter, PLYWriter, STLWriter
            except AttributeError:
                self.writer.SetFileTypeToASCII()  # XMLWriter


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

    @classmethod
    def _get_extension_mappings(cls) -> list[dict[str, type]]:
        import pyvista as pv  # noqa: PLC0415

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
    def writer(self) -> vtkWriter:
        """Return the vtk writer object.

        Returns
        -------
        vtkWriter
            An instance of the vtk writer.

        """
        return self._writer

    @property
    def path(self) -> str:  # numpydoc ignore=RT01
        """Return or set the filename or directory of the writer."""
        return self.writer.GetFileName()

    @path.setter
    def path(self, path: str | Path) -> None:
        self.writer.SetFileName(str(path))

    @property
    def data_object(self) -> DataObject:  # numpydoc ignore=RT01
        """Get or set the dataset to write."""
        return self._data_object

    @data_object.setter
    def data_object(self, data_object: DataObject) -> None:
        self._data_object = data_object
        self.writer.SetInputData(data_object)

    def _execute_before_write(self) -> None:
        """Execute code before calling `write()`.

        Subclasses may optionally define this, e.g. to issue warnings.
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
    """BMPWriter for ``.bmp`` files.

    Wraps :vtk:`vtkBMPWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOImage'
    _vtk_class_name = 'vtkBMPWriter'


class DataSetWriter(BaseWriter, _DataFormatMixin):
    """DataSetWriter for VTK legacy dataset files ``.vtk``.

    Wraps :vtk:`vtkDataSetWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOLegacy'
    _vtk_class_name = 'vtkDataSetWriter'

    def _execute_before_write(self) -> None:
        import pyvista as pv  # noqa: PLC0415

        # Warn if data will be lost
        if isinstance(mesh := self.data_object, pv.ImageData) and not np.allclose(
            mesh.direction_matrix, np.eye(3)
        ):
            msg = (
                'The direction matrix for ImageData will not be saved using the '
                'legacy `.vtk` format.\n'
                'See https://gitlab.kitware.com/vtk/vtk/-/issues/19663 \n'
                'Use the `.vti` extension instead (XML format).'
            )
            warn_external(msg)


class HDFWriter(BaseWriter):
    """HDFWriter for ``.hdf`` and ``.vtkhdf`` files.

    Wraps :vtk:`vtkHDFWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOHDF'
    _vtk_class_name = 'vtkHDFWriter'

    def _execute_before_write(self) -> None:
        import pyvista as pv  # noqa: PLC0415

        if not isinstance(mesh := self.data_object, pv.MultiBlock):
            return
        # Check multiblock block types
        if pv.vtk_version_info < (9, 5, 0):
            if mesh.is_nested:
                msg = (
                    'Nested MultiBlocks are not supported by the .vtkhdf format in VTK 9.4.'
                    '\nUpgrade to VTK>=9.5 for this functionality.'
                )
                raise TypeError(msg)
            if type(None) in mesh.block_types:
                msg = (
                    'Saving None blocks is not supported by the .vtkhdf format in VTK 9.4.'
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
                from pyvista.core.filters.composite import _format_nested_index  # noqa: PLC0415

                index_fmt = _format_nested_index(id_)
                msg = (
                    f"Block at index {index_fmt} with name '{name}' has type "
                    f'{block.__class__.__name__!r} '
                    f'which cannot be saved to the .vtkhdf format.\n'
                    f'Supported types are: {[typ.__name__ for typ in supported_block_types]}.'
                )
                raise TypeError(msg)


class HoudiniPolyDataWriter(BaseWriter):
    """HoudiniPolyDataWriter for Houdini geometry ``.geo`` files.

    Wraps :vtk:`vtkHoudiniPolyDataWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOGeometry'
    _vtk_class_name = 'vtkHoudiniPolyDataWriter'


class IVWriter(BaseWriter):
    """IVWriter for OpenInventor ``.iv`` files.

    Wraps :vtk:`vtkIVWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOGeometry'
    _vtk_class_name = 'vtkIVWriter'


class JPEGWriter(BaseWriter):
    """JPEGWriter for ``.jpeg`` and ``.jpg`` files.

    Wraps :vtk:`vtkJPEGWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOImage'
    _vtk_class_name = 'vtkJPEGWriter'


class NIFTIImageWriter(BaseWriter):
    """NIFTIImageWriter for ``.nii`` and ``.nii.gz`` files.

    Wraps :vtk:`vtkNIFTIImageWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOImage'
    _vtk_class_name = 'vtkNIFTIImageWriter'


class OBJWriter(BaseWriter):
    """OBJWriter for Wavefront ``.obj`` files.

    Wraps :vtk:`vtkOBJWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOGeometry'
    _vtk_class_name = 'vtkOBJWriter'


class PLYWriter(BaseWriter, _DataFormatMixin):
    """PLYWriter for PLY polygonal ``.ply`` files.

    Wraps :vtk:`vtkPLYWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOPLY'
    _vtk_class_name = 'vtkPLYWriter'

    @property
    def texture(self) -> str | None:  # numpydoc ignore=RT01
        """Get or set a texture array to be written."""
        return self.writer.GetArrayName()

    @texture.setter
    def texture(self, texture: str | NumpyArray[float] | None) -> None:
        if texture is None:
            self.writer.SetArrayName(None)
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
        self.writer.SetArrayName(array_name)

        # enable alpha channel if applicable
        enable_alpha = array.shape[-1] == 4
        self.writer.SetEnableAlpha(enable_alpha)


class PNGWriter(BaseWriter):
    """PNGWriter for ``.png`` files.

    Wraps :vtk:`vtkPNGWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOImage'
    _vtk_class_name = 'vtkPNGWriter'


class PNMWriter(BaseWriter):
    """PNMWriter for ``.pnm`` files.

    Wraps :vtk:`vtkPNMWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOImage'
    _vtk_class_name = 'vtkPNMWriter'


class PolyDataWriter(BaseWriter, _DataFormatMixin):
    """PolyDataWriter for legacy VTK PolyData ``.vtk`` files.

    Wraps :vtk:`vtkPolyDataWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOLegacy'
    _vtk_class_name = 'vtkPolyDataWriter'


class RectilinearGridWriter(BaseWriter, _DataFormatMixin):
    """RectilinearGridWriter for legacy VTK rectilinear grid ``.vtk`` files.

    Wraps :vtk:`vtkRectilinearGridWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOLegacy'
    _vtk_class_name = 'vtkRectilinearGridWriter'


class STLWriter(BaseWriter, _DataFormatMixin):
    """STLWriter for stereolithography  ``.stl`` files.

    Wraps :vtk:`vtkSTLWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOGeometry'
    _vtk_class_name = 'vtkSTLWriter'


class SimplePointsWriter(BaseWriter, _DataFormatMixin):
    """SimplePointsWriter for simple point-set ``.xyz`` files.

    Wraps :vtk:`vtkSimplePointsWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOLegacy'
    _vtk_class_name = 'vtkSimplePointsWriter'


class StructuredGridWriter(BaseWriter, _DataFormatMixin):
    """StructuredGridWriter for legacy VTK structured grid ``.vtk`` files.

    Wraps :vtk:`vtkStructuredGridWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOLegacy'
    _vtk_class_name = 'vtkStructuredGridWriter'


class StructuredPointsWriter(BaseWriter, _DataFormatMixin):
    """StructuredPointsWriter for legacy VTK structured points ``.vtk`` files.

    Wraps :vtk:`vtkStructuredPointsWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOLegacy'
    _vtk_class_name = 'vtkStructuredPointsWriter'


class TIFFWriter(BaseWriter):
    """TIFFWriter for ``.tif`` and ``.tiff`` files.

    Wraps :vtk:`vtkTIFFWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOImage'
    _vtk_class_name = 'vtkTIFFWriter'


class UnstructuredGridWriter(BaseWriter, _DataFormatMixin):
    """UnstructuredGridWriter for legacy VTK unstructured grid ``.vtk`` files.

    Wraps :vtk:`vtkUnstructuredGridWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOLegacy'
    _vtk_class_name = 'vtkUnstructuredGridWriter'


@abstract_class
class _XMLWriter(BaseWriter, _DataFormatMixin):
    _format_mapping = _DataFormatMixin._ascii0_binary1

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compression = 'zlib'

    @property
    def compression(self) -> _CompressionOptions:
        return self._compression

    @compression.setter
    def compression(self, compression: _CompressionOptions) -> None:
        supported = get_args(_CompressionOptions)
        _validation.check_contains(supported, must_contain=compression, name='compression')
        self._compression = compression
        if compression is None:
            self.writer.SetCompressorTypeToNone()
        elif compression == 'zlib':
            self.writer.SetCompressorTypeToZLib()
        elif compression == 'lz4':
            self.writer.SetCompressorTypeToLZ4()
        else:
            self.writer.SetCompressorTypeToLZMA()


class XMLImageDataWriter(_XMLWriter):
    """XMLImageDataWriter for VTK XML image data ``.vti`` files.

    Wraps :vtk:`vtkXMLImageDataWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOXML'
    _vtk_class_name = 'vtkXMLImageDataWriter'


class XMLMultiBlockDataWriter(_XMLWriter):
    """XMLMultiBlockDataWriter for VTK XML multiblock ``.vtm`` files.

    Wraps :vtk:`vtkXMLMultiBlockDataWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOXML'
    _vtk_class_name = 'vtkXMLMultiBlockDataWriter'

    def _execute_before_write(self) -> None:
        _warn_multiblock_nested_field_data(self.data_object)


class XMLPartitionedDataSetWriter(_XMLWriter):
    """XMLPartitionedDataSetWriter for VTK XML partitioned datasets ``.vtpd`` files.

    Wraps :vtk:`vtkXMLPartitionedDataSetWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOParallelXML'
    _vtk_class_name = 'vtkXMLPartitionedDataSetWriter'


class XMLPolyDataWriter(_XMLWriter):
    """XMLPolyDataWriter for VTK XML polydata ``.vtp`` files.

    Wraps :vtk:`vtkXMLPolyDataWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOXML'
    _vtk_class_name = 'vtkXMLPolyDataWriter'


class XMLRectilinearGridWriter(_XMLWriter):
    """XMLRectilinearGridWriter for VTK XML rectilinear grid ``.vtr`` files.

    Wraps :vtk:`vtkXMLRectilinearGridWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOXML'
    _vtk_class_name = 'vtkXMLRectilinearGridWriter'


class XMLStructuredGridWriter(_XMLWriter):
    """XMLStructuredGridWriter for VTK XML structured grid ``.vts`` files.

    Wraps :vtk:`vtkXMLStructuredGridWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOXML'
    _vtk_class_name = 'vtkXMLStructuredGridWriter'


class XMLUnstructuredGridWriter(_XMLWriter):
    """XMLUnstructuredGridWriter for VTK XML unstructured grid ``.vtu`` files.

    Wraps :vtk:`vtkXMLUnstructuredGridWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOXML'
    _vtk_class_name = 'vtkXMLUnstructuredGridWriter'
