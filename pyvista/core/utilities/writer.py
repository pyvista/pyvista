"""Writers for writing vtk objects to file."""

from __future__ import annotations

from abc import abstractmethod
import contextlib
from typing import TYPE_CHECKING
from typing import Literal
from typing import get_args

import numpy as np

from pyvista.core import _validation
from pyvista.core.utilities.fileio import _CompressionOptions
from pyvista.core.utilities.misc import _classproperty
from pyvista.core.utilities.misc import _NoNewAttrMixin
from pyvista.core.utilities.misc import abstract_class
from pyvista.core.utilities.reader import _lazy_vtk_import

if TYPE_CHECKING:
    from pathlib import Path

    from vtk import vtkWriter

    from pyvista import DataObject
    from pyvista import NumpyArray

_DataModeOptions = Literal['binary', 'ascii']


@abstract_class
class _DataModeMixin:
    @property
    @abstractmethod
    def writer(self) -> vtkWriter: ...

    @property
    def data_mode(self) -> _DataModeOptions:
        try:
            mode = self.writer.GetDataMode()
        except AttributeError:
            mode = self.writer.GetFileType()
        return 'binary' if mode == 1 else 'ascii'

    @data_mode.setter
    def data_mode(self, mode: _DataModeOptions) -> None:
        _validation.check_contains(get_args(_DataModeOptions), mode, name='data_mode')
        if mode == 'binary':
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
class BaseWriter(_NoNewAttrMixin):
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

    _vtk_module_name: str = ''
    _vtk_class_name: str = ''

    def __init__(self, path: str | Path, data_object: DataObject) -> None:
        """Initialize writer."""
        self._writer = self._vtk_class()
        self.path = path
        self.data_object = data_object

    def __repr__(self) -> str:
        """Representation of a writer object."""
        return f"{self.__class__.__name__}('{self.path}')"

    @_classproperty
    def _vtk_class(self) -> vtkWriter:
        return _lazy_vtk_import(self._vtk_module_name, self._vtk_class_name)

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

    def write(self) -> None:
        """Write data to path."""
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


class DataSetWriter(BaseWriter, _DataModeMixin):
    """DataSetWriter for VTK legacy dataset files ``.vtk``.

    Wraps :vtk:`vtkDataSetWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOLegacy'
    _vtk_class_name = 'vtkDataSetWriter'


class HDFWriter(BaseWriter):
    """HDFWriter for ``.hdf`` and ``.vtkhdf`` files.

    Wraps :vtk:`vtkHDFWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOHDF'
    _vtk_class_name = 'vtkHDFWriter'


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


class PLYWriter(BaseWriter, _DataModeMixin):
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
        else:
            array_name = '_color_array'
            self.data_object[array_name] = texture
        self.writer.SetArrayName(array_name)

        # enable alpha channel if applicable
        enable_alpha = self.data_object[array_name].shape[-1] == 4  # type: ignore[index]
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


class PolyDataWriter(BaseWriter, _DataModeMixin):
    """PolyDataWriter for legacy VTK PolyData ``.vtk`` files.

    Wraps :vtk:`vtkPolyDataWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOLegacy'
    _vtk_class_name = 'vtkPolyDataWriter'


class RectilinearGridWriter(BaseWriter, _DataModeMixin):
    """RectilinearGridWriter for legacy VTK rectilinear grid ``.vtk`` files.

    Wraps :vtk:`vtkRectilinearGridWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOLegacy'
    _vtk_class_name = 'vtkRectilinearGridWriter'


class STLWriter(BaseWriter, _DataModeMixin):
    """STLWriter for stereolithography  ``.stl`` files.

    Wraps :vtk:`vtkSTLWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOGeometry'
    _vtk_class_name = 'vtkSTLWriter'


class SimplePointsWriter(BaseWriter, _DataModeMixin):
    """SimplePointsWriter for simple point-set ``.xyz`` files.

    Wraps :vtk:`vtkSimplePointsWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOLegacy'
    _vtk_class_name = 'vtkSimplePointsWriter'


class StructuredGridWriter(BaseWriter, _DataModeMixin):
    """StructuredGridWriter for legacy VTK structured grid ``.vtk`` files.

    Wraps :vtk:`vtkStructuredGridWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOLegacy'
    _vtk_class_name = 'vtkStructuredGridWriter'


class TIFFWriter(BaseWriter):
    """TIFFWriter for ``.tif`` and ``.tiff`` files.

    Wraps :vtk:`vtkTIFFWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOImage'
    _vtk_class_name = 'vtkTIFFWriter'


class UnstructuredGridWriter(BaseWriter, _DataModeMixin):
    """UnstructuredGridWriter for legacy VTK unstructured grid ``.vtk`` files.

    Wraps :vtk:`vtkUnstructuredGridWriter`.

    .. versionadded:: 0.47.0

    """

    _vtk_module_name = 'vtkIOLegacy'
    _vtk_class_name = 'vtkUnstructuredGridWriter'


@abstract_class
class _XMLWriter(BaseWriter, _DataModeMixin):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compression = 'zlib'
        self.data_mode = 'binary'

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
