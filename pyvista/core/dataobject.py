"""Attributes common to PolyData and Grid Objects."""

import collections.abc
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional, List, Tuple, Iterable, Union, Any, Dict, DefaultDict, Type
import pickle

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import pyvista
from .pyvista_ndarray import pyvista_ndarray
from pyvista.utilities import (FieldAssociation, get_array, is_pyvista_dataset,
                               raise_not_matching, vtk_id_list_to_array, fileio,
                               abstract_class, axis_rotation, transformations)
from .datasetattributes import DataSetAttributes
from .filters import DataSetFilters, _get_output
from .._typing import Vector

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')

# vector array names
DEFAULT_VECTOR_KEY = '_vectors'



class ActiveArrayInfo:
    """Active array info class with support for pickling."""

    def __init__(self, association, name):
        """Initialize."""
        self.association = association
        self.name = name

    def __getstate__(self):
        """Support pickling."""
        state = self.__dict__.copy()
        state['association'] = int(self.association.value)
        return state

    def __setstate__(self, state):
        """Support unpickling."""
        self.__dict__ = state.copy()
        self.association = FieldAssociation(state['association'])

    @property
    def _namedtuple(self):
        """Build a namedtuple on the fly to provide legacy support."""
        named_tuple = collections.namedtuple('ActiveArrayInfo', ['association', 'name'])
        return named_tuple(self.association, self.name)

    def __iter__(self):
        """Provide namedtuple-like __iter__."""
        return self._namedtuple.__iter__()

    def __repr__(self):
        """Provide namedtuple-like __repr__."""
        return self._namedtuple.__repr__()

    def __getitem__(self, item):
        """Provide namedtuple-like __getitem__."""
        return self._namedtuple.__getitem__(item)

    def __setitem__(self, key, value):
        """Provide namedtuple-like __setitem__."""
        self._namedtuple.__setitem__(key, value)

    def __getattr__(self, item):
        """Provide namedtuple-like __getattr__."""
        self._namedtuple.__getattr__(item)

    def __eq__(self, other):
        """Check equivalence (useful for serialize/deserialize tests)."""
        return self.name == other.name and \
               int(self.association.value) == int(other.association.value)


@abstract_class
class DataObject:
    """Methods common to all wrapped data objects."""

    _READERS: Dict[str, Union[Type[vtk.vtkXMLReader], Type[vtk.vtkDataReader]]] = {}
    _WRITERS: Dict[str, Union[Type[vtk.vtkXMLWriter], Type[vtk.vtkDataWriter]]] = {}

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the data object."""
        super().__init__()
        # Remember which arrays come from numpy.bool arrays, because there is no direct
        # conversion from bool to vtkBitArray, such arrays are stored as vtkCharArray.
        self.association_bitarray_names: DefaultDict = collections.defaultdict(set)

    def __getattr__(self, item: str) -> Any:
        """Get attribute from base class if not found."""
        return super().__getattribute__(item)

    def shallow_copy(self, to_copy: vtk.vtkDataObject) -> vtk.vtkDataObject:
        """Shallow copy the given mesh to this mesh."""
        return self.ShallowCopy(to_copy)

    def deep_copy(self, to_copy: vtk.vtkDataObject) -> vtk.vtkDataObject:
        """Overwrite this mesh with the given mesh as a deep copy."""
        return self.DeepCopy(to_copy)

    def _load_file(self, filename: Union[str, Path]) -> vtk.vtkDataObject:
        """Generically load a vtk object from file.

        Parameters
        ----------
        filename : str, pathlib.Path
            Filename of object to be loaded.  File/reader type is inferred from the
            extension of the filename.

        Notes
        -----
        Binary files load much faster than ASCII.

        """
        if self._READERS is None:
            raise NotImplementedError(f'{self.__class__.__name__} readers are not specified,'
                                      ' this should be a dict of (file extension: vtkReader type)')

        file_path = Path(filename).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f'File {filename} does not exist')

        file_ext = file_path.suffix
        if file_ext not in self._READERS:
            valid_extensions = ', '.join(self._READERS.keys())
            raise ValueError(f'Invalid file extension for {self.__class__.__name__}({file_ext}).'
                             f' Must be one of: {valid_extensions}')

        reader = self._READERS[file_ext]()
        if file_ext == ".case":
            reader.SetCaseFileName(str(file_path))
        else:
            reader.SetFileName(str(file_path))
        reader.Update()
        return reader.GetOutputDataObject(0)

    def _from_file(self, filename: Union[str, Path]):
        self.shallow_copy(self._load_file(filename))

    def save(self, filename: str, binary=True):
        """Save this vtk object to file.

        Parameters
        ----------
        filename : str, pathlib.Path
         Filename of output file. Writer type is inferred from
         the extension of the filename.

        binary : bool, optional
         If True, write as binary, else ASCII.

        Notes
        -----
        Binary files write much faster than ASCII and have a smaller
        file size.

        """
        if self._WRITERS is None:
            raise NotImplementedError(f'{self.__class__.__name__} writers are not specified,'
                                      ' this should be a dict of (file extension: vtkWriter type)')

        file_path = Path(filename).resolve()
        file_ext = file_path.suffix
        if file_ext not in self._WRITERS:
            raise ValueError('Invalid file extension for this data type.'
                             f' Must be one of: {self._WRITERS.keys()}')

        writer = self._WRITERS[file_ext]()
        fileio.set_vtkwriter_mode(vtk_writer=writer, use_binary=binary)
        writer.SetFileName(str(file_path))
        writer.SetInputData(self)
        writer.Write()

    @abstractmethod
    def get_data_range(self):  # pragma: no cover
        """Get the non-NaN min and max of a named array."""
        raise NotImplementedError(f'{type(self)} mesh type does not have a `get_data_range` method.')

    def _get_attrs(self):  # pragma: no cover
        """Return the representation methods (internal helper)."""
        raise NotImplementedError('Called only by the inherited class')

    def head(self, display=True, html: bool = None):
        """Return the header stats of this dataset.

        If in IPython, this will be formatted to HTML. Otherwise returns a console friendly string.

        """
        # Generate the output
        if html:
            fmt = ""
            # HTML version
            fmt += "\n"
            fmt += "<table>\n"
            fmt += f"<tr><th>{type(self).__name__}</th><th>Information</th></tr>\n"
            row = "<tr><td>{}</td><td>{}</td></tr>\n"
            # now make a call on the object to get its attributes as a list of len 2 tuples
            for attr in self._get_attrs():
                try:
                    fmt += row.format(attr[0], attr[2].format(*attr[1]))
                except:
                    fmt += row.format(attr[0], attr[2].format(attr[1]))
            if hasattr(self, 'n_arrays'):
                fmt += row.format('N Arrays', self.n_arrays)
            fmt += "</table>\n"
            fmt += "\n"
            if display:
                from IPython.display import display as _display, HTML
                _display(HTML(fmt))
                return
            return fmt
        # Otherwise return a string that is Python console friendly
        fmt = f"{type(self).__name__} ({hex(id(self))})\n"
        # now make a call on the object to get its attributes as a list of len 2 tuples
        row = "  {}:\t{}\n"
        for attr in self._get_attrs():
            try:
                fmt += row.format(attr[0], attr[2].format(*attr[1]))
            except:
                fmt += row.format(attr[0], attr[2].format(attr[1]))
        if hasattr(self, 'n_arrays'):
            fmt += row.format('N Arrays', self.n_arrays)
        return fmt

    def _repr_html_(self):  # pragma: no cover
        """Return a pretty representation for Jupyter notebooks.

        This includes header details and information about all arrays.

        """
        raise NotImplementedError('Called only by the inherited class')

    def copy_meta_from(self, ido):  # pragma: no cover
        """Copy pyvista meta data onto this object from another object."""
        pass  # called only by the inherited class

    def copy(self, deep=True):
        """Return a copy of the object.

        Parameters
        ----------
        deep : bool, optional
            When True makes a full copy of the object.

        Returns
        -------
        newobject : same as input
           Deep or shallow copy of the input.

        """
        thistype = type(self)
        newobject = thistype()
        if deep:
            newobject.deep_copy(self)
        else:
            newobject.shallow_copy(self)
        newobject.copy_meta_from(self)
        return newobject

    def add_field_array(self, scalars: np.ndarray, name: str, deep=True):
        """Add a field array."""
        self.field_arrays.append(scalars, name, deep_copy=deep)

    @property
    def field_arrays(self) -> DataSetAttributes:
        """Return vtkFieldData as DataSetAttributes."""
        return DataSetAttributes(self.GetFieldData(), dataset=self, association=FieldAssociation.NONE)

    def clear_field_arrays(self):
        """Remove all field arrays."""
        self.field_arrays.clear()

    @property
    def memory_address(self) -> str:
        """Get address of the underlying C++ object in format 'Addr=%p'."""
        return self.GetInformation().GetAddressAsString("")

    @property
    def actual_memory_size(self) -> int:
        """Return the actual size of the dataset object.

        Returns
        -------
        int
            The actual size of the dataset object in kibibytes (1024 bytes).

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> mesh.actual_memory_size  # doctest:+SKIP
        93

        """
        return self.GetActualMemorySize()

    def copy_structure(self, dataset: vtk.vtkDataSet):
        """Copy the structure (geometry and topology) of the input dataset object.

        Examples
        --------
        >>> import pyvista as pv
        >>> source = pv.UniformGrid((10, 10, 5))
        >>> target = pv.UniformGrid()
        >>> target.copy_structure(source)
        >>> target.plot(show_edges=True)  # doctest:+SKIP

        """
        self.CopyStructure(dataset)

    def copy_attributes(self, dataset: vtk.vtkDataSet):
        """Copy the data attributes of the input dataset object.

        Examples
        --------
        >>> import pyvista as pv
        >>> source = pv.UniformGrid((10, 10, 5))
        >>> source = source.compute_cell_sizes()
        >>> target = pv.UniformGrid((10, 10, 5))
        >>> target.copy_attributes(source)
        >>> target.plot(scalars='Volume', show_edges=True)  # doctest:+SKIP

        """
        self.CopyAttributes(dataset)

    def __getstate__(self):
        """Support pickle. Serialize the VTK object to ASCII string."""
        state = self.__dict__.copy()
        writer = vtk.vtkDataSetWriter()
        writer.SetInputDataObject(self)
        writer.SetWriteToOutputString(True)
        writer.SetFileTypeToASCII()
        writer.Write()
        to_serialize = writer.GetOutputString()
        state['vtk_serialized'] = to_serialize
        return state

    def __setstate__(self, state):
        """Support unpickle."""
        vtk_serialized = state.pop('vtk_serialized')
        self.__dict__.update(state)

        reader = vtk.vtkDataSetReader()
        reader.ReadFromInputStringOn()
        reader.SetInputString(vtk_serialized)
        reader.Update()
        mesh = pyvista.wrap(reader.GetOutput())

        # copy data
        self.copy_structure(mesh)
        self.copy_attributes(mesh)
