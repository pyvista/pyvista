"""Attributes common to PolyData and Grid Objects."""

import collections
import logging
import os
import warnings

import numpy as np
import vtk

import pyvista
from pyvista.utilities import (FieldAssociation, get_array, is_pyvista_dataset,
                               parse_field_choice, raise_not_matching, vtk_id_list_to_array,
                               fileio, abstract_class)
from .datasetattributes import DataSetAttributes
from .filters import DataSetFilters

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')

# vector array names
DEFAULT_VECTOR_KEY = '_vectors'

ActiveArrayInfo = collections.namedtuple('ActiveInfo', field_names=['association', 'name'])


@abstract_class
class DataObject:
    """Methods common to all wrapped data objects."""

    _READERS = None
    _WRITERS = None

    def __init__(self, *args, **kwargs):
        """Initialize the data object."""
        super().__init__()
        # Remember which arrays come from numpy.bool arrays, because there is no direct
        # conversion from bool to vtkBitArray, such arrays are stored as vtkCharArray.
        self.association_bitarray_names = collections.defaultdict(set)

    def shallow_copy(self, to_copy):
        """Shallow copy the given mesh to this mesh."""
        return self.ShallowCopy(to_copy)

    def deep_copy(self, to_copy):
        """Overwrite this mesh with the given mesh as a deep copy."""
        return self.DeepCopy(to_copy)

    def _load_file(self, filename):
        """Generically load a vtk object from file.

        Parameters
        ----------
        filename : str
            Filename of object to be loaded.  File/reader type is inferred from the
            extension of the filename.

        Notes
        -----
        Binary files load much faster than ASCII.

        """
        if self._READERS is None:
            raise NotImplementedError('{} readers are not specified, this should be a' \
                                      ' dict of (file extension: vtkReader type)'
                                      .format(self.__class__.__name__))

        filename = os.path.abspath(os.path.expanduser(filename))
        if not os.path.isfile(filename):
            raise FileNotFoundError('File %s does not exist' % filename)

        file_ext = fileio.get_ext(filename)
        if file_ext not in self._READERS:
            keys_list = ', '.join(self._READERS.keys())
            raise ValueError('Invalid file extension for {}({}). Must be one of: {}'.format(
                self.__class__.__name__, file_ext, keys_list))

        reader = self._READERS[file_ext]()
        reader.SetFileName(filename)
        reader.Update()
        self.shallow_copy(reader.GetOutput())

    def save(self, filename, binary=True):
        """Save this vtk object to file.

        Parameters
        ----------
        filename : str
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
            raise NotImplementedError('{} writers are not specified, this should be a' \
                                      ' dict of (file extension: vtkWriter type)'
                                      .format(self.__class__.__name__))

        filename = os.path.abspath(os.path.expanduser(filename))
        file_ext = fileio.get_ext(filename)
        if file_ext not in self._WRITERS:
            raise ValueError('Invalid file extension for this data type. Must be one of: {}'.format(
                self._WRITERS.keys()))

        writer = self._WRITERS[file_ext]()
        fileio.set_vtkwriter_mode(vtk_writer=writer, use_binary=binary)
        writer.SetFileName(filename)
        writer.SetInputData(self)
        writer.Write()

    def get_data_range(self, arr=None, preference='field'):  # pragma: no cover
        """Get the non-NaN min and max of a named array.

        Parameters
        ----------
        arr : str, np.ndarray, optional
            The name of the array to get the range. If None, the
            active scalar is used

        preference : str, optional
            When scalars is specified, this is the preferred array type
            to search for in the dataset.  Must be either ``'point'``,
            ``'cell'``, or ``'field'``.

        """
        raise NotImplementedError('{} mesh type does not have a `get_data_range` method.'.format(type(self)))

    def _get_attrs(self):  # pragma: no cover
        """Return the representation methods (internal helper)."""
        raise NotImplementedError('Called only by the inherited class')

    def head(self, display=True, html=None):
        """Return the header stats of this dataset.

        If in IPython, this will be formatted to HTML. Otherwise returns a console friendly string.

        """
        # Generate the output
        if html:
            fmt = ""
            # HTML version
            fmt += "\n"
            fmt += "<table>\n"
            fmt += "<tr><th>{}</th><th>Information</th></tr>\n".format(type(self).__name__)
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
                from IPython.display import display, HTML
                display(HTML(fmt))
                return
            return fmt
        # Otherwise return a string that is Python console friendly
        fmt = "{} ({})\n".format(type(self).__name__, hex(id(self)))
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
        raise NotImplemented('Called only by the inherited class')

    def copy_meta_from(self, ido):  # pragma: no cover
        """Copy pyvista meta data onto this object from another object."""
        pass  # called only by the inherited class

    def copy(self, deep=True):
        """Return a copy of the object.

        Parameters
        ----------
        deep : bool, optional
            When True makes a full copy of the object.

        Return
        ------
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

    def _add_field_array(self, scalars, name, deep=True):
        """Add a field array to the mesh.

        Parameters
        ----------
        scalars : numpy.ndarray
            Numpy array of scalars.  Does not have to match number of points or
            numbers of cells.

        name : str
            Name of field scalars to add.

        deep : bool, optional
            Does not copy scalars when False.  A reference to the scalars
            must be kept to avoid a segfault.

        """
        self.field_arrays.append(scalars, name, deep_copy=deep)

    def _add_field_scalar(self, scalars, name, set_active=False, deep=True):  # pragma: no cover
        """Add a field array.

        DEPRECATED: Please use `_add_field_array` instead.

        """
        warnings.warn('Deprecation Warning: `_add_field_scalar` is now `_add_field_array`', RuntimeWarning)
        return self._add_field_array(scalars, name, deep=deep)

    def add_field_array(self, scalars, name, deep=True):
        """Add a field array."""
        self._add_field_array(scalars, name, deep=deep)

    @property
    def field_arrays(self):
        """Return vtkFieldData as DataSetAttributes."""
        return DataSetAttributes(self.GetFieldData(), dataset=self, association=FieldAssociation.NONE)

    def clear_field_arrays(self):
        """Remove all field arrays."""
        self.field_arrays.clear()

    @property
    def memory_address(self):
        """Get address of the underlying C++ object in format 'Addr=%p'."""
        return self.GetInformation().GetAddressAsString("")
