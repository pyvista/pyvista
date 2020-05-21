"""Contains pyvista_ndarray a numpy ndarray type used in pyvista."""

import numpy as np
from pyvista.utilities.helpers import convert_array, FieldAssociation
from vtk.numpy_interface.dataset_adapter import VTKObjectWrapper, VTKArray
from vtk.vtkCommonCore import vtkWeakReference
from vtk.util.numpy_support import vtk_to_numpy

try:
    from vtk.vtkCommonKitPython import vtkDataArray, vtkAbstractArray
except ImportError:
    from vtk.vtkCommonCore import vtkDataArray, vtkAbstractArray


class pyvista_ndarray(np.ndarray):
    """Link a numpy array with the vtk object the data is attached to.

    When the array is changed it triggers "Modified()" which updates
    all upstream objects, including any render windows holding the
    object.
    """

    def __new__(cls, input_array, proxy=None):
        """Allocate memory for the pyvista ndarray."""
        if proxy is None:
            if isinstance(input_array, vtkDataArray):
                cls._proxy = input_array
                input_array = convert_array(input_array)
        obj = np.asarray(input_array).view(cls)
        return obj

    def __setitem__(self, coords, value):
        """Update the array and update the vtk object."""
        super(pyvista_ndarray, self).__setitem__(coords, value)
        self._proxy.Modified()

    def __getattr__(self, item):
        """Forward unknown attribute requests to VTK array."""
        return self._proxy.__getattribute__(item)
