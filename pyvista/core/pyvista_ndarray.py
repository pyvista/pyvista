"""Contains pyvista_ndarray a numpy ndarray type used in pyvista."""

import numpy as np
from pyvista.utilities.helpers import convert_array, FieldAssociation
from vtk.numpy_interface.dataset_adapter import VTKObjectWrapper, VTKArray
from vtk.vtkCommonCore import vtkWeakReference

try:
    from vtk.vtkCommonKitPython import vtkDataArray, vtkAbstractArray
except (ModuleNotFoundError, ImportError):
    from vtk.vtkCommonCore import vtkDataArray, vtkAbstractArray


class pyvista_ndarray(np.ndarray):
    """Link a numpy array with the vtk object the data is attached to.

    When the array is changed it triggers "Modified()" which updates
    all upstream objects, including any render windows holding the
    object.
    """

    def __new__(cls, input_array, proxy):
        """Allocate memory for the pyvista ndarray."""
        obj = np.asarray(input_array).view(cls)
        cls.proxy = proxy
        return obj

    def __array_finalize__(self, obj):
        """Customize array at creation."""
        if obj is None:
            return

    def __setitem__(self, coords, value):
        """Update the array and update the vtk object."""
        super(pyvista_ndarray, self).__setitem__(coords, value)
        self.proxy.Modified()

