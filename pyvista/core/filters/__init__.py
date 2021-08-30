"""These classes hold methods to apply general filters to any data type.

By inheriting these classes into the wrapped VTK data structures, a user
can easily apply common filters in an intuitive manner.

Example
-------
>>> import pyvista
>>> from pyvista import examples
>>> dataset = examples.load_uniform()

>>> # Threshold
>>> thresh = dataset.threshold([100, 500])

>>> # Slice
>>> slc = dataset.slice()

>>> # Clip
>>> clp = dataset.clip(invert=True)

>>> # Contour
>>> iso = dataset.contour()

"""

import pyvista
from pyvista.utilities import wrap, ProgressMonitor


def _update_alg(alg, progress_bar=False, message=''):
    """Update an algorithm with or without a progress bar."""
    if progress_bar:
        with ProgressMonitor(alg, message=message):
            alg.Update()
    else:
        alg.Update()


def _get_output(algorithm, iport=0, iconnection=0, oport=0, active_scalars=None,
                active_scalars_field='point'):
    """Get the algorithm's output and copy input's pyvista meta info."""
    ido = algorithm.GetInputDataObject(iport, iconnection)
    data = wrap(algorithm.GetOutputDataObject(oport))
    if not isinstance(data, pyvista.MultiBlock):
        data.copy_meta_from(ido)
        if not data.field_data and ido.field_data:
            data.field_data.update(ido.field_data)
        if active_scalars is not None:
            data.set_active_scalars(active_scalars, preference=active_scalars_field)
    return data


# Re-export submodules to maintain the same import paths before filters.py was split into submodules
from .data_set import DataSetFilters
from .composite import CompositeFilters
from .poly_data import PolyDataFilters
from .unstructured_grid import UnstructuredGridFilters
from .structured_grid import StructuredGridFilters
from .uniform_grid import UniformGridFilters


__all__ = [
    '_update_alg', '_get_output', 'DataSetFilters', 'CompositeFilters', 'PolyDataFilters',
    'UnstructuredGridFilters', 'StructuredGridFilters', 'UniformGridFilters',
]
