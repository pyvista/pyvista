"""A transitory module to synchronize the deprecation of default `transform` inplace=True.

Both `DataSetFilters.transform` and `CompositeFilters.transform` use the same `inplace=True`
default. We want to deprecate both simultaneously.
"""

from __future__ import annotations

import warnings

from pyvista import PyVistaDeprecationWarning
from pyvista._version import version_info


def check_inplace(cls: type, inplace: bool | None = None) -> bool:  # numpydoc ignore=RT01
    """Check if user explicitly opted into inplace behavior."""
    # Deprecated v0.45, convert to error in v0.48, remove v0.51
    if inplace is None:
        # if inplace is None user has not explicitly opted into inplace behavior

        if version_info >= (0, 48):  # pragma: no cover
            msg = (
                'Convert this deprecation warning into an error. '
                'Also update docstrs for `DataSetFilters.transform` and `CompositeFilters.transform`.'
            )
            raise RuntimeError(msg)
        if version_info >= (0, 51):  # pragma: no cover
            msg = (
                'Delete this horrid package. '
                'Also update docstrs for `DataSetFilters.transform` and `CompositeFilters.transform`'
            )
            raise RuntimeError(msg)

        msg = (
            f'The default value of `inplace` for the filter `{cls.__name__}.transform` will change in the future. '
            'Previously it defaulted to `True`, but will change to `False`. '
            'Explicitly set `inplace` to `True` or `False` to silence this warning.'
        )
        warnings.warn(msg, PyVistaDeprecationWarning)
        inplace = True  # The old default behavior

    return inplace
