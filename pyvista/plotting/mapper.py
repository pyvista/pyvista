"""An internal module for wrapping the use of mappers."""


def make_mapper(mapper_class):
    """Wrap a mapper.

    This makes a mapper wrapped with a few convenient tools for managing
    mappers with scalar bars in a consistent way since not all mapper classes
    have scalar ranges and lookup tables.
    """

    class MapperHelper(mapper_class):
        """A helper that dynamically inherits the mapper's class."""

        def __init__(self, *args, **kwargs):
            self._scalar_range = None
            self._lut = None

        @property
        def scalar_range(self):
            if hasattr(self, 'GetScalarRange'):
                self._scalar_range = self.GetScalarRange()
            return self._scalar_range

        @scalar_range.setter
        def scalar_range(self, clim):
            if hasattr(self, 'SetScalarRange'):
                self.SetScalarRange(*clim)
            if self.lookup_table is not None:
                self.lookup_table.SetRange(*clim)
            self._scalar_range = clim

        @property
        def lookup_table(self):
            if hasattr(self, 'GetLookupTable'):
                self._lut = self.GetLookupTable()
            return self._lut

        @lookup_table.setter
        def lookup_table(self, lut):
            if hasattr(self, 'SetLookupTable'):
                self.SetLookupTable(lut)
            self._lut = lut

    return MapperHelper()
