"""An internal module for wrapping the use of mappers."""
import logging
import sys

import numpy as np

from pyvista import _vtk
from pyvista.utilities import convert_array, convert_string_array, raise_not_matching

from ._plotting import _has_matplotlib
from .colors import get_cmap_safe
from .tools import normalize, parse_color


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

        def set_scalars(self, mesh, scalars, scalar_bar_args, rgb,
                        component, preference, interpolate_before_map,
                        _custom_opac, annotations, log_scale,
                        nan_color, above_color, below_color, cmap,
                        flip_scalars, opacity, categories, n_colors,
                        clim, theme, show_scalar_bar):
            """Set the scalars on this mapper."""
            if cmap is None:  # Set default map if matplotlib is available
                if _has_matplotlib():
                    cmap = theme.cmap

            # Set the array title for when it is added back to the mesh
            if _custom_opac:
                title = '__custom_rgba'
            else:
                title = scalar_bar_args.get('title', 'Data')

            if not isinstance(scalars, np.ndarray):
                scalars = np.asarray(scalars)

            _using_labels = False
            if not np.issubdtype(scalars.dtype, np.number):
                # raise TypeError('Non-numeric scalars are currently not supported for plotting.')
                # TODO: If str array, digitive and annotate
                cats, scalars = np.unique(scalars.astype('|S'), return_inverse=True)
                values = np.unique(scalars)
                clim = [np.min(values) - 0.5, np.max(values) + 0.5]
                title = f'{title}-digitized'
                n_colors = len(cats)
                scalar_bar_args.setdefault('n_labels', 0)
                _using_labels = True

            if rgb:
                show_scalar_bar = False
                if scalars.ndim != 2 or scalars.shape[1] < 3 or scalars.shape[1] > 4:
                    raise ValueError('RGB array must be n_points/n_cells by 3/4 in shape.')

            if scalars.ndim != 1:
                if rgb:
                    pass
                elif scalars.ndim == 2 and (scalars.shape[0] == mesh.n_points or scalars.shape[0] == mesh.n_cells):
                    if not isinstance(component, (int, type(None))):
                        raise TypeError('component must be either None or an integer')
                    if component is None:
                        scalars = np.linalg.norm(scalars.copy(), axis=1)
                        title = '{}-normed'.format(title)
                    elif component < scalars.shape[1] and component >= 0:
                        scalars = scalars[:, component].copy()
                        title = '{}-{}'.format(title, component)
                    else:
                        raise ValueError(
                            ('component must be nonnegative and less than the '
                             'dimensionality of the scalars array: {}').format(
                                 scalars.shape[1]
                             )
                        )
                else:
                    scalars = scalars.ravel()

            if scalars.dtype == np.bool_:
                scalars = scalars.astype(np.float_)

            self.configure_scalars_mode(
                scalars, mesh, title, n_colors,
                preference, interpolate_before_map, rgb,
                _custom_opac
            )
            table = self.GetLookupTable()

            if _using_labels:
                table.SetAnnotations(convert_array(values), convert_string_array(cats))

            if isinstance(annotations, dict):
                for val, anno in annotations.items():
                    table.SetAnnotation(float(val), str(anno))

            # Set scalars range
            if clim is None:
                clim = [np.nanmin(scalars), np.nanmax(scalars)]
            elif isinstance(clim, (int, float)):
                clim = [-clim, clim]

            if log_scale:
                if clim[0] <= 0:
                    clim = [sys.float_info.min, clim[1]]
                table.SetScaleToLog10()

            if np.any(clim) and not rgb:
                self.scalar_range = clim[0], clim[1]

            table.SetNanColor(nan_color)
            if above_color:
                table.SetUseAboveRangeColor(True)
                table.SetAboveRangeColor(*parse_color(above_color, opacity=1))
                scalar_bar_args.setdefault('above_label', 'Above')
            if below_color:
                table.SetUseBelowRangeColor(True)
                table.SetBelowRangeColor(*parse_color(below_color, opacity=1))
                scalar_bar_args.setdefault('below_label', 'Below')

            if cmap is not None:
                # have to add the attribute to pass it onward to some classes
                if isinstance(cmap, str):
                    self.cmap = cmap
                # ipygany uses different colormaps
                if theme.jupyter_backend == 'ipygany':
                    from ..jupyter.pv_ipygany import check_colormap
                    check_colormap(cmap)
                else:
                    if not _has_matplotlib():
                        cmap = None
                        logging.warning('Please install matplotlib for color maps.')

                    cmap = get_cmap_safe(cmap)
                    if categories:
                        if categories is True:
                            n_colors = len(np.unique(scalars))
                        elif isinstance(categories, int):
                            n_colors = categories
                    ctable = cmap(np.linspace(0, 1, n_colors))*255
                    ctable = ctable.astype(np.uint8)
                    # Set opactities
                    if isinstance(opacity, np.ndarray) and not _custom_opac:
                        ctable[:, -1] = opacity
                    if flip_scalars:
                        ctable = np.ascontiguousarray(ctable[::-1])
                    table.SetTable(_vtk.numpy_to_vtk(ctable))
                    if _custom_opac:
                        # need to round the colors here since we're
                        # directly displaying the colors
                        hue = normalize(scalars, minimum=clim[0], maximum=clim[1])
                        scalars = np.round(hue*n_colors)/n_colors
                        scalars = cmap(scalars)*255
                        scalars[:, -1] *= opacity
                        scalars = scalars.astype(np.uint8)
                        self.configure_scalars_mode(
                            scalars, mesh, title, n_colors,
                            preference, interpolate_before_map, rgb,
                            _custom_opac
                        )

            else:  # no cmap specified
                if flip_scalars:
                    table.SetHueRange(0.0, 0.66667)
                else:
                    table.SetHueRange(0.66667, 0.0)

            return show_scalar_bar, n_colors, clim

        def configure_scalars_mode(self, scalars, mesh, title, n_colors,
                                   preference, interpolate_before_map, rgb,
                                   _custom_opac):
            """Configure scalar mode."""
            if (scalars.shape[0] == mesh.n_points and
                scalars.shape[0] == mesh.n_cells):
                use_points = preference == 'point'
                use_cells = not use_points
            else:
                use_points = scalars.shape[0] == mesh.n_points
                use_cells = scalars.shape[0] == mesh.n_cells

            # Scalars interpolation approach
            if use_points:
                mesh.point_data.set_array(scalars, title, True)
                mesh.active_scalars_name = title
                self.SetScalarModeToUsePointData()
            elif use_cells:
                mesh.cell_data.set_array(scalars, title, True)
                mesh.active_scalars_name = title
                self.SetScalarModeToUseCellData()
            else:
                raise_not_matching(scalars, mesh)

            self.GetLookupTable().SetNumberOfTableValues(n_colors)
            if interpolate_before_map:
                self.InterpolateScalarsBeforeMappingOn()
            if rgb or _custom_opac:
                self.SetColorModeToDirectScalars()
            else:
                self.SetColorModeToMapScalars()

        def set_custom_opacity(self, opacity, color, mesh, n_colors,
                               preference, interpolate_before_map, rgb, theme):
            """Set custom opacity."""
            # create a custom RGBA array to supply our opacity to
            rgb_color = parse_color(color, default_color=theme.color)
            if (opacity.size == mesh.n_points and opacity.size == mesh.n_cells):
                if preference == 'points':
                    rgba = np.empty((mesh.n_points, 4), np.uint8)
                else:
                    rgba = np.empty((mesh.n_cells, 4), np.uint8)
            elif opacity.size == mesh.n_points:
                rgba = np.empty((mesh.n_points, 4), np.uint8)
            elif opacity.size == mesh.n_cells:
                rgba = np.empty((mesh.n_cells, 4), np.uint8)
            else:
                raise ValueError(
                    f"Opacity array size ({opacity.size}) does not equal "
                    f"the number of points {mesh.n_points} or the "
                    f"number of cells ({mesh.n_cells}).")

            rgb_color = np.array(parse_color(color, default_color=theme.color))*255
            rgba[:, :-1] = rgb_color
            rgba[:, -1] = opacity*255

            self.configure_scalars_mode(rgba, mesh, '',
                                        n_colors, preference,
                                        interpolate_before_map, rgb,
                                        True)
            self.SetColorModeToDirectScalars()

    return MapperHelper()
