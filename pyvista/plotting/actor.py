"""Wrap :vtk:`vtkActor` module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal
from typing import get_args

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista._warn_external import warn_external

from . import _vtk
from . import _vtk_gl
from ._property import Property
from .prop3d import Prop3D

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from .mapper import _BaseMapper

ShaderType = Literal['vertex', 'fragment', 'geometry']
PointSpriteShape = Literal['circle', 'triangle', 'hexagon', 'diamond', 'asterisk', 'star']

_SHADER_TYPE_NAMES: tuple[str, ...] = get_args(ShaderType)

_POINT_SPRITE_SHADERS: dict[str, str] = {
    'circle': (
        '//VTK::Color::Impl\n'
        'vec2 p = gl_PointCoord * 2.0 - 1.0;\n'
        'float d = dot(p, p);\n'
        'if (d > 1.0) {\n'
        '  discard;\n'
        '}\n'
    ),
    'triangle': (
        '//VTK::Color::Impl\n'
        'vec2 p = gl_PointCoord * 2.0 - 1.0;\n'
        'float a = 0.5;\n'
        'float b = 0.8660254;\n'
        'vec2 v0 = vec2(0.0, a);\n'
        'vec2 v1 = vec2(-b, -a);\n'
        'vec2 v2 = vec2(b, -a);\n'
        'float area = abs((v1.x - v0.x)*(v2.y - v0.y) - (v2.x - v0.x)*(v1.y - v0.y));\n'
        'float a1 = abs((v1.x - p.x)*(v2.y - p.y) - (v2.x - p.x)*(v1.y - p.y)) / area;\n'
        'float a2 = abs((v2.x - p.x)*(v0.y - p.y) - (v0.x - p.x)*(v2.y - p.y)) / area;\n'
        'float a3 = abs((v0.x - p.x)*(v1.y - p.y) - (v1.x - p.x)*(v0.y - p.y)) / area;\n'
        'if ((a1 + a2 + a3) > 1.01) {\n'
        '  discard;\n'
        '}\n'
    ),
    'hexagon': (
        '//VTK::Color::Impl\n'
        'vec2 p = gl_PointCoord * 2.0 - 1.0;\n'
        'p = abs(p);\n'
        'if (p.x > 1.0 || p.y > 1.0) {\n'
        '  discard;\n'
        '}\n'
        'if (p.x + 0.577 * p.y > 1.0) {\n'
        '  discard;\n'
        '}\n'
    ),
    'diamond': (
        '//VTK::Color::Impl\n'
        'vec2 p = gl_PointCoord * 2.0 - 1.0;\n'
        'if (abs(p.x) + abs(p.y) > 1.0) {\n'
        '  discard;\n'
        '}\n'
    ),
    'asterisk': (
        '//VTK::Color::Impl\n'
        'vec2 p = gl_PointCoord * 2.0 - 1.0;\n'
        'float r = length(p);\n'
        'if (r > 1.0) discard;\n'
        'float theta = atan(p.y, p.x);\n'
        'float N = 5.0;\n'
        'float inner = 0.5;\n'
        'float star_radius = mix(1.0, inner, abs(cos(N * theta)));\n'
        'if (r > star_radius)\n'
        '  discard;\n'
    ),
    'star': (
        '//VTK::Color::Impl\n'
        'vec2 p = gl_PointCoord * 2.0 - 1.0;\n'
        'float r = length(p);\n'
        'if (r > 1.0) discard;\n'
        'float theta = atan(p.y, p.x);\n'
        'theta = mod(theta + 6.2831853 + 1.5707963, 6.2831853);\n'
        'float N = 5.0;\n'
        'float inner = 0.4;\n'
        'float radial = pow(0.5 * (1.0 + cos(N * theta)), 3.0);\n'
        'float radius = mix(inner, 1.0, radial);\n'
        'if (r > radius)\n'
        '  discard;\n'
    ),
}


class Actor(Prop3D, _vtk.vtkActor):
    """Wrap :vtk:`vtkActor`.

    This class represents the geometry & properties in a rendered
    scene. Normally, a :class:`pyvista.Actor` is constructed from
    :func:`pyvista.Plotter.add_mesh`, but there may be times when it is more
    convenient to construct an actor directly from a
    :class:`pyvista.DataSetMapper`.

    Parameters
    ----------
    mapper : pyvista.DataSetMapper, optional
        DataSetMapper.

    prop : pyvista.Property, optional
        Property of the actor.

    name : str, optional
        The name of this actor used when tracking on a plotter.

    Examples
    --------
    Create an actor without using :class:`pyvista.Plotter`.

    >>> import pyvista as pv
    >>> mesh = pv.Sphere()
    >>> mapper = pv.DataSetMapper(mesh)
    >>> actor = pv.Actor(mapper=mapper)
    >>> actor
    Actor (...)
      Center:                     (0.0, 0.0, 0.0)
      Pickable:                   True
      Position:                   (0.0, 0.0, 0.0)
      Scale:                      (1.0, 1.0, 1.0)
      Visible:                    True
      X Bounds                    -4.993E-01, 4.993E-01
      Y Bounds                    -4.965E-01, 4.965E-01
      Z Bounds                    -5.000E-01, 5.000E-01
      User matrix:                Identity
      Has mapper:                 True
    ...

    Change the actor properties and plot the actor.

    >>> import pyvista as pv
    >>> mesh = pv.Sphere()
    >>> mapper = pv.DataSetMapper(mesh)
    >>> actor = pv.Actor(mapper=mapper)
    >>> actor.prop.color = 'blue'
    >>> actor.plot()

    Create an actor using the :class:`pyvista.Plotter` and then change the
    visibility of the actor.

    >>> import pyvista as pv
    >>> pl = pv.Plotter()
    >>> mesh = pv.Sphere()
    >>> actor = pl.add_mesh(mesh)
    >>> actor.visibility = False
    >>> actor.visibility
    False

    """

    def __init__(self, mapper=None, prop=None, name=None) -> None:
        """Initialize actor."""
        super().__init__()
        if mapper is not None:
            self.mapper = mapper
        if prop is None:
            self.prop = Property()
        else:
            self.prop = prop
        self._name = name
        self._shader_replacements: dict[str, list[tuple[str, str, bool]]] = {}

    @property
    def mapper(self) -> _BaseMapper:  # numpydoc ignore=RT01
        """Return or set the mapper of the actor.

        Examples
        --------
        Create an actor and assign a mapper to it.

        >>> import pyvista as pv
        >>> dataset = pv.Sphere()
        >>> actor = pv.Actor()
        >>> actor.mapper = pv.DataSetMapper(dataset)
        >>> actor.mapper
        DataSetMapper (...)
          Scalar visibility:           True
          Scalar range:                (0.0, 1.0)
          Interpolate before mapping:  True
          Scalar map mode:             default
          Color mode:                  direct
        <BLANKLINE>
        Attached dataset:
        PolyData (...)
          N Cells:    1680
          N Points:   842
          N Strips:   0
          X Bounds:   -4.993e-01, 4.993e-01
          Y Bounds:   -4.965e-01, 4.965e-01
          Z Bounds:   -5.000e-01, 5.000e-01
          N Arrays:   1

        """
        return self.GetMapper()  # type: ignore[return-value]

    @mapper.setter
    def mapper(self, obj) -> None:
        self.SetMapper(obj)

    @property
    def prop(self):  # numpydoc ignore=RT01
        """Return or set the property of this actor.

        Examples
        --------
        Modify the properties of an actor after adding a dataset to the plotter.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> prop = actor.prop
        >>> prop.diffuse = 0.6
        >>> pl.show()

        """
        return self.GetProperty()

    @prop.setter
    def prop(self, obj: Property) -> None:
        self.SetProperty(obj)

    @property
    def texture(self):  # numpydoc ignore=RT01
        """Return or set the actor texture.

        Notes
        -----
        The mapper dataset must have texture coordinates for the texture to be
        used.

        Examples
        --------
        Create an actor and add a texture to it. Note how the
        :class:`pyvista.PolyData` has texture coordinates by default.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> plane = pv.Plane()
        >>> plane.active_texture_coordinates is not None
        True
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(plane)
        >>> actor.texture = examples.download_masonry_texture()
        >>> actor.texture
        Texture (...)
          Components:   3
          Cube Map:     False
          Dimensions:   256, 256

        """
        return self.GetTexture()

    @texture.setter
    def texture(self, obj) -> None:
        self.SetTexture(obj)

    @property
    def memory_address(self):  # numpydoc ignore=RT01
        """Return the memory address of this actor."""
        return self.GetAddressAsString('')

    @property
    def pickable(self) -> bool:  # numpydoc ignore=RT01
        """Return or set actor pickability.

        Examples
        --------
        Create an actor using the :class:`pyvista.Plotter` and then make the
        actor unpickable.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> actor.pickable = False
        >>> actor.pickable
        False

        """
        return bool(self.GetPickable())

    @pickable.setter
    def pickable(self, value) -> None:
        self.SetPickable(value)

    @property
    def force_opaque(self) -> bool:  # numpydoc ignore=RT01
        """Return or set actor opacity behavior.

        .. versionadded:: 0.48

        Examples
        --------
        Create an actor using the :class:`pyvista.Plotter` and then force the
        actor to be opaque.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> actor.force_opaque
        False
        >>> actor.force_opaque = True
        >>> actor.force_opaque
        True

        """
        return bool(self.GetForceOpaque())

    @force_opaque.setter
    def force_opaque(self, value: bool) -> None:
        self.SetForceOpaque(value)

    @property
    def visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set actor visibility.

        See Also
        --------
        use_bounds
        pyvista.Plotter.compute_bounds

        Examples
        --------
        Create an actor using the :class:`pyvista.Plotter` and then change the
        visibility of the actor.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(mesh)
        >>> pl.bounds
        BoundsTuple(x_min =  139.06100463867188,
                    x_max = 1654.9300537109375,
                    y_min =   32.09429931640625,
                    y_max = 1319.949951171875,
                    z_min =  -17.741199493408203,
                    z_max =  282.1300048828125)

        >>> actor.visibility = False
        >>> pl.bounds
        BoundsTuple(x_min = -1.0,
                    x_max =  1.0,
                    y_min = -1.0,
                    y_max =  1.0,
                    z_min = -1.0,
                    z_max =  1.0)

        """
        return bool(self.GetVisibility())

    @visibility.setter
    def visibility(self, value: bool) -> None:
        self.SetVisibility(value)

    @property
    def use_bounds(self) -> bool:  # numpydoc ignore=RT01
        """Return or set the use of actor's bounds.

        .. versionadded:: 0.45

        See Also
        --------
        visibility
        pyvista.Plotter.compute_bounds

        Examples
        --------
        Create an actor using the :class:`pyvista.Plotter` and then change the
        use of bounds for the actor.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(mesh)
        >>> pl.bounds
        BoundsTuple(x_min =  139.06100463867188,
                    x_max = 1654.9300537109375,
                    y_min =   32.09429931640625,
                    y_max = 1319.949951171875,
                    z_min =  -17.741199493408203,
                    z_max =  282.1300048828125)

        >>> actor.use_bounds = False
        >>> pl.bounds
        BoundsTuple(x_min = -1.0,
                    x_max =  1.0,
                    y_min = -1.0,
                    y_max =  1.0,
                    z_min = -1.0,
                    z_max =  1.0)

        Although the actor's bounds are no longer used, the actor remains visible.

        >>> actor.visibility
        True

        """
        return bool(self.GetUseBounds())

    @use_bounds.setter
    def use_bounds(self, value: bool) -> None:
        self.SetUseBounds(value)

    def plot(self, **kwargs) -> None:
        """Plot just the actor.

        This may be useful when interrogating or debugging individual actors.

        Parameters
        ----------
        **kwargs : dict, optional
            Optional keyword arguments passed to :func:`pyvista.Plotter.show`.

        Examples
        --------
        Create an actor without the :class:`pyvista.Plotter`, change its
        properties, and plot it.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mapper = pv.DataSetMapper(mesh)
        >>> actor = pv.Actor(mapper=mapper)
        >>> actor.prop.color = 'red'
        >>> actor.prop.show_edges = True
        >>> actor.plot()

        """
        pl = pv.Plotter()
        pl.add_actor(self)
        pl.show(**kwargs)

    @_deprecate_positional_args
    def copy(self: Self, deep: bool = True) -> Self:  # noqa: FBT001, FBT002
        """Create a copy of this actor.

        Parameters
        ----------
        deep : bool, default: True
            Create a shallow or deep copy of the actor. A deep copy will have a
            new property and mapper, while a shallow copy will use the mapper
            and property of this actor.

        Returns
        -------
        Actor
            Deep or shallow copy of this actor.

        Examples
        --------
        Create an actor of a cube by adding it to a :class:`~pyvista.Plotter`
        and then copy the actor, change the properties, and add it back to the
        :class:`~pyvista.Plotter`.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(mesh, color='b')
        >>> new_actor = actor.copy()
        >>> new_actor.prop.style = 'wireframe'
        >>> new_actor.prop.line_width = 5
        >>> new_actor.prop.color = 'r'
        >>> new_actor.prop.lighting = False
        >>> _ = pl.add_actor(new_actor)
        >>> pl.show()

        """
        new_actor = type(self)()
        if deep:
            if self.mapper is not None:
                new_actor.mapper = self.mapper.copy()
            new_actor.prop = self.prop.copy()
        else:
            new_actor.ShallowCopy(self)
        return new_actor

    def __repr__(self):
        """Representation of the actor."""
        mat_info = 'Identity' if np.array_equal(self.user_matrix, np.eye(4)) else 'Set'
        bnd = self.bounds
        attr = [
            f'{type(self).__name__} ({hex(id(self))})',
            f'  Center:                     {self.center}',
            f'  Pickable:                   {self.pickable}',
            f'  Position:                   {self.position}',
            f'  Scale:                      {self.scale}',
            f'  Visible:                    {self.visibility}',
            f'  X Bounds                    {bnd[0]:.3E}, {bnd[1]:.3E}',
            f'  Y Bounds                    {bnd[2]:.3E}, {bnd[3]:.3E}',
            f'  Z Bounds                    {bnd[4]:.3E}, {bnd[5]:.3E}',
            f'  User matrix:                {mat_info}',
            f'  Has mapper:                 {self.mapper is not None}',
            '',
            repr(self.prop),
        ]
        if self.mapper is not None:
            attr.append('')
            attr.append(repr(self.mapper))
        return '\n'.join(attr)

    @property
    def backface_prop(self) -> Property | None:  # numpydoc ignore=RT01
        """Return or set the backface property.

        By default this property matches the frontface property
        :attr:`Actor.prop`. Once accessed or modified, this backface
        property becomes independent of the frontface property. In
        order to restore the fallback to frontface property, assign
        ``None`` to the property.

        Returns
        -------
        pyvista.Property
            The object describing backfaces.

        See Also
        --------
        :ref:`backface_prop_example`

        Examples
        --------
        Clip a sphere by a plane and color the inside of the clipped sphere
        light blue using the ``backface_prop``.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> plane = pv.Plane(i_size=1.5, j_size=1.5)
        >>> mesh = pv.Sphere().clip_surface(plane, invert=False)
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(mesh, smooth_shading=True)
        >>> actor.backface_prop.color = 'lightblue'
        >>> _ = pl.add_mesh(
        ...     plane,
        ...     opacity=0.25,
        ...     show_edges=True,
        ...     color='grey',
        ...     lighting=False,
        ... )
        >>> pl.show()

        """
        if self.GetBackfaceProperty() is None:
            self.SetBackfaceProperty(self.prop.copy())
        return self.GetBackfaceProperty()  # type: ignore[return-value]

    @backface_prop.setter
    def backface_prop(self, value: Property) -> None:
        self.SetBackfaceProperty(value)

    def add_shader_replacement(
        self,
        shader_type: ShaderType,
        original: str,
        replacement: str,
        *,
        replace_first: bool = True,
        replace_all: bool = False,
        _feature_name: str = '_user',
    ) -> None:
        r"""Add a GLSL shader replacement to this actor.

        .. versionadded:: 0.48

        This wraps VTK's shader replacement API, providing conflict detection
        and tracking so that multiple independent shader features can coexist
        safely on the same actor.

        Parameters
        ----------
        shader_type : str
            Type of shader to modify. One of ``'vertex'``, ``'fragment'``,
            or ``'geometry'``.

        original : str
            The VTK shader tag to replace (e.g., ``'//VTK::Color::Impl'``).

        replacement : str
            The GLSL replacement code.

        replace_first : bool, default: True
            Whether the replacement is applied before VTK's standard
            shader substitutions.

        replace_all : bool, default: False
            Whether to replace all occurrences of ``original``.

        _feature_name : str, default: '_user'
            Internal key used to track which feature owns this replacement.
            End users should not need to change this. Built-in features
            use dedicated keys like ``'mip'`` and ``'point_sprite'``.

        Raises
        ------
        ValueError
            If ``shader_type`` is invalid or if another feature already
            targets the same shader tag.

        Examples
        --------
        Add a custom fragment shader that discards pixels outside a circle.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(mesh, style='points', point_size=20)
        >>> actor.add_shader_replacement(
        ...     'fragment',
        ...     '//VTK::Color::Impl',
        ...     '//VTK::Color::Impl\n'
        ...     'vec2 p = gl_PointCoord * 2.0 - 1.0;\n'
        ...     'if (dot(p, p) > 1.0) discard;\n',
        ... )

        """
        if shader_type not in _SHADER_TYPE_NAMES:
            valid = ', '.join(sorted(_SHADER_TYPE_NAMES))
            msg = f'Invalid shader_type {shader_type!r}. Must be one of: {valid}'
            raise ValueError(msg)

        vtk_enum = getattr(_vtk_gl.vtkShader, shader_type.capitalize())
        key = (shader_type, original, replace_first)
        registry = self._shader_replacements

        # Check for conflicts with other features
        for other_feature, entries in registry.items():
            if other_feature != _feature_name and key in entries:
                msg = (
                    f'Shader replacement conflict: feature {other_feature!r} '
                    f'already targets ({shader_type!r}, {original!r}, '
                    f'replace_first={replace_first}). Clear it first with '
                    f'clear_shader_replacements(_feature_name={other_feature!r}).'
                )
                raise ValueError(msg)

        # GetShaderProperty() returns vtkShaderProperty at the type-stub level,
        # but the runtime object is vtkOpenGLShaderProperty which has these methods.
        shader_prop = self.GetShaderProperty()

        # If this feature already owns this slot, clear the old one first
        if _feature_name in registry and key in registry[_feature_name]:
            shader_prop.ClearShaderReplacement(  # type: ignore[attr-defined]
                vtk_enum,
                original,
                replace_first,
            )
            registry[_feature_name].remove(key)

        # Add the replacement
        shader_prop.AddShaderReplacement(  # type: ignore[attr-defined]
            vtk_enum,
            original,
            replace_first,
            replacement,
            replace_all,
        )

        # Register it
        registry.setdefault(_feature_name, []).append(key)

    def clear_shader_replacements(self, *, _feature_name: str | None = None) -> None:
        """Clear shader replacements from this actor.

        .. versionadded:: 0.48

        Parameters
        ----------
        _feature_name : str, optional
            If specified, only clear replacements registered under this
            feature name. If ``None``, clear all shader replacements.

        Examples
        --------
        Clear all shader replacements from an actor.

        >>> import pyvista as pv
        >>> actor = pv.Plotter().add_mesh(pv.Sphere())
        >>> actor.clear_shader_replacements()

        """
        registry = self._shader_replacements
        shader_prop = self.GetShaderProperty()

        if _feature_name is None:
            shader_prop.ClearAllShaderReplacements()
            registry.clear()
        elif _feature_name in registry:
            for shader_type_name, original, replace_first in registry[_feature_name]:
                vtk_enum = getattr(_vtk_gl.vtkShader, shader_type_name.capitalize())
                shader_prop.ClearShaderReplacement(  # type: ignore[attr-defined]
                    vtk_enum,
                    original,
                    replace_first,
                )
            del registry[_feature_name]

    def enable_maximum_intensity_projection(
        self,
        clim: Sequence[float] | None = None,
    ) -> None:
        """Enable maximum intensity projection.

        .. versionadded:: 0.48

        This resets the z screen coordinates so that vertices with higher
        scalar values are rendered closer to the screen, regardless of their
        actual 3D position. This is useful for dense point cloud visualization
        where high-value data points should be visible even when occluded by
        lower-value points.

        Scalar values are normalized to the ``[-1, 0]`` range to stay within
        the OpenGL clip space.

        Parameters
        ----------
        clim : sequence[float], optional
            Two-element sequence ``(min, max)`` specifying the scalar range
            for normalization. If not provided, the range is computed from
            the active scalars on the actor's dataset.

        Raises
        ------
        ValueError
            If no mapper, dataset, or active scalars are available and
            ``clim`` is not provided.

        Warnings
        --------
        Maximum Intensity Projection does not work correctly with
        ``opacity < 1`` unless depth peeling is enabled. See
        :func:`pyvista.Plotter.enable_depth_peeling`.

        References
        ----------
        Cowan, E.J., 2014. 'X-ray Plunge Projection' - Understanding
        Structural Geology from Grade Data. AusIMM Monograph 30: Mineral
        Resource and Ore Reserve Estimation - The AusIMM Guide to Good
        Practice, second edition, 207-220.

        Examples
        --------
        Enable maximum intensity projection on a point cloud actor.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> rng = np.random.default_rng(0)
        >>> cloud = pv.PolyData(rng.random((1000, 3)))
        >>> cloud['values'] = cloud.points[:, 2]
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(cloud, scalars='values', style='points')
        >>> actor.enable_maximum_intensity_projection()

        """
        if clim is not None:
            min_val, max_val = float(clim[0]), float(clim[1])
        else:
            # Use GetMapper() instead of self.mapper to allow the None check;
            # the mapper property is typed as _BaseMapper (never None) but can
            # be None at runtime when no mapper has been assigned.
            mapper = self.GetMapper()
            if mapper is None:
                msg = 'Actor must have a mapper to enable MIP without explicit clim.'
                raise ValueError(msg)
            dataset = mapper.dataset
            if dataset is None:
                msg = 'Mapper must have a dataset to enable MIP without explicit clim.'
                raise ValueError(msg)
            scalars = dataset.active_scalars
            if scalars is None:
                msg = (
                    'Dataset must have active scalars to enable MIP without '
                    'explicit clim. Set scalars on your dataset or pass clim=(min, max).'
                )
                raise ValueError(msg)
            min_val = float(np.nanmin(scalars))
            max_val = float(np.nanmax(scalars))

        if self.prop.opacity < 1.0:
            warn_external(
                'Maximum Intensity Projection does not work correctly with '
                'opacity < 1.0 unless depth peeling is enabled. See '
                'pyvista.Plotter.enable_depth_peeling().'
            )

        denom = max_val - min_val
        if abs(denom) < 1e-12:
            denom = 1.0

        glsl_code = (
            f'float _mip_norm = (colorTCoord[0] - {min_val}) / {denom};\n'
            'gl_Position.z = -_mip_norm;\n'
            '//VTK::LineWidthGLES30::Impl\n'
        )

        self.add_shader_replacement(
            'vertex',
            '//VTK::LineWidthGLES30::Impl',
            glsl_code,
            replace_first=True,
            replace_all=False,
            _feature_name='mip',
        )

    def disable_maximum_intensity_projection(self) -> None:
        """Disable maximum intensity projection.

        .. versionadded:: 0.48

        Clears the vertex shader replacement that reorders z-coordinates
        by scalar value, restoring normal depth-based rendering.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> rng = np.random.default_rng(0)
        >>> cloud = pv.PolyData(rng.random((1000, 3)))
        >>> cloud['values'] = cloud.points[:, 2]
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(cloud, scalars='values', style='points')
        >>> actor.enable_maximum_intensity_projection()
        >>> actor.disable_maximum_intensity_projection()

        """
        self.clear_shader_replacements(_feature_name='mip')

    def set_point_sprite_shape(self, shape: PointSpriteShape) -> None:
        """Set a custom point sprite shape via fragment shader.

        .. versionadded:: 0.48

        Replaces the default square point rendering with a custom shape
        defined by a GLSL fragment shader. This uses the ``discard``
        instruction to clip fragments outside the desired shape boundary.

        Parameters
        ----------
        shape : str
            The sprite shape to use. Must be one of:

            * ``'circle'`` - Circular disc
            * ``'triangle'`` - Upward-pointing triangle
            * ``'hexagon'`` - Regular hexagon
            * ``'diamond'`` - Diamond (rotated square)
            * ``'asterisk'`` - Five-pointed asterisk
            * ``'star'`` - Five-pointed star

        Raises
        ------
        ValueError
            If ``shape`` is not one of the supported shapes.

        Notes
        -----
        Point sprite shapes only produce visible results when
        ``render_points_as_spheres=False`` and ``style='points'``.
        When ``render_points_as_spheres=True``, VTK uses a different
        rendering path that bypasses the fragment shader.

        Examples
        --------
        Render points as circles instead of squares.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> cloud = pv.PolyData(np.random.default_rng(0).random((100, 3)))
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(
        ...     cloud,
        ...     style='points',
        ...     render_points_as_spheres=False,
        ...     point_size=20,
        ... )
        >>> actor.set_point_sprite_shape('circle')

        """
        if shape not in _POINT_SPRITE_SHADERS:
            valid = ', '.join(sorted(_POINT_SPRITE_SHADERS))
            msg = f'Invalid point sprite shape {shape!r}. Must be one of: {valid}'
            raise ValueError(msg)

        self.add_shader_replacement(
            'fragment',
            '//VTK::Color::Impl',
            _POINT_SPRITE_SHADERS[shape],
            replace_first=True,
            replace_all=False,
            _feature_name='point_sprite',
        )

    def clear_point_sprite_shape(self) -> None:
        """Clear the custom point sprite shape.

        .. versionadded:: 0.48

        Restores the default square point rendering by removing the
        fragment shader replacement.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> cloud = pv.PolyData(np.random.default_rng(0).random((100, 3)))
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(cloud, style='points', point_size=20)
        >>> actor.set_point_sprite_shape('circle')
        >>> actor.clear_point_sprite_shape()

        """
        self.clear_shader_replacements(_feature_name='point_sprite')
