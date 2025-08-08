"""A set of common filters that can be applied to any DataSet or MultiBlock."""

from __future__ import annotations

import functools
import itertools
import re
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeVar
from typing import cast
import warnings

import numpy as np

import pyvista
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista._version import version_info
from pyvista.core import _validation
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.errors import VTKVersionError
from pyvista.core.filters import _get_output
from pyvista.core.filters import _update_alg
from pyvista.core.utilities import Transform
from pyvista.core.utilities.geometric_objects import NORMALS
from pyvista.core.utilities.geometric_objects import NormalsLiteral
from pyvista.core.utilities.helpers import generate_plane
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import _reciprocal
from pyvista.core.utilities.misc import abstract_class

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyvista import DataSet
    from pyvista import MultiBlock
    from pyvista import RotationLike
    from pyvista import TransformLike
    from pyvista import VectorLike
    from pyvista import pyvista_ndarray
    from pyvista.core._typing_core import _DataSetOrMultiBlockType
    from pyvista.core._typing_core import _DataSetType
    from pyvista.core.utilities.cell_quality import _CellQualityLiteral

    _MeshType_co = TypeVar('_MeshType_co', DataSet, MultiBlock, covariant=True)


@abstract_class
class DataObjectFilters:
    """A set of common filters that can be applied to any DataSet or MultiBlock."""

    points: pyvista_ndarray

    @_deprecate_positional_args(allowed=['trans'])
    def transform(  # type: ignore[misc]  # noqa: PLR0917
        self: _MeshType_co,
        trans: TransformLike,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool | None = None,  # noqa: FBT001
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Transform this mesh with a 4x4 transform.

        .. warning::
            When using ``transform_all_input_vectors=True``, there is
            no distinction in VTK between vectors and arrays with
            three components.  This may be an issue if you have scalar
            data with three components (e.g. RGB data).  This will be
            improperly transformed as if it was vector data rather
            than scalar data.  One possible (albeit ugly) workaround
            is to store the three components as separate scalar
            arrays.

        .. warning::
            In general, transformations give non-integer results. This
            method converts integer-typed vector data to float before
            performing the transformation. This applies to the points
            array, as well as any vector-valued data that is affected
            by the transformation. To prevent subtle bugs arising from
            in-place transformations truncating the result to integers,
            this conversion always applies to the input mesh.

        .. warning::
            Shear transformations are not supported for :class:`~pyvista.ImageData` or
            :class:`~pyvista.RectilinearGrid`, and rotations are not supported for
            :class:`~pyvista.RectilinearGrid`. If present, these component(s) are removed by the
            filter. To fully support these transformations, the input should be cast to
            :class:`~pyvista.StructuredGrid` `before` applying this filter.

        .. note::
            Transforming :class:`~pyvista.ImageData` modifies its
            :class:`~pyvista.ImageData.origin`,
            :class:`~pyvista.ImageData.spacing`, and
            :class:`~pyvista.ImageData.direction_matrix` properties.

        .. deprecated:: 0.45.0
            `inplace` was previously defaulted to `True`. In the future this will change
            to `False`.

        .. versionchanged:: 0.45.0
            Transforming :class:`~pyvista.ImageData` now returns ``ImageData``.
            Previously, :class:`~pyvista.StructuredGrid` was returned.

        .. versionchanged:: 0.46.0
            Transforming :class:`~pyvista.RectilinearGrid` now returns ``RectilinearGrid``.
            Previously, :class:`~pyvista.StructuredGrid` was returned.

        Parameters
        ----------
        trans : TransformLike
            Accepts a vtk transformation object or a 4x4
            transformation matrix.

        transform_all_input_vectors : bool, default: False
            When ``True``, all arrays with three components are
            transformed. Otherwise, only the normals and vectors are
            transformed.  See the warning for more details.

        inplace : bool, default: True
            When ``True``, modifies the dataset inplace.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        DataSet | MultiBlock
            Transformed dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform
            Describe linear transformations via a 4x4 matrix.
        pyvista.Prop3D.transform
            Transform an actor.

        Examples
        --------
        Translate a mesh by ``(50, 100, 200)``.

        >>> import numpy as np
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()

        Here a 4x4 :class:`numpy.ndarray` is used, but any :class:`~pyvista.TransformLike`
        is accepted.

        >>> transform_matrix = np.array(
        ...     [
        ...         [1, 0, 0, 50],
        ...         [0, 1, 0, 100],
        ...         [0, 0, 1, 200],
        ...         [0, 0, 0, 1],
        ...     ]
        ... )
        >>> transformed = mesh.transform(transform_matrix, inplace=False)
        >>> transformed.plot(show_edges=True)

        """
        # Deprecated v0.45, convert to error in v0.48, remove v0.51
        if inplace is None:
            # if inplace is None user has not explicitly opted into inplace behavior
            if version_info >= (0, 48):  # pragma: no cover
                msg = (
                    'Convert this deprecation warning into an error '
                    'and update the docstring default value/type for inplace.'
                )
                raise RuntimeError(msg)
            if version_info >= (0, 51):  # pragma: no cover
                msg = 'Remove this deprecation and update the docstring value/type for inplace.'
                raise RuntimeError(msg)

            msg = (
                f'The default value of `inplace` for the filter '
                f'`{self.__class__.__name__}.transform` will change in the future. '
                'Previously it defaulted to `True`, but will change to `False`. '
                'Explicitly set `inplace` to `True` or `False` to silence this warning.'
            )
            warnings.warn(msg, PyVistaDeprecationWarning)
            inplace = True  # The old default behavior

        if isinstance(self, pyvista.MultiBlock):
            return self.generic_filter(
                'transform',
                trans=trans,
                transform_all_input_vectors=transform_all_input_vectors,
                inplace=inplace,
                progress_bar=progress_bar,
            )

        t = trans if isinstance(trans, Transform) else Transform(trans)

        if t.matrix[3, 3] == 0:
            msg = 'Transform element (3,3), the inverse scale term, is zero'
            raise ValueError(msg)

        # vtkTransformFilter truncates the result if the input is an integer type
        # so convert input points and relevant vectors to float
        # (creating a new copy would be harmful much more often)
        converted_ints = False
        if not np.issubdtype(self.points.dtype, np.floating):
            self.points = self.points.astype(np.float32)
            converted_ints = True
        if transform_all_input_vectors:
            # all vector-shaped data will be transformed
            point_vectors: list[str | None] = [
                name for name, data in self.point_data.items() if data.shape == (self.n_points, 3)
            ]
            cell_vectors: list[str | None] = [
                name for name, data in self.cell_data.items() if data.shape == (self.n_cells, 3)
            ]
        else:
            # we'll only transform active vectors and normals
            point_vectors = [
                self.point_data.active_vectors_name,
                self.point_data.active_normals_name,
            ]
            cell_vectors = [
                self.cell_data.active_vectors_name,
                self.cell_data.active_normals_name,
            ]
        # dynamically convert each self.point_data[name] etc. to float32
        all_vectors = [point_vectors, cell_vectors]
        all_dataset_attrs = [self.point_data, self.cell_data]
        for vector_names, dataset_attrs in zip(all_vectors, all_dataset_attrs):
            for vector_name in vector_names:
                if vector_name is None:
                    continue
                vector_arr = dataset_attrs[vector_name]
                if not np.issubdtype(vector_arr.dtype, np.floating):
                    dataset_attrs[vector_name] = vector_arr.astype(np.float32)
                    converted_ints = True
        if converted_ints:
            warnings.warn(
                'Integer points, vector and normal data (if any) of the input mesh '
                'have been converted to ``np.float32``. This is necessary in order '
                'to transform properly.',
            )

        # vtkTransformFilter doesn't respect active scalars.  We need to track this
        active_point_scalars_name: str | None = self.point_data.active_scalars_name
        active_cell_scalars_name: str | None = self.cell_data.active_scalars_name

        # vtkTransformFilter sometimes doesn't transform all vector arrays
        # when there are active point/cell scalars. Use this workaround
        self.active_scalars_name = None

        f = _vtk.vtkTransformFilter()
        f.SetInputDataObject(self)
        f.SetTransform(t)
        f.SetTransformAllInputVectors(transform_all_input_vectors)

        _update_alg(f, progress_bar=progress_bar, message='Transforming')
        vtk_filter_output = pyvista.core.filters._get_output(f)

        output = self if inplace else self.__class__()

        if isinstance(output, pyvista.ImageData):
            # vtkTransformFilter returns a StructuredGrid for legacy code (before VTK 9)
            # but VTK 9+ supports oriented images.
            # To keep an ImageData -> ImageData mapping, we copy the transformed data
            # from the filter output but manually transform the structure
            output.copy_structure(self)  # type: ignore[arg-type]
            current_matrix = output.index_to_physical_matrix
            new_matrix = pyvista.Transform(current_matrix).compose(t).matrix
            output.index_to_physical_matrix = new_matrix

            output.point_data.update(vtk_filter_output.point_data, copy=not inplace)
            output.cell_data.update(vtk_filter_output.cell_data, copy=not inplace)
            output.field_data.update(vtk_filter_output.field_data, copy=not inplace)

        elif isinstance(output, pyvista.RectilinearGrid):
            # vtkTransformFilter returns a StructuredGrid, but we can return
            # RectilinearGrid if we ignore shear and rotations
            # Follow similar decomposition performed by ImageData.index_to_physical_matrix
            T, R, N, S, K = t.decompose()

            if not np.allclose(K, np.eye(3)):
                msg = (
                    'The transformation has a shear component which has been removed. Shear is '
                    'not supported\nby RectilinearGrid; cast to StructuredGrid first to support '
                    'shear transformations.'
                )
                warnings.warn(msg)

            # Lump scale and reflection together
            scale = S * N
            if not np.allclose(np.abs(R), np.eye(3)):
                msg = (
                    'The transformation has a non-diagonal rotation component which has been '
                    'removed. Rotation is\nnot supported by RectilinearGrid; cast to '
                    'StructuredGrid first to fully support rotations.'
                )
                warnings.warn(msg)
            else:
                # Lump any reflections from the rotation into the scale
                scale *= np.diagonal(R)

            # Apply transformation to structure
            tx, ty, tz = T
            sx, sy, sz = scale
            output.x = self.x * sx + tx
            output.y = self.y * sy + ty
            output.z = self.z * sz + tz

            # Copy data arrays from the vtkTransformFilter's output
            output.point_data.update(vtk_filter_output.point_data, copy=not inplace)
            output.cell_data.update(vtk_filter_output.cell_data, copy=not inplace)
            output.field_data.update(vtk_filter_output.field_data, copy=not inplace)

        elif inplace:
            output.copy_from(vtk_filter_output, deep=False)
        else:
            # The output from the transform filter contains a shallow copy
            # of the original dataset except for the point arrays.  Here
            # we perform a copy so the two are completely unlinked.
            output.copy_from(vtk_filter_output, deep=True)

        # Make the previously active scalars active again
        self.point_data.active_scalars_name = active_point_scalars_name
        if output is not self:
            output.point_data.active_scalars_name = active_point_scalars_name
        self.cell_data.active_scalars_name = active_cell_scalars_name
        if output is not self:
            output.cell_data.active_scalars_name = active_cell_scalars_name

        return output

    @_deprecate_positional_args(allowed=['normal'])
    def reflect(  # type: ignore[misc]  # noqa: PLR0917
        self: _MeshType_co,
        normal: VectorLike[float],
        point: VectorLike[float] | None = None,
        inplace: bool = False,  # noqa: FBT001, FBT002
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Reflect a dataset across a plane.

        Parameters
        ----------
        normal : array_like[float]
            Normal direction for reflection.

        point : array_like[float]
            Point which, along with ``normal``, defines the reflection
            plane. If not specified, this is the origin.

        inplace : bool, default: False
            When ``True``, modifies the dataset inplace.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are transformed. Otherwise,
            only the points, normals and active vectors are transformed.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        DataSet | MultiBlock
            Reflected dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.reflect
            Concatenate a reflection matrix with a transformation.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> mesh = mesh.reflect((0, 0, 1), point=(0, 0, -100))
        >>> mesh.plot(show_edges=True)

        See the :ref:`reflect_example` for more examples using this filter.

        """
        t = Transform().reflect(normal, point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
            progress_bar=progress_bar,
        )

    @_deprecate_positional_args(allowed=['angle'])
    def rotate_x(  # type: ignore[misc]  # noqa: PLR0917
        self: _MeshType_co,
        angle: float,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Rotate mesh about the x-axis.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the x-axis.

        point : VectorLike[float], optional
            Point to rotate about. Defaults to origin.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        DataSet | MultiBlock
            Rotated dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.rotate_x
            Concatenate a rotation about the x-axis with a transformation.

        Examples
        --------
        Rotate a mesh 30 degrees about the x-axis.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> rot = mesh.rotate_x(30, inplace=False)

        Plot the rotated mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        t = Transform().rotate_x(angle, point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    @_deprecate_positional_args(allowed=['angle'])
    def rotate_y(  # type: ignore[misc]  # noqa: PLR0917
        self: _MeshType_co,
        angle: float,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Rotate mesh about the y-axis.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the y-axis.

        point : VectorLike[float], optional
            Point to rotate about.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are transformed. Otherwise, only
            the points, normals and active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        DataSet | MultiBlock
            Rotated dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.rotate_y
            Concatenate a rotation about the y-axis with a transformation.

        Examples
        --------
        Rotate a cube 30 degrees about the y-axis.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> rot = mesh.rotate_y(30, inplace=False)

        Plot the rotated mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        t = Transform().rotate_y(angle, point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    @_deprecate_positional_args(allowed=['angle'])
    def rotate_z(  # type: ignore[misc]  # noqa: PLR0917
        self: _MeshType_co,
        angle: float,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Rotate mesh about the z-axis.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the z-axis.

        point : VectorLike[float], optional
            Point to rotate about. Defaults to origin.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        DataSet | MultiBlock
            Rotated dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.rotate_z
            Concatenate a rotation about the z-axis with a transformation.

        Examples
        --------
        Rotate a mesh 30 degrees about the z-axis.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> rot = mesh.rotate_z(30, inplace=False)

        Plot the rotated mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        t = Transform().rotate_z(angle, point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    @_deprecate_positional_args(allowed=['vector', 'angle'])
    def rotate_vector(  # type: ignore[misc]  # noqa: PLR0917
        self: _MeshType_co,
        vector: VectorLike[float],
        angle: float,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Rotate mesh about a vector.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        vector : VectorLike[float]
            Vector to rotate about.

        angle : float
            Angle to rotate.

        point : VectorLike[float], optional
            Point to rotate about. Defaults to origin.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        DataSet | MultiBlock
            Rotated dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.rotate_vector
            Concatenate a rotation about a vector with a transformation.

        Examples
        --------
        Rotate a mesh 30 degrees about the ``(1, 1, 1)`` axis.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> rot = mesh.rotate_vector((1, 1, 1), 30, inplace=False)

        Plot the rotated mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        t = Transform().rotate_vector(vector, angle, point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    @_deprecate_positional_args(allowed=['rotation'])
    def rotate(  # type: ignore[misc]  # noqa: PLR0917
        self: _MeshType_co,
        rotation: RotationLike,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Rotate mesh about a point with a rotation matrix or ``Rotation`` object.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        rotation : RotationLike
            3x3 rotation matrix or a SciPy ``Rotation`` object.

        point : VectorLike[float], optional
            Point to rotate about. Defaults to origin.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        DataSet | MultiBlock
            Rotated dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.rotate
            Concatenate a rotation matrix with a transformation.

        Examples
        --------
        Define a rotation. Here, a 3x3 matrix is used which rotates about the z-axis by
        60 degrees.

        >>> import pyvista as pv
        >>> rotation = [
        ...     [0.5, -0.8660254, 0.0],
        ...     [0.8660254, 0.5, 0.0],
        ...     [0.0, 0.0, 1.0],
        ... ]

        Use the rotation to rotate a cone about its tip.

        >>> mesh = pv.Cone()
        >>> tip = (0.5, 0.0, 0.0)
        >>> rot = mesh.rotate(rotation, point=tip)

        Plot the rotated mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        t = Transform().rotate(rotation, point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def translate(  # type: ignore[misc]
        self: _MeshType_co,
        xyz: VectorLike[float],
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Translate the mesh.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        xyz : VectorLike[float]
            A vector of three floats.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        DataSet | MultiBlock
            Translated dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.translate
            Concatenate a translation matrix with a transformation.

        Examples
        --------
        Create a sphere and translate it by ``(2, 1, 2)``.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.center
        (0.0, 0.0, 0.0)
        >>> trans = mesh.translate((2, 1, 2), inplace=False)
        >>> trans.center
        (2.0, 1.0, 2.0)

        """
        transform = Transform().translate(xyz)
        return self.transform(
            transform,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    @_deprecate_positional_args(allowed=['xyz'])
    def scale(  # type: ignore[misc]  # noqa: PLR0917
        self: _MeshType_co,
        xyz: float | VectorLike[float],
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
        point: VectorLike[float] | None = None,
    ) -> _MeshType_co:
        """Scale the mesh.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        xyz : float | VectorLike[float]
            A vector sequence defining the scale factors along x, y, and z. If
            a scalar, the same uniform scale is used along all three axes.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are transformed. Otherwise, only
            the points, normals and active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        point : VectorLike[float], optional
            Point to scale from. Defaults to origin.

        Returns
        -------
        DataSet | MultiBlock
            Scaled dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.scale
            Concatenate a scale matrix with a transformation.

        pyvista.DataObjectFilters.resize
            Resize a mesh.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh1 = examples.download_teapot()
        >>> mesh2 = mesh1.scale([10.0, 10.0, 10.0], inplace=False)

        Plot meshes side-by-side

        >>> pl = pv.Plotter(shape=(1, 2))
        >>> # Create plot with unscaled mesh
        >>> pl.subplot(0, 0)
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.show_axes()
        >>> _ = pl.show_grid()
        >>> # Create plot with scaled mesh
        >>> pl.subplot(0, 1)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show_axes()
        >>> _ = pl.show_grid()
        >>> pl.show(cpos='xy')

        """
        transform = Transform().scale(xyz, point=point)
        return self.transform(
            transform,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def resize(  # type: ignore[misc]
        self: _MeshType_co,
        *,
        bounds: VectorLike[float] | None = None,
        bounds_size: float | VectorLike[float] | None = None,
        center: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ) -> _MeshType_co:
        """Resize the dataset's bounds.

        This filter rescales and translates the mesh to fit specified bounds. This is useful for
        normalizing datasets, changing units, or fitting datasets into specific coordinate ranges.

        Use ``bounds`` to set the mesh's :attr:`~pyvista.DataSet.bounds` directly or use
        ``bounds_size`` and ``center`` to implicitly set the new bounds.

        .. versionadded:: 0.46

        See Also
        --------
        :meth:`scale`, :meth:`translate`
            Scale and/or translate a mesh. Used internally by :meth:`resize`.

        Parameters
        ----------
        bounds : VectorLike[float], optional
            Target :attr:`~pyvista.DataSet.bounds` for the resized dataset in the format
            ``[xmin, xmax, ymin, ymax, zmin, zmax]``. If provided, the dataset is scaled and
            translated to fit exactly within these bounds. Cannot be used together with
            ``bounds_size`` or ``center``.

        bounds_size : float | VectorLike[float], optional
            Target size of the :attr:`~pyvista.DataSet.bounds` for the resized dataset. Use a
            single float to specify the size of all three axes, or a 3-element vector to set the
            size of each axis independently. Cannot be used together with ``bounds``.

        center : VectorLike[float], optional
            Center of the resized dataset in ``[x, y, z]``. By default, the mesh's
            :attr:`~pyvista.DataSet.center` is used. Only used when ``bounds_size`` is specified.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are transformed as part of the resize. Otherwise, only
            the points, normals and active vectors are transformed.

        inplace : bool, default: False
            If True, the dataset is modified in place. If False, a new dataset is returned.

        Returns
        -------
        DataSet | MultiBlock
            Resized dataset. Return type matches input.

        Examples
        --------
        Load a mesh with asymmetric bounds and show them.

        >>> import pyvista as pv
        >>> mesh = pv.Cube(
        ...     x_length=1.0, y_length=2.0, z_length=3.0, center=(1.0, 2.0, 3.0)
        ... )
        >>> mesh.bounds
        BoundsTuple(x_min = 0.5,
                    x_max = 1.5,
                    y_min = 1.0,
                    y_max = 3.0,
                    z_min = 1.5,
                    z_max = 4.5)

        Resize it to fit specific bounds.

        >>> resized = mesh.resize(bounds=[-1, 2, -3, 4, -5, 6])
        >>> resized.bounds
        BoundsTuple(x_min = -1.0,
                    x_max =  2.0,
                    y_min = -3.0,
                    y_max =  4.0,
                    z_min = -5.0,
                    z_max =  6.0)

        Resize the mesh so its size is ``4.0``.

        >>> resized = mesh.resize(bounds_size=4.0)
        >>> resized.bounds_size
        (4.0, 4.0, 4.0)
        >>> resized.bounds
        BoundsTuple(x_min = -1.0,
                    x_max =  3.0,
                    y_min =  0.0,
                    y_max =  4.0,
                    z_min =  1.0,
                    z_max =  5.0)

        Specify a different size for each axis and set the desired center.

        >>> resized = mesh.resize(bounds_size=(2.0, 1.0, 0.5), center=(1.0, 0.5, 0.25))
        >>> resized.bounds_size
        (2.0, 1.0, 0.5)
        >>> resized.center
        (1.0, 0.5, 0.25)

        Center the mesh at the origin and normalize its bounds to ``1.0``.

        >>> resized = mesh.resize(bounds_size=1.0, center=(0.0, 0.0, 0.0))
        >>> resized.bounds
        BoundsTuple(x_min = -0.5,
                    x_max =  0.5,
                    y_min = -0.5,
                    y_max =  0.5,
                    z_min = -0.5,
                    z_max =  0.5)

        """
        if bounds is not None:
            if bounds_size is not None:
                msg = "Cannot specify both 'bounds' and 'bounds_size'. Choose one resizing method."
                raise ValueError(msg)
            if center is not None:
                msg = (
                    "Cannot specify both 'bounds' and 'center'. 'center' can only be used with "
                    "the 'bounds_size' parameter."
                )
                raise ValueError(msg)

            target_bounds3x2 = _validation.validate_array(
                bounds, must_have_shape=6, reshape_to=(3, 2), name='bounds'
            )
            target_size = np.diff(target_bounds3x2.T, axis=0)[0]
            current_center = np.array(self.center)
            target_center = np.mean(target_bounds3x2, axis=1)

        else:
            if bounds_size is None:
                msg = "'bounds_size' and 'bounds' cannot both be None. Choose one resizing method."
                raise ValueError(msg)

            target_size = bounds_size
            current_center = np.array(self.center)
            target_center = (
                current_center
                if center is None
                else _validation.validate_array3(center, name='center')
            )

        current_size = self.bounds_size
        scale_factors = target_size * _reciprocal(current_size, value_if_division_by_zero=1.0)

        # Apply transformation
        transform = pyvista.Transform()
        transform.translate(-current_center)
        transform.scale(scale_factors)
        transform.translate(target_center)
        return self.transform(
            transform, transform_all_input_vectors=transform_all_input_vectors, inplace=inplace
        )

    @_deprecate_positional_args
    def flip_x(  # type: ignore[misc]
        self: _MeshType_co,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Flip mesh about the x-axis.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        point : sequence[float], optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`~pyvista.DataSet.center`.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        DataSet | MultiBlock
            Flipped dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.flip_x
            Concatenate a reflection about the x-axis with a transformation.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> pl.show_axes()
        >>> mesh1 = examples.download_teapot()
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.subplot(0, 1)
        >>> pl.show_axes()
        >>> mesh2 = mesh1.flip_x(inplace=False)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show(cpos='xy')

        """
        if point is None:
            point = self.center
        t = Transform().reflect((1, 0, 0), point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    @_deprecate_positional_args
    def flip_y(  # type: ignore[misc]
        self: _MeshType_co,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Flip mesh about the y-axis.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        point : VectorLike[float], optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`~pyvista.DataSet.center`.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        DataSet | MultiBlock
            Flipped dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.flip_y
            Concatenate a reflection about the y-axis with a transformation.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> pl.show_axes()
        >>> mesh1 = examples.download_teapot()
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.subplot(0, 1)
        >>> pl.show_axes()
        >>> mesh2 = mesh1.flip_y(inplace=False)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show(cpos='xy')

        """
        if point is None:
            point = self.center
        t = Transform().reflect((0, 1, 0), point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    @_deprecate_positional_args
    def flip_z(  # type: ignore[misc]
        self: _MeshType_co,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Flip mesh about the z-axis.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        point : VectorLike[float], optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`~pyvista.DataSet.center`.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        DataSet | MultiBlock
            Flipped dataset. Return type matches input.

        See Also
        --------
        pyvista.Transform.flip_z
            Concatenate a reflection about the z-axis with a transformation.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> pl.show_axes()
        >>> mesh1 = examples.download_teapot().rotate_x(90, inplace=False)
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.subplot(0, 1)
        >>> pl.show_axes()
        >>> mesh2 = mesh1.flip_z(inplace=False)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show(cpos='xz')

        """
        if point is None:
            point = self.center
        t = Transform().reflect((0, 0, 1), point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    @_deprecate_positional_args(allowed=['normal'])
    def flip_normal(  # type: ignore[misc]  # noqa: PLR0917
        self: _MeshType_co,
        normal: VectorLike[float],
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _MeshType_co:
        """Flip mesh about the normal.

        .. note::
            See also the notes at :func:`transform` which is used by this filter
            under the hood.

        Parameters
        ----------
        normal : VectorLike[float]
           Normal vector to flip about.

        point : VectorLike[float], optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`~pyvista.DataSet.center`.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        DataSet | MultiBlock
            Dataset flipped about its normal. Return type matches input.

        See Also
        --------
        pyvista.Transform.reflect
            Concatenate a reflection matrix with a transformation.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> pl.show_axes()
        >>> mesh1 = examples.download_teapot()
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.subplot(0, 1)
        >>> pl.show_axes()
        >>> mesh2 = mesh1.flip_normal([1.0, 1.0, 1.0], inplace=False)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show(cpos='xy')

        """
        if point is None:
            point = self.center
        t = Transform().reflect(normal, point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def _clip_with_function(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        function: _vtk.vtkImplicitFunction,
        *,
        invert: bool = True,
        value: float = 0.0,
        return_clipped: bool = False,
        progress_bar: bool = False,
        crinkle: bool = False,
    ):
        """Clip using an implicit function (internal helper)."""
        if crinkle:
            CELL_IDS_KEY = 'cell_ids'
            VTK_POINT_IDS_KEYS = 'vtkOriginalPointIds'
            VTK_CELL_IDS_KEYS = 'vtkOriginalCellIds'
            INT_DTYPE = np.int64
            ITER_KWARGS = dict(skip_none=True)

            def extract_cells(dataset, ids, active_scalars_info_):
                # Extract cells and remove arrays, and restore active scalars
                output = dataset.extract_cells(ids)
                if VTK_POINT_IDS_KEYS in (point_data := output.point_data):
                    del point_data[VTK_POINT_IDS_KEYS]
                if VTK_CELL_IDS_KEYS in (cell_data := output.cell_data):
                    del cell_data[VTK_CELL_IDS_KEYS]
                association, name = active_scalars_info_
                if not dataset.is_empty:
                    dataset.set_active_scalars(name, preference=association)
                if not output.is_empty:
                    output.set_active_scalars(name, preference=association)
                return output

            def extract_crinkle_cells(dataset, a_, b_, _):
                if b_ is None:
                    # Extract cells when `return_clipped=False`
                    def extract_cells_from_block(block_, clipped_a, _, active_scalars_info_):
                        return extract_cells(
                            block_,
                            np.unique(clipped_a.cell_data[CELL_IDS_KEY]),
                            active_scalars_info_,
                        )
                else:
                    # Extract cells when `return_clipped=True`
                    def extract_cells_from_block(  # noqa: PLR0917
                        block_, clipped_a, clipped_b, active_scalars_info_
                    ):
                        set_a = (
                            set(clipped_a.cell_data[CELL_IDS_KEY])
                            if CELL_IDS_KEY in clipped_a.cell_data.keys()
                            else set()
                        )
                        set_b = (
                            set(clipped_b.cell_data[CELL_IDS_KEY])
                            if CELL_IDS_KEY in clipped_b.cell_data.keys()
                            else set()
                        )
                        set_b = set_b - set_a
                        # Need to cast as int dtype explicitly to ensure empty arrays have
                        # the right type required by extract_cells
                        array_a = np.array(list(set_a), dtype=INT_DTYPE)
                        array_b = np.array(list(set_b), dtype=INT_DTYPE)

                        clipped_a = extract_cells(block_, array_a, active_scalars_info_)
                        clipped_b = extract_cells(block_, array_b, active_scalars_info_)
                        return clipped_a, clipped_b

                def extract_cells_from_multiblock(  # noqa: PLR0917
                    multi_in, multi_a, multi_b, active_scalars_info_
                ):
                    # Iterate though input and output multiblocks
                    # `multi_b` may be None depending on `return_clipped`
                    self_iter = multi_in.recursive_iterator('all', **ITER_KWARGS)
                    a_iter = multi_a.recursive_iterator(**ITER_KWARGS)
                    b_iter = (
                        multi_b.recursive_iterator(**ITER_KWARGS)
                        if multi_b is not None
                        else itertools.repeat(None)
                    )

                    for (ids, _, block_self), block_a, block_b, scalars_info in zip(
                        self_iter, a_iter, b_iter, active_scalars_info_
                    ):
                        crinkled = extract_cells_from_block(
                            block_self, block_a, block_b, scalars_info
                        )
                        # Replace blocks with crinkled ones
                        if block_b is None:
                            # Only need to replace one block
                            multi_a.replace(ids, crinkled)
                        else:
                            multi_a.replace(ids, crinkled[0])
                            multi_b.replace(ids, crinkled[1])
                    return multi_a if multi_b is None else (multi_a, multi_b)

                if isinstance(dataset, pyvista.MultiBlock):
                    return extract_cells_from_multiblock(dataset, a_, b_, active_scalars_info)
                return extract_cells_from_block(dataset, a_, b_, active_scalars_info[0])

            # Add Cell IDs to all blocks and keep track of scalars to restore later
            active_scalars_info = []
            if isinstance(self, pyvista.MultiBlock):
                blocks = self.recursive_iterator('blocks', **ITER_KWARGS)  # type: ignore[call-overload]
            else:
                blocks = [self]
            for block in blocks:
                active_scalars_info.append(block.active_scalars_info)
                block.cell_data[CELL_IDS_KEY] = np.arange(block.n_cells, dtype=INT_DTYPE)

        # Need to cast PointSet to PolyData since vtkTableBasedClipDataSet is broken
        # with vtk 9.4.X, see https://gitlab.kitware.com/vtk/vtk/-/issues/19649
        apply_vtk_94x_patch = (
            isinstance(self, pyvista.PointSet)
            and pyvista.vtk_version_info >= (9, 4)
            and pyvista.vtk_version_info < (9, 5)
        )
        mesh_in = self.cast_to_poly_points() if apply_vtk_94x_patch else self

        if isinstance(mesh_in, pyvista.PolyData):
            alg: _vtk.vtkClipPolyData | _vtk.vtkTableBasedClipDataSet = _vtk.vtkClipPolyData()
        # elif isinstance(self, vtk.vtkImageData):
        #     alg = vtk.vtkClipVolume()
        #     alg.SetMixed3DCellGeneration(True)
        else:
            alg = _vtk.vtkTableBasedClipDataSet()
        alg.SetInputDataObject(mesh_in)  # Use the grid as the data we desire to cut
        alg.SetValue(value)
        alg.SetClipFunction(function)  # the implicit function
        alg.SetInsideOut(invert)  # invert the clip if needed
        alg.SetGenerateClippedOutput(return_clipped)
        _update_alg(alg, progress_bar=progress_bar, message='Clipping with Function')

        def _maybe_cast_to_point_set(in_):
            return in_.cast_to_pointset() if apply_vtk_94x_patch else in_

        if return_clipped:
            a = _get_output(alg, oport=0)
            b = _get_output(alg, oport=1)
            if crinkle:
                a, b = extract_crinkle_cells(self, a, b, active_scalars_info)
            return _maybe_cast_to_point_set(a), _maybe_cast_to_point_set(b)
        clipped = _get_output(alg)
        if crinkle:
            clipped = extract_crinkle_cells(self, clipped, None, active_scalars_info)
        return _maybe_cast_to_point_set(clipped)

    @_deprecate_positional_args(allowed=['normal'])
    def clip(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        normal: VectorLike[float] | NormalsLiteral = 'x',
        origin: VectorLike[float] | None = None,
        invert: bool = True,  # noqa: FBT001, FBT002
        value: float = 0.0,
        inplace: bool = False,  # noqa: FBT001, FBT002
        return_clipped: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        crinkle: bool = False,  # noqa: FBT001, FBT002
    ):
        """Clip a dataset by a plane by specifying the origin and normal.

        If no parameters are given the clip will occur in the center
        of that dataset.

        Parameters
        ----------
        normal : tuple(float) | str, default: 'x'
            Length 3 tuple for the normal vector direction. Can also
            be specified as a string conventional direction such as
            ``'x'`` for ``(1, 0, 0)`` or ``'-x'`` for ``(-1, 0, 0)``, etc.

        origin : sequence[float], optional
            The center ``(x, y, z)`` coordinate of the plane on which the clip
            occurs. The default is the center of the dataset.

        invert : bool, default: True
            Flag on whether to flip/invert the clip.

        value : float, default: 0.0
            Set the clipping value along the normal direction.

        inplace : bool, default: False
            Updates mesh in-place.

        return_clipped : bool, default: False
            Return both unclipped and clipped parts of the dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        crinkle : bool, default: False
            Crinkle the clip by extracting the entire cells along the
            clip. This adds the ``"cell_ids"`` array to the ``cell_data``
            attribute that tracks the original cell IDs of the original
            dataset.

        Returns
        -------
        DataSet | MultiBlock | tuple[DataSet | MultiBlock, DataSet | MultiBlock]
            Clipped mesh when ``return_clipped=False`` or a tuple containing the
            unclipped and clipped meshes. Output mesh type matches input type for
            :class:`~pyvista.PointSet`, :class:`~pyvista.PolyData`, and
            :class:`~pyvista.MultiBlock`; otherwise the output type is
            :class:`~pyvista.UnstructuredGrid`.

        Examples
        --------
        Clip a cube along the +X direction.  ``triangulate`` is used as
        the cube is initially composed of quadrilateral faces and
        subdivide only works on triangles.

        >>> import pyvista as pv
        >>> cube = pv.Cube().triangulate().subdivide(3)
        >>> clipped_cube = cube.clip()
        >>> clipped_cube.plot()

        Clip a cube in the +Z direction.  This leaves half a cube
        below the XY plane.

        >>> import pyvista as pv
        >>> cube = pv.Cube().triangulate().subdivide(3)
        >>> clipped_cube = cube.clip('z')
        >>> clipped_cube.plot()

        See :ref:`clip_with_surface_example` for more examples using this filter.

        """
        normal_: VectorLike[float] = NORMALS[normal.lower()] if isinstance(normal, str) else normal
        # find center of data if origin not specified
        origin_ = self.center if origin is None else origin
        # create the plane for clipping
        function = generate_plane(normal_, origin_)
        # run the clip
        result = self._clip_with_function(
            function,
            invert=invert,
            value=value,
            return_clipped=return_clipped,
            progress_bar=progress_bar,
            crinkle=crinkle,
        )

        if isinstance(result, tuple):
            result = (
                _cast_output_to_match_input_type(result[0], self),
                _cast_output_to_match_input_type(result[1], self),
            )
        else:
            result = _cast_output_to_match_input_type(result, self)
        if inplace:
            if return_clipped:
                self.copy_from(result[0], deep=False)
                return self, result[1]
            else:
                self.copy_from(result, deep=False)
                return self
        return result

    @_deprecate_positional_args(allowed=['bounds'])
    def clip_box(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        bounds: float | VectorLike[float] | pyvista.PolyData | None = None,
        invert: bool = True,  # noqa: FBT001, FBT002
        factor: float = 0.35,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        merge_points: bool = True,  # noqa: FBT001, FBT002
        crinkle: bool = False,  # noqa: FBT001, FBT002
    ):
        """Clip a dataset by a bounding box defined by the bounds.

        If no bounds are given, a corner of the dataset bounds will be removed.

        Parameters
        ----------
        bounds : sequence[float], optional
            Length 6 sequence of floats: ``(x_min, x_max, y_min, y_max, z_min, z_max)``.
            Length 3 sequence of floats: distances from the min coordinate of
            of the input mesh. Single float value: uniform distance from the
            min coordinate. Length 12 sequence of length 3 sequence of floats:
            a plane collection (normal, center, ...).
            :class:`pyvista.PolyData`: if a poly mesh is passed that represents
            a box with 6 faces that all form a standard box, then planes will
            be extracted from the box to define the clipping region.

        invert : bool, default: True
            Flag on whether to flip/invert the clip.

        factor : float, default: 0.35
            If bounds are not given this is the factor along each axis to
            extract the default box.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        merge_points : bool, default: True
            If ``True``, coinciding points of independently defined mesh
            elements will be merged.

        crinkle : bool, default: False
            Crinkle the clip by extracting the entire cells along the
            clip. This adds the ``"cell_ids"`` array to the ``cell_data``
            attribute that tracks the original cell IDs of the original
            dataset.

        Returns
        -------
        DataSet | MultiBlock
            Clipped mesh. Output mesh type matches input type for
            :class:`~pyvista.PointSet`, :class:`~pyvista.PolyData`, and
            :class:`~pyvista.MultiBlock`; otherwise the output type is
            :class:`~pyvista.UnstructuredGrid`.

            .. versionchanged:: 0.47

                The output type now matches the input type for :class:`~pyvista.PointSet` and
                :class:`~pyvista.PolyData`. This matches the behavior of :meth:`clip`.
                Previously, these types would return :class:`~pyvista.UnstructuredGrid`.

        Examples
        --------
        Clip a corner of a cube.  The bounds of a cube are normally
        ``[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]``, and this removes 1/8 of
        the cube's surface.

        >>> import pyvista as pv
        >>> cube = pv.Cube().triangulate().subdivide(3)
        >>> clipped_cube = cube.clip_box([0, 1, 0, 1, 0, 1])
        >>> clipped_cube.plot()

        See :ref:`clip_with_plane_box_example` for more examples using this filter.

        """
        CELL_IDS = 'cell_ids'

        def add_cell_ids_to_self() -> None:
            def add_ids_to_mesh(dataset: DataSet):
                if not isinstance(dataset, pyvista.PointSet):
                    dataset.cell_data[CELL_IDS] = np.arange(dataset.n_cells)

            if isinstance(self, pyvista.MultiBlock):
                for block in self.recursive_iterator(skip_none=True):
                    add_ids_to_mesh(block)
                return
            add_ids_to_mesh(self)
            return

        def extract_crinkle_cells_from_output(input_mesh, output_mesh):
            def extract_crinkle_cells(mesh_in: DataSet, mesh_out: DataSet):
                if CELL_IDS in mesh_out.cell_data.keys():
                    return mesh_in.extract_cells(np.unique(mesh_out.cell_data[CELL_IDS]))
                return mesh_in

            if isinstance(output_mesh, pyvista.MultiBlock):
                for (ids, _, block_in), block_out in zip(
                    output_mesh.recursive_iterator('all', skip_none=True),
                    input_mesh.recursive_iterator(skip_none=True),
                ):
                    extracted = extract_crinkle_cells(block_in, block_out)
                    output_mesh.replace(ids, extracted)
                return output_mesh
            return extract_crinkle_cells(input_mesh, output_mesh)

        if bounds is None:

            def _get_quarter(dmin, dmax):
                """Get a section of the given range (internal helper)."""
                return dmax - ((dmax - dmin) * factor)

            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            xmin = _get_quarter(xmin, xmax)
            ymin = _get_quarter(ymin, ymax)
            zmin = _get_quarter(zmin, zmax)
            bounds = [xmin, xmax, ymin, ymax, zmin, zmax]
        if isinstance(bounds, (float, int)):
            bounds = [bounds, bounds, bounds]
        elif isinstance(bounds, pyvista.PolyData):
            poly = bounds
            if poly.n_cells != 6:
                msg = 'The bounds mesh must have only 6 faces.'
                raise ValueError(msg)
            bounds = []
            poly.compute_normals(inplace=True)
            for cid in range(6):
                cell = poly.extract_cells(cid)
                normal = cell['Normals'][0]
                bounds.append(normal)
                bounds.append(cell.center)
        bounds_ = _validation.validate_array(
            bounds, dtype_out=float, must_have_length=[3, 6, 12], name='bounds'
        )
        if len(bounds_) == 3:
            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            bounds_ = np.array(
                (
                    xmin,
                    xmin + bounds_[0],
                    ymin,
                    ymin + bounds_[1],
                    zmin,
                    zmin + bounds_[2],
                )
            )
        if crinkle:
            add_cell_ids_to_self()
        alg = _vtk.vtkBoxClipDataSet()
        if not merge_points:
            # vtkBoxClipDataSet uses vtkMergePoints by default
            alg.SetLocator(_vtk.vtkNonMergingPointLocator())
        alg.SetInputDataObject(self)
        alg.SetBoxClip(*bounds_)
        port = 0
        if invert:
            # invert the clip if needed
            port = 1
            alg.GenerateClippedOutputOn()
        _update_alg(alg, progress_bar=progress_bar, message='Clipping a Dataset by a Bounding Box')
        clipped = _get_output(alg, oport=port)
        if crinkle:
            clipped = extract_crinkle_cells_from_output(self, clipped)
        return _cast_output_to_match_input_type(clipped, self)

    @_deprecate_positional_args(allowed=['implicit_function'])
    def slice_implicit(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        implicit_function: _vtk.vtkImplicitFunction,
        generate_triangles: bool = False,  # noqa: FBT001, FBT002
        contour: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Slice a dataset by a VTK implicit function.

        Parameters
        ----------
        implicit_function : :vtk:`vtkImplicitFunction`
            Specify the implicit function to perform the cutting.

        generate_triangles : bool, default: False
            If this is enabled (``False`` by default), the output will
            be triangles. Otherwise the output will be the intersection
            polygons. If the cutting function is not a plane, the
            output will be 3D polygons, which might be nice to look at
            but hard to compute with downstream.

        contour : bool, default: False
            If ``True``, apply a ``contour`` filter after slicing.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Sliced dataset.

        See Also
        --------
        slice
        slice_orthogonal
        slice_along_axis
        slice_along_line
        :meth:`~pyvista.ImageDataFilters.slice_index`

        Examples
        --------
        Slice the surface of a sphere.

        >>> import pyvista as pv
        >>> import vtk
        >>> sphere = vtk.vtkSphere()
        >>> sphere.SetRadius(10)
        >>> mesh = pv.Wavelet()
        >>> slice = mesh.slice_implicit(sphere)
        >>> slice.plot(show_edges=True, line_width=5)

        >>> cylinder = vtk.vtkCylinder()
        >>> cylinder.SetRadius(10)
        >>> mesh = pv.Wavelet()
        >>> slice = mesh.slice_implicit(cylinder)
        >>> slice.plot(show_edges=True, line_width=5)

        """
        alg = _vtk.vtkCutter()  # Construct the cutter object
        alg.SetInputDataObject(self)  # Use the grid as the data we desire to cut
        alg.SetCutFunction(implicit_function)  # the cutter to use the function
        alg.SetGenerateTriangles(generate_triangles)
        _update_alg(alg, progress_bar=progress_bar, message='Slicing')
        output = _get_output(alg)
        if contour:
            return output.contour()
        return output

    @_deprecate_positional_args(allowed=['normal'])
    def slice(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        normal: VectorLike[float] | NormalsLiteral = 'x',
        origin: VectorLike[float] | None = None,
        generate_triangles: bool = False,  # noqa: FBT001, FBT002
        contour: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Slice a dataset by a plane at the specified origin and normal vector orientation.

        If no origin is specified, the center of the input dataset will be used.

        Parameters
        ----------
        normal : sequence[float] | str, default: 'x'
            Length 3 tuple for the normal vector direction. Can also be
            specified as a string conventional direction such as ``'x'`` for
            ``(1, 0, 0)`` or ``'-x'`` for ``(-1, 0, 0)``, etc.

        origin : sequence[float], optional
            The center ``(x, y, z)`` coordinate of the plane on which
            the slice occurs.

        generate_triangles : bool, default: False
            If this is enabled (``False`` by default), the output will
            be triangles. Otherwise the output will be the intersection
            polygons.

        contour : bool, default: False
            If ``True``, apply a ``contour`` filter after slicing.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Sliced dataset.

        See Also
        --------
        slice_implicit
        slice_orthogonal
        slice_along_axis
        slice_along_line
        :meth:`~pyvista.ImageDataFilters.slice_index`

        Examples
        --------
        Slice the surface of a sphere.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> slice_x = sphere.slice(normal='x')
        >>> slice_y = sphere.slice(normal='y')
        >>> slice_z = sphere.slice(normal='z')
        >>> slices = slice_x + slice_y + slice_z
        >>> slices.plot(line_width=5)

        See :ref:`slice_example` for more examples using this filter.

        """
        normal_: VectorLike[float] = NORMALS[normal.lower()] if isinstance(normal, str) else normal
        # find center of data if origin not specified
        origin_ = self.center if origin is None else origin

        # create the plane for clipping
        plane = generate_plane(normal_, origin_)
        return self.slice_implicit(
            plane,
            generate_triangles=generate_triangles,
            contour=contour,
            progress_bar=progress_bar,
        )

    @_deprecate_positional_args
    def slice_orthogonal(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        generate_triangles: bool = False,  # noqa: FBT001, FBT002
        contour: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Create three orthogonal slices through the dataset on the three cartesian planes.

        Yields a MutliBlock dataset of the three slices.

        Parameters
        ----------
        x : float, optional
            The X location of the YZ slice.

        y : float, optional
            The Y location of the XZ slice.

        z : float, optional
            The Z location of the XY slice.

        generate_triangles : bool, default: False
            When ``True``, the output will be triangles. Otherwise the output
            will be the intersection polygons.

        contour : bool, default: False
            If ``True``, apply a ``contour`` filter after slicing.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Sliced dataset.

        See Also
        --------
        slice
        slice_implicit
        slice_along_axis
        slice_along_line
        :meth:`~pyvista.ImageDataFilters.slice_index`

        Examples
        --------
        Slice the random hills dataset with three orthogonal planes.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> slices = hills.slice_orthogonal(contour=False)
        >>> slices.plot(line_width=5)

        See :ref:`slice_example` for more examples using this filter.

        """
        # Create the three slices
        if x is None:
            x = self.center[0]
        if y is None:
            y = self.center[1]
        if z is None:
            z = self.center[2]
        output = pyvista.MultiBlock()
        if isinstance(self, pyvista.MultiBlock):
            for i in range(self.n_blocks):
                data = self[i]
                output.append(
                    data.slice_orthogonal(
                        x=x,
                        y=y,
                        z=z,
                        generate_triangles=generate_triangles,
                        contour=contour,
                    )
                    if data is not None
                    else data
                )
            return output
        output.append(
            self.slice(
                normal='x',
                origin=[x, y, z],
                generate_triangles=generate_triangles,
                progress_bar=progress_bar,
            ),
            'YZ',
        )
        output.append(
            self.slice(
                normal='y',
                origin=[x, y, z],
                generate_triangles=generate_triangles,
                progress_bar=progress_bar,
            ),
            'XZ',
        )
        output.append(
            self.slice(
                normal='z',
                origin=[x, y, z],
                generate_triangles=generate_triangles,
                progress_bar=progress_bar,
            ),
            'XY',
        )
        return output

    @_deprecate_positional_args(allowed=['n', 'axis'])
    def slice_along_axis(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        n: int = 5,
        axis: Literal['x', 'y', 'z', 0, 1, 2] = 'x',
        tolerance: float | None = None,
        generate_triangles: bool = False,  # noqa: FBT001, FBT002
        contour: bool = False,  # noqa: FBT001, FBT002
        bounds=None,
        center=None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Create many slices of the input dataset along a specified axis.

        Parameters
        ----------
        n : int, default: 5
            The number of slices to create.

        axis : str | int, default: 'x'
            The axis to generate the slices along. Perpendicular to the
            slices. Can be string name (``'x'``, ``'y'``, or ``'z'``) or
            axis index (``0``, ``1``, or ``2``).

        tolerance : float, optional
            The tolerance to the edge of the dataset bounds to create
            the slices. The ``n`` slices are placed equidistantly with
            an absolute padding of ``tolerance`` inside each side of the
            ``bounds`` along the specified axis. Defaults to 1% of the
            ``bounds`` along the specified axis.

        generate_triangles : bool, default: False
            When ``True``, the output will be triangles. Otherwise the output
            will be the intersection polygons.

        contour : bool, default: False
            If ``True``, apply a ``contour`` filter after slicing.

        bounds : sequence[float], optional
            A 6-length sequence overriding the bounds of the mesh.
            The bounds along the specified axis define the extent
            where slices are taken.

        center : sequence[float], optional
            A 3-length sequence specifying the position of the line
            along which slices are taken. Defaults to the center of
            the mesh.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Sliced dataset.

        See Also
        --------
        slice
        slice_implicit
        slice_orthogonal
        slice_along_line
        :meth:`~pyvista.ImageDataFilters.slice_index`

        Examples
        --------
        Slice the random hills dataset in the X direction.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> slices = hills.slice_along_axis(n=10)
        >>> slices.plot(line_width=5)

        Slice the random hills dataset in the Z direction.

        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> slices = hills.slice_along_axis(n=10, axis='z')
        >>> slices.plot(line_width=5)

        See :ref:`slice_example` for more examples using this filter.

        """
        # parse axis input
        XYZLiteral = Literal['x', 'y', 'z']
        labels: list[XYZLiteral] = ['x', 'y', 'z']
        label_to_index: dict[Literal['x', 'y', 'z'], Literal[0, 1, 2]] = {
            'x': 0,
            'y': 1,
            'z': 2,
        }
        if isinstance(axis, int):
            ax_index = axis
            ax_label = labels[ax_index]
        elif isinstance(axis, str):
            ax_str = axis.lower()
            if ax_str in labels:
                ax_label = cast('XYZLiteral', ax_str)
                ax_index = label_to_index[ax_label]
            else:
                msg = f'Axis ({axis!r}) not understood. Choose one of {labels}.'
                raise ValueError(msg) from None
        # get the locations along that axis
        if bounds is None:
            bounds = self.bounds
        if center is None:
            center = self.center
        if tolerance is None:
            tolerance = (bounds[ax_index * 2 + 1] - bounds[ax_index * 2]) * 0.01
        rng = np.linspace(
            bounds[ax_index * 2] + tolerance, bounds[ax_index * 2 + 1] - tolerance, n
        )
        center = list(center)
        # Make each of the slices
        output = pyvista.MultiBlock()
        if isinstance(self, pyvista.MultiBlock):
            for i in range(self.n_blocks):
                data = self[i]
                output.append(
                    data.slice_along_axis(
                        n=n,
                        axis=ax_label,
                        tolerance=tolerance,
                        generate_triangles=generate_triangles,
                        contour=contour,
                        bounds=bounds,
                        center=center,
                    )
                    if data is not None
                    else data
                )
            return output
        for i in range(n):
            center[ax_index] = rng[i]
            slc = self.slice(
                normal=ax_label,
                origin=center,
                generate_triangles=generate_triangles,
                contour=contour,
                progress_bar=progress_bar,
            )
            output.append(slc, f'slice{i}')
        return output

    @_deprecate_positional_args(allowed=['line'])
    def slice_along_line(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        line: pyvista.PolyData,
        generate_triangles: bool = False,  # noqa: FBT001, FBT002
        contour: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Slice a dataset using a polyline/spline as the path.

        This also works for lines generated with :func:`pyvista.Line`.

        Parameters
        ----------
        line : pyvista.PolyData
            A PolyData object containing one single PolyLine cell.

        generate_triangles : bool, default: False
            When ``True``, the output will be triangles. Otherwise the output
            will be the intersection polygons.

        contour : bool, default: False
            If ``True``, apply a ``contour`` filter after slicing.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Sliced dataset.

        See Also
        --------
        slice
        slice_implicit
        slice_orthogonal
        slice_along_axis
        :meth:`~pyvista.ImageDataFilters.slice_index`

        Examples
        --------
        Slice the random hills dataset along a circular arc.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> hills = examples.load_random_hills()
        >>> center = np.array(hills.center)
        >>> point_a = center + np.array([5, 0, 0])
        >>> point_b = center + np.array([-5, 0, 0])
        >>> arc = pv.CircularArc(
        ...     pointa=point_a, pointb=point_b, center=center, resolution=100
        ... )
        >>> line_slice = hills.slice_along_line(arc)

        Plot the circular arc and the hills mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(hills, smooth_shading=True, style='wireframe')
        >>> _ = pl.add_mesh(
        ...     line_slice,
        ...     line_width=10,
        ...     render_lines_as_tubes=True,
        ...     color='k',
        ... )
        >>> _ = pl.add_mesh(arc, line_width=10, color='grey')
        >>> pl.show()

        See :ref:`slice_example` for more examples using this filter.

        """
        # check that we have a PolyLine cell in the input line
        if line.GetNumberOfCells() != 1:
            msg = 'Input line must have only one cell.'
            raise ValueError(msg)
        polyline = line.GetCell(0)
        if not isinstance(polyline, _vtk.vtkPolyLine):
            msg = f'Input line must have a PolyLine cell, not ({type(polyline)})'
            raise TypeError(msg)
        # Generate PolyPlane
        polyplane = _vtk.vtkPolyPlane()
        polyplane.SetPolyLine(polyline)
        # Create slice
        alg = _vtk.vtkCutter()  # Construct the cutter object
        alg.SetInputDataObject(self)  # Use the grid as the data we desire to cut
        alg.SetCutFunction(polyplane)  # the cutter to use the poly planes
        if not generate_triangles:
            alg.GenerateTrianglesOff()
        _update_alg(alg, progress_bar=progress_bar, message='Slicing along Line')
        output = _get_output(alg)
        if contour:
            return output.contour()
        return output

    @_deprecate_positional_args
    def extract_all_edges(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        use_all_points: bool = False,  # noqa: FBT001, FBT002
        clear_data: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Extract all the internal/external edges of the dataset as PolyData.

        This produces a full wireframe representation of the input dataset.

        Parameters
        ----------
        use_all_points : bool, default: False
            Indicates whether all of the points of the input mesh should exist
            in the output. When ``True``, point numbering does not change and
            a threaded approach is used, which avoids the use of a point locator
            and is quicker.

            By default this is set to ``False``, and unused points are omitted
            from the output.

            This parameter can only be set to ``True`` with ``vtk==9.1.0`` or newer.

        clear_data : bool, default: False
            Clear any point, cell, or field data. This is useful
            if wanting to strictly extract the edges.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Edges extracted from the dataset.

        Examples
        --------
        Extract the edges of a sample unstructured grid and plot the edges.
        Note how it plots interior edges.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> hex_beam = pv.read(examples.hexbeamfile)
        >>> edges = hex_beam.extract_all_edges()
        >>> edges.plot(line_width=5, color='k')

        See :ref:`cell_centers_example` for more examples using this filter.

        """
        alg = _vtk.vtkExtractEdges()
        alg.SetInputDataObject(self)
        if use_all_points:
            try:
                alg.SetUseAllPoints(use_all_points)
            except AttributeError:  # pragma: no cover
                msg = (
                    'This version of VTK does not support `use_all_points=True`. '
                    'VTK v9.1 or newer is required.'
                )
                raise VTKVersionError(msg)
        # Suppress improperly used INFO for debugging messages in vtkExtractEdges
        with pyvista.vtk_verbosity('off'):
            _update_alg(alg, progress_bar=progress_bar, message='Extracting All Edges')
        output = _get_output(alg)
        if clear_data:
            output.clear_data()
        return output

    @_deprecate_positional_args
    def elevation(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        low_point: VectorLike[float] | None = None,
        high_point: VectorLike[float] | None = None,
        scalar_range: str | VectorLike[float] | None = None,
        preference: Literal['point', 'cell'] = 'point',
        set_active: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Generate scalar values on a dataset.

        The scalar values lie within a user specified range, and are
        generated by computing a projection of each dataset point onto
        a line.  The line can be oriented arbitrarily.  A typical
        example is to generate scalars based on elevation or height
        above a plane.

        .. warning::
           This will create a scalars array named ``'Elevation'`` on the
           point data of the input dataset and overwrite the array
           named ``'Elevation'`` if present.

        Parameters
        ----------
        low_point : sequence[float], optional
            The low point of the projection line in 3D space. Default is bottom
            center of the dataset. Otherwise pass a length 3 sequence.

        high_point : sequence[float], optional
            The high point of the projection line in 3D space. Default is top
            center of the dataset. Otherwise pass a length 3 sequence.

        scalar_range : str | sequence[float], optional
            The scalar range to project to the low and high points on the line
            that will be mapped to the dataset. If None given, the values will
            be computed from the elevation (Z component) range between the
            high and low points. Min and max of a range can be given as a length
            2 sequence. If ``str``, name of scalar array present in the
            dataset given, the valid range of that array will be used.

        preference : str, default: "point"
            When an array name is specified for ``scalar_range``, this is the
            preferred array type to search for in the dataset.
            Must be either ``'point'`` or ``'cell'``.

        set_active : bool, default: True
            A boolean flag on whether or not to set the new
            ``'Elevation'`` scalar as the active scalars array on the
            output dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        DataSet | MultiBlock
            Dataset containing elevation scalars in the
            ``"Elevation"`` array in ``point_data``.

        Examples
        --------
        Generate the "elevation" scalars for a sphere mesh.  This is
        simply the height in Z from the XY plane.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere_elv = sphere.elevation()
        >>> sphere_elv.plot(smooth_shading=True)

        Access the first 4 elevation scalars.  This is a point-wise
        array containing the "elevation" of each point.

        >>> sphere_elv['Elevation'][:4]  # doctest:+SKIP
        array([-0.5       ,  0.5       , -0.49706897, -0.48831028], dtype=float32)

        See :ref:`using_filters_example` for more examples using this filter.

        """
        # Fix the projection line:
        if low_point is None:
            low_point_ = list(self.center)
            low_point_[2] = self.bounds.z_min
        else:
            low_point_ = _validation.validate_array3(low_point)
        if high_point is None:
            high_point_ = list(self.center)
            high_point_[2] = self.bounds.z_max
        else:
            high_point_ = _validation.validate_array3(high_point)
        # Fix scalar_range:
        if scalar_range is None:
            scalar_range_ = (low_point_[2], high_point_[2])
        elif isinstance(scalar_range, str):
            scalar_range_ = self.get_data_range(scalar_range, preference=preference)
        else:
            scalar_range_ = _validation.validate_data_range(scalar_range)

        # Construct the filter
        alg = _vtk.vtkElevationFilter()
        alg.SetInputDataObject(self)
        # Set the parameters
        alg.SetScalarRange(scalar_range_)
        alg.SetLowPoint(low_point_)
        alg.SetHighPoint(high_point_)
        _update_alg(alg, progress_bar=progress_bar, message='Computing Elevation')
        # Decide on updating active scalars array
        output = _get_output(alg)
        if not set_active:
            # 'Elevation' is automatically made active by the VTK filter
            output.point_data.active_scalars_name = self.point_data.active_scalars_name
        return output

    @_deprecate_positional_args
    def compute_cell_sizes(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        length: bool = True,  # noqa: FBT001, FBT002
        area: bool = True,  # noqa: FBT001, FBT002
        volume: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        vertex_count: bool = False,  # noqa: FBT001, FBT002
    ):
        """Compute sizes for 0D (vertex count), 1D (length), 2D (area) and 3D (volume) cells.

        Parameters
        ----------
        length : bool, default: True
            Specify whether or not to compute the length of 1D cells.

        area : bool, default: True
            Specify whether or not to compute the area of 2D cells.

        volume : bool, default: True
            Specify whether or not to compute the volume of 3D cells.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        vertex_count : bool, default: False
            Specify whether or not to compute sizes for vertex and polyvertex cells (0D cells).
            The computed value is the number of points in the cell.

        Returns
        -------
        DataSet | MultiBlock
            Dataset with `cell_data` containing the ``"VertexCount"``,
            ``"Length"``, ``"Area"``, and ``"Volume"`` arrays if set
            in the parameters.  Return type matches input.

        Notes
        -----
        If cells do not have a dimension (for example, the length of
        hexahedral cells), the corresponding array will be all zeros.

        Examples
        --------
        Compute the face area of the example airplane mesh.

        >>> from pyvista import examples
        >>> surf = examples.load_airplane()
        >>> surf = surf.compute_cell_sizes(length=False, volume=False)
        >>> surf.plot(show_edges=True, scalars='Area')

        """
        alg = _vtk.vtkCellSizeFilter()
        alg.SetInputDataObject(self)
        alg.SetComputeArea(area)
        alg.SetComputeVolume(volume)
        alg.SetComputeLength(length)
        alg.SetComputeVertexCount(vertex_count)
        _update_alg(alg, progress_bar=progress_bar, message='Computing Cell Sizes')
        return _get_output(alg)

    @_deprecate_positional_args
    def cell_centers(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        vertex: bool = True,  # noqa: FBT001, FBT002
        pass_cell_data: bool = True,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Generate points at the center of the cells in this dataset.

        These points can be used for placing glyphs or vectors.

        Parameters
        ----------
        vertex : bool, default: True
            Enable or disable the generation of vertex cells.

        pass_cell_data : bool, default: True
            If enabled, pass the input cell data through to the output.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Polydata where the points are the cell centers of the
            original dataset.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Plane()
        >>> mesh.point_data.clear()
        >>> centers = mesh.cell_centers()
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(mesh, show_edges=True)
        >>> actor = pl.add_points(
        ...     centers,
        ...     render_points_as_spheres=True,
        ...     color='red',
        ...     point_size=20,
        ... )
        >>> pl.show()

        See :ref:`cell_centers_example` for more examples using this filter.

        """
        input_mesh = self.cast_to_poly_points() if isinstance(self, pyvista.PointSet) else self
        alg = _vtk.vtkCellCenters()
        alg.SetInputDataObject(input_mesh)
        alg.SetVertexCells(vertex)
        alg.SetCopyArrays(pass_cell_data)
        _update_alg(
            alg, progress_bar=progress_bar, message='Generating Points at the Center of the Cells'
        )
        return _get_output(alg)

    @_deprecate_positional_args
    def cell_data_to_point_data(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        pass_cell_data: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Transform cell data into point data.

        Point data are specified per node and cell data specified
        within cells.  Optionally, the input point data can be passed
        through to the output.

        The method of transformation is based on averaging the data
        values of all cells using a particular point. Optionally, the
        input cell data can be passed through to the output as well.

        Parameters
        ----------
        pass_cell_data : bool, default: False
            If enabled, pass the input cell data through to the output.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        DataSet | MultiBlock
            Dataset with the point data transformed into cell data.
            Return type matches input.

        See Also
        --------
        point_data_to_cell_data
            Similar transformation applied to point data.
        :meth:`~pyvista.ImageDataFilters.cells_to_points`
            Re-mesh :class:`~pyvista.ImageData` to a points-based representation.

        Examples
        --------
        First compute the face area of the example airplane mesh and
        show the cell values.  This is to show discrete cell data.

        >>> from pyvista import examples
        >>> surf = examples.load_airplane()
        >>> surf = surf.compute_cell_sizes(length=False, volume=False)
        >>> surf.plot(scalars='Area')

        These cell scalars can be applied to individual points to
        effectively smooth out the cell data onto the points.

        >>> from pyvista import examples
        >>> surf = examples.load_airplane()
        >>> surf = surf.compute_cell_sizes(length=False, volume=False)
        >>> surf = surf.cell_data_to_point_data()
        >>> surf.plot(scalars='Area')

        """
        alg = _vtk.vtkCellDataToPointData()
        alg.SetInputDataObject(self)
        alg.SetPassCellData(pass_cell_data)
        _update_alg(
            alg, progress_bar=progress_bar, message='Transforming cell data into point data.'
        )
        active_scalars = None
        if not isinstance(self, pyvista.MultiBlock):
            active_scalars = self.active_scalars_name
        return _get_output(alg, active_scalars=active_scalars)

    @_deprecate_positional_args
    def ctp(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        pass_cell_data: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        **kwargs,
    ):
        """Transform cell data into point data.

        Point data are specified per node and cell data specified
        within cells.  Optionally, the input point data can be passed
        through to the output.

        This method is an alias for :func:`cell_data_to_point_data`.

        Parameters
        ----------
        pass_cell_data : bool, default: False
            If enabled, pass the input cell data through to the output.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        **kwargs : dict, optional
            Deprecated keyword argument ``pass_cell_arrays``.

        Returns
        -------
        DataSet | MultiBlock
            Dataset with the cell data transformed into point data.
            Return type matches input.

        """
        return self.cell_data_to_point_data(
            pass_cell_data=pass_cell_data,
            progress_bar=progress_bar,
            **kwargs,
        )

    @_deprecate_positional_args
    def point_data_to_cell_data(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        pass_point_data: bool = False,  # noqa: FBT001, FBT002
        categorical: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Transform point data into cell data.

        Point data are specified per node and cell data specified within cells.
        Optionally, the input point data can be passed through to the output.

        Parameters
        ----------
        pass_point_data : bool, default: False
            If enabled, pass the input point data through to the output.

        categorical : bool, default: False
            Control whether the source point data is to be treated as
            categorical. If ``True``,  histograming is used to assign the
            cell data. Specifically, a histogram is populated for each cell
            from the scalar values at each point, and the bin with the most
            elements is selected. In case of a tie, the smaller value is selected.

            .. note::

                If the point data is continuous, values that are almost equal (within
                ``1e-6``) are merged into a single bin. Otherwise, for discrete data
                the number of bins equals the number of unique values.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        DataSet | MultiBlock
            Dataset with the point data transformed into cell data.
            Return type matches input.

        See Also
        --------
        cell_data_to_point_data
            Similar transformation applied to cell data.
        :meth:`~pyvista.ImageDataFilters.points_to_cells`
            Re-mesh :class:`~pyvista.ImageData` to a cells-based representation.

        Examples
        --------
        Color cells by their z coordinates.  First, create point
        scalars based on z-coordinates of a sample sphere mesh.  Then
        convert this point data to cell data.  Use a low resolution
        sphere for emphasis of cell valued data.

        First, plot these values as point values to show the
        difference between point and cell data.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
        >>> sphere['Z Coordinates'] = sphere.points[:, 2]
        >>> sphere.plot()

        Now, convert these values to cell data and then plot it.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
        >>> sphere['Z Coordinates'] = sphere.points[:, 2]
        >>> sphere = sphere.point_data_to_cell_data()
        >>> sphere.plot()

        """
        alg = _vtk.vtkPointDataToCellData()
        alg.SetInputDataObject(self)
        alg.SetPassPointData(pass_point_data)
        alg.SetCategoricalData(categorical)
        _update_alg(
            alg, progress_bar=progress_bar, message='Transforming point data into cell data'
        )
        active_scalars = None
        if not isinstance(self, pyvista.MultiBlock):
            active_scalars = self.active_scalars_name
        return _get_output(alg, active_scalars=active_scalars)

    @_deprecate_positional_args
    def ptc(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        pass_point_data: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        **kwargs,
    ):
        """Transform point data into cell data.

        Point data are specified per node and cell data specified
        within cells.  Optionally, the input point data can be passed
        through to the output.

        This method is an alias for :func:`point_data_to_cell_data`.

        Parameters
        ----------
        pass_point_data : bool, default: False
            If enabled, pass the input point data through to the output.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        **kwargs : dict, optional
            Deprecated keyword argument ``pass_point_arrays``.

        Returns
        -------
        DataSet | MultiBlock
            Dataset with the point data transformed into cell data.
            Return type matches input.

        """
        return self.point_data_to_cell_data(
            pass_point_data=pass_point_data,
            progress_bar=progress_bar,
            **kwargs,
        )

    @_deprecate_positional_args
    def triangulate(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        inplace: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
        """Return an all triangle mesh.

        More complex polygons will be broken down into triangles.

        Parameters
        ----------
        inplace : bool, default: False
            Updates mesh in-place.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh containing only triangles.

        Examples
        --------
        Generate a mesh with quadrilateral faces.

        >>> import pyvista as pv
        >>> plane = pv.Plane()
        >>> plane.point_data.clear()
        >>> plane.plot(show_edges=True, line_width=5)

        Convert it to an all triangle mesh.

        >>> mesh = plane.triangulate()
        >>> mesh.plot(show_edges=True, line_width=5)

        """
        alg = _vtk.vtkDataSetTriangleFilter()
        alg.SetInputData(self)
        _update_alg(alg, progress_bar=progress_bar, message='Converting to triangle mesh')

        mesh = _get_output(alg)
        if inplace:
            self.copy_from(mesh, deep=False)
            return self
        return mesh

    @_deprecate_positional_args(allowed=['target'])
    def sample(  # type: ignore[misc]  # noqa: PLR0917
        self: _DataSetOrMultiBlockType,
        target: DataSet | _vtk.vtkDataSet,
        tolerance: float | None = None,
        pass_cell_data: bool = True,  # noqa: FBT001, FBT002
        pass_point_data: bool = True,  # noqa: FBT001, FBT002
        categorical: bool = False,  # noqa: FBT001, FBT002
        progress_bar: bool = False,  # noqa: FBT001, FBT002
        locator: Literal['cell', 'cell_tree', 'obb_tree', 'static_cell']
        | _vtk.vtkAbstractCellLocator
        | None = 'static_cell',
        pass_field_data: bool = True,  # noqa: FBT001, FBT002
        mark_blank: bool = True,  # noqa: FBT001, FBT002
        snap_to_closest_point: bool = False,  # noqa: FBT001, FBT002
    ):
        """Resample array data from a passed mesh onto this mesh.

        For `mesh1.sample(mesh2)`, the arrays from `mesh2` are sampled onto
        the points of `mesh1`.  This function interpolates within an
        enclosing cell.  This contrasts with
        :func:`pyvista.DataSetFilters.interpolate` that uses a distance
        weighting for nearby points.  If there is cell topology, `sample` is
        usually preferred.

        The point data 'vtkValidPointMask' stores whether the point could be sampled
        with a value of 1 meaning successful sampling. And a value of 0 means
        unsuccessful.

        This uses :vtk:`vtkResampleWithDataSet`.

        Parameters
        ----------
        target : pyvista.DataSet
            The vtk data object to sample from - point and cell arrays from
            this object are sampled onto the nodes of the ``dataset`` mesh.

        tolerance : float, optional
            Tolerance used to compute whether a point in the source is
            in a cell of the input.  If not given, tolerance is
            automatically generated.

        pass_cell_data : bool, default: True
            Preserve source mesh's original cell data arrays.

        pass_point_data : bool, default: True
            Preserve source mesh's original point data arrays.

        categorical : bool, default: False
            Control whether the source point data is to be treated as
            categorical. If the data is categorical, then the resultant data
            will be determined by a nearest neighbor interpolation scheme.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        locator : :vtk:`vtkAbstractCellLocator` or str or None, default: 'static_cell'
            Prototype cell locator to perform the ``FindCell()``
            operation.  If ``None``, uses the DataSet ``FindCell`` method.
            Valid strings with mapping to vtk cell locators are

                * 'cell' - :vtk:`vtkCellLocator`
                * 'cell_tree' - :vtk:`vtkCellTreeLocator`
                * 'obb_tree' - :vtk:`vtkOBBTree`
                * 'static_cell' - :vtk:`vtkStaticCellLocator`

        pass_field_data : bool, default: True
            Preserve source mesh's original field data arrays.

        mark_blank : bool, default: True
            Whether to mark blank points and cells in "vtkGhostType".

        snap_to_closest_point : bool, default: False
            Whether to snap to cell with closest point if no cell is found. Useful
            when sampling from data with vertex cells. Requires vtk >=9.3.0.

            .. versionadded:: 0.43

        Returns
        -------
        DataSet | MultiBlock
            Dataset containing resampled data.

        See Also
        --------
        pyvista.DataSetFilters.interpolate
            Interpolate values from one mesh onto another.

        pyvista.ImageDataFilters.resample
            Resample image data to modify its dimensions and spacing.

        Examples
        --------
        Resample data from another dataset onto a sphere.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = pv.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
        >>> data_to_probe = examples.load_uniform()
        >>> result = mesh.sample(data_to_probe)
        >>> result.plot(scalars='Spatial Point Data')

        If sampling from a set of points represented by a ``(n, 3)``
        shaped ``numpy.ndarray``, they need to be converted to a
        PyVista DataSet, e.g. :class:`pyvista.PolyData`, first.

        >>> import numpy as np
        >>> points = np.array([[1.5, 5.0, 6.2], [6.7, 4.2, 8.0]])
        >>> mesh = pv.PolyData(points)
        >>> result = mesh.sample(data_to_probe)
        >>> result['Spatial Point Data']
        pyvista_ndarray([ 46.5 , 225.12])

        See :ref:`resampling_example` and :ref:`interpolate_sample_example`
        for more examples using this filter.

        """
        alg = _vtk.vtkResampleWithDataSet()  # Construct the ResampleWithDataSet object
        alg.SetInputData(
            self
        )  # Set the Input data (actually the source i.e. where to sample from)
        # Set the Source data (actually the target, i.e. where to sample to)
        alg.SetSourceData(wrap(target))
        alg.SetPassCellArrays(pass_cell_data)
        alg.SetPassPointArrays(pass_point_data)
        alg.SetPassFieldArrays(pass_field_data)

        alg.SetMarkBlankPointsAndCells(mark_blank)
        alg.SetCategoricalData(categorical)

        if tolerance is not None:
            alg.SetComputeTolerance(False)
            alg.SetTolerance(tolerance)
        if locator:
            if isinstance(locator, str):
                locator_map = {
                    'cell': _vtk.vtkCellLocator(),
                    'cell_tree': _vtk.vtkCellTreeLocator(),
                    'obb_tree': _vtk.vtkOBBTree(),
                    'static_cell': _vtk.vtkStaticCellLocator(),
                }
                try:
                    locator = locator_map[locator]
                except KeyError as err:
                    msg = f'locator must be a string from {locator_map.keys()}, got {locator}'
                    raise ValueError(msg) from err
            alg.SetCellLocatorPrototype(locator)

        if snap_to_closest_point:
            try:
                alg.SnapToCellWithClosestPointOn()
            except AttributeError:  # pragma: no cover
                msg = '`snap_to_closest_point=True` requires vtk 9.3.0 or newer'
                raise VTKVersionError(msg)
        _update_alg(
            alg,
            progress_bar=progress_bar,
            message='Resampling array Data from a Passed Mesh onto Mesh',
        )
        return _get_output(alg)

    def cell_quality(  # type: ignore[misc]
        self: _DataSetOrMultiBlockType,
        quality_measure: Literal['all', 'all_valid']
        | _CellQualityLiteral
        | Sequence[_CellQualityLiteral] = 'scaled_jacobian',
        *,
        null_value: float = -1.0,
        progress_bar: bool = False,
    ) -> _DataSetOrMultiBlockType:
        r"""Compute a function of (geometric) quality for each cell of a mesh.

        The per-cell quality is added to the mesh's cell data, in an array with
        the same name as the quality measure. Cell types not supported by this
        filter or undefined quality of supported cell types will have an
        entry of ``-1``.

        See the :ref:`cell_quality_measures_table` below for all measures and the
        :class:`~pyvista.CellType` supported by each one.
        Defaults to computing the ``scaled_jacobian`` quality measure.

        .. _cell_quality_measures_table:

        .. include:: /api/core/cell_quality/cell_quality_measures_table.rst

        .. note::

            Refer to the `Verdict Library Reference Manual <https://github.com/sandialabs/verdict/raw/master/SAND2007-2853p.pdf>`_
            for low-level technical information about how each metric is computed.

        .. versionadded:: 0.45

        Parameters
        ----------
        quality_measure : str | sequence[str], default: 'scaled_jacobian'
            The cell quality measure(s) to use. May be either:

            - A single measure or a sequence of measures listed in
              :ref:`cell_quality_measures_table`.
            - ``'all'`` to compute all measures.
            - ``'all_valid'`` to only keep quality measures that are valid for the mesh's
              cell type(s).

            A separate array is created for each measure.

        null_value : float, default: -1.0
            Float value for undefined quality. Undefined quality are qualities
            that could be addressed by this filter but is not well defined for
            the particular geometry of cell in question, e.g. a volume query
            for a triangle. Undefined quality will always be undefined.
            The default value is -1.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        DataSet | MultiBlock
            Dataset with the computed mesh quality. Return type matches input.
            Cell data array(s) with the computed quality measure(s) are included.

        See Also
        --------
        :func:`~pyvista.cell_quality_info`
            Return information about a cell's quality measure, e.g. acceptable range.

        Examples
        --------
        Compute and plot the minimum angle of a sample sphere mesh.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=20, phi_resolution=20)
        >>> cqual = sphere.cell_quality('min_angle')
        >>> cqual.plot(show_edges=True)

        Quality measures like ``'volume'`` do not apply to 2D cells, and a null value
        of ``-1`` is returned.

        >>> qual = sphere.cell_quality('volume')
        >>> qual.get_data_range('volume')
        (np.float64(-1.0), np.float64(-1.0))

        Compute all valid quality measures for the sphere. These measures all return
        non-null values for :attr:`~pyvista.CellType.TRIANGLE` cells.

        >>> cqual = sphere.cell_quality('all_valid')
        >>> valid_measures = cqual.cell_data.keys()
        >>> valid_measures  # doctest: +NORMALIZE_WHITESPACE
        ['area',
         'aspect_frobenius',
         'aspect_ratio',
         'condition',
         'distortion',
         'max_angle',
         'min_angle',
         'radius_ratio',
         'relative_size_squared',
         'scaled_jacobian',
         'shape',
         'shape_and_size']

        See :ref:`mesh_quality_example` for more examples using this filter.

        """
        # Validate measures
        _validation.check_instance(quality_measure, (str, list, tuple), name='quality_measure')
        keep_valid_only = quality_measure == 'all_valid'
        measures_available = _get_cell_quality_measures()
        measures_available_names = cast(
            'list[_CellQualityLiteral]', list(measures_available.keys())
        )
        if quality_measure in ['all', 'all_valid']:
            measures_requested = measures_available_names
        else:
            measures = [quality_measure] if isinstance(quality_measure, str) else quality_measure
            for measure in measures:
                _validation.check_contains(
                    measures_available_names, must_contain=measure, name='quality_measure'
                )
            measures_requested = cast('list[_CellQualityLiteral]', measures)

        cell_quality = functools.partial(
            DataObjectFilters._dataset_cell_quality,
            measures_requested=measures_requested,
            measures_available=measures_available,
            keep_valid_only=keep_valid_only,
            null_value=null_value,
            progress_bar=progress_bar,
        )
        return (
            self.generic_filter(cell_quality)  # type: ignore[return-value]
            if isinstance(self, pyvista.MultiBlock)
            else cell_quality(self)
        )

    def _dataset_cell_quality(  # type: ignore[misc]
        self: _DataSetType,
        *,
        measures_requested,
        measures_available,
        keep_valid_only,
        null_value,
        progress_bar,
    ) -> _DataSetType:
        """Compute cell quality of a DataSet (internal method)."""
        CELL_QUALITY = 'CellQuality'

        alg = _vtk.vtkCellQuality()
        alg.SetUndefinedQuality(null_value)

        if 'size' in ''.join(measures_requested):
            # Need to compute mesh quality statistics to get average cell size.
            # We only need to do this once. This will create field data arrays:
            # 'TriArea', 'QuadArea', 'TetVolume', 'PyrVolume', 'WedgeVolume', 'HexVolume'
            # which are used later by vtkCellQuality
            mesh_quality = _vtk.vtkMeshQuality()
            mesh_quality.SaveCellQualityOff()
            mesh_quality.SetInputData(self)
            # Setting any 'Size' measure for any cell (tri, quad, etc.) is sufficient to
            # ensure all necessary base stats are computed for all cell types and for
            # all 'Size' measures
            mesh_quality.SetTriangleQualityMeasureToShapeAndSize()
            mesh_quality.Update()

            alg.SetInputDataObject(mesh_quality.GetOutput())
        else:
            alg.SetInputDataObject(self)

        output = self.copy()

        # Compute all measures
        for measure in measures_requested:
            # Set measure and update
            getattr(alg, measures_available[measure])()
            _update_alg(
                alg, progress_bar=progress_bar, message=f"Computing Cell Quality '{measure}'"
            )

            # Store the cell quality array with the output
            cell_quality_array = _get_output(alg).cell_data[CELL_QUALITY]
            if keep_valid_only and (
                np.max(cell_quality_array) == np.min(cell_quality_array) == null_value
            ):
                continue
            output.cell_data[measure] = cell_quality_array
        return output


def _get_cell_quality_measures() -> dict[str, str]:
    """Return snake case quality measure keys and vtkCellQuality attribute setter names."""
    # Get possible quality measures dynamically
    str_start = 'SetQualityMeasureTo'
    measures = {}
    for attr in dir(_vtk.vtkCellQuality):
        if attr.startswith(str_start):
            # Get the part after 'SetQualityMeasureTo'
            measure_name = attr[len(str_start) :]
            # Convert to snake case
            # Add underscore before uppercase letters, except the first one
            measure_name = re.sub(r'([a-z])([A-Z])', r'\1_\2', measure_name).lower()
            measures[measure_name] = attr
    return measures


def _cast_output_to_match_input_type(
    output_mesh: DataSet | MultiBlock, input_mesh: DataSet | MultiBlock
):
    # Ensure output type matches input type

    def cast_output(mesh_out: DataSet, mesh_in: DataSet):
        if isinstance(mesh_in, pyvista.PolyData) and not isinstance(mesh_out, pyvista.PolyData):
            return mesh_out.extract_geometry()
        elif isinstance(mesh_in, pyvista.PointSet) and not isinstance(mesh_out, pyvista.PointSet):
            return mesh_out.cast_to_pointset()
        return mesh_out

    def cast_output_blocks(mesh_out: MultiBlock, mesh_in: MultiBlock):
        # Replace all blocks in the output mesh with cast versions that match the input
        for (ids, _, block_out), block_in in zip(
            mesh_out.recursive_iterator('all', skip_none=True),
            mesh_in.recursive_iterator(skip_none=True),
        ):
            mesh_out.replace(ids, cast_output(block_out, block_in))
        return mesh_out

    return (
        cast_output_blocks(output_mesh, input_mesh)  # type: ignore[arg-type]
        if isinstance(output_mesh, pyvista.MultiBlock)
        else cast_output(output_mesh, input_mesh)  # type: ignore[arg-type]
    )
