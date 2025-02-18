"""Filters module with a class of common filters that can be applied to any vtkDataSet."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core.filters import _update_alg
from pyvista.core.utilities import Transform

if TYPE_CHECKING:
    from pyvista import MultiBlock
    from pyvista import RotationLike
    from pyvista import TransformLike
    from pyvista import VectorLike
    from pyvista.core._typing_core import ConcreteDataSetType
    from pyvista.core._typing_core._dataset_types import ConcreteDataSetAlias


class DataObjectFilters:
    """A set of common filters that can be applied to any DataSet or MultiBlock."""

    def transform(  # type: ignore[misc]
        self: ConcreteDataSetType | MultiBlock,
        trans: TransformLike,
        transform_all_input_vectors: bool = False,
        inplace: bool | None = None,
        progress_bar: bool = False,
    ):
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
            Shear transformations are not supported for ':class:`~pyvista.ImageData`.
            If present, any shear component is removed by the filter.

        .. note::
            Transforming :class:`~pyvista.ImageData` modifies its :class:`~pyvista.ImageData.origin`,
            :class:`~pyvista.ImageData.spacing`, and :class:`~pyvista.ImageData.direction_matrix`
            properties.

        .. deprecated:: 0.45.0
            `inplace` was previously defaulted to `True`. In the future this will change to `False`.

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
        pyvista.DataSet
            Transformed dataset.  Return type matches input unless
            input dataset is a :class:`pyvista.RectilinearGrid`, in which
            case the output datatype is a :class:`pyvista.StructuredGrid`.

        See Also
        --------
        :class:`pyvista.Transform`
            Describe linear transformations via a 4x4 matrix.

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
        from ._deprecate_transform_inplace_default_true import check_inplace

        inplace = check_inplace(cls=type(self), inplace=inplace)

        if isinstance(self, pyvista.MultiBlock):
            return self.generic_filter(
                'transform',
                trans=trans,
                transform_all_input_vectors=transform_all_input_vectors,
                inplace=inplace,
                progress_bar=progress_bar,
            )

        if inplace and isinstance(self, pyvista.RectilinearGrid):
            raise TypeError(f'Cannot transform a {self.__class__} inplace')

        t = trans if isinstance(trans, Transform) else Transform(trans)

        if t.matrix[3, 3] == 0:
            raise ValueError('Transform element (3,3), the inverse scale term, is zero')

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

        _update_alg(f, progress_bar, 'Transforming')
        res = pyvista.core.filters._get_output(f)

        def _restore_active_scalars(input_: ConcreteDataSetAlias, output_: ConcreteDataSetAlias):
            # make the previously active scalars active again
            input_.point_data.active_scalars_name = active_point_scalars_name
            input_.cell_data.active_scalars_name = active_cell_scalars_name

            # Only update output if necessary
            if input_ is not output_:
                output_.point_data.active_scalars_name = active_point_scalars_name
                output_.cell_data.active_scalars_name = active_cell_scalars_name

        if isinstance(self, pyvista.RectilinearGrid):
            output: ConcreteDataSetAlias = pyvista.StructuredGrid()
        elif inplace:
            output = self
        else:
            output = self.__class__()

        if isinstance(output, pyvista.ImageData):
            # vtkTransformFilter returns a StructuredGrid for legacy code (before VTK 9)
            # but VTK 9+ supports oriented images.
            # To keep an ImageData -> ImageData mapping, we copy the transformed data
            # from the filter output but manually transform the structure
            output.copy_structure(self)  # type: ignore[arg-type]
            current_matrix = output.index_to_physical_matrix
            new_matrix = pyvista.Transform(current_matrix).compose(t).matrix
            output.index_to_physical_matrix = new_matrix

            output.point_data.update(res.point_data, copy=False)
            output.cell_data.update(res.cell_data, copy=False)
            output.field_data.update(res.field_data, copy=False)
            _restore_active_scalars(self, output)
            return output

        _restore_active_scalars(self, res)

        # The output from the transform filter contains a shallow copy
        # of the original dataset except for the point arrays.  Here
        # we perform a copy so the two are completely unlinked.
        if inplace:
            output.copy_from(res, deep=False)
        else:
            output.copy_from(res, deep=True)
        return output

    def reflect(  # type: ignore[misc]
        self: ConcreteDataSetType | MultiBlock,
        normal: VectorLike[float],
        point: VectorLike[float] | None = None,
        inplace: bool = False,
        transform_all_input_vectors: bool = False,
        progress_bar: bool = False,
    ):
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
        pyvista.DataSet
            Reflected dataset.  Return type matches input.

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

    def rotate_x(  # type: ignore[misc]
        self: ConcreteDataSetType | MultiBlock,
        angle: float,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Rotate mesh about the x-axis.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
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
        pyvista.DataSet
            Rotated dataset.

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

    def rotate_y(  # type: ignore[misc]
        self: ConcreteDataSetType | MultiBlock,
        angle: float,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Rotate mesh about the y-axis.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
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
        pyvista.DataSet
            Rotated dataset.

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

    def rotate_z(  # type: ignore[misc]
        self: ConcreteDataSetType | MultiBlock,
        angle: float,
        point: VectorLike[float] = (0.0, 0.0, 0.0),
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Rotate mesh about the z-axis.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
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
        pyvista.DataSet
            Rotated dataset.

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

    def rotate_vector(  # type: ignore[misc]
        self: ConcreteDataSetType | MultiBlock,
        vector: VectorLike[float],
        angle: float,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Rotate mesh about a vector.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
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
        pyvista.DataSet
            Rotated dataset.

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

    def rotate(  # type: ignore[misc]
        self: ConcreteDataSetType | MultiBlock,
        rotation: RotationLike,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Rotate mesh about a point with a rotation matrix or ``Rotation`` object.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
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
        pyvista.DataSet
            Rotated dataset.

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
        self: ConcreteDataSetType | MultiBlock,
        xyz: VectorLike[float],
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Translate the mesh.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
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
        pyvista.DataSet
            Translated dataset.

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

    def scale(  # type: ignore[misc]
        self: ConcreteDataSetType | MultiBlock,
        xyz: float | VectorLike[float],
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
        point: VectorLike[float] | None = None,
    ):
        """Scale the mesh.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
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
        pyvista.DataSet
            Scaled dataset.

        See Also
        --------
        pyvista.Transform.scale
            Concatenate a scale matrix with a transformation.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> pl.show_axes()
        >>> _ = pl.show_grid()
        >>> mesh1 = examples.download_teapot()
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.subplot(0, 1)
        >>> pl.show_axes()
        >>> _ = pl.show_grid()
        >>> mesh2 = mesh1.scale([10.0, 10.0, 10.0], inplace=False)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show(cpos='xy')

        """
        transform = Transform().scale(xyz, point=point)
        return self.transform(
            transform,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def flip_x(  # type: ignore[misc]
        self: ConcreteDataSetType | MultiBlock,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Flip mesh about the x-axis.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        point : sequence[float], optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`center <pyvista.DataSet.center>`.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Flipped dataset.

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

    def flip_y(  # type: ignore[misc]
        self: ConcreteDataSetType | MultiBlock,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Flip mesh about the y-axis.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        point : VectorLike[float], optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`center <pyvista.DataSet.center>`.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Flipped dataset.

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

    def flip_z(  # type: ignore[misc]
        self: ConcreteDataSetType | MultiBlock,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Flip mesh about the z-axis.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        point : VectorLike[float], optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`center <pyvista.DataSet.center>`.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Flipped dataset.

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

    def flip_normal(  # type: ignore[misc]
        self: ConcreteDataSetType | MultiBlock,
        normal: VectorLike[float],
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Flip mesh about the normal.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        normal : VectorLike[float]
           Normal vector to flip about.

        point : VectorLike[float], optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`center <pyvista.DataSet.center>`.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Dataset flipped about its normal.

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
