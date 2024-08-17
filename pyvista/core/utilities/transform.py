"""Module containing the Transform class."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

import numpy as np

from pyvista.core import _validation
from pyvista.core import _vtk_core as _vtk
from pyvista.core.utilities.arrays import array_from_vtkmatrix
from pyvista.core.utilities.arrays import vtkmatrix_from_array
from pyvista.core.utilities.transformations import axis_angle_rotation
from pyvista.core.utilities.transformations import reflection

if TYPE_CHECKING:  # pragma: no cover
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import RotationLike
    from pyvista.core._typing_core import TransformLike
    from pyvista.core._typing_core import VectorLike


class Transform(_vtk.vtkTransform):
    """Describes linear transformations via a 4x4 matrix.

    A :class:`Transform` can be used to describe the full range of linear (also known
    as affine) coordinate transformations in three dimensions, which are internally
    represented as a 4x4 homogeneous transformation matrix.

    The transformation methods (e.g. :meth:`translate`, :meth:`rotate`,
    :meth:`concatenate`) can operate in either :meth:`pre_multiply` or
    :meth:`post_multiply` mode. In pre-multiply mode, any additional transformations
    will occur *before* any transformations represented by the current :attr:`matrix`.
    In post-multiply mode (the default), the additional transformation will occur
    *after* any transformations represented by the current matrix.

    .. note::

        This class performs all of its operations in a right-handed coordinate system
        with right-handed rotations. Some other graphics libraries use left-handed
        coordinate systems and rotations.

    .. versionadded:: 0.45

    Parameters
    ----------
    trans : TransformLike, optional
        Initialize the transform with a transformation. By default, the transform
        is initialized as the identity matrix.

    point : VectorLike[float], optional
        Point to use when concatenating some transformations such as scale, rotation, etc.
        If set, two additional transformations are concatenated and added to
        the :attr:`matrix_list`:

            - :meth:`translate` to ``point`` before the transformation
            - :meth:`translate` away from ``point`` after the transformation

        By default, this value is ``None``, which means that the scale, rotation, etc.
        transformations are performed about the origin ``(0, 0, 0)``.

    See Also
    --------
    :meth:`pyvista.DataSetFilters.transform`
        Apply a transformation to a mesh.

    Examples
    --------
    Create a transformation and use ``+`` to concatenate a translation.

    >>> import numpy as np
    >>> import pyvista as pv
    >>> position = (-0.6, -0.8, 2.1)
    >>> translation_T = pv.Transform() + position
    >>> translation_T.matrix
    array([[ 1. ,  0. ,  0. , -0.6],
           [ 0. ,  1. ,  0. , -0.8],
           [ 0. ,  0. ,  1. ,  2.1],
           [ 0. ,  0. ,  0. ,  1. ]])

    Using ``+`` performs the same concatenation as calling :meth:`translate`.

    >>> np.array_equal(
    ...     translation_T.matrix, pv.Transform().translate(position).matrix
    ... )
    True

    Create a transformation and use ``*`` to concatenate a scaling matrix.

    >>> scale_factor = 2.0
    >>> scaling_T = pv.Transform() * scale_factor
    >>> scaling_T.matrix
    array([[2., 0., 0., 0.],
           [0., 2., 0., 0.],
           [0., 0., 2., 0.],
           [0., 0., 0., 1.]])

    Using ``*`` performs the same concatenation as calling :meth:`scale`.

    >>> np.array_equal(
    ...     scaling_T.matrix, pv.Transform().scale(scale_factor).matrix
    ... )
    True

    Concatenate the two transformations using addition. This will concatenate with
    post-multiplication such that the transformations are applied in order from left to
    right, i.e. translate first, then scale.

    >>> transform_post = translation_T + scaling_T
    >>> transform_post.matrix
    array([[ 2. ,  0. ,  0. , -1.2],
           [ 0. ,  2. ,  0. , -1.6],
           [ 0. ,  0. ,  2. ,  4.2],
           [ 0. ,  0. ,  0. ,  1. ]])

    Post-multiplication is equivalent to using matrix multiplication on the
    arrays directly but with the arguments reversed:

    >>> mat_mul = scaling_T.matrix @ translation_T.matrix
    >>> np.array_equal(transform_post.matrix, mat_mul)
    True

    Alternatively, concatenate the transformations by chaining the methods with a
    single :class:`Transform` instance. Note that post-multiply is used by default.

    >>> transform_post = pv.Transform()
    >>> transform_post.multiply_mode
    'post'
    >>> _ = transform_post.translate(position).scale(scale_factor)
    >>> transform_post.matrix
    array([[ 2. ,  0. ,  0. , -1.2],
           [ 0. ,  2. ,  0. , -1.6],
           [ 0. ,  0. ,  2. ,  4.2],
           [ 0. ,  0. ,  0. ,  1. ]])

    Use :attr:`n_transformations` to check that there are two transformations.

    >>> transform_post.n_transformations
    2

    Use :attr:`matrix_list` to get a list of the transformations. Since
    post-multiplication is used, the translation matrix is first in the list since
    it was applied first, and the scale matrix is second.

    >>> transform_post.matrix_list[0]  # translation
    array([[ 1. ,  0. ,  0. , -0.6],
           [ 0. ,  1. ,  0. , -0.8],
           [ 0. ,  0. ,  1. ,  2.1],
           [ 0. ,  0. ,  0. ,  1. ]])

    >>> transform_post.matrix_list[1]  # scaling
    array([[2., 0., 0., 0.],
           [0., 2., 0., 0.],
           [0., 0., 2., 0.],
           [0., 0., 0., 1.]])

    Create a similar transform but use pre-multiplication this time. Concatenate the
    transformations in the same order as before using :meth:`translate` and :meth`scale`.

    >>> transform_pre = pv.Transform().pre_multiply()
    >>> _ = transform_pre.translate(position).scale(scale_factor)

    Alternatively, create the transform using matrix multiplication. Matrix
    multiplication concatenates the transformations using pre-multiply semantics such
    that the transformations are applied in order from right to left, i.e. scale first,
    then translate.

    >>> transform_pre = translation_T @ scaling_T
    >>> transform_pre.matrix
    array([[ 2. ,  0. ,  0. , -0.6],
           [ 0. ,  2. ,  0. , -0.8],
           [ 0. ,  0. ,  2. ,  2.1],
           [ 0. ,  0. ,  0. ,  1. ]])

    This is equivalent to using matrix multiplication directly on the arrays:

    >>> mat_mul = translation_T.matrix @ scaling_T.matrix
    >>> np.array_equal(transform_pre.matrix, mat_mul)
    True

    Show the matrix list again. Note how the order with pre-multiplication is the
    reverse of post-multiplication.

    >>> transform_pre.matrix_list[0]  # scaling
    array([[2., 0., 0., 0.],
           [0., 2., 0., 0.],
           [0., 0., 2., 0.],
           [0., 0., 0., 1.]])

    >>> transform_pre.matrix_list[1]  # translation
    array([[ 1. ,  0. ,  0. , -0.6],
           [ 0. ,  1. ,  0. , -0.8],
           [ 0. ,  0. ,  1. ,  2.1],
           [ 0. ,  0. ,  0. ,  1. ]])

    Apply the two post- and pre-multiplied transformations to a dataset and plot them.
    Note how the meshes have different positions since post- and pre-multiplication
    produce different transformations.

    >>> mesh_post = pv.Sphere().transform(transform_post)
    >>> mesh_pre = pv.Cone().transform(transform_pre)
    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh_post, color='goldenrod')
    >>> _ = pl.add_mesh(mesh_pre, color='teal')
    >>> _ = pl.add_axes_at_origin()
    >>> pl.show()

    Get the concatenated inverse transformation matrix of the pre-multiplication case.

    >>> inverse_matrix = transform_pre.inverse_matrix
    >>> inverse_matrix
    array([[ 0.5 ,  0.  ,  0.  ,  0.3 ],
           [ 0.  ,  0.5 ,  0.  ,  0.4 ],
           [ 0.  ,  0.  ,  0.5 , -1.05],
           [ 0.  ,  0.  ,  0.  ,  1.  ]])

    Similar to using :attr:`matrix_list`, we can inspect the individual transformation
    inverses with :attr:`inverse_matrix_list`.

    >>> transform_pre.inverse_matrix_list[0]  # inverse scaling
    array([[0.5, 0. , 0. , 0. ],
           [0. , 0.5, 0. , 0. ],
           [0. , 0. , 0.5, 0. ],
           [0. , 0. , 0. , 1. ]])

    >>> transform_pre.inverse_matrix_list[1]  # inverse translation
    array([[ 1. ,  0. ,  0. ,  0.6],
           [ 0. ,  1. ,  0. ,  0.8],
           [ 0. ,  0. ,  1. , -2.1],
           [ 0. ,  0. ,  0. ,  1. ]])

    Transform the mesh by its inverse to restore it to its original un-scaled state
    and positioning at the origin.

    >>> mesh_pre_inverted = mesh_pre.transform(inverse_matrix)
    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh_pre_inverted, color='teal')
    >>> _ = pl.add_axes_at_origin()
    >>> pl.show()
    """

    def __init__(
        self, trans: TransformLike | None = None, *, point: VectorLike[float] | None = None
    ):
        super().__init__()
        self.multiply_mode = 'post'
        self.point = point  # type: ignore[assignment]
        self.check_finite = True
        if trans is not None:
            self.matrix = trans  # type: ignore[assignment]

    def __add__(self, other: VectorLike[float] | TransformLike) -> Transform:
        """:meth:`concatenate` this transform using post-multiply semantics.

        Use :meth:`translate` for length-3 vector inputs, and :meth:`concatenate`
        otherwise for transform-like inputs.
        """
        copied = self.copy()
        try:
            transform = copied.translate(other, multiply_mode='post')
        except (ValueError, TypeError):
            try:
                transform = copied.concatenate(other, multiply_mode='post')
            except TypeError:
                raise TypeError(
                    f"Unsupported operand type(s) for +: '{self.__class__.__name__}' and '{type(other).__name__}'\n"
                    f"The right-side argument must be transform-like."
                )
            except ValueError:
                raise ValueError(
                    f"Unsupported operand value(s) for +: '{self.__class__.__name__}' and '{type(other).__name__}'\n"
                    f"The right-side argument must be a length-3 vector or have 3x3 or 4x4 shape."
                )
        return transform

    def __radd__(self, other: VectorLike[float] | TransformLike) -> Transform:
        """:meth:`translate` this transform using pre-multiply semantics."""
        try:
            return self.copy().translate(other, multiply_mode='pre')
        except TypeError:
            raise TypeError(
                f"Unsupported operand type(s) for +: '{type(other).__name__}' and '{self.__class__.__name__}'\n"
                f"The left-side argument must be a length-3 vector."
            )
        except ValueError:
            raise ValueError(
                f"Unsupported operand value(s) for +: '{type(other).__name__}' and '{self.__class__.__name__}'\n"
                f"The left-side argument must be a length-3 vector."
            )

    def __mul__(self, other: float | VectorLike[float] | TransformLike) -> Transform:
        """:meth:`scale` this transform using post-multiply semantics."""
        try:
            return self.copy().scale(other, multiply_mode='post')

        except TypeError:
            raise TypeError(
                f"Unsupported operand type(s) for *: '{self.__class__.__name__}' and '{type(other).__name__}'\n"
                f"The right-side argument must be a single number or a length-3 vector."
            )
        except ValueError:
            raise ValueError(
                f"Unsupported operand value(s) for *: '{self.__class__.__name__}' and '{type(other).__name__}'\n"
                f"The right-side argument must be a single number or a length-3 vector."
            )

    def __rmul__(self, other: float | VectorLike[float] | TransformLike) -> Transform:
        """:meth:`scale` this transform using pre-multiply semantics."""
        try:
            return self.copy().scale(other, multiply_mode='pre')
        except TypeError:
            raise TypeError(
                f"Unsupported operand type(s) for *: '{type(other).__name__}' and '{self.__class__.__name__}'\n"
                f"The left-side argument must be a single number or a length-3 vector."
            )
        except ValueError:
            raise ValueError(
                f"Unsupported operand value(s) for *: '{type(other).__name__}' and '{self.__class__.__name__}'\n"
                f"The left-side argument must be a single number or a length-3 vector."
            )

    def __matmul__(self, other: TransformLike) -> Transform:
        """:meth:`concatenate` this transform using pre-multiply semantics."""
        try:
            return self.copy().concatenate(other, multiply_mode='pre')
        except TypeError:
            raise TypeError(
                f"Unsupported operand type(s) for @: '{self.__class__.__name__}' and '{type(other).__name__}'\n"
                f"The right-side argument must be transform-like."
            )
        except ValueError:
            raise ValueError(
                f"Unsupported operand value(s) for @: '{self.__class__.__name__}' and '{type(other).__name__}'\n"
                f"The right-side argument must be transform-like."
            )

    def copy(self) -> Transform:
        """Return a deep copy of the transform.

        Returns
        -------
        Transform
            Deep copy of this transform.

        Examples
        --------
        Create a scaling transform.

        >>> import pyvista as pv
        >>> transform = pv.Transform().scale(1, 2, 3)
        >>> transform.matrix
        array([[1., 0., 0., 0.],
               [0., 2., 0., 0.],
               [0., 0., 3., 0.],
               [0., 0., 0., 1.]])

        Copy the transform.

        >>> copied = transform.copy()
        >>> copied.matrix
        array([[1., 0., 0., 0.],
               [0., 2., 0., 0.],
               [0., 0., 3., 0.],
               [0., 0., 0., 1.]])

        >>> copied is transform
        False
        """
        new_transform = Transform()
        new_transform.DeepCopy(self)

        # Need to copy other props not stored by vtkTransform
        new_transform.multiply_mode = self.multiply_mode

        return new_transform

    def __repr__(self):
        """Representation of the transform."""

        def _matrix_repr():
            repr_ = np.array_repr(self.matrix)
            return repr_.replace('array(', '      ').replace(')', '').replace('      [', '[')

        matrix_repr_lines = _matrix_repr().split('\n')
        lines = [
            f'{type(self).__name__} ({hex(id(self))})',
            f'  Num Transformations: {self.n_transformations}',
            f'  Matrix:  {matrix_repr_lines[0]}',
            f'           {matrix_repr_lines[1]}',
            f'           {matrix_repr_lines[2]}',
            f'           {matrix_repr_lines[3]}',
        ]
        return "\n".join(lines)

    @property
    def point(self) -> tuple[float, float, float] | None:  # numpydoc ignore=RT01
        """Point to use when concatenating some transformations such as scale, rotation, etc.

        If set, two additional transformations are concatenated and added to
        the :attr:`matrix_list`:

            - :meth:`translate` to ``point`` before the transformation
            - :meth:`translate` away from ``point`` after the transformation

        By default, this value is ``None``, which means that the scale, rotation, etc.
        transformations are performed about the origin ``(0, 0, 0)``.
        """
        return self._point

    @point.setter
    def point(self, point: VectorLike[float] | None):  # numpydoc ignore=GL08
        self._point = (
            None
            if point is None
            else _validation.validate_array3(point, dtype_out=float, to_tuple=True, name='point')
        )

    @property
    def multiply_mode(self) -> Literal['pre', 'post']:  # numpydoc ignore=RT01
        """Set or get the multiplication mode.

        Set this to ``'pre'`` to set the multiplication mode to :meth:`pre_multiply`.
        Set this to ``'post'`` to set it to :meth:`post_multiply`.

        In pre-multiply mode, any additional transformations (e.g. using
        :meth:`translate`,:meth:`concatenate`, etc.) will occur *before* any
        transformations represented by the current :attr:`matrix`.
        In post-multiply mode, the additional transformation will occur *after* any
        transformations represented by the current matrix.
        """
        return self._multiply_mode

    @multiply_mode.setter
    def multiply_mode(self, multiply_mode: Literal['pre', 'post']) -> None:  # numpydoc ignore=GL08
        _validation.check_contains(
            item=multiply_mode, container=['pre', 'post'], name='multiply mode'
        )
        self.pre_multiply() if multiply_mode == 'pre' else self.post_multiply()

    def pre_multiply(self) -> Transform:  # numpydoc ignore=RT01
        """Set the multiplication mode to pre-multiply.

        In pre-multiply mode, any additional transformations (e.g. using
        :meth:`translate`,:meth:`concatenate`, etc.) will occur *before* any
        transformations represented by the current :attr:`matrix`.

        Examples
        --------
        >>> import pyvista as pv
        >>> transform = pv.Transform().pre_multiply()
        >>> transform.multiply_mode
        'pre'
        """
        self._multiply_mode: Literal['pre', 'post'] = 'pre'
        self.PreMultiply()
        return self

    def post_multiply(self) -> Transform:  # numpydoc ignore=RT01
        """Set the multiplication mode to post-multiply.

        In post-multiply mode, any additional transformations (e.g. using
        :meth:`translate`,:meth:`concatenate`, etc.) will occur *after* any
        transformations represented by the current :attr:`matrix`.

        Examples
        --------
        >>> import pyvista as pv
        >>> transform = pv.Transform().post_multiply()
        >>> transform.multiply_mode
        'post'
        """
        self._multiply_mode = 'post'
        self.PostMultiply()
        return self

    def scale(
        self,
        *factor,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Concatenate a scale matrix.

        Create a scale matrix and :meth:`concatenate` it with the current
        transformation :attr:`matrix` according to pre-multiply or post-multiply
        semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        *factor : float | VectorLike[float]
            Scale factor(s) to use. Use a single number for uniform scaling or
            three numbers for non-uniform scaling.

        point : VectorLike[float], optional
            Point to scale from. By default, the object's :attr:`point` is used,
            but this can be overridden.
            If set, two additional transformations are concatenated and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the scaling
                - :meth:`translate` away from ``point`` after the scaling

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when concatenating the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        :meth:`pyvista.DataSet.scale`
            Scale a mesh.

        Examples
        --------
        Concatenate a scale matrix.

        >>> import pyvista as pv
        >>> transform = pv.Transform().scale(1, 2, 3)
        >>> transform.matrix
        array([[1., 0., 0., 0.],
               [0., 2., 0., 0.],
               [0., 0., 3., 0.],
               [0., 0., 0., 1.]])

        Concatenate a second scale matrix using ``*``.

        >>> transform = transform * 2
        >>> transform.matrix
        array([[2., 0., 0., 0.],
               [0., 4., 0., 0.],
               [0., 0., 6., 0.],
               [0., 0., 0., 1.]])

        Scale from a point. Check the :attr:`matrix_list` to see that a translation
        is added before and after the scaling.

        >>> transform = pv.Transform().scale(7, point=(1, 2, 3))
        >>> translation_to_origin = transform.matrix_list[0]
        >>> translation_to_origin
        array([[ 1.,  0.,  0., -1.],
               [ 0.,  1.,  0., -2.],
               [ 0.,  0.,  1., -3.],
               [ 0.,  0.,  0.,  1.]])

        >>> scale = transform.matrix_list[1]
        >>> scale
        array([[7., 0., 0., 0.],
               [0., 7., 0., 0.],
               [0., 0., 7., 0.],
               [0., 0., 0., 1.]])

        >>> translation_from_origin = transform.matrix_list[2]
        >>> translation_from_origin
        array([[1., 0., 0., 1.],
               [0., 1., 0., 2.],
               [0., 0., 1., 3.],
               [0., 0., 0., 1.]])
        """
        valid_factor = _validation.validate_array3(
            factor, broadcast=True, dtype_out=float, name='scale factor'
        )
        transform = _vtk.vtkTransform()
        transform.Scale(valid_factor)
        return self._concatenate_with_translations(
            transform, point=point, multiply_mode=multiply_mode
        )

    def reflect(
        self,
        *normal,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Concatenate a reflection matrix.

        Create a reflection matrix and :meth:`concatenate` it with the current
        transformation :attr:`matrix` according to pre-multiply or post-multiply
        semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        *normal : VectorLike[float]
            Normal direction for reflection.

        point : VectorLike[float], optional
            Point to reflect about. By default, the object's :attr:`point` is used,
            but this can be overridden.
            If set, two additional transformations are concatenated and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the reflection
                - :meth:`translate` away from ``point`` after the reflection

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when concatenating the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        :meth:`pyvista.DataSet.reflect`
            Reflect a mesh.

        Examples
        --------
        Concatenate a reflection matrix.

        >>> import pyvista as pv
        >>> transform = pv.Transform()
        >>> _ = transform.reflect(0, 0, 1)
        >>> transform.matrix
        array([[ 1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0., -1.,  0.],
               [ 0.,  0.,  0.,  1.]])

        Concatenate a second reflection matrix.

        >>> _ = transform.reflect((1, 0, 0))
        >>> transform.matrix
        array([[-1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0., -1.,  0.],
               [ 0.,  0.,  0.,  1.]])
        """
        valid_normal = _validation.validate_array3(
            normal, dtype_out=float, name='reflection normal'
        )
        transform = reflection(valid_normal)
        return self._concatenate_with_translations(
            transform, point=point, multiply_mode=multiply_mode
        )

    def translate(
        self, *vector, multiply_mode: Literal['pre', 'post'] | None = None
    ) -> Transform:  # numpydoc ignore=RT01
        """Concatenate a translation matrix.

        Create a translation matrix and :meth:`concatenate` it with the current
        transformation :attr:`matrix` according to pre-multiply or post-multiply
        semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        *vector : VectorLike[float]
            Vector to use for the translation.

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when concatenating the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        :meth:`pyvista.DataSet.translate`
            Translate a mesh.

        Examples
        --------
        Concatenate a translation matrix.

        >>> import pyvista as pv
        >>> transform = pv.Transform().translate(1, 2, 3)
        >>> transform.matrix
        array([[1., 0., 0., 1.],
               [0., 1., 0., 2.],
               [0., 0., 1., 3.],
               [0., 0., 0., 1.]])

        Concatenate a second translation matrix using ``+``.

        >>> transform = transform + (1, 1, 1)
        >>> transform.matrix
        array([[1., 0., 0., 2.],
               [0., 1., 0., 3.],
               [0., 0., 1., 4.],
               [0., 0., 0., 1.]])
        """
        valid_vector = _validation.validate_array3(
            vector, dtype_out=float, name='translation vector'
        )
        transform = _vtk.vtkTransform()
        transform.Translate(valid_vector)
        return self.concatenate(transform, multiply_mode=multiply_mode)

    def rotate(
        self,
        rotation: RotationLike,
        *,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Concatenate a rotation matrix.

        Create a rotation matrix and :meth:`concatenate` it with the current
        transformation :attr:`matrix` according to pre-multiply or post-multiply
        semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        rotation : RotationLike
            3x3 rotation matrix or a SciPy ``Rotation`` object.

        point : VectorLike[float], optional
            Point to rotate about. By default, the object's :attr:`point` is used,
            but this can be overridden.
            If set, two additional transformations are concatenated and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the rotation
                - :meth:`translate` away from ``point`` after the rotation

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when concatenating the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        pyvista.DataSet.rotate
            Rotate a mesh.

        Examples
        --------
        Concatenate a rotation matrix. In this case the rotation rotates about the
        z-axis by 90 degrees.

        >>> import pyvista as pv
        >>> rotation_z_90 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        >>> transform = pv.Transform().rotate(rotation_z_90)
        >>> transform.matrix
        array([[ 0., -1.,  0.,  0.],
               [ 1.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  1.]])

        Concatenate a second rotation matrix. In this case we use the same rotation as
        before.

        >>> _ = transform.rotate(rotation_z_90)

        The result is a matrix that rotates about the z-axis by 180 degrees.

        >>> transform.matrix
        array([[-1.,  0.,  0.,  0.],
               [ 0., -1.,  0.,  0.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  1.]])

        Rotate about a point. Check the :attr:`matrix_list` to see that a translation
        is added before and after the rotation.

        >>> transform = pv.Transform().rotate(
        ...     rotation_z_90, point=(1, 2, 3)
        ... )
        >>> translation_to_origin = transform.matrix_list[0]
        >>> translation_to_origin
        array([[ 1.,  0.,  0., -1.],
               [ 0.,  1.,  0., -2.],
               [ 0.,  0.,  1., -3.],
               [ 0.,  0.,  0.,  1.]])

        >>> rotation = transform.matrix_list[1]
        >>> rotation
        array([[ 0., -1.,  0.,  0.],
               [ 1.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  1.]])

        >>> translation_from_origin = transform.matrix_list[2]
        >>> translation_from_origin
        array([[1., 0., 0., 1.],
               [0., 1., 0., 2.],
               [0., 0., 1., 3.],
               [0., 0., 0., 1.]])
        """
        valid_rotation = _validation.validate_transform3x3(
            rotation, must_be_finite=self.check_finite, name='rotation'
        )
        return self._concatenate_with_translations(
            valid_rotation, point=point, multiply_mode=multiply_mode
        )

    def rotate_x(
        self,
        angle: float,
        *,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Concatenate a rotation about the x-axis.

        Create a matrix for rotation about the x-axis and :meth:`concatenate`
        it with the current transformation :attr:`matrix` according to pre-multiply or
        post-multiply semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the x-axis.

        point : VectorLike[float], optional
            Point to rotate about. By default, the object's :attr:`point` is used,
            but this can be overridden.
            If set, two additional transformations are concatenated and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the rotation
                - :meth:`translate` away from ``point`` after the rotation

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when concatenating the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        pyvista.DataSet.rotate_x
            Rotate a mesh about the x-axis.

        Examples
        --------
        Concatenate a rotation about the x-axis.

        >>> import pyvista as pv
        >>> transform = pv.Transform().rotate_x(90)
        >>> transform.matrix
        array([[ 1.,  0.,  0.,  0.],
               [ 0.,  0., -1.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  0.,  1.]])

        Concatenate a second rotation about the x-axis.

        >>> _ = transform.rotate_x(45)

        The result is a matrix that rotates about the x-axis by 135 degrees.

        >>> transform.matrix
        array([[ 1.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        , -0.70710678, -0.70710678,  0.        ],
               [ 0.        ,  0.70710678, -0.70710678,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  1.        ]])
        """
        transform = axis_angle_rotation((1, 0, 0), angle, deg=True)
        return self._concatenate_with_translations(
            transform, point=point, multiply_mode=multiply_mode
        )

    def rotate_y(
        self,
        angle: float,
        *,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Concatenate a rotation about the y-axis.

        Create a matrix for rotation about the y-axis and :meth:`concatenate`
        it with the current transformation :attr:`matrix` according to pre-multiply or
        post-multiply semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the y-axis.

        point : VectorLike[float], optional
            Point to rotate about. By default, the object's :attr:`point` is used,
            but this can be overridden.
            If set, two additional transformations are concatenated and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the rotation
                - :meth:`translate` away from ``point`` after the rotation

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when concatenating the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        pyvista.DataSet.rotate_y
            Rotate a mesh about the y-axis.

        Examples
        --------
        Concatenate a rotation about the y-axis.

        >>> import pyvista as pv
        >>> transform = pv.Transform().rotate_y(90)
        >>> transform.matrix
        array([[ 0.,  0.,  1.,  0.],
               [ 0.,  1.,  0.,  0.],
               [-1.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  1.]])

        Concatenate a second rotation about the y-axis.

        >>> _ = transform.rotate_y(45)

        The result is a matrix that rotates about the y-axis by 135 degrees.

        >>> transform.matrix
        array([[-0.70710678,  0.        ,  0.70710678,  0.        ],
               [ 0.        ,  1.        ,  0.        ,  0.        ],
               [-0.70710678,  0.        , -0.70710678,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  1.        ]])
        """
        transform = axis_angle_rotation((0, 1, 0), angle, deg=True)
        return self._concatenate_with_translations(
            transform, point=point, multiply_mode=multiply_mode
        )

    def rotate_z(
        self,
        angle: float,
        *,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Concatenate a rotation about the z-axis.

        Create a matrix for rotation about the z-axis and :meth:`concatenate`
        it with the current transformation :attr:`matrix` according to pre-multiply or
        post-multiply semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the z-axis.

        point : VectorLike[float], optional
            Point to rotate about. By default, the object's :attr:`point` is used,
            but this can be overridden.
            If set, two additional transformations are concatenated and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the rotation
                - :meth:`translate` away from ``point`` after the rotation

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when concatenating the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        pyvista.DataSet.rotate_z
            Rotate a mesh about the z-axis.

        Examples
        --------
        Concatenate a rotation about the z-axis.

        >>> import pyvista as pv
        >>> transform = pv.Transform().rotate_z(90)
        >>> transform.matrix
        array([[ 0., -1.,  0.,  0.],
               [ 1.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  1.]])

        Concatenate a second rotation about the z-axis.

        >>> _ = transform.rotate_z(45)

        The result is a matrix that rotates about the z-axis by 135 degrees.

        >>> transform.matrix
        array([[-0.70710678, -0.70710678,  0.        ,  0.        ],
               [ 0.70710678, -0.70710678,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  1.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  1.        ]])
        """
        transform = axis_angle_rotation((0, 0, 1), angle, deg=True)
        return self._concatenate_with_translations(
            transform, point=point, multiply_mode=multiply_mode
        )

    def rotate_vector(
        self,
        vector: VectorLike[float],
        angle: float,
        *,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Concatenate a rotation about a vector.

        Create a matrix for rotation about the vector and :meth:`concatenate`
        it with the current transformation :attr:`matrix` according to pre-multiply or
        post-multiply semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        vector : VectorLike[float]
            Vector to rotate about.

        angle : float
            Angle in degrees to rotate about the vector.

        point : VectorLike[float], optional
            Point to rotate about. By default, the object's :attr:`point` is used,
            but this can be overridden.
            If set, two additional transformations are concatenated and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the rotation
                - :meth:`translate` away from ``point`` after the rotation

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when concatenating the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        pyvista.DataSet.rotate_vector
            Rotate a mesh about a vector.

        Examples
        --------
        Concatenate a rotation of 30 degrees about the ``(1, 1, 1)`` axis.

        >>> import pyvista as pv
        >>> transform = pv.Transform().rotate_vector((1, 1, 1), 30)
        >>> transform.matrix
        array([[ 0.9106836 , -0.24401694,  0.33333333,  0.        ],
               [ 0.33333333,  0.9106836 , -0.24401694,  0.        ],
               [-0.24401694,  0.33333333,  0.9106836 ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  1.        ]])

        Concatenate a second rotation of 45 degrees about the ``(1, 2, 3)`` axis.

        >>> _ = transform.rotate_vector((1, 2, 3), 45)
        >>> transform.matrix
        array([[ 0.38042304, -0.50894634,  0.77217351,  0.        ],
               [ 0.83349512,  0.55045308, -0.04782562,  0.        ],
               [-0.40070461,  0.66179682,  0.63360933,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  1.        ]])
        """
        transform = axis_angle_rotation(vector, angle, deg=True)
        return self._concatenate_with_translations(
            transform, point=point, multiply_mode=multiply_mode
        )

    def concatenate(
        self, transform: TransformLike, *, multiply_mode: Literal['pre', 'post'] | None = None
    ) -> Transform:  # numpydoc ignore=RT01
        """Concatenate a transformation matrix.

        Create a 4x4 matrix from any transform-like input and concatenate it with the
        current transformation :attr:`matrix` according to pre-multiply or post-multiply
        semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        transform : TransformLike
            Any transform-like input such as a 3x3 or 4x4 array or matrix.

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when concatenating the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        Examples
        --------
        Define an arbitrary 4x4 affine transformation matrix and concatenate it.

        >>> import pyvista as pv
        >>> array = [
        ...     [0.707, -0.707, 0, 0],
        ...     [0.707, 0.707, 0, 0],
        ...     [0, 0, 1, 1.5],
        ...     [0, 0, 0, 2],
        ... ]
        >>> transform = pv.Transform().concatenate(array)
        >>> transform.matrix
        array([[ 0.707, -0.707,  0.   ,  0.   ],
               [ 0.707,  0.707,  0.   ,  0.   ],
               [ 0.   ,  0.   ,  1.   ,  1.5  ],
               [ 0.   ,  0.   ,  0.   ,  2.   ]])

        Define a second transformation and use ``+`` to concatenate it.

        >>> array = [[1, 0, 0], [0, 0, -1], [0, -1, 0]]
        >>> transform = transform + array
        >>> transform.matrix
        array([[ 0.707, -0.707,  0.   ,  0.   ],
               [ 0.   ,  0.   , -1.   , -1.5  ],
               [-0.707, -0.707,  0.   ,  0.   ],
               [ 0.   ,  0.   ,  0.   ,  2.   ]])
        """
        # Make sure we have a vtkTransform
        if isinstance(transform, _vtk.vtkTransform):
            vtk_transform = transform
        else:
            array = _validation.validate_transform4x4(
                transform, must_be_finite=self.check_finite, name='matrix'
            )
            vtk_transform = _vtk.vtkTransform()
            vtk_transform.SetMatrix(vtkmatrix_from_array(array))

        if multiply_mode is not None:
            # Override multiply mode
            original_mode = self.multiply_mode
            self.multiply_mode = multiply_mode

        self.Concatenate(vtk_transform)

        if multiply_mode is not None:
            self.multiply_mode = original_mode

        return self

    @property
    def matrix(self) -> NumpyArray[float]:
        """Return or set the current transformation matrix.

        Notes
        -----
        This matrix is a single 4x4 matrix computed from concatenating all
        transformations. Use :attr:`matrix_list` instead to get a list of the
        individual transformations.

        See Also
        --------
        inverse_matrix
            Return the inverse of the current transformation.

        Returns
        -------
        NDArray[float]
            Current transformation matrix.
        """
        array = array_from_vtkmatrix(self.GetMatrix())
        if self.check_finite:
            _validation.check_finite(array, name='matrix')
        return array

    @matrix.setter
    def matrix(self, trans: TransformLike):  # numpydoc ignore=GL08
        self.identity()
        self.concatenate(trans)

    @property
    def inverse_matrix(self) -> NumpyArray[float]:
        """Return the inverse of the current transformation :attr:`matrix`.

        Notes
        -----
        This matrix is a single 4x4 matrix computed from concatenating the inverse of
        all transformations. Use :attr:`inverse_matrix_list` instead to get a list of
        the individual inverse transformations.

        See Also
        --------
        matrix
            Return the current transformation matrix.

        Returns
        -------
        NDArray[float]
            Current inverse transformation matrix.
        """
        array = array_from_vtkmatrix(self.GetInverse().GetMatrix())
        if self.check_finite:
            _validation.check_finite(array, name='matrix')
        return array

    @property
    def matrix_list(self) -> list[NumpyArray[float]]:
        """Return a list of all current transformation matrices.

        Notes
        -----
        The list comprises all 4x4 transformation matrices. Use :attr:`matrix` instead
        to get the concatenated result as a single 4x4 matrix.

        See Also
        --------
        inverse_matrix_list
            Return the current transformation matrix.

        Returns
        -------
        list[NDArray[float]]
            List of all current transformation matrices.
        """
        return [
            array_from_vtkmatrix(self.GetConcatenatedTransform(i).GetMatrix())
            for i in range(self.n_transformations)
        ]

    @property
    def inverse_matrix_list(self) -> list[NumpyArray[float]]:
        """Return a list of all inverse transformations applied by this :class:`Transform`.

        Notes
        -----
        The list comprises all 4x4 inverse transformation matrices. Use
        :attr:`inverse_matrix` instead to get the concatenated result as a single
        4x4 matrix.

        See Also
        --------
        matrix_list
            Return a list of all transformation matrices.

        Returns
        -------
        list[NDArray[float]]
            List of all current inverse transformation matrices.
        """
        return [
            array_from_vtkmatrix(self.GetConcatenatedTransform(i).GetInverse().GetMatrix())
            for i in range(self.n_transformations)
        ]

    @property
    def n_transformations(self) -> int:  # numpydoc ignore: RT01
        """Return the current number of concatenated transformations."""
        return self.GetNumberOfConcatenatedTransforms()

    def invert(self) -> Transform:  # numpydoc ignore: RT01
        """Invert the current transformation.

        The current transformation :attr:`matrix` (including all matrices in the
        :attr:`matrix_list`) is inverted every time :meth:`invert` is called.

        Use :attr:`is_inverted` to check if the transformations are currently inverted.

        Examples
        --------
        Concatenate an arbitrary transformation.

        >>> import pyvista as pv
        >>> transform = pv.Transform().scale(2.0)
        >>> transform.matrix
        array([[2., 0., 0., 0.],
               [0., 2., 0., 0.],
               [0., 0., 2., 0.],
               [0., 0., 0., 1.]])

        Check if the transformation is inverted.

        >>> transform.is_inverted
        False

        Invert the transformation and show the matrix.

        >>> _ = transform.invert()
        >>> transform.matrix
        array([[0.5, 0. , 0. , 0. ],
               [0. , 0.5, 0. , 0. ],
               [0. , 0. , 0.5, 0. ],
               [0. , 0. , 0. , 1. ]])

        Check that the transformation is inverted.
        >>> transform.is_inverted
        True

        Invert it again to restore it back to its original state.

        >>> _ = transform.invert()
        >>> transform.matrix
        array([[2., 0., 0., 0.],
               [0., 2., 0., 0.],
               [0., 0., 2., 0.],
               [0., 0., 0., 1.]])
        >>> transform.is_inverted
        False
        """
        self.Inverse()
        return self

    def identity(self) -> Transform:  # numpydoc ignore: RT01
        """Set the transformation to the identity transformation.

        This can be used to "reset" the transform.

        Examples
        --------
        Concatenate an arbitrary transformation.

        >>> import pyvista as pv
        >>> transform = pv.Transform().scale(2.0)
        >>> transform.matrix
        array([[2., 0., 0., 0.],
               [0., 2., 0., 0.],
               [0., 0., 2., 0.],
               [0., 0., 0., 1.]])

        Reset the transformation to the identity matrix.

        >>> _ = transform.identity()
        >>> transform.matrix
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])
        """
        self.Identity()
        return self

    @property
    def is_inverted(self) -> bool:  # numpydoc ignore: RT01
        """Get the inverse flag of the transformation.

        This flag is modified whenever :meth:`invert` is called.
        """
        return bool(self.GetInverseFlag())

    def _concatenate_with_translations(
        self,
        transform: TransformLike,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ):
        translate_before, translate_after = self._get_point_translations(
            point=point, multiply_mode=multiply_mode
        )
        if translate_before:
            self.concatenate(translate_before, multiply_mode=multiply_mode)

        self.concatenate(transform, multiply_mode=multiply_mode)

        if translate_after:
            self.concatenate(translate_after, multiply_mode=multiply_mode)

        return self

    def _get_point_translations(
        self, point: VectorLike[float] | None, multiply_mode: Literal['pre', 'post'] | None
    ):
        point = point if point is not None else self.point
        if point is not None:
            point_array = _validation.validate_array3(point, dtype_out=float, name='point')
            translate_away = Transform().translate(-point_array)
            translate_toward = Transform().translate(point_array)
            if multiply_mode == 'post' or self._multiply_mode == 'post':
                return translate_away, translate_toward
            else:
                return translate_toward, translate_away
        return None, None

    @property
    def check_finite(self) -> bool:  # numpydoc ignore: RT01
        """Check that the :attr:`matrix` and :attr:`inverse_matrix` have finite values.

        If ``True``, all transformations are checked to ensure they only contain
        finite values (i.e. no ``NaN`` or ``Inf`` values) and a ``ValueError`` is raised
        otherwise. This is useful to catch cases where the transformation(s) are poorly
        defined and/or are numerically unstable.

        This flag is enabled by default.
        """
        return self._check_finite

    @check_finite.setter
    def check_finite(self, value: bool):  # numpydoc ignore: GL08
        self._check_finite = bool(value)
