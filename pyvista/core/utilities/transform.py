"""Module containing the Transform class."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast
from typing import overload

import numpy as np

import pyvista
from pyvista.core import _validation
from pyvista.core import _vtk_core as _vtk
from pyvista.core.utilities.arrays import array_from_vtkmatrix
from pyvista.core.utilities.arrays import vtkmatrix_from_array
from pyvista.core.utilities.misc import _NoNewAttrMixin
from pyvista.core.utilities.misc import assert_empty_kwargs
from pyvista.core.utilities.transformations import _decomposition_as_homogeneous
from pyvista.core.utilities.transformations import apply_transformation_to_points
from pyvista.core.utilities.transformations import axis_angle_rotation
from pyvista.core.utilities.transformations import decomposition
from pyvista.core.utilities.transformations import reflection

if TYPE_CHECKING:  # pragma: no cover
    from scipy.spatial.transform import Rotation

    from pyvista import DataSet
    from pyvista import MultiBlock
    from pyvista import Prop3D
    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import RotationLike
    from pyvista.core._typing_core import TransformLike
    from pyvista.core._typing_core import VectorLike
    from pyvista.core._typing_core import _DataSetOrMultiBlockType
    from pyvista.core.utilities.transformations import _FiveArrays


class Transform(
    _NoNewAttrMixin,
    _vtk.DisableVtkSnakeCase,
    _vtk.vtkPyVistaOverride,
    _vtk.vtkTransform,
):
    """Describes linear transformations via a 4x4 matrix.

    A :class:`Transform` can be used to describe the full range of linear (also known
    as affine) coordinate transformations in three dimensions, which are internally
    represented as a 4x4 homogeneous transformation matrix.

    The transformation methods (e.g. :meth:`translate`, :meth:`rotate`,
    :meth:`compose`) can operate in either :meth:`pre_multiply` or
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
    trans : TransformLike | Sequence[TransformLike], optional
        Initialize the transform with a transformation or sequence of transformations.
        By default, the transform is initialized as the identity matrix.

    point : VectorLike[float], optional
        Point to use when composing transformations.
        If set, two additional transformations are composed and added to
        the :attr:`matrix_list`:

            - :meth:`translate` to ``point`` before the transformation
            - :meth:`translate` away from ``point`` after the transformation

        By default, this value is ``None``, which means that the scale, rotation, etc.
        transformations are performed about the origin ``(0, 0, 0)``.

    multiply_mode : 'pre' | 'post', optional
        Multiplication mode to use when composing. Set this to ``'pre'`` for
        pre-multiplication or ``'post'`` for post-multiplication.

    See Also
    --------
    pyvista.DataObjectFilters.transform
        Apply a transformation to a mesh.
    pyvista.Prop3D.transform
        Transform an actor.

    Examples
    --------
    Create a transformation and use ``+`` to compose a translation.

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

    Create a transformation and use ``*`` to compose a scaling matrix.

    >>> scale_factor = 2.0
    >>> scaling_T = pv.Transform() * scale_factor
    >>> scaling_T.matrix
    array([[2., 0., 0., 0.],
           [0., 2., 0., 0.],
           [0., 0., 2., 0.],
           [0., 0., 0., 1.]])

    Using ``*`` performs the same concatenation as calling :meth:`scale`.

    >>> np.array_equal(scaling_T.matrix, pv.Transform().scale(scale_factor).matrix)
    True

    Compose the two transformations using ``*``. This will compose with
    post-multiplication such that the transformations are applied in order from left to
    right, i.e. translate first, then scale.

    >>> transform_post = translation_T * scaling_T
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

    Alternatively, compose the transformations by chaining the methods with a
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

    Create a similar transform but use pre-multiplication this time. Compose the
    transformations in the same order as before using :meth:`translate` and :meth:`scale`.

    >>> transform_pre = pv.Transform().pre_multiply()
    >>> _ = transform_pre.translate(position).scale(scale_factor)

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

    >>> mesh_post = pv.Sphere().transform(transform_post, inplace=False)
    >>> mesh_pre = pv.Cone().transform(transform_pre, inplace=False)
    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh_post, color='goldenrod')
    >>> _ = pl.add_mesh(mesh_pre, color='teal')
    >>> _ = pl.add_axes_at_origin()
    >>> pl.show()

    Get the composed inverse transformation matrix of the pre-multiplication case.

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

    >>> mesh_pre_inverted = mesh_pre.transform(inverse_matrix, inplace=False)
    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh_pre_inverted, color='teal')
    >>> _ = pl.add_axes_at_origin()
    >>> pl.show()

    """

    def __init__(
        self: Transform,
        trans: TransformLike | Sequence[TransformLike] | None = None,
        *,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] = 'post',
    ) -> None:
        super().__init__()
        self.multiply_mode = multiply_mode
        self.point = point
        self.check_finite = True
        if trans is not None:
            if isinstance(trans, Sequence):
                if all(isinstance(item, Sequence) for item in trans):
                    # Init from a nested sequence array
                    self.compose(trans)
                else:
                    # Init from sequence of transformations
                    [self.compose(t) for t in trans]
            else:
                self.compose(trans)

        self._decomposition_cache: _FiveArrays | None = None
        self._decomposition_mtime = -1

    def __add__(self: Transform, other: VectorLike[float]) -> Transform:
        """:meth:`translate` this transform using post-multiply semantics."""
        try:
            return self.copy().translate(other, multiply_mode='post')
        except TypeError:
            msg = (
                f"Unsupported operand type(s) for +: '{self.__class__.__name__}' "
                f"and '{type(other).__name__}'\n"
                f'The right-side argument must be a length-3 vector.'
            )
            raise TypeError(msg)
        except ValueError:
            msg = (
                f"Unsupported operand value(s) for +: '{self.__class__.__name__}' "
                f"and '{type(other).__name__}'\n"
                f'The right-side argument must be a length-3 vector.'
            )
            raise ValueError(msg)

    def __radd__(self: Transform, other: VectorLike[float]) -> Transform:
        """:meth:`translate` this transform using pre-multiply semantics."""
        try:
            return self.copy().translate(other, multiply_mode='pre')
        except TypeError:
            msg = (
                f"Unsupported operand type(s) for +: '{type(other).__name__}' "
                f"and '{self.__class__.__name__}'\n"
                f'The left-side argument must be a length-3 vector.'
            )
            raise TypeError(msg)
        except ValueError:
            msg = (
                f"Unsupported operand value(s) for +: '{type(other).__name__}' "
                f"and '{self.__class__.__name__}'\n"
                f'The left-side argument must be a length-3 vector.'
            )
            raise ValueError(msg)

    def __mul__(self: Transform, other: float | VectorLike[float] | TransformLike) -> Transform:
        """:meth:`compose` this transform using post-multiply semantics.

        Use :meth:`scale` for single numbers and length-3 vector inputs, and
        :meth:`compose` otherwise for transform-like inputs.
        """
        copied = self.copy()
        try:
            transform = copied.scale(other, multiply_mode='post')  # type: ignore[arg-type]
        except (ValueError, TypeError):
            try:
                transform = copied.compose(other, multiply_mode='post')
            except TypeError:
                msg = (
                    f"Unsupported operand type(s) for *: '{self.__class__.__name__}' "
                    f"and '{type(other).__name__}'\n"
                    f'The right-side argument must be transform-like.'
                )
                raise TypeError(msg)
            except ValueError:
                msg = (
                    f"Unsupported operand value(s) for *: '{self.__class__.__name__}' "
                    f"and '{type(other).__name__}'\n"
                    f'The right-side argument must be a single number or a length-3 vector '
                    f'or have 3x3 or 4x4 shape.'
                )
                raise ValueError(msg)
        return transform

    def __rmul__(self: Transform, other: float | VectorLike[float]) -> Transform:
        """:meth:`scale` this transform using pre-multiply semantics."""
        try:
            return self.copy().scale(other, multiply_mode='pre')
        except TypeError:
            msg = (
                f"Unsupported operand type(s) for *: '{type(other).__name__}' "
                f"and '{self.__class__.__name__}'\n"
                f'The left-side argument must be a single number or a length-3 vector.'
            )
            raise TypeError(msg)
        except ValueError:
            msg = (
                f"Unsupported operand value(s) for *: '{type(other).__name__}' "
                f"and '{self.__class__.__name__}'\n"
                f'The left-side argument must be a single number or a length-3 vector.'
            )
            raise ValueError(msg)

    def copy(self: Transform) -> Transform:
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

    def __repr__(self: Transform) -> str:
        """Representation of the transform."""

        def _matrix_repr() -> str:
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
        return '\n'.join(lines)

    @property
    def point(self: Transform) -> tuple[float, float, float] | None:  # numpydoc ignore=RT01
        """Point to use when composing some transformations such as scale, rotation, etc.

        If set, two additional transformations are composed and added to
        the :attr:`matrix_list`:

            - :meth:`translate` to ``point`` before the transformation
            - :meth:`translate` away from ``point`` after the transformation

        By default, this value is ``None``, which means that the scale, rotation, etc.
        transformations are performed about the origin ``(0, 0, 0)``.

        """
        return self._point

    @point.setter
    def point(self: Transform, point: VectorLike[float] | None) -> None:
        self._point = (
            None
            if point is None
            else _validation.validate_array3(point, dtype_out=float, to_tuple=True, name='point')
        )

    @property
    def multiply_mode(self: Transform) -> Literal['pre', 'post']:  # numpydoc ignore=RT01
        """Set or get the multiplication mode.

        Set this to ``'pre'`` to set the multiplication mode to :meth:`pre_multiply`.
        Set this to ``'post'`` to set it to :meth:`post_multiply`.

        In pre-multiply mode, any additional transformations (e.g. using
        :meth:`translate`, :meth:`compose`, etc.) will occur *before* any
        transformations represented by the current :attr:`matrix`.
        In post-multiply mode, the additional transformation will occur *after* any
        transformations represented by the current matrix.
        """
        return self._multiply_mode

    @multiply_mode.setter
    def multiply_mode(self: Transform, multiply_mode: Literal['pre', 'post']) -> None:
        _validation.check_contains(
            ['pre', 'post'], must_contain=multiply_mode, name='multiply mode'
        )
        self.pre_multiply() if multiply_mode == 'pre' else self.post_multiply()

    def pre_multiply(self: Transform) -> Transform:  # numpydoc ignore=RT01
        """Set the multiplication mode to pre-multiply.

        In pre-multiply mode, any additional transformations (e.g. using
        :meth:`translate`, :meth:`compose`, etc.) will occur *before* any
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

    def post_multiply(self: Transform) -> Transform:  # numpydoc ignore=RT01
        """Set the multiplication mode to post-multiply.

        In post-multiply mode, any additional transformations (e.g. using
        :meth:`translate`, :meth:`compose`, etc.) will occur *after* any
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
        self: Transform,
        *factor: float | VectorLike[float],
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Compose a scale matrix.

        Create a scale matrix and :meth:`compose` it with the current
        transformation :attr:`matrix` according to pre-multiply or post-multiply
        semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        *factor : float | VectorLike[float]
            Scale factor(s) to use. Use a single number for uniform scaling or
            three numbers for non-uniform scaling. The three factors may be
            passed as a single vector (one arg) or an unpacked vector (three args).

        point : VectorLike[float], optional
            Point to scale from. By default, the object's :attr:`point` is used,
            but this can be overridden.
            If set, two additional transformations are composed and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the scaling
                - :meth:`translate` away from ``point`` after the scaling

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when composing the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        pyvista.DataObjectFilters.scale
            Scale a mesh.

        pyvista.DataObjectFilters.resize
            Resize a mesh.

        scale_factors, has_scale
            Get info about the transform's scale component.

        Examples
        --------
        Compose a scale matrix.

        >>> import pyvista as pv
        >>> transform = pv.Transform().scale(1, 2, 3)
        >>> transform.matrix
        array([[1., 0., 0., 0.],
               [0., 2., 0., 0.],
               [0., 0., 3., 0.],
               [0., 0., 0., 1.]])

        Compose a second scale matrix using ``*``.

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
            factor,  # type: ignore[arg-type]
            broadcast=True,
            dtype_out=float,
            name='scale factor',
        )
        transform = _vtk.vtkTransform()
        transform.Scale(valid_factor)
        return self._compose_with_translations(transform, point=point, multiply_mode=multiply_mode)

    def reflect(
        self: Transform,
        *normal: float | VectorLike[float],
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Compose a reflection matrix.

        Create a reflection matrix and :meth:`compose` it with the current
        transformation :attr:`matrix` according to pre-multiply or post-multiply
        semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        *normal : float | VectorLike[float]
            Normal direction for reflection. May be a single vector (one arg) or
            unpacked vector (three args).

        point : VectorLike[float], optional
            Point to reflect about. By default, the object's :attr:`point` is used,
            but this can be overridden.
            If set, two additional transformations are composed and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the reflection
                - :meth:`translate` away from ``point`` after the reflection

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when composing the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        flip_x, flip_y, flip_z
            Convenience methods for reflecting about the x-, y-, or z-axis.

        pyvista.DataObjectFilters.reflect
            Reflect a mesh.

        reflection, has_reflection
            Get info about the transform's reflection component.

        Examples
        --------
        Compose a reflection matrix.

        >>> import pyvista as pv
        >>> transform = pv.Transform()
        >>> _ = transform.reflect(0, 0, 1)
        >>> transform.matrix
        array([[ 1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0., -1.,  0.],
               [ 0.,  0.,  0.,  1.]])

        Compose a second reflection matrix.

        >>> _ = transform.reflect((1, 0, 0))
        >>> transform.matrix
        array([[-1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0., -1.,  0.],
               [ 0.,  0.,  0.,  1.]])

        """
        valid_normal = _validation.validate_array3(
            normal,  # type: ignore[arg-type]
            dtype_out=float,
            name='reflection normal',
        )
        transform = reflection(valid_normal)
        return self._compose_with_translations(transform, point=point, multiply_mode=multiply_mode)

    def flip_x(
        self: Transform,
        *,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Compose a reflection about the x-axis.

        Create a reflection about the x-axis and :meth:`compose` it
        with the current transformation :attr:`matrix` according to pre-multiply or
        post-multiply semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        point : VectorLike[float], optional
            Point to reflect about. By default, the object's :attr:`point` is used,
            but this can be overridden.
            If set, two additional transformations are composed and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the reflection
                - :meth:`translate` away from ``point`` after the reflection

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when composing the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        reflect, flip_y, flip_z
            Similar reflection methods.

        pyvista.DataObjectFilters.flip_x
            Flip a mesh about the x-axis.

        reflection, has_reflection
            Get info about the transform's reflection component.

        Examples
        --------
        Compose a reflection about the x-axis.

        >>> import pyvista as pv
        >>> transform = pv.Transform()
        >>> _ = transform.flip_x()
        >>> transform.matrix
        array([[-1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  1.]])

        Compose a second reflection, but this time about a point.

        >>> _ = transform.flip_x(point=(4, 5, 6))
        >>> transform.matrix
        array([[1., 0., 0., 8.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])

        """
        return self.reflect((1, 0, 0), point=point, multiply_mode=multiply_mode)

    def flip_y(
        self: Transform,
        *,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Compose a reflection about the y-axis.

        Create a reflection about the y-axis and :meth:`compose` it
        with the current transformation :attr:`matrix` according to pre-multiply or
        post-multiply semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        point : VectorLike[float], optional
            Point to reflect about. By default, the object's :attr:`point` is used,
            but this can be overridden.
            If set, two additional transformations are composed and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the reflection
                - :meth:`translate` away from ``point`` after the reflection

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when composing the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        reflect, flip_x, flip_z
            Similar reflection methods.

        pyvista.DataObjectFilters.flip_y
            Flip a mesh about the y-axis.

        reflection, has_reflection
            Get info about the transform's reflection component.

        Examples
        --------
        Compose a reflection about the y-axis.

        >>> import pyvista as pv
        >>> transform = pv.Transform()
        >>> _ = transform.flip_y()
        >>> transform.matrix
        array([[ 1.,  0.,  0.,  0.],
               [ 0., -1.,  0.,  0.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  1.]])

        Compose a second reflection, but this time about a point.

        >>> _ = transform.flip_y(point=(4, 5, 6))
        >>> transform.matrix
        array([[ 1.,  0.,  0.,  0.],
               [ 0.,  1.,  0., 10.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  1.]])

        """
        return self.reflect((0, 1, 0), point=point, multiply_mode=multiply_mode)

    def flip_z(
        self: Transform,
        *,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Compose a reflection about the z-axis.

        Create a reflection about the z-axis and :meth:`compose` it
        with the current transformation :attr:`matrix` according to pre-multiply or
        post-multiply semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        point : VectorLike[float], optional
            Point to reflect about. By default, the object's :attr:`point` is used,
            but this can be overridden.
            If set, two additional transformations are composed and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the reflection
                - :meth:`translate` away from ``point`` after the reflection

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when composing the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        reflect, flip_x, flip_y
            Similar reflection methods.

        pyvista.DataObjectFilters.flip_z
            Flip a mesh about the z-axis.

        reflection, has_reflection
            Get info about the transform's reflection component.

        Examples
        --------
        Compose a reflection about the z-axis.

        >>> import pyvista as pv
        >>> transform = pv.Transform()
        >>> _ = transform.flip_z()
        >>> transform.matrix
        array([[ 1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0., -1.,  0.],
               [ 0.,  0.,  0.,  1.]])

        Compose a second reflection, but this time about a point.

        >>> _ = transform.flip_z(point=(4, 5, 6))
        >>> transform.matrix
        array([[ 1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  1., 12.],
               [ 0.,  0.,  0.,  1.]])

        """
        return self.reflect((0, 0, 1), point=point, multiply_mode=multiply_mode)

    def translate(
        self: Transform,
        *vector: float | VectorLike[float],
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Compose a translation matrix.

        Create a translation matrix and :meth:`compose` it with the current
        transformation :attr:`matrix` according to pre-multiply or post-multiply
        semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        *vector : float | VectorLike[float]
            Vector to use for translation. May be a single vector (one arg) or
            unpacked vector (three args).

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when composing the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        pyvista.DataObjectFilters.translate
            Translate a mesh.

        translation, has_translation
            Get info about the transform's translation component.

        Examples
        --------
        Compose a translation matrix.

        >>> import pyvista as pv
        >>> transform = pv.Transform().translate(1, 2, 3)
        >>> transform.matrix
        array([[1., 0., 0., 1.],
               [0., 1., 0., 2.],
               [0., 0., 1., 3.],
               [0., 0., 0., 1.]])

        Compose a second translation matrix using ``+``.

        >>> transform = transform + (1, 1, 1)
        >>> transform.matrix
        array([[1., 0., 0., 2.],
               [0., 1., 0., 3.],
               [0., 0., 1., 4.],
               [0., 0., 0., 1.]])

        """
        valid_vector = _validation.validate_array3(
            vector,  # type: ignore[arg-type]
            dtype_out=float,
            name='translation vector',
        )
        transform = _vtk.vtkTransform()
        transform.Translate(valid_vector)
        return self.compose(transform, multiply_mode=multiply_mode)

    def rotate(
        self: Transform,
        rotation: RotationLike,
        *,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Compose a rotation matrix.

        Create a rotation matrix and :meth:`compose` it with the current
        transformation :attr:`matrix` according to pre-multiply or post-multiply
        semantics. The rotation may be right-handed or left-handed.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        rotation : RotationLike
            3x3 rotation matrix or a SciPy ``Rotation`` object.

        point : VectorLike[float], optional
            Point to rotate about. By default, the object's :attr:`point` is used,
            but this can be overridden.
            If set, two additional transformations are composed and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the rotation
                - :meth:`translate` away from ``point`` after the rotation

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when composing the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        rotate_x, rotate_y, rotate_z, rotate_vector
            Similar rotation methods.
        rotation_matrix, rotation_axis_angle, as_rotation, has_rotation
            Get this transform's rotation component.
        pyvista.DataObjectFilters.rotate
            Rotate a mesh.

        Examples
        --------
        Compose a rotation matrix. In this case the rotation rotates about the
        z-axis by 90 degrees.

        >>> import pyvista as pv
        >>> rotation_z_90 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        >>> transform = pv.Transform().rotate(rotation_z_90)
        >>> transform.matrix
        array([[ 0., -1.,  0.,  0.],
               [ 1.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  1.]])

        Compose a second rotation matrix. In this case we use the same rotation as
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

        >>> transform = pv.Transform().rotate(rotation_z_90, point=(1, 2, 3))
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
        valid_rotation = _validation.validate_rotation(rotation)
        return self._compose_with_translations(
            valid_rotation, point=point, multiply_mode=multiply_mode
        )

    def rotate_x(
        self: Transform,
        angle: float,
        *,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Compose a rotation about the x-axis.

        Create a matrix for rotation about the x-axis and :meth:`compose`
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
            If set, two additional transformations are composed and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the rotation
                - :meth:`translate` away from ``point`` after the rotation

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when composing the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        rotate_y, rotate_z, rotate_vector, rotate
            Similar rotation methods.
        rotation_matrix, rotation_axis_angle, as_rotation, has_rotation
            Get this transform's rotation component.
        pyvista.DataObjectFilters.rotate_x
            Rotate a mesh about the x-axis.

        Examples
        --------
        Compose a rotation about the x-axis.

        >>> import pyvista as pv
        >>> transform = pv.Transform().rotate_x(90)
        >>> transform.matrix
        array([[ 1.,  0.,  0.,  0.],
               [ 0.,  0., -1.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  0.,  1.]])

        Compose a second rotation about the x-axis.

        >>> _ = transform.rotate_x(45)

        The result is a matrix that rotates about the x-axis by 135 degrees.

        >>> transform.matrix
        array([[ 1.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        , -0.70710678, -0.70710678,  0.        ],
               [ 0.        ,  0.70710678, -0.70710678,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  1.        ]])

        """
        transform = axis_angle_rotation((1, 0, 0), angle, deg=True)
        return self._compose_with_translations(transform, point=point, multiply_mode=multiply_mode)

    def rotate_y(
        self: Transform,
        angle: float,
        *,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Compose a rotation about the y-axis.

        Create a matrix for rotation about the y-axis and :meth:`compose`
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
            If set, two additional transformations are composed and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the rotation
                - :meth:`translate` away from ``point`` after the rotation

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when composing the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        rotate_x, rotate_z, rotate_vector, rotate
            Similar rotation methods.
        rotation_matrix, rotation_axis_angle, as_rotation, has_rotation
            Get this transform's rotation component.
        pyvista.DataObjectFilters.rotate_y
            Rotate a mesh about the y-axis.

        Examples
        --------
        Compose a rotation about the y-axis.

        >>> import pyvista as pv
        >>> transform = pv.Transform().rotate_y(90)
        >>> transform.matrix
        array([[ 0.,  0.,  1.,  0.],
               [ 0.,  1.,  0.,  0.],
               [-1.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  1.]])

        Compose a second rotation about the y-axis.

        >>> _ = transform.rotate_y(45)

        The result is a matrix that rotates about the y-axis by 135 degrees.

        >>> transform.matrix
        array([[-0.70710678,  0.        ,  0.70710678,  0.        ],
               [ 0.        ,  1.        ,  0.        ,  0.        ],
               [-0.70710678,  0.        , -0.70710678,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  1.        ]])

        """
        transform = axis_angle_rotation((0, 1, 0), angle, deg=True)
        return self._compose_with_translations(transform, point=point, multiply_mode=multiply_mode)

    def rotate_z(
        self: Transform,
        angle: float,
        *,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Compose a rotation about the z-axis.

        Create a matrix for rotation about the z-axis and :meth:`compose`
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
            If set, two additional transformations are composed and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the rotation
                - :meth:`translate` away from ``point`` after the rotation

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when composing the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        rotate_x, rotate_y, rotate_vector, rotate
            Similar rotation methods.
        rotation_matrix, rotation_axis_angle, as_rotation, has_rotation
            Get this transform's rotation component.
        pyvista.DataObjectFilters.rotate_z
            Rotate a mesh about the z-axis.

        Examples
        --------
        Compose a rotation about the z-axis.

        >>> import pyvista as pv
        >>> transform = pv.Transform().rotate_z(90)
        >>> transform.matrix
        array([[ 0., -1.,  0.,  0.],
               [ 1.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  1.]])

        Compose a second rotation about the z-axis.

        >>> _ = transform.rotate_z(45)

        The result is a matrix that rotates about the z-axis by 135 degrees.

        >>> transform.matrix
        array([[-0.70710678, -0.70710678,  0.        ,  0.        ],
               [ 0.70710678, -0.70710678,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  1.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  1.        ]])

        """
        transform = axis_angle_rotation((0, 0, 1), angle, deg=True)
        return self._compose_with_translations(transform, point=point, multiply_mode=multiply_mode)

    def rotate_vector(
        self: Transform,
        vector: VectorLike[float],
        angle: float,
        *,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Compose a rotation about a vector.

        Create a matrix for rotation about the vector and :meth:`compose`
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
            If set, two additional transformations are composed and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the rotation
                - :meth:`translate` away from ``point`` after the rotation

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when composing the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        rotate_x, rotate_y, rotate_z, rotate
            Similar rotation methods.
        rotation_matrix, rotation_axis_angle, as_rotation, has_rotation
            Get this transform's rotation component.
        pyvista.DataObjectFilters.rotate_vector
            Rotate a mesh about a vector.

        Examples
        --------
        Compose a rotation of 30 degrees about the ``(1, 1, 1)`` axis.

        >>> import pyvista as pv
        >>> transform = pv.Transform().rotate_vector((1, 1, 1), 30)
        >>> transform.matrix
        array([[ 0.9106836 , -0.24401694,  0.33333333,  0.        ],
               [ 0.33333333,  0.9106836 , -0.24401694,  0.        ],
               [-0.24401694,  0.33333333,  0.9106836 ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  1.        ]])

        Compose a second rotation of 45 degrees about the ``(1, 2, 3)`` axis.

        >>> _ = transform.rotate_vector((1, 2, 3), 45)
        >>> transform.matrix
        array([[ 0.38042304, -0.50894634,  0.77217351,  0.        ],
               [ 0.83349512,  0.55045308, -0.04782562,  0.        ],
               [-0.40070461,  0.66179682,  0.63360933,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  1.        ]])

        """
        transform = axis_angle_rotation(vector, angle, deg=True)
        return self._compose_with_translations(transform, point=point, multiply_mode=multiply_mode)

    def compose(
        self: Transform,
        transform: TransformLike,
        *,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
        """Compose a transformation matrix.

        Create a 4x4 matrix from any transform-like input and compose it with the
        current transformation :attr:`matrix` according to pre-multiply or post-multiply
        semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        transform : TransformLike
            Any transform-like input such as a 3x3 or 4x4 array or matrix.

        point : VectorLike[float], optional
            Point to transform about. By default, the object's :attr:`point` is used,
            but this can be overridden.
            If set, two additional transformations are composed and added to
            the :attr:`matrix_list`:

                - :meth:`translate` to ``point`` before the transformation
                - :meth:`translate` away from ``point`` after the transformation

            .. versionadded:: 0.47

        multiply_mode : 'pre' | 'post', optional
            Multiplication mode to use when composing the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        See Also
        --------
        decompose

        Examples
        --------
        Define an arbitrary 4x4 affine transformation matrix and compose it.

        >>> import pyvista as pv
        >>> array = [
        ...     [0.707, -0.707, 0, 0],
        ...     [0.707, 0.707, 0, 0],
        ...     [0, 0, 1, 1.5],
        ...     [0, 0, 0, 2],
        ... ]
        >>> transform = pv.Transform().compose(array)
        >>> transform.matrix
        array([[ 0.707, -0.707,  0.   ,  0.   ],
               [ 0.707,  0.707,  0.   ,  0.   ],
               [ 0.   ,  0.   ,  1.   ,  1.5  ],
               [ 0.   ,  0.   ,  0.   ,  2.   ]])

        Define a second transformation and use ``*`` to compose it.

        >>> array = [[1, 0, 0], [0, 0, -1], [0, -1, 0]]
        >>> transform = transform * array
        >>> transform.matrix
        array([[ 0.707, -0.707,  0.   ,  0.   ],
               [ 0.   ,  0.   , -1.   , -1.5  ],
               [-0.707, -0.707,  0.   ,  0.   ],
               [ 0.   ,  0.   ,  0.   ,  2.   ]])

        Compose the transform about a point. Check the :attr:`matrix_list` to see that a
        translation is added before and after the transform.

        >>> transform = pv.Transform().compose(transform, point=(1, 2, 3))
        >>> transform.matrix_list  # doctest: +NORMALIZE_WHITESPACE
        [array([[ 1.,  0.,  0., -1.],
                [ 0.,  1.,  0., -2.],
                [ 0.,  0.,  1., -3.],
                [ 0.,  0.,  0.,  1.]]),
         array([[ 0.707, -0.707,  0.   ,  0.   ],
                [ 0.   ,  0.   , -1.   , -1.5  ],
                [-0.707, -0.707,  0.   ,  0.   ],
                [ 0.   ,  0.   ,  0.   ,  2.   ]]),
         array([[1., 0., 0., 1.],
                [0., 1., 0., 2.],
                [0., 0., 1., 3.],
                [0., 0., 0., 1.]])]


        """
        return self._compose_with_translations(transform, point=point, multiply_mode=multiply_mode)

    def _compose(
        self: Transform,
        transform: TransformLike,
        *,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:  # numpydoc ignore=RT01
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
    def matrix(self: Transform) -> NumpyArray[float]:
        """Return or set the current transformation matrix.

        Notes
        -----
        This matrix is a single 4x4 matrix computed from composing all
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
    def matrix(self: Transform, trans: TransformLike) -> None:
        array = _validation.validate_transform4x4(
            trans, must_be_finite=self.check_finite, name='matrix'
        )
        self.SetMatrix(vtkmatrix_from_array(array))

    @property
    def inverse_matrix(self: Transform) -> NumpyArray[float]:
        """Return the inverse of the current transformation :attr:`~Transform.matrix`.

        Notes
        -----
        This matrix is a single 4x4 matrix computed from composing the inverse of
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
        array = array_from_vtkmatrix(self.GetInverse().GetMatrix())  # type: ignore[attr-defined]
        if self.check_finite:
            _validation.check_finite(array, name='matrix')
        return array

    @property
    def matrix_list(self: Transform) -> list[NumpyArray[float]]:
        """Return a list of all current transformation matrices.

        Notes
        -----
        The list comprises all 4x4 transformation matrices. Use :attr:`matrix` instead
        to get the composed result as a single 4x4 matrix.

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
    def inverse_matrix_list(self: Transform) -> list[NumpyArray[float]]:
        """Return a list of all inverse transformations applied by this :class:`Transform`.

        Notes
        -----
        The list comprises all 4x4 inverse transformation matrices. Use
        :attr:`inverse_matrix` instead to get the composed result as a single
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
            array_from_vtkmatrix(self.GetConcatenatedTransform(i).GetInverse().GetMatrix())  # type: ignore[attr-defined]
            for i in range(self.n_transformations)
        ]

    @property
    def n_transformations(self: Transform) -> int:  # numpydoc ignore: RT01
        """Return the current number of composed transformations."""
        return self.GetNumberOfConcatenatedTransforms()

    @overload
    def apply(
        self: Transform,
        obj: VectorLike[float] | MatrixLike[float],
        /,
        mode: Literal['points', 'vectors'] | None = ...,
        *,
        inverse: bool = ...,
        copy: bool = ...,
    ) -> NumpyArray[float]: ...
    @overload
    def apply(
        self: Transform,
        obj: _DataSetOrMultiBlockType,
        /,
        mode: Literal['active_vectors', 'all_vectors'] = ...,
        *,
        inverse: bool = ...,
        copy: bool = ...,
    ) -> _DataSetOrMultiBlockType: ...
    @overload
    def apply(
        self: Transform,
        obj: Prop3D,
        /,
        mode: Literal['replace', 'pre-multiply', 'post-multiply'] = ...,
        *,
        inverse: bool = ...,
        copy: bool = ...,
    ) -> Prop3D: ...
    def apply(
        self: Transform,
        obj: VectorLike[float] | MatrixLike[float] | DataSet | MultiBlock | Prop3D,
        /,
        mode: Literal[
            'points',
            'vectors',
            'active_vectors',
            'all_vectors',
            'replace',
            'pre-multiply',
            'post-multiply',
        ]
        | None = None,
        *,
        inverse: bool = False,
        copy: bool = True,
    ):
        """Apply the current transformation :attr:`~Transform.matrix` to points, vectors, a dataset, or actor.

        .. note::

            Points with integer values are cast to a float type before the
            transformation is applied. A similar casting is also performed when
            transforming datasets. See also the notes at
            :func:`~pyvista.DataObjectFilters.transform`
            which is used by this filter under the hood.

        Parameters
        ----------
        obj : VectorLike[float] | MatrixLike[float] | DataSet | MultiBlock | Prop3D
            Object to apply the transformation to.

        mode : str, optional
            Define how to apply the transformation. Different modes may be used depending
            on the input type.

            #.  For array inputs:

                - ``'points'`` transforms point arrays.
                - ``'vectors'`` transforms vector arrays. The translation component of
                  the transformation is removed for vectors.

                By default, ``'points'`` mode is used for array inputs.

            #.  For dataset inputs:

                The dataset's points are always transformed, and its vectors are
                transformed based on the mode:

                - ``'active_vectors'`` transforms active normals and active vectors
                  arrays only.
                - ``'all_vectors'`` transforms `all` input vectors, i.e. all arrays
                  with three components. This mode is equivalent to setting
                  ``transform_all_input_vectors=True``
                  with :meth:`pyvista.DataObjectFilters.transform`.

                By default, only ``'active_vectors'`` are transformed.

            #.  For actor inputs:

                - ``'pre-multiply'`` pre-multiplies this transform with the actor's
                  :attr:`~pyvista.Prop3D.user_matrix`.
                - ``'post-multiply'`` post-multiplies this transform with the actor's
                  user-matrix.
                - ``'replace'`` replaces the actor's user-matrix with this transform's
                  :attr:`matrix`.

                By default, ``'post-multiply'`` mode is used for actors.

        inverse : bool, default: False
            Apply the transformation using the :attr:`inverse_matrix` instead of the
            :attr:`matrix`.

        copy : bool, default: True
            Return a copy of the input with the transformation applied. Set this to
            ``False`` to transform the input directly and return it. Setting this to
            ``False`` only applies to NumPy float arrays, datasets, and actors; a copy
            is always returned for tuple and list inputs or arrays with integers.

        Returns
        -------
        np.ndarray | DataSet | MultiBlock | Prop3D
            Transformed array, dataset, or actor.

        See Also
        --------
        apply_to_points
            Equivalent to ``apply(obj, 'points')`` for point-array inputs.
        apply_to_vectors
            Equivalent to ``apply(obj, 'vectors')`` for vector-array inputs.
        apply_to_dataset
            Equivalent to ``apply(obj, mode)`` for dataset inputs where ``mode`` may be
            ``'active_vectors'`` or ``'all_vectors'``.
        apply_to_actor
            Equivalent to ``apply(obj, mode)`` for actor inputs where ``mode`` may be
            ``'pre-multiply'``, ``'post-multiply'``, or ``'replace'``.
        pyvista.DataObjectFilters.transform
            Transform a dataset.
        pyvista.Prop3D.transform
            Transform an actor.

        Examples
        --------
        Apply a transformation to a point.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> point = (1.0, 2.0, 3.0)
        >>> transform = pv.Transform().scale(2).translate((0.0, 0.0, 1.0))
        >>> transformed = transform.apply(point)
        >>> transformed
        array([2., 4., 7.])

        Apply a transformation to a points array.

        >>> array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> transformed = transform.apply(array)
        >>> transformed
        array([[ 2.,  4.,  7.],
               [ 8., 10., 13.]])

        Apply a transformation to a vector array. Use the same array as before but
        use the ``'vectors'`` mode. Note how the translation component is not applied
        to vectors.

        >>> transformed = transform.apply(array, 'vectors')
        >>> transformed
        array([[ 2.,  4.,  6.],
               [ 8., 10., 12.]])

        Apply a transformation to a dataset.

        >>> dataset = pv.PolyData(array)
        >>> transformed = transform.apply(dataset)
        >>> transformed.points
        pyvista_ndarray([[ 2.,  4.,  6.],
                         [ 8., 10., 12.]])

        Apply the inverse.

        >>> transformed = transform.apply(dataset, inverse=True)
        >>> transformed.points
        pyvista_ndarray([[0.5, 1. , 1. ],
                         [2. , 2.5, 2.5]])

        Apply a transformation to an actor.

        >>> actor = pv.Actor()
        >>> transformed_actor = transform.apply(actor)
        >>> transformed_actor.user_matrix
        array([[2., 0., 0., 0.],
               [0., 2., 0., 0.],
               [0., 0., 2., 0.],
               [0., 0., 0., 1.]])

        """

        def _check_mode(kind: str, mode_: str | None, allowed_modes: list[str | None]) -> None:
            if mode_ not in allowed_modes:
                msg = (
                    f"Transformation mode '{mode_}' is not supported for {kind}. "
                    'Mode must be one of'
                    f'\n{allowed_modes}'
                )
                raise ValueError(msg)

        _validation.check_contains(
            [
                'points',
                'vectors',
                'active_vectors',
                'all_vectors',
                'replace',
                'pre-multiply',
                'post-multiply',
                None,
            ],
            must_contain=mode,
            name='mode',
        )
        _validation.check_instance(
            obj,
            (
                np.ndarray,
                Sequence,
                pyvista.DataSet,
                pyvista.MultiBlock,
                pyvista.Prop3D,
            ),
        )

        inplace = not copy
        # Transform dataset
        if isinstance(obj, (pyvista.DataSet, pyvista.MultiBlock)):
            allowed = ['active_vectors', 'all_vectors', None]
            _check_mode('datasets', mode, allowed)
            if mode in ['active_vectors', None]:
                mode = None

            return obj.transform(
                self.copy().invert() if inverse else self,
                inplace=inplace,
                transform_all_input_vectors=bool(mode),
            )

        matrix = self.inverse_matrix if inverse else self.matrix

        # Transform actor
        if isinstance(obj, pyvista.Prop3D):
            allowed = ['replace', 'pre-multiply', 'post-multiply', None]
            _check_mode('actors', mode, allowed)
            if mode in ['post-multiply', None]:
                return obj.transform(matrix, 'post', inplace=inplace)
            elif mode == 'pre-multiply':
                return obj.transform(matrix, 'pre', inplace=inplace)
            else:
                actor = obj.copy() if copy else obj
                actor.user_matrix = matrix
                return actor

        # Transform array
        allowed = ['points', 'vectors', None]
        _check_mode('arrays', mode, allowed)

        if mode == 'vectors':
            # Remove translation
            matrix[:3, 3] = 0

        # Validate array - make sure we have floats
        array: NumpyArray[float] = _validation.validate_array(obj, must_have_shape=[(3,), (-1, 3)])
        array = array if np.issubdtype(array.dtype, np.floating) else array.astype(float)

        # Transform a 1D array
        out: NumpyArray[float] | None
        if array.shape == (3,):
            out = (matrix @ (*array, 1))[:3]
            if inplace:
                array[:] = out
                out = array
            return out

        # Transform a 2D array
        out = apply_transformation_to_points(matrix, array, inplace=inplace)
        return out if out is not None else array

    def apply_to_points(
        self,
        points: VectorLike[float] | MatrixLike[float],
        /,
        *,
        inverse: bool = False,
        copy: bool = True,
    ) -> NumpyArray[float]:
        """Apply the current transformation :attr:`~Transform.matrix` to a point or points.

        This is equivalent to ``apply(points, 'points')``. See :meth:`apply` for
        details and examples.

        Parameters
        ----------
        points : VectorLike[float] | MatrixLike[float]
            Single point or ``Nx3`` points array to apply the transformation to.

        inverse : bool, default: False
            Apply the transformation using the :attr:`inverse_matrix` instead of the
            :attr:`matrix`.

        copy : bool, default: True
            Return a copy of the input with the transformation applied. Set this to
            ``False`` to transform the input directly and return it. Only applies to
            NumPy arrays. A copy is always returned for tuple and list
            inputs or point arrays with integers.

        Returns
        -------
        np.ndarray
            Transformed points.

        See Also
        --------
        apply
            Apply this transformation to any input.
        apply_to_vectors
            Apply this transformation to vectors.
        apply_to_dataset
            Apply this transformation to a dataset.
        apply_to_actor
            Apply this transformation to an actor.

        """
        return self.apply(points, 'points', inverse=inverse, copy=copy)

    def apply_to_vectors(
        self,
        vectors: VectorLike[float] | MatrixLike[float],
        /,
        *,
        inverse: bool = False,
        copy: bool = True,
    ) -> NumpyArray[float]:
        """Apply the current transformation :attr:`~Transform.matrix` to a vector or vectors.

        This is equivalent to ``apply(vectors, 'vectors')``. See :meth:`apply` for
        details and examples.

        Parameters
        ----------
        vectors : VectorLike[float] | MatrixLike[float]
            Single vector or ``Nx3`` vectors array to apply the transformation to.

        inverse : bool, default: False
            Apply the transformation using the :attr:`inverse_matrix` instead of the
            :attr:`matrix`.

        copy : bool, default: True
            Return a copy of the input with the transformation applied. Set this to
            ``False`` to transform the input directly and return it. Only applies to
            NumPy arrays. A copy is always returned for tuple and list
            inputs or point arrays with integers.

        Returns
        -------
        np.ndarray
            Transformed vectors.

        See Also
        --------
        apply
            Apply this transformation to any input.
        apply_to_points
            Apply this transformation to points.
        apply_to_dataset
            Apply this transformation to a dataset.
        apply_to_actor
            Apply this transformation to an actor.

        """
        return self.apply(vectors, 'vectors', inverse=inverse, copy=copy)

    def apply_to_dataset(
        self,
        dataset: _DataSetOrMultiBlockType,
        /,
        mode: Literal['active_vectors', 'all_vectors'] = 'active_vectors',
        *,
        copy: bool = True,
        inverse: bool = False,
    ) -> _DataSetOrMultiBlockType:
        """Apply the current transformation :attr:`~Transform.matrix` to a dataset.

        This is equivalent to ``apply(dataset, mode)``. See :meth:`apply` for details
        and examples.

        Parameters
        ----------
        dataset : DataSet | MultiBlock
            Object to apply the transformation to.

        mode : 'active_vectors' | 'all_vectors', default: 'active_vectors'
            Mode for transforming the dataset's vectors:

            - ``'active_vectors'`` transforms active normals and active vectors arrays
              only.
            - ``'all_vectors'`` transforms `all` input vectors, i.e. all arrays with
              three components. This mode is equivalent to setting
              ``transform_all_input_vectors=True``
              with :meth:`pyvista.DataObjectFilters.transform`.

        inverse : bool, default: False
            Apply the transformation using the :attr:`inverse_matrix` instead of the
            :attr:`matrix`.

        copy : bool, default: True
            Return a copy of the input with the transformation applied. Set this to
            ``False`` to transform the input directly and return it.

        Returns
        -------
        DataSet | MultiBlock
            Transformed dataset.

        See Also
        --------
        apply
            Apply this transformation to any input.
        apply_to_points
            Apply this transformation to points.
        apply_to_vectors
            Apply this transformation to vectors.
        apply_to_actor
            Apply this transformation to an actor.
        pyvista.DataObjectFilters.transform
            Transform a dataset.

        """
        return self.apply(dataset, mode, inverse=inverse, copy=copy)

    def apply_to_actor(
        self,
        actor: Prop3D,
        /,
        mode: Literal['pre-multiply', 'post-multiply', 'replace'] = 'post-multiply',
        *,
        copy: bool = True,
        inverse: bool = False,
    ) -> Prop3D:
        """Apply the current transformation :attr:`~Transform.matrix` to an actor.

        This is equivalent to ``apply(actor, mode)``. See :meth:`apply` for details and
        examples.

        Parameters
        ----------
        actor : Prop3D
            Actor to apply the transformation to.

        mode : 'pre-multiply', 'post-multiply', 'replace', default: 'post-multiply'
            Mode for transforming the actor:

            - ``'pre-multiply'`` pre-multiplies this transform with the actor's
              :attr:`~pyvista.Prop3D.user_matrix`.
            - ``'post-multiply'``  post-multiplies this transform with the actor's
              user-matrix.
            - ``'replace'`` replaces the actor's user-matrix with this transform's
              :attr:`matrix`.

            By default, ``'post-multiply'`` mode is used.

        inverse : bool, default: False
            Apply the transformation using the :attr:`inverse_matrix` instead of the
            :attr:`matrix`.

        copy : bool, default: True
            Return a copy of the input with the transformation applied. Set this to
            ``False`` to transform the input directly and return it.

        Returns
        -------
        Prop3D
            Transformed actor.

        See Also
        --------
        apply
            Apply this transformation to any input.
        apply_to_points
            Apply this transformation to points.
        apply_to_vectors
            Apply this transformation to vectors.
        apply_to_dataset
            Apply this transformation to a dataset.
        pyvista.Prop3D.transform
            Transform an actor.

        """
        return self.apply(actor, mode, inverse=inverse, copy=copy)

    def decompose(self: Transform, *, homogeneous: bool = False) -> _FiveArrays:
        """Decompose the current transformation into its components.

        Decompose the :attr:`matrix` ``M`` into

        - translation ``T``
        - rotation ``R``
        - reflection ``N``
        - scaling ``S``
        - shearing ``K``

        such that, when represented as 4x4 matrices, ``M = TRNSK``. The decomposition is
        unique and is computed with polar matrix decomposition.

        By default, compact representations of the transformations are returned (e.g. as a
        3-element vector or a 3x3 matrix). Optionally, 4x4 matrices may be returned instead.

        .. note::

            - The rotation is orthonormal and right-handed with positive determinant.
            - The scaling factors are positive.
            - The reflection is either ``1`` (no reflection) or ``-1`` (has reflection)
              and can be used like a scaling factor.

        Parameters
        ----------
        homogeneous : bool, default: False
            If ``True``, return the components (translation, rotation, etc.) as 4x4
            homogeneous matrices. By default, reflection is a scalar, translation and
            scaling are length-3 vectors, and rotation and shear are 3x3 matrices.

        Returns
        -------
        numpy.ndarray
            Translation component ``T``. Returned as a 3-element vector (or a 4x4
            translation matrix if ``homogeneous`` is ``True``).

        numpy.ndarray
            Rotation component ``R``. Returned as a 3x3 orthonormal rotation matrix of row
            vectors (or a 4x4 rotation matrix if ``homogeneous`` is ``True``).

        numpy.ndarray
            Reflection component ``N``. Returned as a NumPy scalar (or a 4x4 reflection
            matrix if ``homogeneous`` is ``True``).

        numpy.ndarray
            Scaling component ``S``. Returned as a 3-element vector (or a 4x4 scaling matrix
            if ``homogeneous`` is ``True``).

        numpy.ndarray
            Shear component ``K``. Returned as a 3x3 matrix with ones on the diagonal and
            shear values in the off-diagonals (or as a 4x4 shearing matrix if ``homogeneous``
            is ``True``).

        See Also
        --------
        compose
            Compose a transformation.

        translation, has_translation
            Get info about this transform's translation component.

        rotation_matrix, rotation_axis_angle, as_rotation, has_rotation
            Get info about this transform's rotation component.

        reflection, has_reflection
            Get info about this transform's reflection component.

        scale_factors, has_scale
            Get info about this transform's scale component.

        shear_matrix, has_shear
            Get info about this transform's shear component.

        Examples
        --------
        Create a transform by composing scaling, rotation, and translation
        matrices.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> transform = pv.Transform()
        >>> _ = transform.scale(1, 2, 3)
        >>> _ = transform.rotate_z(90)
        >>> _ = transform.translate(4, 5, 6)
        >>> transform
        Transform (...)
          Num Transformations: 3
          Matrix:  [[ 0., -2.,  0.,  4.],
                    [ 1.,  0.,  0.,  5.],
                    [ 0.,  0.,  3.,  6.],
                    [ 0.,  0.,  0.,  1.]]

        Decompose the matrix.

        >>> T, R, N, S, K = transform.decompose()

        Since the input has no shear this component is the identity matrix.
        Similarly, there are no reflections so its value is ``1``. All other components
        are recovered perfectly and match the input.

        >>> K  # shear
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])

        >>> S  # scale
        array([1., 2., 3.])

        >>> N  # reflection
        array(1.)

        >>> R  # rotation
        array([[ 0., -1.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])

        >>> T  # translation
        array([4., 5., 6.])

        Compose a shear component using pre-multiplication so that shearing is
        the first transformation.

        >>> shear = np.eye(4)
        >>> shear[0, 1] = 0.1  # xy shear
        >>> _ = transform.compose(shear, multiply_mode='pre')

        Repeat the decomposition and show its components. Note how the decomposed shear
        does not perfectly match the input shear matrix values. The values of the
        scaling and rotation components are also affected and do not exactly match the
        input. This is expected, because the shear can be partially factored as a
        combination of rotation and scaling.

        >>> T, R, N, S, K = transform.decompose()

        >>> K  # shear
        array([[1.        , 0.03333333, 0.        ],
               [0.01663894, 1.        , 0.        ],
               [0.        , 0.        , 1.        ]])

        >>> S  # scale
        array([0.99944491, 2.0022213 , 3.        ])

        >>> N  # reflection
        array(1.)

        >>> R  # rotation
        array([[ 0.03331483, -0.99944491,  0.        ],
               [ 0.99944491,  0.03331483,  0.        ],
               [ 0.        ,  0.        ,  1.        ]])

        >>> T  # translation
        array([4., 5., 6.])

        Although the values may not match the input exactly, the decomposition is
        nevertheless valid and can be used to re-compose the original transformation.

        >>> T, R, N, S, K = transform.decompose(homogeneous=True)
        >>> T @ R @ N @ S @ K
        array([[-5.76153045e-17, -2.00000000e+00,  0.00000000e+00,
                 4.00000000e+00],
               [ 1.00000000e+00,  1.00000000e-01,  0.00000000e+00,
                 5.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  3.00000000e+00,
                 6.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 1.00000000e+00]])

        Alternatively, re-compose the transformation as a new
        :class:`~pyvista.Transform` with pre-multiplication.

        >>> recomposed = pv.Transform([T, R, N, S, K], multiply_mode='pre')
        >>> np.allclose(recomposed.matrix, transform.matrix)
        True

        Compose a reflection and decompose the transform again.

        >>> _ = transform.flip_x()
        >>> T, R, N, S, K = transform.decompose()

        The reflection component is now ``-1``.

        >>> N  # reflection
        array(-1.)

        The decomposition may be simplified to a ``TRSK`` decomposition by combining
        the reflection component with either the rotation or the scaling term.

        Multiplying the reflection with the rotation will make it a left-handed rotation
        with negative determinant:

        >>> R = R * N
        >>> np.linalg.det(R) < 0
        np.True_

        Alternatively, keep the rotation right-handed but make the scaling factors negative:

        >>> S = S * N
        >>> S  # scale
        array([-0.99944491, -2.0022213 , -3.        ])

        """
        if (current_mtime := self.GetMTime()) != self._decomposition_mtime:
            # Recompute and cache
            self._decomposition_cache = decomposition(self.matrix, homogeneous=False)
            self._decomposition_mtime = current_mtime

        cache = cast('_FiveArrays', self._decomposition_cache)
        if homogeneous:
            return _decomposition_as_homogeneous(*cache)
        return cache

    def invert(self: Transform) -> Transform:  # numpydoc ignore: RT01
        """Invert the current transformation.

        The current transformation :attr:`matrix` (including all matrices in the
        :attr:`matrix_list`) is inverted every time :meth:`invert` is called.

        Use :attr:`is_inverted` to check if the transformations are currently inverted.

        Examples
        --------
        Compose an arbitrary transformation.

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

    def identity(self: Transform) -> Transform:  # numpydoc ignore: RT01
        """Set the transformation to the identity transformation.

        This can be used to "reset" the transform.

        Examples
        --------
        Compose an arbitrary transformation.

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
    def is_inverted(self: Transform) -> bool:  # numpydoc ignore: RT01
        """Get the inverse flag of the transformation.

        This flag is modified whenever :meth:`invert` is called.
        """
        return bool(self.GetInverseFlag())

    def _compose_with_translations(
        self: Transform,
        transform: TransformLike,
        point: VectorLike[float] | None = None,
        multiply_mode: Literal['pre', 'post'] | None = None,
    ) -> Transform:
        translate_before, translate_after = self._get_point_translations(
            point=point, multiply_mode=multiply_mode
        )
        if translate_before:
            self._compose(translate_before, multiply_mode=multiply_mode)

        self._compose(transform, multiply_mode=multiply_mode)

        if translate_after:
            self._compose(translate_after, multiply_mode=multiply_mode)

        return self

    def _get_point_translations(
        self: Transform,
        point: VectorLike[float] | None,
        multiply_mode: Literal['pre', 'post'] | None,
    ) -> tuple[None | Transform, None | Transform]:
        point = point if point is not None else self.point
        if point is not None:
            point_array = _validation.validate_array3(point, dtype_out=float, name='point')
            translate_away = Transform().translate(-point_array)
            translate_toward = Transform().translate(point_array)
            if multiply_mode == 'post' or (
                multiply_mode is None and self._multiply_mode == 'post'
            ):
                return translate_away, translate_toward
            else:
                return translate_toward, translate_away
        return None, None

    @property
    def check_finite(self: Transform) -> bool:  # numpydoc ignore: RT01
        """Check that the :attr:`~Transform.matrix` and :attr:`~Transform.inverse_matrix` have finite values.

        If ``True``, all transformations are checked to ensure they only contain
        finite values (i.e. no ``NaN`` or ``Inf`` values) and a ``ValueError`` is raised
        otherwise. This is useful to catch cases where the transformation(s) are poorly
        defined and/or are numerically unstable.

        This flag is enabled by default.
        """
        return self._check_finite

    @check_finite.setter
    def check_finite(self: Transform, value: bool) -> None:
        self._check_finite = bool(value)

    @property
    def translation(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return the translation component of the current transformation :attr:`~Transform.matrix`.

        .. versionadded:: 0.47

        See Also
        --------
        has_translation, translate, decompose

        Examples
        --------
        Compose a translation and get the translation component.

        >>> import pyvista as pv
        >>> trans = pv.Transform() + (1, 2, 3)
        >>> trans.translation
        (1.0, 2.0, 3.0)

        Compose a second translation and get the component again.

        >>> trans += (4, 5, 6)
        >>> trans.translation
        (5.0, 7.0, 9.0)

        """
        return self.GetPosition()

    @property
    def rotation_axis_angle(
        self,
    ) -> tuple[tuple[float, float, float], float]:  # numpydoc ignore=RT01
        """Return the rotation component of the current transformation :attr:`~Transform.matrix` as a vector and angle.

        .. versionadded:: 0.47

        See Also
        --------
        has_rotation, rotation_matrix, rotate_vector, as_rotation, decompose

        Examples
        --------
        Compose a rotation from a vector and angle.

        >>> import pyvista as pv
        >>> trans = pv.Transform().rotate_vector((1, 2, 3), 30)

        Get the rotation axis and angle.

        >>> axis, angle = trans.rotation_axis_angle
        >>> axis
        (0.2672, 0.5345, 0.8017)
        >>> angle
        30.0

        Compose a second rotation around the same axis and get the axis and angle again.

        >>> _ = trans.rotate_vector((1, 2, 3), 40)
        >>> axis, angle = trans.rotation_axis_angle
        >>> axis
        (0.2672, 0.5345, 0.8017)
        >>> angle
        70.0

        """
        # Decompose first to ensure we have a proper rotation
        _, R, _, _, _ = self.decompose()
        wxyz = Transform(R).GetOrientationWXYZ()
        return wxyz[1:4], wxyz[0]

    @property
    def rotation_matrix(self) -> NumpyArray[float]:  # numpydoc ignore=RT01
        """Return the rotation component of the current transformation :attr:`~Transform.matrix` as a 3x3 matrix.

        The rotation is orthonormal and right-handed with positive determinant.

        .. versionadded:: 0.47

        See Also
        --------
        has_rotation, rotation_axis_angle, rotate, as_rotation, decompose

        Examples
        --------
        Compose a rotation about the z-axis.

        >>> import pyvista as pv
        >>> trans = pv.Transform().rotate_z(90)

        Get the rotation matrix.

        >>> trans.rotation_matrix
        array([[ 0., -1.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])

        Compose a second rotation and get the rotation matrix again.

        >>> _ = trans.rotate_y(-90)
        >>> trans.rotation_matrix
        array([[ 0.,  0., -1.],
               [ 1.,  0.,  0.],
               [ 0., -1.,  0.]])

        """
        _, R, _, _, _ = self.decompose()
        return R

    @property
    def reflection(self) -> Literal[1, -1]:  # numpydoc ignore=RT01
        """Return the reflection component of the current transformation :attr:`~Transform.matrix` as an integer.

        ``1`` is returned if there is no reflection, and ``-1`` is returned if there
        is a reflection.

        See Also
        --------
        has_reflection, reflect, decompose

        Examples
        --------
        Create a transform and get its reflection.

        >>> import pyvista as pv
        >>> trans = pv.Transform()
        >>> trans.reflection
        1

        Compose a reflection about the x-axis and get the reflection again.

        >>> _ = trans.flip_x()
        >>> trans.reflection
        -1

        Compose a second reflection and get the reflection again.

        >>> _ = trans.flip_y()
        >>> trans.reflection
        1

        """
        _, _, N, _, _ = self.decompose()
        return N.astype(int).tolist()

    @property
    def scale_factors(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return the scaling component of the current transformation :attr:`~Transform.matrix`.

        The scaling factors are always positive.

        .. versionadded:: 0.47

        See Also
        --------
        has_scale, scale, decompose

        Examples
        --------
        Compose a scale matrix and get the scale factors.

        >>> import pyvista as pv
        >>> trans = pv.Transform() * (1, 2, 3)
        >>> trans.scale_factors
        (1.0, 2.0, 3.0)

        Compose a second scale matrix and get the factors again.

        >>> trans *= (4, 5, 6)
        >>> trans.scale_factors
        (4.0, 10.0, 18.0)

        """
        # Use PyVista's decompose instead of vtk's GetScale() method
        _, _, _, S, _ = self.decompose()
        return tuple(S.tolist())

    @property
    def shear_matrix(self) -> NumpyArray[float]:  # numpydoc ignore=RT01
        """Return the shear component of the current transformation :attr:`~Transform.matrix` as a 3x3 matrix.

        .. versionadded:: 0.47

        See Also
        --------
        has_shear, compose, decompose

        Examples
        --------
        Compose a symmetric shear matrix.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> shear = np.eye(4)
        >>> shear[0, 1] = 0.1
        >>> shear[1, 0] = 0.1
        >>> trans = pv.Transform(shear)

        Get the shear matrix. The shear matrix is the same as the input in this particular example,
        but in general this is not the case.

        >>> trans.shear_matrix
        array([[1. , 0.1, 0. ],
               [0.1, 1. , 0. ],
               [0. , 0. , 1. ]])

        Compose an asymmetric shear matrix instead.

        >>> shear = np.eye(4)
        >>> shear[0, 1] = 0.1
        >>> trans = pv.Transform(shear)

        Get the shear matrix. In this case, shear differs from the input because asymmetric shear
        can be decomposed into scale factors and a rotation.

        >>> trans.shear_matrix
        array([[1.        , 0.05      , 0.        ],
               [0.04975124, 1.        , 0.        ],
               [0.        , 0.        , 1.        ]])

        >>> trans.scale_factors
        (0.9987523388778445, 1.0037461005722337, 1.0)

        >>> axis, angle = trans.rotation_axis_angle
        >>> axis
        (0.0, 0.0, -1.0)
        >>> angle
        2.8624

        """
        _, _, _, _, K = self.decompose()
        return K

    @property
    def has_translation(self) -> bool:  # numpydoc ignore=RT01
        """Return ``True`` if the current transformation :attr:`~Transform.matrix` has a translation component.

        .. versionadded:: 0.47

        See Also
        --------
        translation, translate, decompose

        """
        return not np.allclose(self.translation, np.zeros((3,)))

    @property
    def has_rotation(self) -> bool:  # numpydoc ignore=RT01
        """Return ``True`` if the current transformation :attr:`~Transform.matrix` has a rotation component.

        .. versionadded:: 0.47

        See Also
        --------
        rotation_matrix, rotate, decompose

        """
        return not np.allclose(self.rotation_matrix, np.eye(3))

    @property
    def has_reflection(self) -> bool:  # numpydoc ignore=RT01
        """Return ``True`` if the current transformation :attr:`~Transform.matrix` has a reflection component.

        .. versionadded:: 0.47

        See Also
        --------
        reflection, reflect, decompose

        """
        return self.reflection == -1

    @property
    def has_scale(self) -> bool:  # numpydoc ignore=RT01
        """Return ``True`` if the current transformation :attr:`~Transform.matrix` has a scale component.

        .. versionadded:: 0.47

        See Also
        --------
        scale_factors, scale, decompose

        """
        return not np.allclose(self.scale_factors, np.ones((3,)))

    @property
    def has_shear(self) -> bool:  # numpydoc ignore=RT01
        """Return ``True`` if the current transformation :attr:`~Transform.matrix` has a shear component.

        .. versionadded:: 0.47

        See Also
        --------
        shear_matrix, compose, decompose

        """
        return not np.allclose(self.shear_matrix, np.eye(3))

    def as_rotation(
        self,
        representation: Literal['quat', 'matrix', 'rotvec', 'mrp', 'euler', 'davenport']
        | None = None,
        *args,
        **kwargs,
    ) -> Rotation | NumpyArray[float]:
        """Return the rotation component as a SciPy ``Rotation`` or any of its representations.

        The current :attr:`matrix` is first decomposed to extract the rotation component
        and then returned with the specified representation.

        .. note::

            This method depends on the ``scipy`` package which must be installed to use it.

        Parameters
        ----------
        representation : str, optional
            Representation of the rotation.

            - ``'quat'``: Represent as a quaternion using
              :meth:`~scipy.spatial.transform.Rotation.as_quat`. Returns a length-4 vector.
            - ``'matrix'``: Represent as a 3x3 matrix using
              :meth:`~scipy.spatial.transform.Rotation.as_matrix`.
            - ``'rotvec'``: Represent as a rotation vector using
              :meth:`~scipy.spatial.transform.Rotation.as_rotvec`.
            - ``'mrp'``: Represent as a Modified Rodrigues Parameters (MRPs) vector using
              :meth:`~scipy.spatial.transform.Rotation.as_mrp`.
            - ``'euler'``: Represent as Euler angles using
              :meth:`~scipy.spatial.transform.Rotation.as_euler`.
            - ``'davenport'``: Represent as Davenport angles using
              :meth:`~scipy.spatial.transform.Rotation.as_davenport`.

            If no representation is given, then an instance of
            :class:`scipy.spatial.transform.Rotation` is returned by default.

        *args
            Arguments passed to the ``Rotation`` method for the specified
            representation.

        **kwargs
            Keyword arguments passed to the ``Rotation`` method for the specified
            representation.

        Returns
        -------
        scipy.spatial.transform.Rotation | np.ndarray
            Rotation object or array depending on the representation.

        See Also
        --------
        rotation_matrix, rotation_axis_angle, decompose
            Get this transform's rotation component `without` using SciPy.
        rotate, rotate_x, rotate_y, rotate_z, rotate_vector
            Compose a rotation matrix.

        Examples
        --------
        Create a rotation matrix and initialize a :class:`Transform` from it.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> matrix = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        >>> transform = pv.Transform(matrix)

        Represent the rotation as :class:`scipy.spatial.transform.Rotation` instance.

        >>> rot = transform.as_rotation()
        >>> rot
        Rotation.from_matrix(array([[ 0., -1.,  0.],
                                    [ 1.,  0.,  0.],
                                    [ 0.,  0.,  1.]]))

        Represent the rotation as a quaternion.

        >>> rot = transform.as_rotation('quat')
        >>> rot
        array([0.        , 0.        , 0.70710678, 0.70710678])

        Represent the rotation as a rotation vector. The vector has a direction
        ``(0, 0, 1)`` and magnitude of ``pi/2``.

        >>> rot = transform.as_rotation('rotvec')
        >>> rot
        array([0.        , 0.        , 1.57079633])

        Represent the rotation as a Modified Rodrigues Parameters vector.

        >>> rot = transform.as_rotation('mrp')
        >>> rot
        array([0.        , 0.        , 0.41421356])

        Represent the rotation as x-y-z Euler angles in degrees.

        >>> rot = transform.as_rotation('euler', 'xyz', degrees=True)
        >>> rot
        array([ 0.,  0., 90.])

        Represent the rotation as extrinsic x-y-z Davenport angles in degrees.

        >>> rot = transform.as_rotation(
        ...     'davenport', np.eye(3), 'extrinsic', degrees=True
        ... )
        >>> rot
        array([-1.27222187e-14,  0.00000000e+00,  9.00000000e+01])

        """
        try:
            from scipy.spatial.transform import Rotation  # noqa: PLC0415
        except ImportError:
            msg = "The 'scipy' package must be installed to use `as_rotation`"
            raise ImportError(msg)

        if isinstance(representation, str):
            representation = representation.lower()  # type: ignore[assignment]

        _validation.check_contains(
            ['quat', 'matrix', 'rotvec', 'mrp', 'euler', 'davenport', None],
            must_contain=representation,
            name='representation',
        )
        if representation in ['rotation', 'matrix']:
            assert_empty_kwargs(**kwargs)

        _, R, _, _, _ = self.decompose()

        if representation == 'matrix':
            out = R
        else:
            rotation = Rotation.from_matrix(R)
            if representation is None:
                out = rotation
            elif representation == 'quat':
                out = rotation.as_quat(*args, **kwargs)
            elif representation == 'rotvec':
                out = rotation.as_rotvec(*args, **kwargs)
            elif representation == 'mrp':
                out = rotation.as_mrp(*args, **kwargs)
            elif representation == 'euler':
                out = rotation.as_euler(*args, **kwargs)
            elif representation == 'davenport':
                out = rotation.as_davenport(*args, **kwargs)
            else:  # pragma: no cover
                msg = f"Unexpected rotation type '{representation}'"  # type: ignore[unreachable]
                raise RuntimeError(msg)
        return out
