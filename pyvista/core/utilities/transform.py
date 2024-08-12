"""Module containing the Transform class."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

from pyvista.core import _validation
from pyvista.core import _vtk_core as _vtk
from pyvista.core.utilities.arrays import array_from_vtkmatrix
from pyvista.core.utilities.arrays import vtkmatrix_from_array

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
    Concatenate two transformations with :meth:`translate` and :meth:`scale` using
    post-multiplication (default).

    >>> import pyvista as pv
    >>> transform = pv.Transform()
    >>> transform.multiply_mode
    'post'
    >>> position = (-0.6, -0.8, 2.1)
    >>> scale = 2.0
    >>> _ = transform.translate(position)
    >>> _ = transform.scale(scale)

    Use :attr:`n_transformations` to verify that there are two transformations.

    >>> transform.n_transformations
    2

    Use :attr:`matrix_list` to get a list of the transformations. Since
    post-multiplication is used, the translation matrix is first in the list since
    it was applied first, and the scale matrix is second.

    >>> translation = transform.matrix_list[0]
    >>> translation
    array([[ 1. ,  0. ,  0. , -0.6],
           [ 0. ,  1. ,  0. , -0.8],
           [ 0. ,  0. ,  1. ,  2.1],
           [ 0. ,  0. ,  0. ,  1. ]])

    >>> scaling = transform.matrix_list[1]
    >>> scaling
    array([[2., 0., 0., 0.],
           [0., 2., 0., 0.],
           [0., 0., 2., 0.],
           [0., 0., 0., 1.]])

    Show the concatenated matrix. Note that the position has doubled from
    ``(-0.6, -0.8, 2.1)`` to ``(-1.2, -1.6, 4.2)``, indicating that the scaling is
    applied *after* the translation.

    >>> post_matrix = transform.matrix
    >>> post_matrix
    array([[ 2. ,  0. ,  0. , -1.2],
           [ 0. ,  2. ,  0. , -1.6],
           [ 0. ,  0. ,  2. ,  4.2],
           [ 0. ,  0. ,  0. ,  1. ]])

    Reset the transform to the identity matrix and set :attr:`multiply_mode` to
    use pre-multiplication instead.

    >>> _ = transform.identity()
    >>> transform.multiply_mode = 'pre'

    Apply the same two transformations as before and in the same order. Note that the
    function calls can be chained together.

    >>> _ = transform.translate(position).scale(scale)

    Show the matrix list again. Note how with pre-multiplication, the order is
    reversed from post-multiplication, and the scaling matrix is now first
    followed by the translation.

    >>> scaling = transform.matrix_list[0]
    >>> scaling
    array([[2., 0., 0., 0.],
           [0., 2., 0., 0.],
           [0., 0., 2., 0.],
           [0., 0., 0., 1.]])

    >>> translation = transform.matrix_list[1]
    >>> translation
    array([[ 1. ,  0. ,  0. , -0.6],
           [ 0. ,  1. ,  0. , -0.8],
           [ 0. ,  0. ,  1. ,  2.1],
           [ 0. ,  0. ,  0. ,  1. ]])

    Show the concatenated matrix. Unlike before, the position is not scaled,
    and is ``(-0.6, -0.8, 2.1)`` instead of ``(-1.2, -1.6, 4.2)``.

    >>> pre_matrix = transform.matrix
    >>> pre_matrix
    array([[ 2. ,  0. ,  0. , -0.6],
           [ 0. ,  2. ,  0. , -0.8],
           [ 0. ,  0. ,  2. ,  2.1],
           [ 0. ,  0. ,  0. ,  1. ]])

    Another way to compare pre- and post-multiplcation is by manually concatenating
    the transformations using matrix multiplication:

    >>> # Apply translation first, then scaling
    >>> np.array_equal(post_matrix, scaling @ translation)
    True

    >>> # Apply scaling first, then translation
    >>> np.array_equal(pre_matrix, translation @ scaling)
    True

    Apply the two transformations to a dataset and plot them. Note how the meshes
    have different positions since pre- and post-multiplication produce different
    transformations.

    >>> mesh_post = pv.Sphere().transform(post_matrix)
    >>> mesh_pre = pv.Cone().transform(pre_matrix)
    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh_post, color='goldenrod')
    >>> _ = pl.add_mesh(mesh_pre, color='teal')
    >>> _ = pl.add_axes_at_origin()
    >>> pl.show()

    Get the concatenated inverse transformation matrix of the pre-multiplication case.

    >>> inverse_matrix = transform.inverse_matrix
    >>> inverse_matrix
    array([[ 0.5 ,  0.  ,  0.  ,  0.3 ],
           [ 0.  ,  0.5 ,  0.  ,  0.4 ],
           [ 0.  ,  0.  ,  0.5 , -1.05],
           [ 0.  ,  0.  ,  0.  ,  1.  ]])

    Similar to using :attr:`matrix_list`, we can inspect the individual transformation
    inverses with :attr:`inverse_matrix_list`.

    >>> transform.inverse_matrix_list[0]  # inverse scaling
    array([[0.5, 0. , 0. , 0. ],
           [0. , 0.5, 0. , 0. ],
           [0. , 0. , 0.5, 0. ],
           [0. , 0. , 0. , 1. ]])

    >>> transform.inverse_matrix_list[1]  # inverse translation
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
        if trans is not None:
            self.matrix = trans  # type: ignore[assignment]

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
            else _validation.validate_array3(point, dtype_out=float, to_tuple=True)
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
        >>> transform = pv.Transform()
        >>> _ = transform.pre_multiply()
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
        >>> transform = pv.Transform()
        >>> _ = transform.post_multiply()
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
            Point to scale about. By default, the object's :attr:`point` is used,
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
        >>> transform = pv.Transform()
        >>> _ = transform.scale(1, 2, 3)
        >>> transform.matrix
        array([[1., 0., 0., 0.],
               [0., 2., 0., 0.],
               [0., 0., 3., 0.],
               [0., 0., 0., 1.]])

        Concatenate a second scale matrix.

        >>> _ = transform.scale(2)
        >>> transform.matrix
        array([[2., 0., 0., 0.],
               [0., 4., 0., 0.],
               [0., 0., 6., 0.],
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
        >>> transform = pv.Transform()
        >>> _ = transform.translate(1, 2, 3)
        >>> transform.matrix
        array([[1., 0., 0., 1.],
               [0., 1., 0., 2.],
               [0., 0., 1., 3.],
               [0., 0., 0., 1.]])

        Concatenate a second translation matrix.

        >>> _ = transform.translate((1, 1, 1))
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

        Examples
        --------
        Concatenate a rotation matrix. In this case the rotation rotates about the
        z-axis by 90 degrees.

        >>> import pyvista as pv
        >>> rotation_z_90 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        >>> transform = pv.Transform()
        >>> _ = transform.rotate(rotation_z_90)
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
        """
        valid_rotation = _validation.validate_transform3x3(rotation, name='rotation')

        return self._concatenate_with_translations(
            valid_rotation, point=point, multiply_mode=multiply_mode
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
        >>> transform = pv.Transform()
        >>> _ = transform.concatenate(array)
        >>> transform.matrix
        array([[ 0.707, -0.707,  0.   ,  0.   ],
               [ 0.707,  0.707,  0.   ,  0.   ],
               [ 0.   ,  0.   ,  1.   ,  1.5  ],
               [ 0.   ,  0.   ,  0.   ,  2.   ]])

        Define a second transformation and concatenate it.

        >>> array = [[1, 0, 0], [0, 0, -1], [0, -1, 0]]
        >>> _ = transform.concatenate(array)
        >>> transform.matrix
        array([[ 0.707, -0.707,  0.   ,  0.   ],
               [ 0.   ,  0.   , -1.   , -1.5  ],
               [-0.707, -0.707,  0.   ,  0.   ],
               [ 0.   ,  0.   ,  0.   ,  2.   ]])
        """
        # Make sure we have a vtkTransform
        if isinstance(transform, _vtk.vtkTransform):
            vtk_transform = transform
        elif isinstance(transform, _vtk.vtkMatrix4x4):
            vtk_transform = _vtk.vtkTransform()
            vtk_transform.SetMatrix(transform)
        else:
            array = _validation.validate_transform4x4(transform, name='matrix')
            vtk_transform = _vtk.vtkTransform()
            vtk_transform.SetMatrix(vtkmatrix_from_array(array))

        if multiply_mode is not None:
            # Override multiply mode
            original_mode = self.multiply_mode
            self.multiply_mode = multiply_mode

        self.Concatenate(vtk_transform)

        if multiply_mode:
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
        return array_from_vtkmatrix(self.GetMatrix())

    @matrix.setter
    def matrix(self, trans: TransformLike):  # numpydoc ignore=GL08
        array = _validation.validate_transform4x4(trans)
        self.SetMatrix(vtkmatrix_from_array(array))

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
        return array_from_vtkmatrix(self.GetInverse().GetMatrix())

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
        >>> transform = pv.Transform()
        >>> _ = transform.scale(2.0)
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
        >>> transform = pv.Transform()
        >>> _ = transform.scale(2.0)
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
            point_array = _validation.validate_array3(point, dtype_out=float)
            translate_away = Transform().translate(-point_array)
            translate_toward = Transform().translate(point_array)
            if multiply_mode == 'post' or self._multiply_mode == 'post':
                return translate_away, translate_toward
            else:
                return translate_toward, translate_away
        return None, None
