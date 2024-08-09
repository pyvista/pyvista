"""Module containing Transform classes."""

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
    :meth:`post_multiply` (the default) mode.  In pre-multiply mode, any additional
    transformations will occur *before* any transformations represented by the current
    :attr:`matrix`. In post-multiply mode, the additional transformation will occur
    *after* any transformations represented by the current matrix.

    .. note::

        This class performs all of its operations in a right-handed coordinate system
        with right-handed rotations. Some other graphics libraries use left-handed
        coordinate systems and rotations.
    """

    def __init__(self):
        super().__init__()
        self.multiply_mode = 'post'

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
    def multiply_mode(self, multiply_mode: Literal['pre', 'post']):  # numpydoc ignore=GL08
        _validation.check_contains(
            item=multiply_mode, container=['pre', 'post'], name='multiply mode'
        )
        self.pre_multiply() if multiply_mode == 'pre' else self.post_multiply()

    def pre_multiply(self):
        """Set the multiplication mode to pre-multiply.

        In pre-multiply mode, any additional transformations (e.g. using
        :meth:`translate`,:meth:`concatenate`, etc.) will occur *before* any
        transformations represented by the current :attr:`matrix`.
        """
        self._multiply_mode: Literal['pre'] = 'pre'
        self.PreMultiply()

    def post_multiply(self):
        """Set the multiplication mode to post-multiply.

        In post-multiply mode, any additional transformations (e.g. using
        :meth:`translate`,:meth:`concatenate`, etc.) will occur *after* any
        transformations represented by the current :attr:`matrix`.
        """
        self._multiply_mode: Literal['post'] = 'post'
        self.PostMultiply()

    def scale(self, *factor, multiply_mode: Literal['pre', 'post'] | None = None) -> None:
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

        multiply_mode : 'pre' | 'post' | None, optional
            Multiplication mode to use when concatenating the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.
        """
        valid_factor = _validation.validate_array3(
            factor, broadcast=True, dtype_out=float, name='scale factor'
        )
        self._concatenate('Scale', valid_factor, multiply_mode=multiply_mode)

    def translate(
        self, vector: VectorLike[float], multiply_mode: Literal['pre', 'post'] | None = None
    ) -> None:
        """Concatenate a translation matrix.

        Create a translation matrix and :meth:`concatenate` it with the current
        transformation :attr:`matrix` according to pre-multiply or post-multiply
        semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        vector : VectorLike[float]
            Vector to use for the translation.

        multiply_mode : 'pre' | 'post' | None, optional
            Multiplication mode to use when concatenating the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        """
        valid_vector = _validation.validate_array3(
            vector, broadcast=True, dtype_out=float, name='translation vector'
        )
        self._concatenate('Translate', valid_vector, multiply_mode=multiply_mode)

    def rotate(
        self, rotation: RotationLike, multiply_mode: Literal['pre', 'post'] | None = None
    ) -> None:
        """Concatenate a rotation matrix.

        Create a rotation matrix and :meth:`concatenate` it with the current
        transformation :attr:`matrix` according to pre-multiply or post-multiply
        semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        rotation : RotationLike
            3x3 rotation matrix or a SciPy ``Rotation`` object.

        multiply_mode : 'pre' | 'post' | None, optional
            Multiplication mode to use when concatenating the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.
        """
        valid_rotation = _validation.validate_transform3x3(rotation, name='rotation')
        self.concatenate(valid_rotation, multiply_mode=multiply_mode)

    def concatenate(
        self, matrix: TransformLike, multiply_mode: Literal['pre', 'post'] | None = None
    ) -> None:
        """Concatenate a transformation matrix.

        Create a 4x4 matrix from any transform-like input and :meth:`concatenate` it
        with the current transformation :attr:`matrix` according to pre-multiply or
        post-multiply semantics.

        Internally, the matrix is stored in the :attr:`matrix_list`.

        Parameters
        ----------
        matrix : TransformLike
            3x3 rotation matrix or a SciPy ``Rotation`` object.

        multiply_mode : 'pre' | 'post' | None, optional
            Multiplication mode to use when concatenating the matrix. By default, the
            object's :attr:`multiply_mode` is used, but this can be overridden. Set this
            to ``'pre'`` for pre-multiplication or ``'post'`` for post-multiplication.

        """
        if not isinstance(matrix, _vtk.vtkMatrix4x4):
            matrix = _validation.validate_transform4x4(matrix, name='matrix')
            matrix = vtkmatrix_from_array(matrix)
        self._concatenate('Concatenate', matrix, multiply_mode=multiply_mode)

    @property
    def matrix(self) -> NumpyArray[float]:
        """Return the current transformation matrix.

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

    def invert(self) -> None:  # numpydoc ignore: RT01
        """Invert the current transformation.

        The current transformation :attr:`matrix` (including all matrices in the
        :attr:`matrix_list`) is inverted every time :meth:`invert` is called.

        Use :attr:`is_inverted` to check if the transformations are currently inverted.
        """
        self.Inverse()

    @property
    def is_inverted(self) -> bool:  # numpydoc ignore: RT01
        """Get the inverse flag of the transformation."""
        return bool(self.GetInverseFlag())

    def _concatenate(
        self,
        method: Literal['Scale', 'Translate', 'Concatenate'],
        arg,
        multiply_mode: Literal['pre', 'post'] | None,
    ):
        if multiply_mode is not None:
            original_mode = self.multiply_mode
            self.multiply_mode = multiply_mode

        method_obj = getattr(self, method)
        method_obj(arg)

        if multiply_mode:
            self.multiply_mode = original_mode
