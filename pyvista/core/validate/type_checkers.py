"""Functions that check object types.

.. versionadded:: 0.43.0

A ``check`` function typically:

* Performs a simple validation on a single input variable.
* Raises an error if the check fails due to invalid input.
* Does not modify input or return anything.

"""
from collections.abc import Iterable, Sequence
from numbers import Number, Real
from typing import Any, Literal, Tuple, Union, get_args, get_origin

import numpy as np


def check_is_number(
    num: Union[float, int, complex, np.number, Number],
    /,
    *,
    definition: Literal['abstract', 'builtin', 'numpy'] = 'abstract',
    must_be_real=True,
    name: str = 'Object',
):
    """Check if an object is a number.

    By default, the number must be an instance of the abstract base class :class:`numbers.Real`.
    Optionally, the number can also be complex. The definition can also be restricted
    to strictly check if the number is a built-in numeric type (e.g. ``int``, ``float``)
    or numpy numeric data types (e.g. ``np.floating``, ``np.integer``).

    Notes
    -----
    - This check fails for instances of :class:`numpy.ndarray`. Use :func:`check_is_scalar`
      instead to also allow for 0-dimensional arrays.
    - Values such as ``float('inf')`` and ``float('NaN')`` are valid numbers and
      will not raise an error. Use :func:`check_is_finite` to check for finite values.

    .. warning::

        - Some NumPy numeric data types are subclasses of the built-in types whereas other are
          not. For example, ``numpy.float_`` is a subclass of ``float`` and ``numpy.complex_``
          is a subclass ``complex``. However, ``numpy.int_`` is not a subclass of ``int`` and
          ``numpy.bool_`` is not a subclass of ``bool``.
        - The built-in ``bool`` type is a subclass of ``int`` whereas NumPy's``.bool_`` type
          is not a subclass of ``np.int_`` (``np.bool_`` is not a numeric type).

        This can lead to unexpected results:

        - This check will always fail for ``np.bool_`` types.
        - This check will pass for ``np.float_`` or ``np.complex_``, even if
          ``definition='builtin'``.

    Parameters
    ----------
    num : float | int | complex | numpy.number | Number
        Number to check.

    definition : str, default: 'abstract'
        Control the base class(es) to use when checking the number's type. Must be
        one of:

        - ``'abstract'`` : number must be an instance of one of the abstract types
          in :py:mod:`numbers`.
        - ``'builtin'`` : number must be an instance of one of the built-in numeric
          types.
        - ``'numpy'`` : number must be an instance of NumPy's data types.

    must_be_real : bool, default: True
        If ``True``, the number must be real, i.e. an integer or
        floating type. Set to ``False`` to allow complex numbers.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If input is not an instance of a numeric type.

    See Also
    --------
    check_is_scalar
        Similar function which allows 0-dimensional ndarrays.
    check_is_numeric
        Similar function for any dimensional array of numbers.
    check_is_real
        Similar function for any dimensional array of real numbers.
    check_is_finite


    Examples
    --------
    Check if a float is a number.
    >>> from pyvista.core import validate
    >>> num = 42.0
    >>> type(num)
    <class 'float'>
    >>> validate.check_is_number(num)

    Check if an element of a NumPy array is a number.

    >>> import numpy as np
    >>> num_array = np.array([1, 2, 3])
    >>> num = num_array[0]
    >>> type(num)
    <class 'numpy.int64'>
    >>> validate.check_is_number(num)

    Check if a complex number is a number.
    >>> num = 1 + 2j
    >>> type(num)
    <class 'complex'>
    >>> validate.check_is_number(num, must_be_real=False)

    """
    check_is_string_in_iterable(definition, ['abstract', 'builtin', 'numpy'])

    valid_type: Any
    if definition == 'abstract':
        valid_type = Real if must_be_real else Number
    elif definition == 'builtin':
        valid_type = (float, int) if must_be_real else (float, int, complex)
    elif definition == 'numpy':
        valid_type = (np.floating, np.integer) if must_be_real else np.number
    else:
        raise NotImplementedError  # pragma: no cover

    try:
        check_is_instance(num, valid_type, allow_subclass=True, name=name)
    except TypeError:
        raise


def check_is_string(obj: str, /, *, allow_subclass: bool = True, name: str = 'Object'):
    """Check if an object is an instance of ``str``.

    Parameters
    ----------
    obj : str
        Object to check.

    allow_subclass : bool, default: True
        If ``True``, the object's type must be ``str`` or a subclass of
        ``str``. Otherwise, subclasses are not allowed.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If input is not an instance of ``str``.

    See Also
    --------
    check_is_string_in_iterable
    check_is_iterable_of_strings
    check_is_sequence
    check_is_instance

    Examples
    --------
    Check if an object is a string.

    >>> from pyvista.core import validate
    >>> validate.check_is_string("eggs")

    """
    try:
        check_is_instance(obj, str, allow_subclass=allow_subclass, name=name)
    except TypeError:
        raise


def check_is_sequence(obj: Sequence, /, *, name: str = 'Object'):
    """Check if an object is an instance of ``Sequence``.

    Parameters
    ----------
    obj : Sequence
        Object to check.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If input is not an instance of ``Sequence``.

    See Also
    --------
    check_is_iterable
    check_is_instance

    Examples
    --------
    Check if an object is a sequence.

    >>> import numpy as np
    >>> from pyvista.core import validate
    >>> validate.check_is_sequence([1, 2, 3])
    >>> validate.check_is_sequence("A")

    """
    try:
        check_is_instance(obj, Sequence, allow_subclass=True, name=name)
    except TypeError:
        raise


def check_is_iterable(obj: Iterable, /, *, name: str = 'Object'):
    """Check if an object is an instance of ``Iterable``.

    Parameters
    ----------
    obj : Iterable
        Iterable object to check.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If input is not an instance of ``Iterable``.

    See Also
    --------
    check_is_sequence
    check_is_instance
    check_is_iterable_of_some_type

    Examples
    --------
    Check if an object is iterable.

    >>> import numpy as np
    >>> from pyvista.core import validate
    >>> validate.check_is_iterable([1, 2, 3])
    >>> validate.check_is_iterable(np.array((4, 5, 6)))

    """
    try:
        check_is_instance(obj, Iterable, allow_subclass=True, name=name)
    except TypeError:
        raise


def check_is_instance(
    obj: Any,
    /,
    classinfo: Union[type, Tuple[type, ...]],
    *,
    allow_subclass: bool = True,
    name: str = 'Object',
):
    """Check if an object is an instance of the given type or types.

    Parameters
    ----------
    obj : Any
        Object to check.

    classinfo : type | tuple[type, ...]
        A type, tuple of types, or a Union of types. Object must be an instance
        of one of the types.

    allow_subclass : bool, default: True
        If ``True``, the object's type must be specified by ``classinfo``
        or any of its subclasses. Otherwise, subclasses are not allowed.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If object is not an instance of any of the given types.

    See Also
    --------
    check_is_type
    check_is_number
    check_is_string
    check_is_iterable
    check_is_sequence

    Examples
    --------
    Check if an object is an instance of ``complex``.

    >>> from pyvista.core import validate
    >>> validate.check_is_instance(1 + 2j, complex)

    Check if an object is an instance of one of several types.

    >>> validate.check_is_instance("eggs", (int, str))

    """
    if not isinstance(name, str):
        raise TypeError(f"Name must be a string, got {type(name)} instead.")

    # Get class info from generics
    if get_origin(classinfo) is Union:
        classinfo = get_args(classinfo)

    # Count num classes
    if isinstance(classinfo, tuple) and all(map(lambda cls: isinstance(cls, type), classinfo)):
        num_classes = len(classinfo)
    else:
        num_classes = 1

    # Check if is instance
    is_instance = isinstance(obj, classinfo)

    # Set flag to raise error if not instance
    is_error = False
    if allow_subclass and not is_instance:
        is_error = True
        if num_classes == 1:
            msg_body = "must be an instance of"
        else:
            msg_body = "must be an instance of any type"

    # Set flag to raise error if not type
    elif not allow_subclass:
        if isinstance(classinfo, tuple):
            if type(obj) not in classinfo:
                is_error = True
                msg_body = "must have one of the following types"
        elif type(obj) is not classinfo:
            is_error = True
            msg_body = "must have type"

    if is_error:
        msg = f"{name} {msg_body} {classinfo}. Got {type(obj)} instead."
        raise TypeError(msg)


def check_is_type(obj: Any, /, classinfo: Union[type, Tuple[type, ...]], *, name: str = 'Object'):
    """Check if an object is one of the given type or types.

    Notes
    -----
    The use of :func:`check_is_instance` is generally preferred as it
    allows subclasses. Use :func:`check_is_type` only for cases where
    exact types are necessary.

    Parameters
    ----------
    obj : Any
        Object to check.

    classinfo : type | tuple[type, ...]
        A type, tuple of types, or a Union of types. Object must be one
        of the types.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If object is not any of the given types.

    See Also
    --------
    check_is_instance

    Examples
    --------
    Check if an object is type ``dict`` or ``set``.

    >>> from pyvista.core import validate
    >>> validate.check_is_type({'spam': "eggs"}, (dict, set))

    """
    try:
        check_is_instance(obj, classinfo, allow_subclass=False, name=name)
    except TypeError:
        raise


def check_is_iterable_of_some_type(
    iterable_obj: Iterable,
    /,
    some_type: Union[type, Tuple[type, ...]],
    *,
    allow_subclass: bool = True,
    name: str = 'Iterable',
):
    """Check if an iterable's items all have a specified type.

    Parameters
    ----------
    iterable_obj : Iterable
        Iterable to check.

    some_type : type | tuple[type, ...]
        Class type(s) to check for. Each element of the sequence must
        have the type or one of the types specified.

    allow_subclass : bool, default: True
        If ``True``, the type of the iterable's items must be any of the
        given types or a subclass thereof. Otherwise, subclasses are not
        allowed.

    name : str, default: "Iterable"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If any of the items in the iterable have an incorrect type.

    See Also
    --------
    check_is_instance
    check_is_iterable
    check_is_iterable_of_strings

    Examples
    --------
    Check if a ``tuple`` only has ``int`` or ``float`` elements.

    >>> from pyvista.core import validate
    >>> validate.check_is_iterable_of_some_type((1, 2, 3.0), (int, float))

    Check if a ``list`` only has ``list`` elements.

    >>> from pyvista.core import validate
    >>> validate.check_is_iterable_of_some_type([[1], [2], [3]], list)

    """
    check_is_iterable(iterable_obj, name=name)
    try:
        # TODO: add bool return to check functions and convert this statement
        # to a generator with all()
        [
            check_is_instance(
                item, some_type, allow_subclass=allow_subclass, name=f"All items of {name}"
            )
            for item in iterable_obj
        ]

    except TypeError:
        raise


def check_is_iterable_of_strings(
    iterable_obj: Iterable, /, *, allow_subclass: bool = True, name: str = 'String Iterable'
):
    """Check if an iterable's items are all strings.

    Parameters
    ----------
    iterable_obj : Iterable
        Iterable of strings to check.

    allow_subclass : bool, default: True
        If ``True``, the type of the iterable's items must be any of the
        given types or a subclass thereof. Otherwise, subclasses are not
        allowed.

    name : str, default: "String Iterable"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If any of the elements have an incorrect type.

    See Also
    --------
    check_is_iterable
    check_is_string
    check_is_string_in_iterable

    Examples
    --------
    Check if a ``tuple`` only has ``str`` elements.

    >>> from pyvista.core import validate
    >>> validate.check_is_iterable_of_strings(("cat", "dog"))

    """
    try:
        check_is_iterable_of_some_type(iterable_obj, str, allow_subclass=allow_subclass, name=name)
    except TypeError:
        raise


def check_is_string_in_iterable(
    string_in: str, /, string_iterable: Iterable, *, name: str = 'String'
):
    """Check if a given string is in an iterable of strings.

    Parameters
    ----------
    string_in : str
        String to check.

    string_iterable : Iterable
        Iterable containing only strings.

    name : str, default: "String"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If the string is not in the iterable.

    See Also
    --------
    check_is_iterable
    check_is_string
    check_is_iterable_of_strings

    Examples
    --------
    Check if ``"A"`` is in a list of strings.

    >>> from pyvista.core import validate
    >>> validate.check_is_string_in_iterable("A", ["A", "B", "C"])

    """
    check_is_string(string_in, name=name)
    check_is_iterable_of_strings(string_iterable)
    if string_in not in string_iterable:
        raise ValueError(
            f"{name} '{string_in}' is not in the iterable. "
            f"{name} must be one of: \n\t" + str(string_iterable)
        )
