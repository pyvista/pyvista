"""Functions that check object types.

.. versionadded:: 0.43.0

A ``check`` function typically:

* Performs a simple validation on a single input variable.
* Raises an error if the check fails due to invalid input.
* Does not modify input or return anything.

"""
from collections.abc import Iterable, Sequence
from typing import Any, Tuple, Union, get_args, get_origin


def check_string(obj: str, /, *, allow_subclass: bool = True, name: str = 'Object'):
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
    >>> validate.check_string("eggs")

    """
    try:
        check_instance(obj, str, allow_subclass=allow_subclass, name=name)
    except TypeError:
        raise


def check_sequence(obj: Sequence, /, *, name: str = 'Object'):
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
    >>> validate.check_sequence([1, 2, 3])
    >>> validate.check_sequence("A")

    """
    try:
        check_instance(obj, Sequence, allow_subclass=True, name=name)
    except TypeError:
        raise


def check_iterable(obj: Iterable, /, *, name: str = 'Object'):
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
    >>> validate.check_iterable([1, 2, 3])
    >>> validate.check_iterable(np.array((4, 5, 6)))

    """
    try:
        check_instance(obj, Iterable, allow_subclass=True, name=name)
    except TypeError:
        raise


def check_instance(
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
    >>> validate.check_instance(1 + 2j, complex)

    Check if an object is an instance of one of several types.

    >>> validate.check_instance("eggs", (int, str))

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


def check_type(obj: Any, /, classinfo: Union[type, Tuple[type, ...]], *, name: str = 'Object'):
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
    >>> validate.check_type({'spam': "eggs"}, (dict, set))

    """
    try:
        check_instance(obj, classinfo, allow_subclass=False, name=name)
    except TypeError:
        raise


def check_iterable_item_type(
    iterable_obj: Iterable,
    /,
    item_type: Union[type, Tuple[type, ...]],
    *,
    allow_subclass: bool = True,
    name: str = 'Iterable',
):
    """Check if an iterable's items all have a specified type.

    Parameters
    ----------
    iterable_obj : Iterable
        Iterable to check.

    item_type : type | tuple[type, ...]
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
    >>> validate.check_iterable_item_type((1, 2, 3.0), (int, float))

    Check if a ``list`` only has ``list`` elements.

    >>> from pyvista.core import validate
    >>> validate.check_iterable_item_type([[1], [2], [3]], list)

    """
    check_iterable(iterable_obj, name=name)
    try:
        # TODO: add bool return to check functions and convert this statement
        # to a generator with all()
        [
            check_instance(
                item, item_type, allow_subclass=allow_subclass, name=f"All items of {name}"
            )
            for item in iterable_obj
        ]

    except TypeError:
        raise


def check_iterable_of_strings(
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
    >>> validate.check_iterable_of_strings(("cat", "dog"))

    """
    try:
        check_iterable_item_type(iterable_obj, str, allow_subclass=allow_subclass, name=name)
    except TypeError:
        raise


def check_contains(obj: Any, /, container: Any, *, name: str = 'Input'):
    """Check if an object is in a container.

    Parameters
    ----------
    obj : Any
        Object to check.

    container : Any
        Container the object is expected to be in.

    name : str, default: "Input"
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
    >>> check_contains("A", ["A", "B", "C"])

    """
    if obj not in container:
        if isinstance(container, (list, tuple)):
            qualifier = "one of"
        else:
            qualifier = "in"
        msg = f"{name} '{obj}' is not valid. {name} must be " f"{qualifier}: \n\t{container}"
        raise ValueError(msg)
