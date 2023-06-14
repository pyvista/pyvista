"""Miscellaneous core utilities."""
from collections.abc import Iterable
import enum
from functools import lru_cache
import importlib
import sys
import threading
import traceback
import warnings


def assert_empty_kwargs(**kwargs):
    """Assert that all keyword arguments have been used (internal helper).

    If any keyword arguments are passed, a ``TypeError`` is raised.
    """
    n = len(kwargs)
    if n == 0:
        return True
    caller = sys._getframe(1).f_code.co_name
    keys = list(kwargs.keys())
    bad_arguments = ', '.join([f'"{key}"' for key in keys])
    if n == 1:
        grammar = "is an invalid keyword argument"
    else:
        grammar = "are invalid keyword arguments"
    message = f"{bad_arguments} {grammar} for `{caller}`"
    raise TypeError(message)


def check_valid_vector(point, name=''):
    """Check if a vector contains three components."""
    if not isinstance(point, Iterable):
        raise TypeError(f'{name} must be a length three iterable of floats.')
    if len(point) != 3:
        if name == '':
            name = 'Vector'
        raise ValueError(f'{name} must be a length three iterable of floats.')


def abstract_class(cls_):
    """Decorate a class, overriding __new__.

    Preventing a class from being instantiated similar to abc.ABCMeta
    but does not require an abstract method.
    """

    def __new__(cls, *args, **kwargs):
        if cls is cls_:
            raise TypeError(f'{cls.__name__} is an abstract class and may not be instantiated.')
        return object.__new__(cls)

    cls_.__new__ = __new__
    return cls_


class AnnotatedIntEnum(int, enum.Enum):
    """Annotated enum type."""

    def __new__(cls, value, annotation):
        """Initialize."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.annotation = annotation
        return obj

    @classmethod
    def from_str(cls, input_str):
        """Create from string."""
        for value in cls:
            if value.annotation.lower() == input_str.lower():
                return value
        raise ValueError(f"{cls.__name__} has no value matching {input_str}")

    @classmethod
    def from_any(cls, value):
        """Create from string, int, etc."""
        if isinstance(value, cls):
            return value
        elif isinstance(value, int):
            return cls(value)
        elif isinstance(value, str):
            return cls.from_str(value)
        else:
            raise ValueError(f"{cls.__name__} has no value matching {value}")


@lru_cache(maxsize=None)
def has_module(module_name):
    """Return if a module can be imported."""
    module_spec = importlib.util.find_spec(module_name)
    return module_spec is not None


def try_callback(func, *args):
    """Wrap a given callback in a try statement.

    Parameters
    ----------
    func : callable
        Callable object.

    *args
        Any arguments.

    """
    try:
        func(*args)
    except Exception:
        etype, exc, tb = sys.exc_info()
        stack = traceback.extract_tb(tb)[1:]
        formatted_exception = 'Encountered issue in callback (most recent call last):\n' + ''.join(
            traceback.format_list(stack) + traceback.format_exception_only(etype, exc)
        ).rstrip('\n')
        warnings.warn(formatted_exception)


def threaded(fn):
    """Call a function using a thread.

    Parameters
    ----------
    fn : callable
        Callable object.

    Returns
    -------
    function
        Wrapped function.

    """

    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


class conditional_decorator:
    """Conditional decorator for methods.

    Parameters
    ----------
    dec
        Decorator
    condition
        Condition to match.

    """

    def __init__(self, dec, condition):
        """Initialize."""
        self.decorator = dec
        self.condition = condition

    def __call__(self, func):
        """Call the decorated function if condition is matched."""
        if not self.condition:
            # Return the function unchanged, not decorated.
            return func
        return self.decorator(func)


def _check_range(value, rng, parm_name):
    """Check if a parameter is within a range."""
    if value < rng[0] or value > rng[1]:
        raise ValueError(
            f'The value {float(value)} for `{parm_name}` is outside the acceptable range {tuple(rng)}.'
        )


def no_new_attr(cls):
    """Override __setattr__ to not permit new attributes."""
    if not hasattr(cls, '_new_attr_exceptions'):
        cls._new_attr_exceptions = []

    def __setattr__(self, name, value):
        """Do not allow setting attributes."""
        if hasattr(self, name) or name in cls._new_attr_exceptions:
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f'Attribute "{name}" does not exist and cannot be added to type '
                f'{self.__class__.__name__}'
            )

    setattr(cls, '__setattr__', __setattr__)
    return cls
