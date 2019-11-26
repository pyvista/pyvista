"""Module containing helper function for the documentation."""


def copy_function_doc(source):
    """Copy the docstring from another function (decorator).

    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator.

    Parameters
    ----------
    source : function
        Function to copy the docstring from.

    Return
    ------
    wrapper : function
        The decorated function.

    """
    def wrapper(func):
        if source.__doc__ is None or len(source.__doc__) == 0:
            raise ValueError('Cannot copy docstring: docstring was empty.')
        # issue is here:
        doc = source.__doc__
        if func.__doc__ is not None:
            if not doc.rstrip(' ').endswith('\n'):
                doc += '\n'
            doc += func.__doc__
        func.__doc__ = doc
        return func
    return wrapper


def deprecated_function_doc(source):
    """Specify that the decorated function is a deprecated version of source."""
    def wrapper(func):
        doc = func.__doc__
        if func.__doc__ is not None:
            if not doc.rstrip(' ').endswith('\n'):
                doc += '\n'
        else:
            func.__doc__ = ''
        func.__doc__ += 'DEPRECATED: Please use ``' + source.__name__ + \
            '`` instead.\n'
        return func
    return wrapper


def aliased_function_doc(source):
    """Specify that the decorated function is an alias version of source."""
    def wrapper(func):
        doc = func.__doc__
        if func.__doc__ is not None:
            if not doc.rstrip(' ').endswith('\n'):
                doc += '\n'
        else:
            func.__doc__ = ''
        func.__doc__ += 'Alias for: ``' + source.__name__ + '``.\n'
        return func
    return wrapper
