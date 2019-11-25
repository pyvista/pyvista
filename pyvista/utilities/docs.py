"""Module containing helper function for the documentation."""


def copy_function_doc(source, alias=False, deprecated=False):
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
        doc = source.__doc__
        if func.__doc__ is not None:
            if not doc.rstrip(' ').endswith('\n'):
                doc += '\n'
            doc += func.__doc__
        elif alias or deprecated:
            doc += '\n'
        if alias:
            doc += 'Alias for: ``' + source.__name__ + '``.\n'
        elif deprecated:
            doc += 'DEPRECATED: Please use ``' + source.__name__ \
                + '`` instead.\n'
        func.__doc__ = doc
        return func
    return wrapper
