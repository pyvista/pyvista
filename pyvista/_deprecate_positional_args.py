from __future__ import annotations

from functools import wraps
import inspect
from inspect import Parameter
from inspect import Signature
import os
from pathlib import Path
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import overload

from typing_extensions import ParamSpec

from pyvista._version import version_info
from pyvista._warn_external import warn_external

if TYPE_CHECKING:
    from collections.abc import Callable

_MAX_POSITIONAL_ARGS = 3  # Should match value in pyproject.toml


P = ParamSpec('P')
T = TypeVar('T')


@overload
def _deprecate_positional_args(
    func: Callable[P, T],
    *,
    version: tuple[int, int] = ...,
    allowed: list[str] | None = ...,
    n_allowed: int = ...,
) -> Callable[P, T]: ...
@overload
def _deprecate_positional_args(
    *, version: tuple[int, int] = ..., allowed: list[str] | None = ..., n_allowed: int = ...
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...
def _deprecate_positional_args(
    func: Callable[..., T] | None = None,
    *,
    version: tuple[int, int] = (0, 50),
    allowed: list[str] | None = None,
    n_allowed: int | None = None,
) -> Callable[..., T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Use a decorator to deprecate positional arguments.

    Parameters
    ----------
    func : callable, default=None
        Function to check arguments on.

    version : tuple[int, int], default: (0, 50)
        The version (major, minor) when positional arguments will result in RuntimeError.

    allowed : list[str], optional
        List of argument names which are allowed to be positional. This value is limited
        based on rule PLR0917.

    n_allowed : int, optional
        Override the number of allowed positional arguments to this value.

    """

    def _inner_deprecate_positional_args(f: Callable[P, T]) -> Callable[P, T]:
        def qualified_name() -> str:
            return f.__qualname__ if hasattr(f, '__qualname__') else f.__name__

        decorator_name = _deprecate_positional_args.__name__
        sig = inspect.signature(f)
        param_names = list(sig.parameters)

        # Validate n_allowed itself
        if n_allowed:
            if n_allowed <= _MAX_POSITIONAL_ARGS:
                msg = (
                    f'In decorator {decorator_name!r} for function {qualified_name()!r}:\n'
                    f'`n_allowed` must be greater than {_MAX_POSITIONAL_ARGS} for it to be useful.'
                )
                raise ValueError(msg)
            n_allowed_ = n_allowed
        else:
            n_allowed_ = _MAX_POSITIONAL_ARGS

        if allowed is not None:
            # Validate input type
            if not isinstance(allowed, list):
                msg = (  # type: ignore[unreachable]
                    f'In decorator {decorator_name!r} for function {qualified_name()!r}:\n'
                    f'Allowed arguments must be a list, got {type(allowed)}.'
                )
                raise TypeError(msg)

            # Validate number of allowed args
            if len(allowed) > n_allowed_:
                msg = (
                    f'In decorator {decorator_name!r} for function {qualified_name()!r}:\n'
                    f'A maximum of {n_allowed_} positional arguments are allowed.\n'
                    f'Got {len(allowed)}: {allowed}'
                )
                raise ValueError(msg)

            # Validate allowed against actual parameter names
            for name in allowed:
                if name not in param_names:
                    msg = (
                        f'Allowed positional argument {name!r} in decorator '
                        f'{decorator_name!r}\n'
                        f'is not a parameter of function {qualified_name()!r}.'
                    )
                    raise ValueError(msg)

            # Check that allowed args appears in the same order as in the signature
            sig_allowed = [name for name in param_names if name in allowed]
            if sig_allowed != allowed:
                msg = (
                    f'The `allowed` list {allowed} in decorator {decorator_name!r} is not in the\n'
                    f'same order as the parameters in {qualified_name()!r}.\n'
                    f'Expected order: {sig_allowed}.'
                )
                raise ValueError(msg)

            # Check that allowed args are not already kwonly
            for name in allowed:
                if sig.parameters[name].kind == Parameter.KEYWORD_ONLY:
                    msg = (
                        f'Parameter {name!r} in decorator {decorator_name!r} is already '
                        f'keyword-only\nand should be removed from the allowed list.'
                    )
                    raise ValueError(msg)

        # Check if the decorator is even needed at all
        n_positional = 0
        for name in param_names:
            if name not in ['cls', 'self'] and sig.parameters[name].kind in [
                Parameter.POSITIONAL_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
            ]:
                n_positional += 1
        actual_n_allowed = len(allowed) if allowed else 0
        if n_positional <= actual_n_allowed:
            msg = (
                f'Function {qualified_name()!r} has {actual_n_allowed} positional arguments, '
                f'which is less than or equal to the\nmaximum number of allowed positional '
                f'arguments ({n_allowed_}).\nThis decorator is not necessary and can be removed.'
            )
            raise RuntimeError(msg)

        # Raise error post-deprecation
        if version_info >= version:
            # Construct expected positional args and signature
            new_parameters = []
            max_args_to_print = actual_n_allowed + 2
            cls_or_self = 'cls' in param_names or 'self' in param_names
            max_args_to_print = (max_args_to_print + 1) if cls_or_self else max_args_to_print
            has_too_many_to_print = False
            for i, name in enumerate(param_names):
                if i > max_args_to_print:
                    has_too_many_to_print = True
                    break
                if name in ['cls', 'self', *(allowed or [])]:
                    current_kind = sig.parameters[name].kind
                    new_kind = (
                        current_kind
                        if current_kind != Parameter.KEYWORD_ONLY
                        else Parameter.KEYWORD_ONLY
                    )
                    new_parameters.append(Parameter(name, kind=new_kind))
                else:
                    new_parameters.append(Parameter(name, kind=Parameter.KEYWORD_ONLY))

            signature_string = f'{qualified_name()}{Signature(new_parameters)}'
            if has_too_many_to_print:
                # Replace ending bracket with ellipses
                signature_string = f'{signature_string[:-1]}, ...)'

            # Get source file and line number
            file = Path(inspect.getfile(f)).as_posix()
            lineno = inspect.getsourcelines(f)[1]
            location = f'{file}:{lineno}'

            msg = (
                f'Positional arguments are no longer allowed in {qualified_name()!r}.\n'
                f'Update the function signature at:\n'
                f'{location} to enforce keyword-only args:\n'
                f'    {signature_string}\n'
                f'and remove the {decorator_name!r} decorator.'
            )
            raise RuntimeError(msg)

        @wraps(f)
        def inner_f(*args: P.args, **kwargs: P.kwargs) -> T:
            passed_positional_names = param_names[: len(args)]

            # Exclude allowed ones
            if allowed:
                offending_args = [name for name in passed_positional_names if name not in allowed]
            else:
                offending_args = passed_positional_names

            if 'self' in offending_args:
                offending_args.remove('self')
            if 'cls' in offending_args:
                offending_args.remove('cls')

            if offending_args:
                # Craft a message to print a warning or raise an error
                if len(offending_args) == 1:
                    a = ' a '
                    s = ''
                    this = 'this'
                else:
                    a = ' '
                    s = 's'
                    this = 'these'

                if version_info < version:
                    # Print warning
                    version_str = '.'.join(map(str, version))
                    arg_list = ', '.join(f'{a!r}' for a in offending_args)
                    stack_level = 3

                    def call_site() -> str:
                        # Get location where the function is called
                        frame = inspect.stack()[stack_level]
                        file = Path(frame.filename).as_posix()
                        return f'{file}:{frame.lineno}'

                    def warn_positional_args() -> None:
                        from pyvista.core.errors import PyVistaDeprecationWarning  # noqa: PLC0415

                        msg = (
                            f'\n{call_site()}: '
                            f'Argument{s} {arg_list} must be passed as{a}keyword argument{s} '
                            f'to function {qualified_name()!r}.\n'
                            f'From version {version_str}, passing {this} as{a}positional '
                            f'argument{s} will result in a TypeError.'
                        )
                        warn_external(msg, PyVistaDeprecationWarning)

                    warn_positional_args()

            return f(*args, **kwargs)

        return inner_f

    if func is not None:
        return _inner_deprecate_positional_args(func)
    return _inner_deprecate_positional_args
