from __future__ import annotations

from functools import wraps
import inspect
import os
from pathlib import Path
from typing import Callable
from typing import TypeVar
from typing import overload
import warnings

from typing_extensions import ParamSpec

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
    n_allowed: int = _MAX_POSITIONAL_ARGS,
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
        from pyvista._version import version_info

        def qualified_name() -> str:
            return f.__qualname__ if hasattr(f, '__qualname__') else f.__name__

        decorator_name = _deprecate_positional_args.__name__
        sig = inspect.signature(f)
        param_names = list(sig.parameters)

        # Raise error post-deprecation
        if version_info >= version:
            # Construct expected positional args and signature
            positional_args = (
                ['self'] if 'self' in param_names else ['cls'] if 'cls' in param_names else []
            )
            if allowed is not None:
                for name in allowed:
                    positional_args.append(name)  # noqa: PERF402
            new_signature = f'{qualified_name()}({", ".join(positional_args)}, *, ...)'

            # Get source file and line number
            file = Path(os.path.relpath(inspect.getfile(f), start=os.getcwd())).as_posix()  # noqa: PTH109
            lineno = inspect.getsourcelines(f)[1]
            location = f'{file}:{lineno}'

            msg = (
                f'Positional arguments are no longer allowed in {qualified_name()!r}.\n'
                f'Update the function signature at:\n'
                f'{location} to:\n'
                f'    {new_signature}\n'
                f'and remove the {decorator_name!r} decorator.'
            )
            raise RuntimeError(msg)

        if allowed is not None:
            # Validate input type
            if not isinstance(allowed, list):
                msg = (  # type: ignore[unreachable]
                    f'In decorator {decorator_name} for function {qualified_name()!r}:\n'
                    f'Allowed arguments must be a list, got {type(allowed)}.'
                )
                raise TypeError(msg)

            # Validate number of allowed args
            if len(allowed) > n_allowed:
                msg = (
                    f'In decorator {decorator_name!r} for function {qualified_name()!r}:\n'
                    f'A maximum of {n_allowed} positional arguments are allowed.\n'
                    f'Got {len(allowed)}: {allowed}'
                )
                raise ValueError(msg)

            # Validate `allowed` against actual parameter names
            for name in allowed:
                if name not in param_names:
                    msg = (
                        f'Allowed positional argument {name!r} in decorator '
                        f'{decorator_name!r}\n'
                        f'is not a parameter of function {qualified_name()!r}.'
                    )
                    raise ValueError(msg)

            # Check that `allowed` appears in the same order as in the signature
            sig_allowed = [name for name in param_names if name in allowed]
            if sig_allowed != allowed:
                msg = (
                    f'The `allowed` list {allowed} in decorator {decorator_name!r} is not in the\n'
                    f'same order as the parameters in {qualified_name()!r}.\n'
                    f'Expected order: {sig_allowed}.'
                )
                raise ValueError(msg)

        @wraps(f)
        def inner_f(*args: P.args, **kwargs: P.kwargs) -> T:
            from pyvista.core.errors import PyVistaDeprecationWarning

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
                        file = Path(os.path.relpath(frame.filename, start=os.getcwd())).as_posix()  # noqa: PTH109
                        return f'{file}:{frame.lineno}'

                    def warn_positional_args() -> None:
                        msg = (
                            f'\n{call_site()}: '
                            f'Argument{s} {arg_list} must be passed as{a}keyword argument{s} '
                            f'to function {qualified_name()!r}.\n'
                            f'From version {version_str}, passing {this} as{a}positional '
                            f'argument{s} will result in a TypeError.'
                        )
                        warnings.warn(msg, PyVistaDeprecationWarning, stacklevel=stack_level)

                    warn_positional_args()

            return f(*args, **kwargs)

        return inner_f

    if func is not None:
        return _inner_deprecate_positional_args(func)
    return _inner_deprecate_positional_args
