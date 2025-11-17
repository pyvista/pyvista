"""Enforce warnings style.

Python script to enforce using the custom `warn_external` function instead of
plain `warnings.warn`, to allow for dynamic stacklevel value.
"""

from __future__ import annotations

from inspect import Parameter
from inspect import Signature
import sys

import libcst as cst
from libcst.codemod import VisitorBasedCodemodCommand
from libcst.codemod.visitors import AddImportsVisitor
from libcst.codemod.visitors import RemoveImportsVisitor
import libcst.matchers as m


def needs_replace(node: cst.Call) -> bool:  # noqa: D103
    return m.matches(
        node.func,
        m.Attribute(value=m.Name('warnings'), attr=m.Name('warn')) | m.Name('warn'),
    )


def get_args_kwargs(args: tuple[cst.Arg]) -> tuple[list[cst.Arg], dict[str, cst.Arg]]:  # noqa: D103
    a = [a for a in args if a.keyword is None]
    kw = {kw.value: a for a in args if (kw := a.keyword) is not None}
    return a, kw


# Need to manually build the `warnings.warn` signature because `inspect.signature`
# is raising an error for some builtins https://github.com/python/cpython/issues/123473
_WARN_PARAMS = [
    Parameter(
        name='message',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
    ),
    Parameter(name='category', kind=Parameter.POSITIONAL_OR_KEYWORD, default=None),
    Parameter(name='stacklevel', kind=Parameter.POSITIONAL_OR_KEYWORD, default=1),
    Parameter(name='source', kind=Parameter.POSITIONAL_OR_KEYWORD, default=None),
]

if sys.version_info[:2] >= (3, 12):
    _WARN_PARAMS.append(
        Parameter(name='skip_file_prefixes', kind=Parameter.KEYWORD_ONLY, default=())
    )


_WARN_SIGNATURE = Signature(parameters=_WARN_PARAMS)


class ConvertWarningsToExternal(VisitorBasedCodemodCommand):
    """Class responsible to parse/modify the syntax tree if warnings calls are found."""

    def leave_Call(  # noqa: D102, N802
        self,
        original_node: cst.Call,
        updated_node: cst.Call,
    ) -> cst.Call:
        if needs_replace(original_node):
            AddImportsVisitor.add_needed_import(
                self.context,
                module='pyvista._warn_external',
                obj='warn_external',
            )
            RemoveImportsVisitor.remove_unused_import(self.context, 'warnings')

            # Remove stacklevel from call. Relies on the fact that pos args are
            # passed before keyword args
            a, kw = get_args_kwargs(original_node.args)
            bound = _WARN_SIGNATURE.bind(*a, **kw)
            b_arguments = bound.arguments
            b_arguments.pop('stacklevel', None)

            args: list[cst.Arg] = list(b_arguments.values())

            # Remove trailing comma
            args[-1] = args[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

            return updated_node.with_changes(
                func=cst.Name('warn_external'),
                args=args,
            )
        return updated_node
