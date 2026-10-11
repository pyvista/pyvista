"""Test the pre-commit hooks"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope='session')
def pre_commit_config(request: pytest.FixtureRequest):
    with (request.config.rootpath / '.pre-commit-config.yaml').open() as f:
        return yaml.safe_load(f)


def test_warnings_converter(
    tmp_path: Path,
    pre_commit_config: dict,
    request: pytest.FixtureRequest,
):
    test = """\
    import warnings

    warnings.warn("foo")
    warnings.warn("foo", UserWarning)
    warnings.warn("foo", UserWarning, 1)
    warnings.warn("foo", UserWarning, stacklevel=1)
    warnings.warn("foo", category=UserWarning, stacklevel=1)
    warnings.warn(message="foo", category=UserWarning, stacklevel=1)
    warnings.warn(category=UserWarning, stacklevel=1, message="foo")
    """

    if sys.version_info[:2] >= (3, 12):
        test += """
    warnings.warn(category=UserWarning, stacklevel=1, message="foo", source='bar', skip_file_prefixes=('',))
    """  # noqa: E501

    with (file := (tmp_path / 'file.py')).open('w') as f:
        f.write(textwrap.dedent(test))

    local = next(v for v in pre_commit_config['repos'] if v['repo'] == 'local')
    warning_hook = next(v for v in local['hooks'] if v['id'] == 'warn_external')
    cml = warning_hook['entry']

    ret = subprocess.run(
        [sys.executable, *shlex.split(cml)[1:], str(file.absolute())],
        check=True,
        cwd=request.config.rootpath,
    )
    assert ret.returncode == 0

    with file.open('r') as f:
        lines = f.readlines()

    expected = """\
        from pyvista._warn_external import warn_external

        warn_external("foo")
        warn_external("foo", UserWarning)
        warn_external("foo", UserWarning)
        warn_external("foo", UserWarning)
        warn_external("foo", category=UserWarning)
        warn_external(message="foo", category=UserWarning)
        warn_external(message="foo", category=UserWarning)
        """

    if sys.version_info[:2] >= (3, 12):
        expected += """
        warn_external(message="foo", category=UserWarning)
        """

    assert textwrap.dedent(expected) == ''.join(lines)


def _run_private_class_hook(
    source: str,
    tmp_path: Path,
    pre_commit_config: dict,
    request: pytest.FixtureRequest,
    *,
    encoding: str = 'utf-8',
) -> tuple[int, str]:
    """Write ``source`` to a file and return the hook's exit code and output."""
    file = tmp_path / 'file.py'
    file.write_text(textwrap.dedent(source), encoding=encoding)

    local = next(v for v in pre_commit_config['repos'] if v['repo'] == 'local')
    hook = next(v for v in local['hooks'] if v['id'] == 'private-class-docstrings')

    ret = subprocess.run(
        [sys.executable, *shlex.split(hook['entry'])[1:], str(file.absolute())],
        capture_output=True,
        text=True,
        check=False,
        cwd=request.config.rootpath,
    )
    return ret.returncode, ret.stdout


def test_private_class_docstrings_reports(
    tmp_path: Path, pre_commit_config: dict, request: pytest.FixtureRequest
):
    code, out = _run_private_class_hook(
        '''\
        class _Private:
            def undocumented(self): ...

            @property
            def undocumented_property(self): ...

            def _private_method(self): ...

            @documented.setter
            def documented(self, value): ...

            @overload
            def stub(self) -> int: ...

            def documented_method(self):
                """Documented."""

        class Public:
            def undocumented(self): ...
        ''',
        tmp_path,
        pre_commit_config,
        request,
    )
    assert code == 1
    # The private class itself, and only its two undocumented public members
    assert "class '_Private' is private, so it needs a docstring" in out
    assert "public 'undocumented' is in private class '_Private'" in out
    assert "public 'undocumented_property' is in private class '_Private'" in out
    # Exempt: private members, setters, overload stubs, documented members
    assert '_private_method' not in out
    assert "'documented'" not in out
    assert "'stub'" not in out
    assert "'documented_method'" not in out
    # Ruff's D102 already covers a public class
    assert "'Public'" not in out


def test_private_class_docstrings_nested_and_clean(
    tmp_path: Path, pre_commit_config: dict, request: pytest.FixtureRequest
):
    code, out = _run_private_class_hook(
        '''\
        class _Private:
            """Documented."""

            class Nested:
                """Documented."""

                def member(self):
                    """Documented."""

        class Public:
            class _AlsoPrivate:
                """Documented."""
        ''',
        tmp_path,
        pre_commit_config,
        request,
    )
    assert code == 0, out
    assert out == ''


def test_private_class_docstrings_nested_class_needs_docstring(
    tmp_path: Path, pre_commit_config: dict, request: pytest.FixtureRequest
):
    code, out = _run_private_class_hook(
        '''\
        class _Private:
            """Documented."""

            class Nested: ...
        ''',
        tmp_path,
        pre_commit_config,
        request,
    )
    assert code == 1
    assert "class 'Nested' is in private class '_Private'" in out


def test_private_class_docstrings_reads_non_ascii(
    tmp_path: Path, pre_commit_config: dict, request: pytest.FixtureRequest, monkeypatch
):
    """The hook must not depend on the locale's preferred encoding."""
    monkeypatch.setenv('PYTHONUTF8', '0')
    monkeypatch.setenv('PYTHONCOERCECLOCALE', '0')
    monkeypatch.setenv('LC_ALL', 'C')
    code, out = _run_private_class_hook(
        '''\
        class _Private:
            """Documented — with an em dash."""
        ''',
        tmp_path,
        pre_commit_config,
        request,
    )
    assert code == 0, out


def _run_sentence_spacing_hook(
    relpath: str,
    source: str,
    tmp_path: Path,
    pre_commit_config: dict,
    request: pytest.FixtureRequest,
) -> tuple[int, str]:
    """Write ``source`` to ``relpath`` and return the hook's exit code and output."""
    file = tmp_path / relpath
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(textwrap.dedent(source), encoding='utf-8')

    local = next(v for v in pre_commit_config['repos'] if v['repo'] == 'local')
    hook = next(v for v in local['hooks'] if v['id'] == 'sentence-spacing')

    ret = subprocess.run(
        [sys.executable, *shlex.split(hook['entry'])[1:], relpath],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(request.config.rootpath)},
    )
    return ret.returncode, ret.stdout


@pytest.mark.parametrize(
    ('relpath', 'source', 'line'),
    [
        pytest.param('doc/page.rst', 'Plot the mesh.  Then close it.\n', 1, id='rst'),
        pytest.param('doc/page.rst', 'See ``pad``.  Then close it.\n', 1, id='after-code'),
        pytest.param('doc/page.rst', 'Plot it.  ``show`` opens it.\n', 1, id='before-code'),
        pytest.param('doc/page.rst', "Use it.  'all' needs all.\n", 1, id='before-quote'),
        pytest.param('doc/page.rst', 'See (the docs.)  Then go.\n', 1, id='after-paren'),
        pytest.param('doc/page.rst', 'Plot it.  (Then close it.)\n', 1, id='before-paren'),
        pytest.param('doc/page.rst', 'Added in\n2024.  Then changed.\n', 2, id='wrapped-number'),
        pytest.param('doc/page.rst', '.. note::\n\n   Check it.  Then go.\n', 3, id='note-body'),
        pytest.param(
            'doc/page.rst',
            'Example::\n\n    code.  Here\n\nBack in prose.  Then go.\n',
            5,
            id='after-literal-block',
        ),
        pytest.param(
            'pyvista/mod.py',
            '''\
            def f():
                """Summary.

                Plot the mesh.  Then close it.
                """
            ''',
            4,
            id='function-docstring',
        ),
        pytest.param(
            'pyvista/mod.py',
            '''\
            def f():
                """Summary.

                Example::

                    code"""


            class C:
                def g(self):
                    """Plot the mesh.  Then close it."""
            ''',
            11,
            id='docstring-after-literal-block',
        ),
        pytest.param(
            'examples/ex.py',
            '''\
            """Example.

            Plot the mesh.  Then close it.
            """
            ''',
            3,
            id='example-module-docstring',
        ),
        pytest.param(
            'examples/ex.py',
            '''\
            """Example."""

            # %%
            # Plot the mesh.  Then close it.
            ''',
            4,
            id='gallery-text',
        ),
        pytest.param(
            'examples/ex.py',
            '''\
            """Example."""

            ###############################################################################
            # Plot the mesh.  Then close it.
            ''',
            4,
            id='gallery-separator',
        ),
    ],
)
def test_sentence_spacing_flags(
    relpath: str,
    source: str,
    line: int,
    tmp_path: Path,
    pre_commit_config: dict,
    request: pytest.FixtureRequest,
):
    code, out = _run_sentence_spacing_hook(relpath, source, tmp_path, pre_commit_config, request)
    assert code == 1
    assert len(out.splitlines()) == 1, out
    assert out.startswith(f'{relpath}:{line}: ')


@pytest.mark.parametrize(
    ('relpath', 'source'),
    [
        pytest.param('doc/page.rst', '#.  First item\n#.  Second item\n', id='auto-enumerator'),
        pytest.param('doc/page.rst', '1.  First item\na.  Lettered item\n', id='enumerator'),
        pytest.param('doc/page.rst', 'Plot the mesh. Then close it.\n', id='one-space'),
        pytest.param('doc/page.rst', 'Wait for it...  Then go.\n', id='ellipsis'),
        pytest.param('doc/page.rst', 'Example::\n\n    code.  Here\n', id='literal-block'),
        pytest.param(
            'doc/page.rst',
            '.. code-block:: python\n\n    x = 1.  # Then\n',
            id='code-block',
        ),
        pytest.param(
            'doc/page.rst',
            '#.  First item,\n    continued.\n#.  Second item.\n',
            id='enumerator-continued',
        ),
        pytest.param(
            'pyvista/mod.py',
            '''\
            def f():
                """Run it.

                >>> print('Done.  Then go.')
                Done.  Then go.
                """
                x = """A.  B"""
                # A comment.  Not prose.
                msg = 'An error.  Not prose.'
            ''',
            id='code-and-doctest',
        ),
        pytest.param(
            'examples/ex.py',
            '''\
            """Example."""

            # %%
            # Gallery text.

            # A code comment.  Not prose.


            def f():
                """Shown as code in the gallery.  Not prose."""
            ''',
            id='example-code-cell',
        ),
    ],
)
def test_sentence_spacing_passes(
    relpath: str,
    source: str,
    tmp_path: Path,
    pre_commit_config: dict,
    request: pytest.FixtureRequest,
):
    code, out = _run_sentence_spacing_hook(relpath, source, tmp_path, pre_commit_config, request)
    assert code == 0, out
    assert out == ''
