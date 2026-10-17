"""Test the pre-commit hooks"""

from __future__ import annotations

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
