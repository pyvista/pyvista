"""Tests for the ``pyvista build`` command and its interactive wizard."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
from rich.console import Console

from pyvista._build import BuildConfig
from pyvista._build import Builder
from pyvista._build import phases
from pyvista._build import tools

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _config(tmp_path: Path, **kwargs) -> BuildConfig:
    defaults = dict(backend='vtk', build_dir=tmp_path / 'vtk-build', dry_run=True, os_name='linux')
    defaults.update(kwargs)
    return BuildConfig(**defaults)


@pytest.fixture
def no_tools(mocker: MockerFixture):
    """Pretend every tool is present and pip is available."""
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(tools, 'command_succeeds', return_value=True)
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])


@pytest.mark.usefixtures('no_tools')
def test_interactive_source_prompt(tmp_path: Path, mocker: MockerFixture):
    """The clone prompt takes a fork URL, a local checkout, or a plain yes or no."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(_config(tmp_path, dry_run=False, interactive=True), console)
    git = mocker.patch.object(builder, 'git', return_value='')
    answers = iter(['nonsense', 'https://example.com/me/vtk.git', 'mybranch'])
    mocker.patch.object(phases.Prompt, 'ask', side_effect=lambda *_, **__: next(answers))
    builder.phase_source()
    out = console.export_text()
    assert 'neither a git checkout nor a git URL' in out
    assert 'To build a fork instead' in out
    assert git.call_args_list[0].args[3] == 'https://example.com/me/vtk.git'
    assert git.call_args_list[1].args[-1] == 'mybranch'

    fork = tmp_path / 'fork'
    (fork / '.git').mkdir(parents=True)
    builder = Builder(_config(tmp_path, dry_run=False, interactive=True), Console(quiet=True))
    mocker.patch.object(phases.Prompt, 'ask', return_value=str(fork))
    builder.phase_source()
    assert builder.config.source == fork.resolve()
    assert builder.manifest.name == 'vtk'

    builder = Builder(_config(tmp_path, dry_run=False, interactive=True), Console(quiet=True))
    mocker.patch.object(phases.Prompt, 'ask', return_value='n')
    with pytest.raises(phases.BuildError, match='Nothing to build'):
        builder.phase_source()


@pytest.mark.usefixtures('no_tools')
def test_interactive_reuse_prompt(tmp_path: Path, mocker: MockerFixture):
    """An existing checkout in the build tree is confirmed, replaced, or swapped for another."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    cfg = _config(tmp_path, dry_run=False, interactive=True)
    (cfg.src_dir / '.git').mkdir(parents=True)
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(cfg, console)
    git = mocker.patch.object(builder, 'git', return_value='')
    mocker.patch.object(phases.Prompt, 'ask', return_value='y')
    builder.phase_source()
    assert 'Reusing the checkout' in console.export_text()
    assert not git.called

    fork = tmp_path / 'fork'
    (fork / '.git').mkdir(parents=True)
    builder = Builder(_config(tmp_path, dry_run=False, interactive=True), Console(quiet=True))
    mocker.patch.object(phases.Prompt, 'ask', return_value=str(fork))
    builder.phase_source()
    assert builder.config.source == fork.resolve()

    builder = Builder(_config(tmp_path, dry_run=False, interactive=True), Console(quiet=True))
    git = mocker.patch.object(builder, 'git', return_value='')
    mocker.patch.object(
        phases.Prompt, 'ask', side_effect=['https://example.com/me/vtk.git', 'dev']
    )
    mocker.patch.object(phases.Confirm, 'ask', return_value=False)
    with pytest.raises(phases.BuildError, match='Keeping the existing checkout'):
        builder.phase_source()
    assert (cfg.src_dir / '.git').exists()

    cfg_url = _config(tmp_path, dry_run=False, interactive=True, url='https://example.com/f.git')
    builder = Builder(cfg_url, Console(quiet=True))
    mocker.patch.object(builder, 'git', return_value='https://example.com/vtk.git')
    prompt = mocker.patch.object(phases.Prompt, 'ask', return_value='y')
    with pytest.raises(phases.BuildError, match='--url is not used'):
        builder.phase_source()
    assert not prompt.called

    builder = Builder(_config(tmp_path, dry_run=False, interactive=True), Console(quiet=True))
    git = mocker.patch.object(builder, 'git', return_value='')
    mocker.patch.object(
        phases.Prompt, 'ask', side_effect=['https://example.com/me/vtk.git', 'dev']
    )
    mocker.patch.object(phases.Confirm, 'ask', return_value=True)
    builder.phase_source()
    assert not cfg.src_dir.exists()
    assert git.call_args_list[0].args[3] == 'https://example.com/me/vtk.git'
    assert git.call_args_list[1].args[-1] == 'dev'


@pytest.mark.usefixtures('no_tools')
def test_source_dir_that_is_not_a_checkout(tmp_path: Path, mocker: MockerFixture):
    """A non-empty ``<build-dir>/src`` without git is kept; an empty one is cloned into."""
    cfg = _config(tmp_path, dry_run=False)
    cfg.src_dir.mkdir(parents=True)
    (cfg.src_dir / 'pyproject.toml').write_text('')
    builder = Builder(cfg, Console(quiet=True))
    git = mocker.patch.object(builder, 'git', return_value='')
    with pytest.raises(phases.BuildError, match='exists but is not a git checkout'):
        builder.phase_source()
    assert (cfg.src_dir / 'pyproject.toml').exists()
    assert not git.called
    (cfg.src_dir / 'pyproject.toml').unlink()
    builder.phase_source()
    assert git.call_args_list[0].args[0] == 'clone'


@pytest.mark.usefixtures('no_tools')
def test_url_with_existing_checkout_stops(tmp_path: Path, mocker: MockerFixture):
    """A ``--url`` the existing checkout was not cloned from stops before anything runs."""
    cfg = _config(tmp_path, url='https://example.com/f.git')
    (cfg.src_dir / '.git').mkdir(parents=True)
    builder = Builder(cfg, Console(quiet=True))
    git = mocker.patch.object(builder, 'git', return_value='https://example.com/vtk.git')
    with pytest.raises(
        phases.BuildError, match=r'checkout of https://example\.com/vtk\.git already'
    ):
        builder.plan()
    with pytest.raises(phases.BuildError, match='--url is not used'):
        builder.phase_source()
    for interactive in (False, True):
        mocker.patch.object(phases.sys.stdin, 'isatty', return_value=interactive)
        cfg.dry_run, cfg.interactive = not interactive, interactive
        builder.console = Console(record=True, width=4000, force_terminal=False)
        with pytest.raises(phases.BuildError, match='--url is not used'):
            builder.build()
        assert 'steps' not in builder.console.export_text()
    git.side_effect = subprocess.CalledProcessError(2, 'git')
    with pytest.raises(phases.BuildError, match='checkout of another URL'):
        builder.plan()
    git.side_effect = None
    for origin in (cfg.url, 'https://example.com/f', 'https://example.com/f/'):
        git.return_value = origin
        assert builder._unused_url() is None
    cfg.url = 'https://example.com/f/'
    git.return_value = 'https://example.com/f.git'
    assert builder._unused_url() is None


def test_default_ref_without_shipped_manifest(tmp_path: Path):
    builder = Builder(_config(tmp_path, backend='demo'), Console(quiet=True))
    assert builder._default_ref() is None


def test_fork_ref_defaults_to_its_manifest(tmp_path: Path, mocker: MockerFixture):
    """Enter at the ref prompt leaves the ref to the cloned fork's own manifest."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    cfg = _config(tmp_path, backend='cvista', interactive=True, dry_run=False)
    builder = Builder(cfg, Console(quiet=True))
    ask = mocker.patch.object(
        phases.Prompt, 'ask', side_effect=['https://example.com/me/cvista.git', '']
    )
    assert builder._ask_source('Clone it?', stop='stopped') == (None, True)
    assert builder.config.ref is None
    assert "default ref in the fork's manifest" in ask.call_args.args[0]
    assert ask.call_args.kwargs['show_default'] is False


@pytest.mark.usefixtures('no_tools')
def test_build_tree_from_another_source(tmp_path: Path, mocker: MockerFixture):
    """A build tree configured from another checkout is replaced only on a yes."""
    cfg = _config(tmp_path, dry_run=False, yes=True)
    cfg.cmake_dir.mkdir(parents=True)
    other = tmp_path / 'other-vtk'
    cache = cfg.cmake_dir / 'CMakeCache.txt'
    cache.write_text(f'# comment\nCMAKE_HOME_DIRECTORY:INTERNAL={other}\n')
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(cfg, console)
    run = mocker.patch.object(builder, 'run')
    builder.phase_source()
    builder.phase_tools()
    builder.phase_groups()
    with pytest.raises(phases.BuildError, match='Pass --clean to replace it'):
        builder.phase_configure()
    assert f'was configured from {other}' in console.export_text()
    assert cache.exists()

    mocker.patch.object(builder, 'confirm', return_value=True)
    builder.phase_configure()
    assert not cache.exists()
    assert run.call_args.args[0][:2] == ['cmake', '-Wno-dev']

    cfg.cmake_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={cfg.src_dir}\n')
    assert builder._configured_source() == cfg.src_dir
    cache.write_text('CMAKE_BUILD_TYPE:STRING=Release\n')
    assert builder._configured_source() is None

    cache.write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={other}\n')
    builder.config.dry_run = True
    builder.phase_configure()
    assert cache.exists()


def test_existing_checkout_declined(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'src'
    src.mkdir()
    builder = Builder(_config(tmp_path, dry_run=False, ref='v1'), Console(quiet=True))
    git = mocker.patch.object(
        builder, 'git', side_effect=lambda *a, **_: a[-1] if a[0] == 'rev-parse' else ''
    )
    mocker.patch.object(builder, 'confirm', return_value=False)
    builder._existing_checkout(src)
    assert not any('checkout' in call.args for call in git.call_args_list)


def _walk_answers(smp: str, answers: list[str]):
    """Return a ``Prompt.ask`` stand-in: ``smp`` for the smp option, else the next answer."""
    queue = iter(answers)
    return lambda question='', **_: smp if question.startswith('smp') else next(queue)
