"""The build itself: source, tools, groups, configure, build, wheel, repair, install."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

from rich.console import Console
from rich.prompt import Confirm
from rich.prompt import Prompt
from rich.table import Table

from . import manylinux as _manylinux
from . import tools as _tools
from .manifest import MANIFEST_FILENAME
from .manifest import ManifestError
from .manifest import cmake_entries
from .manifest import option_environment
from .manifest import resolve_groups
from .manifest import resolve_options
from .registry import BACKENDS
from .registry import find_manifest
from .registry import known_backends
from .registry import shipped_manifest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .manifest import Manifest

STATE_FILENAME = '.pyvista-build.json'
_TAG_LIKE = re.compile(r'v?\d+\.\d+')
_join = subprocess.list2cmdline if os.name == 'nt' else shlex.join
_NOISY_BUILD_LINE = re.compile(
    r': warning C\d+:|: note:|^\s+(with|\[|\])\s*$|^\s+[A-Za-z_][A-Za-z0-9_]*='
)


class BuildError(Exception):
    """A step cannot continue; the message tells the user what to do."""


@dataclass
class BuildConfig:
    """Everything the command line decided."""

    backend: str
    build_dir: Path
    source: Path | None = None
    url: str | None = None
    ref: str | None = None
    with_: list[str] = field(default_factory=list)
    without: list[str] = field(default_factory=list)
    all_: bool = False
    minimal: bool = False
    build_type: str = 'Release'
    jobs: int | None = None
    defines: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    clean: bool = False
    reconfigure: bool = False
    update: bool = False
    compiler_cache: str = 'auto'
    cache_dir: Path | None = None
    version_suffix: str | None = None
    repair: bool = True
    install: bool = False
    interactive: bool = False
    dry_run: bool = False
    manylinux: str | None = None
    os_name: str = field(default_factory=_tools.host_os)

    @property
    def src_dir(self) -> Path:  # numpydoc ignore=RT01
        """Source checkout: ``--source`` when given, else ``<build-dir>/src``."""
        return self.source if self.source is not None else self.build_dir / 'src'

    @property
    def cmake_dir(self) -> Path:  # numpydoc ignore=RT01
        """CMake build tree."""
        return self.build_dir / 'build'

    @property
    def dist_dir(self) -> Path:  # numpydoc ignore=RT01
        """Where ``bdist_wheel`` writes the raw wheel."""
        return self.cmake_dir / 'dist'

    @property
    def wheelhouse(self) -> Path:  # numpydoc ignore=RT01
        """Where the repaired wheel goes."""
        return self.build_dir / 'wheelhouse'

    @property
    def compiler_cache_dir(self) -> Path:  # numpydoc ignore=RT01
        """Compiler cache directory."""
        return self.cache_dir if self.cache_dir is not None else self.build_dir / 'compiler-cache'

    @property
    def prompting(self) -> bool:  # numpydoc ignore=RT01
        """Whether to ask questions: interactive, on a terminal, and not a dry run."""
        return self.interactive and not self.dry_run and sys.stdin.isatty()


class Builder:
    """Run the phases in order, confirming each one with the user.

    Parameters
    ----------
    config : BuildConfig
        What to build and how.

    console : rich.console.Console, optional
        Console to print to.

    """

    def __init__(self, config: BuildConfig, console: Console | None = None) -> None:
        self.config = config
        self.console = console or Console()
        self.env: dict[str, str] = dict(os.environ)
        scripts = str(Path(sys.executable).parent)
        self.env['PATH'] = f'{scripts}{os.pathsep}{self.env.get("PATH", "")}'
        self._manifest: Manifest | None = None
        self.groups: list[str] = []
        self.option_values: dict[str, bool | int | str] = {}
        self.launcher: _tools.Tool | None = None
        self.wheel: Path | None = None

    @property
    def manifest(self) -> Manifest:  # numpydoc ignore=RT01
        """Manifest of the backend being built."""
        if self._manifest is None:
            msg = 'The manifest is only available after the source phase.'
            raise BuildError(msg)
        return self._manifest

    def step(self, title: str) -> None:
        """Print a phase heading.

        Parameters
        ----------
        title : str
            Heading text.

        """
        self.console.rule(f'[bold]{title}')

    def say(self, message: str) -> None:
        """Print a line of narration.

        Parameters
        ----------
        message : str
            Rich markup to print.

        """
        self.console.print(message)

    def confirm(self, question: str, *, default: bool = True) -> bool:
        """Ask a yes/no question, or take the default when not interactive.

        Parameters
        ----------
        question : str
            Question to ask.

        default : bool, default: True
            Answer taken on Enter, and without ``--interactive`` or under ``--dry-run``.

        Returns
        -------
        bool
            The answer.

        """
        if self.config.prompting:
            return Confirm.ask(question, default=default, console=self.console)
        if self.config.interactive:
            self.say(f'[dim]{question} -> {"yes" if default else "no"} (dry run)[/dim]')
        return default

    def run(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        log: Path | None = None,
        fatal: bool = True,
    ) -> None:
        """Print and run a command; raise :class:`BuildError` when it fails.

        Parameters
        ----------
        cmd : Sequence[str]
            Command and arguments.

        cwd : Path, optional
            Working directory, created when missing.

        env : Mapping[str, str], optional
            Environment entries added to the builder's environment.

        log : Path, optional
            File that receives the full output; the console gets a filtered view.

        fatal : bool, default: True
            Raise on a non-zero exit status instead of only reporting it.

        """
        where = f' (in {cwd})' if cwd is not None else ''
        prefix = ''.join(f'{k}={shlex.quote(v)} ' for k, v in (env or {}).items())
        self.say(f'[cyan]$ {prefix}{_join(cmd)}[/cyan]{where}')
        if self.config.dry_run:
            return
        if cwd is not None:
            cwd.mkdir(parents=True, exist_ok=True)
        full_env = dict(self.env)
        full_env.update(env or {})
        if log is None:
            status = subprocess.run(cmd, cwd=cwd, env=full_env, check=False).returncode
        else:
            status = self._run_logged(cmd, cwd=cwd, env=full_env, log=log)
        if status != 0 and fatal:
            msg = f'Command failed with exit status {status}: {_join(cmd)}'
            raise BuildError(msg)

    def _run_logged(
        self, cmd: Sequence[str], *, cwd: Path | None, env: dict[str, str], log: Path
    ) -> int:
        """Stream a command's output with compiler noise filtered; keep the full log."""
        with (
            log.open('w', encoding='utf-8') as log_file,
            subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors='replace',
            ) as proc,
        ):
            for line in proc.stdout or ():
                log_file.write(line)
                if not _NOISY_BUILD_LINE.search(line):
                    self.console.out(line, end='', highlight=False)
        if proc.returncode != 0:
            self.say(f'[red]Build failed; the full log is in {log}[/red]')
        return proc.returncode

    def git(self, *args: str, cwd: Path | None = None, capture: bool = False) -> str:
        """Run git.

        Parameters
        ----------
        *args : str
            Arguments after ``git``.

        cwd : Path, optional
            Repository to run in.

        capture : bool, default: False
            Return the output instead of streaming it; errors then raise
            :class:`subprocess.CalledProcessError`.

        Returns
        -------
        str
            Stripped output when ``capture`` is set, else an empty string.

        """
        cmd = ['git', *args]
        if capture:
            return subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, check=True
            ).stdout.strip()
        self.run(cmd, cwd=cwd)
        return ''

    @property
    def python(self) -> str:  # numpydoc ignore=RT01
        """Interpreter the wheel is built for."""
        return sys.executable

    def hooks(self, name: str) -> None:
        """Run the manifest's hook commands inside the source tree.

        Parameters
        ----------
        name : str
            Hook name, for example ``'post_build'``.

        """
        cfg = self.config
        for template in self.manifest.hooks.get(name, ()):
            self.run(self._expand(template), cwd=cfg.src_dir)

    def _expand(self, template: str) -> list[str]:
        """Split a manifest command template, then substitute the placeholders per part."""
        cfg = self.config
        fields = {
            'python': self.python,
            'src_dir': cfg.src_dir,
            'build_dir': cfg.cmake_dir,
            'dist_dir': cfg.dist_dir,
            'wheelhouse': cfg.wheelhouse,
        }
        return [part.format(**fields) for part in shlex.split(template)]

    def build(self) -> Path | None:
        """Run every phase.

        Returns
        -------
        Path | None
            The final wheel, or ``None`` on a dry run.

        """
        self.console.print(f'[bold green]Building backend: {self.config.backend}[/bold green]')
        if self.config.manylinux:
            return self.build_in_manylinux()
        self.say(
            'This walks through eight steps: source, tools, build options, configure, build, '
            'wheel, repair, and install. Each step explains what it will do and asks before '
            'running any command.'
        )
        if not self.confirm('Proceed?'):
            msg = 'Stopped before the first step.'
            raise BuildError(msg)
        self.phase_source()
        self.phase_tools()
        self.phase_groups()
        self.phase_configure()
        self.phase_build()
        self.phase_wheel()
        self.phase_repair()
        self.phase_install()
        self.summary()
        return self.wheel

    def phase_source(self) -> None:
        """Clone or reuse the checkout, then load its manifest."""
        cfg = self.config
        self.step('1/8 Source')
        git = _tools.Tool(
            'git', ('git',), instructions='Install git: https://git-scm.com/downloads'
        )
        if not _tools.detect(git, self.env):
            msg = f'git was not found on PATH. {git.instructions}'
            raise BuildError(msg)

        shipped = shipped_manifest(cfg.backend)
        try:
            self._source_phase(shipped)
        except ManifestError as err:
            raise BuildError(str(err)) from None

    def _source_phase(self, shipped: Path | None) -> None:
        """Body of :meth:`phase_source`; manifest errors propagate to the caller."""
        cfg = self.config
        url = cfg.url or BACKENDS.get(cfg.backend)
        if url is None and shipped is not None:
            url = find_manifest(cfg.backend, None).url
        if cfg.source is not None:
            if not (cfg.source / '.git').exists():
                msg = f'--source {cfg.source} is not a git checkout.'
                raise BuildError(msg)
            self.say(f'Using the existing checkout at [bold]{cfg.source}[/bold].')
            self._existing_checkout(cfg.source)
        elif (cfg.src_dir / '.git').exists():
            self.say(f'Reusing the checkout at [bold]{cfg.src_dir}[/bold]; local edits are kept.')
            self._existing_checkout(cfg.src_dir)
        else:
            if url is None:
                msg = (
                    f'Unknown backend {cfg.backend!r}. Known backends: {known_backends()}. '
                    'Pass --url <git url> or --source <checkout> to build another one.'
                )
                raise BuildError(msg)
            ref = cfg.ref
            if ref is None and shipped is not None:
                ref = find_manifest(cfg.backend, None).default_ref
            self.say(
                f'The source will be cloned from [bold]{url}[/bold] into '
                f'[bold]{cfg.src_dir}[/bold] (blob-less partial clone: history without '
                'file contents until needed).'
            )
            if not self.confirm('Clone it now?'):
                msg = 'Nothing to build without a source checkout.'
                raise BuildError(msg)
            self.git('clone', '--filter=blob:none', url, str(cfg.src_dir))
            if ref is None and not cfg.dry_run:
                ref = find_manifest(cfg.backend, cfg.src_dir).default_ref
            if ref is not None:
                self.git('-c', 'advice.detachedHead=false', 'checkout', ref, cwd=cfg.src_dir)

        if cfg.dry_run and not (cfg.src_dir / MANIFEST_FILENAME).is_file() and shipped is None:
            msg = (
                f'--dry-run needs an existing checkout for {cfg.backend!r}: its '
                f'{MANIFEST_FILENAME} lives in the source tree.'
            )
            raise BuildError(msg)
        source = cfg.src_dir if cfg.src_dir.exists() else None
        self._manifest = find_manifest(cfg.backend, source)
        self.say(f'Manifest: [bold]{self.manifest.path}[/bold]')

    def _existing_checkout(self, src: Path) -> None:
        """Apply ``--update`` / ``--ref`` to a checkout that already exists."""
        cfg = self.config
        if not cfg.update and cfg.ref is None:
            return
        dirty = self.git('status', '--porcelain', cwd=src, capture=True)
        if dirty:
            msg = f'{src} has local changes; commit or stash them before using --ref or --update.'
            raise BuildError(msg)
        if cfg.update:
            self.git('fetch', '--tags', 'origin', cwd=src)
        ref = cfg.ref
        if ref is None:
            ref = find_manifest(cfg.backend, src).default_ref
        if cfg.update or self.confirm(f'Check out {ref!r} in {src}?'):
            self.git('-c', 'advice.detachedHead=false', 'checkout', ref, cwd=src)

    def phase_tools(self) -> None:
        """Check for every tool, offering to install the pip-installable ones."""
        cfg = self.config
        self.step('2/8 Tools')
        tools = _tools.required_tools(cfg.os_name, self.manifest.tools, repair=cfg.repair)
        self._activate_msvc(tools)
        table = Table(show_header=True, header_style='bold purple')
        table.add_column('Tool')
        table.add_column('Found')
        table.add_column('Status')
        missing_pip: list[_tools.Tool] = []
        blocking: list[_tools.Tool] = []
        for tool in tools:
            ok = _tools.detect(tool, self.env)
            found = f'{tool.path} ({tool.version})' if tool.version else (tool.path or '-')
            if ok:
                status = '[green]ok[/green]'
            elif tool.pip:
                status = (
                    '[yellow]missing[/yellow]' if tool.path is None else '[yellow]too old[/yellow]'
                )
                missing_pip.append(tool)
            else:
                status = '[red]missing[/red]'
                blocking.append(tool)
            table.add_row(tool.label, found, status)
        missing_packages = _tools.python_build_packages()
        for name in _tools.PYTHON_BUILD_PACKAGES:
            if name in missing_packages:
                table.add_row(f'{name} (python)', '-', '[yellow]missing[/yellow]')
            else:
                table.add_row(f'{name} (python)', sys.prefix, '[green]ok[/green]')
        table.add_row(*self._compiler_cache_row())
        self.console.print(table)
        if blocking:
            lines = [f'{t.name}: {t.instructions}' for t in blocking]
            msg = 'Install these and run again:\n  ' + '\n  '.join(lines)
            raise BuildError(msg)

        specs = [f'{t.pip}{t.spec}' for t in missing_pip] + missing_packages
        if specs:
            cmd = _tools.pip_install_command(*specs)
            if cmd is None:
                msg = (
                    f'Install {", ".join(specs)} into {sys.prefix} and run again '
                    '(no pip or uv found).'
                )
                raise BuildError(msg)
            self.say(f'Missing Python packages: [bold]{", ".join(specs)}[/bold].')
            if not self.confirm(f'Install them into {sys.prefix}?'):
                msg = 'Cannot build without them.'
                raise BuildError(msg)
            self.run(cmd)
            if not cfg.dry_run:
                bad = [t.label for t in missing_pip if not _tools.detect(t, self.env)]
                if bad:
                    msg = f'Still not usable after install: {", ".join(bad)}.'
                    raise BuildError(msg)

        self._activate_nasm()
        self._activate_compiler_cache()

    def _activate_msvc(self, tools: list[_tools.Tool]) -> None:
        """On Windows, import the MSVC environment when ``cl`` is not already on PATH."""
        if self.config.os_name != 'windows':
            return
        msvc = next(t for t in tools if t.name == 'msvc')
        if _tools.detect(msvc, self.env):
            return
        vcvarsall = _tools.find_vcvarsall()
        if vcvarsall is None:
            return
        self.say(
            f'Visual Studio found; importing its build environment from [bold]{vcvarsall}[/bold].'
        )
        if self.confirm('Use it?') and not self.config.dry_run:
            self.env = _tools.msvc_environment(vcvarsall, env=self.env)

    def _activate_nasm(self) -> None:
        """On Windows, fetch NASM with VTK's own script when it is missing."""
        cfg = self.config
        if cfg.os_name != 'windows':
            return
        nasm = _tools.Tool('nasm', ('nasm',))
        if _tools.detect(nasm, self.env):
            return
        script = cfg.src_dir / '.gitlab' / 'ci' / 'download_nasm.cmake'
        if not script.is_file():
            self.say(
                '[yellow]nasm not found; assembler-optimised codecs will be disabled.[/yellow]'
            )
            return
        target = cfg.src_dir / '.gitlab' / 'nasm'
        self.say(f'nasm not found; VTK can download it into [bold]{target}[/bold].')
        if self.confirm('Download nasm?'):
            self.run(
                ['cmake', '-P', str(script)],
                cwd=cfg.src_dir,
                env={'CMAKE_CONFIGURATION': 'windows'},
            )
            self.env['PATH'] = f'{target}{os.pathsep}{self.env.get("PATH", "")}'

    def _compiler_cache_row(self) -> tuple[str, str, str]:
        """Return the tools-table row for the ``--compiler-cache`` choice."""
        choice = self.config.compiler_cache
        if choice == 'none':
            return 'compiler cache', '-', '[dim]off (--compiler-cache none)[/dim]'
        tool = _tools.compiler_cache_tool(choice, self.config.os_name, self.env)
        if tool is None:
            label = 'ccache/sccache' if choice == 'auto' else choice
            status = (
                '[yellow]missing, optional[/yellow]' if choice == 'auto' else '[red]missing[/red]'
            )
            return f'{label} (compiler cache)', '-', status
        found = f'{tool.path} ({tool.version})' if tool.version else (tool.path or '-')
        return f'{tool.name} (compiler cache)', found, '[green]ok[/green]'

    def _activate_compiler_cache(self) -> None:
        """Pick the compiler launcher and point it at the cache directory."""
        cfg = self.config
        if cfg.compiler_cache == 'none':
            self.say('Compiler cache: off')
            return
        self.launcher = _tools.compiler_cache_tool(cfg.compiler_cache, cfg.os_name, self.env)
        if self.launcher is None:
            hint = _tools.compiler_cache_instructions(cfg.os_name)
            if cfg.compiler_cache != 'auto':
                msg = f'{cfg.compiler_cache} was not found on PATH. Install it: {hint}'
                raise BuildError(msg)
            self.say(
                '[yellow]No ccache/sccache on PATH; rebuilds will be slow. '
                f'Install one: {hint}[/yellow]'
            )
            return
        cache_dir = cfg.compiler_cache_dir
        self.say(f'Compiler cache: [bold]{self.launcher.name}[/bold] in [bold]{cache_dir}[/bold]')
        var = 'SCCACHE_DIR' if self.launcher.name == 'sccache' else 'CCACHE_DIR'
        self.env[var] = str(cache_dir)
        if self.launcher.name == 'ccache':
            self.env.setdefault('CCACHE_COMPILERCHECK', '%compiler% --version')

    def phase_groups(self) -> None:
        """Resolve the groups and options and show them."""
        cfg = self.config
        self.step('3/8 Build options')
        try:
            self.groups = resolve_groups(
                self.manifest,
                cfg.os_name,
                with_=cfg.with_,
                without=cfg.without,
                all_=cfg.all_,
                minimal=cfg.minimal,
            )
            self.option_values = resolve_options(self.manifest, cfg.options)
        except ManifestError as err:
            raise BuildError(str(err)) from None
        self._print_selection()
        self.say(
            'Change the selection with --with/--without/--all/--minimal and --opt NAME=VALUE.'
        )

    def _print_selection(self) -> None:
        """Print the group and option table."""
        table = Table(show_header=True, header_style='bold purple')
        table.add_column('Group / option')
        table.add_column('Value')
        table.add_column('Description')
        for group in self.manifest.groups_for(self.config.os_name):
            on = group.name in self.groups
            value = '[green]on[/green]' if on else '[dim]off[/dim]'
            if group.required:
                value += ' (required)'
            table.add_row(group.name, value, group.help)
        for option in self.manifest.options.values():
            value = str(self.option_values[option.name])
            table.add_row(f'--opt {option.name}', value, option.help)
        self.console.print(table)

    def version_suffix(self) -> str | None:
        """Return the wheel version suffix.

        Returns
        -------
        str | None
            ``--version-suffix`` when given, nothing on a tag, else ``dev0+g<sha>``.

        """
        cfg = self.config
        if cfg.version_suffix is not None:
            return cfg.version_suffix or None
        if self.manifest.version_command:
            if cfg.dry_run:
                return '<version_command>'
            command = self._expand(self.manifest.version_command)
            try:
                out = subprocess.run(
                    command, cwd=cfg.src_dir, capture_output=True, text=True, check=True
                ).stdout
            except subprocess.CalledProcessError as err:
                msg = f'version_command failed with exit status {err.returncode}: {err.stderr}'
                raise BuildError(msg) from None
            return out.strip() or None
        if not (cfg.src_dir / '.git').exists():
            ref = cfg.ref or self.manifest.default_ref
            if cfg.dry_run and not (ref and _TAG_LIKE.match(ref)):
                return 'dev0+g<sha>'
            return None
        return self._git_version_suffix()

    def _git_version_suffix(self) -> str | None:
        """Return ``dev0+g<sha>`` for the checkout, or ``None`` on a tag or without git."""
        cfg = self.config
        try:
            self.git('describe', '--exact-match', '--tags', 'HEAD', cwd=cfg.src_dir, capture=True)
        except subprocess.CalledProcessError:
            pass
        else:
            return None
        try:
            sha = self.git('rev-parse', '--short=8', 'HEAD', cwd=cfg.src_dir, capture=True)
        except subprocess.CalledProcessError:
            return None
        return f'dev0+g{sha}'

    def configure_command(self) -> list[str]:
        """Assemble the ``cmake`` configure command.

        Returns
        -------
        list[str]
            Command and arguments.

        """
        cfg = self.config
        cmd = ['cmake', '-Wno-dev', '-S', str(cfg.src_dir), '-B', str(cfg.cmake_dir), '-GNinja']
        init_cache = self.manifest.init_cache.get(cfg.os_name)
        if init_cache:
            cmd += ['-C', str(cfg.src_dir / init_cache)]
        cmd += [
            f'-DCMAKE_BUILD_TYPE={cfg.build_type}',
            f'-DPython3_EXECUTABLE={self.python}',
            '-DPython3_FIND_STRATEGY=LOCATION',
        ]
        if self.launcher is not None:
            cmd += [
                f'-DCMAKE_C_COMPILER_LAUNCHER={self.launcher.name}',
                f'-DCMAKE_CXX_COMPILER_LAUNCHER={self.launcher.name}',
            ]
        cmd.append(f'-DVTK_VERSION_SUFFIX={self.version_suffix() or ""}')
        entries = cmake_entries(
            self.manifest, cfg.os_name, groups=self.groups, options=self.option_values
        )
        cmd += [f'-D{e}' for e in entries]
        cmd += [f'-D{e}' for e in cfg.defines]
        return cmd

    def configure_env(self) -> dict[str, str]:
        """Return the environment variables the manifest options export to cmake.

        Returns
        -------
        dict[str, str]
            Variable name to value.

        """
        return option_environment(self.manifest, self.option_values)

    def _state(self) -> dict[str, object]:
        """Return what the configure step depends on, for change detection between runs."""
        return {
            'groups': self.groups,
            'options': self.option_values,
            'defines': self.config.defines,
            'build_type': self.config.build_type,
            'version_suffix': self.version_suffix(),
        }

    def phase_configure(self) -> None:
        """Run cmake, or skip it when the tree is already configured the same way."""
        cfg = self.config
        self.step('4/8 Configure')
        state_file = cfg.cmake_dir / STATE_FILENAME
        if cfg.clean and cfg.cmake_dir.exists():
            self.say(f'--clean removes [bold]{cfg.cmake_dir}[/bold].')
            if self.confirm('Delete it?') and not cfg.dry_run:
                shutil.rmtree(cfg.cmake_dir)
        configured = (cfg.cmake_dir / 'CMakeCache.txt').exists()
        changed = configured and (
            not state_file.is_file() or json.loads(state_file.read_text()) != self._state()
        )
        if configured and not cfg.reconfigure and not changed:
            self.say(
                f'[bold]{cfg.cmake_dir}[/bold] is already configured; skipping '
                '(use --reconfigure).'
            )
            return
        if changed:
            self.say('The build options differ from the last configure.')
            if not self.confirm('Reconfigure with the new options?'):
                msg = 'Keeping the old configuration; run again without changing the options.'
                raise BuildError(msg)
        self.say(
            f'Build type [bold]{cfg.build_type}[/bold], build tree [bold]{cfg.cmake_dir}[/bold].'
        )
        if not self.confirm('Run cmake?'):
            msg = 'Stopped before configure.'
            raise BuildError(msg)
        self.run(self.configure_command(), env=self.configure_env())
        if not cfg.dry_run:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(json.dumps(self._state(), indent=2))
        self.hooks('post_configure')

    def phase_build(self) -> None:
        """Compile with ninja through cmake."""
        cfg = self.config
        self.step('5/8 Build')
        cores = os.cpu_count() or 1
        jobs = cfg.jobs or cores
        flag = ' (--jobs)' if cfg.jobs else ''
        self.say(
            f'{cores} CPU cores detected; compiling with [bold]{jobs}[/bold] jobs{flag}. '
            'A full VTK build takes tens of minutes to hours.'
        )
        jobs = self._ask_jobs(jobs)
        self.run(
            ['cmake', '--build', str(cfg.cmake_dir), '--parallel', str(jobs)],
            log=cfg.build_dir / 'build.log',
        )
        if self.launcher is not None:
            self.run([self.launcher.name, '--show-stats'], fatal=False)
        self.hooks('post_build')

    def _ask_jobs(self, jobs: int) -> int:
        """Confirm the build, taking a number as the job count instead of a yes.

        Parameters
        ----------
        jobs : int
            Proposed job count.

        Returns
        -------
        int
            Job count to build with.

        """
        if not self.config.prompting:
            if not self.confirm('Start the build?'):
                msg = 'Stopped before building.'
                raise BuildError(msg)
            return jobs
        while True:
            answer = (
                Prompt.ask(
                    'Start the build? [y/n, or a number of jobs]',
                    default='y',
                    console=self.console,
                )
                .strip()
                .lower()
            )
            if answer in ('y', 'yes'):
                return jobs
            if answer in ('n', 'no'):
                msg = 'Stopped before building.'
                raise BuildError(msg)
            if answer.isdigit() and int(answer) > 0:
                return int(answer)
            self.say('[red]Answer y, n or a positive number.[/red]')

    def wheel_env(self) -> dict[str, str]:
        """Return the extra environment for ``bdist_wheel``.

        Returns
        -------
        dict[str, str]
            Platform tag variables on macOS; empty elsewhere.

        """
        cfg = self.config
        if cfg.os_name != 'macos':
            return {}
        arch = platform.machine()
        prefix = 'CMAKE_OSX_DEPLOYMENT_TARGET='
        targets = [
            e[len(prefix) :] for e in self.manifest.cmake_os['macos'] if e.startswith(prefix)
        ]
        target = targets[0] if targets else '11.0'
        return {'_PYTHON_HOST_PLATFORM': f'macosx-{target}-{arch}', 'ARCHFLAGS': f'-arch {arch}'}

    def phase_wheel(self) -> None:
        """Package the build tree into a wheel."""
        cfg = self.config
        self.step('6/8 Wheel')
        self.say(f'The wheel is written to [bold]{cfg.dist_dir}[/bold].')
        if not self.confirm('Build the wheel?'):
            msg = 'Stopped before packaging.'
            raise BuildError(msg)
        self.run([self.python, 'setup.py', 'bdist_wheel'], cwd=cfg.cmake_dir, env=self.wheel_env())
        self.hooks('post_wheel')
        self.wheel = self._newest_wheel(cfg.dist_dir)

    def _newest_wheel(self, folder: Path) -> Path | None:
        """Return the most recently written wheel in a folder."""
        wheels = sorted(folder.glob('*.whl'), key=lambda p: p.stat().st_mtime)
        if not wheels:
            if self.config.dry_run:
                return None
            msg = f'No wheel found in {folder}.'
            raise BuildError(msg)
        return wheels[-1]

    def repair_command(self, wheel: Path) -> tuple[list[str], dict[str, str]]:
        """Assemble the repair command for this operating system.

        Parameters
        ----------
        wheel : Path
            Wheel to repair.

        Returns
        -------
        list[str]
            Command and arguments.

        dict[str, str]
            Extra environment for the command.

        """
        cfg = self.config
        out = str(cfg.wheelhouse)
        if cfg.os_name == 'linux':
            return ['auditwheel', 'repair', str(wheel), '-w', out], {}
        if cfg.os_name == 'windows':
            bin_dir = str(cfg.cmake_dir / 'bin')
            return ['delvewheel', 'repair', '--add-path', bin_dir, '-w', out, str(wheel)], {}
        dylibs = cfg.cmake_dir.rglob('libvtksys*.dylib')
        libdir = next((p.parent for p in dylibs), cfg.cmake_dir / 'lib')
        arch = platform.machine()
        return (
            ['delocate-wheel', '--require-archs', arch, '-w', out, '-v', str(wheel)],
            {'DYLD_LIBRARY_PATH': str(libdir)},
        )

    def phase_repair(self) -> None:
        """Bundle the shared libraries into the wheel."""
        cfg = self.config
        self.step('7/8 Repair')
        if not cfg.repair:
            self.say('Skipped (--no-repair). The wheel only works while the build tree exists.')
            return
        self.say(
            'Repairing copies the VTK shared libraries into the wheel so it installs anywhere, '
            'not just next to this build tree.'
        )
        if cfg.os_name == 'linux':
            self.say(
                "[dim]On Linux the wheel is tagged for this machine's glibc; portable manylinux "
                'wheels need a manylinux container.[/dim]'
            )
        if not self.confirm('Repair the wheel?'):
            self.say('Skipped.')
            return
        wheel = self.wheel or cfg.dist_dir / '<backend>-*.whl'
        cmd, env = self.repair_command(wheel)
        self.run(cmd, env=env)
        if not cfg.dry_run:
            self.wheel = self._newest_wheel(cfg.wheelhouse)

    def install_command(self, wheel: Path) -> list[str] | None:
        """Assemble the command that installs a wheel into this interpreter.

        Parameters
        ----------
        wheel : Path
            Wheel to install.

        Returns
        -------
        list[str] | None
            Command, or ``None`` when neither pip nor uv is available.

        """
        return _tools.pip_install_command(str(wheel), args=('--no-deps', '--force-reinstall'))

    def phase_install(self) -> None:
        """Install the wheel into the current environment when asked."""
        cfg = self.config
        self.step('8/8 Install')
        wheel = self.wheel or cfg.wheelhouse / '<backend>-*.whl'
        cmd = self.install_command(wheel)
        if cmd is None:
            self.say(f'No pip or uv found; install the wheel yourself: {wheel}')
            return
        wanted = cfg.install or self.confirm(
            f'Install {wheel.name} into {sys.prefix}?', default=False
        )
        if not wanted:
            self.say(f'Not installed. To install later:\n  [cyan]{shlex.join(cmd)}[/cyan]')
            return
        self.run(cmd)

    def build_in_manylinux(self) -> Path | None:
        """Run the build inside a manylinux container, then install on the host.

        Returns
        -------
        Path or None
            The repaired wheel, or ``None`` on a dry run.

        """
        cfg = self.config
        image = cfg.manylinux or _manylinux.default_image()
        self.say(
            f'The build runs inside the [bold]{image}[/bold] container through Docker, so '
            'the wheel is portable across Linux machines. The container installs the '
            f'build dependencies and PyVista, then runs the eight steps with '
            f'[bold]{cfg.build_dir}[/bold] mounted as its build tree.'
        )
        docker = _tools.Tool('docker', ('docker',), instructions=_manylinux.DOCKER_INSTRUCTIONS)
        if not _tools.detect(docker, self.env):
            msg = f'docker was not found on PATH. {docker.instructions}'
            raise BuildError(msg)
        if not cfg.dry_run:
            probe = subprocess.run([str(docker.path), 'info'], capture_output=True, check=False)
            if probe.returncode != 0:
                msg = 'Docker is installed but its daemon is not running or not reachable.'
                raise BuildError(msg)
        source_tree = cfg.src_dir if (cfg.src_dir / '.git').exists() else None
        try:
            packages = find_manifest(cfg.backend, source_tree).manylinux_packages
        except ManifestError as err:
            raise BuildError(str(err)) from None
        pv_source = _manylinux.pyvista_source()
        mounts: dict[str, Path] = {_manylinux.WORK: cfg.build_dir}
        if cfg.source is not None:
            mounts[_manylinux.SOURCE] = cfg.source
        if cfg.cache_dir is not None:
            mounts[_manylinux.CACHE] = cfg.cache_dir
        if pv_source is not None:
            mounts[_manylinux.PYVISTA] = pv_source
        owner = _manylinux.file_owner()
        script = _manylinux.container_script(
            python=_manylinux.container_python(),
            packages=packages,
            pyvista=_manylinux.pyvista_requirement(pv_source),
            build_args=self.container_args(),
            ccache=cfg.compiler_cache != 'none',
            owner=owner,
            paths=[m for m in mounts if m != _manylinux.PYVISTA],
        )
        cmd = _manylinux.docker_command(
            image, mounts=mounts, script=script, interactive=cfg.prompting, env=self.env
        )
        if not self.confirm('Run the build in the container? (n stops)'):
            msg = 'Stopped before the container build.'
            raise BuildError(msg)
        if not cfg.dry_run:
            for folder in mounts.values():
                folder.mkdir(parents=True, exist_ok=True)
        self.run(cmd)
        if not cfg.dry_run:
            self.wheel = self._newest_wheel(cfg.wheelhouse)
        self.phase_install()
        self.summary()
        return self.wheel

    def container_args(self) -> list[str]:
        """Return the ``pyvista build`` arguments that rerun this configuration in a container.

        Returns
        -------
        list[str]
            Arguments after ``pyvista build``, with host paths mapped to the mounts.

        """
        cfg = self.config
        args = [cfg.backend, '--build-dir', _manylinux.WORK, '--no-install']
        if cfg.source is not None:
            args += ['--source', _manylinux.SOURCE]
        if cfg.cache_dir is not None:
            args += ['--cache-dir', _manylinux.CACHE]
        if cfg.url is not None:
            args += ['--url', cfg.url]
        if cfg.ref is not None:
            args += ['--ref', cfg.ref]
        if cfg.with_:
            args += ['--with', ','.join(cfg.with_)]
        if cfg.without:
            args += ['--without', ','.join(cfg.without)]
        if cfg.all_:
            args.append('--all')
        if cfg.minimal:
            args.append('--minimal')
        args += ['--build-type', cfg.build_type]
        if cfg.jobs is not None:
            args += ['--jobs', str(cfg.jobs)]
        for define in cfg.defines:
            args += ['-D', define]
        for option in cfg.options:
            args += ['--opt', option]
        for flag, on in (
            ('--clean', cfg.clean),
            ('--reconfigure', cfg.reconfigure),
            ('--update', cfg.update),
            ('--no-repair', not cfg.repair),
            ('--interactive', cfg.prompting),
        ):
            if on:
                args.append(flag)
        args += ['--compiler-cache', cfg.compiler_cache]
        if cfg.version_suffix is not None:
            args += ['--version-suffix', cfg.version_suffix]
        return args

    def summary(self) -> None:
        """Print where things ended up."""
        cfg = self.config
        self.console.rule('[bold green]Done')
        if self.wheel is not None:
            self.say(f'Wheel: [bold]{self.wheel}[/bold]')
        self.say(f'Source: {cfg.src_dir}\nBuild tree: {cfg.cmake_dir}')
        self.say('Edit the source and run the same command again for an incremental rebuild.')
