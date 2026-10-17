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
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .manifest import Manifest

STATE_FILENAME = '.pyvista-build.json'
BUILD_TYPES = ('Release', 'RelWithDebInfo', 'Debug')
COMPILER_CACHES = ('auto', 'ccache', 'sccache', 'none')
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
            self.say(
                f'A fork or branch of {cfg.backend} can be given instead (--url, --ref, '
                f'--source); another backend is a separate run, e.g. `pyvista build '
                f'{self._other_backend()}`.'
            )
            source = self._ask_source()
            if source is not None:
                self.say(f'Using the existing checkout at [bold]{source}[/bold].')
                cfg.source = source
                self._existing_checkout(source)
                self._load_manifest(shipped)
                return
            url, ref = cfg.url or url, cfg.ref or ref
            self.git('clone', '--filter=blob:none', url, str(cfg.src_dir))
            if ref is None and not cfg.dry_run:
                ref = find_manifest(cfg.backend, cfg.src_dir).default_ref
            if ref is not None:
                self.git('-c', 'advice.detachedHead=false', 'checkout', ref, cwd=cfg.src_dir)

        self._load_manifest(shipped)

    def _load_manifest(self, shipped: Path | None) -> None:
        """Load the manifest from the checkout, falling back to the shipped one."""
        cfg = self.config
        if cfg.dry_run and not (cfg.src_dir / MANIFEST_FILENAME).is_file() and shipped is None:
            msg = (
                f'--dry-run needs an existing checkout for {cfg.backend!r}: its '
                f'{MANIFEST_FILENAME} lives in the source tree.'
            )
            raise BuildError(msg)
        source = cfg.src_dir if cfg.src_dir.exists() else None
        self._manifest = find_manifest(cfg.backend, source)
        self.say(f'Manifest: [bold]{self.manifest.path}[/bold]')

    def _other_backend(self) -> str:
        """Return a known backend other than the one being built, for examples."""
        others = [name for name in BACKENDS if name != self.config.backend]
        return others[0] if others else 'cvista'

    def _ask_source(self) -> Path | None:
        """Confirm the clone, taking a git URL or a local checkout instead of a yes.

        Returns
        -------
        Path or None
            An existing checkout to use instead of cloning, else ``None``.

        """
        cfg = self.config
        if not cfg.prompting:
            if not self.confirm('Clone it now?'):
                msg = 'Nothing to build without a source checkout.'
                raise BuildError(msg)
            return None
        while True:
            answer = Prompt.ask(
                'Clone it now? [y/n, or a git URL or local checkout to use instead]',
                default='y',
                console=self.console,
            ).strip()
            if answer.lower() in ('y', 'yes'):
                return None
            if answer.lower() in ('n', 'no'):
                msg = 'Nothing to build without a source checkout.'
                raise BuildError(msg)
            path = Path(answer).expanduser()
            if (path / '.git').exists():
                return path.resolve()
            if path.exists() or ('://' not in answer and not answer.startswith('git@')):
                self.say(f'[red]{answer} is neither a git checkout nor a git URL.[/red]')
                continue
            cfg.url = answer
            cfg.ref = Prompt.ask(
                'Tag, branch or commit to check out',
                default=cfg.ref or self._default_ref(),
                console=self.console,
            )
            self.say(f'Cloning [bold]{cfg.url}[/bold] at [bold]{cfg.ref}[/bold] instead.')
            return None

    def _default_ref(self) -> str:
        """Return the shipped manifest's default ref, or ``'master'`` without one."""
        if shipped_manifest(self.config.backend) is None:
            return 'master'
        return find_manifest(self.config.backend, None).default_ref

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
        """Show the groups and options, letting the user adjust them interactively."""
        cfg = self.config
        self.step('3/8 Build options')
        self._resolve_selection()
        self._print_selection()
        if not cfg.prompting:
            self.say(
                'Change the selection with --with/--without/--all/--minimal and --opt NAME=VALUE.'
            )
            return
        self.say(
            'To change the selection, answer with flags, for example '
            '[bold]--without web --opt smp=TBB[/bold].'
        )
        self._ask_until_yes(
            'Build with these? [y / n = choose one by one / '
            '--with --without --all --minimal --opt]',
            apply=self._apply_selection_flags,
            resolve=self._resolve_selection,
            show=self._print_selection,
            walk=self._adjust_selection,
        )

    def _ask_until_yes(
        self,
        question: str,
        *,
        apply: Callable[[str], None],
        resolve: Callable[[], None],
        show: Callable[[], None],
        walk: Callable[[], None],
    ) -> None:
        """Ask until yes: ``n`` walks through the settings one by one, flags change them."""
        self.say(
            '[dim]Enter or y continues, n asks about each setting in turn, '
            'and flags change settings directly. Ctrl-C stops.[/dim]'
        )
        while True:
            answer = Prompt.ask(question, default='y', console=self.console).strip()
            if answer.lower() in ('y', 'yes'):
                return
            if answer.lower() in ('n', 'no'):
                walk()
                show()
                continue
            try:
                apply(answer)
                resolve()
            except BuildError as err:
                self.say(f'[red]{err}[/red]')
                continue
            show()

    def _resolve_selection(self) -> None:
        """Resolve the groups and options from the configuration."""
        cfg = self.config
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

    @staticmethod
    def _parse_flags(text: str, switches: set[str], valued: set[str]) -> list[tuple[str, str]]:
        """Split a prompt answer into ``(flag, value)`` pairs.

        Parameters
        ----------
        text : str
            Answer typed at the prompt.

        switches : set[str]
            Flags that take no value.

        valued : set[str]
            Flags that take one value, inline after ``=`` or as the next word.

        Returns
        -------
        list[tuple[str, str]]
            Flags in order, switches with an empty value.

        """
        try:
            words = shlex.split(text)
        except ValueError as err:
            msg = f'Cannot parse {text!r}: {err}'
            raise BuildError(msg) from None
        pairs: list[tuple[str, str]] = []
        i = 0
        while i < len(words):
            flag, _, inline = words[i].partition('=')
            if flag in switches:
                pairs.append((flag, ''))
            elif flag in valued:
                if not inline:
                    i += 1
                    if i == len(words):
                        msg = f'{flag} needs a value.'
                        raise BuildError(msg)
                    inline = words[i]
                pairs.append((flag, inline))
            else:
                msg = f'Unknown flag {words[i]!r}. Use {", ".join(sorted(switches | valued))}.'
                raise BuildError(msg)
            i += 1
        return pairs

    def _apply_selection_flags(self, text: str) -> None:
        """Apply selection flags typed at the prompt to the configuration."""
        cfg = self.config
        pairs = self._parse_flags(text, {'--all', '--minimal'}, {'--with', '--without', '--opt'})
        for flag, value in pairs:
            names = [n for n in value.split(',') if n]
            if flag == '--all':
                cfg.all_, cfg.minimal = True, False
            elif flag == '--minimal':
                cfg.all_, cfg.minimal = False, True
            elif flag == '--with':
                cfg.with_ += [n for n in names if n not in cfg.with_]
                cfg.without = [n for n in cfg.without if n not in names]
            elif flag == '--without':
                cfg.without += [n for n in names if n not in cfg.without]
                cfg.with_ = [n for n in cfg.with_ if n not in names]
            else:
                key = value.partition('=')[0]
                cfg.options = [o for o in cfg.options if o.partition('=')[0] != key] + [value]

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

    def _adjust_selection(self) -> None:
        """Ask about every group and option in turn."""
        chosen: list[str] = []
        for group in self.manifest.groups_for(self.config.os_name):
            clash = sorted(set(group.conflicts) & set(chosen))
            if clash:
                self.say(f'[dim]{group.name}: off, conflicts with {clash}[/dim]')
                continue
            if group.required or Confirm.ask(
                f'Enable {group.name!r} ({group.help})?',
                default=group.name in self.groups,
                console=self.console,
            ):
                chosen.append(group.name)
        for option in self.manifest.options.values():
            raw = Prompt.ask(
                f'{option.name} ({option.help})',
                default=str(self.option_values[option.name]),
                choices=list(option.choices) or None,
                console=self.console,
            )
            try:
                self.option_values[option.name] = option.coerce(raw)
            except ManifestError as err:
                raise BuildError(str(err)) from None
        try:
            self.groups = resolve_groups(
                self.manifest, self.config.os_name, with_=chosen, without=[], minimal=True
            )
        except ManifestError as err:
            raise BuildError(str(err)) from None

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
        self._print_configure_settings()
        if cfg.prompting:
            self.say(
                'To change a setting, answer with its flag, for example '
                '[bold]--build-type Debug -D VTK_ENABLE_LOGGING=OFF[/bold].'
            )
            self._ask_until_yes(
                'Run cmake with these? [y / n = choose one by one / '
                '--build-type --compiler-cache --version-suffix -D]',
                apply=self._apply_configure_flags,
                resolve=lambda: None,
                show=self._print_configure_settings,
                walk=self._walk_configure_settings,
            )
        elif not self.confirm('Run cmake with these?'):
            msg = 'Stopped before configure.'
            raise BuildError(msg)
        self.run(self.configure_command(), env=self.configure_env())
        if not cfg.dry_run:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(json.dumps(self._state(), indent=2))
        self.hooks('post_configure')

    def _print_configure_settings(self) -> None:
        """Print the settings the cmake command is assembled from."""
        cfg = self.config
        table = Table(show_header=True, header_style='bold purple')
        table.add_column('Setting')
        table.add_column('Value')
        table.add_column('Flag')
        launcher = self.launcher.name if self.launcher is not None else 'none'
        table.add_row('Build type', cfg.build_type, '--build-type')
        table.add_row('Build tree', str(cfg.cmake_dir), '--build-dir')
        table.add_row('Python', str(self.python), '')
        table.add_row('Compiler cache', launcher, '--compiler-cache')
        table.add_row('Version suffix', self.version_suffix() or '(release)', '--version-suffix')
        table.add_row('Extra cache entries', ' '.join(cfg.defines) or '(none)', '-D KEY=VALUE')
        self.console.print(table)

    def _walk_configure_settings(self) -> None:
        """Ask for each configure setting in turn."""
        cfg = self.config
        cfg.build_type = Prompt.ask(
            'Build type', default=cfg.build_type, choices=list(BUILD_TYPES), console=self.console
        )
        cache = Prompt.ask(
            'Compiler cache',
            default=cfg.compiler_cache,
            choices=list(COMPILER_CACHES),
            console=self.console,
        )
        if cache != cfg.compiler_cache:
            self._apply_configure_flags(f'--compiler-cache {cache}')
        cfg.version_suffix = Prompt.ask(
            'Version suffix (empty for a release version)',
            default=self.version_suffix() or '',
            console=self.console,
        )
        raw = Prompt.ask(
            'Extra CMake cache entries, KEY=VALUE separated by spaces',
            default=' '.join(cfg.defines),
            console=self.console,
        )
        cfg.defines = raw.split()

    def _apply_configure_flags(self, text: str) -> None:
        """Apply configure flags typed at the prompt to the configuration."""
        cfg = self.config
        valued = {'--build-type', '--version-suffix', '--compiler-cache', '-D', '--define'}
        for flag, value in self._parse_flags(text, set(), valued):
            if flag == '--build-type':
                if value not in BUILD_TYPES:
                    msg = f'--build-type must be one of {", ".join(BUILD_TYPES)}.'
                    raise BuildError(msg)
                cfg.build_type = value
            elif flag == '--version-suffix':
                cfg.version_suffix = value
            elif flag == '--compiler-cache':
                if value not in COMPILER_CACHES:
                    msg = f'--compiler-cache must be one of {", ".join(COMPILER_CACHES)}.'
                    raise BuildError(msg)
                cfg.compiler_cache = value
                self.launcher = None
                self._activate_compiler_cache()
            else:
                key = value.partition('=')[0]
                cfg.defines = [d for d in cfg.defines if d.partition('=')[0] != key] + [value]

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

    def summary(self) -> None:
        """Print where things ended up."""
        cfg = self.config
        self.console.rule('[bold green]Done')
        if self.wheel is not None:
            self.say(f'Wheel: [bold]{self.wheel}[/bold]')
        self.say(f'Source: {cfg.src_dir}\nBuild tree: {cfg.cmake_dir}')
        self.say('Edit the source and run the same command again for an incremental rebuild.')
