"""The build itself: tools, source, groups, configure, build, wheel, repair, install."""

from __future__ import annotations

from collections import deque
import contextlib
from dataclasses import dataclass
from dataclasses import field
import hashlib
from io import TextIOWrapper
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING
from typing import cast
import zipfile

from rich.console import Console
from rich.highlighter import Highlighter
from rich.markup import escape
from rich.prompt import Confirm as _RichConfirm
from rich.prompt import Prompt as _RichPrompt
from rich.style import Style
from rich.table import Table
from rich.text import Text

from . import manylinux as _manylinux
from . import tools as _tools
from .manifest import MANIFEST_FILENAME
from .manifest import ManifestError
from .manifest import cmake_entries
from .manifest import cmake_unset
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
    from io import BufferedReader
    from typing import IO

    from .manifest import Manifest

STATE_FILENAME = '.pyvista-build.json'
VERSION_SUFFIX_DEFINE = '-DVTK_VERSION_SUFFIX='
DEFAULT_MACOS_TARGET = '11.0'


class _NarrationHighlighter(Highlighter):
    """Color URLs, paths, command-line flags and version numbers.

    Parameters
    ----------
    links : bool, default: True
        Link each path to its location, for a terminal that can open it.

    """

    rules = (
        (re.compile(r'(https?://|git@)\S*[^\s.,)]'), 'underline blue'),
        (re.compile(r'(?<![\w-])--?[a-zA-Z][\w-]*'), 'bold cyan'),
        (re.compile(r'(?<![\w.])v?\d+(\.\d+)*(?!\w)'), 'cyan'),
        (re.compile(r'\b[Ss]kipped\b'), 'yellow'),
    )
    path = re.compile(r'(?<![\w/.:])(~(?=/)|/)[-\w.+~/]*[-\w+~/]')

    def __init__(self, *, links: bool = True) -> None:
        self.links = links

    def highlight(self, text: Text) -> None:
        """Apply every rule to the text in place, leaving paths whole.

        Parameters
        ----------
        text : Text
            Text to color.

        """
        paths = []
        for match in self.path.finditer(text.plain):
            link = None
            if self.links:
                with contextlib.suppress(ValueError, RuntimeError):
                    link = Path(match.group()).expanduser().as_uri()
            text.stylize(Style(color='magenta', link=link), *match.span())
            paths.append(match.span())
        for pattern, style in self.rules:
            for match in pattern.finditer(text.plain):
                start, end = match.span()
                if not any(a <= start < b for a, b in paths):
                    text.stylize(style, start, end)


_HIGHLIGHT = _NarrationHighlighter()
_HIGHLIGHT_UNLINKED = _NarrationHighlighter(links=False)
_TAG_LIKE = re.compile(r'v?\d+\.\d+')
_join = subprocess.list2cmdline if os.name == 'nt' else shlex.join
_NOISY_BUILD_LINE = re.compile(
    r': warning C\d+:|: note:|^\s+(with|\[|\])\s*$|^\s+[A-Za-z_][A-Za-z0-9_]*='
)
_TAIL_LINES = 40
_REPAIR_STAMP = '.pyvista-repair.json'
# One message from one module each; setuptools loads its distutils under either name.
_SETUPTOOLS_WARNINGS = (
    'ignore:License classifiers are deprecated::setuptools.dist',
    'ignore:setup.py install is deprecated::distutils.cmd',
    'ignore:setup.py install is deprecated::setuptools._distutils.cmd',
)
_TICK_SECONDS = 1.0


def _short_name(cmd: Sequence[str]) -> str:
    """Return a command's program name, plus its subcommand when it has one."""
    words = [Path(cmd[0]).stem]
    if len(cmd) > 1 and not cmd[1].startswith('-') and not set('/\\') & set(cmd[1]):
        words.append(cmd[1])
    return ' '.join(words)


def _lf(line: str) -> str:
    """Return a line with its Windows line ending turned into a plain newline."""
    return line[:-2] + '\n' if line.endswith('\r\n') else line


def _wheel_fingerprint(wheel: Path, cmd: Sequence[str]) -> str:
    """Return a hash of the repair command and each file's name, checksum, and size in a wheel."""
    with zipfile.ZipFile(wheel) as archive:
        entries = sorted((i.filename, i.CRC, i.file_size) for i in archive.infolist())
    return hashlib.sha256(json.dumps([entries, list(cmd)]).encode()).hexdigest()


def _repaired_wheel(stamp: Path, fingerprint: str) -> Path | None:
    """Return the wheel an earlier repair made from the same input, if it is still there."""
    try:
        record = json.loads(stamp.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get('fingerprint') != fingerprint:
        return None
    wheel = stamp.parent / str(record.get('wheel', ''))
    return wheel if wheel.is_file() else None


def _duration(seconds: float) -> str:
    """Return an elapsed time as ``1h 02m``, ``3m 04s`` or ``5s``."""
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours}h {minutes:02d}m'
    if minutes:
        return f'{minutes}m {secs:02d}s'
    return f'{secs}s'


def _interrupt(proc: subprocess.Popen[bytes]) -> None:
    """Pass Ctrl-C on to a command running in its own session."""
    if sys.platform != 'win32':
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGINT)


class BuildError(Exception):
    """A step cannot continue; the message tells the user what to do."""


class Prompt(_RichPrompt):
    """Text prompt whose question is printed verbatim.

    Parameters
    ----------
    prompt : str, default: ''
        Question text, shown as is.

    **kwargs
        Passed to the rich prompt.

    """

    def __init__(self, prompt: str = '', **kwargs) -> None:
        super().__init__(Text(prompt, style='prompt'), **kwargs)


class Confirm(_RichConfirm):
    """Yes/no prompt whose question is printed verbatim.

    Parameters
    ----------
    prompt : str, default: ''
        Question text, shown as is.

    **kwargs
        Passed to the rich prompt.

    """

    def __init__(self, prompt: str = '', **kwargs) -> None:
        super().__init__(Text(prompt, style='prompt'), **kwargs)


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
    minimal: bool = False
    build_type: str = 'Release'
    jobs: int | None = None
    defines: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    clean: bool = False
    reconfigure: bool = False
    update: bool = False
    compiler_cache: str = 'none'
    cache_dir: Path | None = None
    version_suffix: str | None = None
    repair: bool = True
    install: bool = False
    yes: bool = False
    interactive: bool = False
    dry_run: bool = False
    verbose: bool = False
    manylinux: str | None = None
    in_container: bool = False
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
    def prompting(self) -> bool:  # numpydoc ignore=RT01
        """Whether to ask questions: interactive, on a terminal, and not a dry run."""
        return self.interactive and not self.dry_run and sys.stdin.isatty()


class Builder:
    """Run the phases in order.

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
        self.env['PATH'] = f'{_tools.scripts_dir()}{os.pathsep}{self.env.get("PATH", "")}'
        self._manifest: Manifest | None = None
        self._checked_tools: dict[str, str] | None = None
        self._planned_installs: list[str] = []
        self._approved_installs: set[str] = set()
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
        highlight = _HIGHLIGHT if self._links else _HIGHLIGHT_UNLINKED
        self.console.print(highlight(Text.from_markup(message)))

    @property
    def _links(self) -> bool:
        """Whether output goes to a real terminal, which can open linked paths."""
        try:
            return self.console.is_terminal and self.console.file.isatty()
        except (AttributeError, ValueError):
            return False

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
            why = 'dry run' if self.config.dry_run else 'no terminal'
            answer = 'yes' if default else 'no'
            self.say(f'[dim]{escape(question)} -> {answer} ({why})[/dim]')
        return default

    def confirm_installs(self, items: Sequence[str]) -> None:
        """List everything the build will install, and ask unless the start already approved it.

        Parameters
        ----------
        items : Sequence[str]
            One line per thing to install, saying what and where.

        """
        if not items:
            return
        self.say('The build installs:')
        for item in items:
            self.say(f'  • {escape(item)}')
        cfg = self.config
        if cfg.yes or cfg.dry_run:
            return
        if not cfg.interactive and set(items) <= self._approved_installs:
            return
        if not sys.stdin.isatty():
            msg = 'Pass --yes to install these, or install them yourself and run again.'
            raise BuildError(msg)
        if not Confirm.ask('Install them? (n stops)', default=True, console=self.console):
            msg = 'Nothing was installed; install them yourself and run again.'
            raise BuildError(msg)

    def _planned_manifest(self) -> Manifest | None:
        """Return the manifest the build will most likely use, or ``None`` before a clone."""
        cfg = self.config
        try:
            if (cfg.src_dir / '.git').exists():
                return find_manifest(cfg.backend, cfg.src_dir)
            if cfg.url is None and shipped_manifest(cfg.backend) is not None:
                return find_manifest(cfg.backend, None)
        except ManifestError:
            pass
        return None

    def _unknown_backend(self) -> str:
        """Return the error for a backend with no known URL."""
        return (
            f'Unknown backend {self.config.backend!r}. Known backends: {known_backends()}. '
            'Pass --url <git url> or --source <checkout> to build another one.'
        )

    def _check_source(self) -> None:
        """Raise for a ``--source`` or ``--dry-run`` that cannot work, before any step runs."""
        cfg = self.config
        if cfg.source is not None and not (cfg.source / '.git').exists():
            msg = f'--source {cfg.source} is not a git checkout.'
            raise BuildError(msg)
        if (
            cfg.dry_run
            and (cfg.url or BACKENDS.get(cfg.backend))
            and shipped_manifest(cfg.backend) is None
            and not (cfg.src_dir / '.git').exists()
        ):
            msg = (
                f'A dry run cannot clone {cfg.backend!r}, and its {MANIFEST_FILENAME} lives in '
                'its source tree. Pass --source <checkout> to dry-run it.'
            )
            raise BuildError(msg)

    def plan(self) -> list[tuple[str, str]]:
        """Describe what each step will do, before any of them runs.

        Returns
        -------
        list[tuple[str, str]]
            Step name and description, in order.

        """
        cfg = self.config
        self._check_source()
        container = bool(cfg.manylinux) and not cfg.in_container
        os_name = 'linux' if container else cfg.os_name
        manifest = self._planned_manifest()
        if (cfg.src_dir / '.git').exists():
            source = f'reuse the checkout at {cfg.src_dir}'
            if cfg.ref:
                source += (
                    f', checked out at {cfg.ref} (local edits stop it unless it is already there)'
                )
            else:
                source += ', local edits kept'
        else:
            url = cfg.url or BACKENDS.get(cfg.backend) or (manifest.url if manifest else None)
            if url is None:
                raise BuildError(self._unknown_backend())
            ref = cfg.ref or (manifest.default_ref if manifest else None) or 'the default branch'
            source = f'clone {url} at {ref} into {cfg.src_dir}'
        tools = _tools.required_tools(
            os_name, manifest.tools if manifest else {}, repair=cfg.repair
        )
        names = ', '.join([t.label for t in tools] + list(_tools.PYTHON_BUILD_PACKAGES))
        if container:
            tools_text = f'check {names} inside the container, installing anything missing there'
            self._planned_installs = self.install_plan([])
        else:
            specs = [f'{t.pip}{t.spec}' for t in tools if t.pip and not _tools.detect(t, self.env)]
            specs += _tools.python_build_packages()
            self._planned_installs = self.install_plan(specs)
            tools_text = f'check {names}. ' + (
                f'Install the missing {", ".join(specs)} into {sys.prefix}'
                if specs
                else 'Nothing to install'
            )
        if manifest is None:
            options_text = "the default module groups and options from the checkout's manifest"
        else:
            try:
                groups = resolve_groups(
                    manifest,
                    os_name,
                    with_=cfg.with_,
                    without=cfg.without,
                    minimal=cfg.minimal,
                )
                values = resolve_options(manifest, cfg.options)
            except ManifestError as err:
                raise BuildError(str(err)) from None
            options_text = f'module groups {", ".join(groups) or "(none)"}'
            if values:
                options_text += '. Options ' + ', '.join(f'{k}={v}' for k, v in values.items())
        configure = f'cmake, {cfg.build_type} build with Ninja, into {cfg.cmake_dir}'
        if cfg.defines:
            configure += ', plus ' + ' '.join(f'-D{d}' for d in cfg.defines)
        cache = {'auto': 'ccache' if container else 'ccache or sccache, whichever is on PATH'}
        configure += f'. Compiler cache: {cache.get(cfg.compiler_cache, cfg.compiler_cache)}'
        log = cfg.build_dir / 'build.log'
        if container and not cfg.jobs:
            build = (
                'compile with one job per CPU core the container sees, at most one per 2 GB '
                f'of its memory, logging to {log}'
            )
        else:
            jobs, cores, _ = self._jobs()
            limit = ' (one per 2 GB of memory)' if not cfg.jobs and jobs < cores else ''
            build = f'compile with {jobs} parallel jobs{limit}, logging to {log}'
        wheel = f'package the wheel with setup.py bdist_wheel into {cfg.dist_dir}'
        if cfg.repair:
            repairer = _tools.REPAIRERS[os_name]
            repair = (
                f'bundle the libraries the wheel links against with {repairer}, '
                f'into {cfg.wheelhouse}, so it installs on other machines. '
                '--no-repair skips it'
            )
        else:
            repair = 'skipped (--no-repair), so the wheel may only work on this machine'
        if cfg.install and container and cfg.os_name != 'linux':
            install = (
                f'skipped: the wheel targets Linux, so it is not installed on this '
                f'{cfg.os_name} host'
            )
        elif cfg.install:
            install = f'install the wheel into {sys.prefix}'
        else:
            install = 'skipped. The install command is printed at the end, or pass --install'
        return [
            ('Tools', tools_text),
            ('Source', source),
            ('Build options', options_text),
            ('Configure', configure),
            ('Build', build),
            ('Wheel', wheel),
            ('Repair', repair),
            ('Install', install),
        ]

    def confirm_start(self) -> None:
        """Say where the build goes and ask before the first step; ``--yes`` answers yes."""
        cfg = self.config
        where = 'Build tree and wheel go' if cfg.source else 'Source, build tree and wheel go'
        self.say(
            f'{where} in [bold]{escape(str(cfg.build_dir))}[/bold]. '
            'Compiling takes from minutes to over an hour and needs several GB of disk.'
        )
        if cfg.yes or cfg.dry_run:
            if cfg.interactive:
                why = '--yes' if cfg.yes else 'dry run'
                self.say(f'[dim]Proceed? -> yes ({why})[/dim]')
            return
        if not sys.stdin.isatty():
            msg = 'Pass --yes to build without a terminal.'
            raise BuildError(msg)
        if not Confirm.ask('Proceed?', default=True, console=self.console):
            msg = 'Stopped before the first step.'
            raise BuildError(msg)
        self._approved_installs = set(self._planned_installs)

    def run(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        log: Path | None = None,
        fatal: bool = True,
        passthrough: bool = False,
    ) -> None:
        """Print and run a command; raise :class:`BuildError` when it fails.

        Without ``verbose``, the printed command leaves out its environment variables and its
        standard output shows as one updating status line.

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

        passthrough : bool, default: False
            Leave the command's output to the terminal, whatever ``verbose`` says.

        """
        where = f' (in {cwd})' if cwd is not None else ''
        shown_env = (env or {}) if self.config.verbose else {}
        prefix = ''.join(f'{k}={shlex.quote(v)} ' for k, v in shown_env.items())
        self.say(f'[cyan]$ {escape(prefix + _join(cmd))}[/cyan]{escape(where)}')
        if self.config.dry_run:
            return
        if cwd is not None:
            cwd.mkdir(parents=True, exist_ok=True)
        full_env = dict(self.env)
        full_env.update(env or {})
        resolved = [_tools.which(cmd[0], full_env), *cmd[1:]]
        started = time.monotonic()
        try:
            if passthrough or (self.config.verbose and log is None):
                status = subprocess.run(resolved, cwd=cwd, env=full_env, check=False).returncode
            else:
                status = self._run_captured(resolved, cwd=cwd, env=full_env, log=log)
        except OSError as err:
            msg = f'Cannot run {cmd[0]}: {err}'
            raise BuildError(msg) from None
        if status == 0:
            self.say(f'[green]Done[/green] in {_duration(time.monotonic() - started)}')
        elif fatal:
            msg = f'Command failed with exit status {status}: {_join(cmd)}'
            raise BuildError(msg)

    def _run_captured(
        self, cmd: Sequence[str], *, cwd: Path | None, env: dict[str, str], log: Path | None
    ) -> int:
        """Run a command, folding its standard output into a status line unless verbose."""
        verbose = self.config.verbose
        tail: deque[str] = deque(maxlen=_TAIL_LINES)
        lock = threading.Lock()
        stopped = threading.Event()
        started = time.monotonic()
        current = _short_name(cmd)

        def show(line: str) -> None:
            if log is None or not _NOISY_BUILD_LINE.search(line):
                self.console.out(line, end='', highlight=False)

        def lines(stream: IO[bytes] | None) -> TextIOWrapper:
            buffer = cast('BufferedReader', stream)
            return TextIOWrapper(buffer, encoding='utf-8', errors='replace', newline='')

        def render() -> None:
            if status is not None:
                elapsed = _duration(time.monotonic() - started)
                status.update(f'{escape(current)} [dim]{elapsed}[/dim]')

        def update(line: str) -> None:
            nonlocal current
            current = line.strip().expandtabs()[: max(self.console.width - 14, 20)]
            render()

        def tick() -> None:
            while not stopped.wait(_TICK_SECONDS):
                with lock:
                    render()

        with contextlib.ExitStack() as stack:
            log_file = stack.enter_context(log.open('w', encoding='utf-8')) if log else None
            proc = stack.enter_context(
                subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
            )

            def stop(kind: type[BaseException] | None, *_: object) -> None:
                if kind is KeyboardInterrupt:
                    _interrupt(proc)

            stack.push(stop)
            status = None
            if not verbose and self.console.is_terminal:
                status = stack.enter_context(self.console.status('', spinner='dots'))
                render()
                ticker = threading.Thread(target=tick, daemon=True)
                ticker.start()
                stack.callback(ticker.join)
                stack.callback(stopped.set)

            def copy_stderr() -> None:
                with lines(proc.stderr) as stream:
                    for line in map(_lf, stream):
                        with lock:
                            if not line.endswith('\n'):
                                update(line)
                                continue
                            if log_file is not None:
                                log_file.write(line)
                            show(line)

            reader = threading.Thread(target=copy_stderr, daemon=True)
            reader.start()
            with lines(proc.stdout) as stream:
                for line in map(_lf, stream):
                    with lock:
                        if not line.endswith('\n'):
                            update(line)
                            continue
                        if log_file is not None:
                            log_file.write(line)
                        if verbose:
                            show(line)
                            continue
                        tail.append(line)
                        update(line)
            reader.join()
        if proc.returncode != 0:
            if not verbose:
                for line in tail:
                    show(line)
            if log is not None:
                self.say(f'[red]Build failed; the full log is in {escape(str(log))}[/red]')
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
                cmd, cwd=cwd, env=self.env, capture_output=True, text=True, check=True
            ).stdout.strip()
        self.run(cmd, cwd=cwd)
        return ''

    @property
    def python(self) -> str:  # numpydoc ignore=RT01
        """Interpreter the wheel is built for."""
        return sys.executable

    def hooks(self, name: str) -> None:
        """Run the manifest's hook commands inside the source tree, with the options' variables.

        Parameters
        ----------
        name : str
            Hook name, for example ``'post_build'``.

        """
        cfg = self.config
        for template in self.manifest.hooks.get(name, ()):
            self.run(self._expand(template), cwd=cfg.src_dir, env=self.configure_env())

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

    def print_plan(self) -> None:
        """Print the eight steps, each in full unless every step asks before it runs."""
        steps = self.plan()
        if self.config.prompting:
            self.say(
                'Eight steps: tools, source, build options, configure, build, wheel, repair, '
                'and install.'
            )
            self.say(
                'Each step explains what it will do and asks before running any command; '
                'Enter takes the default.'
            )
            return
        self.say('The build runs these eight steps:')
        for number, (name, text) in enumerate(steps, start=1):
            self.say(f'  {number}. [bold]{name}[/bold]: {escape(text)}')

    def build(self) -> Path | None:
        """Run every phase.

        Returns
        -------
        Path | None
            The final wheel, or ``None`` on a dry run.

        """
        self.console.print(
            f'[bold green]Building backend: {escape(self.config.backend)}[/bold green]'
        )
        if self.config.manylinux:
            return self.build_in_manylinux()
        if not self.config.in_container:
            self.print_plan()
            self.confirm_start()
        self.phase_tools()
        self.phase_source()
        self.recheck_tools()
        self.phase_groups()
        self.phase_configure()
        self.phase_build()
        self.phase_wheel()
        self.phase_repair()
        if not self.config.in_container:
            self.phase_install()
            self.summary()
        return self.wheel

    def phase_source(self) -> None:
        """Clone or reuse the checkout, then load its manifest."""
        cfg = self.config
        self.step('2/8 Source')
        git = _tools.git_tool()
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
            self.say(f'Using the existing checkout at [bold]{escape(str(cfg.source))}[/bold].')
            self._existing_checkout(cfg.source)
        elif (cfg.src_dir / '.git').exists():
            where = escape(str(cfg.src_dir))
            self.say(f'Reusing the checkout at [bold]{where}[/bold]; local edits are kept.')
            self._existing_checkout(cfg.src_dir)
        else:
            if url is None:
                raise BuildError(self._unknown_backend())
            ref = cfg.ref
            if ref is None and shipped is not None:
                ref = find_manifest(cfg.backend, None).default_ref
            self.say(
                f'The source will be cloned from [bold]{escape(url)}[/bold] into [bold]'
                f'{escape(str(cfg.src_dir))}[/bold] (blob-less partial clone: history without '
                'file contents until needed).'
            )
            if not self.confirm('Clone it now?'):
                msg = 'Nothing to build without a source checkout.'
                raise BuildError(msg)
            self.git('clone', '--progress', '--filter=blob:none', url, str(cfg.src_dir))
            if ref is None:
                ref = find_manifest(cfg.backend, cfg.src_dir).default_ref
            self.git('-c', 'advice.detachedHead=false', 'checkout', ref, cwd=cfg.src_dir)

        source = cfg.src_dir if cfg.src_dir.exists() else None
        self._manifest = find_manifest(cfg.backend, source)
        self.say(f'Manifest: [bold]{escape(str(self.manifest.path))}[/bold]')

    def _existing_checkout(self, src: Path) -> None:
        """Apply ``--update`` / ``--ref`` to a checkout that already exists."""
        cfg = self.config
        if not cfg.update and (cfg.ref is None or self._at_ref(src, cfg.ref)):
            return
        try:
            dirty = self.git(
                'status', '--porcelain', '--untracked-files=no', cwd=src, capture=True
            )
        except subprocess.CalledProcessError as err:
            msg = f'git status failed in {src}: {(err.stderr or "").strip()}'
            raise BuildError(msg) from None
        if dirty:
            msg = f'{src} has local changes; commit or stash them before using --ref or --update.'
            raise BuildError(msg)
        if cfg.update:
            self.git('fetch', '--progress', '--tags', 'origin', cwd=src)
        ref = cfg.ref
        if ref is None:
            ref = find_manifest(cfg.backend, src).default_ref
        if cfg.update and self._has_ref(src, f'origin/{ref}'):
            ref = f'origin/{ref}'
        if cfg.update or self.confirm(f'Check out {ref!r} in {src}?'):
            self.git('-c', 'advice.detachedHead=false', 'checkout', ref, cwd=src)

    def _has_ref(self, src: Path, ref: str) -> bool:
        """Return whether ``ref`` names a commit in the checkout."""
        try:
            self.git(
                'rev-parse', '--verify', '--quiet', f'{ref}^{{commit}}', cwd=src, capture=True
            )
        except subprocess.CalledProcessError:
            return False
        return True

    def _at_ref(self, src: Path, ref: str) -> bool:
        """Return whether a checkout's HEAD is already the commit ``ref`` names."""
        try:
            head = self.git('rev-parse', 'HEAD', cwd=src, capture=True)
            target = self.git('rev-parse', f'{ref}^{{commit}}', cwd=src, capture=True)
        except subprocess.CalledProcessError:
            return False
        return head == target

    def phase_tools(self) -> None:
        """Check every tool, then install what is missing after one confirmation."""
        cfg = self.config
        self.step('1/8 Tools')
        manifest = self._planned_manifest()
        self._checked_tools = dict(manifest.tools) if manifest else {}
        tools = _tools.required_tools(cfg.os_name, self._checked_tools, repair=cfg.repair)
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
            found = escape(found)
            if ok:
                status = '[green]ok[/green]'
            elif tool.pip:
                status = (
                    '[yellow]missing[/yellow]'
                    if tool.path is None
                    else '[yellow]wrong version[/yellow]'
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
                table.add_row(f'{name} (python)', escape(sys.prefix), '[green]ok[/green]')
        table.add_row(*self._compiler_cache_row())
        self.console.print(table)
        choice = cfg.compiler_cache
        if choice not in ('none', 'auto') and not _tools.compiler_cache_tool(
            choice, cfg.os_name, self.env
        ):
            hint = _tools.compiler_cache_instructions(cfg.os_name, choice)
            blocking.append(_tools.Tool(choice, (choice,), instructions=hint))
        if blocking:
            lines = [f'{t.name}: {t.instructions}' for t in blocking]
            msg = 'Install these and run again:\n  ' + '\n  '.join(lines)
            raise BuildError(msg)

        specs = [f'{t.pip}{t.spec}' for t in missing_pip] + missing_packages
        cmd = _tools.pip_install_command(*specs) if specs else None
        if specs and cmd is None:
            msg = (
                f'Install {", ".join(specs)} into {sys.prefix} and run again (no pip or uv found).'
            )
            raise BuildError(msg)
        if cfg.install:
            self._check_installer()
        self.confirm_installs(self.install_plan(specs))
        if cmd is not None:
            self.run(cmd)
            if not cfg.dry_run:
                bad = [t.label for t in missing_pip if not _tools.detect(t, self.env)]
                if bad:
                    msg = f'Still not usable after install: {", ".join(bad)}.'
                    raise BuildError(msg)
        self._activate_compiler_cache()

    def _check_installer(self) -> None:
        """Raise when ``--install`` has no pip or uv to install the wheel with."""
        if _tools.pip_install_command() is None:
            msg = '--install needs pip or uv in this environment.'
            raise BuildError(msg)

    def _activate_msvc(self, tools: list[_tools.Tool]) -> None:
        """On Windows, import the MSVC environment when ``cl`` is not already on PATH."""
        if self.config.os_name != 'windows':
            return
        msvc = next(t for t in tools if t.name == 'msvc')
        if _tools.detect(msvc, self.env):
            return
        arch = _tools.msvc_arch()
        vcvarsall = _tools.find_vcvarsall(arch)
        if vcvarsall is None:
            return
        self.say(
            f'Visual Studio found; importing its build environment from [bold]{vcvarsall}[/bold].'
        )
        if self.confirm('Use it?'):
            try:
                self.env = _tools.msvc_environment(vcvarsall, arch, env=self.env)
            except subprocess.CalledProcessError as err:
                msg = f'{vcvarsall} failed with exit status {err.returncode}.'
                raise BuildError(msg) from None

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

    def compiler_cache_dir(self, var: str) -> Path | None:
        """Return the compiler cache directory the user chose.

        Parameters
        ----------
        var : str
            Cache tool's directory variable, ``CCACHE_DIR`` or ``SCCACHE_DIR``.

        Returns
        -------
        Path | None
            ``--cache-dir``, else ``var`` when it is set, else ``None`` for the tool's default.

        """
        if self.config.cache_dir is not None:
            return self.config.cache_dir
        if self.env.get(var):
            return Path(self.env[var]).resolve()
        return None

    def _activate_compiler_cache(self) -> None:
        """Pick the compiler launcher and point it at the chosen cache directory, if any."""
        cfg = self.config
        if cfg.compiler_cache == 'none':
            return
        self.launcher = _tools.compiler_cache_tool(cfg.compiler_cache, cfg.os_name, self.env)
        if self.launcher is None:
            hint = _tools.compiler_cache_instructions(cfg.os_name, cfg.compiler_cache)
            self.say(
                '[yellow]No ccache/sccache on PATH; rebuilds will be slow. '
                f'Install one: {hint}[/yellow]'
            )
            return
        var = 'SCCACHE_DIR' if self.launcher.name == 'sccache' else 'CCACHE_DIR'
        cache_dir = self.compiler_cache_dir(var)
        if cache_dir is None:
            self.say(f'Compiler cache: [bold]{self.launcher.name}[/bold] in its default location')
        else:
            self.say(
                f'Compiler cache: [bold]{self.launcher.name}[/bold] in [bold]{cache_dir}[/bold]'
            )
            self.env[var] = str(cache_dir)
        if self.launcher.name == 'ccache':
            if cfg.os_name != 'windows':
                self.env.setdefault('CCACHE_COMPILERCHECK', '%compiler% --version')
            try:
                base = Path(os.path.commonpath([cfg.src_dir.resolve(), cfg.cmake_dir.resolve()]))
            except ValueError:
                base = Path(cfg.cmake_dir.resolve().anchor)
            if base != Path(base.anchor):
                self.env.setdefault('CCACHE_BASEDIR', str(base))
                self.env.setdefault('CCACHE_NOHASHDIR', '1')

    def recheck_tools(self) -> None:
        """Check again the tool versions the checkout's manifest asks for, if they changed."""
        cfg = self.config
        specs = self.manifest.tools
        if self._checked_tools is None or dict(specs) == self._checked_tools:
            return
        tools = [
            t
            for t in _tools.required_tools(cfg.os_name, specs, repair=False)
            if t.spec and t.spec != self._checked_tools.get(t.name, '')
        ]
        self._checked_tools = dict(specs)
        if not tools:
            return
        labels = ', '.join(t.label for t in tools)
        self.say(f"The checkout's manifest asks for {escape(labels)}. Checking them again.")
        missing = [t for t in tools if not _tools.detect(t, self.env)]
        if not missing:
            return
        wanted = [f'{t.pip}{t.spec}' for t in missing]
        cmd = _tools.pip_install_command(*wanted)
        if cmd is None:
            msg = (
                f'Install {", ".join(wanted)} into {sys.prefix} and run again '
                '(no pip or uv found).'
            )
            raise BuildError(msg)
        self.confirm_installs([f'{spec} into {sys.prefix}' for spec in wanted])
        self.run(cmd)
        if not cfg.dry_run:
            bad = [t.label for t in missing if not _tools.detect(t, self.env)]
            if bad:
                msg = f'Still not usable after install: {", ".join(bad)}.'
                raise BuildError(msg)

    def install_plan(self, specs: Sequence[str]) -> list[str]:
        """Return one line per thing the build installs.

        Parameters
        ----------
        specs : Sequence[str]
            Missing Python packages with their version requirements.

        Returns
        -------
        list[str]
            What gets installed, and where.

        """
        items = [f'{spec} into {sys.prefix}' for spec in specs]
        if self.config.install:
            items.append(f'the {self.config.backend} wheel into {sys.prefix}, once it is built')
        return items

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
                minimal=cfg.minimal,
            )
            self.option_values = resolve_options(self.manifest, cfg.options)
        except ManifestError as err:
            raise BuildError(str(err)) from None
        self._print_selection()
        self.say('Change the selection with --with/--without/--minimal and --opt NAME=VALUE.')

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
            table.add_row(escape(group.name), value, escape(group.help))
        for option in self.manifest.options.values():
            value = escape(str(self.option_values[option.name]))
            table.add_row(escape(f'--opt {option.name}'), value, escape(option.help))
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
            except OSError as err:
                msg = f'Cannot run version_command {command[0]}: {err}'
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
        cmd += [f'-U{key}' for key in cmake_unset(self.manifest, cfg.os_name, groups=self.groups)]
        init_cache = self.manifest.init_cache.get(cfg.os_name)
        if init_cache:
            cmd += ['-C', str(cfg.src_dir / init_cache)]
        cmd += [
            f'-DCMAKE_BUILD_TYPE={cfg.build_type}',
            f'-DPython3_EXECUTABLE={self.python}',
            '-DPython3_FIND_STRATEGY=LOCATION',
        ]
        launcher = self.launcher.name if self.launcher is not None else ''
        cmd += [
            f'-DCMAKE_C_COMPILER_LAUNCHER={launcher}',
            f'-DCMAKE_CXX_COMPILER_LAUNCHER={launcher}',
        ]
        cmd.append(f'{VERSION_SUFFIX_DEFINE}{self.version_suffix() or ""}')
        entries = cmake_entries(
            self.manifest, cfg.os_name, groups=self.groups, options=self.option_values
        )
        cmd += [f'-D{e}' for e in entries]
        if cfg.os_name == 'macos' and not self._macos_targets():
            cmd.append(f'-DCMAKE_OSX_DEPLOYMENT_TARGET={DEFAULT_MACOS_TARGET}')
        cmd += [f'-D{e}' for e in cfg.defines]
        return cmd

    def configure_env(self) -> dict[str, str]:
        """Return the environment cmake runs in: the manifest options and the wheel platform.

        Returns
        -------
        dict[str, str]
            Variable name to value.

        """
        return {**option_environment(self.manifest, self.option_values), **self.wheel_env()}

    def _stored_state(self) -> object:
        """Return the configure state saved by the last run, or ``None`` when unreadable."""
        state_file = self.config.cmake_dir / STATE_FILENAME
        try:
            return json.loads(state_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None

    def phase_configure(self) -> None:
        """Run cmake, or skip it when the tree is already configured the same way."""
        cfg = self.config
        self.step('4/8 Configure')
        if cfg.clean and cfg.cmake_dir.exists():
            self.say(f'--clean removes [bold]{escape(str(cfg.cmake_dir))}[/bold].')
            if self.confirm('Delete it?') and not cfg.dry_run:
                shutil.rmtree(cfg.cmake_dir)
        cmd = self.configure_command()
        env = self.configure_env()
        state = {'command': cmd, 'env': env}
        configured = (cfg.cmake_dir / 'CMakeCache.txt').exists() and not cfg.clean
        stored = self._stored_state() if configured else None
        if configured and not cfg.reconfigure and stored == state:
            self.say(
                f'[bold]{escape(str(cfg.cmake_dir))}[/bold] is already configured; skipping '
                '(use --reconfigure).'
            )
            return
        if configured and stored != state:
            if self._only_version_changed(stored, cmd, env):
                self.say('The source is at a new commit, so the version suffix changes.')
            else:
                if stored is None:
                    self.say('There is no record of how this tree was configured.')
                else:
                    self.say(
                        'The configure command differs from the one this tree was set up with.'
                    )
                if not self.confirm('Reconfigure?'):
                    msg = 'Stopped; the build tree keeps its earlier configuration.'
                    raise BuildError(msg)
        tree = escape(str(cfg.cmake_dir))
        self.say(f'Build type [bold]{cfg.build_type}[/bold], build tree [bold]{tree}[/bold].')
        if not self.confirm('Run cmake?'):
            msg = 'Stopped before configure.'
            raise BuildError(msg)
        self.run(cmd, env=env)
        self.hooks('post_configure')
        if not cfg.dry_run:
            state_file = cfg.cmake_dir / STATE_FILENAME
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(json.dumps(state, indent=2), encoding='utf-8')

    @staticmethod
    def _only_version_changed(stored: object, cmd: list[str], env: dict[str, str]) -> bool:
        """Return whether a stored state differs from this one only in the version suffix."""
        if not isinstance(stored, dict) or stored.get('env') != env:
            return False
        old = stored.get('command')
        if not isinstance(old, list) or len(old) != len(cmd):
            return False
        prefix = VERSION_SUFFIX_DEFINE
        return all(
            a == b or (str(a).startswith(prefix) and b.startswith(prefix))
            for a, b in zip(old, cmd, strict=True)
        )

    def phase_build(self) -> None:
        """Compile with ninja through cmake."""
        cfg = self.config
        self.step('5/8 Build')
        jobs, cores, memory = self._jobs()
        found = f'{cores} CPU cores'
        if memory is not None:
            found += f' and {memory / 2**30:.0f} GB of memory'
        if cfg.jobs:
            why = ' (--jobs)'
        elif jobs < cores:
            why = ', one per 2 GB of memory so the compilers do not run out of it'
        else:
            why = ''
        self.say(
            f'{found} detected; compiling with [bold]{jobs}[/bold] jobs{why}. '
            'A full VTK build takes tens of minutes to hours.'
        )
        jobs = self._ask_jobs(jobs)
        self.run(
            ['cmake', '--build', str(cfg.cmake_dir), '--parallel', str(jobs)],
            env=self.configure_env(),
            log=cfg.build_dir / 'build.log',
        )
        if self.launcher is not None:
            self.run([self.launcher.name, '--show-stats'], fatal=False, passthrough=True)
        self.hooks('post_build')

    def _jobs(self) -> tuple[int, int, int | None]:
        """Return the job count, the CPU cores and the physical memory it was chosen from."""
        cores = os.cpu_count() or 1
        memory = _tools.physical_memory()
        return self.config.jobs or _tools.default_jobs(cores, memory), cores, memory

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

    def _macos_targets(self) -> list[str]:
        """Return the ``CMAKE_OSX_DEPLOYMENT_TARGET`` values from the manifest and ``-D``."""
        entries = [
            *self.manifest.cmake_common,
            *self.manifest.cmake_os['macos'],
            *self.config.defines,
        ]
        return [
            value
            for key, _, value in (e.partition('=') for e in entries)
            if key.split(':')[0] == 'CMAKE_OSX_DEPLOYMENT_TARGET'
        ]

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
        targets = self._macos_targets()
        target = targets[-1] if targets else DEFAULT_MACOS_TARGET
        return {'_PYTHON_HOST_PLATFORM': f'macosx-{target}-{arch}', 'ARCHFLAGS': f'-arch {arch}'}

    def phase_wheel(self) -> None:
        """Package the build tree into a wheel."""
        cfg = self.config
        self.step('6/8 Wheel')
        self.say(f'The wheel is written to [bold]{escape(str(cfg.dist_dir))}[/bold].')
        if not self.confirm('Build the wheel?'):
            msg = 'Stopped before packaging.'
            raise BuildError(msg)
        env = self.wheel_env()
        if not cfg.verbose:
            given = self.env.get('PYTHONWARNINGS', '')
            env['PYTHONWARNINGS'] = ','.join(filter(None, (given, *_SETUPTOOLS_WARNINGS)))
        self.run([self.python, 'setup.py', 'bdist_wheel'], cwd=cfg.cmake_dir, env=env)
        self.hooks('post_wheel')
        self.wheel = self._newest_wheel(cfg.dist_dir)

    def _newest_wheel(self, folder: Path, *, since: float | None = None) -> Path | None:
        """Return the most recently written wheel in a folder, optionally newer than ``since``."""
        wheels = sorted(folder.glob('*.whl'), key=lambda p: p.stat().st_mtime)
        if since is not None:
            wheels = [w for w in wheels if w.stat().st_mtime >= since]
        if not wheels:
            if self.config.dry_run:
                return None
            msg = f'No new wheel found in {folder}.' if since else f'No wheel found in {folder}.'
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
            [
                'delocate-wheel',
                '--require-archs',
                arch,
                '-w',
                out,
                *(['-v'] if self.config.verbose else []),
                str(wheel),
            ],
            {'DYLD_LIBRARY_PATH': str(libdir)},
        )

    def phase_repair(self) -> None:
        """Bundle the shared libraries into the wheel."""
        cfg = self.config
        self.step('7/8 Repair')
        if not cfg.repair:
            self.say('Skipped (--no-repair).')
            return
        self.say(
            'Repairing bundles the libraries the wheel links against from outside it, '
            'so it installs on other machines.'
        )
        if cfg.os_name == 'linux':
            self.say(
                "[dim]On Linux the wheel is tagged for this machine's glibc; portable manylinux "
                'wheels need a manylinux container.[/dim]'
            )
        wheel = self.wheel or cfg.dist_dir / f'{cfg.backend}-*.whl'
        cmd, env = self.repair_command(wheel)
        stamp = cfg.wheelhouse / _REPAIR_STAMP
        fingerprint = None
        if self.wheel is not None and not cfg.dry_run:
            fingerprint = _wheel_fingerprint(self.wheel, cmd)
            repaired = _repaired_wheel(stamp, fingerprint)
            if repaired is not None:
                self.say(
                    f'Skipped: [bold]{escape(str(repaired))}[/bold] was repaired from '
                    'this same wheel.'
                )
                self.wheel = repaired
                return
        if not self.confirm('Repair the wheel?'):
            self.say('Skipped.')
            return
        self.run(cmd, env=env)
        if not cfg.dry_run:
            self.wheel = self._newest_wheel(cfg.wheelhouse)
            if fingerprint is not None and self.wheel is not None:
                record = {'fingerprint': fingerprint, 'wheel': self.wheel.name}
                stamp.write_text(json.dumps(record), encoding='utf-8')

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
        wheel = self.wheel or (cfg.wheelhouse if cfg.repair else cfg.dist_dir) / (
            f'{cfg.backend}-*.whl'
        )
        cmd = self.install_command(wheel)
        if cmd is None:
            self.say(f'No pip or uv found; install the wheel yourself: {escape(str(wheel))}')
            return
        wanted = cfg.install or self.confirm(
            f'Install {wheel.name} into {sys.prefix}?', default=False
        )
        if not wanted:
            self.say(f'Not installed. To install later:\n  [cyan]{escape(_join(cmd))}[/cyan]')
            return
        self.run(cmd)
        self.run([self.python, '-c', f'import {self.manifest.import_name}'])

    def build_in_manylinux(self) -> Path | None:
        """Run the build inside a manylinux container, then install on the host.

        Returns
        -------
        Path or None
            The repaired wheel, or ``None`` on a dry run.

        """
        cfg = self.config
        image = cfg.manylinux or _manylinux.default_image()
        if cfg.compiler_cache == 'sccache':
            msg = '--manylinux builds use ccache; pass --compiler-cache ccache, auto or none.'
            raise BuildError(msg)
        self.say(
            f'The build runs inside the [bold]{image}[/bold] container through Docker, so '
            'the wheel is portable across Linux machines. The source is prepared here, then '
            'the container installs the build dependencies and runs the remaining steps with '
            f'[bold]{cfg.build_dir}[/bold] mounted as its build tree.'
        )
        docker = _tools.Tool('docker', ('docker',), instructions=_manylinux.DOCKER_INSTRUCTIONS)
        if not _tools.detect(docker, self.env):
            msg = f'docker was not found on PATH. {docker.instructions}'
            raise BuildError(msg)
        if not cfg.dry_run:
            probe = subprocess.run(
                [str(docker.path), 'info'], capture_output=True, text=True, check=False
            )
            if probe.returncode != 0:
                msg = 'Docker is installed but its daemon is not running or not reachable.'
                detail = (probe.stderr or '').strip()
                if detail:
                    msg += f'\n{detail}'
                raise BuildError(msg)
        self.print_plan()
        self.confirm_start()
        self.phase_source()
        if cfg.install and cfg.os_name == 'linux':
            self._check_installer()
            self.confirm_installs(self.install_plan([]))
        mounts: dict[str, Path] = {_manylinux.WORK: cfg.build_dir, _manylinux.SOURCE: cfg.src_dir}
        if cfg.compiler_cache != 'none':
            cache_dir = self.compiler_cache_dir('CCACHE_DIR')
            mounts[_manylinux.CACHE] = cache_dir or cfg.build_dir / 'compiler-cache'
        package_mount = f'{_manylinux.PYVISTA_ROOT}/pyvista'
        script = _manylinux.container_script(
            python=_manylinux.container_python(),
            packages=self.manifest.manylinux_packages,
            requirements=_manylinux.pyvista_requirements(),
            config=json.dumps(self.container_config()),
            ccache=cfg.compiler_cache != 'none',
            owner=_manylinux.file_owner(),
            paths=list(mounts),
            quiet=not cfg.verbose,
        )
        mounts[package_mount] = _manylinux.pyvista_package()
        cmd = _manylinux.docker_command(
            image, mounts=mounts, script=script, interactive=cfg.prompting, env=self.env
        )
        if not self.confirm('Run the build in the container? (n stops)'):
            msg = 'Stopped before the container build.'
            raise BuildError(msg)
        if not cfg.dry_run:
            for inside, folder in mounts.items():
                if inside != package_mount:
                    folder.mkdir(parents=True, exist_ok=True)
        started = time.time()
        self.run(cmd, passthrough=True)
        if not cfg.dry_run:
            folder = cfg.wheelhouse if cfg.repair else cfg.dist_dir
            self.wheel = self._newest_wheel(folder, since=started)
        if cfg.os_name == 'linux':
            self.phase_install()
        else:
            self.say(
                f'The wheel targets Linux, so it is not installed on this {cfg.os_name} host.'
            )
        self.summary()
        return self.wheel

    def container_config(self) -> dict[str, object]:
        """Return the container's build configuration, with host paths mapped to the mounts.

        The version suffix is computed here, so the container never reads the checkout's git
        metadata, which may point at host paths it cannot see.

        Returns
        -------
        dict[str, object]
            :class:`BuildConfig` fields for ``python -m pyvista._build``.

        """
        cfg = self.config
        cache = cfg.compiler_cache != 'none'
        return {
            'backend': cfg.backend,
            'build_dir': _manylinux.WORK,
            'source': _manylinux.SOURCE,
            'with_': cfg.with_,
            'without': cfg.without,
            'minimal': cfg.minimal,
            'build_type': cfg.build_type,
            'jobs': cfg.jobs,
            'defines': cfg.defines,
            'options': cfg.options,
            'clean': cfg.clean,
            'reconfigure': cfg.reconfigure,
            'version_suffix': self.version_suffix() or '',
            'repair': cfg.repair,
            'install': False,
            'yes': True,
            'interactive': cfg.prompting,
            'verbose': cfg.verbose,
            'in_container': True,
            'os_name': 'linux',
            'compiler_cache': cfg.compiler_cache,
            'cache_dir': _manylinux.CACHE if cache else None,
        }

    def summary(self) -> None:
        """Print where things ended up."""
        cfg = self.config
        self.console.rule('[bold green]Done')
        if self.wheel is not None:
            self.say(f'Wheel: [bold]{escape(str(self.wheel))}[/bold]')
        self.say(f'Source: {escape(str(cfg.src_dir))}\nBuild tree: {escape(str(cfg.cmake_dir))}')
        again = 'the same command without --clean' if cfg.clean else 'the same command'
        self.say(f'Edit the source and run {again} again for an incremental rebuild.')
