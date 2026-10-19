"""The build itself: tools, source, groups, configure, build, wheel, repair, install."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from dataclasses import field
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import TYPE_CHECKING
from typing import cast

from rich.console import Console
from rich.highlighter import Highlighter
from rich.markup import escape
from rich.prompt import Confirm as _RichConfirm
from rich.prompt import Prompt as _RichPrompt
from rich.style import Style
from rich.table import Table
from rich.text import Text

from . import tools as _tools
from .manifest import MANIFEST_FILENAME
from .manifest import ManifestError
from .manifest import resolve_groups
from .manifest import resolve_options
from .registry import BACKENDS
from .registry import find_manifest
from .registry import known_backends
from .registry import shipped_manifest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import IO

    from .manifest import Manifest


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
    version_suffix: str | None = None
    repair: bool = True
    install: bool = False
    yes: bool = False
    interactive: bool = False
    dry_run: bool = False
    verbose: bool = False
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

        Without ``verbose``, the printed command leaves out its environment variables.

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
        try:
            if log is None:
                status = subprocess.run(resolved, cwd=cwd, env=full_env, check=False).returncode
            else:
                status = self._run_logged(resolved, cwd=cwd, env=full_env, log=log)
        except OSError as err:
            msg = f'Cannot run {cmd[0]}: {err}'
            raise BuildError(msg) from None
        if status != 0 and fatal:
            msg = f'Command failed with exit status {status}: {_join(cmd)}'
            raise BuildError(msg)

    def _run_logged(
        self, cmd: Sequence[str], *, cwd: Path | None, env: dict[str, str], log: Path
    ) -> int:
        """Run a command, writing its full output to a log and a filtered view to the console."""
        with (
            log.open('w', encoding='utf-8') as log_file,
            subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding='utf-8',
                errors='replace',
            ) as proc,
        ):
            for line in cast('IO[str]', proc.stdout):
                log_file.write(line)
                if not _NOISY_BUILD_LINE.search(line):
                    self.console.out(line, end='', highlight=False)
        if proc.returncode != 0:
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
        self.console.print(table)
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
        if cfg.install and _tools.pip_install_command() is None:
            msg = '--install needs pip or uv in this environment.'
            raise BuildError(msg)
        self.confirm_installs(self.install_plan(specs))
        if cmd is not None:
            self.run(cmd)
            if not cfg.dry_run:
                bad = [t.label for t in missing_pip if not _tools.detect(t, self.env)]
                if bad:
                    msg = f'Still not usable after install: {", ".join(bad)}.'
                    raise BuildError(msg)

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
