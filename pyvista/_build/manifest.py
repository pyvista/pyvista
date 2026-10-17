"""Backend build manifest (``pyvista-build.toml``) schema and resolution."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import tomllib
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from pathlib import Path

MANIFEST_FILENAME = 'pyvista-build.toml'
MANIFEST_VERSION = 1
OS_NAMES = ('linux', 'macos', 'windows')
HOOK_NAMES = ('post_configure', 'post_build', 'post_wheel')


class ManifestError(ValueError):
    """Raised when a manifest is missing, malformed, or self-contradictory."""


@dataclass(frozen=True)
class Group:
    """Named set of CMake cache entries that can be switched on or off together."""

    name: str
    help: str
    default: bool
    cmake: tuple[str, ...]
    cmake_off: tuple[str, ...] = ()
    os: tuple[str, ...] = OS_NAMES
    conflicts: tuple[str, ...] = ()
    required: bool = False


@dataclass(frozen=True)
class Option:
    """Scalar build knob rendered into a CMake cache entry or an environment variable."""

    name: str
    help: str
    default: bool | int | str
    cmake: str | None = None
    env: str | None = None
    choices: tuple[str, ...] = ()

    def render(self, *, value: bool | int | str) -> str | None:
        """Render the CMake cache entry for a value.

        Parameters
        ----------
        value : bool | int | str
            Value to substitute; booleans become ``ON``/``OFF``.

        Returns
        -------
        str | None
            ``KEY=VALUE`` cache entry, or ``None`` for an environment-only option.

        """
        if self.cmake is None:
            return None
        text = ('ON' if value else 'OFF') if isinstance(value, bool) else str(value)
        return self.cmake.format(value=text)

    def render_env(self, *, value: bool | int | str) -> tuple[str, str] | None:
        """Render the environment variable for a value.

        Parameters
        ----------
        value : bool | int | str
            Value to export; booleans become ``1``/``0``.

        Returns
        -------
        tuple[str, str] | None
            Variable name and value, or ``None`` for a CMake-only option.

        """
        if self.env is None:
            return None
        text = ('1' if value else '0') if isinstance(value, bool) else str(value)
        return self.env, text

    def coerce(self, raw: str) -> bool | int | str:
        """Convert a command-line string to this option's value type.

        Parameters
        ----------
        raw : str
            Text as typed on the command line.

        Returns
        -------
        bool | int | str
            Value of the same type as :attr:`default`.

        """
        if isinstance(self.default, bool):
            lowered = raw.strip().lower()
            if lowered in ('1', 'true', 'on', 'yes'):
                return True
            if lowered in ('0', 'false', 'off', 'no'):
                return False
            msg = f'Option {self.name!r} expects a boolean, got {raw!r}.'
            raise ManifestError(msg)
        if isinstance(self.default, int):
            try:
                return int(raw)
            except ValueError:
                msg = f'Option {self.name!r} expects an integer, got {raw!r}.'
                raise ManifestError(msg) from None
        if self.choices and raw not in self.choices:
            msg = f'Option {self.name!r} must be one of {list(self.choices)}, got {raw!r}.'
            raise ManifestError(msg)
        return raw


@dataclass(frozen=True)
class Manifest:
    """Parsed ``pyvista-build.toml``."""

    name: str
    import_name: str
    default_ref: str
    url: str | None
    version_command: str | None
    tools: dict[str, str]
    cmake_common: tuple[str, ...]
    cmake_os: dict[str, tuple[str, ...]]
    init_cache: dict[str, str]
    groups: dict[str, Group]
    options: dict[str, Option]
    hooks: dict[str, tuple[str, ...]]
    path: Path | None = field(default=None, compare=False)

    def groups_for(self, os_name: str) -> list[Group]:
        """Return the groups that apply on one operating system, in manifest order.

        Parameters
        ----------
        os_name : str
            ``'linux'``, ``'macos'`` or ``'windows'``.

        Returns
        -------
        list[Group]
            Groups whose ``os`` list includes ``os_name``.

        """
        return [g for g in self.groups.values() if os_name in g.os]


def _expect(table: dict[str, Any], key: str, type_: type, *, where: str) -> Any:
    """Return ``table[key]`` checked against ``type_``."""
    if key not in table:
        msg = f'{where}: missing required key {key!r}.'
        raise ManifestError(msg)
    value = table[key]
    if not isinstance(value, type_) or (type_ is int and isinstance(value, bool)):
        msg = f'{where}: {key!r} must be {type_.__name__}, got {type(value).__name__}.'
        raise ManifestError(msg)
    return value


def _str_list(
    table: dict[str, Any], key: str, *, where: str, default: list[str] | None = None
) -> tuple[str, ...]:
    """Return ``table[key]`` as a tuple of strings, or ``default`` when absent."""
    value = table.get(key, [] if default is None else default)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        msg = f'{where}: {key!r} must be a list of strings.'
        raise ManifestError(msg)
    return tuple(value)


def _parse_group(name: str, table: dict[str, Any]) -> Group:
    """Build a :class:`Group` from its TOML table."""
    where = f'[groups.{name}]'
    os_names = _str_list(table, 'os', where=where, default=list(OS_NAMES))
    unknown = set(os_names) - set(OS_NAMES)
    if unknown:
        msg = f'{where}: unknown os {sorted(unknown)}; expected {list(OS_NAMES)}.'
        raise ManifestError(msg)
    return Group(
        name=name,
        help=_expect(table, 'help', str, where=where),
        default=bool(table.get('default', False)),
        cmake=_str_list(table, 'cmake', where=where),
        cmake_off=_str_list(table, 'cmake_off', where=where),
        os=os_names,
        conflicts=_str_list(table, 'conflicts', where=where),
        required=bool(table.get('required', False)),
    )


def _parse_option(name: str, table: dict[str, Any]) -> Option:
    """Build an :class:`Option` from its TOML table."""
    where = f'[options.{name}]'
    default = table.get('default')
    if not isinstance(default, (bool, int, str)):
        msg = f'{where}: "default" must be a bool, int or str.'
        raise ManifestError(msg)
    cmake = table.get('cmake')
    env = table.get('env')
    if (cmake is None) == (env is None):
        msg = f'{where}: set exactly one of "cmake" or "env".'
        raise ManifestError(msg)
    if cmake is not None and '{value}' not in cmake:
        msg = f'{where}: "cmake" must contain the {{value}} placeholder.'
        raise ManifestError(msg)
    return Option(
        name=name,
        help=_expect(table, 'help', str, where=where),
        default=default,
        cmake=cmake,
        env=env,
        choices=_str_list(table, 'choices', where=where),
    )


def parse_manifest(data: dict[str, Any], path: Path | None = None) -> Manifest:
    """Validate a decoded TOML document.

    Parameters
    ----------
    data : dict[str, Any]
        Decoded ``pyvista-build.toml``.

    path : Path, optional
        Where ``data`` came from, kept for messages.

    Returns
    -------
    Manifest
        Validated manifest.

    """
    version = data.get('manifest_version')
    if version != MANIFEST_VERSION:
        msg = (
            f'Unsupported manifest_version {version!r}; '
            f'this PyVista reads version {MANIFEST_VERSION}.'
        )
        raise ManifestError(msg)
    backend = _expect(data, 'backend', dict, where='manifest')
    tools = data.get('tools', {})
    if not isinstance(tools, dict) or not all(isinstance(v, str) for v in tools.values()):
        msg = '[tools]: every entry must be a version specifier string.'
        raise ManifestError(msg)
    cmake = data.get('cmake', {})
    if not isinstance(cmake, dict):
        msg = '[cmake] must be a table.'
        raise ManifestError(msg)
    init_cache = cmake.get('init_cache', {})
    if not isinstance(init_cache, dict) or set(init_cache) - set(OS_NAMES):
        msg = f'[cmake]: "init_cache" must be a table keyed by {list(OS_NAMES)}.'
        raise ManifestError(msg)
    groups = {name: _parse_group(name, t) for name, t in data.get('groups', {}).items()}
    for group in groups.values():
        missing = set(group.conflicts) - set(groups)
        if missing:
            msg = f'[groups.{group.name}]: conflicts with unknown group(s) {sorted(missing)}.'
            raise ManifestError(msg)
    options = {name: _parse_option(name, t) for name, t in data.get('options', {}).items()}
    hooks_table = data.get('hooks', {})
    unknown_hooks = set(hooks_table) - set(HOOK_NAMES)
    if unknown_hooks:
        msg = f'[hooks]: unknown hook(s) {sorted(unknown_hooks)}; expected {list(HOOK_NAMES)}.'
        raise ManifestError(msg)
    return Manifest(
        name=_expect(backend, 'name', str, where='[backend]'),
        import_name=_expect(backend, 'import_name', str, where='[backend]'),
        default_ref=_expect(backend, 'default_ref', str, where='[backend]'),
        url=backend.get('url'),
        version_command=backend.get('version_command'),
        tools=dict(tools),
        cmake_common=_str_list(cmake, 'common', where='[cmake]'),
        cmake_os={name: _str_list(cmake, name, where='[cmake]') for name in OS_NAMES},
        init_cache=dict(init_cache),
        groups=groups,
        options=options,
        hooks={name: _str_list(hooks_table, name, where='[hooks]') for name in HOOK_NAMES},
        path=path,
    )


def load_manifest(path: Path) -> Manifest:
    """Read and validate a manifest file.

    Parameters
    ----------
    path : Path
        ``pyvista-build.toml`` to read.

    Returns
    -------
    Manifest
        Validated manifest.

    """
    try:
        data = tomllib.loads(path.read_text(encoding='utf-8'))
    except tomllib.TOMLDecodeError as err:
        msg = f'{path}: {err}'
        raise ManifestError(msg) from None
    return parse_manifest(data, path)


def resolve_groups(
    manifest: Manifest,
    os_name: str,
    *,
    with_: list[str] | None = None,
    without: list[str] | None = None,
    all_: bool = False,
    minimal: bool = False,
) -> list[str]:
    """Return the enabled group names after applying the command-line selection.

    Parameters
    ----------
    manifest : Manifest
        Manifest declaring the groups.

    os_name : str
        Operating system the build runs on.

    with_ : list[str], optional
        Groups to switch on.

    without : list[str], optional
        Groups to switch off.

    all_ : bool, default: False
        Start from every group available on ``os_name``.

    minimal : bool, default: False
        Start from the required groups only.

    Returns
    -------
    list[str]
        Enabled group names in manifest order.

    """
    available = {g.name: g for g in manifest.groups_for(os_name)}
    unknown = set(with_ or []) | set(without or [])
    unknown -= set(manifest.groups)
    if unknown:
        msg = f'Unknown group(s) {sorted(unknown)}; available: {sorted(manifest.groups)}.'
        raise ManifestError(msg)
    unavailable = (set(with_ or []) & set(manifest.groups)) - set(available)
    if unavailable:
        msg = f'Group(s) {sorted(unavailable)} are not available on {os_name}.'
        raise ManifestError(msg)
    if all_ and minimal:
        msg = '--all and --minimal are mutually exclusive.'
        raise ManifestError(msg)

    enabled: list[str] = []
    for group in available.values():
        on = group.required or all_ or (group.default and not minimal)
        if group.name in (without or []):
            if group.required:
                msg = f'Group {group.name!r} is required and cannot be disabled.'
                raise ManifestError(msg)
            on = False
        if group.name in (with_ or []):
            on = True
        if on:
            enabled.append(group.name)

    for name in enabled:
        clash = sorted(set(available[name].conflicts) & set(enabled))
        if clash:
            msg = f'Group {name!r} conflicts with {clash}; disable one with --without.'
            raise ManifestError(msg)
    return enabled


def resolve_options(
    manifest: Manifest, assignments: list[str] | None
) -> dict[str, bool | int | str]:
    """Return every option's value after applying ``NAME=VALUE`` assignments.

    Parameters
    ----------
    manifest : Manifest
        Manifest declaring the options.

    assignments : list[str], optional
        ``NAME=VALUE`` strings from the command line.

    Returns
    -------
    dict[str, bool | int | str]
        Value for every option in the manifest.

    """
    values: dict[str, bool | int | str] = {o.name: o.default for o in manifest.options.values()}
    for item in assignments or []:
        name, sep, raw = item.partition('=')
        if not sep or name not in manifest.options:
            msg = (
                f'Unknown option {item!r}; use NAME=VALUE with one of {sorted(manifest.options)}.'
            )
            raise ManifestError(msg)
        values[name] = manifest.options[name].coerce(raw)
    return values


def cmake_entries(
    manifest: Manifest,
    os_name: str,
    *,
    groups: list[str],
    options: dict[str, bool | int | str],
) -> list[str]:
    """Return the ``KEY=VALUE`` cache entries the manifest contributes to this build.

    Parameters
    ----------
    manifest : Manifest
        Manifest to read.

    os_name : str
        Operating system the build runs on.

    groups : list[str]
        Enabled group names; other groups contribute their ``cmake_off`` entries.

    options : dict[str, bool | int | str]
        Value for every option.

    Returns
    -------
    list[str]
        Cache entries without the ``-D`` prefix.

    """
    entries = list(manifest.cmake_common) + list(manifest.cmake_os.get(os_name, ()))
    for group in manifest.groups_for(os_name):
        entries.extend(group.cmake if group.name in groups else group.cmake_off)
    for name, value in options.items():
        entry = manifest.options[name].render(value=value)
        if entry is not None:
            entries.append(entry)
    return entries


def option_environment(manifest: Manifest, options: dict[str, bool | int | str]) -> dict[str, str]:
    """Return the environment variables the environment-backed options export.

    Parameters
    ----------
    manifest : Manifest
        Manifest to read.

    options : dict[str, bool | int | str]
        Value for every option.

    Returns
    -------
    dict[str, str]
        Variable name to value.

    """
    env: dict[str, str] = {}
    for name, value in options.items():
        pair = manifest.options[name].render_env(value=value)
        if pair is not None:
            env[pair[0]] = pair[1]
    return env
