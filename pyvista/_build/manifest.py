"""Backend build manifest (``pyvista-build.toml``) schema and resolution."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import re
import string
import tomllib
from typing import TYPE_CHECKING
from typing import Any

from packaging.specifiers import InvalidSpecifier
from packaging.specifiers import SpecifierSet

if TYPE_CHECKING:
    from pathlib import Path

MANIFEST_FILENAME = 'pyvista-build.toml'
MANIFEST_VERSION = 1
OS_NAMES = ('linux', 'macos', 'windows')
HOOK_NAMES = ('post_configure', 'post_build', 'post_wheel')
PLACEHOLDERS = ('python', 'src_dir', 'build_dir', 'dist_dir', 'wheelhouse')
TOP_LEVEL_KEYS = ('manifest_version', 'backend', 'tools', 'cmake', 'groups', 'options', 'hooks')
_CMAKE_ENTRY = re.compile(r'[A-Za-z_][A-Za-z0-9_.+-]*(:[A-Za-z]+)?=')


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


_TOML_TYPES = {
    dict: 'a table',
    list: 'an array',
    str: 'a string',
    bool: 'a boolean',
    int: 'an integer',
    float: 'a float',
}


def _expect(table: dict[str, Any], key: str, type_: type, *, where: str) -> Any:
    """Return ``table[key]`` checked against ``type_``."""
    if key not in table:
        msg = f'{where}: missing required key {key!r}.'
        raise ManifestError(msg)
    value = table[key]
    if not isinstance(value, type_) or (type_ is int and isinstance(value, bool)):
        expected = _TOML_TYPES.get(type_, type_.__name__)
        got = _TOML_TYPES.get(type(value), type(value).__name__)
        msg = f'{where}: {key!r} must be {expected}, got {got}.'
        raise ManifestError(msg)
    return value


def _optional(table: dict[str, Any], key: str, type_: type, *, where: str, default: Any) -> Any:
    """Return ``table[key]`` checked against ``type_``, or ``default`` when absent."""
    if key not in table:
        return default
    return _expect(table, key, type_, where=where)


def _str_list(
    table: dict[str, Any], key: str, *, where: str, default: list[str] | None = None
) -> tuple[str, ...]:
    """Return ``table[key]`` as a tuple of strings, or ``default`` when absent."""
    value = table.get(key, [] if default is None else default)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        msg = f'{where}: {key!r} must be a list of strings.'
        raise ManifestError(msg)
    return tuple(value)


def _table(table: dict[str, Any], key: str, *, where: str) -> dict[str, Any]:
    """Return ``table[key]`` as a TOML table, empty when absent."""
    return _optional(table, key, dict, where=where, default={})


def _check_keys(table: dict[str, Any], allowed: tuple[str, ...], *, where: str) -> None:
    """Reject keys outside ``allowed``, so a typo is an error rather than ignored."""
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        msg = f'{where}: unknown key(s) {unknown}; expected some of {list(allowed)}.'
        raise ManifestError(msg)


def _check_placeholders(template: str, allowed: tuple[str, ...], *, where: str) -> None:
    """Reject ``{name}`` fields outside ``allowed``; ``{{`` and ``}}`` are literal braces."""
    try:
        names = {f for _, f, _, _ in string.Formatter().parse(template) if f is not None}
    except ValueError as err:
        msg = f'{where}: {template!r}: {err}; write a literal brace as {{{{ or }}}}.'
        raise ManifestError(msg) from None
    unknown = sorted(names - set(allowed))
    if unknown:
        msg = (
            f'{where}: {template!r} uses unknown placeholder(s) {unknown}; '
            f'expected some of {list(allowed)}. Write a literal brace as {{{{ or }}}}.'
        )
        raise ManifestError(msg)


def _cmake_entries(table: dict[str, Any], key: str, *, where: str) -> tuple[str, ...]:
    """Return ``table[key]`` as CMake cache entries, each ``KEY=VALUE`` or ``KEY:TYPE=VALUE``."""
    entries = _str_list(table, key, where=where)
    for entry in entries:
        _check_cmake_entry(entry, where=f'{where} {key!r}')
    return entries


def _check_cmake_entry(entry: str, *, where: str) -> None:
    """Raise unless ``entry`` is a CMake cache entry without the ``-D`` prefix."""
    if entry.startswith('-D'):
        msg = f'{where}: write {entry!r} without the -D prefix.'
        raise ManifestError(msg)
    if not _CMAKE_ENTRY.match(entry):
        msg = f'{where}: {entry!r} must be KEY=VALUE or KEY:TYPE=VALUE.'
        raise ManifestError(msg)


def _parse_group(name: str, table: Any) -> Group:
    """Build a :class:`Group` from its TOML table."""
    where = f'[groups.{name}]'
    if not isinstance(table, dict):
        msg = f'{where} must be a table.'
        raise ManifestError(msg)
    _check_keys(
        table,
        ('help', 'default', 'cmake', 'cmake_off', 'os', 'conflicts', 'required'),
        where=where,
    )
    os_names = _str_list(table, 'os', where=where, default=list(OS_NAMES))
    unknown = set(os_names) - set(OS_NAMES)
    if unknown:
        msg = f'{where}: unknown os {sorted(unknown)}; expected {list(OS_NAMES)}.'
        raise ManifestError(msg)
    return Group(
        name=name,
        help=_expect(table, 'help', str, where=where),
        default=_optional(table, 'default', bool, where=where, default=False),
        cmake=_cmake_entries(table, 'cmake', where=where),
        cmake_off=_cmake_entries(table, 'cmake_off', where=where),
        os=os_names,
        conflicts=_str_list(table, 'conflicts', where=where),
        required=_optional(table, 'required', bool, where=where, default=False),
    )


def _parse_option(name: str, table: Any) -> Option:
    """Build an :class:`Option` from its TOML table."""
    where = f'[options.{name}]'
    if not isinstance(table, dict):
        msg = f'{where} must be a table.'
        raise ManifestError(msg)
    _check_keys(table, ('help', 'default', 'cmake', 'env', 'choices'), where=where)
    if 'default' not in table:
        msg = f"{where}: missing required key 'default'."
        raise ManifestError(msg)
    default = table['default']
    if not isinstance(default, (bool, int, str)):
        msg = f"{where}: 'default' must be a bool, int or str."
        raise ManifestError(msg)
    cmake = _optional(table, 'cmake', str, where=where, default=None)
    env = _optional(table, 'env', str, where=where, default=None)
    if (cmake is None) == (env is None):
        msg = f"{where}: set exactly one of 'cmake' or 'env'."
        raise ManifestError(msg)
    if cmake is not None:
        if '{value}' not in cmake:
            msg = f"{where}: 'cmake' must contain the {{value}} placeholder."
            raise ManifestError(msg)
        _check_placeholders(cmake, ('value',), where=where)
        _check_cmake_entry(cmake.replace('{value}', 'x'), where=f"{where} 'cmake'")
    choices = _str_list(table, 'choices', where=where)
    if choices and not isinstance(default, str):
        msg = f"{where}: 'choices' needs a string 'default'."
        raise ManifestError(msg)
    if choices and default not in choices:
        msg = f'{where}: default {default!r} is not one of the choices {list(choices)}.'
        raise ManifestError(msg)
    return Option(
        name=name,
        help=_expect(table, 'help', str, where=where),
        default=default,
        cmake=cmake,
        env=env,
        choices=choices,
    )


def parse_manifest(data: dict[str, Any], path: Path | None = None) -> Manifest:
    """Validate a decoded TOML document.

    Parameters
    ----------
    data : dict[str, Any]
        Decoded ``pyvista-build.toml``.

    path : Path, optional
        Where ``data`` came from.

    Returns
    -------
    Manifest
        Validated manifest.

    """
    version = _expect(data, 'manifest_version', int, where='manifest')
    if version != MANIFEST_VERSION:
        msg = (
            f'Unsupported manifest_version {version!r}; '
            f'this PyVista reads version {MANIFEST_VERSION}.'
        )
        raise ManifestError(msg)
    _check_keys(data, TOP_LEVEL_KEYS, where='manifest')
    backend = _expect(data, 'backend', dict, where='manifest')
    _check_keys(
        backend,
        ('name', 'import_name', 'default_ref', 'url', 'version_command'),
        where='[backend]',
    )
    backend_name = _expect(backend, 'name', str, where='[backend]')
    import_name = _optional(backend, 'import_name', str, where='[backend]', default=backend_name)
    if not all(part.isidentifier() for part in import_name.split('.')):
        msg = f"[backend]: 'import_name' must be a Python module name, got {import_name!r}."
        raise ManifestError(msg)
    version_command = _optional(backend, 'version_command', str, where='[backend]', default=None)
    if version_command is not None:
        _check_placeholders(version_command, PLACEHOLDERS, where='[backend] version_command')
    tools = _table(data, 'tools', where='manifest')
    _check_keys(tools, ('cmake', 'ninja'), where='[tools]')
    for name in tools:
        spec = _expect(tools, name, str, where='[tools]')
        try:
            SpecifierSet(spec)
        except InvalidSpecifier:
            msg = f'[tools] {name}: {spec!r} is not a version specifier such as ">=3.22".'
            raise ManifestError(msg) from None
    cmake = _table(data, 'cmake', where='manifest')
    _check_keys(cmake, ('common', 'init_cache', *OS_NAMES), where='[cmake]')
    init_cache = _table(cmake, 'init_cache', where='[cmake]')
    if set(init_cache) - set(OS_NAMES) or not all(isinstance(v, str) for v in init_cache.values()):
        msg = f"[cmake]: 'init_cache' must map {list(OS_NAMES)} to file paths."
        raise ManifestError(msg)
    groups = {
        name: _parse_group(name, t) for name, t in _table(data, 'groups', where='manifest').items()
    }
    for group in groups.values():
        missing = set(group.conflicts) - set(groups)
        if missing:
            msg = f'[groups.{group.name}]: conflicts with unknown group(s) {sorted(missing)}.'
            raise ManifestError(msg)
    for group in groups.values():
        for other in group.conflicts:
            if group.name not in groups[other].conflicts:
                msg = (
                    f"[groups.{other}]: list {group.name!r} in 'conflicts' too; "
                    f'[groups.{group.name}] conflicts with it.'
                )
                raise ManifestError(msg)
    options = {
        name: _parse_option(name, t)
        for name, t in _table(data, 'options', where='manifest').items()
    }
    hooks_table = _table(data, 'hooks', where='manifest')
    unknown_hooks = set(hooks_table) - set(HOOK_NAMES)
    if unknown_hooks:
        msg = f'[hooks]: unknown hook(s) {sorted(unknown_hooks)}; expected {list(HOOK_NAMES)}.'
        raise ManifestError(msg)
    hooks = {name: _str_list(hooks_table, name, where='[hooks]') for name in HOOK_NAMES}
    for name, templates in hooks.items():
        for template in templates:
            _check_placeholders(template, PLACEHOLDERS, where=f'[hooks] {name}')
    return Manifest(
        name=backend_name,
        import_name=import_name,
        default_ref=_expect(backend, 'default_ref', str, where='[backend]'),
        url=_optional(backend, 'url', str, where='[backend]', default=None),
        version_command=version_command,
        tools=dict(tools),
        cmake_common=_cmake_entries(cmake, 'common', where='[cmake]'),
        cmake_os={name: _cmake_entries(cmake, name, where='[cmake]') for name in OS_NAMES},
        init_cache=dict(init_cache),
        groups=groups,
        options=options,
        hooks=hooks,
        path=path,
    )


def load_manifest(path: Path, *, default_ref: str | None = None) -> Manifest:
    """Read and validate a manifest file.

    Parameters
    ----------
    path : Path
        ``pyvista-build.toml`` to read.

    default_ref : str, optional
        ``[backend] default_ref`` to use when the file sets none.

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
    if default_ref is not None and isinstance(data.get('backend'), dict):
        data['backend'].setdefault('default_ref', default_ref)
    try:
        return parse_manifest(data, path)
    except ManifestError as err:
        msg = f'{path}: {err}'
        raise ManifestError(msg) from None


def resolve_groups(
    manifest: Manifest,
    os_name: str,
    *,
    with_: list[str] | None = None,
    without: list[str] | None = None,
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
    enabled: list[str] = []
    for group in available.values():
        on = group.required or (group.default and not minimal)
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
        if name not in manifest.options:
            msg = f'Unknown option {name!r}; available: {sorted(manifest.options)}.'
            raise ManifestError(msg)
        if not sep:
            msg = f'Option {name!r} needs a value: write {name}=VALUE.'
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


def cmake_unset(manifest: Manifest, os_name: str, *, groups: list[str]) -> list[str]:
    """Return the cache keys a disabled group set without giving an off value.

    Parameters
    ----------
    manifest : Manifest
        Manifest to read.

    os_name : str
        Operating system the build runs on.

    groups : list[str]
        Enabled group names.

    Returns
    -------
    list[str]
        Keys of disabled groups without ``cmake_off`` that no enabled entry sets.

    """
    kept = {
        e.partition('=')[0].partition(':')[0]
        for e in cmake_entries(manifest, os_name, groups=groups, options={})
    }
    keys: list[str] = []
    for group in manifest.groups_for(os_name):
        if group.name in groups or group.cmake_off:
            continue
        for entry in group.cmake:
            key = entry.partition('=')[0].partition(':')[0]
            if key not in kept and key not in keys:
                keys.append(key)
    return keys


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
