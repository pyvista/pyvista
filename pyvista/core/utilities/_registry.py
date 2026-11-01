"""Shared machinery for PyVista's plugin registries."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module  # noqa: ICN003  # tests patch this name
from importlib.metadata import EntryPoint
from importlib.metadata import entry_points
import threading
from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import TypedDict
from typing import TypeVar
from typing import cast

from pyvista._warn_external import warn_external

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

_V = TypeVar('_V')
_R = TypeVar('_R', bound=tuple[Any, ...])

# Marks an attribute that did not exist before a descriptor replaced it.
_MISSING: Any = object()


def handler_source(handler: object) -> str:  # numpydoc ignore=PR01
    """Return ``module.qualname`` for ``handler``, with placeholders for what is missing."""
    module = getattr(handler, '__module__', None) or '<unknown>'
    qualname = getattr(handler, '__qualname__', None) or repr(handler)
    return f'{module}.{qualname}'


class _HandlerTableState(TypedDict, Generic[_V]):
    """Snapshot of a :class:`_HandlerTable`."""

    handlers: dict[str, _V]
    sources: dict[str, str]


class _HandlerTable(Generic[_V]):
    """Registered handlers keyed by name, each with the source it came from.

    Parameters
    ----------
    label : str
        What is registered, as it appears in messages.
    builtin : callable
        Says whether a key names a handler PyVista ships.
    builtin_label : str
        How a built-in handler is described in messages.

    """

    def __init__(
        self,
        label: str = 'handler',
        *,
        builtin: Callable[[str], bool] = lambda _key: False,
        builtin_label: str = 'a built-in',
    ) -> None:  # numpydoc ignore=PR01
        self._handlers: dict[str, _V] = {}
        self._sources: dict[str, str] = {}
        self.label = label
        self._builtin = builtin
        self._builtin_label = builtin_label

    def __contains__(self, key: object) -> bool:
        return key in self._handlers

    def __iter__(self) -> Iterator[str]:
        return iter(self._handlers)

    def __len__(self) -> int:
        return len(self._handlers)

    def __bool__(self) -> bool:
        return bool(self._handlers)

    def get(self, key: str) -> _V | None:  # numpydoc ignore=PR01
        """Return the handler registered under ``key``, if any."""
        return self._handlers.get(key)

    def source(self, key: str) -> str:  # numpydoc ignore=PR01
        """Return where the handler under ``key`` came from."""
        return self._sources.get(key, '<unknown>')

    def items(self) -> list[tuple[str, _V]]:
        """Return ``(key, handler)`` pairs in registration order."""
        return list(self._handlers.items())

    def keys(self) -> set[str]:
        """Return every registered key."""
        return set(self._handlers)

    def set(self, key: str, handler: _V, source: str) -> None:  # numpydoc ignore=PR01
        """Register ``handler`` under ``key``, replacing any previous one."""
        self._handlers[key] = handler
        self._sources[key] = source

    def pop(self, key: str) -> _V | None:  # numpydoc ignore=PR01
        """Remove and return the handler under ``key``, if any."""
        self._sources.pop(key, None)
        return self._handlers.pop(key, None)

    def clear(self) -> None:
        """Remove every registration."""
        self._handlers.clear()
        self._sources.clear()

    def register(
        self, key: str, handler: _V, source: str | None = None, *, override: bool = False
    ) -> None:  # numpydoc ignore=PR01
        """Register an explicit handler; a built-in key raises and a replacement warns."""
        if not override and self._builtin(key):
            msg = (
                f'Cannot register custom {self.label} "{key}": collides with '
                f'{self._builtin_label}. Use override=True to replace it.'
            )
            raise ValueError(msg)
        if not override and key in self._handlers:
            warn_external(
                f'Registering {self.label} "{key}" replaces an existing custom '
                f'registration from {self.source(key)}.',
            )
        self.set(key, handler, handler_source(handler) if source is None else source)

    def snapshot(self) -> _HandlerTableState[_V]:
        """Copy the table for :meth:`restore`."""
        return {'handlers': dict(self._handlers), 'sources': dict(self._sources)}

    def restore(self, state: _HandlerTableState[_V]) -> None:  # numpydoc ignore=PR01
        """Replace the table with a :meth:`snapshot`."""
        self._handlers.clear()
        self._handlers.update(state['handlers'])
        self._sources.clear()
        self._sources.update(state['sources'])


class EntryPointRefusedError(ValueError):
    """Raised by a registry to refuse an entry point; it stays pending and is not marked failed."""


class _EntryPointLoaderState(TypedDict):
    """Snapshot of an :class:`_EntryPointLoader`."""

    loaded: bool
    pending: dict[str, list[EntryPoint]]
    failed: dict[str, str]


class _EntryPointLoader:
    """Scan one or more entry-point groups once and load each plugin on first request.

    Parameters
    ----------
    group : str
        Entry-point group to scan.
    *more_groups : str
        Further groups, scanned after ``group``.
    key : callable
        Maps an entry point and its group to a pending key, or to ``None`` to skip it.
    load : callable
        Receives a key and its entry points and records what they contribute.
    keep_causes : bool
        Keep the exception that failed a key for :meth:`take_cause`.

    """

    def __init__(
        self,
        group: str,
        *more_groups: str,
        key: Callable[[EntryPoint, str], str | None],
        load: Callable[[str, list[EntryPoint]], None],
        keep_causes: bool = False,
    ) -> None:  # numpydoc ignore=PR01
        self.group = group
        self._keep_causes = keep_causes
        self.groups = (group, *more_groups)
        self._key = key
        self._load = load
        self.lock = threading.RLock()
        self.loaded = False
        self.pending: dict[str, list[EntryPoint]] = {}
        self.failed: dict[str, str] = {}
        self.causes: dict[str, BaseException] = {}
        self._loading: dict[str, tuple[int, threading.Event]] = {}

    def ensure_scanned(self) -> None:
        """Read the group's metadata once, importing nothing."""
        if not self.loaded:
            with self.lock:
                self._scan()

    def _scan(self) -> None:
        """Fill the pending map from the groups' metadata, under the lock."""
        if not self.loaded:
            seen: set[tuple[str, str]] = set()
            for group in self.groups:
                for ep in entry_points(group=group):
                    identity = (str(ep.name), str(ep.value))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    key = self._key(ep, group)
                    if key is not None:
                        self.pending.setdefault(key, []).append(ep)
            self.loaded = True

    def pending_keys(self) -> tuple[str, ...]:
        """Return the keys whose plugin has not been loaded, failed ones left out."""
        self.ensure_scanned()
        with self.lock:
            return tuple(key for key in self.pending if key not in self.failed)

    def failure(self, key: str) -> str | None:  # numpydoc ignore=PR01
        """Return the message recorded when loading ``key`` failed."""
        return self.failed.get(key)

    def take_cause(self, key: str) -> BaseException | None:  # numpydoc ignore=PR01
        """Return and forget the exception that failed ``key``, so it is chained once."""
        return self.causes.pop(key, None)

    def resolve(self, key: str) -> bool:  # numpydoc ignore=PR01,RT01
        """Load the plugin pending under ``key``; ``False`` when this call loaded nothing.

        The plugin imports outside the lock, so a plugin that reaches back into
        the registry while importing cannot deadlock against the import lock.
        Another thread asking for the same key waits for that import to finish.
        """
        self.ensure_scanned()
        with self.lock:
            eps = self.pending.get(key)
            if not eps or key in self.failed:
                return False
            loading = self._loading.get(key)
            if loading is not None:
                thread, done = loading
                if thread == threading.get_ident():
                    return False
            else:
                done = threading.Event()
                self._loading[key] = (threading.get_ident(), done)
        if loading is not None:
            done.wait()
            return False
        try:
            self._load(key, eps)
        except EntryPointRefusedError:
            raise
        except Exception as exc:  # noqa: BLE001
            winner = eps[0]
            msg = (
                f'Failed to load {self.group} entry point '
                f'"{winner.name}" from {winner.value}: {exc}'
            )
            with self.lock:
                self.failed[key] = msg
                if self._keep_causes:
                    self.causes[key] = exc
            warn_external(msg)
            return False
        else:
            with self.lock:
                self.pending.pop(key, None)
        finally:
            with self.lock:
                self._loading.pop(key, None)
            done.set()
        if len(eps) > 1:
            providers = ', '.join(ep.value for ep in eps)
            warn_external(
                f'Multiple {self.group} entry points registered for '
                f'"{key}": {providers}. Using {eps[0].value}.'
            )
        return True

    def ensure_loaded(self) -> None:
        """Load every pending plugin that has not failed."""
        for key in self.pending_keys():
            self.resolve(key)

    def resolve_all(self) -> None:
        """Load every pending plugin, retrying the ones that failed before; a refusal warns."""
        self.ensure_scanned()
        with self.lock:
            self.failed.clear()
            self.causes.clear()
            keys = list(self.pending)
        for key in keys:
            try:
                self.resolve(key)
            except EntryPointRefusedError as exc:
                warn_external(str(exc))

    def reset(self) -> None:
        """Forget the scan so the next request reads the metadata again."""
        with self.lock:
            self.loaded = False
            self.pending.clear()
            self.failed.clear()
            self.causes.clear()

    def snapshot(self) -> _EntryPointLoaderState:
        """Copy the loader state for :meth:`restore`."""
        with self.lock:
            return {
                'loaded': self.loaded,
                'pending': {k: list(v) for k, v in self.pending.items()},
                'failed': dict(self.failed),
            }

    def restore(self, state: _EntryPointLoaderState) -> None:  # numpydoc ignore=PR01
        """Replace the loader state with a :meth:`snapshot`."""
        with self.lock:
            self.loaded = state['loaded']
            self.pending.clear()
            self.pending.update({k: list(v) for k, v in state['pending'].items()})
            self.failed.clear()
            self.failed.update(state['failed'])
            self.causes.clear()


def discovered_items(
    group: str, ep: EntryPoint, loaded: object, *, label: str
) -> list[tuple[str, object]]:  # numpydoc ignore=PR01
    """Return the ``(name, handler)`` pairs an entry point contributes; a mapping gives several."""
    if not isinstance(loaded, Mapping):
        return [(ep.name, loaded)]
    items: list[tuple[str, object]] = []
    for name, handler in loaded.items():
        if isinstance(name, str):
            items.append((name, handler))
        else:
            warn_external(
                f'Ignoring {group} entry point from {ep.value}: {label} names must be strings.'
            )
    return items


class _HandlerRegistryState(TypedDict):
    """Snapshot of a :class:`_HandlerRegistry`."""

    explicit: _HandlerTableState[Any]
    discovered: _HandlerTableState[Any]
    plugins: _EntryPointLoaderState


class _HandlerRegistry(Generic[_V]):
    """Handlers registered by name, explicitly or through an entry-point group.

    Parameters
    ----------
    group : str
        Entry-point group whose entry points name the handlers.
    *more_groups : str
        Further groups, scanned after ``group``.
    label : str
        What is registered, as it appears in messages (``'writer for'``).
    normalize : callable
        Maps a user-supplied name to the registry key.
    builtin : callable
        Says whether a key names a handler PyVista ships.
    builtin_label : str
        How a built-in handler is described in messages.
    accept : callable
        Returns why a discovered handler cannot serve its key, or ``None`` to accept it.
    separate_discovered : bool
        Keep entry-point handlers in their own table so explicit ones always win.

    """

    def __init__(
        self,
        group: str,
        *more_groups: str,
        label: str,
        normalize: Callable[[str], str] = str.lower,
        builtin: Callable[[str], bool] = lambda _key: False,
        builtin_label: str = 'a built-in',
        accept: Callable[[str, object], str | None] = lambda _key, _handler: None,
        separate_discovered: bool = False,
    ) -> None:  # numpydoc ignore=PR01
        self.normalize = normalize
        self._builtin = builtin
        self._accept = accept
        self.explicit: _HandlerTable[_V] = _HandlerTable(
            label, builtin=builtin, builtin_label=builtin_label
        )
        self.discovered: _HandlerTable[_V] = (
            _HandlerTable(label) if separate_discovered else self.explicit
        )
        self.plugins = _EntryPointLoader(group, *more_groups, key=self.pending_key, load=self.load)

    @property
    def group(self) -> str:
        """Return the primary entry-point group."""
        return self.plugins.group

    def pending_key(self, ep: EntryPoint, group: str) -> str | None:  # noqa: ARG002  # numpydoc ignore=PR01
        """Return the key ``ep`` claims, or ``None`` when it is empty or registered explicitly."""
        key = self.normalize(ep.name)
        if not key:
            warn_external(
                f'Ignoring {self.group} entry point from {ep.value}: name must not be empty.'
            )
            return None
        return None if key in self.explicit else key

    def load(self, key: str, eps: list[EntryPoint]) -> None:  # noqa: ARG002  # numpydoc ignore=PR01
        """Load the first entry point and record every handler it contributes."""
        winner = eps[0]
        loaded = load_entry_point(eps)
        for name, handler in discovered_items(
            self.group, winner, loaded, label=self.explicit.label
        ):
            self.discover(name, handler, winner.value)

    def discover(self, name: str, handler: object, source: str) -> None:  # numpydoc ignore=PR01
        """Record a handler an entry point contributed, unless something already claims its key."""
        key = self.normalize(name)
        if not key:
            warn_external(
                f'Ignoring {self.group} entry point from {source}: name must not be empty.'
            )
            return
        if key in self.explicit:
            return
        if key in self.discovered:
            existing = self.discovered.source(key)
            warn_external(
                f'Multiple {self.group} providers registered for "{key}": '
                f'{existing}, {source}. Using {existing}.'
            )
            return
        reason = self._accept(key, handler)
        if reason is not None:
            warn_external(f'Ignoring {self.group} entry point "{key}" from {source}: {reason}')
            return
        self.discovered.set(key, cast('_V', handler), source)

    def register(
        self, name: str, handler: _V, source: str | None = None, *, override: bool = False
    ) -> str:  # numpydoc ignore=PR01
        """Register an explicit handler and return its key."""
        key = self.normalize(name)
        self.explicit.register(key, handler, source, override=override)
        return key

    def get(self, name: str) -> _V | None:  # numpydoc ignore=PR01
        """Return the handler under ``name``, loading its plugin on first request."""
        key = self.normalize(name)
        handler = self.explicit.get(key)
        if handler is None:
            self.plugins.resolve(key)
            handler = self.explicit.get(key) or self.discovered.get(key)
        return handler

    def keys(self) -> set[str]:
        """Return every key that is registered or still pending, loading nothing."""
        return self.explicit.keys() | self.discovered.keys() | set(self.plugins.pending_keys())

    def items(self) -> list[tuple[str, _V, str]]:
        """Return ``(key, handler, source)`` for every handler, explicit ones first."""
        records = [
            (key, handler, self.explicit.source(key)) for key, handler in self.explicit.items()
        ]
        if self.discovered is not self.explicit:
            records += [
                (key, handler, self.discovered.source(key))
                for key, handler in self.discovered.items()
                if key not in self.explicit
            ]
        return records

    def registered(self) -> list[tuple[str, _V, str]]:
        """Return :meth:`items` after loading every pending plugin, retrying failed ones."""
        self.plugins.resolve_all()
        return self.items()

    def snapshot(self) -> _HandlerRegistryState:
        """Copy the registry state for :meth:`restore`."""
        return {
            'explicit': self.explicit.snapshot(),
            'discovered': self.discovered.snapshot(),
            'plugins': self.plugins.snapshot(),
        }

    def restore(self, state: _HandlerRegistryState) -> None:  # numpydoc ignore=PR01
        """Replace the registry state with a :meth:`snapshot`."""
        self.explicit.restore(state['explicit'])
        self.discovered.restore(state['discovered'])
        self.plugins.restore(state['plugins'])


class _DescriptorRegistryState(TypedDict):
    """Snapshot of a :class:`_DescriptorRegistry`."""

    registrations: list[Any]
    prior: dict[tuple[type, str], Any]
    attached: dict[tuple[type, str], Any]
    plugins: _EntryPointLoaderState


class _DescriptorRegistry(Generic[_R]):
    """Namespaced classes attached to a target class through a caching descriptor.

    Parameters
    ----------
    group : str
        Entry-point group whose entry points name modules that register on import.
    descriptor : type
        Descriptor class, constructed as ``descriptor(name, cls)``.
    record : callable
        Builds the registration record from ``(name, target, cls, source)``.
    noun : str
        What is registered, as it appears in messages (``'accessor'``).
    label : str
        The noun capitalized for the start of a message (``'Accessor'``).

    """

    def __init__(
        self,
        group: str,
        *,
        descriptor: type,
        record: Callable[[str, type, type, str], _R],
        noun: str,
        label: str,
    ) -> None:  # numpydoc ignore=PR01
        self._descriptor = descriptor
        self._record = record
        self._noun = noun
        self._label = label
        self.registrations: list[_R] = []
        self.prior: dict[tuple[type, str], Any] = {}
        self.plugins = _EntryPointLoader(
            group, key=lambda ep, _group: ep.name, load=import_entry_point_module, keep_causes=True
        )

    def validate_name(self, name: object) -> str:  # numpydoc ignore=PR01
        """Strip ``name`` and check it is a public identifier."""
        if not isinstance(name, str):
            msg = f'{self._label} name must be a string, got {type(name).__name__}.'
            raise TypeError(msg)
        normalized = name.strip()
        if not normalized:
            msg = f'{self._label} name must not be empty.'
            raise ValueError(msg)
        if normalized.startswith('_'):
            msg = (
                f'{self._label} name must not start with an underscore, got {name!r}. '
                'Leading underscores are reserved for PyVista internals.'
            )
            raise ValueError(msg)
        if not normalized.isidentifier():
            msg = f'{self._label} name must be a valid Python identifier, got {name!r}.'
            raise ValueError(msg)
        return normalized

    @staticmethod
    def validate_target(target_cls: Any) -> type:  # numpydoc ignore=PR01
        """Check that ``target_cls`` is a class."""
        if not isinstance(target_cls, type):
            msg = f'target_cls must be a class, got {type(target_cls).__name__}.'
            raise TypeError(msg)
        return target_cls

    def validate_class(self, cls: object) -> type:  # numpydoc ignore=PR01
        """Check that ``cls`` is a class."""
        if not isinstance(cls, type):
            msg = f'{self._label} must be a class, got {type(cls).__name__}.'
            raise TypeError(msg)
        return cls

    def owner_on_mro(self, target_cls: type, name: str) -> type | None:  # numpydoc ignore=PR01
        """Return the class whose ``__dict__`` holds the descriptor under ``name``."""
        for klass in target_cls.__mro__:
            if isinstance(klass.__dict__.get(name), self._descriptor):
                return klass
        return None

    def shadow_on_mro(self, target_cls: type, name: str) -> type | None:  # numpydoc ignore=PR01
        """Return the class defining a non-descriptor attribute under ``name``, if any."""
        for klass in target_cls.__mro__:
            if klass is object or name not in klass.__dict__:
                continue
            if isinstance(klass.__dict__[name], self._descriptor):
                return None
            return klass
        return None

    def attach(
        self, name: str, target_cls: type, cls: type, *, override: bool, source: str | None = None
    ) -> None:  # numpydoc ignore=PR01
        """Attach ``cls`` to ``target_cls`` under ``name`` after collision checks."""
        owner = self.owner_on_mro(target_cls, name)
        if owner is not None:
            if owner is target_cls and owner.__dict__[name]._cls is cls:
                return  # the same registration run twice, as under ``pytest --cov``
            if not override:
                warn_external(
                    f'Registering {self._noun} {name!r} on {target_cls.__qualname__} '
                    f'replaces an existing registered {self._noun} on '
                    f'{self._location(owner, target_cls)}.',
                )
        else:
            shadow = self.shadow_on_mro(target_cls, name)
            if shadow is not None and not override:
                msg = (
                    f'Cannot register {self._noun} {name!r} on {target_cls.__qualname__}: '
                    f'shadows built-in attribute on {self._location(shadow, target_cls)}. '
                    'Pass override=True to force.'
                )
                raise ValueError(msg)

        key = (target_cls, name)
        if key not in self.prior:
            self.prior[key] = target_cls.__dict__.get(name, _MISSING)
        setattr(target_cls, name, self._descriptor(name, cls))
        self._forget(target_cls, name)
        self.registrations.append(
            self._record(name, target_cls, cls, handler_source(cls) if source is None else source)
        )

    def detach(self, name: str, target_cls: type) -> None:  # numpydoc ignore=PR01
        """Remove the descriptor attached directly on ``target_cls`` and restore what it hid."""
        if not isinstance(target_cls.__dict__.get(name), self._descriptor):
            msg = (
                f'No registered {self._noun} {name!r} attached directly to '
                f'{target_cls.__qualname__}.'
            )
            raise ValueError(msg)  # noqa: TRY004
        prior = self.prior.pop((target_cls, name), _MISSING)
        if prior is _MISSING:
            delattr(target_cls, name)
        else:
            setattr(target_cls, name, prior)
        self._forget(target_cls, name)

    def resolve_pending(self, name: str) -> bool:  # numpydoc ignore=PR01
        """Import the plugin pending under ``name``; a failed one raises ``AttributeError``."""
        if name.startswith('_'):
            return False  # registered names never do, so dunder lookups skip the scan
        if self.plugins.resolve(name):
            return True
        failure = self.plugins.failure(name)
        if failure is not None:
            raise AttributeError(failure) from self.plugins.take_cause(name)
        return False

    def registered(self) -> tuple[_R, ...]:
        """Return every registration after loading every pending plugin."""
        self.plugins.resolve_all()
        return tuple(self.registrations)

    def snapshot(self) -> _DescriptorRegistryState:
        """Copy the registry state for :meth:`restore`."""
        attached = {
            (r.target, r.name): r.target.__dict__.get(r.name, _MISSING)  # type: ignore[attr-defined]
            for r in self.registrations
        }
        return {
            'registrations': list(self.registrations),
            'prior': dict(self.prior),
            'attached': attached,
            'plugins': self.plugins.snapshot(),
        }

    def restore(self, state: _DescriptorRegistryState) -> None:  # numpydoc ignore=PR01
        """Detach what was added since the :meth:`snapshot` and re-attach what was removed."""
        current = {(r.target, r.name) for r in self.registrations}  # type: ignore[attr-defined]
        snapshot = {(r.target, r.name) for r in state['registrations']}
        for record in self.registrations:
            key = (record.target, record.name)  # type: ignore[attr-defined]
            if key in snapshot:
                continue
            if isinstance(record.target.__dict__.get(record.name), self._descriptor):  # type: ignore[attr-defined]
                delattr(record.target, record.name)  # type: ignore[attr-defined]
            prior = self.prior.pop(key, _MISSING)
            if prior is not _MISSING:
                setattr(record.target, record.name, prior)  # type: ignore[attr-defined]
        for record in state['registrations']:
            key = (record.target, record.name)
            attached = state['attached'].get(key, _MISSING)
            if attached is _MISSING:
                if record.name in record.target.__dict__:
                    delattr(record.target, record.name)
            elif key not in current or record.target.__dict__.get(record.name) is not attached:
                setattr(record.target, record.name, attached)
        self.registrations[:] = state['registrations']
        self.prior.clear()
        self.prior.update(state['prior'])
        self.plugins.restore(state['plugins'])

    def _forget(self, target_cls: type, name: str) -> None:
        """Drop the registration record for ``(target_cls, name)``."""
        self.registrations[:] = [
            r
            for r in self.registrations
            if not (r.target is target_cls and r.name == name)  # type: ignore[attr-defined]
        ]

    @staticmethod
    def _location(owner: type, target_cls: type) -> str:
        """Describe where on the MRO an attribute lives."""
        if owner is target_cls:
            return target_cls.__qualname__
        return f'{owner.__qualname__} (inherited by {target_cls.__qualname__})'


def import_entry_point_module(key: str, eps: list[EntryPoint]) -> None:  # noqa: ARG001  # numpydoc ignore=PR01
    """Import the module the first entry point names, for plugins that register on import."""
    import_module(eps[0].value)


def load_entry_point(eps: list[EntryPoint]) -> Any:  # numpydoc ignore=PR01
    """Return the object the first entry point names."""
    return eps[0].load()
