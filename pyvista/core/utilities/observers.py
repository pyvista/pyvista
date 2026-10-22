"""Core error utilities."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import re
import signal
import sys
import threading
import traceback
from typing import TYPE_CHECKING
from typing import NamedTuple

from pyvista import _vtk
from pyvista._warn_external import warn_external
from pyvista.core.config import _validate_vtk_message_level
from pyvista.core.config import _VtkMessageLevel
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.errors import VTKExecutionError
from pyvista.core.errors import VTKExecutionWarning
from pyvista.core.utilities.misc import _NoNewAttrMixin

log = logging.getLogger(__name__)
_vtk_log = logging.getLogger('pyvista.vtk')

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType
    from typing import Self

    from tqdm import tqdm

    from pyvista._vtk import vtkObject
    from pyvista.core.utilities.reader import BaseVTKReader


def set_error_output_file(
    filename: str | Path,
) -> tuple[_vtk.vtkFileOutputWindow, _vtk.vtkOutputWindow]:
    """Set a file to write out the VTK errors.

    .. deprecated:: 0.50

        Set :attr:`pyvista.core.config.Config.vtk_errors` to ``'log'`` and add a
        :class:`logging.FileHandler` to the ``'pyvista.vtk'`` logger instead.

    Parameters
    ----------
    filename : str, Path
        Path to the file to write VTK errors to.

    Returns
    -------
    :vtk:`vtkFileOutputWindow`
        VTK file output window.
    :vtk:`vtkOutputWindow`
        VTK output window.

    """
    # deprecated 0.50.0, convert to error in 0.53.0, remove 0.54.0
    msg = (
        '`set_error_output_file` is deprecated. Set `pv.global_config.vtk_errors` to '
        "'log' and add a `logging.FileHandler` to the 'pyvista.vtk' logger instead."
    )
    warn_external(msg, PyVistaDeprecationWarning)
    filename = Path(filename).expanduser().resolve()
    fileOutputWindow = _vtk.vtkFileOutputWindow()
    fileOutputWindow.SetFileName(filename)  # type: ignore[arg-type]
    outputWindow = _vtk.vtkOutputWindow()
    outputWindow.SetInstance(fileOutputWindow)
    return fileOutputWindow, outputWindow


class vtk_messages:  # noqa: N801
    """Record the errors and warnings VTK reports, and optionally change how they are handled.

    VTK reports errors and warnings as messages. By default each one becomes a Python
    warning. :attr:`pyvista.core.config.Config.vtk_errors` and
    :attr:`pyvista.core.config.Config.vtk_warnings` set this globally, and this context
    manager overrides them for a block of code. Each level is one of:

    - ``'raise'``: raise :class:`~pyvista.VTKExecutionError` when the block ends.
    - ``'warn'``: emit a :class:`~pyvista.VTKExecutionWarning`.
    - ``'log'``: log with the ``'pyvista.vtk'`` logger.
    - ``'ignore'``: do nothing.

    A PyVista operation whose VTK algorithm fails raises
    :class:`~pyvista.VTKExecutionError` at every level, with the messages VTK reported.

    Contexts can be nested, and each one records every message reported while it is
    active, whatever the level. The levels are process-wide, so a context in one thread
    also applies to messages reported from other threads while it is active.

    .. versionadded:: 0.50

    Parameters
    ----------
    errors : 'raise' | 'warn' | 'log' | 'ignore', optional
        Level for VTK errors while the context is active. The default keeps
        :attr:`pyvista.core.config.Config.vtk_errors`.

    warnings : 'raise' | 'warn' | 'log' | 'ignore', optional
        Level for VTK warnings while the context is active. The default keeps
        :attr:`pyvista.core.config.Config.vtk_warnings`.

    Examples
    --------
    Record VTK messages without changing how they are handled.

    >>> import pyvista as pv
    >>> with pv.vtk_messages() as messages:
    ...     sphere = pv.Sphere()
    >>> messages.events
    []

    Raise VTK errors as Python errors and ignore VTK warnings.

    >>> with pv.vtk_messages(errors='raise', warnings='ignore'):
    ...     sphere = pv.Sphere()

    """

    def __init__(
        self,
        *,
        errors: _VtkMessageLevel | None = None,
        warnings: _VtkMessageLevel | None = None,
    ) -> None:
        """Initialize context manager."""
        self._errors = None if errors is None else _validate_vtk_message_level(errors, 'errors')
        self._warnings = (
            None if warnings is None else _validate_vtk_message_level(warnings, 'warnings')
        )
        self._events: list[tuple[str, VtkEvent]] = []
        self._pending: list[tuple[str, VtkEvent, _VtkMessageLevel]] = []
        self._saved: dict[str, _VtkMessageLevel] = {}

    def __enter__(self: Self) -> Self:
        """Start recording VTK errors and warnings."""
        import pyvista as pv  # noqa: PLC0415

        self._events = []
        self._pending = []
        config = pv.global_config
        self._saved = {}
        if self._errors is not None:
            self._saved['vtk_errors'] = config.vtk_errors
            config.vtk_errors = self._errors
        if self._warnings is not None:
            self._saved['vtk_warnings'] = config.vtk_warnings
            config.vtk_warnings = self._warnings
        _vtk_event_dispatcher.add(self)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *args: object) -> None:
        """Stop recording, then handle the messages reported inside the block.

        Messages at the ``'raise'`` level are dropped when the block exits with an
        exception, so the exception propagates. A warning filter that turns a VTK warning
        into an exception keeps a pending ``VTKExecutionError`` as its cause.
        """
        import pyvista as pv  # noqa: PLC0415

        _vtk_event_dispatcher.remove(self)
        for name, level in self._saved.items():
            setattr(pv.global_config, name, level)
        self._saved = {}
        raised: list[VtkEvent] = []
        warning_exc: VTKExecutionWarning | None = None
        for event_type, event, level in self._pending:
            if level == 'raise':
                if exc_type is None:
                    raised.append(event)
                continue
            try:
                _handle(event_type, event, level)
            except VTKExecutionWarning as exc:
                if exc_type is not None or warning_exc is not None:
                    _handle(event_type, event, 'log')
                else:
                    warning_exc = exc
        if raised:
            raise VTKExecutionError(_join(raised)) from warning_exc
        if warning_exc is not None:
            raise warning_exc

    def _record(self, event_type: str, event: VtkEvent) -> None:
        """Record one VTK event."""
        self._events.append((event_type, event))

    def _defer(self, event_type: str, event: VtkEvent, level: _VtkMessageLevel) -> None:
        """Handle one VTK event when the block ends."""
        self._pending.append((event_type, event, level))

    def _drop_pending_errors(self) -> None:
        """Leave the recorded errors to the caller, which raises them itself."""
        self._pending = [item for item in self._pending if item[0] != 'ErrorEvent']

    @property
    def events(self) -> list[VtkEvent]:  # numpydoc ignore=RT01
        """List of all VTK warning and error events recorded, in the order reported."""
        return [event for _, event in self._events]

    @property
    def error_events(self) -> list[VtkEvent]:  # numpydoc ignore=RT01
        """List of VTK error events recorded."""
        return [event for event_type, event in self._events if event_type == 'ErrorEvent']

    @property
    def warning_events(self) -> list[VtkEvent]:  # numpydoc ignore=RT01
        """List of VTK warning events recorded."""
        return [event for event_type, event in self._events if event_type == 'WarningEvent']


class VtkErrorCatcher(vtk_messages):
    """Context manager to temporarily catch VTK errors.

    .. deprecated:: 0.50

        Use :class:`pyvista.vtk_messages` instead. Messages this catcher does not
        raise or emit are logged with the ``'pyvista.vtk'`` logger, or ignored when
        ``send_to_logging`` is ``False``.

    Parameters
    ----------
    raise_errors : bool, default: False
        Raise a ``pyvista.VTKExecutionError`` when a VTK error is recorded.

    send_to_logging : bool, default: True
        Log the messages that are not raised or emitted.

    emit_warnings : bool, default: False
        Emit a ``pyvista.VTKExecutionWarning`` when a VTK warning is recorded.

    """

    def __init__(
        self,
        *,
        raise_errors: bool = False,
        send_to_logging: bool = True,
        emit_warnings: bool = False,
    ) -> None:
        """Initialize context manager."""
        # deprecated 0.50.0, convert to error in 0.53.0, remove 0.54.0
        msg = '`VtkErrorCatcher` is deprecated. Use `pv.vtk_messages` instead.'
        warn_external(msg, PyVistaDeprecationWarning)
        self.raise_errors = raise_errors
        self.send_to_logging = send_to_logging
        self.emit_warnings = emit_warnings
        fallback: _VtkMessageLevel = 'log' if send_to_logging else 'ignore'
        super().__init__(
            errors='raise' if raise_errors else fallback,
            warnings='warn' if emit_warnings else fallback,
        )


def _join(events: list[VtkEvent]) -> str:
    """Format VTK events as one message."""
    return '\n'.join(map(str, events))


def _handle(event_type: str, event: VtkEvent, level: _VtkMessageLevel) -> None:
    """Warn or log one VTK event; ``'raise'`` is handled by the context that owns it."""
    if level == 'warn':
        warn_external(str(event), VTKExecutionWarning)
    elif level == 'log':
        is_error = event_type == 'ErrorEvent'
        _vtk_log.log(logging.ERROR if is_error else logging.WARNING, str(event))


class _VtkEventDispatcher:
    """Handle the errors and warnings VTK reports and pass them to the active contexts."""

    CallDataType = 'string0'

    def __init__(self) -> None:
        """Initialize with no active contexts."""
        self._contexts: list[vtk_messages] = []

    def __call__(self, _obj: vtkObject, event_type: str, message: str = '') -> None:
        """Handle one VTK event."""
        import pyvista as pv  # noqa: PLC0415

        event = Observer.parse_message(message)
        for context in self._contexts:
            context._record(event_type, event)
        is_error = event_type == 'ErrorEvent'
        level = pv.global_config.vtk_errors if is_error else pv.global_config.vtk_warnings
        if self._contexts:
            self._contexts[-1]._defer(event_type, event, level)
            return
        try:
            _handle(event_type, event, 'warn' if level == 'raise' else level)
        except VTKExecutionWarning:
            # Raised by a warning filter; VTK would swallow it, so log it instead
            _handle(event_type, event, 'log')

    def observe(self) -> None:
        """Observe the current output window, once per window."""
        window = _vtk.vtkOutputWindow.GetInstance()
        if not getattr(window, '_pyvista_event_dispatcher', False):
            window.AddObserver('ErrorEvent', self)  # type: ignore[arg-type]
            window.AddObserver('WarningEvent', self)  # type: ignore[arg-type]
            window._pyvista_event_dispatcher = True  # type: ignore[attr-defined]

    def add(self, context: vtk_messages) -> None:
        """Activate a context.

        Parameters
        ----------
        context : vtk_messages
            Context to pass events to.

        """
        self.observe()
        self._contexts.append(context)

    def remove(self, context: vtk_messages) -> None:
        """Deactivate a context.

        Parameters
        ----------
        context : vtk_messages
            Context to stop passing events to.

        """
        self._contexts.remove(context)


_vtk_event_dispatcher = _VtkEventDispatcher()


class VtkEvent(NamedTuple):
    """Named tuple to store VTK event information."""

    kind: str
    path: str
    address: str
    alert: str
    line: str
    name: str

    def __str__(self) -> str:
        if all(self):
            return (
                f'{self.kind}: In {self.path}, line {self.line}\n'
                f'{self.name} ({self.address}): {self.alert}'
            ).strip()
        return self.alert


class Observer(_NoNewAttrMixin):
    """A standard class for observing VTK objects.

    Parameters
    ----------
    event_type : str, default: "ErrorEvent"
        Event type to observe.

    log : bool, default: True
        Log the caught messages.

    store_history : bool, default: False
        Store each caught event.

    """

    def __init__(
        self,
        event_type: str = 'ErrorEvent',
        *,
        log: bool = True,
        store_history: bool = False,
    ) -> None:
        """Initialize observer."""
        self.__event_occurred: bool = False
        self.__message: str | None = None
        self.__message_etc: str | None = None
        self.CallDataType = 'string0'
        self.__observing: bool = False
        self.event_type = event_type
        self.__log = log

        self.store_history = store_history
        self.event_history: list[VtkEvent] = []
        self._event_history_etc: list[str] = []

    @staticmethod
    def parse_message(message: str) -> VtkEvent:
        """Parse the given message.

        Parameters
        ----------
        message : str
            Message from the VTK event.

        Returns
        -------
        VtkEvent
            Parsed event fields.

        """
        regex = re.compile(
            r'(?P<kind>[a-zA-Z]+):\sIn\s(?P<path>.+?),\sline\s(?P<line>\d+)\r?\n'
            r'(?P<name>\w+) \((?P<address>0x[0-9a-fA-F]+)\):\s(?P<alert>.+)',
            re.DOTALL,
        )

        match = regex.match(message)
        if match:
            d = match.groupdict()
            kind = d.get('kind', '')
            path = d.get('path', '')
            line = d.get('line', '')
            name = d.get('name', '')
            address = d.get('address', '')
            alert = d.get('alert', '').strip()
            return VtkEvent(
                kind=kind, path=path, line=line, name=name, address=address, alert=alert
            )
        return VtkEvent(kind='', path='', line='', name='', address='', alert=message.strip())

    def log_message(self, kind: str, alert: str) -> None:
        """Log a VTK event with the ``pyvista.vtk`` logger.

        .. versionchanged:: 0.50

            Logs with the ``pyvista.vtk`` logger instead of the root logger.

        Parameters
        ----------
        kind : str
            Event kind; ``'ERROR'`` logs as an error, anything else as a warning.

        alert : str
            Message text to log.

        """
        if kind == 'ERROR':
            _vtk_log.error(alert)
        else:
            _vtk_log.warning(alert)

    def __call__(self, _obj: vtkObject, _event: str, message: str = '') -> None:
        """Declare standard call function for the observer.

        On an event occurrence, this function executes.

        """
        try:
            self.__event_occurred = True
            self.__message_etc = message
            event = self.parse_message(message)
            self.__message = event.alert
            if self.store_history:
                self.event_history.append(event)
                self._event_history_etc.append(message)
            if self.__log:
                self.log_message(event.kind, event.alert)
        except Exception:  # noqa: BLE001  # pragma: no cover
            try:
                if len(message) > 120:
                    message = f'{message[:100]!r} ... ({len(message)} characters)'
                else:
                    message = repr(message)
                print(
                    f'PyVista error in handling VTK error message:\n{message}',
                    file=sys.__stdout__,
                )
                traceback.print_tb(sys.last_traceback, file=sys.__stderr__)
            except Exception as e:  # noqa: BLE001
                log.debug('Failed to print VTK error message: %s', e)

    def has_event_occurred(self) -> bool:  # numpydoc ignore=RT01
        """Ask self if an error has occurred since last queried.

        This resets the observer's status.

        """
        occ = self.__event_occurred
        self.__event_occurred = False
        return occ

    def get_message(self, *, etc: bool = False) -> str | None:
        """Get the last set error message.

        Parameters
        ----------
        etc : bool, default: False
            Return the full event information instead of only the message.

        Returns
        -------
        str or None
            The last set error message. It is ``None`` until an event has been observed.

        """
        if etc:
            return self.__message_etc
        return self.__message

    def observe(self, algorithm: vtkObject | BaseVTKReader) -> None:
        """Make this an observer of an algorithm.

        Parameters
        ----------
        algorithm : :vtk:`vtkObject` | BaseVTKReader
            Algorithm to observe.

        """
        if self.__observing:
            msg = 'This error observer is already observing an algorithm.'
            raise RuntimeError(msg)
        if hasattr(algorithm, 'GetExecutive') and algorithm.GetExecutive() is not None:
            algorithm.GetExecutive().AddObserver(self.event_type, self)
        algorithm.AddObserver(self.event_type, self)  # type: ignore[arg-type]
        self.__observing = True


def _install_vtk_output_window() -> None:
    """Route VTK messages through PyVista and turn off VTK's own stderr logging."""
    _vtk.vtkOutputWindow.SetInstance(_vtk.vtkStringOutputWindow())
    _vtk_event_dispatcher.observe()
    _vtk.vtkLogger.SetStderrVerbosity(_vtk.vtkLogger.VERBOSITY_OFF)


def send_errors_to_logging() -> None:
    """Send VTK errors and warnings to the ``'pyvista.vtk'`` logger.

    .. deprecated:: 0.50

        Set :attr:`pyvista.core.config.Config.vtk_errors` and
        :attr:`pyvista.core.config.Config.vtk_warnings` to ``'log'`` instead.

    """
    import pyvista as pv  # noqa: PLC0415

    # deprecated 0.50.0, convert to error in 0.53.0, remove 0.54.0
    msg = (
        '`send_errors_to_logging` is deprecated. Set `pv.global_config.vtk_errors` and '
        "`pv.global_config.vtk_warnings` to 'log' instead."
    )
    warn_external(msg, PyVistaDeprecationWarning)
    pv.global_config.vtk_errors = 'log'
    pv.global_config.vtk_warnings = 'log'


class ProgressMonitor(_NoNewAttrMixin):
    """A standard class for monitoring the progress of a VTK algorithm.

    This must be use in a ``with`` context and it will block keyboard
    interrupts from happening until the exit event as interrupts will crash
    the kernel if the VTK algorithm is still executing.

    Parameters
    ----------
    algorithm : :vtk:`vtkAlgorithm` | BaseVTKReader
        VTK algorithm or filter.

    message : str, default: ""
        Message to display in the progress bar.

    """

    def __init__(self, algorithm: _vtk.vtkAlgorithm | BaseVTKReader, message: str = '') -> None:
        """Initialize observer."""
        if not importlib.util.find_spec('tqdm'):
            msg = 'Please install `tqdm` to monitor algorithms.'
            raise ImportError(msg)
        self.event_type = _vtk.vtkCommand.ProgressEvent
        self.progress = 0.0
        self._last_progress = self.progress
        self.algorithm = algorithm
        self.message = message
        self._interrupt_signal_received = False
        self._old_progress: float = 0
        self._old_handler: Callable[[int, FrameType | None], object] | int | None = None
        self._progress_bar: tqdm | None = None

    def handler(self, sig: int, frame: FrameType | None) -> None:
        """Pass signal to custom interrupt handler.

        Parameters
        ----------
        sig : int
            Signal number.

        frame : types.FrameType
            Current stack frame.

        """
        self._interrupt_signal_received = (sig, frame)  # type: ignore[assignment]
        log.debug('SIGINT received. Delaying KeyboardInterrupt until VTK algorithm finishes.')

    def __call__(
        self,
        obj: _vtk.vtkAlgorithm | BaseVTKReader,
        *args: int | str,  # noqa: ARG002
    ) -> None:
        """Call progress update callback.

        On an event occurrence, this function executes.
        """
        if self._interrupt_signal_received:
            if (abort := getattr(obj, 'AbortExecuteOn', None)) is not None:
                abort()
        else:
            progress = obj.GetProgress()
            step = progress - self._old_progress
            self._progress_bar.update(step)  # type: ignore[union-attr]
            self._old_progress = progress

    def __enter__(self) -> tqdm:
        """Enter event for ``with`` context."""
        from tqdm import tqdm  # noqa: PLC0415

        # check if in main thread
        if threading.current_thread().__class__.__name__ == '_MainThread':
            self._old_handler = signal.signal(signal.SIGINT, self.handler)
        self._progress_bar = tqdm(
            total=1,
            leave=True,
            bar_format='{l_bar}{bar}[{elapsed}<{remaining}]',
        )
        self._progress_bar.set_description(self.message)
        self.algorithm.AddObserver(self.event_type, self)
        return self._progress_bar

    def __exit__(self, *args: object) -> None:
        """Exit event for ``with`` context."""
        self._progress_bar.total = 1  # type: ignore[union-attr]
        self._progress_bar.refresh()  # type: ignore[union-attr]
        self._progress_bar.close()  # type: ignore[union-attr]
        self.algorithm.RemoveObservers(self.event_type)
        if threading.current_thread().__class__.__name__ == '_MainThread':
            signal.signal(signal.SIGINT, self._old_handler)
