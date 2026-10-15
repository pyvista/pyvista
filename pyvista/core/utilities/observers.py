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
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.errors import VTKExecutionError
from pyvista.core.errors import VTKExecutionWarning
from pyvista.core.utilities.misc import _NoNewAttrMixin

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType

    from tqdm import tqdm
    from typing_extensions import Self

    from pyvista._vtk import vtkObject
    from pyvista.core.utilities.reader import BaseVTKReader


def set_error_output_file(
    filename: str | Path,
) -> tuple[_vtk.vtkFileOutputWindow, _vtk.vtkOutputWindow]:
    """Set a file to write out the VTK errors.

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
    filename = Path(filename).expanduser().resolve()
    fileOutputWindow = _vtk.vtkFileOutputWindow()
    fileOutputWindow.SetFileName(filename)  # type: ignore[arg-type]
    outputWindow = _vtk.vtkOutputWindow()
    outputWindow.SetInstance(fileOutputWindow)
    return fileOutputWindow, outputWindow


class VtkErrorCatcher:
    """Context manager to record the errors and warnings VTK reports.

    Recording leaves VTK's output untouched: messages still reach the current VTK output
    window, and with it the ``pyvista.vtk`` logger set up by
    :func:`~pyvista.send_errors_to_logging`. Catchers can be nested, and each one records
    every message reported while it is active.

    Parameters
    ----------
    raise_errors : bool, default: False
        Raise a ``pyvista.VTKExecutionError`` (a runtime error) on exit when a VTK error
        was recorded.

        .. versionchanged:: 0.47

            A ``pyvista.VTKExecutionError`` is now raised instead of a generic
            ``RuntimeError``.

    send_to_logging : bool, optional
        Has no effect. VTK messages are logged by the ``pyvista.vtk`` logger whether or
        not they are recorded.

        .. deprecated:: 0.50

    emit_warnings : bool, default: False
        Emit a ``pyvista.VTKExecutionWarning`` (a runtime warning) on exit when a VTK
        warning was recorded.

        .. versionadded:: 0.47

    Examples
    --------
    Record VTK errors and warnings.

    >>> import pyvista as pv
    >>> with pv.VtkErrorCatcher() as error_catcher:
    ...     sphere = pv.Sphere()
    >>> error_catcher.events
    []

    Raise VTK errors as Python errors and emit VTK warnings as Python warnings.

    >>> with pv.VtkErrorCatcher(
    ...     raise_errors=True, emit_warnings=True
    ... ) as error_catcher:
    ...     sphere = pv.Sphere()

    """

    def __init__(
        self,
        *,
        raise_errors: bool = False,
        send_to_logging: bool | None = None,
        emit_warnings: bool = False,
    ) -> None:
        """Initialize context manager."""
        if send_to_logging is not None:
            # deprecated 0.50.0, convert to error in 0.53.0, remove 0.54.0
            msg = (
                '`send_to_logging` is deprecated and has no effect. VTK messages are '
                "logged by the 'pyvista.vtk' logger."
            )
            warn_external(msg, PyVistaDeprecationWarning)
        self.raise_errors = raise_errors
        self.emit_warnings = emit_warnings
        self._events: list[tuple[str, VtkEvent]] = []

    def __enter__(self: Self) -> Self:
        """Start recording VTK errors and warnings."""
        self._events = []
        _vtk_event_dispatcher.add(self)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *args: object) -> None:
        """Stop recording, then warn and raise unless an exception is propagating."""
        _vtk_event_dispatcher.remove(self)
        if exc_type is None:
            if self.emit_warnings and self.warning_events:
                warn_external(_join(self.warning_events), VTKExecutionWarning)
            if self.raise_errors and self.error_events:
                raise VTKExecutionError(_join(self.error_events))

    def _record(self, event_type: str, event: VtkEvent) -> None:
        """Record one VTK event."""
        self._events.append((event_type, event))

    @property
    def events(self) -> list[VtkEvent]:  # numpydoc ignore=RT01
        """List of all VTK warning and error events recorded, in the order reported.

        .. versionadded:: 0.47

        """
        return [event for _, event in self._events]

    @property
    def error_events(self) -> list[VtkEvent]:  # numpydoc ignore=RT01
        """List of VTK error events recorded.

        .. versionadded:: 0.47

        """
        return [event for event_type, event in self._events if event_type == 'ErrorEvent']

    @property
    def warning_events(self) -> list[VtkEvent]:  # numpydoc ignore=RT01
        """List of VTK warning events recorded.

        .. versionadded:: 0.47

        """
        return [event for event_type, event in self._events if event_type == 'WarningEvent']


def _join(events: list[VtkEvent]) -> str:
    """Format VTK events as one message."""
    return '\n'.join(map(str, events))


class _VtkEventDispatcher:
    """Pass the errors and warnings VTK reports to the active catchers."""

    CallDataType = 'string0'

    def __init__(self) -> None:
        """Initialize with no active catchers."""
        self._catchers: list[VtkErrorCatcher] = []

    def __call__(self, _obj: vtkObject, event_type: str, message: str = '') -> None:
        """Pass one VTK event to the active catchers."""
        if self._catchers:
            event = Observer.parse_message(message)
            for catcher in self._catchers:
                catcher._record(event_type, event)

    def add(self, catcher: VtkErrorCatcher) -> None:
        """Activate a catcher, observing the current output window once per window.

        Parameters
        ----------
        catcher : VtkErrorCatcher
            Catcher to pass events to.

        """
        window = _vtk.vtkOutputWindow.GetInstance()
        if not getattr(window, '_pyvista_event_dispatcher', False):
            window.AddObserver('ErrorEvent', self)  # type: ignore[arg-type]
            window.AddObserver('WarningEvent', self)  # type: ignore[arg-type]
            window._pyvista_event_dispatcher = True  # type: ignore[attr-defined]
        self._catchers.append(catcher)

    def remove(self, catcher: VtkErrorCatcher) -> None:
        """Deactivate a catcher.

        Parameters
        ----------
        catcher : VtkErrorCatcher
            Catcher to stop passing events to.

        """
        self._catchers.remove(catcher)


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
        """Parse different event types and passes them to logging.

        Parameters
        ----------
        kind : str
            Event kind; ``'ERROR'`` logs as an error, anything else as a warning.

        alert : str
            Message text to log.

        """
        if kind == 'ERROR':
            logging.error(alert)  # noqa: LOG015
        else:
            logging.warning(alert)  # noqa: LOG015

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


def send_errors_to_logging() -> None:  # numpydoc ignore=RT01
    """Send all VTK error/warning messages to Python's logging module."""
    error_output = _vtk.vtkStringOutputWindow()
    error_win = _vtk.vtkOutputWindow()
    error_win.SetInstance(error_output)
    obs = Observer()
    return obs.observe(error_output)


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
        logging.debug('SIGINT received. Delaying KeyboardInterrupt until VTK algorithm finishes.')  # noqa: LOG015

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
