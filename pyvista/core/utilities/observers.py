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
from typing import NamedTuple

from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _vtk_core as _vtk
from pyvista.core.utilities.misc import _NoNewAttrMixin


def set_error_output_file(filename):
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
    fileOutputWindow.SetFileName(filename)
    outputWindow = _vtk.vtkOutputWindow()
    outputWindow.SetInstance(fileOutputWindow)
    return fileOutputWindow, outputWindow


class VtkErrorCatcher:
    """Context manager to temporarily catch VTK errors.

    Parameters
    ----------
    raise_errors : bool, default: False
        Raise a ``RuntimeError`` when a VTK error is encountered.

    send_to_logging : bool, default: True
        Determine whether VTK errors raised within the context should
        also be sent to logging.

    Examples
    --------
    Catch VTK errors using the context manager.

    >>> import pyvista as pv
    >>> with pv.VtkErrorCatcher() as error_catcher:
    ...     sphere = pv.Sphere()

    """

    @_deprecate_positional_args
    def __init__(self, raise_errors: bool = False, send_to_logging: bool = True) -> None:  # noqa: FBT001, FBT002
        """Initialize context manager."""
        self.raise_errors = raise_errors
        self.send_to_logging = send_to_logging

    def __enter__(self) -> None:
        """Observe VTK string output window for errors."""
        error_output = _vtk.vtkStringOutputWindow()
        error_win = _vtk.vtkOutputWindow()
        self._error_output_orig = error_win.GetInstance()
        error_win.SetInstance(error_output)
        obs = Observer(log=self.send_to_logging, store_history=True)
        obs.observe(error_output)
        self._observer = obs

    def __exit__(self, *args):
        """Stop observing VTK string output window."""
        error_win = _vtk.vtkOutputWindow()
        error_win.SetInstance(self._error_output_orig)
        self.events = self._observer.event_history
        if self.raise_errors and self.events:
            errors = [RuntimeError(f'{e.kind}: {e.alert}', e.path, e.address) for e in self.events]
            raise RuntimeError(errors)


class VtkEvent(NamedTuple):
    """Named tuple to store VTK event information."""

    kind: str
    path: str
    address: str
    alert: str


class Observer(_NoNewAttrMixin):
    """A standard class for observing VTK objects."""

    @_deprecate_positional_args(allowed=['event_type'])
    def __init__(
        self,
        event_type='ErrorEvent',
        log: bool = True,  # noqa: FBT001, FBT002
        store_history: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize observer."""
        self.__event_occurred = False
        self.__message = None
        self.__message_etc = None
        self.CallDataType = 'string0'
        self.__observing = False
        self.event_type = event_type
        self.__log = log

        self.store_history = store_history
        self.event_history: list[VtkEvent] = []

    @staticmethod
    def parse_message(message):  # numpydoc ignore=RT01
        """Parse the given message."""
        # Message format
        regex = re.compile(r'([A-Z]+):\sIn\s(.+),\sline\s.+\n\w+\s\((.+)\):\s(.+)')
        try:
            kind, path, address, alert = regex.findall(message)[0]
        except Exception:  # noqa: BLE001
            return '', '', '', message
        else:
            return kind, path, address, alert

    def log_message(self, kind, alert) -> None:
        """Parse different event types and passes them to logging."""
        if kind == 'ERROR':
            logging.error(alert)  # noqa: LOG015
        else:
            logging.warning(alert)  # noqa: LOG015

    def __call__(self, _obj, _event, message) -> None:
        """Declare standard call function for the observer.

        On an event occurrence, this function executes.

        """
        try:
            self.__event_occurred = True
            self.__message_etc = message
            kind, path, address, alert = self.parse_message(message)
            self.__message = alert
            if self.store_history:
                self.event_history.append(VtkEvent(kind, path, address, alert))
            if self.__log:
                self.log_message(kind, alert)
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
            except Exception:  # noqa: BLE001
                pass

    def has_event_occurred(self):  # numpydoc ignore=RT01
        """Ask self if an error has occurred since last queried.

        This resets the observer's status.

        """
        occ = self.__event_occurred
        self.__event_occurred = False
        return occ

    @_deprecate_positional_args
    def get_message(self, etc: bool = False):  # noqa: FBT001, FBT002
        """Get the last set error message.

        Returns
        -------
        str
            The last set error message.

        """
        if etc:
            return self.__message_etc
        return self.__message

    def observe(self, algorithm):
        """Make this an observer of an algorithm."""
        if self.__observing:
            msg = 'This error observer is already observing an algorithm.'
            raise RuntimeError(msg)
        if hasattr(algorithm, 'GetExecutive') and algorithm.GetExecutive() is not None:
            algorithm.GetExecutive().AddObserver(self.event_type, self)
        algorithm.AddObserver(self.event_type, self)
        self.__observing = True


def send_errors_to_logging():  # numpydoc ignore=RT01
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
    algorithm
        VTK algorithm or filter.

    message : str, default: ""
        Message to display in the progress bar.

    """

    def __init__(self, algorithm, message=''):
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
        self._old_progress = 0
        self._old_handler = None
        self._progress_bar = None

    def handler(self, sig, frame) -> None:
        """Pass signal to custom interrupt handler."""
        self._interrupt_signal_received = (sig, frame)  # type: ignore[assignment]
        logging.debug('SIGINT received. Delaying KeyboardInterrupt until VTK algorithm finishes.')  # noqa: LOG015

    def __call__(self, obj, *args) -> None:  # noqa: ARG002
        """Call progress update callback.

        On an event occurrence, this function executes.
        """
        if self._interrupt_signal_received:
            obj.AbortExecuteOn()
        else:
            progress = obj.GetProgress()
            step = progress - self._old_progress
            self._progress_bar.update(step)  # type: ignore[union-attr]
            self._old_progress = progress

    def __enter__(self):
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

    def __exit__(self, *args) -> None:
        """Exit event for ``with`` context."""
        self._progress_bar.total = 1  # type: ignore[union-attr]
        self._progress_bar.refresh()  # type: ignore[union-attr]
        self._progress_bar.close()  # type: ignore[union-attr]
        self.algorithm.RemoveObservers(self.event_type)
        if threading.current_thread().__class__.__name__ == '_MainThread':
            signal.signal(signal.SIGINT, self._old_handler)
