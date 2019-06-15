import logging
import os
import re

import vtk


def set_error_output_file(filename):
    """Sets a file to write out the VTK errors"""
    filename = os.path.abspath(os.path.expanduser(filename))
    fileOutputWindow = vtk.vtkFileOutputWindow()
    fileOutputWindow.SetFileName(filename)
    outputWindow = vtk.vtkOutputWindow()
    outputWindow.SetInstance(fileOutputWindow)
    return fileOutputWindow, outputWindow


class Observer:
    """A standerd class for observing VTK objects.
    """
    def __init__(self, event_type='ErrorEvent', log=True):
        self.__event_occurred = False
        self.__message = None
        self.__message_etc = None
        self.CallDataType = 'string0'
        self.__observing = False
        self.event_type = event_type
        self.__log = log

    @staticmethod
    def parse_message(message):
        # Message format
        regex = re.compile(r'([A-Z]+):\sIn\s(.+),\sline\s.+\n\w+\s(.+):\s(.+)')
        try:
            kind, path, address, alert = regex.findall(message)[0]
            return kind, path, address, alert
        except:
            return '', '', '', message

    def log_message(self, kind, alert):
        """Parses different event types and passes them to logging"""
        if kind == 'ERROR':
            logging.error(alert)
        else:
            logging.warning(alert)
        return

    def __call__(self, obj, event, message):
        """On an event occurence, this function executes"""
        self.__event_occurred = True
        self.__message_etc = message
        kind, path, address, alert = self.parse_message(message)
        self.__message = alert
        if self.__log:
            self.log_message(kind, alert)

    def has_event_occurred(self):
        """Ask self if an error has occured since last querried.
        This resets the observer's status.
        """
        occ = self.__event_occurred
        self.__event_occurred = False
        return occ

    def get_message(self, etc=False):
        """Get the last set error message

        Return:
            str: the last set error message
        """
        if etc:
            return self.__message_etc
        return self.__message

    def observe(self, algorithm):
        """Make this an observer of an algorithm
        """
        if self.__observing:
            raise RuntimeError('This error observer is already observing an algorithm.')
        if hasattr(algorithm, 'GetExecutive') and algorithm.GetExecutive() is not None:
            algorithm.GetExecutive().AddObserver(self.event_type, self)
        algorithm.AddObserver(self.event_type, self)
        self.__observing = True
        return


def send_errors_to_logging():
    """Send all VTK error/warning messages to Python's logging module"""
    error_output = vtk.vtkStringOutputWindow()
    error_win = vtk.vtkOutputWindow()
    error_win.SetInstance(error_output)
    obs = Observer()
    return obs.observe(error_output)
