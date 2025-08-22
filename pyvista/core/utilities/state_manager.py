"""Context manager for controlling global state variables."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import contextlib
import os
from typing import TYPE_CHECKING
from typing import Generic
from typing import Literal
from typing import TypeVar
from typing import cast
from typing import final
from typing import get_args
from typing import overload

from pyvista.core import _vtk_core as _vtk

if TYPE_CHECKING:
    from typing_extensions import Self


T = TypeVar('T')


class _StateManager(contextlib.AbstractContextManager[None], ABC, Generic[T]):
    """Abstract base class for managing a global state variable.

    Subclasses must:

    - Specify a `Literal` as the subclass' type argument. The literal's
      arguments must specify all allowable options for the state variable.
    - Define a getter and setter for the state. Input validation is not
      required - the input is automatically validated when setting the state.

    Examples
    --------
    >>> from pyvista.core.utilities.state_manager import _StateManager
    >>> from typing import Literal

    Define the available options as a ``Literal`` and initialize a global state variable.

    >>> _StateOptions = Literal['on', 'off']
    >>> _GLOBAL_STATE = ['off']  # Init global state. Use list to make it mutable.

    Define the class and its state property.

    >>> class MyState(_StateManager[_StateOptions]):
    ...     @property
    ...     def _state(self) -> _StateOptions:
    ...         return _GLOBAL_STATE[0]
    ...
    ...     @_state.setter
    ...     def _state(self, state: _StateOptions) -> None:
    ...         _GLOBAL_STATE[0] = state

    Finally, create an instance of the state manager.

    >>> my_state = MyState()

    Get the state.

    >>> my_state()
    'off'

    Set the state.

    >>> _ = my_state('on')
    >>> my_state()
    'on'

    Use it as a context manager to set the state temporarily:

    >>> with my_state('off'):
    ...     pass

    """

    @classmethod
    def _get_state_options_from_literal(cls) -> tuple[str | int | bool]:
        state_manager_fullname = f'{_StateManager.__module__}.{_StateManager.__name__}'
        for base in getattr(cls, '__orig_bases__', ()):
            if str(base).startswith(state_manager_fullname):
                # Get StateManager's typing args
                state_manager_args = get_args(base)
                if len(state_manager_args) == 1:
                    # There must only be one arg and it must be a non-empty Literal
                    literal = state_manager_args[0]
                    if str(literal).startswith('typing.Literal'):
                        args = get_args(literal)
                        if len(args) >= 1:
                            return args
        msg = (
            'Type argument for subclasses must be a single non-empty Literal with all state '
            'options provided.'
        )
        raise TypeError(msg)

    def __init__(self) -> None:
        """Initialize context manager."""
        self._valid_states = self._get_state_options_from_literal()
        self._original_state: T | None = None

    @property
    @abstractmethod
    def _state(self) -> T:
        """Get the current global state."""

    @_state.setter
    @abstractmethod
    def _state(self, state: T) -> None:
        """Set the global state."""

    @final
    def _validate_state(self, state: T) -> T:
        from pyvista import _validation  # noqa: PLC0415

        _validation.check_contains(self._valid_states, must_contain=state, name='state')
        return state

    def __enter__(self) -> None:
        """Enter context manager."""
        if self._original_state is None:
            msg = 'State must be set before using it as a context manager.'
            raise ValueError(msg)

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001, ANN204
        """Exit context manager and restore original state."""
        self._state = cast('T', self._original_state)
        self._original_state = None  # Reset

    @overload
    def __call__(self: Self, state: None) -> T: ...
    @overload
    def __call__(self: Self, state: T) -> Self: ...
    def __call__(self: Self, state: T | None = None) -> Self | T:
        """Call the context manager."""
        if state is None:
            return self._state

        self._validate_state(state)

        # Create new instance and store the local state to be restored when exiting
        output = self.__class__()
        output._original_state = self._state
        output._state = state
        return output


_VerbosityOptions = Literal[
    'off',
    'error',
    'warning',
    'info',
    'max',
]


class _VTKVerbosity(_StateManager[_VerbosityOptions]):
    """Context manager to set VTK verbosity level.

    .. versionadded:: 0.45

    Parameters
    ----------
    verbosity : str
        Verbosity of the :vtk:`vtkLogger` to set.

        - ``'off'``: No output.
        - ``'error'``: Only error messages.
        - ``'warning'``: Errors and warnings.
        - ``'info'``: Errors, warnings, and info messages.
        - ``'max'``: All messages, including debug info.

    Examples
    --------
    Get the current vtk verbosity.

    >>> import pyvista as pv
    >>> pv.vtk_verbosity()
    'info'

    Set verbosity to max.

    >>> _ = pv.vtk_verbosity('max')
    >>> pv.vtk_verbosity()
    'max'

    Create a :func:`~pyvista.Sphere`. Note how many VTK debugging messages are now
    generated as the sphere is created.

    >>> mesh = pv.Sphere()

    Use it as a context manager to temporarily turn it off.

    >>> with pv.vtk_verbosity('off'):
    ...     mesh = mesh.cell_quality('volume')

    The state is restored to its previous value outside the context.

    >>> pv.vtk_verbosity()
    'max'

    Note that the verbosity state is global and will persist between function
    calls. If the context manager isn't used, the state needs to be reset explicitly.
    Here, we set it back to its default value.

    >>> _ = pv.vtk_verbosity('info')

    """

    @property
    def _state(self) -> _VerbosityOptions:
        int_to_string: dict[int, _VerbosityOptions] = {
            -9: 'off',
            -2: 'error',
            -1: 'warning',
            0: 'info',
            9: 'max',
        }
        state = _vtk.vtkLogger.GetCurrentVerbosityCutoff()
        try:
            return int_to_string[state]
        except KeyError:
            # Unsupported state, raise error using validation method
            self._validate_state(state)  # type: ignore[arg-type]
            msg = 'This line should not be reachable.'  # pragma: no cover
            raise RuntimeWarning(msg)  # pragma: no cover

    @_state.setter
    def _state(self, state: _VerbosityOptions) -> None:
        verbosity_int = _vtk.vtkLogger.ConvertToVerbosity(state.upper())
        _vtk.vtkLogger.SetStderrVerbosity(verbosity_int)


vtk_verbosity = _VTKVerbosity()


_VtkSnakeCaseOptions = Literal['allow', 'warning', 'error']


class _vtkSnakeCase(_StateManager[_VtkSnakeCaseOptions]):  # noqa: N801
    """Context manager to control access to VTK's pythonic snake_case API.

    VTK 9.4 introduced pythonic snake_case attributes, e.g. `output_port` instead
    of `GetOutputPort`. These can easily be confused for PyVista attributes
    which also use a snake_case convention. This class controls access to vtk's
    new interface.

    .. versionadded:: 0.45

    Parameters
    ----------
    state : 'allow' | 'warning' | 'error'
        Allow or disallow the use of VTK's pythonic snake_case API with
        PyVista-wrapped VTK classes.

        - 'allow': Allow accessing VTK-defined snake_case attributes.
        - 'warning': Print a RuntimeWarning when accessing VTK-defined snake_case
          attributes.
        - 'error': Raise a ``PyVistaAttributeError`` when accessing
          VTK-defined snake_case attributes.

    Examples
    --------
    Get the current access state for VTK's snake_case api.

    >>> import pyvista as pv
    >>> pv.vtk_snake_case()
    'error'

    The following will raise an error because the `information` property is defined
    by :vtk:`vtkDataObject` and is not part of PyVista's API.

    >>> # pv.PolyData().information

    Allow use of VTK's snake_case attributes. No warning or error is raised.

    >>> _ = pv.vtk_snake_case('allow')
    >>> pv.PolyData().information
    <vtkmodules.vtkCommonCore.vtkInformation...

    Note that this state is global and will persist between function calls. Set it
    back to its original state explicitly.

    >>> _ = pv.vtk_snake_case('error')

    Use it as a context manager instead. This way, the state is only temporarily
    modified and is automatically restored.

    >>> with pv.vtk_snake_case('allow'):
    ...     _ = pv.PolyData().information

    >>> pv.vtk_snake_case()
    'error'

    """

    @property
    def _state(self) -> _VtkSnakeCaseOptions:
        import pyvista as pv  # noqa: PLC0415

        return pv._VTK_SNAKE_CASE_STATE

    @_state.setter
    def _state(self, state: _VtkSnakeCaseOptions) -> None:
        import pyvista as pv  # noqa: PLC0415

        pv._VTK_SNAKE_CASE_STATE = state


vtk_snake_case = _vtkSnakeCase()


_AllowNewAttributesOptions = Literal['private', 'any', 'none']


class _AllowNewAttributes(_StateManager[_AllowNewAttributesOptions]):
    """Context manager to control setting new attributes on PyVista classes.

    Python allows arbitrarily setting new attributes on objects at any time,
    but PyVista's classes do not always allow this. By default, setting a
    new attribute is only allowed when the attribute's name has a leading
    underscore "``_``", i.e. it is a private attribute; attempting to
    set a new public attribute raises an ``AttributeError``.

    This context manager may be used to allow or disallow setting `any` new
    attribute, public or private, either globally or within a context.

    .. versionadded:: 0.47

    Parameters
    ----------
    mode : 'private' | 'any' | 'none'
        Control if setting new attributes is allowed.

        - 'private': Allow setting private attributes, but do not allow setting public attributes.
        - 'any': Allow setting any new attribute, either private or public.
        - 'none': Do not allow setting new attributes, regardless if they are private or public.

        ``'private'`` is used by default, ``'any'`` removes all restrictions, and ``'none'``
         is most strict.

        .. note::

            An attribute is considered private if its name has a leading underscore "``_``".

    See Also
    --------
    pyvista.set_new_attribute
        Function for setting any new attribute on a PyVista object.

    Examples
    --------
    Get the default attribute mode.

    >>> import pyvista as pv
    >>> pv.allow_new_attributes()
    'private'

    Setting new `private` attributes on PyVista objects is allowed by default.

    >>> mesh = pv.PolyData()
    >>> mesh._foo = 42  # OK
    >>> mesh.foo = 42  # ERROR # doctest: +SKIP

    Do not allow setting new attributes.

    >>> _ = pv.allow_new_attributes('none')
    >>> mesh._foo = 42  # ERROR # doctest:+SKIP
    >>> mesh.foo = 42  # ERROR # doctest:+SKIP

    Note that this state is global and will persist between function calls. Set it
    back to its original state explicitly.

    >>> _ = pv.allow_new_attributes('private')

    Use it as a context manager instead. This way, the state is only temporarily
    modified and is automatically restored.

    >>> with pv.allow_new_attributes('any'):
    ...     mesh._foo = 42  # OK
    ...     mesh.foo = 42  # OK

    >>> pv.allow_new_attributes()
    'private'

    """

    @property
    def _state(self) -> _AllowNewAttributesOptions:
        from pyvista import _ALLOW_NEW_ATTRIBUTES_MODE  # noqa: PLC0415

        return _ALLOW_NEW_ATTRIBUTES_MODE

    @_state.setter
    def _state(self, state: _AllowNewAttributesOptions) -> None:
        import pyvista as pv  # noqa: PLC0415

        pv._ALLOW_NEW_ATTRIBUTES_MODE = state


allow_new_attributes = _AllowNewAttributes()


def _get_default_allow_new_attributes() -> _AllowNewAttributesOptions:
    # Use stricter 'none' mode with CI
    return 'none' if os.environ.get('CI', 'false').lower() == 'true' else 'private'
