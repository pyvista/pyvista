"""Typing cases for :func:`pyvista.plotting.colors.get_cycler`."""

from __future__ import annotations

from typing import Any

from cycler import Cycler
from type_assert import assert_types

from pyvista.plotting.colors import get_cycler

# A name, a sequence of colors or a cycler each give a cycler back
assert_types(get_cycler('default'), Cycler[str, Any])
assert_types(get_cycler(['red', 'green']), Cycler[str, Any])
assert_types(get_cycler(get_cycler('all')), Cycler[str, Any])

# Only ``None`` has no cycler
assert_types(get_cycler(None), None)
