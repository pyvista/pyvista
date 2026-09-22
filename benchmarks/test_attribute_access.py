"""Benchmarks for array lookup and for the active-attribute properties."""

from __future__ import annotations

import operator

from pyvista.core.utilities.arrays import get_array
from pyvista.core.utilities.arrays import get_array_association


def test_array_names(sphere, benchmark):
    """Read the array names of a dataset."""
    assert benchmark(operator.attrgetter('array_names'), sphere)


def test_active_scalars_name(sphere, benchmark):
    """Read the active scalars name."""
    assert benchmark(operator.attrgetter('active_scalars_name'), sphere) == 'data'


def test_active_scalars_info(sphere, benchmark):
    """Read the active scalars association and name."""
    assert benchmark(operator.attrgetter('active_scalars_info'), sphere)


def test_active_scalars(sphere, benchmark):
    """Read the active scalars array."""
    assert benchmark(operator.attrgetter('active_scalars'), sphere).size


def test_active_scalars_info_unchosen(unchosen_sphere, benchmark):
    """Read the active scalars info of a mesh whose active array was never chosen."""
    assert benchmark(operator.attrgetter('active_scalars_info'), unchosen_sphere)


def test_active_scalars_unchosen(unchosen_sphere, benchmark):
    """Read the active scalars of a mesh whose active array was never chosen."""
    assert benchmark(operator.attrgetter('active_scalars'), unchosen_sphere).size


def test_active_vectors_info(sphere, benchmark):
    """Read the active vectors association and name."""
    assert benchmark(operator.attrgetter('active_vectors_info'), sphere) is not None


def test_active_normals(sphere, benchmark):
    """Read the active normals array."""
    assert benchmark(operator.attrgetter('active_normals'), sphere).size


def test_point_data_property(sphere, benchmark):
    """Build the point-data attributes wrapper."""
    assert benchmark(operator.attrgetter('point_data'), sphere) is not None


def test_get_array(sphere, benchmark):
    """Look up an array through the dataset helper."""
    assert benchmark(get_array, sphere, 'data').size


def test_get_array_association(sphere, benchmark):
    """Resolve which attribute table holds an array."""
    assert benchmark(get_array_association, sphere, 'data') is not None


def test_dataset_getitem(sphere, benchmark):
    """Look up an array with dataset item access."""
    assert benchmark(sphere.__getitem__, 'data').size


def test_attributes_getitem(sphere, benchmark):
    """Look up a vector array through the attributes wrapper."""
    assert benchmark(sphere.point_data.__getitem__, 'vec').size


def test_get_data_range(sphere, benchmark):
    """Read the range of a named array."""
    assert benchmark(sphere.get_data_range, 'data')


def test_dataset_setitem(mutable_sphere, scalars, benchmark):
    """Assign an array with dataset item access."""
    benchmark(mutable_sphere.__setitem__, 'new', scalars)
    assert 'new' in mutable_sphere.point_data


def test_attributes_setitem(mutable_sphere, scalars, benchmark):
    """Assign an array through the attributes wrapper."""
    benchmark(mutable_sphere.point_data.__setitem__, 'new', scalars)
    assert 'new' in mutable_sphere.point_data


def test_set_active_scalars(mutable_sphere, benchmark):
    """Set the active scalars by name."""
    benchmark(mutable_sphere.set_active_scalars, 'data')
    assert mutable_sphere.active_scalars_name == 'data'
