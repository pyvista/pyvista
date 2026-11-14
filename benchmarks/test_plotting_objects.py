"""Benchmarks for building and updating plotting objects without rendering."""

from __future__ import annotations

import pyvista as pv


def test_axes_assembly(benchmark):
    """Build an axes assembly."""
    assert benchmark(pv.AxesAssembly).parts


def test_axes_assembly_set_shaft_length(benchmark):
    """Update the shaft length of an axes assembly."""
    axes = pv.AxesAssembly()
    benchmark(setattr, axes, 'shaft_length', 0.5)
    assert axes.shaft_length == (0.5, 0.5, 0.5)


def test_property(benchmark):
    """Build an actor property, which snapshots the global theme."""
    assert benchmark(pv.Property).opacity == 1.0


def test_set_camera_position(plotter, benchmark):
    """Assign a camera position to a plotter."""
    position = [(5.0, 5.0, 5.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    benchmark(setattr, plotter, 'camera_position', position)
    assert plotter.camera_position.focal_point == (0.0, 0.0, 0.0)


def test_add_mesh_shared_scalar_bar(plotter, sphere, benchmark):
    """Add a mesh whose scalars share the scalar bar of twenty earlier meshes."""
    for _ in range(20):
        plotter.add_mesh(sphere, scalars='data')
    assert benchmark(plotter.add_mesh, sphere, scalars='data').mapper
