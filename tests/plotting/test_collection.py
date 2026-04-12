"""This module contains any tests which cause memory leaks."""

from __future__ import annotations

import gc
import weakref

import numpy as np

import pyvista as pv
from pyvista.core import _vtk_core as _vtk


def test_pyvistandarray_assign(sphere):
    sphere.point_data['data'] = range(sphere.n_points)

    # this might leave a reference behind if we don't properly use the pointer
    # to the vtk array.
    sphere.point_data['data'] = sphere.point_data['data']


def test_pyvistandarray_strides(sphere):
    sphere['test_scalars'] = sphere.points[:, 2]
    assert np.allclose(sphere['test_scalars'], sphere.points[:, 2])


def test_complex_collection(plane):
    name = 'my_data'
    data = np.random.default_rng().random((plane.n_points, 2)).view(np.complex128).ravel()
    plane.point_data[name] = data

    # ensure shallow copy
    assert np.shares_memory(plane.point_data[name], data)

    # ensure data remains but original numpy object does not
    ref = weakref.ref(data)
    data_copy = data.copy()
    del data
    assert np.allclose(plane.point_data[name], data_copy)

    assert ref() is None


def test_add_array(sphere):
    """Ensure data added dynamically to a plotter is collected."""
    pl = pv.Plotter()
    pl.add_mesh(sphere, scalars=range(sphere.n_points))


def test_plotting_collection():
    """Ensure that we don't leak Plotter, Renderer and Charts instances."""
    pl = pv.Plotter()
    pl.add_chart(pv.Chart2D())
    ref_plotter = weakref.ref(pl)
    ref_renderers = weakref.ref(pl.renderers)
    ref_renderer = weakref.ref(pl.renderer)
    ref_charts = weakref.ref(pl.renderer._charts)

    # delete known references to Plotter
    del pv.plotting.plotter._ALL_PLOTTERS[pl._id_name]
    del pl

    # check that everything is eventually destroyed
    gc.collect()  # small reference cycles allowed
    assert ref_plotter() is None, gc.get_referrers(ref_plotter())
    assert ref_renderers() is None
    assert ref_renderer() is None
    assert ref_charts() is None


def test_vtk_points_slice():
    mesh = pv.Sphere()
    n = 10
    orig_points = np.array(mesh.points[:n])
    pts = pv.vtk_points(mesh.points[:n], deep=False)
    assert isinstance(pts, _vtk.vtkPoints)

    del mesh
    gc.collect()
    assert np.allclose(_vtk.vtk_to_numpy(pts.GetData()), orig_points)


def test_vtk_points():
    mesh = pv.Sphere()
    orig_points = np.array(mesh.points)
    pts = pv.vtk_points(mesh.points, deep=False)
    assert isinstance(pts, _vtk.vtkPoints)
    assert np.shares_memory(mesh.points, _vtk.vtk_to_numpy(pts.GetData()))

    del mesh
    gc.collect()
    assert np.allclose(_vtk.vtk_to_numpy(pts.GetData()), orig_points)
