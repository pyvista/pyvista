"""Regression tests for boundary and non-manifold edge classification."""

from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv


def _issue_mesh():
    """Build the three-quad mesh from issue #4738."""
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [1, -1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
    ]
    faces = [4, 0, 1, 3, 2, 4, 0, 1, 5, 4, 4, 0, 1, 7, 6]
    return pv.PolyData(np.asarray(points, dtype=float), faces)


def _closed_non_manifold_mesh():
    """Build two closed tetrahedra that share just one edge."""
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1]], dtype=float
    )
    triangles = [
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
        [0, 1, 4],
        [0, 5, 1],
        [1, 5, 4],
        [4, 5, 0],
    ]
    faces = np.array([[3, *triangle] for triangle in triangles]).ravel()
    return pv.PolyData(np.asarray(points, dtype=float), faces)


@pytest.mark.parametrize(
    ('mesh_factory', 'boundary_count', 'non_manifold_count', 'is_manifold'),
    [
        (_issue_mesh, 9, 1, False),
        (_closed_non_manifold_mesh, 0, 1, False),
        (pv.Sphere, 0, 0, True),
        (lambda: pv.Plane(i_resolution=1, j_resolution=1), 4, 0, False),
    ],
)
def test_open_edges_count_only_boundary_edges(
    mesh_factory, boundary_count, non_manifold_count, is_manifold
):
    """Count exposed edges without hiding non-manifold edges from validity checks."""
    mesh = mesh_factory()
    extracted = mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False,
    )
    non_manifold = mesh.extract_feature_edges(
        boundary_edges=False,
        non_manifold_edges=True,
        feature_edges=False,
        manifold_edges=False,
    )
    assert mesh.n_open_edges == extracted.n_cells == boundary_count
    assert non_manifold.n_cells == non_manifold_count
    assert mesh.is_manifold is is_manifold


def test_closed_non_manifold_surface_still_rejected():
    """Filters requiring manifold input reject a surface with no boundary edges."""
    mesh = _closed_non_manifold_mesh()
    assert mesh.n_open_edges == 0
    with pytest.raises(ValueError, match='non-manifold'):
        mesh.clip_closed_surface()
    with pytest.raises(RuntimeError, match='not closed'):
        with pytest.warns(pv.PyVistaDeprecationWarning):
            pv.PolyData([[0.5, 0, 0]]).select_enclosed_points(mesh)


def test_image_clip_avoids_fast_path_for_non_manifold_surface(monkeypatch):
    """Image clipping only uses the fast path for a closed manifold surface."""
    from pyvista.core.filters import data_set

    mesh = _closed_non_manifold_mesh()
    assert mesh.n_open_edges == 0

    fast_path_calls = []

    def record_fast_path(*_args):
        fast_path_calls.append(None)

    monkeypatch.setattr(data_set, '_signed_distance_near_surface', record_fast_path)
    image = pv.ImageData(dimensions=(4, 4, 4), spacing=(0.5, 0.5, 0.5))
    image.clip_surface(mesh)
    assert not fast_path_calls

    image.clip_surface(pv.Sphere())
    assert len(fast_path_calls) == 1
