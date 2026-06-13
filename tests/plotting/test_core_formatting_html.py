"""Tests for core HTML formatting that require plotting classes."""

from __future__ import annotations

import pyvista as pv


def test_camera_position_repr():
    cp = pv.CameraPosition((1.0, 2.0, 3.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    html = cp._repr_html_()
    assert 'CameraPosition' in html
    assert 'position' in html
    assert 'focal_point' in html
    assert 'viewup' in html
    assert 'pv-copy-btn' in html
