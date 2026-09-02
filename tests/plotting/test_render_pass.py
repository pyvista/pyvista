from __future__ import annotations

import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista.plotting.render_passes import RenderPasses


# this ideally would be a fixture, but if it's a fixture the renderer object
# collects immediately since RenderPasses only holds a weakref
def make_passes():
    ren = _vtk.vtkRenderer()
    passes = RenderPasses(ren)
    return ren, passes


def test_render_passes_init():
    ren, passes = make_passes()
    assert passes._renderer is ren
    del ren

    # ensure renderer is collected
    assert passes._renderer is None


def test_blur_pass():
    _ren, passes = make_passes()
    assert not passes._blur_passes
    blur_pass = passes.add_blur_pass()
    assert isinstance(blur_pass, _vtk.vtkGaussianBlurPass)
    assert len(passes._blur_passes) == 1

    passes.remove_blur_pass()
    assert not passes._blur_passes

    # double pass should work
    blur_pass = passes.add_blur_pass()
    blur_pass = passes.add_blur_pass()
    assert len(passes._blur_passes) == 2


def test_ssaa_pass():
    _ren, passes = make_passes()
    assert not passes._passes
    ssaa_pass = passes.enable_ssaa_pass()
    assert isinstance(ssaa_pass, _vtk.vtkSSAAPass)
    assert list(passes._passes.keys()).count('vtkSSAAPass') == 1

    # enabling again should not add the pass again
    ssaa_pass = passes.enable_ssaa_pass()
    assert list(passes._passes.keys()).count('vtkSSAAPass') == 1

    passes.disable_ssaa_pass()
    assert not passes._passes

    # disabling again should just do nothing
    passes.disable_ssaa_pass()
    assert not passes._passes


def test_depth_of_field_pass():
    _ren, passes = make_passes()
    assert not passes._passes
    ren_pass = passes.enable_depth_of_field_pass()
    assert isinstance(ren_pass, _vtk.vtkDepthOfFieldPass)
    assert list(passes._passes.keys()).count('vtkDepthOfFieldPass') == 1

    # enabling again should not add the pass again
    ren_pass = passes.enable_depth_of_field_pass()
    assert list(passes._passes.keys()).count('vtkDepthOfFieldPass') == 1

    passes.disable_depth_of_field_pass()
    assert not passes._passes

    # disabling again should just do nothing
    passes.disable_depth_of_field_pass()
    assert not passes._passes


def test_depth_of_field_raise_no_ssao():
    _ren, passes = make_passes()
    passes.enable_ssao_pass(radius=0.5, bias=0.005, kernel_size=16, blur=False)
    with pytest.raises(RuntimeError, match='Depth of field pass is incompatible'):
        passes.enable_depth_of_field_pass()


def test_ssao_raise_no_depth_of_field():
    _ren, passes = make_passes()
    passes.enable_depth_of_field_pass()
    with pytest.raises(RuntimeError, match='SSAO pass is incompatible'):
        passes.enable_ssao_pass(radius=0.5, bias=0.005, kernel_size=16, blur=False)


def test_shadow_pass():
    ren, passes = make_passes()
    ren_pass = passes.enable_shadow_pass()
    assert isinstance(ren_pass, _vtk.vtkShadowMapPass)

    assert passes._pass_collection.IsItemPresent(ren_pass)
    assert passes._pass_collection.IsItemPresent(ren_pass.GetShadowMapBakerPass())
    assert ren.GetPass() is not None

    passes.disable_shadow_pass()
    assert not passes._pass_collection.IsItemPresent(ren_pass)
    assert not passes._pass_collection.IsItemPresent(ren_pass.GetShadowMapBakerPass())
    assert passes._shadow_map_pass is None
    assert ren.GetPass() is None

    # enabling again after disabling should add a new pass
    new_pass = passes.enable_shadow_pass()
    assert isinstance(new_pass, _vtk.vtkShadowMapPass)
    assert new_pass is not ren_pass
    assert passes._pass_collection.IsItemPresent(new_pass)
    assert passes._pass_collection.IsItemPresent(new_pass.GetShadowMapBakerPass())


def test_edl_pass():
    _ren, passes = make_passes()
    assert not passes._passes
    ren_pass = passes.enable_edl_pass()
    assert isinstance(ren_pass, _vtk.vtkEDLShading)
    assert list(passes._passes.keys()).count('vtkEDLShading') == 1

    # enabling again should just not add the pass again
    ren_pass = passes.enable_edl_pass()
    assert list(passes._passes.keys()).count('vtkEDLShading') == 1

    passes.disable_edl_pass()
    assert not passes._passes

    # disabling again should just do nothing
    passes.disable_edl_pass()
    assert not passes._passes


def test_ssao_pass():
    _ren, passes = make_passes()
    assert not passes._passes

    ren_pass = passes.enable_ssao_pass(radius=0.5, bias=0.005, kernel_size=16, blur=False)
    assert isinstance(ren_pass, _vtk.vtkSSAOPass)
    assert list(passes._passes.keys()).count('vtkSSAOPass') == 1

    # enabling again should just not add the pass again
    ren_pass = passes.enable_ssao_pass(radius=0.5, bias=0.005, kernel_size=16, blur=False)
    assert list(passes._passes.keys()).count('vtkSSAOPass') == 1

    passes.disable_ssao_pass()
    assert not passes._passes

    # disabling again should just do nothing
    passes.disable_ssao_pass()
    assert not passes._passes


def test_render_passes_deep_clean():
    ren, passes = make_passes()
    passes.add_blur_pass()
    passes.enable_depth_of_field_pass()
    passes.enable_edl_pass()
    passes.enable_shadow_pass()
    passes.enable_ssaa_pass()

    passes.deep_clean()
    del ren
    assert passes._renderer is None

    assert passes._RenderPasses__seq_pass is None
    assert passes._RenderPasses__pass_collection is None
    assert passes._RenderPasses__camera_pass is None
    assert passes._passes == {}
    assert passes._shadow_map_pass is None
    assert passes._edl_pass is None
    assert passes._dof_pass is None
    assert passes._ssaa_pass is None
    assert passes._blur_passes == []


@pytest.mark.parametrize(
    ('enable', 'disable'),
    [
        pytest.param(
            'enable_eye_dome_lighting',
            'disable_eye_dome_lighting',
            marks=pytest.mark.skip_windows('No testing on windows for EDL'),
        ),
        ('enable_shadows', 'disable_shadows'),
        ('enable_depth_of_field', 'disable_depth_of_field'),
        ('add_blurring', 'remove_blurring'),
    ],
)
def test_render_pass_releases_graphics_resources(enable, disable):
    with pv.VtkErrorCatcher() as catcher:
        pl = pv.Plotter()
        pl.add_mesh(pv.Sphere())
        getattr(pl, enable)()
        pl.show()
    assert catcher.error_events == []

    with pv.VtkErrorCatcher() as catcher:
        pl = pv.Plotter()
        pl.add_mesh(pv.Sphere())
        getattr(pl, enable)()
        pl.show(auto_close=False)
        getattr(pl, disable)()
        pl.close()
    assert catcher.error_events == []
