from pyvista import _vtk
from pyvista.plotting.render_passes import RenderPasses


# this ideally would be a fixture, but if it's a fixture renderer collects
# immediately
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
    ren, passes = make_passes()
    assert not passes._blur_passes
    blur_pass = passes.add_blur_pass()
    assert isinstance(blur_pass, _vtk.vtkGaussianBlurPass)
    assert len(passes._blur_passes) == 1

    passes.remove_blur_pass()
    assert not passes._blur_passes


def test_ssaa_pass():
    ren, passes = make_passes()
    assert not passes._passes
    ssaa_pass = passes.enable_ssaa_pass()
    assert isinstance(ssaa_pass, _vtk.vtkSSAAPass)

    assert 'vtkSSAAPass' in passes._passes

    passes.disable_ssaa_pass()
    assert not passes._passes


def test_depth_of_field_pass():
    ren, passes = make_passes()
    assert not passes._passes
    ren_pass = passes.enable_depth_of_field_pass()
    assert isinstance(ren_pass, _vtk.vtkDepthOfFieldPass)

    assert 'vtkDepthOfFieldPass' in passes._passes

    passes.disable_depth_of_field_pass()
    assert not passes._passes


def test_shadow_pass():
    ren, passes = make_passes()
    ren_pass = passes.enable_shadow_pass()
    assert isinstance(ren_pass, _vtk.vtkShadowMapPass)

    assert passes._pass_collection.IsItemPresent(ren_pass)

    passes.disable_shadow_pass()
    assert not passes._pass_collection.IsItemPresent(ren_pass)


def test_edl_pass():
    ren, passes = make_passes()
    assert not passes._passes
    ren_pass = passes.enable_edl_pass()
    assert isinstance(ren_pass, _vtk.vtkEDLShading)

    assert 'vtkEDLShading' in passes._passes

    passes.disable_edl_pass()
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
