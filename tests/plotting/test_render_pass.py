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

    steps = passes._pass_collection.GetItemAsObject(0)
    opaque = steps.GetOpaquePass().GetPasses()
    assert opaque.GetItemAsObject(0) is ren_pass.GetShadowMapBakerPass()
    assert opaque.GetItemAsObject(1) is ren_pass
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
    opaque = steps.GetOpaquePass().GetPasses()
    assert opaque.GetItemAsObject(0) is new_pass.GetShadowMapBakerPass()
    assert opaque.GetItemAsObject(1) is new_pass


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

    updated_pass = passes.enable_ssao_pass(radius=2.0, bias=0.02, kernel_size=64, blur=True)
    assert updated_pass is ren_pass
    assert ren_pass.GetRadius() == 2.0
    assert ren_pass.GetBias() == 0.02
    assert ren_pass.GetKernelSize() == 64
    assert ren_pass.GetBlur()
    assert list(passes._passes.keys()).count('vtkSSAOPass') == 1

    modification_time = ren_pass.GetMTime()
    passes.enable_ssao_pass(radius=2.0, bias=0.02, kernel_size=64, blur=True)
    assert ren_pass.GetMTime() == modification_time

    passes.disable_ssao_pass()
    assert not passes._passes

    # disabling again should just do nothing
    passes.disable_ssao_pass()
    assert not passes._passes


@pytest.mark.parametrize('effect', ['enable_depth_of_field_pass', 'enable_shadow_pass'])
def test_render_passes_deep_clean(effect):
    ren, passes = make_passes()
    passes.add_blur_pass()
    getattr(passes, effect)()
    passes.enable_edl_pass()
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


def test_ssao_pass_closed():
    """A closed renderer rejects SSAO before allocating a pass."""
    _ren, passes = make_passes()
    passes.close()
    with pytest.raises(RuntimeError, match='closed'):
        passes.enable_ssao_pass(radius=0.5, bias=0.005, kernel_size=256, blur=True)
    assert passes._ssao_pass is None
    assert not passes._passes


@pytest.mark.parametrize('image_pass', [None, 'vtkToneMappingPass', 'vtkSSAAPass'])
def test_external_base_restore(image_pass):
    """A custom translucent pipeline survives managed effects and their removal."""
    ren, passes = make_passes()
    steps = _vtk.vtkRenderStepsPass()
    peeling = _vtk.vtkDualDepthPeelingPass()
    translucent = _vtk.vtkTranslucentPass()
    peeling.SetTranslucentPass(translucent)
    steps.SetTranslucentPass(peeling)
    base = steps if image_pass is None else getattr(_vtk, image_pass)()
    if base is not steps:
        base.SetDelegatePass(steps)
    ren.SetPass(base)
    passes.enable_ssao_pass(radius=0.5, bias=0.005, kernel_size=256, blur=True)
    assert passes._pass_collection.GetItemAsObject(0) is steps
    assert steps.GetTranslucentPass() is peeling
    assert peeling.GetTranslucentPass() is translucent
    passes.disable_ssao_pass()
    assert ren.GetPass() is base
    if base is not steps:
        assert base.GetDelegatePass() is steps
    passes.enable_edl_pass()
    passes.deep_clean()
    assert ren.GetPass() is base
    assert steps.GetTranslucentPass() is peeling


def test_external_base_replacement():
    """Replacing or clearing a base between toggles never resurrects the old base."""
    ren, passes = make_passes()
    first = _vtk.vtkRenderStepsPass()
    second = _vtk.vtkRenderStepsPass()
    ren.SetPass(first)
    passes.enable_ssaa_pass()
    ren.SetPass(second)
    passes.add_blur_pass()
    assert passes._pass_collection.GetItemAsObject(0) is second
    passes.remove_blur_pass()
    passes.disable_ssaa_pass()
    assert ren.GetPass() is second
    ren.SetPass(None)
    passes.enable_ssaa_pass()
    passes.disable_ssaa_pass()
    assert ren.GetPass() is None


def test_external_delegate_edit():
    """An edit beneath an unchanged external root is retained on disable."""
    ren, passes = make_passes()
    tone = _vtk.vtkToneMappingPass()
    first = _vtk.vtkRenderStepsPass()
    second = _vtk.vtkRenderStepsPass()
    tone.SetDelegatePass(first)
    ren.SetPass(tone)
    passes.enable_ssaa_pass()
    assert ren.GetPass() is tone
    tone.SetDelegatePass(second)
    passes.add_blur_pass()
    assert passes._pass_collection.GetItemAsObject(0) is second
    passes.remove_blur_pass()
    passes.disable_ssaa_pass()
    assert ren.GetPass() is tone
    assert tone.GetDelegatePass() is second


@pytest.mark.parametrize('wrapper', ['vtkCameraPass', 'vtkGaussianBlurPass'])
def test_external_managed_cycle_rejected(wrapper):
    """Wrapping an active managed graph fails before allocating another pass."""
    ren, passes = make_passes()
    active = passes.enable_ssao_pass(radius=0.5, bias=0.005, kernel_size=256, blur=True)
    external = getattr(_vtk, wrapper)()
    external.SetDelegatePass(active)
    ren.SetPass(external)
    with pytest.raises(ValueError, match='cycles or managed render passes'):
        passes.add_blur_pass()
    assert not passes._blur_passes
    assert list(passes._passes) == ['vtkSSAOPass']
    assert ren.GetPass() is external
    external.SetDelegatePass(None)
    ren.SetPass(active)
    passes.deep_clean()


@pytest.mark.parametrize('first', ['dof', 'shadows', 'peeling'])
def test_depth_of_field_incompatible_passes(first):
    """Reject unsupported depth-buffer combinations in either enable order."""
    ren, passes = make_passes()
    if first == 'dof':
        passes.enable_depth_of_field_pass()
        with pytest.raises(RuntimeError, match='incompatible'):
            passes.enable_shadow_pass()
        assert passes._shadow_map_pass is None
    else:
        if first == 'shadows':
            passes.enable_shadow_pass()
        else:
            ren.SetUseDepthPeeling(True)
        with pytest.raises(RuntimeError, match='incompatible'):
            passes.enable_depth_of_field_pass()
        assert passes._dof_pass is None


def test_render_steps_edits_preserved():
    """Restoring borrowed geometry stages preserves caller changes."""
    ren, passes = make_passes()
    steps = _vtk.vtkRenderStepsPass()
    ren.SetPass(steps)
    passes.enable_shadow_pass()
    translucent = _vtk.vtkTranslucentPass()
    steps.SetTranslucentPass(translucent)
    passes.disable_shadow_pass()
    assert steps.GetTranslucentPass() is translucent
    assert ren.GetPass() is steps


@pytest.mark.parametrize('slot', ['OpaquePass', 'CameraPass'])
def test_external_render_steps_cycle_rejected(slot):
    """Edits to borrowed geometry or camera stages cannot introduce managed cycles."""
    ren, passes = make_passes()
    steps = _vtk.vtkRenderStepsPass()
    original = getattr(steps, 'Get' + slot)()
    ren.SetPass(steps)
    active = passes.enable_ssao_pass(radius=0.5, bias=0.005, kernel_size=256, blur=True)
    getattr(steps, 'Set' + slot)(active if slot == 'OpaquePass' else passes._camera_pass)
    with pytest.raises(ValueError, match='cycles or managed render passes'):
        passes.add_blur_pass()
    assert not passes._blur_passes
    getattr(steps, 'Set' + slot)(original)
    passes.deep_clean()


def test_reenable_after_external_replacement():
    """Re-enabling SSAO reconnects it to a newly installed external base."""
    ren, passes = make_passes()
    active = passes.enable_ssao_pass(radius=0.5, bias=0.005, kernel_size=256, blur=True)
    steps = _vtk.vtkRenderStepsPass()
    ren.SetPass(steps)
    passes.enable_ssao_pass(radius=0.8, bias=0.005, kernel_size=256, blur=True)
    assert ren.GetPass() is active
    assert passes._pass_collection.GetItemAsObject(0) is steps
    passes.disable_ssao_pass()
    assert ren.GetPass() is steps


def test_external_same_class_order():
    """Distinct tone-mapping operations retain their original relative order."""
    ren, passes = make_passes()
    outer = _vtk.vtkToneMappingPass()
    inner = _vtk.vtkToneMappingPass()
    steps = _vtk.vtkRenderStepsPass()
    outer.SetDelegatePass(inner)
    inner.SetDelegatePass(steps)
    ren.SetPass(outer)
    passes.enable_ssaa_pass()
    assert ren.GetPass() is outer
    assert outer.GetDelegatePass() is inner
    assert inner.GetDelegatePass() is passes._ssaa_pass
    passes.disable_ssaa_pass()
    assert outer.GetDelegatePass() is inner
    assert inner.GetDelegatePass() is steps


def test_external_fxaa_not_duplicated():
    """The caller's FXAA pass satisfies the renderer's FXAA flag."""
    ren, passes = make_passes()
    fxaa = _vtk.vtkOpenGLFXAAPass()
    fxaa.SetDelegatePass(_vtk.vtkRenderStepsPass())
    ren.SetPass(fxaa)
    ren.SetUseFXAA(True)
    passes.enable_ssao_pass(radius=0.5, bias=0.005, kernel_size=256, blur=True)
    assert passes._fxaa_pass is None
    assert ren.GetPass() is fxaa
    assert fxaa.GetDelegatePass() is passes._ssao_pass
    passes.disable_ssao_pass()
    assert ren.GetPass() is fxaa


@pytest.mark.parametrize('enable', ['enable_shadow_pass', 'enable_ssao_pass'])
def test_external_depth_of_field_rejected(enable):
    """The incompatibility checks include externally installed image passes."""
    ren, passes = make_passes()
    dof = _vtk.vtkDepthOfFieldPass()
    dof.SetDelegatePass(_vtk.vtkRenderStepsPass())
    ren.SetPass(dof)
    kwargs = (
        dict(radius=0.5, bias=0.005, kernel_size=256, blur=True)
        if enable == 'enable_ssao_pass'
        else {}
    )
    with pytest.raises(RuntimeError, match='incompatible'):
        getattr(passes, enable)(**kwargs)
    assert not passes._passes
    assert ren.GetPass() is dof


def test_release_uses_owning_context(mocker):
    """Removing a pass activates its live window before freeing GPU resources."""
    pl = pv.Plotter()
    pl.enable_ssao()
    render_pass = pl.renderer._render_passes._ssao_pass
    calls = []
    mocker.patch.object(pl.ren_win, 'MakeCurrent', side_effect=lambda: calls.append('current'))
    mocker.patch.object(render_pass, 'ReleaseGraphicsResources', side_effect=calls.append)
    pl.disable_ssao()
    assert calls == ['current', pl.ren_win]
    assert render_pass.GetDelegatePass() is None
    pl.close()
