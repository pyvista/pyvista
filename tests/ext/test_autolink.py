"""Unit tests for pyvista.ext._autolink internals, without a full Sphinx build."""

from __future__ import annotations

import re
from types import SimpleNamespace

from pyvista.ext import _autolink


def test_accessed_names_syntax_error():
    assert _autolink._accessed_names('def bad(:\n') == set()


def test_accessed_names_call_chain_not_rooted_in_name():
    # `.plot` on a call result has nothing to look up, but the inner `pv.Sphere` does.
    assert _autolink._accessed_names('pv.Sphere().plot()') == {'pv.Sphere'}


def test_module_path_candidates_no_module():
    class Foo:
        pass

    Foo.__module__ = 'pyvista._nonexistent_test_module_xyz'
    assert list(_autolink._module_path_candidates(Foo, [])) == []


def test_candidate_names_unresolvable():
    assert _autolink._candidate_names('totally_undefined_name', {}) == []


def test_candidate_names_property():
    class Widget:
        @property
        def name(self):
            return 'widget'

    namespace = {'widget': Widget()}
    candidates = _autolink._candidate_names('widget.name.upper', namespace)
    assert any(c.endswith('Widget.name') for c in candidates)


def test_candidate_names_getattr_raises():
    # object() has no .nonexistent attribute -- like a variable reassigned mid-script.
    namespace = {'x': object()}
    assert _autolink._candidate_names('x.nonexistent.deep', namespace) == list(
        _autolink._module_path_candidates(object, [])
    )


def test_candidate_names_module_reexported_function():
    # load_uniform is documented at pyvista.examples.examples, not the public path.
    from pyvista import examples

    candidates = _autolink._candidate_names('examples.load_uniform', {'examples': examples})
    assert candidates == [
        'pyvista.examples.examples.load_uniform',
        'pyvista.examples.load_uniform',
        'pyvista.load_uniform',
    ]


def test_candidate_names_module_attribute_is_module():
    import pyvista as pv

    assert _autolink._candidate_names('pv.examples', {'pv': pv}) == ['pyvista.examples']


def test_call_chains_no_intermediate_variable():
    assert _autolink._call_chains('pv.Sphere().plot()') == {('pv.Sphere', ('plot',))}


def test_call_chains_bound_method():
    assert _autolink._call_chains('mesh.copy().plot()') == {('mesh.copy', ('plot',))}


def test_call_chains_multi_attribute_trailing():
    assert _autolink._call_chains('pv.Sphere().points.size') == {('pv.Sphere', ('points', 'size'))}


def test_resolve_object():
    import pyvista as pv

    assert _autolink._resolve_object('pv.Sphere', {'pv': pv}) is pv.Sphere
    assert _autolink._resolve_object('pv.nonexistent', {'pv': pv}) is None
    assert _autolink._resolve_object('undefined', {}) is None


def test_call_return_type_resolves_via_fallback_namespace():
    import pyvista as pv

    # 'PolyData' isn't in Sphere's own module globals (TYPE_CHECKING-only there), but is on pv.
    assert 'PolyData' not in pv.Sphere.__globals__
    assert _autolink._call_return_type(pv.Sphere, {'pv': pv}) is pv.PolyData


def test_call_return_type_rejects_complex_annotation():
    def make_widget_or_string() -> Widget | str:  # noqa: F821
        return ''

    assert _autolink._call_return_type(make_widget_or_string, {}) is None


def test_call_return_type_no_annotation():
    def plain():
        return None

    assert _autolink._call_return_type(plain, {}) is None


def test_call_chain_candidates():
    import pyvista as pv

    candidates = _autolink._call_chain_candidates('pv.Sphere', ('plot',), {'pv': pv})
    assert 'pyvista.PolyData.plot' in candidates


def test_call_chain_candidates_unresolvable_target():
    assert _autolink._call_chain_candidates('undefined', ('plot',), {}) == []


def test_intersphinx_inventory():
    env = SimpleNamespace(
        intersphinx_cache={},
        intersphinx_inventory={
            'py:function': {
                'external.thing': ('external', '1.0', 'https://example.invalid/thing.html', '-'),
            },
        },
        intersphinx_named_inventory={},
    )
    app = SimpleNamespace(env=env)
    assert _autolink._intersphinx_inventory(app) == {
        'external.thing': 'https://example.invalid/thing.html',
    }


def test_resolve_link_external():
    link = _autolink._resolve_link(
        ('external.thing',),
        docname='index',
        app=None,
        local={},
        external={'external.thing': 'https://example.invalid/thing.html'},
    )
    assert link == 'https://example.invalid/thing.html'


def test_embed_links_skips_on_exception():
    app = SimpleNamespace(builder=SimpleNamespace(format='html'))
    assert _autolink._embed_links(app, Exception('build failed')) is None


def test_embed_links_skips_non_html_builder():
    app = SimpleNamespace(builder=SimpleNamespace(format='text'))
    assert _autolink._embed_links(app, None) is None


def _fake_env():
    return SimpleNamespace(
        intersphinx_cache={},
        intersphinx_inventory={},
        intersphinx_named_inventory={},
        domains={'py': SimpleNamespace(objects={})},
    )


def test_embed_links_missing_output_file(tmp_path):
    env = _fake_env()
    setattr(env, _autolink._ENV_ATTR, {'missing_doc': [_autolink._Candidate('name', ('x.Foo',))]})
    app = SimpleNamespace(
        builder=SimpleNamespace(format='html', get_target_uri=lambda docname: f'{docname}.html'),
        outdir=str(tmp_path),
        env=env,
    )
    assert _autolink._embed_links(app, None) is None


def test_embed_links_no_resolved_candidates(tmp_path):
    out_file = tmp_path / 'exists_doc.html'
    out_file.write_text('<html><body><span class="n">name</span></body></html>')

    env = _fake_env()
    setattr(env, _autolink._ENV_ATTR, {'exists_doc': [_autolink._Candidate('name', ('x.Foo',))]})
    app = SimpleNamespace(
        builder=SimpleNamespace(format='html', get_target_uri=lambda docname: f'{docname}.html'),
        outdir=str(tmp_path),
        env=env,
    )
    _autolink._embed_links(app, None)
    assert out_file.read_text() == '<html><body><span class="n">name</span></body></html>'


def test_embed_links_call_chain(tmp_path):
    # `pv.Sphere().plot()`, syntax-highlighted.
    html = (
        '<pre><span class="n">pv</span><span class="o">.</span><span class="n">Sphere</span>'
        '<span class="p">()</span><span class="o">.</span><span class="n">plot</span>'
        '<span class="p">()</span></pre>'
    )
    out_file = tmp_path / 'index.html'
    out_file.write_text(html)

    env = _fake_env()
    env.domains['py'].objects['pyvista.PolyData.plot'] = SimpleNamespace(
        docname='api', node_id='pyvista.PolyData.plot'
    )
    setattr(
        env,
        _autolink._ENV_ATTR,
        {'index': [_autolink._CallCandidate('pv.Sphere', ('plot',), ('pyvista.PolyData.plot',))]},
    )
    app = SimpleNamespace(
        builder=SimpleNamespace(
            format='html',
            get_target_uri=lambda docname: f'{docname}.html',
            get_relative_uri=lambda _from, to: to,
        ),
        outdir=str(tmp_path),
        env=env,
    )
    _autolink._embed_links(app, None)
    result = out_file.read_text()

    # only `.plot` is linked -- `pv.Sphere()` and the trailing `()` stay outside.
    assert '<span class="n">Sphere</span></a>' not in result
    assert (
        '<a class="pyvista-autolink-a" href="api#pyvista.PolyData.plot">'
        '<span class="o">.</span><span class="n">plot</span></a>' in result
    )
    assert re.search(r'<a\b[^>]*><a\b', result) is None


def test_embed_links_call_chain_and_plain_candidate_coexist(tmp_path):
    html = (
        '<pre><span class="n">pv</span><span class="o">.</span><span class="n">Sphere</span>'
        '<span class="p">()</span><span class="o">.</span><span class="n">plot</span>'
        '<span class="p">()</span></pre>'
    )
    out_file = tmp_path / 'index.html'
    out_file.write_text(html)

    env = _fake_env()
    env.domains['py'].objects['pyvista.PolyData.plot'] = SimpleNamespace(
        docname='api', node_id='pyvista.PolyData.plot'
    )
    env.domains['py'].objects['pyvista.Sphere'] = SimpleNamespace(
        docname='api2', node_id='pyvista.Sphere'
    )
    setattr(
        env,
        _autolink._ENV_ATTR,
        {
            'index': [
                _autolink._CallCandidate('pv.Sphere', ('plot',), ('pyvista.PolyData.plot',)),
                _autolink._Candidate('pv.Sphere', ('pyvista.Sphere',)),
            ]
        },
    )
    app = SimpleNamespace(
        builder=SimpleNamespace(
            format='html',
            get_target_uri=lambda docname: f'{docname}.html',
            get_relative_uri=lambda _from, to: to,
        ),
        outdir=str(tmp_path),
        env=env,
    )
    _autolink._embed_links(app, None)
    result = out_file.read_text()

    assert 'href="api2#pyvista.Sphere"' in result
    assert 'href="api#pyvista.PolyData.plot"' in result
    assert re.search(r'<a\b[^>]*><a\b', result) is None
