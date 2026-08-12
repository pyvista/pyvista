"""Unit tests for pyvista.ext._autolink internals, without a full Sphinx build."""

from __future__ import annotations

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
    # `object()` has no `.nonexistent` attribute -- the same situation as a variable
    # reassigned to a different type between two accesses.
    namespace = {'x': object()}
    assert _autolink._candidate_names('x.nonexistent.deep', namespace) == list(
        _autolink._module_path_candidates(object, [])
    )


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
