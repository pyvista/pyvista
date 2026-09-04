"""Tests for pyvista.ext._expanded_sidebar, checked against the theme it patches."""

from __future__ import annotations

import posixpath
import re
from typing import TYPE_CHECKING

import pytest
from sphinx.application import Sphinx
from sphinx.util.docutils import docutils_namespace

if TYPE_CHECKING:
    from pathlib import Path

# Raised by sphinx_book_theme 1.4.0 under Sphinx 9, not by anything under test.
pytestmark = pytest.mark.filterwarnings('ignore::sphinx.deprecation.RemovedInSphinx11Warning')

SECTIONS = 2
SUBSECTIONS = 2
PAGES = 2

CONF = """\
project = 'tinypages_expanded_sidebar'
html_theme = 'sphinx_book_theme'
exclude_patterns = ['_build']
extensions = {extensions!r}
html_theme_options = {{
    'collapse_navbar': False,
    'show_navbar_depth': 1,
    'max_navbar_depth': 4,
    'navbar_persistent': [],
}}
"""

NAV = re.compile(r'<nav class="bd-links bd-docs-nav".*?</nav>', re.DOTALL)
HREF = re.compile(r'href="([^"]*)"')


def write_project(root: Path, *, extensions: list[str]) -> None:
    """Write a small nested Sphinx project using the documentation theme."""
    root.mkdir(parents=True)
    (root / 'conf.py').write_text(CONF.format(extensions=extensions), encoding='utf-8')

    def page(path: Path, title: str, entries: list[str]) -> None:
        body = [title, '=' * len(title), '', 'Body text.', '']
        if entries:
            body += ['.. toctree::', '   :maxdepth: 2', '', *(f'   {e}' for e in entries), '']
        path.write_text('\n'.join(body), encoding='utf-8')

    page(root / 'index.rst', 'Root', [f'sec{s}/index' for s in range(SECTIONS)])
    for s in range(SECTIONS):
        section = root / f'sec{s}'
        section.mkdir()
        page(section / 'index.rst', f'Section {s}', [f'sub{b}/index' for b in range(SUBSECTIONS)])
        for b in range(SUBSECTIONS):
            subsection = section / f'sub{b}'
            subsection.mkdir()
            leaves = [f'page{p}' for p in range(PAGES)]
            page(subsection / 'index.rst', f'Subsection {s}.{b}', leaves)
            for p in range(PAGES):
                page(subsection / f'page{p}.rst', f'Page {s}.{b}.{p}', [])


def build(source: Path) -> Path:
    """Build the project to HTML and return the output directory."""
    out = source / '_build'
    with docutils_namespace():
        app = Sphinx(
            str(source), str(source), str(out), str(out / '.doctrees'), 'html', status=None
        )
        app.build()
    return out


def sidebar(out: Path, page: str) -> str:
    """Return one page's sidebar markup, with every link resolved to a canonical path."""
    match = NAV.search((out / page).read_text(encoding='utf-8'))
    assert match is not None, f'no sidebar in {page}'
    directory = posixpath.dirname(page)

    def resolve(href: re.Match[str]) -> str:
        target = href[1]
        if target in {'', '#'} or ':' in target or target.startswith('/'):
            return href[0]
        return f'href="{posixpath.normpath(posixpath.join(directory, target))}"'

    return HREF.sub(resolve, match[0])


@pytest.fixture(scope='module')
def built(tmp_path_factory) -> tuple[Path, Path, list[str]]:
    """Build the same project twice, with the extension and without it."""
    root = tmp_path_factory.mktemp('expanded_sidebar')
    outputs = []
    for name, extensions in (('plain', []), ('cached', ['pyvista.ext._expanded_sidebar'])):
        source = root / name
        write_project(source, extensions=extensions)
        outputs.append(build(source))
    plain, cached = outputs
    pages = sorted(
        str(p.relative_to(plain))
        for p in plain.rglob('*.html')
        if '_static' not in p.relative_to(plain).parts
    )
    return plain, cached, pages


def test_sidebar_matches_theme(built):
    """The cached sidebar is what the theme renders, page for page."""
    plain, cached, pages = built
    assert pages, 'no pages were built'
    for page in pages:
        assert sidebar(cached, page) == sidebar(plain, page), page


def test_sidebar_is_expanded(built):
    """Every page appears in every sidebar, so the comparison is not of pruned trees."""
    plain, _, pages = built
    links = HREF.findall(sidebar(plain, 'index.html'))
    assert set(pages) <= set(links) | {'index.html', 'genindex.html', 'search.html'}


def test_current_page_is_marked(built):
    """A deep page's own entry and each of its ancestors are marked current."""
    _, cached, _ = built
    markup = sidebar(cached, 'sec1/sub1/page1.html')
    assert markup.count('current active') == 3  # section, subsection, page
    assert '<a class="current reference internal" href="#">' in markup
    assert markup.count('<details open="open">') == 2  # section, subsection
