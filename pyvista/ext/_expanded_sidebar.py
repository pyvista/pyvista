"""Render the expanded sidebar navigation once and reuse it on every page.

The theme rebuilds the whole ``toctree`` for every page, and renders the sidebar
template twice on top of that, which is why an expanded sidebar was only affordable on
tag builds (#9023, #9070). This works around that: the tree is rendered once against the
root document and the current-page markers are spliced into the cached string (#9082).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Any

from pydata_sphinx_theme.toctree import add_toctree_functions
from sphinx.environment.adapters.toctree import global_toctree_for_doc

if TYPE_CHECKING:
    from bs4 import BeautifulSoup
    from bs4 import Tag
    from sphinx.application import Sphinx

#: Stands in for the ``../`` chain that makes a root-relative href work on a page.
_PREFIX = '\x00P\x00'

#: Attribute that tags a ``<details>`` element until the markers are compiled out.
_DETAILS_ATTR = 'data-pv-nav'

#: Every marker left in the rendered tree, in the order the compiler resolves them.
_TOKENS = re.compile(
    r'\x00L:(\d+)\x00'
    r'|\x00U:(\d+)\x00'
    r'| class="\x00u:(\d+)\x00"'
    rf'| {_DETAILS_ATTR}="\x00D:(\d+)\x00"'
    r'|\x00A:(\d+)\x00'
    r'|href="\x00I:(\d+)\x00(\x00P\x00[^"]*)"'
)

_LIST_ITEM_MARKER = re.compile(r'\x00L:(\d+)\x00')

#: An edit is the span of the cached tree to replace and the text to put there.
_Edit = tuple[int, int, str]


class _ExpandedSidebar:
    """One rendering of the expanded sidebar plus the edits that adapt it to a page."""

    def __init__(self, app: Sphinx, show_nav_level: int, kwargs: dict[str, Any]) -> None:
        soup = self._render_for_root(app, show_nav_level, kwargs)
        ancestors, lists = self._mark(soup, self._docnames(app))
        self._compile(str(soup), ancestors, lists)

    @staticmethod
    def _docnames(app: Sphinx) -> dict[str, str]:
        """Map every document's output URI back to its document name."""
        get_target_uri = app.builder.get_target_uri
        return {get_target_uri(docname): docname for docname in app.env.found_docs}

    @staticmethod
    def _render_for_root(
        app: Sphinx, show_nav_level: int, kwargs: dict[str, Any]
    ) -> BeautifulSoup:
        """Build the sidebar soup as the theme would build it on the root document."""
        root = app.config.root_doc
        builder = app.builder

        def toctree(**kw: Any) -> str:
            node = global_toctree_for_doc(app.env, root, builder, tags=builder.tags, **kw)
            return '' if node is None else builder.render_partial(node)['fragment']

        context: dict[str, Any] = {'toctree': toctree}
        add_toctree_functions(app, root, 'page.html', context, None)
        return context['generate_toctree_html'](
            'sidebar', startdepth=0, show_nav_level=show_nav_level, **kwargs
        )

    def _mark(
        self, soup: BeautifulSoup, docnames: dict[str, str]
    ) -> tuple[list[list[int]], list[int | None]]:
        """Tag every internal entry in the soup and record its list-item ancestry."""
        self.by_docname: dict[str, list[int]] = {}
        ancestors: list[list[int]] = []
        lists: list[int | None] = []
        holder_ids: dict[int, int] = {}
        holders: list[Tag] = []
        for anchor in soup.select('a.reference.internal'):
            href = anchor.get('href', '')
            docname = docnames.get(href)
            item = anchor.find_parent('li')
            if docname is None or item is None:
                continue
            index = len(ancestors)
            ancestors.append(
                [
                    marker
                    for parent in item.find_parents('li')
                    if (marker := self._index_of(parent)) is not None
                ]
            )
            item_classes = item.get('class') or ['']
            item['class'] = [f'{item_classes[0]}\x00L:{index}\x00', *item_classes[1:]]
            anchor_classes = anchor.get('class') or ['']
            anchor['class'] = [f'\x00A:{index}\x00{anchor_classes[0]}', *anchor_classes[1:]]
            anchor['href'] = f'\x00I:{index}\x00{_PREFIX}{href}'
            holder = item.parent
            if holder is not None and holder.name == 'ul':
                lists.append(holder_ids.setdefault(id(holder), len(holder_ids)))
                holders.append(holder)
            else:
                lists.append(None)
            details = item.find('details', recursive=False)
            if details is not None:
                details[_DETAILS_ATTR] = f'\x00D:{index}\x00'
            self.by_docname.setdefault(docname, []).append(index)
        for holder in dict.fromkeys(holders):
            self._mark_list(holder, holder_ids[id(holder)])
        return ancestors, lists

    @staticmethod
    def _mark_list(holder: Tag, index: int) -> None:
        """Tag a list holding entries, which Sphinx also marks as current."""
        classes = holder.get('class')
        if classes:
            holder['class'] = [f'\x00U:{index}\x00{classes[0]}', *classes[1:]]
        else:
            holder['class'] = [f'\x00u:{index}\x00']

    @staticmethod
    def _index_of(item: Tag) -> int | None:
        """Return the occurrence index tagged onto a list item, if it has one."""
        for token in item.get('class') or ():
            match = _LIST_ITEM_MARKER.search(token)
            if match:
                return int(match[1])
        return None

    def _compile(self, raw: str, ancestors: list[list[int]], lists: list[int | None]) -> None:
        """Strip the markers back out, recording where each per-page edit belongs."""
        branch: list[list[_Edit]] = [[] for _ in ancestors]
        page: list[list[_Edit]] = [[] for _ in ancestors]
        holders: dict[int, _Edit] = {}
        parts: list[str] = []
        read = written = 0
        for match in _TOKENS.finditer(raw):
            parts.append(raw[read : match.start()])
            written += match.start() - read
            read = match.end()
            item, listed, bare, details, anchor, href_index, href = match.groups()
            if item is not None:
                branch[int(item)].append((written, written, ' current active'))
            elif listed is not None:
                holders[int(listed)] = (written, written, 'current ')
            elif bare is not None:
                holders[int(bare)] = (written, written, ' class="current"')
            elif details is not None:
                branch[int(details)].append((written, written, ' open="open"'))
            elif anchor is not None:
                page[int(anchor)].append((written, written, 'current '))
            else:
                kept = f'href="{href}"'
                parts.append(kept)
                start = written + len('href="')
                page[int(href_index)].append((start, start + len(href), '#'))
                written += len(kept)
        parts.append(raw[read:])
        for index, holder in enumerate(lists):
            if holder is not None:
                branch[index].append(holders[holder])
        self.base = ''.join(parts)
        self.branch_edits = [tuple(edits) for edits in branch]
        self.page_edits = [tuple(edits) for edits in page]
        self.ancestors = [tuple(chain) for chain in ancestors]

    def render(self, pagename: str, prefix: str) -> str:
        """Return the sidebar HTML for one page."""
        edits: list[_Edit] = []
        for index in self.by_docname.get(pagename, ()):
            edits.extend(self.page_edits[index])
            edits.extend(self.branch_edits[index])
            for ancestor in self.ancestors[index]:
                edits.extend(self.branch_edits[ancestor])
        if not edits:
            return self.base.replace(_PREFIX, prefix)
        pieces: list[str] = []
        read = 0
        for start, end, text in sorted(set(edits)):
            pieces.append(self.base[read:start])
            pieces.append(text)
            read = end
        pieces.append(self.base[read:])
        return ''.join(pieces).replace(_PREFIX, prefix)


def _prefix_to_root(app: Sphinx, pagename: str) -> str:
    """Return the relative path from a page back to the root document's directory."""
    root = app.config.root_doc
    root_uri = app.builder.get_target_uri(root)
    relative = app.builder.get_relative_uri(pagename, root)
    if root_uri and relative.endswith(root_uri):
        return relative[: len(relative) - len(root_uri)]
    return ''


class _SidebarCache:
    """Build the expanded sidebar on demand and render it for any page."""

    def __init__(self, app: Sphinx) -> None:
        self._app = app
        self._sidebars: dict[Any, _ExpandedSidebar] = {}

    def renderer(self, pagename: str, original: Any) -> Any:
        """Return a stand-in for the theme's sidebar renderer, bound to one page."""

        def generate_toctree_html(
            kind: str, startdepth: int = 1, show_nav_level: int = 1, **kwargs: Any
        ) -> Any:
            if kind != 'sidebar' or startdepth != 0 or kwargs.get('collapse', True):
                return original(
                    kind, startdepth=startdepth, show_nav_level=show_nav_level, **kwargs
                )
            key = (show_nav_level, tuple(sorted(kwargs.items())))
            sidebar = self._sidebars.get(key)
            if sidebar is None:
                sidebar = self._sidebars[key] = _ExpandedSidebar(self._app, show_nav_level, kwargs)
            return sidebar.render(pagename, _prefix_to_root(self._app, pagename))

        return generate_toctree_html


def _install(app: Sphinx) -> None:
    """Route every template render through the cached sidebar renderer."""
    # The theme renders the sidebar twice per page, so the swap cannot go in the context.
    templates = getattr(app.builder, 'templates', None)
    if templates is None:
        return
    render = templates.render
    cache = _SidebarCache(app)
    # The 404 page needs the site-root absolute links `notfound.extension` builds for it.
    skip = getattr(app.config, 'notfound_pagename', None)

    def render_with_cached_sidebar(template: str, context: dict[str, Any]) -> str:
        original = context.get('generate_toctree_html')
        pagename = context.get('pagename')
        if original is None or pagename is None or pagename == skip:
            return render(template, context)
        renderer = cache.renderer(pagename, original)
        return render(template, {**context, 'generate_toctree_html': renderer})

    templates.render = render_with_cached_sidebar


def setup(app: Sphinx) -> dict[str, Any]:
    """Connect the sidebar reuse handler."""
    app.connect('builder-inited', _install)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
