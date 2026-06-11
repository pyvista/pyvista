"""HTML representation for PyVista objects in Jupyter notebooks.

Uses pure HTML/CSS with checkbox-driven expand/collapse.  The only
JavaScript is a minimal ``navigator.clipboard.writeText`` call for
copy-to-clipboard buttons.
Works in Jupyter Notebook, JupyterLab, VS Code, Colab, and nbviewer.
"""

from __future__ import annotations

from functools import lru_cache
from html import escape
from importlib.resources import files as _resources_files
from typing import TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from collections.abc import Sequence


@lru_cache(None)
def _load_css() -> str:
    """Load the CSS stylesheet (cached)."""
    return (
        _resources_files('pyvista.core.static').joinpath('style.css').read_text(encoding='utf-8')
    )


@lru_cache(None)
def _load_pyvista_logo() -> str:
    """Load the PyVista brand mark SVG (cached)."""
    return (
        _resources_files('pyvista.core.static')
        .joinpath('pyvista_logo.svg')
        .read_text(encoding='utf-8')
    )


_MESH_TYPE_ICONS: dict[str, str] = {
    'PolyData': 'polydata.svg',
    'UnstructuredGrid': 'unstructured.svg',
    'ImageData': 'imagedata.svg',
    'RectilinearGrid': 'rectilinear.svg',
    'StructuredGrid': 'structured.svg',
    'MultiBlock': 'multiblock.svg',
}


@lru_cache(None)
def _load_mesh_icon(mesh_type: str) -> str | None:
    """Load a mesh-type SVG icon, or ``None`` if unavailable."""
    filename = _MESH_TYPE_ICONS.get(mesh_type)
    if filename is None:
        return None
    icon_dir = _resources_files('pyvista.core.static').joinpath('mesh_icons')
    return icon_dir.joinpath(filename).read_text(encoding='utf-8')


def collapsible_section(
    header: str,
    *,
    inline_details: str = '',
    details: str = '',
    n_items: int | None = None,
    enabled: bool = True,
    collapsed: bool = False,
) -> str:
    """Build an expand/collapse section using a hidden checkbox.

    Parameters
    ----------
    header : str
        Section label (e.g. ``'Point Data:'``).
    inline_details : str
        Content shown when collapsed.
    details : str
        Content shown when expanded.
    n_items : int, optional
        Item count displayed in parentheses. A value of ``0``
        disables the section.
    enabled : bool
        Whether the section is expandable.
    collapsed : bool
        Whether the section starts collapsed.

    Returns
    -------
    str
        HTML fragment for the section.

    """
    data_id = 'section-' + str(uuid.uuid4())

    has_items = n_items is not None and n_items > 0
    n_items_span = '' if n_items is None else f' <span>({n_items})</span>'
    enabled_attr = '' if enabled and has_items else ' disabled'
    collapsed_attr = '' if collapsed or not has_items else ' checked'
    tip = " title='Expand/collapse section'" if enabled_attr == '' else ''

    html = (
        f"<input id='{data_id}' class='pv-section-summary-in'"
        f" type='checkbox'{enabled_attr}{collapsed_attr} />"
        f"<label for='{data_id}' class='pv-section-summary'{tip}>"
        f'{header}{n_items_span}</label>'
        f"<div class='pv-section-inline-details'>{inline_details}</div>"
    )
    if details:
        html += f"<div class='pv-section-details'>{details}</div>"
    return html


def _badge(label: str, css_class: str) -> str:
    """Return a small colored badge ``<span>``."""
    return f" <span class='pv-badge {css_class}'>{escape(label)}</span>"


def _copy_btn(text: str) -> str:
    """Return a small copy-to-clipboard button.

    Uses a ``data-copy`` attribute to hold the text and reads it via
    ``this.dataset.copy`` in the ``onclick`` handler.  This avoids
    XSS risks from inline string interpolation: HTML entities in the
    attribute value are decoded safely by the browser before reaching
    the JavaScript API.
    """
    return (
        f"<button class='pv-copy-btn' title='Copy to clipboard'"
        f" data-copy='{escape(text)}'"
        f''' onclick="navigator.clipboard.writeText(this.dataset.copy)"'''
        f'>\u29c9</button>'
    )


def _summarize_array(
    name: str,
    ncomp: int,
    dtype: str,
    *,
    badges: str = '',
    is_active: bool = False,
    shape: str = '',
    range_str: str = '',
) -> str:
    """Format a single data-array row for the grid layout."""
    safe_name = escape(name)
    dims_str = escape(shape) if shape else f'{ncomp} comp' if ncomp > 1 else 'scalar'
    active_cls = ' pv-var-name-active' if is_active else ''
    copy = _copy_btn(name)

    return (
        f"<div class='pv-var-name{active_cls}'><span>{safe_name}</span>{copy}</div>"
        f"<div class='pv-var-dims'>{dims_str}</div>"
        f"<div class='pv-var-dtype'>{escape(dtype)}</div>"
        f"<div class='pv-var-range'>{escape(range_str)}</div>"
        f"<div class='pv-var-badges'>{badges}</div>"
    )


def _array_badges(
    name: str,
    *,
    active_scalars: str | None = None,
    active_vectors: str | None = None,
    active_normals: str | None = None,
    active_tcoords: str | None = None,
) -> str:
    """Return concatenated badge HTML for an array."""
    badges = ''
    if name == active_scalars:
        badges += _badge('active', 'pv-badge-active')
    if name == active_vectors:
        badges += _badge('vectors', 'pv-badge-vectors')
    if name == active_normals:
        badges += _badge('normals', 'pv-badge-normals')
    if name == active_tcoords:
        badges += _badge('tcoords', 'pv-badge-tcoords')
    return badges


def _data_array_section(
    title: str,
    arrays: list[tuple[str, int, str, str, str]],
    *,
    active_scalars: str | None = None,
    active_vectors: str | None = None,
    active_normals: str | None = None,
    active_tcoords: str | None = None,
    collapsed: bool = False,
) -> str:
    """Build a collapsible section for a set of data arrays.

    Parameters
    ----------
    title : str
        Section title (e.g. ``'Point Data'``).
    arrays : list[tuple[str, int, str, str, str]]
        Each tuple is ``(name, n_components, dtype_str, shape_str,
        range_str)``.  When ``shape_str`` is non-empty it is shown
        instead of the default ``'scalar'`` / ``'N comp'`` label.
        ``range_str`` is displayed as the data range (e.g.
        ``'[-0.5, 0.5]'``).
    active_scalars : str, optional
        Name of the active scalars array.
    active_vectors : str, optional
        Name of the active vectors array.
    active_normals : str, optional
        Name of the active normals array.
    active_tcoords : str, optional
        Name of the active texture coordinates array.
    collapsed : bool
        Whether the section starts collapsed.

    Returns
    -------
    str
        HTML fragment for the section.

    """
    li_parts: list[str] = []
    for name, ncomp, dtype, shape_str, range_str in arrays:
        badges = _array_badges(
            name,
            active_scalars=active_scalars,
            active_vectors=active_vectors,
            active_normals=active_normals,
            active_tcoords=active_tcoords,
        )
        row = _summarize_array(
            name,
            ncomp,
            dtype,
            badges=badges,
            is_active=(name == active_scalars),
            shape=shape_str,
            range_str=range_str,
        )
        li_parts.append(f"<li class='pv-var-item'>{row}</li>")

    items_li = ''.join(li_parts)
    details = f"<ul class='pv-var-list'>{items_li}</ul>"

    # Show active scalar name inline when collapsed
    inline = ''
    if active_scalars and any(name == active_scalars for name, _, _, _, _ in arrays):
        inline = f'{escape(active_scalars)}{_badge("active", "pv-badge-active")}'

    return collapsible_section(
        f'{title}:',
        inline_details=inline,
        details=details,
        n_items=len(arrays),
        collapsed=collapsed,
    )


def _metadata_html(
    rows: list[tuple[str, list[tuple[str, str]], str]],
) -> str:
    """Build always-visible metadata rows.

    Parameters
    ----------
    rows : list[tuple[str, list[tuple[str, str]], str]]
        Each entry is ``(row_label, [(key, value), ...], copy_text)``.
        If ``copy_text`` is non-empty, a clipboard button is shown.

    Returns
    -------
    str
        HTML ``<div>`` fragment.

    """
    row_parts: list[str] = []
    for label, items, copy_text in rows:
        entries = ''.join(
            f"<span class='pv-meta-entry'>"
            f"<span class='pv-meta-label'>{escape(k)}</span> {escape(v)}"
            f'</span>'
            for k, v in items
        )
        copy_html = _copy_btn(copy_text) if copy_text else ''
        row_parts.append(
            f"<div class='pv-meta-row pv-copyable'>"
            f"<span class='pv-meta-row-label'>{escape(label)}</span>"
            f'{copy_html}{entries}'
            f'</div>'
        )
    return f"<div class='pv-metadata'>{''.join(row_parts)}</div>"


def _fmt_memory(kib: int) -> str:
    """Format kibibytes into a human-readable string."""
    if kib < 1024:
        return f'{kib} KiB'
    mib = kib / 1024
    if mib < 1024:
        return f'{mib:.1f} MiB'
    gib = mib / 1024
    return f'{gib:.2f} GiB'


def _children_section(
    title: str,
    children: list[tuple[str, str, str]],
) -> str:
    """Build a collapsible section listing child blocks/partitions.

    Parameters
    ----------
    title : str
        Section title (e.g. ``'Blocks'``).
    children : list[tuple[str, str, str]]
        Each tuple is ``(name, type_name, detail_str)``.

    Returns
    -------
    str
        HTML fragment for the section.

    """
    items_li = ''.join(
        f"<li><span class='pv-child-name'>{escape(name)}</span>"
        f"<span class='pv-child-type'>{escape(ctype)}</span>"
        f"<span class='pv-child-detail'>{escape(detail)}</span></li>"
        for name, ctype, detail in children
    )
    details = f"<ul class='pv-children-list'>{items_li}</ul>"
    return collapsible_section(
        f'{title}:',
        details=details,
        n_items=len(children),
        collapsed=False,
    )


def _sections_html(sections: list[str]) -> str:
    """Wrap sections into a ``<ul>`` CSS grid."""
    items = ''.join(f"<li class='pv-section-item'>{s}</li>" for s in sections)
    return f"<ul class='pv-sections'>{items}</ul>"


def build_repr_html(
    obj_type: str,
    mesh_type: str,
    *,
    header_badges: Sequence[str] = (),
    metadata: Sequence[tuple[str, list[tuple[str, str]], str]] = (),
    sections: Sequence[str] = (),
    text_repr: str = '',
) -> str:
    """Assemble a complete HTML repr with header, metadata, sections, and fallback.

    Parameters
    ----------
    obj_type : str
        Display name (e.g. ``'PolyData'``).
    mesh_type : str
        Key for mesh icon lookup (e.g. ``'PolyData'``).
    header_badges : Sequence[str]
        Labels shown as small badges in the header (e.g. memory size).
    metadata : Sequence[tuple[str, list[tuple[str, str]], str]]
        Always-visible key-value rows rendered between the header and
        collapsible sections.  Each entry is
        ``(row_label, [(key, val), ...], copy_text)``.
    sections : Sequence[str]
        List of collapsible-section HTML fragments.
    text_repr : str
        Plain-text fallback for non-HTML environments.

    Returns
    -------
    str
        Complete HTML string.

    """
    css = _load_css()

    # Mesh icon
    icon_svg = _load_mesh_icon(mesh_type) or _load_mesh_icon('ImageData') or ''
    icon_html = f"<span class='pv-logo'>{icon_svg}</span>"

    # PyVista brand logo (top-right)
    pv_logo = _load_pyvista_logo()
    pv_logo_html = f"<span class='pv-brand-logo'>{pv_logo}</span>"

    # Header badges
    badges_html = ''.join(
        f" <span class='pv-header-badge'>{escape(b)}</span>" for b in header_badges
    )

    header = (
        "<div class='pv-header'>"
        f'{icon_html}'
        "<div class='pv-header-text'>"
        f"<div class='pv-obj-type'>{escape(obj_type)}{badges_html}</div>"
        '</div>'
        f'{pv_logo_html}'
        '</div>'
    )

    meta_html = _metadata_html(list(metadata)) if metadata else ''
    text_fallback = escape(text_repr)

    return (
        '<div>'
        f'<style>{css}</style>'
        f"<pre class='pv-text-repr-fallback'>{text_fallback}</pre>"
        "<div class='pv-wrap' style='display:none'>"
        f'{header}'
        f'{meta_html}'
        f'{_sections_html(list(sections))}'
        '</div>'
        '</div>'
    )
