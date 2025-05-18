"""Sphinx role for linking to VTK documentation."""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup  # pydata-sphinx-theme dependency
from docutils import nodes
import requests
from sphinx.roles import ReferenceRole

if TYPE_CHECKING:
    from typing import ClassVar


class VTKRole(ReferenceRole):
    """Link to vtk class documentation using a custom role.

    E.g. use :vtk:`vtkPolyData` for linking to the `vtkPolyData` class docs.
    """

    base_url = 'https://vtk.org/doc/nightly/html/'
    class_url_template = base_url + 'class{cls}.html'
    validated_urls: ClassVar[dict[str, bool]] = {}
    resolved_anchors: ClassVar[dict[tuple[str, str], str]] = {}

    def run(self):
        """Run the :vtk: role."""
        cls_full = self.target
        title = self.title or cls_full

        # Split member if present
        if '.' in cls_full:
            cls_name, member_name = cls_full.split('.', 1)
        else:
            cls_name, member_name = cls_full, None

        class_url = self.class_url_template.format(cls=cls_name)
        url = class_url

        # Validate URL
        is_valid = self.validated_urls.get(class_url)
        html = None
        if is_valid is None:
            try:
                response = requests.get(class_url, timeout=3)
                is_valid = response.status_code == HTTPStatus.OK
                html = response.text if is_valid else html
            except requests.exceptions.RequestException:
                is_valid = False
            self.validated_urls[class_url] = is_valid

        if not is_valid:
            msg = f"Invalid VTK class reference: '{cls_name}' → {class_url}"
            self.inliner.reporter.warning(msg, line=self.lineno, subtype='ref')

        # If method specified, find anchor
        if member_name and is_valid:
            cache_key = (cls_name, member_name)
            anchor = self.resolved_anchors.get(cache_key)
            if anchor is None and html:
                anchor = find_member_anchor(html, member_name)
                if anchor:
                    self.resolved_anchors[cache_key] = anchor

            if anchor:
                url += f'#{anchor}'

        # Emit link
        node = nodes.reference(title, title, refuri=url)
        return [node], []


def find_member_anchor(html: str, member_name: str) -> str | None:
    """Try to find the anchor ID for a method/attribute in the HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    headers = soup.find_all(['h2', 'h3'], class_='memtitle')
    for header in headers:
        if member_name in header.get_text():
            anchor = header.find_previous('a', id=True)
            if anchor:
                return anchor['id']
    return None
