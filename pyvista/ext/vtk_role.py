from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup
from docutils import nodes
import requests
from sphinx.util.docutils import ReferenceRole

if TYPE_CHECKING:
    from typing import ClassVar


class VTKRole(ReferenceRole):
    """Link to vtk class documentation using a custom role.

    E.g. use :vtk:`vtkPolyData` for linking to the `vtkPolyData` class docs.
    """

    # Cache for (class, member) keys with urls as values
    resolved_urls: ClassVar[dict[tuple[str, str | None], str]] = {}

    def run(self):  # numpydoc ignore=RT01
        """Run the :vtk: role."""
        INVALID_URL = ''  # URL is set to empty string if not valid
        cls_full = self.target
        title = self.title or cls_full

        # Split member if present
        if '.' in cls_full:
            cls_name, member_name = cls_full.split('.', 1)
        else:
            cls_name, member_name = cls_full, None
        cls_url = _vtk_class_url(cls_name)

        cache_key = (cls_name, member_name)
        cached_url = self.resolved_urls.get(cache_key)
        if cached_url is not None:
            # Cache hit, check if valid or not
            if cached_url == INVALID_URL:
                # Not valid, report the error source
                has_valid_class_url = self.resolved_urls.get((cls_name, None))
                if member_name and has_valid_class_url:
                    # Class is valid but member is not
                    self._warn_invalid_class_member_ref(cls_name, member_name)
                else:
                    self._warn_invalid_class_ref(cls_name)

                # Use class URL fallback for invalid member anchor
                refuri = cls_url
            else:
                # Cached url is valid
                refuri = cached_url

            node = nodes.reference(title, title, refuri=refuri)
            return [node], []

        # Not cached, build URL and validate
        try:
            response = requests.get(cls_url, timeout=3)
            if response.status_code != HTTPStatus.OK:
                msg = f'Status code {response.status_code}'
                raise requests.RequestException(msg)
            html = response.text
        except requests.RequestException:
            # Invalid class url
            self._warn_invalid_class_ref(cls_name)

            # Create cache entries
            self.resolved_urls[cache_key] = INVALID_URL
            if member_name:
                self.resolved_urls[(cls_name, None)] = INVALID_URL

            # We return the reference even though the URL is bad
            node = nodes.reference(title, title, refuri=cls_url)
            return [node], []

        if member_name:
            anchor = _find_member_anchor(html, member_name)
            if anchor:
                full_url = f'{cls_url}#{anchor}'
                self.resolved_urls[cache_key] = full_url
                node = nodes.reference(title, title, refuri=full_url)
                return [node], []
            else:
                # Anchor not found, mark cache as invalid but still fallback to class URL
                self.resolved_urls[cache_key] = INVALID_URL
                self._warn_invalid_class_member_ref(cls_name, member_name)

                node = nodes.reference(title, title, refuri=cls_url)
                return [node], []

        # No member, just class URL
        self.resolved_urls[cache_key] = cls_url
        node = nodes.reference(title, title, refuri=cls_url)
        return [node], []

    def _warn_invalid_class_ref(self, cls_name):
        msg = f"Invalid VTK class reference: '{cls_name}' → {_vtk_class_url(cls_name)}"
        self.inliner.reporter.warning(msg, line=self.lineno, subtype='ref')

    def _warn_invalid_class_member_ref(self, cls_name, member_name):
        msg = f"VTK method anchor not found for: '{cls_name}.{member_name}' → {_vtk_class_url(cls_name)}#<anchor>, the class URL is used instead."
        self.inliner.reporter.warning(msg, line=self.lineno, subtype='ref')


def _vtk_class_url(cls_name):
    """Return the URL to the documentation for a VTK class."""
    return f'https://vtk.org/doc/nightly/html/class{cls_name}.html'


def _find_member_anchor(html: str, member_name: str) -> str | None:
    """Try to find the anchor ID for a method/attribute in the HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    headers = soup.find_all(['h2', 'h3'], class_='memtitle')
    for header in headers:
        if member_name in header.get_text():
            anchor = header.find_previous('a', id=True)
            if anchor:
                return anchor['id']
    return None


def setup(app):
    app.add_role('vtk', VTKRole())
