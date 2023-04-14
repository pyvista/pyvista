"""Supporting functions for documentation build.

Redirection code sourced and modified from:
https://gitlab.com/documatt/sphinx-reredirects/

License follows:

BSD 3-Clause License

Copyright (c) 2020, documatt
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
from fnmatch import fnmatch
import inspect
import os
import os.path as op
from pathlib import Path
import re
from string import Template
import sys
from typing import Dict, Mapping, Sequence

from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx.util.osutil import SEP

logger = logging.getLogger(__name__)

OPTION_REDIRECTS = "redirects"
OPTION_REDIRECTS_DEFAULT: Dict[str, str] = {}

OPTION_TEMPLATE_FILE = "redirect_html_template_file"
OPTION_TEMPLATE_FILE_DEFAULT = None

REDIRECT_FILE_DEFAULT_TEMPLATE = (
    '<html><head><meta http-equiv="refresh" content="0; url=${to_uri}"></head></html>'  # noqa: E501
)
wildcard_pattern = re.compile(r"[\*\?\[\]]")


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to a Python object.

    Parameters
    ----------
    domain : str
        Only useful when 'py'.

    info : dict
        With keys "module" and "fullname".

    Returns
    -------
    url : str
        The code URL.

    Notes
    -----
    This has been adapted to deal with our "verbose" decorator.

    Adapted from mne (mne/utils/docs.py), which was adapted from SciPy (doc/source/conf.py).
    """
    import pyvista

    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # deal with our decorators properly
    while hasattr(obj, 'fget'):
        obj = obj.fget

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:  # pragma: no cover
        fn = None

    if not fn:  # pragma: no cover
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            return None
        return None

    fn = op.relpath(fn, start=op.dirname(pyvista.__file__))
    fn = '/'.join(op.normpath(fn).split(os.sep))  # in case on Windows

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:  # pragma: no cover
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    if 'dev' in pyvista.__version__:
        kind = 'main'
    else:  # pragma: no cover
        kind = 'release/%s' % ('.'.join(pyvista.__version__.split('.')[:2]))

    return f"http://github.com/pyvista/pyvista/blob/{kind}/pyvista/{fn}{linespec}"


class Reredirects:
    """Generate redirects."""

    def __init__(self, app: Sphinx) -> None:
        self.app = app
        self.redirects_option: Dict[str, str] = getattr(app.config, OPTION_REDIRECTS)

    def grab_redirects(self) -> Mapping[str, str]:
        """Inspect redirects returns dict mapping docname to target."""
        # docname-target dict
        to_be_redirected = {}

        # For each source-target redirect pair in conf.py
        for source, target in self.redirects_option.items():
            # no wildcard, append source as-is
            if not self._contains_wildcard(source):
                to_be_redirected[source] = target
                continue

            assert self.app.env

            # wildcarded source, expand to docnames
            expanded_docs = [doc for doc in self.app.env.found_docs if fnmatch(doc, source)]

            if not expanded_docs:
                logger.warning(f"No documents match to '{source}' redirect.")
                continue

            for doc in expanded_docs:
                new_target = self._apply_placeholders(doc, target)
                to_be_redirected[doc] = new_target

        return to_be_redirected

    def docname_out_path(self, docname: str, suffix: str) -> Sequence[str]:
        """Return path to outfile that would be created by the used builder.

        This is for a Sphinx docname (the path to a source document without
        suffix).
        """
        # Return as-is, if the docname already has been passed with a suffix
        if docname.endswith(suffix):
            return [docname]

        # Remove any trailing slashes, except for "/"" index
        if len(docname) > 1 and docname.endswith(SEP):
            docname = docname.rstrip(SEP)

        # Figure out whether we have dirhtml builder
        out_uri = self.app.builder.get_target_uri(docname=docname)  # type: ignore

        if not out_uri.endswith(suffix):
            # If dirhtml builder is used, need to append "index"
            return [out_uri, "index"]

        # Otherwise, convert e.g. 'source' to 'source.html'
        return [out_uri]

    def create_redirects(self, to_be_redirected: Mapping[str, str]) -> None:
        """Create actual redirect file for each pair in passed mapping of docnames to targets."""
        # Corresponds to value of `html_file_suffix`, but takes into account
        # modifications done by the builder class
        try:
            suffix = self.app.builder.out_suffix  # type: ignore
        except Exception:
            suffix = ".html"

        for docname, target in to_be_redirected.items():
            out = self.docname_out_path(docname, suffix)
            redirect_file_abs = Path(self.app.outdir).joinpath(*out).with_suffix(suffix)

            redirect_file_rel = redirect_file_abs.relative_to(self.app.outdir)

            if redirect_file_abs.exists():
                logger.info(f"Overwriting '{redirect_file_rel}' with redirect to '{target}'.")
            else:
                logger.info(f"Creating redirect '{redirect_file_rel}' to '{target}'.")

            self._create_redirect_file(redirect_file_abs, target)

    @staticmethod
    def _contains_wildcard(text: str) -> bool:
        """Tells whether passed argument contains wildcard characters."""
        return bool(wildcard_pattern.search(text))

    @staticmethod
    def _apply_placeholders(source: str, target: str) -> str:
        """Expand "source" placeholder in target and return it."""
        return Template(target).substitute({"source": source})

    def _create_redirect_file(self, at_path: Path, to_uri: str) -> None:
        """Create a redirect file according to redirect template."""
        content = self._render_redirect_template(to_uri)

        # create any missing parent folders
        at_path.parent.mkdir(parents=True, exist_ok=True)

        at_path.write_text(content)

    def _render_redirect_template(self, to_uri: str) -> str:
        # HTML used as redirect file content
        redirect_template = REDIRECT_FILE_DEFAULT_TEMPLATE
        content = Template(redirect_template).substitute({"to_uri": to_uri})

        return content


def make_redirects(app: Sphinx, exception):
    """Make redirects using sphinx."""
    rr = Reredirects(app)
    to_be_redirected = rr.grab_redirects()
    rr.create_redirects(to_be_redirected)
