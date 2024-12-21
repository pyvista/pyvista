"""Measure document reading durations.

This is a modified version of `sphinx.ext.duration`.
It has been modified to show more entries (default is 5, which is too few).

License for Sphinx
==================

Unless otherwise indicated, all code in the Sphinx project is licenced under the
two clause BSD licence below.

Copyright (c) 2007-2024 by the Sphinx team (see AUTHORS file).
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
# pragma: no cover

from __future__ import annotations

from itertools import islice
from operator import itemgetter
import time
from typing import TYPE_CHECKING

from sphinx.domains import Domain
from sphinx.locale import __
from sphinx.util import logging

if TYPE_CHECKING:
    from typing import TypedDict

    from docutils import nodes
    from sphinx.application import Sphinx

    class _DurationDomainData(TypedDict):
        reading_durations: dict[str, float]


N_DURATIONS = 100  # Set how many entries to show duration for
logger = logging.getLogger(__name__)


class DurationDomain(Domain):
    """A domain for durations of Sphinx processing."""

    name = 'duration'

    @property
    def reading_durations(self) -> dict[str, float]:
        return self.data.setdefault('reading_durations', {})

    def note_reading_duration(self, duration: float) -> None:
        self.reading_durations[self.env.docname] = duration

    def clear(self) -> None:
        self.reading_durations.clear()

    def clear_doc(self, docname: str) -> None:
        self.reading_durations.pop(docname, None)

    def merge_domaindata(self, docnames: set[str], otherdata: _DurationDomainData) -> None:  # type: ignore[override]
        other_reading_durations = otherdata.get('reading_durations', {})
        docnames_set = frozenset(docnames)
        for docname, duration in other_reading_durations.items():
            if docname in docnames_set:
                self.reading_durations[docname] = duration


def on_builder_inited(app: Sphinx) -> None:
    """Initialize DurationDomain on bootstrap.

    This clears the results of the last build.
    """
    domain = app.env.domains['duration']
    domain.clear()


def on_source_read(app: Sphinx, docname: str, content: list[str]) -> None:
    """Start to measure reading duration."""
    app.env.temp_data['started_at'] = time.monotonic()


def on_doctree_read(app: Sphinx, doctree: nodes.document) -> None:
    """Record a reading duration."""
    started_at = app.env.temp_data['started_at']
    duration = time.monotonic() - started_at
    domain = app.env.domains['duration']
    domain.note_reading_duration(duration)


def on_build_finished(app: Sphinx, error: Exception) -> None:
    """Display duration ranking on the current build."""
    domain = app.env.domains['duration']
    if not domain.reading_durations:
        return
    durations = sorted(domain.reading_durations.items(), key=itemgetter(1), reverse=True)

    logger.info('')
    logger.info(
        __(
            f'====================== slowest {N_DURATIONS} reading durations (seconds) ======================='
        )
    )
    for docname, d in islice(durations, N_DURATIONS):
        logger.info(f'{d:.3f} {docname}')


def setup(app):
    app.add_domain(DurationDomain)
    app.connect('builder-inited', on_builder_inited)
    app.connect('source-read', on_source_read)
    app.connect('doctree-read', on_doctree_read)
    app.connect('build-finished', on_build_finished)
