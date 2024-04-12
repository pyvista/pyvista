import os
from pathlib import Path

import pytest

from doc.source.make_tables import make_all_tables

ROOT = Path(__file__).parent.parent.parent
DOC_SOURCE = ROOT / 'doc' / 'source'


@pytest.fixture()
def _docs_dir():
    cwd = Path.cwd()
    os.chdir(DOC_SOURCE)
    yield
    os.chdir(cwd)


@pytest.mark.usefixtures(_docs_dir)
def test_make_all_tables():
    make_all_tables()
