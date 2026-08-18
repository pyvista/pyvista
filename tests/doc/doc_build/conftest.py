"""Shared paths for tests that inspect the built documentation."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = str(Path(__file__).parent.parent.parent.parent)
BUILD_DIR = str(Path(ROOT_DIR) / 'doc' / '_build')
HTML_DIR = str(Path(BUILD_DIR) / 'html')
