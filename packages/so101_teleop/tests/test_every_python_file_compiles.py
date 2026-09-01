"""Every tracked .py file must at least parse.

Three separate syntax errors reached the tree during this migration, all from
automated import insertion landing inside a multi-line parenthesised import, and
none was caught by 260+ passing tests because nothing imported those modules.
Parsing is the cheapest possible guard against that whole class.
"""

import ast
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def tracked_python_files():
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "*.py"],
                         capture_output=True, text=True, check=True).stdout
    return sorted(REPO / line for line in out.split("\n") if line.strip())


FILES = tracked_python_files()


def test_there_are_files_to_check():
    assert len(FILES) > 50, f"only found {len(FILES)} tracked .py files"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_file_parses(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
