"""Make `tools/` importable.

The PV sender and calibration utilities are standalone programs, not library
modules, so they live in `tools/` rather than under `src/`. Their tests import
them by bare name, the way they are run.
"""

import sys
from pathlib import Path

TOOLS = Path(__file__).parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
