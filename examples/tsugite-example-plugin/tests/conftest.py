"""Make the example plugin importable without installing it.

Unlike the first-party plugins under ``plugins/`` (uv workspace members), this
example is a standalone copy-paste reference that is not installed into the dev
environment, so its package dir is added to ``sys.path`` for the test run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
