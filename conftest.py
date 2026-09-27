"""
conftest.py
-----------
Loaded automatically by pytest before any test runs. Adds the project root
to the import path, so tests that use `from src import ...` work however
pytest is started. Tests that import modules directly (`from sentiment
import ...`) are covered by `pythonpath = src` in pytest.ini.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
