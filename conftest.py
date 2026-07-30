"""
conftest.py
-----------
Ensures the project root is on sys.path for every pytest run, regardless of
how pytest is invoked (bare `pytest`, `python -m pytest`, from a subfolder,
etc.).

Two import styles coexist in tests/:
  - test_emotion_detector.py uses `from src import emotion_detector as ed`,
    which needs the PROJECT ROOT on sys.path (so `src` resolves as a package).
  - test_sentiment.py, test_profile_generator.py, and test_session.py use
    `sys.path.insert(..., ".../src")` and import flat module names instead
    (`from sentiment import ...`), which needs SRC ITSELF on the path -
    already handled by pytest.ini's `pythonpath = src`.

Both styles were already present in the project before this file existed;
pytest.ini's `pythonpath = src` only satisfies the second one. This file
adds the missing piece for the first, so `pytest -v` from the project root
works the same way `python -m pytest -v` already did (which works only
because Python's -m flag happens to prepend the current directory to
sys.path automatically - this file makes that not be a coincidence).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
