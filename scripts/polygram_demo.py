"""Phase 3: econ-sae -> polygram Dictionary + cancellation runs.

Requires the polygram extra:
    pip install -e ".[polygram]"

Usage:
    python scripts/polygram_demo.py
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from econsae.polygram_bridge import main

if __name__ == "__main__":
    main()
