"""Locations of the data, caches, results and figures.

Paths are resolved from the repository root, so the scripts can be started
from any working directory. QAL_CACHE_DIR, QAL_RESULTS_DIR and QAL_FONT_DIR
override the default locations.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'data'
PAPER_DATA_DIR = REPO_ROOT / 'paper_data'
FIGURES_DIR = REPO_ROOT / 'figures'
CACHE_DIR = Path(os.environ.get('QAL_CACHE_DIR', REPO_ROOT / 'cache'))
RESULTS_DIR = Path(os.environ.get('QAL_RESULTS_DIR', REPO_ROOT / 'results'))
FONT_DIR = Path(os.environ.get('QAL_FONT_DIR', DATA_DIR / 'fonts'))
