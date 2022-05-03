"""System module."""
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
processed_path: str = os.path.join(ROOT_DIR, 'data/processed')
raw_path: str = os.path.join(ROOT_DIR, 'data/raw')
