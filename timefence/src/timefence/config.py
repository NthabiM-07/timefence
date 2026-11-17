"""Project-wide constants: timeline, paths and load schedule.

All timestamps are South African Standard Time (SAST, UTC+2) stored tz-naive.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("TIMEFENCE_HOME") or Path(__file__).resolve().parents[2])
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FRAMES_DIR = DATA_DIR / "frames"
ONLINE_DB = DATA_DIR / "online" / "online_store.sqlite"
REPORTS_DIR = ROOT / "reports"
REGISTRY_PATH = ROOT / "features" / "credit_features.yaml"

# Simulation horizon: 60 month-ends of history.
HISTORY_START = pd.Timestamp("2021-01-01")
N_MONTHS = 60  # Jan 2021 .. Dec 2025
DATA_END = pd.Timestamp("2025-12-31 23:59:59")

# Credit applications are generated in this window.
APPS_START = pd.Timestamp("2022-01-01")
APPS_END = pd.Timestamp("2024-12-31 23:59:59")

# Model development split (by decision time).
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")  # train: 2022-2023
OOT_START = pd.Timestamp("2024-01-01")  # out-of-time test: 2024

# Default label: 90+ DPD on any account at any of the 12 month-ends starting
# with the decision month. The label matures at the last of those month-ends.
PERFORMANCE_MONTHS = 12
BAD_DPD = 90

# The "naive" strategy builds every feature as at the day the analyst
# extracted the data, which is after the whole performance window.
EXTRACT_TIME = pd.Timestamp("2026-01-05 09:00:00")

# Every warehouse load lands at 02:00. Decisions happen 08:00-19:59, so the
# warehouse's state of knowledge is constant through each decision day.
LOAD_HOUR = 2
