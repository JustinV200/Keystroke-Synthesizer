"""Central configuration for KeyForge.

All paths, model hyperparameters, and UI defaults live here so they can be
tweaked in one place instead of being scattered across modules.
"""
import os

from pynput.keyboard import Key


# --- Paths ---------------------------------------------------------------
KEYFORGE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(KEYFORGE_DIR, "Model")
OUTPUT_DIR = os.path.join(KEYFORGE_DIR, "output")

CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_model.pt")
STATS_PATH = os.path.join(MODEL_DIR, "cont_stats.json")


# --- Model ---------------------------------------------------------------
BASE_MODEL = "microsoft/deberta-v3-base"
NUM_CONTINUOUS = 3
MAX_TOKEN_LENGTH = 512


# --- Output / physical bounds (match training preprocessing) -------------
# (min, max) clamp for each continuous feature in output order.
FEATURE_BOUNDS = {
    "DwellTime":    (0.0, 300.0),
    "FlightTime":   (0.0, 900.0),
    "typing_speed": (0.0, 490.0),
}

FEATURE_COLUMNS = ["char", "prev_char", "DwellTime", "FlightTime", "typing_speed"]


# --- UI / replay ---------------------------------------------------------
WINDOW_TITLE = "KeyForge"
WINDOW_SIZE = "420x460"

# Average typist speed; replay is scaled relative to this baseline.
BASELINE_WPM = 40.0
WPM_MIN = 20
WPM_MAX = 120
COUNTDOWN_SECONDS = 3

# Characters that need a pynput Key enum instead of a raw string.
SPECIAL_KEYS = {
    " ": Key.space,
    "\n": Key.enter,
    "\t": Key.tab,
}
