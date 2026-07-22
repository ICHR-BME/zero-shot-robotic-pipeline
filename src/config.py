"""Global configuration for the FrED vision and robotic control system."""

from __future__ import annotations

import logging
import os

# Must be configured before importing PyTorch in vision.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ══════════════════════════════════════════════════════════════════════════════
# FILES AND NETWORK
# ══════════════════════════════════════════════════════════════════════════════
ROBOT_IP = "192.168.0.184"
ROBOT2_IP = "192.168.0.181"
CALIB_MATRIX_FILE = "matriz_vision_xarm.npy"
DB_FILE = "production_log.db"

GROUNDING_DINO_MODEL_NAME = "IDEA-Research/grounding-dino-tiny"
SAM_CHECKPOINT = "models/sam_vit_b_01ec64.pth"
DINO_MODEL_NAME = "facebook/dinov2-small"

# ══════════════════════════════════════════════════════════════════════════════
# ROBOT 1 — PICKER
# ══════════════════════════════════════════════════════════════════════════════
HANDOFF_X, HANDOFF_Y, HANDOFF_Z = 260.2, -300.4, 185.6
HANDOFF_ROLL, HANDOFF_PITCH, HANDOFF_YAW = -178.4, -2.9, -0.2

Z_HOME_SEGURO = 220.0
Z_OFFSET_AGARRE = 0.0
Z_GRASP_MIN = 130.0
X_RETRACT, Y_RETRACT = 180.0, 0.0

# ══════════════════════════════════════════════════════════════════════════════
# ROBOT 2 — SORTER
# All coordinates are expressed in Robot 2's base frame.
# ══════════════════════════════════════════════════════════════════════════════
R2_Z_SAFE = 282.2
R2_X_RETRACT = 219.6
R2_Y_RETRACT = 0.1

R2_HANDOFF_X = 257.5
R2_HANDOFF_Y = 339.1
R2_HANDOFF_Z = 173.1
R2_HANDOFF_ROLL = -179.8
R2_HANDOFF_PITCH = 0.0
R2_HANDOFF_YAW = 0.1

R2_APPROVED_X = 209.5
R2_APPROVED_Y = -62.9
R2_APPROVED_Z = 94.0
R2_APPROVED_ROLL = -179.8
R2_APPROVED_PITCH = 0.0
R2_APPROVED_YAW = 0.1

R2_REJECTED_X = 378.0
R2_REJECTED_Y = -71.6
R2_REJECTED_Z = 88.5
R2_REJECTED_ROLL = -179.8
R2_REJECTED_PITCH = 0.0
R2_REJECTED_YAW = 0.1

# ══════════════════════════════════════════════════════════════════════════════
# CAMERAS AND VISION
# ══════════════════════════════════════════════════════════════════════════════
CAM_TOP_IDX = 0
CAM_SIDE_IDX = 1
CAM_W, CAM_H = 640, 460

DINO_DIM = 384
ROI_SIZE = 224
VISION_TEXT_PROMPT = "3D printed part . plastic object"
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
SHOW_SIDE_DEBUG = True

DEFAULT_IDENTITY_TOLERANCE = 1000
DEFAULT_QUALITY_TOLERANCE = 750

# ══════════════════════════════════════════════════════════════════════════════
# GUI DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════
C_BG = "#0D0F14"
C_SURFACE = "#13161E"
C_SURFACE2 = "#1A1E2A"
C_BORDER = "#252A38"
C_ACCENT = "#00C6FF"
C_ACCENT2 = "#7B5EFF"
C_GREEN = "#00E676"
C_RED = "#FF3D57"
C_YELLOW = "#FFD740"
C_GRAY = "#4A5068"
C_TEXT = "#E8EAF0"

FONT_TITLE = ("Consolas", 13, "bold")
FONT_MONO = ("Consolas", 11)
FONT_SMALL = ("Consolas", 10)
FONT_BIG = ("Consolas", 22, "bold")
FONT_MED = ("Consolas", 14, "bold")
