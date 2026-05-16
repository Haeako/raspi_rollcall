"""
Centralized path management for raspi_rollcall project.
Resolves all paths relative to project root for consistency.
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Project root: one level up from core/
PROJECT_ROOT = Path(__file__).parent.parent

# Core directories
CORE_DIR = PROJECT_ROOT / "core"
CORE_SRC_DIR = CORE_DIR / "src"

# External dependencies
EXTERNAL_DIR = PROJECT_ROOT / "external"
FACE_REID_DIR = EXTERNAL_DIR / "face-reidentification"

# Weights
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Data
DATA_DIR = PROJECT_ROOT / "data"
CAPTURES_DIR = DATA_DIR / "captures"

# Config
CONFIG_DIR = PROJECT_ROOT / "app"
CONFIG_FILE = CONFIG_DIR / "config.json"


def get_face_reid_path() -> Path:
    """Get face-reidentification module path."""
    if not FACE_REID_DIR.exists():
        logger.error(f"Face-reid dir not found: {FACE_REID_DIR}")
        raise RuntimeError(f"Missing face-reidentification module at {FACE_REID_DIR}")
    return FACE_REID_DIR


def get_weight_path(weight_file: str) -> Path:
    """Get absolute path for weight file."""
    weight_path = WEIGHTS_DIR / weight_file
    if not weight_path.exists():
        logger.warning(f"Weight file not found: {weight_path}")
    return weight_path


def ensure_capture_dir() -> Path:
    """Ensure captures directory exists."""
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    return CAPTURES_DIR


def get_config_path() -> Path:
    """Get config file path."""
    return CONFIG_FILE


# Export frequently used paths
__all__ = [
    "PROJECT_ROOT",
    "CORE_DIR",
    "CORE_SRC_DIR",
    "EXTERNAL_DIR",
    "FACE_REID_DIR",
    "WEIGHTS_DIR",
    "DATA_DIR",
    "CAPTURES_DIR",
    "CONFIG_FILE",
    "CONFIG_DIR",
    "get_face_reid_path",
    "get_weight_path",
    "ensure_capture_dir",
    "get_config_path",
]
