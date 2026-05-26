"""Project path helpers."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / "core"
CORE_SRC_DIR = CORE_DIR / "src"

EXTERNAL_DIR = PROJECT_ROOT / "external"
FACE_REID_DIR = EXTERNAL_DIR / "face-reidentification"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DATA_DIR = PROJECT_ROOT / "data"
CAPTURES_DIR = DATA_DIR / "captures"
CONFIG_DIR = PROJECT_ROOT / "app"
CONFIG_FILE = CONFIG_DIR / "config.json"


def get_face_reid_path() -> Path:
    if not FACE_REID_DIR.exists():
        logger.error("Face-reid dir not found: %s", FACE_REID_DIR)
        raise RuntimeError(f"Missing face-reidentification module at {FACE_REID_DIR}")
    return FACE_REID_DIR


def get_weight_path(weight_file: str) -> Path:
    weight_path = WEIGHTS_DIR / weight_file
    if not weight_path.exists():
        logger.warning("Weight file not found: %s", weight_path)
    return weight_path


def ensure_capture_dir() -> Path:
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    return CAPTURES_DIR


def get_config_path() -> Path:
    return CONFIG_FILE


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
