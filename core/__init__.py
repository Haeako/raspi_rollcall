"""Core hardware, model, database, and path exports."""

from .paths import (
    CAPTURES_DIR,
    CORE_DIR,
    DATA_DIR,
    PROJECT_ROOT,
    WEIGHTS_DIR,
    ensure_capture_dir,
    get_config_path,
    get_weight_path,
)
from .src.AS608 import AS608_HAL
from .src.HW_201 import HW_201_HAL
from .src.camera import PiCamera
from .src.database import Qdrant_db
from .src.model import FaceModel, FuzzyModel

__all__ = [
    "AS608_HAL",
    "CAPTURES_DIR",
    "CORE_DIR",
    "DATA_DIR",
    "FaceModel",
    "FuzzyModel",
    "HW_201_HAL",
    "PROJECT_ROOT",
    "PiCamera",
    "Qdrant_db",
    "WEIGHTS_DIR",
    "ensure_capture_dir",
    "get_config_path",
    "get_weight_path",
]
