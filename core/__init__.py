"""
Core module for raspi_rollcall.
Exports hardware and ML models with proper error handling.
"""

import logging

logger = logging.getLogger(__name__)

# Import and export all core components
try:
    from .src.model import FaceModel, FuzzyModel
except ImportError as e:
    logger.error(f"Failed to import models: {e}")
    raise

try:
    from .src.AS608 import AS608_HAL
except ImportError as e:
    logger.error(f"Failed to import AS608_HAL: {e}")
    raise

try:
    from .src.HW_201 import HW_201_HAL
except ImportError as e:
    logger.error(f"Failed to import HW_201_HAL: {e}")
    raise

try:
    from .src.camera import PiCamera
except ImportError as e:
    logger.error(f"Failed to import PiCamera: {e}")
    raise

try:
    from .src.database import Qdrant_db
except ImportError as e:
    logger.error(f"Failed to import Qdrant_db: {e}")
    raise

# Path utilities
from .paths import (
    PROJECT_ROOT,
    CORE_DIR,
    WEIGHTS_DIR,
    DATA_DIR,
    CAPTURES_DIR,
    get_weight_path,
    ensure_capture_dir,
    get_config_path,
)

__all__ = [
    # Models
    "FaceModel",
    "FuzzyModel",
    # Hardware
    "AS608_HAL",
    "HW_201_HAL",
    "PiCamera",
    # Database
    "Qdrant_db",
    # Path utilities
    "PROJECT_ROOT",
    "CORE_DIR",
    "WEIGHTS_DIR",
    "DATA_DIR",
    "CAPTURES_DIR",
    "get_weight_path",
    "ensure_capture_dir",
    "get_config_path",
]
