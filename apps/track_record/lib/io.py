import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from apps.track_record.lib.constants import TRACK_RECORD_DIR, REPORT_JSON_NAME, REPORT_MD_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_track_record_json(root_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Loads the main track record JSON file."""
    path = (root_dir or TRACK_RECORD_DIR) / REPORT_JSON_NAME
    if not path.exists():
        logger.warning(f"Track record JSON not found at {path}")
        return None
    
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON at {path}: {e}")
        return None

def load_track_record_md(root_dir: Optional[Path] = None) -> Optional[str]:
    """Loads the markdown report content."""
    path = (root_dir or TRACK_RECORD_DIR) / REPORT_MD_NAME
    if not path.exists():
        logger.warning(f"Track record Markdown not found at {path}")
        return None
    
    return path.read_text(encoding="utf-8")

def check_artifacts_exist(root_dir: Optional[Path] = None) -> bool:
    """Checks if essential artifacts are present."""
    base = root_dir or TRACK_RECORD_DIR
    return (base / REPORT_JSON_NAME).exists()
