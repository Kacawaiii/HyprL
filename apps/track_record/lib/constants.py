from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
TRACK_RECORD_DIR = ROOT_DIR / "docs/reports/track_record"
REPORT_JSON_NAME = "track_record_latest.json"
REPORT_MD_NAME = "TRACK_RECORD_latest.md"
DISCLAIMER_PATH = ROOT_DIR / "docs/legal/DISCLAIMER.md"

# UI / Design
THEME_COLOR_BG = "#0b0f19"
THEME_COLOR_ACCENT_1 = "#7C3AED"  # Violet
THEME_COLOR_ACCENT_2 = "#22D3EE"  # Cyan
THEME_COLOR_ACCENT_3 = "#F472B6"  # Pink
THEME_COLOR_ACCENT_4 = "#34D399"  # Green

# Assets
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
STYLE_CSS_PATH = ASSETS_DIR / "styles.css"
THEME_JS_PATH = ASSETS_DIR / "theme.js"
BG_JS_PATH = ASSETS_DIR / "bg.js"
