"""Prepare licensed IBM Plex TTFs from local site assets, without networking."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def prepare_fonts(source_dir, output_dir):
    from fontTools.ttLib import TTFont

    source_dir, output_dir = Path(source_dir), Path(output_dir)
    license_file = source_dir / "LICENSE-OFL.txt"
    license_text = license_file.read_text()
    if "SIL OPEN FONT LICENSE Version 1.1" not in license_text or "IBM" not in license_text:
        raise ValueError("IBM SIL OFL 1.1 licence required")
    if output_dir.exists():
        raise ValueError("font output already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="plex-", dir=output_dir.parent) as scratch:
        stage = Path(scratch) / "fonts"
        stage.mkdir(mode=0o700)
        records = []
        for weight, style in (("400", "Regular"), ("600", "SemiBold")):
            path = source_dir / f"ibm-plex-sans-latin-{weight}-normal.woff2"
            with TTFont(path, recalcTimestamp=False) as face:
                family = face["name"].getDebugName(1)
                if "IBM Plex Sans" not in family:
                    raise ValueError("IBM Plex Sans font required")
                face.flavor = None
                out = stage / f"IBMPlexSans-{style}.ttf"
                face.save(out)
            records.append(
                {
                    "file": out.name,
                    "family": family,
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "ttf_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                }
            )
        shutil.copyfile(license_file, stage / license_file.name)
        record = {
            "license": "SIL OFL 1.1",
            "license_sha256": hashlib.sha256(license_file.read_bytes()).hexdigest(),
            "conversion": "WOFF2 to TTF; original font tables preserved",
            "fonts": records,
            "audio": "none",
            "music": "none",
            "tts": "none",
        }
        (stage / "license-record.json").write_text(json.dumps(record, indent=2) + "\n")
        stage.rename(output_dir)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_fonts(args.source_dir, args.output_dir)
    print("IBM Plex Sans prepared; licence and font digests recorded.")


if __name__ == "__main__":
    main()
