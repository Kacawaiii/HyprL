"""Generate the public B2B OpenAPI artifact from its authoritative route registry."""
import argparse
import json
from pathlib import Path

from scripts.trading_lab.b2b.contracts import openapi


def generated():
    return json.dumps(openapi(), sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/artifacts/b2b_openapi_v1.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.output)
    value = generated()
    if args.check:
        if not path.is_file() or path.read_text() != value:
            raise SystemExit("OpenAPI artifact differs from the code")
        print("B2B OpenAPI matches the code")
    else:
        path.write_text(value)


if __name__ == "__main__":
    main()
