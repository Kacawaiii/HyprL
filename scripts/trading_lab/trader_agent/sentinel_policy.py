"""Narrow PM expectations and an offline patch for the operator-owned sentinel."""
import argparse
import ast
import hashlib
from pathlib import Path

from .config import TraderError, instant
from .pm_retirement_spec import PM_SERVICES, PM_TIMERS, active, spec


def expected_unit(unit, at):
    return not (active(at) and unit in PM_SERVICES | PM_TIMERS)


def expected_alert(alert, at):
    if alert.get('code') not in {'MISSING_PM_RUN', 'FAILED_PM_RUN'} or not active(at):
        return True
    try:
        return not active(instant(alert['at']))
    except (KeyError, TypeError, ValueError, AttributeError):
        # A corrupt/missing timestamp must never hide an operational failure.
        return True


def patch_sentinel(source, repo, expected_sha256):
    """Stage only; refuse an unreviewed script and require each exact anchor once."""
    spec()
    if hashlib.sha256(source.encode()).hexdigest() != expected_sha256:
        raise TraderError('SENTINEL_SOURCE_CHANGED')
    replacements = {
        'issues = []\n': 'import sys\nsys.path.insert(0, ' + repr(str(Path(repo).resolve())) + ')\n'
            'from scripts.trading_lab.trader_agent.sentinel_policy import expected_unit, expected_alert\n'
            'issues = []\n',
        'if unit: issues.append(': 'if unit and expected_unit(unit, now): issues.append(',
        'if n >= 3: issues.append(': 'if n >= 3 and expected_unit(u, now): issues.append(',
        'if a.get("code") not in noisy: issues.append(':
            'if a.get("code") not in noisy and expected_alert(a, now): issues.append(',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise TraderError('SENTINEL_PATCH_ANCHOR_MISMATCH')
        source = source.replace(old, new)
    ast.parse(source)
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repo', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.source.resolve() == args.output.resolve():
        parser.error('stage to a separate file; the operator installs it after review')
    patched = patch_sentinel(args.source.read_text(), args.repo, args.source_sha256)
    # Exclusive creation also prevents an accidental overwrite or symlink follow.
    with args.output.open('x') as output:
        output.write(patched)
    args.output.chmod(0o700)


if __name__ == '__main__':
    main()
