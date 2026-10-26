#!/usr/bin/env python3
"""Research-only notification adapter; frozen research and ledger bytes stay intact."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.public_agent_names import public_research_message


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('channel', choices=('brief', 'brouillons', 'alertes', 'ordres'))
    parser.add_argument('--render-only', action='store_true', help='offline preview; no delivery')
    args = parser.parse_args(argv)
    original = sys.stdin.read()
    if args.channel == 'ordres':
        # Research publishes immutable hash commitments on this existing route.
        # Never transform their keys or admit an order/trading message here.
        lines = original.splitlines()
        if (len(lines) < 3 or not re.fullmatch(r'Engagement HyprL [a-z-]+ [\w:-]+', lines[0])
                or not re.fullmatch(r'ledger ledger/forward-duel @ [0-9a-f]{40}', lines[1])
                or any(not re.fullmatch(r'[\w:-]{1,120}: [0-9a-f]{64}', line) for line in lines[2:])):
            parser.error('Only unchanged hash-only research commitments may use this route')
        rendered = original
    else:
        rendered = public_research_message(original)
    if args.render_only:
        sys.stdout.write(rendered)
        return 0
    notify = os.environ.get('HYPRL_RESEARCH_NOTIFY_TRANSPORT')
    if not notify or Path(notify).resolve() == Path(__file__).resolve():
        parser.error('Set HYPRL_RESEARCH_NOTIFY_TRANSPORT to the existing notify executable')
    # Inherit stdout and exit status: Telegram fallback must never become a
    # successful Discord receipt. The transport retains its own chunking.
    return subprocess.run([notify, args.channel], input=rendered, text=True).returncode


if __name__ == '__main__':
    sys.exit(main())
