"""Presentation names only. Never apply these helpers to records or model identities."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_AGENT_NAMES = json.loads((ROOT / 'configs/public_agent_names.json').read_text())
PUBLIC_METHOD = 'agents propriétaires HyprL construits sur des modèles de langage tiers'


def public_agent_label(identity):
    """Render a known agent role; keep unknown identities explicit."""
    family = {'claude opus': 'opus', 'claude': 'opus', 'codex': 'gpt', 'analyst_claude': 'opus',
              'reviewer_claude': 'opus', 'analyst_gpt': 'gpt', 'reviewer_gpt': 'gpt'}.get(
                  identity.lower(), identity.lower())
    return PUBLIC_AGENT_NAMES.get(family, identity)


# Human labels, rather than substrings in opaque keys, paths, URLs or model ids.
_LABEL = re.compile(r'(?<![\w./:@-])(Claude(?: Opus)?|Opus|GPT|Codex|analyst_claude|'
                    r'reviewer_claude|analyst_gpt|reviewer_gpt)(?![\w./@-])', re.I)
_EVIDENCE = re.compile(r'^[^\s:]+: [0-9a-f]{64}\s*$|\b(?:model_id|model|family|configuration_hash)\s*["\']?\s*[:=]', re.I)
_LITERAL = re.compile(r'https?://\S+|`[^`]+`')


def _prose(line):
    pieces, start = [], 0
    for literal in _LITERAL.finditer(line):
        pieces.extend((_LABEL.sub(lambda m: public_agent_label(m.group()), line[start:literal.start()]), literal.group()))
        start = literal.end()
    pieces.append(_LABEL.sub(lambda m: public_agent_label(m.group()), line[start:]))
    return ''.join(pieces)


def public_agent_text(text):
    """Render owned prose/labels, retaining evidence lines verbatim.

    Source quotations and arbitrary structured records must not be passed here.
    In particular, ledger branch/hash lines and explicit model identity fields
    remain evidence even when they appear in a public message.
    """
    return ''.join(line if line.startswith(('ledger ', 'Engagement HyprL ')) or _EVIDENCE.search(line)
                   else _prose(line)
                   for line in text.splitlines(keepends=True))


def public_research_message(text):
    """Render the frozen research notification template at the delivery boundary."""
    rendered = public_agent_text(text)
    if 'LIVE_FORWARD — duel hebdomadaire ' in rendered:
        # Put the disclosure in the first message, including when weekly output
        # has already been divided into transport-sized parts.
        first, sep, rest = rendered.partition('\n')
        rendered = first + '\nMéthode : ' + PUBLIC_METHOD + '.' + sep + rest
    return rendered
