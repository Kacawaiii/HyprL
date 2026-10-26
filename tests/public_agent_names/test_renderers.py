"""Synthetic presentation/delivery tests. Never call a source, model or webhook."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.public_agent_names import PUBLIC_METHOD, public_agent_label, public_research_message
from scripts.public_brief_install import patched_source
from scripts.public_weekly import weekly
from scripts.trading_lab.replay_bench.grant import ReplayError, instant
from scripts.trading_lab.replay_seal_forward import experiment as ex, forward, spec
from scripts.trading_lab.sources.canonical import sha256_canonical


def scores():
    return {'scored_at': '2026-10-23T21:15:00Z', 'through_session': '2026-10-23',
            'rows': [{'session': '2026-10-22', 'group': 'ab', 'asset': 'AAA', 'horizon': '1d',
                      'snapshot_hash': 'a' * 64, 'family': f} for f in ('opus', 'gpt')],
            'primary_endpoint': {'brier_skill': {'opus': .2, 'gpt': .1}, 'ci95': [-.1, .2], 'state': 'COMPUTED'},
            'verdict_wording': 'Opus wins on the primary endpoint (synthetic wording only).',
            'full_table': {f'panel-{i}': {'opus': {'n': 17, 'brier': .19}, 'gpt': {'n': 17, 'brier': .21}}
                           for i in range(40)}, 'trades': [{'family': 'gpt', 'hit': True}],
            'failures': {'opus': {'count': 1, 'failed_sessions': [{'session': '2026-10-20', 'error': 'SYNTHETIC_ERROR'}]}}}


@pytest.mark.parametrize('identity,expected', [('opus', 'HypR-A1'), ('Claude', 'HypR-A1'),
    ('Claude Opus', 'HypR-A1'), ('analyst_claude', 'HypR-A1'), ('reviewer_claude', 'HypR-A1'),
    ('gpt', 'HypR-E1'), ('Codex', 'HypR-E1'), ('analyst_gpt', 'HypR-E1'), ('unknown', 'unknown')])
def test_agent_labels(identity, expected):
    assert public_agent_label(identity) == expected


def test_weekly_delivery_names_numbers_hashes_and_immutable_inputs(tmp_path):
    base = tmp_path / forward.STATE_DIR
    base.mkdir()
    source = base / 'scores.json'
    source.write_text(json.dumps(scores()))
    before = source.read_bytes()
    messages = []
    def send(args, **kw):
        assert args == ['synthetic-notify', 'brouillons']
        messages.append(kw['input'])
        return subprocess.CompletedProcess(args, 0, 'discord\n')
    result = weekly(tmp_path, notify='synthetic-notify', clock=lambda: instant('2026-10-23T21:15:00Z'), runner=send)
    assert result['paired_rows'] == 1
    text = ''.join(messages)
    assert 'HypR-A1' in text and 'HypR-E1' in text and PUBLIC_METHOD in text
    assert all(old not in text for old in ('opus', 'gpt', 'Claude', 'Opus', 'GPT', 'Codex'))
    assert '"n": 17' in text and '"brier": 0.19' in text and 'jours en échec=1' in text
    assert len(messages) > 1 and all(len(m) <= 1800 for m in messages)
    receipt = base / 'weekly-2026-W43.json'
    saved = receipt.read_bytes()
    assert json.loads(saved)['text_hash'] == sha256_canonical(text)
    assert json.loads(saved)['score_sha256'] == sha256_canonical(scores())
    assert source.read_bytes() == before
    with pytest.raises(ReplayError, match='WEEKLY_DELIVERY_ALREADY_ATTEMPTED'):
        weekly(tmp_path, notify='synthetic-notify', clock=lambda: instant('2026-10-23T21:15:00Z'), runner=send)
    assert receipt.read_bytes() == saved


def test_weekly_correction_retains_historical_receipt(tmp_path):
    base = tmp_path / forward.STATE_DIR
    base.mkdir()
    source = scores()
    source['scored_at'] = '2026-10-25T21:15:00Z'
    (base / 'scores.json').write_text(json.dumps(source))
    receipt = base / 'weekly-2026-W43.json'
    old = {'state': 'DELIVERED', 'text_hash': 'b' * 64, 'score_sha256': 'c' * 64, 'messages': 2}
    receipt.write_text(json.dumps(old))
    before = receipt.read_bytes()
    weekly(tmp_path, notify='synthetic', clock=lambda: instant('2026-10-25T21:15:00Z'),
           correction_week='2026-W43', runner=lambda a, **k: subprocess.CompletedProcess(a, 0, 'discord'))
    assert receipt.read_bytes() == before
    correction = json.loads((base / ('weekly-2026-W43-correction-' + 'b' * 64 + '.json')).read_text())
    assert correction['corrects_text_hash'] == old['text_hash']


@pytest.mark.parametrize('failure', ['stale', 'future', 'outside_window', 'score_digest', 'telegram'])
def test_weekly_preserves_admission_and_delivery_gates(tmp_path, failure):
    base = tmp_path / forward.STATE_DIR
    base.mkdir()
    value = scores()
    if failure == 'stale': value['scored_at'] = '2026-10-23T18:00:00Z'
    if failure == 'future': value['scored_at'] = '2026-10-23T22:00:00Z'
    (base / 'scores.json').write_text(json.dumps(value))
    at = '2026-10-22T21:15:00Z' if failure == 'outside_window' else '2026-10-23T21:15:00Z'
    options = {}
    if failure == 'score_digest':
        (base / 'score-runs').mkdir()
        (base / 'score-runs' / ('d' * 64 + '.json')).write_text(json.dumps(value))
        options['score_run'] = 'd' * 64
    calls = []
    def send(args, **kw):
        calls.append(kw['input'])
        return subprocess.CompletedProcess(args, 0, 'telegram-fallback')
    with pytest.raises(ReplayError):
        weekly(tmp_path, notify='synthetic', clock=lambda: instant(at), runner=send, **options)
    assert bool(calls) == (failure == 'telegram')
    if failure == 'telegram':
        assert json.loads((base / 'weekly-2026-W43.json').read_text())['state'] == 'INTENT'


def test_research_transport_preserves_ledger_model_ids_and_evidence():
    evidence = ('ledger public-ledger/forward-duel @ ' + 'a' * 40 + '\n'
                'opus: ' + 'b' * 64 + '\n'
                'model_id: claude-opus-4-6\n{"model": "opus", "family": "gpt"}\n'
                'https://example.test/opus/gpt\n')
    assert public_research_message(evidence) == evidence
    text = public_research_message('Opus vs GPT\nanalyst_claude: ready\nCodex: ready\n')
    assert text == 'HypR-A1 vs HypR-E1\nHypR-A1: ready\nHypR-E1: ready\n'
    assert public_research_message('Opus: source https://example.test/?agent=gpt and `gpt`\n') == (
        'HypR-A1: source https://example.test/?agent=gpt and `gpt`\n')


def test_notification_preview_never_sends():
    result = subprocess.run([sys.executable, '-m', 'scripts.public_research_notify', 'brouillons', '--render-only'],
                            input='opus: 3\ngpt: 4\n', text=True, capture_output=True, check=True)
    assert result.stdout == 'HypR-A1: 3\nHypR-E1: 4\n'


@pytest.mark.parametrize('route,status', [('discord', 0), ('telegram-fallback', 0), ('failed', 1)])
def test_research_commitment_transport_is_verbatim_and_keeps_confirmation(tmp_path, route, status):
    transport = tmp_path / 'notify'
    output = tmp_path / 'sent.txt'
    transport.write_text('#!' + sys.executable + '\nimport os,sys\nfrom pathlib import Path\n'
                         'Path(os.environ["TEST_SENT"]).write_text(sys.stdin.read())\n'
                         'print(os.environ["TEST_ROUTE"])\nsys.exit(int(os.environ["TEST_STATUS"]))\n')
    transport.chmod(0o700)
    evidence = ('Engagement HyprL forward 2026-10-23\nledger ledger/forward-duel @ ' + 'a'*40 +
                '\nopus: ' + 'b'*64 + '\ngpt: ' + 'c'*64 + '\n')
    result = subprocess.run([sys.executable, '-m', 'scripts.public_research_notify', 'ordres'],
        input=evidence, text=True, capture_output=True,
        env=dict(os.environ, HYPRL_RESEARCH_NOTIFY_TRANSPORT=str(transport), TEST_SENT=str(output),
                 TEST_ROUTE=route, TEST_STATUS=str(status)))
    assert result.returncode == status and result.stdout.strip() == route
    assert output.read_text() == evidence
    rejected = subprocess.run([sys.executable, '-m', 'scripts.public_research_notify', 'ordres', '--render-only'],
                              input='BUY AAA 10 shares', text=True, capture_output=True)
    assert rejected.returncode == 2 and rejected.stdout == ''


def test_brief_patch_only_changes_owned_prose_and_method():
    source = '''import sys
from pathlib import Path
INTERNAL = {'model': 'claude-opus-4-6', 'family': 'opus'}
DISCLOSURE = ("Conflits : la rédaction est assistée par Claude, un modèle d'IA d'Anthropic ; "
              "Anthropic peut être cité.")
def render():
    L = []
    L += ['', '> ' + DISCLOSURE, '']
    L += ['### Méthode et limites', 'déterministe']
    return '\\n'.join(L)
'''
    ns = {}
    exec(patched_source(source, Path(__file__).resolve().parents[2]), ns)
    original = copy.deepcopy(ns['INTERNAL'])
    text = ns['render']()
    assert 'HypR-A1' in text and PUBLIC_METHOD in text and 'Anthropic' in text
    assert all(old not in text for old in ('Claude', 'Opus', 'GPT', 'Codex'))
    assert ns['INTERNAL'] == original == {'model': 'claude-opus-4-6', 'family': 'opus'}
    with pytest.raises(ValueError, match='VERSION_UNKNOWN'):
        patched_source('different brief', Path('.'))


def test_frozen_research_code_still_matches_committed_freeze():
    assert spec.code_hash() == json.loads(ex.FREEZE.read_text())['code_hash']
