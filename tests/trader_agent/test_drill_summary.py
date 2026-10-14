from scripts.trading_lab.trader_agent.drill_summary import first_label_at, summarize


def test_first_label_checkpoint_matures_the_actual_calendar_population():
    assert first_label_at('2026-10-15') == '2026-10-15T21:30:00Z'
    assert first_label_at('2026-10-17') == '2026-10-18T21:30:00Z'
    assert first_label_at('2026-12-25') == '2026-12-26T21:30:00Z'


def test_nested_too_early_does_not_hide_a_completed_decision():
    result = {'status': 'COMPLETE', 'synthetic': True, 'decision': {'views': [{}]},
              'scorecard': {'comparison': {'state': 'TOO_EARLY'}}}
    assert summarize(result, 'run')['verified']
    assert summarize(result, 'run')['state'] == 'COMPLETE'


def test_zero_labels_or_an_empty_decision_cannot_pass_the_drill():
    assert not summarize({'state': 'COMPLETE', 'labels_added': 0, 'pending': 0}, 'label')['verified']
    assert not summarize({'status': 'COMPLETE', 'synthetic': True, 'decision': {'views': []}}, 'run')['verified']
    assert not summarize({'state': 'COMPLETE', 'labels_added': 10, 'pending': 1}, 'matured')['verified']


def test_busy_or_changed_paper_snapshot_is_an_unproven_check():
    result = {'state': 'COMPLETE', 'production_writes': 0, 'broker_methods': ['GET'], 'snapshot_unchanged': True}
    assert summarize(result, 'paper')['verified']
    assert not summarize({**result, 'snapshot_unchanged': False}, 'paper')['verified']
    assert not summarize({'state': 'PAPER_EXECUTION_IN_PROGRESS'}, 'paper')['verified']


def test_equity_skip_must_match_the_pinned_closed_calendar():
    result = {'state': 'SKIPPED_CLOSED_EQUITY_MARKET', 'at': '2026-10-17T22:45:00Z',
              'production_writes': 0, 'broker_methods': ['GET'], 'snapshot_unchanged': True}
    assert summarize(result, 'paper')['verified']
    assert not summarize({**result, 'at': '2026-10-15T22:45:00Z'}, 'paper')['verified']
