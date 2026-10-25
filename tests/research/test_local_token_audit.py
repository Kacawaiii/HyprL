import json

import pytest

from scripts.trading_lab.local_token_audit import account


def result():
    return {"type": "result", "session_id": "synthetic", "usage": {
        "input_tokens": 2, "cache_creation_input_tokens": 10,
        "cache_read_input_tokens": 100, "output_tokens": 4,
        "iterations": [{"input_tokens": 99999}]},
        "modelUsage": {"synthetic_main": {"inputTokens": 2, "cacheCreationInputTokens": 10,
            "cacheReadInputTokens": 100, "outputTokens": 4},
            "synthetic_helper": {"inputTokens": 20, "outputTokens": 5}}}


def test_helpers_counted_once_and_iterations_never_added():
    measured = account(json.dumps(result()))
    assert measured["new_tokens"] == 41
    assert measured["all_tokens"] == 141
    assert measured["cache_share_of_input"] == pytest.approx(100 / 132)


def test_streaming_frames_and_repeated_result_not_added():
    r = json.dumps(result())
    raw = '\n'.join([json.dumps({"type": "assistant", "usage": {"input_tokens": 99999}}), r, r])
    assert account(raw)["all_tokens"] == 141


def test_absent_helper_usage_is_explicitly_unknown():
    r = result()
    del r["modelUsage"]
    measured = account(json.dumps(r))
    assert measured["auxiliary_coverage"] == "UNKNOWN"


def test_ambiguous_totals_fail_instead_of_undercounting():
    r = result()
    r["usage"]["input_tokens"] = 999
    with pytest.raises(ValueError):
        account(json.dumps(r))


def test_missing_final_result_is_not_zero():
    with pytest.raises(ValueError):
        account('{"type":"assistant"}')


def test_single_streaming_assistant_frame_is_not_final_usage():
    with pytest.raises(ValueError):
        account('{"type":"assistant","usage":{"input_tokens":30}}')
