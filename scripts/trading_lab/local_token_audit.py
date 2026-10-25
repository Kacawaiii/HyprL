"""Pure offline accounting for saved Claude CLI results, including web helpers."""
import json

FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
MODEL_FIELDS = ("inputTokens", "cacheCreationInputTokens", "cacheReadInputTokens", "outputTokens")


def final_result(raw):
    try:
        value = json.loads(raw)
    except ValueError:
        events = []
        for line in raw.splitlines():
            if line.strip():
                events.append(json.loads(line))
        results = [e for e in events if isinstance(e, dict) and e.get("type") == "result"]
        if not results:
            raise ValueError("no complete final result")
        value = results[-1]
    if (not isinstance(value, dict) or not isinstance(value.get("usage"), dict)
            or value.get("type") not in (None, "result")):
        raise ValueError("no complete final result")
    return value


def counts(usage, keys):
    values = [usage.get(key, 0) for key in keys]
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError("invalid token counter")
    return dict(zip(FIELDS, values))


def account(raw):
    event = final_result(raw)
    main = counts(event["usage"], FIELDS)
    models = {name: counts(value, MODEL_FIELDS) for name, value in event.get("modelUsage", {}).items()}
    total = {key: sum(v[key] for v in models.values()) for key in FIELDS} if models else main
    if models and main != total and main not in models.values():
        raise ValueError("aggregate usage disagrees with model counters")
    incoming = sum(total[k] for k in FIELDS[:3])
    return {"tokens": total, "model_tokens": models,
            "new_tokens": total["input_tokens"] + total["cache_creation_input_tokens"] + total["output_tokens"],
            "all_tokens": sum(total.values()),
            "cache_share_of_input": total["cache_read_input_tokens"] / incoming if incoming else None,
            "auxiliary_coverage": "per_model" if models else "UNKNOWN",
            "identity": event.get("session_id") or event.get("uuid"),
            "turns": event.get("num_turns"), "is_error": event.get("is_error"),
            "duration_ms": event.get("duration_ms", 0)}
