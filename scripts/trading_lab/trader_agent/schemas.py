"""Strict output contracts plus universe, direction and causal-source validation."""
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from .config import TraderError, instant

HERE = Path(__file__).parent


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def array(items, maximum=100):
    return {"type": "array", "items": items, "maxItems": maximum}


STRING = {"type": "string", "maxLength": 2000}
TIME = {"type": "string", "format": "date-time"}
HASH = {"type": "string", "pattern": "^[a-f0-9]{64}$"}
P = {"type": "number", "minimum": 0.2, "maximum": 0.8}
HORIZON = {"enum": ["1d", "5d"]}
SOURCE = obj({"url": {"type": "string", "format": "uri", "maxLength": 2048}, "published_at": TIME, "fact": STRING})
VIEW = obj({"asset": STRING, "horizon": HORIZON, "view": {"enum": ["UP", "DOWN", "ABSTAIN"]},
    "p_outperform": P, "confidence_reason": STRING, "catalysts": array(SOURCE, 10),
    "priced_in_assessment": STRING, "second_order": STRING, "counter_thesis": STRING,
    "falsifier": STRING, "event_risk": array(STRING, 10)})
ANALYST = obj({"regime": array(STRING, 6), "views": array(VIEW)})
VERDICT = obj({"analyst": {"enum": ["analyst_claude", "analyst_gpt"]}, "asset": STRING,
    "horizon": HORIZON, "verdict": {"enum": ["KEEP", "DOWNGRADE", "REJECT"]},
    "reason_code": {"enum": ["supported", "unsupported", "priced_in", "generic", "counter_evidence", "overconfidence", "disagreement", "abstained"]},
    "adjusted_p": {"anyOf": [P, {"type": "null"}]}, "note": STRING})
REVIEWER = obj({"verdicts": array(VERDICT, 200)})
DECISION_VIEW = obj({"asset": STRING, "horizon": HORIZON,
    "analyst": {"enum": ["analyst_claude", "analyst_gpt", "reviewer_claude", "reviewer_gpt", "consensus"]},
    "view": {"enum": ["UP", "DOWN", "ABSTAIN"]}, "p_outperform": P,
    "verdict": {"enum": ["KEEP", "DOWNGRADE", "REJECT", "ABSTAIN", "MISSING"]},
    "raw_view": {"anyOf": [VIEW, {"type": "null"}]},
    "review": {"anyOf": [VERDICT, {"type": "null"}]}})
# Only on MISSING views (a degraded run): the error code of the analyst that produced nothing. Optional, so every
# earlier decision stays valid.
DECISION_VIEW["properties"]["error"] = {"type": "string", "maxLength": 64}
DECISION = obj({"schema": {"const": "trader-decision-v1"}, "run_id": STRING, "session": STRING,
    "decision_at": TIME, "context_hash": HASH, "authorization_hash": HASH, "preregistration_hash": HASH,
    "skill_hashes": obj({"TRADER_SKILL.md": HASH, "REVIEWER_SKILL.md": HASH}),
    "models": obj({role: obj({"model": STRING, "cli_version": STRING, "reported_version": STRING})
                   for role in ("analyst_claude", "analyst_gpt", "reviewer")}),
    "synthetic": {"type": "boolean"}, "views": array(DECISION_VIEW, 500)})
SCHEMAS = {"analyst": ANALYST, "reviewer": REVIEWER, "decision": DECISION}

# OpenAI strict structured outputs (Codex --output-schema) accept only a subset of JSON Schema: 'format: uri' and
# 'maxLength' made the first real run fail with invalid_json_schema. GPT gets a derived schema; every local validator
# (validate / validate_analyst / validate_reviewer) keeps the full SCHEMAS, including URL checks.
OPENAI_STRICT_KEYWORDS = {"type", "properties", "required", "additionalProperties", "items", "enum", "anyOf",
                          "format", "pattern", "minimum", "maximum", "minItems", "maxItems"}
OPENAI_STRICT_FORMATS = {"date-time", "time", "date", "duration", "email", "hostname", "ipv4", "ipv6", "uuid"}


def for_openai_strict(schema):
    if isinstance(schema, list):
        return [for_openai_strict(x) for x in schema]
    if not isinstance(schema, dict):
        return schema
    result = {}
    for key, value in schema.items():
        if key not in OPENAI_STRICT_KEYWORDS and key not in {"description", "title"}:
            continue
        if key == "format" and value not in OPENAI_STRICT_FORMATS:
            continue
        # property NAMES are data, not keywords: recurse into their schemas only
        result[key] = ({k: for_openai_strict(v) for k, v in value.items()} if key == "properties" else for_openai_strict(value))
    return result


def openai_strict_problems(schema, path="$", depth=0):
    """Offline compatibility check against OpenAI's documented strict-mode rules; returns a list of problems."""
    problems = []
    if path == "$" and schema.get("type") != "object":
        problems.append("root must be an object")
    depth += schema.get("type") == "object"   # nesting counts object levels (root = 1)
    if depth > 5:
        problems.append(f"{path}: object nesting deeper than 5")
    for key in schema:
        if key not in OPENAI_STRICT_KEYWORDS and key not in {"description", "title"}:
            problems.append(f"{path}: unsupported keyword {key}")
    if "format" in schema and schema["format"] not in OPENAI_STRICT_FORMATS:
        problems.append(f"{path}: unsupported format {schema['format']}")
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is not False:
            problems.append(f"{path}: additionalProperties must be false")
        if sorted(schema.get("required", [])) != sorted(properties):
            problems.append(f"{path}: every property must be required")
        for name, child in properties.items():
            problems += openai_strict_problems(child, f"{path}.{name}", depth)
    if isinstance(schema.get("items"), dict):
        problems += openai_strict_problems(schema["items"], path + "[]", depth)
    for i, option in enumerate(schema.get("anyOf", [])):
        problems += openai_strict_problems(option, f"{path}|{i}", depth)
    return problems


GPT_SCHEMAS = {name: for_openai_strict(SCHEMAS[name]) for name in ("analyst", "reviewer")}


def validate(kind, payload):
    try:
        Draft202012Validator(SCHEMAS[kind], format_checker=FormatChecker()).validate(payload)
    except Exception as error:
        from jsonschema import ValidationError
        if isinstance(error, ValidationError):
            raise TraderError("SCHEMA_INVALID") from None
        raise
    return payload


def validate_analyst(payload, assets, at):
    validate("analyst", payload)
    keys = [(v["asset"], v["horizon"]) for v in payload["views"]]
    wanted = {(asset, h) for asset in assets for h in ("1d", "5d")}
    if len(keys) != len(set(keys)) or set(keys) != wanted:
        raise TraderError("UNIVERSE_INVALID")
    for view in payload["views"]:
        p, direction = view["p_outperform"], view["view"]
        if (direction == "UP" and p <= .5) or (direction == "DOWN" and p >= .5) or (direction == "ABSTAIN" and p != .5):
            raise TraderError("DIRECTION_INVALID")
        if direction != "ABSTAIN" and not view["catalysts"]:
            raise TraderError("UNSUPPORTED_VIEW")
        for source in view["catalysts"]:
            if instant(source["published_at"]) > at:
                raise TraderError("FUTURE_SOURCE")
            if not source["url"].startswith("https://"):
                raise TraderError("UNSAFE_SOURCE_URL")
    return payload


def validate_reviewer(payload, analysts):
    validate("reviewer", payload)
    views = {(role, v["asset"], v["horizon"]): v for role, output in analysts.items() for v in output["views"]}
    keys = [(v["analyst"], v["asset"], v["horizon"]) for v in payload["verdicts"]]
    if len(keys) != len(set(keys)) or set(keys) != set(views):
        raise TraderError("REVIEW_COVERAGE_INVALID")
    for review in payload["verdicts"]:
        original = views[(review["analyst"], review["asset"], review["horizon"])]
        p, adjusted = original["p_outperform"], review["adjusted_p"]
        if review["verdict"] == "DOWNGRADE":
            if adjusted is None or not min(.5, p) <= adjusted <= max(.5, p):
                raise TraderError("REVIEW_RAISED_PROBABILITY")
        elif adjusted is not None:
            raise TraderError("UNEXPECTED_ADJUSTED_PROBABILITY")
        if original["view"] == "ABSTAIN" and review["reason_code"] != "abstained":
            raise TraderError("REVIEW_CREATED_VIEW")
    return payload


def write_schemas():
    for name, schema in SCHEMAS.items():
        (HERE / "schemas" / (name + ".json")).write_text(json.dumps(schema, indent=2) + "\n")
    for name, schema in GPT_SCHEMAS.items():
        (HERE / "schemas" / (name + ".gpt.json")).write_text(json.dumps(schema, indent=2) + "\n")


if __name__ == "__main__":
    write_schemas()
