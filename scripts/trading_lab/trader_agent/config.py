"""Operator grants are private, explicit and rechecked at dispatch time."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from scripts.trading_lab.sources.canonical import sha256_canonical

UTC = timezone.utc
ROLES = ("analyst_claude", "analyst_gpt", "reviewer")
HOSTS = {"yahoo_chart": "query1.finance.yahoo.com",
         "coinbase_exchange_public": "api.exchange.coinbase.com",
         "gdelt_doc_api": "api.gdeltproject.org"}


class TraderError(ValueError):
    def __init__(self, code, evidence=None):
        self.code = code
        self.evidence = evidence
        super().__init__(code)


def instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise TraderError("NAIVE_TIME")
    return result.astimezone(UTC)


def iso(value):
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def now():
    return datetime.now(UTC)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise TraderError("DUPLICATE_JSON_KEY")
            result[key] = value
        return result
    def constant(_):
        raise TraderError("NONFINITE_JSON")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


@dataclass(frozen=True)
class Authorization:
    payload: dict
    identity: str

    @classmethod
    def load(cls, path):
        payload = strict_json(Path(path).read_text())
        grant = cls(payload, sha256_canonical(payload))
        grant.validate()
        return grant

    def validate(self):
        p = self.payload
        if p.get("mode") != "PAPER_SHADOW_ONLY":
            raise TraderError("PAPER_ONLY")
        if instant(p["not_after"]) <= instant(p["granted_at"]):
            raise TraderError("INVALID_AUTHORIZATION_WINDOW")
        if set(p["external_models"]) != set(ROLES):
            raise TraderError("UNAUTHORIZED_MODEL")
        for role, model in p["external_models"].items():
            expected = "codex" if role == "analyst_gpt" else "claude"
            if model["cli"] != expected:
                raise TraderError("UNAUTHORIZED_CLI")
            if role != "analyst_gpt" and (set(model["tools"]) != {"WebSearch", "WebFetch"}
                    or model["model"] != ("sonnet" if role == "reviewer" else "opus")):
                raise TraderError("UNAUTHORIZED_MODEL_TOOLS")
            for key in ("calls_per_day", "timeout_minutes"):
                if type(model[key]) is not int or model[key] <= 0:
                    raise TraderError("INVALID_MODEL_BUDGET")
            if type(model["retries_per_day"]) is not int or model["retries_per_day"] < 0:
                raise TraderError("INVALID_RETRY_BUDGET")
        for source, host in HOSTS.items():
            if p["data_sources"][source]["host"] != host:
                raise TraderError("UNAUTHORIZED_HOST")
            count = p["data_sources"][source]["max_requests_per_day"]
            if type(count) is not int or count < 1:
                raise TraderError("INVALID_SOURCE_BUDGET")
        if set(p["universe"]) != {"stocks", "sector_etfs", "crypto", "benchmarks_not_predicted"}:
            raise TraderError("INVALID_UNIVERSE")
        all_assets = [x for group in p["universe"].values() for x in group]
        if len(set(all_assets)) != len(all_assets) or not all_assets:
            raise TraderError("INVALID_UNIVERSE")
        if any(not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,15}", x) for x in all_assets):
            raise TraderError("INVALID_SYMBOL")
        if not set(p["universe"]["crypto"]) <= {"BTC-USD", "ETH-USD"}:
            raise TraderError("UNAUTHORIZED_CRYPTO")
        if "SPY" not in p["universe"]["benchmarks_not_predicted"]:
            raise TraderError("MISSING_BENCHMARK")
        if p["budgets"]["max_runs_per_day"] != 1 or p["cadence"]["runs_per_us_trading_day"] != 1:
            raise TraderError("INVALID_CADENCE")
        if type(p["budgets"]["max_llm_calls_per_day"]) is not int or not 1 <= p["budgets"]["max_llm_calls_per_day"] <= 6:
            raise TraderError("INVALID_LLM_BUDGET")
        if p["budgets"]["paper_capital_usd"] <= 0:
            raise TraderError("INVALID_PAPER_CAPITAL")
        self.gpt_model

    def check(self, at):
        if not instant(self.payload["granted_at"]) <= at < instant(self.payload["not_after"]):
            raise TraderError("AUTHORIZATION_EXPIRED_OR_NOT_STARTED")

    @property
    def gpt_model(self):
        value = self.payload["external_models"]["analyst_gpt"]["model"]
        match = re.fullmatch(r"operator default \(([-a-zA-Z0-9.]+)\)", value)
        name = match[1] if match else value
        if not re.fullmatch(r"gpt-[-a-zA-Z0-9.]+", name):
            raise TraderError("UNRESOLVED_GPT_MODEL")
        return name

    @property
    def universe(self):
        u = self.payload["universe"]
        return u["stocks"] + u["sector_etfs"] + u["crypto"]


def private_root(root):
    root = Path(root).expanduser().resolve()
    repo = Path(__file__).resolve().parents[3]
    if root == repo or repo in root.parents:
        raise TraderError("RUNTIME_MUST_BE_OUTSIDE_GIT")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    return root
