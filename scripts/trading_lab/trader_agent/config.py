"""Operator grants are private, explicit and rechecked at dispatch time."""
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.authorization_extensions import ExtensionError, expiry_receipt

AMENDMENT_NAME = 'trader-agent-crypto-amendment-v1.json'
AMENDMENT_HASH = '2bf0d9d5a3f62eb9ea94ffafe8a5ec630d0986dc184428c0340e49f714c3b6d7'
AMENDED_CRYPTO = ('SOL-USD', 'AVAX-USD', 'LINK-USD', 'DOGE-USD', 'LTC-USD')
WEEKEND_DECISION_NAME = 'trader-operator-decisions-2026-10-10.json'
WEEKEND_DECISION_HASH = 'ceb0c46dd68b68d1db0dd995bf39b009934ce5e21de09adbffa7758143215cff'
CRYPTO_PRODUCTS = frozenset({'BTC-USD', 'ETH-USD', *AMENDED_CRYPTO})
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
    base_identity: str | None = None
    amendment_identity: str | None = None
    amendment: dict | None = None
    weekend_decision: dict | None = None
    news_decision: dict | None = None
    pm_amendment: dict | None = None
    pm_identity: str | None = None
    retirement_decision: dict | None = None
    source_path: Path | None = None
    source_payload: dict | None = None
    amendment_source_path: Path | None = None
    work_budget_approval: dict | None = None

    @classmethod
    def load(cls, path, *, amendment_path=None):
        payload = strict_json(Path(path).read_text())
        grant = cls(payload, sha256_canonical(payload), source_path=Path(path).resolve(),
                    source_payload=deepcopy(payload))
        grant.validate()
        explicit_amendment = amendment_path is not None
        amendment_path = Path(amendment_path) if amendment_path else Path(path).parent / AMENDMENT_NAME
        if not amendment_path.exists():
            if explicit_amendment:
                raise TraderError('CRYPTO_AMENDMENT_UNAVAILABLE')
            return grant.load_weekend(Path(path).parent)
        amendment = strict_json(amendment_path.read_text())
        body = {k: v for k, v in amendment.items() if k != 'canonical_sha256'}
        if amendment.get('canonical_sha256') != AMENDMENT_HASH or sha256_canonical(body) != AMENDMENT_HASH:
            raise TraderError('CRYPTO_AMENDMENT_HASH_MISMATCH')
        if (body.get('base_grant_hash') != grant.identity or body.get('extends') != 'trader-agent-v1'
                or body.get('crypto') != list(AMENDED_CRYPTO) or body.get('excluded') != ['BTC-USD', 'ETH-USD']
                or body.get('coinbase_max_requests_per_day') != 40
                or instant(body['not_after']) > instant(payload['not_after'])):
            raise TraderError('CRYPTO_AMENDMENT_BINDING_MISMATCH')
        from scripts.trading_lab.research_protection import protection_table
        from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1
        table = protection_table()
        if (sha256_canonical(table) != body['protection_table_hash'] or
                any(a in table or PROTECTED_WINDOW_V1.protects_product(a) for a in body['crypto'])):
            raise TraderError('CRYPTO_AMENDMENT_PROTECTED')
        effective = deepcopy(payload)
        effective['universe']['crypto'] = body['crypto']
        effective['data_sources']['coinbase_exchange_public']['max_requests_per_day'] = 40
        identity = sha256_canonical({'base_grant_hash': grant.identity, 'amendment_hash': AMENDMENT_HASH,
                                     'effective_payload': effective})
        amended = replace(grant, payload=effective, identity=identity, base_identity=grant.identity,
                          amendment_identity=AMENDMENT_HASH, amendment=amendment,
                          amendment_source_path=amendment_path.resolve())
        amended.validate()
        return amended.load_weekend(Path(path).parent)

    def load_weekend(self, folder):
        from .pm_retirement_spec import DECISION_NAME as pm_retirement_name, check_decision
        if (folder / pm_retirement_name).exists():
            check_decision(folder / pm_retirement_name)
        from .work_budget import APPROVAL_NAME
        work_approval = folder / APPROVAL_NAME
        if work_approval.exists():
            self = replace(self, work_budget_approval=strict_json(work_approval.read_text()))
        from .retirement_spec import DECISION_NAME as retirement_name, DECISION_HASH as retirement_hash
        path = folder / retirement_name
        if path.exists():
            decision = strict_json(path.read_text())
            if sha256_canonical(decision) != retirement_hash:
                raise TraderError('GPT_RETIREMENT_OPERATOR_DECISION_HASH_MISMATCH')
            self = replace(self, retirement_decision=decision)
        from .slots import load_amendment
        self = load_amendment(self, folder)
        from .news_spec import DECISION_NAME, DECISION_HASH
        news_path = folder / DECISION_NAME
        if news_path.exists():
            news = strict_json(news_path.read_text())
            if sha256_canonical(news) != DECISION_HASH:
                raise TraderError('NEWS_OPERATOR_DECISION_HASH_MISMATCH')
            self = replace(self, news_decision=news)
        path = folder / WEEKEND_DECISION_NAME
        if not path.exists():
            return self
        decision = strict_json(path.read_text())
        if sha256_canonical(decision) != WEEKEND_DECISION_HASH:
            raise TraderError('WEEKEND_OPERATOR_DECISION_HASH_MISMATCH')
        return replace(self, weekend_decision=decision)

    def check_slot(self, slot, at, *, preflight=False):
        self.check(at, slot=slot)
        if slot == 'am':
            return
        from .pm_retirement_spec import active
        if slot == 'pm' and active(at):
            raise TraderError('PM_SLOT_RETIRED')
        if slot != 'pm' or not self.pm_amendment or not self.pm_identity:
            raise TraderError('PM_SLOT_NOT_AUTHORIZED')
        start = self.pm_amendment['granted_at'] if preflight else self.pm_amendment['effective_from']
        if not instant(start) <= at < instant(self.pm_amendment['not_after']):
            raise TraderError('PM_SLOT_NOT_STARTED_OR_EXPIRED')

    def check_news(self, at):
        from .news_spec import DECISION_HASH, spec
        self.check(at)
        if (not self.news_decision or sha256_canonical(self.news_decision) != DECISION_HASH
                or at < instant(spec()['effective_from'])):
            raise TraderError('NEWS_NOT_AUTHORIZED')

    def check_weekend(self, at):
        self.check(at)
        if (not self.weekend_decision or instant(self.weekend_decision['decided_at']) > at
                or self.amendment_identity is None
                or self.payload['universe']['crypto'] != list(AMENDED_CRYPTO)):
            raise TraderError('WEEKEND_CRYPTO_NOT_AUTHORIZED')

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
        if not (set(p["universe"]["crypto"]) <= {"BTC-USD", "ETH-USD"} if self.amendment_identity is None else
                self.amendment_identity == AMENDMENT_HASH and p["universe"]["crypto"] == list(AMENDED_CRYPTO)):
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

    def extension_metadata(self, at, *, slot='am'):
        try:
            return expiry_receipt(self.source_path, self.source_payload or self.payload, at, slot=slot)
        except ExtensionError:
            raise TraderError("AUTHORIZATION_EXPIRED_OR_NOT_STARTED")

    def check(self, at, *, slot='am'):
        # Preserve original and amendment identities; never rewrite their payloads.
        metadata = self.extension_metadata(at, slot=slot)
        if not metadata and not instant(self.payload['granted_at']) <= at < instant(self.payload['not_after']):
            raise TraderError('AUTHORIZATION_EXPIRED_OR_NOT_STARTED')
        if self.amendment and not instant(self.amendment['granted_at']) <= at < instant(self.amendment['not_after']):
            try:
                receipt = expiry_receipt(self.amendment_source_path, self.amendment, at, slot=slot)
                metadata = {**metadata, 'crypto_amendment_extension_sha256': receipt['authorization_extension_sha256']}
            except (ExtensionError, KeyError):
                raise TraderError('CRYPTO_AMENDMENT_EXPIRED_OR_NOT_STARTED') from None
        return metadata

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
