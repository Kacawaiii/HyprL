"""Prospective run -> independent analysts -> reviewer -> immutable evidence."""
import hashlib
import json
from pathlib import Path

from scripts.trading_lab.platform.contracts import ModelContract, PredictionRecord
from scripts.trading_lab.research.contracts import PredictionEvidence
from scripts.trading_lab.research.store import ResearchStore
from scripts.trading_lab.sources.canonical import sha256_canonical

from .config import TraderError, instant, iso, now
from .data import build_context, calendar_session, label_window
from .portfolio import portfolios
from .runners import HERE, prompt
from .schemas import validate, validate_analyst, validate_reviewer

VARIANTS = ["analyst_claude", "analyst_gpt", "reviewer_claude", "reviewer_gpt", "consensus",
            "reviewer_kept", "reviewer_rejected", "reviewer_downgraded", "always_up", "momentum20",
            "random_seeded", "spy_relative_zero", "unhedged", "spy_hedged"]
ARTIFACT = Path(__file__).resolve().parents[3] / "docs/artifacts/trader_agent_preregistration_v1.json"


def preregistration():
    artifact = json.loads(ARTIFACT.read_text())
    expected = artifact.pop("canonical_sha256")
    if sha256_canonical(artifact) != expected:
        raise TraderError("PREREGISTRATION_HASH_MISMATCH")
    return expected


def skills():
    texts = {name: (HERE / "skills" / name).read_text() for name in ("TRADER_SKILL.md", "REVIEWER_SKILL.md")}
    hashes = {name: hashlib.sha256((HERE / "skills" / name).read_bytes()).hexdigest() for name in texts}
    artifact = json.loads(ARTIFACT.read_text())
    if artifact["skill_hashes"] != hashes:
        raise TraderError("SKILL_HASH_MISMATCH")
    return texts, hashes


def views_with_review(analysts, reviewer):
    reviews = {(r["analyst"], r["asset"], r["horizon"]): r for r in reviewer["verdicts"]}
    output, kept = [], {}
    for analyst, result in analysts.items():
        for v in result["views"]:
            review = reviews[(analyst, v["asset"], v["horizon"])]
            common = {k: v[k] for k in ("asset", "horizon", "view", "p_outperform")}
            output.append({**common, "analyst": analyst, "verdict": review["verdict"], "raw_view": v, "review": review})
            reviewed = {**common, "analyst": "reviewer_" + analyst.split("_")[-1],
                        "verdict": review["verdict"], "raw_view": v, "review": review}
            if review["verdict"] == "REJECT" or v["view"] == "ABSTAIN":
                reviewed.update(view="ABSTAIN", p_outperform=.5)
            elif review["verdict"] == "DOWNGRADE":
                reviewed["p_outperform"] = review["adjusted_p"]
                if review["adjusted_p"] == .5:
                    reviewed["view"] = "ABSTAIN"
            output.append(reviewed)
            kept.setdefault((v["asset"], v["horizon"]), []).append(reviewed)
    for (asset, horizon), pair in kept.items():
        agrees = len(pair) == 2 and pair[0]["view"] == pair[1]["view"] and pair[0]["view"] != "ABSTAIN"
        p = min((r["p_outperform"] for r in pair), key=lambda p: abs(p - .5)) if agrees else .5
        output.append({"asset": asset, "horizon": horizon, "analyst": "consensus",
                       "view": pair[0]["view"] if agrees else "ABSTAIN", "p_outperform": p,
                       "verdict": "KEEP" if agrees else "ABSTAIN", "raw_view": None, "review": None})
    return output


class TraderService:
    def __init__(self, ledger, data, runner, *, clock=now, fomc=None, edgar=None):
        self.ledger, self.data, self.runner, self.clock = ledger, data, runner, clock
        self.fomc, self.edgar = fomc, edgar
        self.store = ResearchStore(ledger.root / "evidence")

    def summary(self, run_id, status, at, **extra):
        payload = {"schema": "trader-run-v1", "run_id": run_id, "status": status, "at": iso(at),
                   "synthetic": self.data.synthetic, **extra}
        self.store.append("replay-summary", payload, object_id=run_id + ":" + status, recorded_at=iso(at))
        return payload

    def run(self):
        with self.ledger.owner():
            at = self.clock()
            day = at.date().isoformat()
            run_id = "trader:" + day + (":synthetic" if self.data.synthetic else ":real")
            try:
                self.ledger.grant.check(at)
                if self.ledger.paused:
                    return self.summary(run_id, "PAUSED", at)
                if not self.data.synthetic and at < instant("2026-10-06T12:00:00Z"):
                    return self.summary(run_id, "NOT_STARTED", at)
                session = calendar_session(day)
                if not session:
                    return self.summary(run_id, "SKIPPED_HOLIDAY", at)
                if at >= session.open_at:
                    raise TraderError("MISSED_DECISION_DEADLINE")
                self.runner.deadline = session.open_at
                prereg_hash = preregistration()
                texts, hashes = skills()
                self.ledger.reserve("run")
                self.summary(run_id, "RUNNING", at, preregistration_hash=prereg_hash, multiple_testing_variants=VARIANTS)
                context, context_hash = build_context(self.ledger.grant, self.data, at=at, fomc=self.fomc, edgar=self.edgar)
                self.store.append('replay-summary', {'schema':'trader-context-evidence-v1', 'run_id':run_id,
                    'context_hash':context_hash, 'context':context}, object_id=run_id + ':context',
                    recorded_at=context['decision_time'])
                if context["exclusions"]:
                    self.ledger.alert("PARTIAL_CONTEXT")
                context_at = instant(context["decision_time"])
                entry_deadline = min(label_window(a, day, '1d', self.ledger.grant.payload['universe']['crypto'])[0]
                                     for a in context['universe'])
                self.runner.deadline = entry_deadline
                analysts = {}
                # Neither prompt contains the other analyst's output. Serial to bound host memory.
                for role in ("analyst_claude", "analyst_gpt"):
                    self.before_call(session)
                    analysts[role] = self.runner.infer(role, prompt(context, texts["TRADER_SKILL.md"]),
                        validator=lambda output: validate_analyst(output, context["universe"], context_at))
                self.before_call(session)
                reviewer = self.runner.infer("reviewer", prompt(context, texts["REVIEWER_SKILL.md"], analysts=analysts),
                    validator=lambda output: validate_reviewer(output, analysts))
                decision_at = self.clock()
                if decision_at >= entry_deadline:
                    raise TraderError("MISSED_DECISION_DEADLINE")
                decision = {"schema": "trader-decision-v1", "run_id": run_id, "session": day,
                    "decision_at": iso(decision_at), "context_hash": context_hash,
                    "authorization_hash": self.ledger.grant.identity, "preregistration_hash": prereg_hash,
                    "skill_hashes": hashes, "models": self.runner.metadata, "synthetic": self.data.synthetic,
                    "views": views_with_review(analysts, reviewer)}
                validate("decision", decision)
                self.record(decision, context)
                return self.summary(run_id, "COMPLETE", decision_at, decision=decision,
                    portfolios=portfolios(decision["views"]), exclusions=context["exclusions"],
                    reviewed_analyst_portfolios={a:portfolios(decision['views'],analyst=a)
                                                for a in ('reviewer_claude','reviewer_gpt')},
                    multiple_testing_variants=VARIANTS,
                    budget_counts=self.ledger.counts())
            except TraderError as error:
                self.ledger.alert(error.code)
                return self.summary(run_id, "SKIPPED_QUOTA" if error.code == "SKIPPED_QUOTA" else "FAILED",
                                    self.clock(), error=error.code, budget_counts=self.ledger.counts())
            except Exception:
                self.ledger.alert("UNEXPECTED_FAILURE")
                self.summary(run_id, "FAILED", self.clock(), error="UNEXPECTED_FAILURE")
                raise

    def before_call(self, session):
        self.ledger.grant.check(self.clock())
        if self.clock() >= getattr(self.runner, 'deadline', session.open_at) or self.ledger.paused:
            raise TraderError("MISSED_DECISION_DEADLINE_OR_PAUSED")

    def record(self, decision, context):
        recorded = iso(self.clock())
        crypto = self.ledger.grant.payload["universe"]["crypto"]
        proposals = {a:portfolios(decision['views'],analyst=a) for a in ('reviewer_claude','reviewer_gpt','consensus')}
        context_evidence = self.store.records('replay-summary', object_id=decision['run_id'] + ':context')
        if len(context_evidence) != 1 or context_evidence[0]['payload']['context_hash'] != sha256_canonical(context):
            raise TraderError('CONTEXT_BINDING_INVALID')
        context_record_hash = context_evidence[0]['identity']
        for view in decision["views"]:
            entry, exit_at = label_window(view["asset"], decision["session"], view["horizon"], crypto)
            at = instant(decision["decision_at"])
            if instant(recorded) >= entry:
                raise TraderError("MISSED_RECORDING_DEADLINE")
            identity = sha256_canonical({"run": decision["run_id"], "analyst": view["analyst"],
                                        "asset": view["asset"], "horizon": view["horizon"]})
            price = context["prices"][view["asset"]]
            snapshot = {'schema':'trader-input-reference-v1', 'context_hash':decision['context_hash'],
                        'context_record_hash':context_record_hash, 'decision_time':context['decision_time'],
                        'asset':view['asset'], 'price':price, 'headlines_hash':sha256_canonical(context['headlines'].get(view['asset'])),
                        'macro_politics_trade_hash':sha256_canonical({a:context['headlines'][a] for a in ('macro','politics','trade')}),
                        'archives_hash':sha256_canonical(context['archives']), 'synthetic':decision['synthetic']}
            features = tuple(sorted({**price['returns'], 'vol_20d': price['vol_20d'],
                                     'last_close': price['recent_closes'][-1]}.items()))
            probability = view["p_outperform"]
            seconds = int((exit_at - at).total_seconds())
            contract = ModelContract(model_id="trader:" + view["analyst"], version="1",
                inputs={"context": "trader-context-v1", "skills": decision["skill_hashes"]},
                outputs={"return": None, "target_price": None, "class": "UP_DOWN_ABSTAIN",
                         "probabilities": "uncalibrated_outperformance_judgment", "quantiles": None, "scenarios": None},
                horizons_seconds=(seconds,), capabilities=("infer",),
                limits={"paper_only": True, "model_versions": decision["models"]},
                implementation_version="trader-agent-v1", synthetic=decision["synthetic"])
            # Horizon is the exact elapsed time from decision to the target endpoint, including weekends.
            prediction = PredictionRecord(prediction_id=identity, model_id="trader:" + view["analyst"],
                model_contract_hash=contract.identity, artifact_hash=decision["preregistration_hash"],
                product=view["asset"], decision_at=decision["decision_at"], horizon_seconds=seconds,
                snapshot_hash=sha256_canonical(snapshot), features_hash=sha256_canonical(features), event_ids=(),
                outputs={"return": None, "target_price": None, "class": view["view"], "probabilities": {"outperform": probability},
                         "quantiles": None, "scenarios": None}, signal={"view": view, "session": decision["session"],
                    "label_definition": {"entry_at": iso(entry), "exit_at": iso(exit_at), "horizon": view["horizon"],
                                         "targets": ["raw", "SPY_relative"] if view["asset"] not in crypto else ["raw"]},
                    "models": decision["models"], "skill_hashes": decision["skill_hashes"], "run_id": decision["run_id"]},
                risk={"mode": "PAPER_SHADOW_ONLY", "verdict": view["verdict"]},
                proposed_position={"weight": proposals[view['analyst']][view["horizon"]]["unhedged"].get(view["asset"], 0)
                                   if view["analyst"] in proposals else 0, "entry_at": iso(entry),
                                   "capital_usd":self.ledger.grant.payload['budgets']['paper_capital_usd'],
                                   "capital_fraction":1 if view['horizon']=='1d' else .2,
                                   "variant":view['analyst']},
                uncertainty={"method": "uncalibrated_analyst_judgment", "calibration_claim": False},
                costs={"equity_bps_per_side": 5, "crypto_bps_per_side": 10, "half_spread_bps": 2.5,
                       "spread_method": "preregistered_fixed_assumption_not_measured"},
                synthetic=decision["synthetic"])
            baselines = {"always_up": 1, "momentum20": 1 if dict(features)["20"] > 0 else 0,
                         "random_seeded": int(hashlib.sha256(identity.encode()).hexdigest()[-1], 16) % 2,
                         "spy_relative_zero": .5}
            evidence = PredictionEvidence(prediction_id=identity, prediction_hash=prediction.identity, recorded_at=recorded,
                features=features, snapshot=snapshot, baselines=baselines, input_quality={"state": "AVAILABLE"},
                split="PROSPECTIVE_SYNTHETIC" if decision["synthetic"] else "PROSPECTIVE", provenance={
                    "authorization_hash": decision["authorization_hash"], "skill_hashes": decision["skill_hashes"],
                    "context_hash":decision['context_hash'], 'context_record_hash':context_record_hash,
                    "sources": {'context_record_hash':context_record_hash, 'count':len(context['sources']),
                                'sources_hash':sha256_canonical(context['sources'])}, "model_contract": contract.to_dict(),
                    "preregistration_hash": decision["preregistration_hash"]})
            self.store.issue(prediction, evidence)
