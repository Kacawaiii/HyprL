"""Prospective run -> independent analysts -> reviewer -> immutable evidence."""
from datetime import timedelta
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
from .scoring import rows

# Operator-approved same-day recovery (2026-10-06), extended 2026-10-07 ~12:45Z:
# "on peut pas reprendre le run mtn on perd trop" authorizes recovery after FAILED even if models were called.
# Every dispatch retains its daily role/aggregate limits; exhausted analysts are MISSING in a degraded catch-up.
# Decide before close minus 80 minutes, enter at the close, score apart.
CATCHUP_VARIANT = "catchup_close_entry_v1"
AFTER_HOURS_VARIANT = 'alpaca_after_hours_v1'
VARIANTS = ["analyst_claude", "analyst_gpt", "reviewer_claude", "reviewer_gpt", "consensus",
            "reviewer_kept", "reviewer_rejected", "reviewer_downgraded", "always_up", "momentum20",
            "random_seeded", "spy_relative_zero", "unhedged", "spy_hedged", CATCHUP_VARIANT,
            'alpaca_open_entry_v1', AFTER_HOURS_VARIANT]
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


# Never degrade on these: they are the run's own limits, not a model failing. Everything else an analyst raises
# (MODEL_FAILED, TAINTED_RUN, SCHEMA_INVALID, SKIPPED_QUOTA, MODEL_TIMEOUT, invalid output...) leaves that analyst MISSING.
# The authorized catch-up handles exhausted analyst roles separately; aggregate and reviewer limits still fail closed.
NEVER_DEGRADE = {"BUDGET_EXHAUSTED", "PAUSED", "MISSED_DECISION_DEADLINE", "MISSED_DECISION_DEADLINE_OR_PAUSED",
                 "AUTHORIZATION_EXPIRED_OR_NOT_STARTED"}
ROLES = ("analyst_claude", "analyst_gpt")


def views_with_review(analysts, reviewer, degraded=None):
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
    for role, code in (degraded or {}).items():
        # The missing analyst and its reviewed twin: recorded as MISSING (never predicted, never scored) with the error code.
        for (asset, horizon) in kept:
            for name in (role, "reviewer_" + role.split("_")[-1]):
                output.append({"asset": asset, "horizon": horizon, "analyst": name, "view": "ABSTAIN", "p_outperform": .5,
                               "verdict": "MISSING", "raw_view": None, "review": None, "error": code})
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
        self.entry_at = "open"

    def summary(self, run_id, status, at, **extra):
        payload = {"schema": "trader-run-v1", "run_id": run_id, "status": status, "at": iso(at),
                   "synthetic": self.data.synthetic, **extra}
        # RUNNING/COMPLETE happen once per run; failures and skips can repeat (e.g. a refused catch-up), each kept.
        suffix = status if status in {"RUNNING", "COMPLETE", "DEGRADED"} else f"{status}:{iso(at)}"
        self.store.append("replay-summary", payload, object_id=run_id + ":" + suffix, recorded_at=iso(at))
        return payload

    def role_exhausted(self, role, counts):
        model = self.ledger.grant.payload['external_models'][role]
        return counts.get(role, 0) >= model['calls_per_day'] + model['retries_per_day']

    def catchup_missing(self, daily_run_id):
        counts = self.ledger.counts()
        statuses = {r['payload'].get('status') for r in rows(self.store, 'replay-summary')
                    if r['payload'].get('schema') == 'trader-run-v1'
                    and r['payload'].get('run_id') == daily_run_id}
        if counts.get('run', 0) < 1 or 'FAILED' not in statuses or statuses & {'COMPLETE', 'DEGRADED'}:
            raise TraderError('CATCHUP_NOT_ALLOWED')
        if any(r['payload'].get('signal', {}).get('run_id') == daily_run_id
               for r in rows(self.store, 'prediction')):
            raise TraderError('CATCHUP_NOT_ALLOWED')
        missing = {role: 'BUDGET_EXHAUSTED' for role in ROLES if self.role_exhausted(role, counts)}
        remaining = self.ledger.grant.payload['budgets']['max_llm_calls_per_day'] - sum(
            counts.get(role, 0) for role in (*ROLES, 'reviewer'))
        # Need at least one valid analyst and a reviewer; retries still reserve against the same immutable bank.
        if (counts.get('catchup_run', 0) or len(missing) == len(ROLES)
                or self.role_exhausted('reviewer', counts) or remaining < len(ROLES) - len(missing) + 1):
            raise TraderError('BUDGET_EXHAUSTED')
        return missing

    def run(self, *, catchup=False, after_hours=False):
        """catchup=True: recover a FAILED daily run within the unchanged daily model budgets. Exhausted analysts stay
        MISSING; decide before close minus 80 minutes, enter at the close, and score CATCHUP_VARIANT separately."""
        if after_hours and not catchup:
            raise TraderError('AFTER_HOURS_REQUIRES_CATCHUP')
        self.entry_at = 'after_hours' if after_hours else "close" if catchup else "open"
        with self.ledger.owner():
            at = self.clock()
            from zoneinfo import ZoneInfo
            day = (at.astimezone(ZoneInfo('America/New_York')) if after_hours else at).date().isoformat()
            previous_budget_day = self.ledger.budget_day
            if after_hours:
                self.ledger.budget_day = day
            run_id = "trader:" + day + (":synthetic" if self.data.synthetic else ":real") + (":catchup" if catchup else "")
            try:
                self.ledger.grant.check(at)
                if self.ledger.paused:
                    return self.summary(run_id, "PAUSED", at)
                if not self.data.synthetic and at < instant("2026-10-06T12:00:00Z"):
                    return self.summary(run_id, "NOT_STARTED", at)
                session = calendar_session(day)
                if not session:
                    return self.summary(run_id, "SKIPPED_HOLIDAY", at)
                degraded = {}
                if catchup:
                    degraded = self.catchup_missing(run_id.removesuffix(':catchup'))
                    from .alpaca_paper import after_hours_cutoff
                    cutoff = after_hours_cutoff(session) if after_hours else session.close_at - timedelta(minutes=80)
                    if (after_hours and at < session.close_at) or at >= cutoff - (timedelta(minutes=80) if after_hours else timedelta()):
                        raise TraderError("MISSED_DECISION_DEADLINE")
                elif at >= session.open_at:
                    raise TraderError("MISSED_DECISION_DEADLINE")
                self.runner.deadline = cutoff if catchup else session.open_at
                prereg_hash = preregistration()
                texts, hashes = skills()
                self.ledger.reserve("catchup_run" if catchup else "run")
                self.summary(run_id, "RUNNING", at, preregistration_hash=prereg_hash, multiple_testing_variants=VARIANTS)
                context, context_hash = build_context(self.ledger.grant, self.data, at=at, fomc=self.fomc, edgar=self.edgar)
                if catchup:   # equities only (crypto has no session close); the analysts are told when positions enter and exit
                    crypto_assets = self.ledger.grant.payload["universe"]["crypto"]
                    context = {**context, "universe": [a for a in context["universe"] if a not in crypto_assets],
                               "exclusions": context["exclusions"] + [{"asset": a, "reason": "CATCHUP_EQUITIES_ONLY"}
                                                                      for a in context["universe"] if a in crypto_assets],
                               "label_convention": {"variant": AFTER_HOURS_VARIANT if after_hours else CATCHUP_VARIANT,
                               "entry": "actual after-hours paper fill; unfilled is not traded" if after_hours else "today's session close",
                               "1d_exit": "close of the next session", "5d_exit": "close five sessions later"}}
                    context_hash = sha256_canonical(context)
                self.store.append('replay-summary', {'schema':'trader-context-evidence-v1', 'run_id':run_id,
                    'context_hash':context_hash, 'context':context}, object_id=run_id + ':context',
                    recorded_at=context['decision_time'])
                if context["exclusions"]:
                    self.ledger.alert("PARTIAL_CONTEXT")
                context_at = instant(context["decision_time"])
                entry_deadline = cutoff if after_hours else min(label_window(a, day, '1d', self.ledger.grant.payload['universe']['crypto'], self.entry_at)[0]
                                     for a in context['universe'])
                self.runner.deadline = min(self.runner.deadline, entry_deadline)
                analysts, last = {}, None
                # Neither prompt contains the other analyst's output. Serial to bound host memory.
                for role in ROLES:
                    self.before_call(session)
                    if role in degraded:
                        continue   # no CLI/version request or reservation for an exhausted catch-up analyst
                    try:
                        analysts[role] = self.runner.infer(role, prompt(context, texts["TRADER_SKILL.md"]),
                            validator=lambda output: validate_analyst(output, context["universe"], context_at))
                    except TraderError as error:
                        exhausted_catchup_role = (catchup and error.code == 'BUDGET_EXHAUSTED'
                                                  and self.role_exhausted(role, self.ledger.counts()))
                        if error.code in NEVER_DEGRADE and not exhausted_catchup_role:
                            raise
                        degraded[role], last = error.code, error
                if not analysts:
                    raise last   # both analysts failed: nothing to review, the run fails as before
                self.before_call(session)
                reviewer = self.runner.infer("reviewer", prompt(context, texts["REVIEWER_SKILL.md"], analysts=analysts),
                    validator=lambda output: validate_reviewer(output, analysts))
                decision_at = self.clock()
                if decision_at >= self.runner.deadline:
                    raise TraderError("MISSED_DECISION_DEADLINE")
                decision = {"schema": "trader-decision-v1", "run_id": run_id, "session": day,
                    "decision_at": iso(decision_at), "context_hash": context_hash,
                    "authorization_hash": self.ledger.grant.identity, "preregistration_hash": prereg_hash,
                    "skill_hashes": hashes, "synthetic": self.data.synthetic,
                    "models": {**{role: {"model": self.ledger.grant.gpt_model if role == "analyst_gpt" else
                                        self.ledger.grant.payload["external_models"][role]["model"],
                                        "cli_version": "NOT_RUN", "reported_version": "NOT_RUN"} for role in degraded},
                               **self.runner.metadata},
                    "views": views_with_review(analysts, reviewer, degraded)}
                validate("decision", decision)
                self.record(decision, context)
                return self.summary(run_id, "DEGRADED" if degraded else "COMPLETE", decision_at, decision=decision,
                    **({"degraded": degraded} if degraded else {}),
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
            finally:
                self.ledger.budget_day = previous_budget_day

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
            if view["verdict"] == "MISSING":
                continue   # a failed analyst has no forecast: it stays visible in the decision, never predicted or scored
            entry, exit_at = label_window(view["asset"], decision["session"], view["horizon"], crypto,
                                          'close' if self.entry_at == 'after_hours' else self.entry_at)
            at = instant(decision["decision_at"])
            if self.entry_at == 'after_hours':
                from .alpaca_paper import after_hours_cutoff
                entry = at
                if instant(recorded) >= after_hours_cutoff(calendar_session(decision['session'])):
                    raise TraderError('MISSED_RECORDING_DEADLINE')
            elif instant(recorded) >= entry:
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
                                         "targets": ["raw", "SPY_relative"] if view["asset"] not in crypto else ["raw"],
                                         **({"entry_price": "alpaca_actual_fill", "variant": AFTER_HOURS_VARIANT} if self.entry_at == 'after_hours'
                                            else {"entry_price": "close", "variant": CATCHUP_VARIANT} if self.entry_at == "close" else {})},
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
