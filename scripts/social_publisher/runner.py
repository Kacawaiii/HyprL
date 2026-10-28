"""Daily orchestration. Dry runs cannot construct a network transport."""

from __future__ import annotations

import fcntl
from datetime import datetime
from pathlib import Path

from .adapters import ADAPTERS, x_text
from .core import (
    FACTORY_REVISION,
    PLATFORMS,
    UTC,
    Blocked,
    Ledger,
    load_bundle,
    private_dir,
    session,
)
from .http import Credentials, Grant, Transport
from .media import MediaPool


def default_config():
    home = Path.home()
    return {
        "schema": "hyprl-social-publisher-v1",
        "dry_run": True,
        "factory_revision": FACTORY_REVISION,
        "factory_root": str(home / "data" / "videos"),
        "factory_state": str(home / "data" / "video-factory-v1" / "state"),
        "state_dir": str(home / "data" / "social-publisher-v1" / "runtime"),
        "authorization": str(home / "authorizations" / "social-publisher-v1.json"),
        "platforms": {
            p: {
                "enabled": False,
                "account_id": "",
                **(
                    {"graph_version": "v25.0"}
                    if p in ("instagram", "facebook")
                    else {"zero_cost_write_access_confirmed": False, "monthly_post_cap": 25}
                    if p == "x"
                    else {"privacy": "private", "audit_approved": False}
                    if p == "youtube"
                    else {"mode": "private"}
                ),
            }
            for p in PLATFORMS
        },
    }


def validate_config(config):
    if (
        config.get("schema") != "hyprl-social-publisher-v1"
        or config.get("factory_revision") != FACTORY_REVISION
        or type(config.get("dry_run")) is not bool
        or not isinstance(config.get("platforms"), dict)
        or set(config["platforms"]) - set(PLATFORMS)
    ):
        raise Blocked("unsupported_config_or_factory_revision")
    for options in config["platforms"].values():
        if not isinstance(options, dict) or type(options.get("enabled")) is not bool:
            raise Blocked("invalid_platform_switch")
        if options["enabled"] and (
            not isinstance(options.get("account_id"), str) or not options["account_id"]
        ):
            raise Blocked("platform_account_required")


def preflight(platform, bundle, options):
    if platform in ("instagram", "facebook") and len(bundle.caption) > 2200:
        raise Blocked("meta_caption_limit")
    if platform == "x":
        if options.get("zero_cost_write_access_confirmed") is not True:
            raise Blocked("x_zero_cost_access_unproven")
        cap = options.get("monthly_post_cap", 25)
        if type(cap) is not int or not 1 <= cap <= 25:
            raise Blocked("x_monthly_cap_invalid")
        x_text(bundle)
    if platform == "tiktok":
        # The pinned factory renderer always burns in logo/site/promotional text.
        # A human consent record cannot override that incompatible source contract.
        raise Blocked("factory_v2_branding_incompatible_with_tiktok_guidelines")


def daily(
    config,
    *,
    now=None,
    dry_run=False,
    loader=load_bundle,
    transport_factory=Transport,
    credential_factory=Credentials,
):
    validate_config(config)
    now = now or datetime.now(UTC)
    day = session(now)
    if day is None:
        return {"status": "skipped", "reason": "outside_current_nyse_session_after_factory"}
    enabled = {p: o for p, o in config["platforms"].items() if o["enabled"]}
    if not enabled:
        return {"status": "disabled", "day": day}
    bundle = loader(config["factory_root"], config["factory_state"], day)
    is_dry = dry_run or config["dry_run"]
    results = {}
    if is_dry:
        for platform, options in enabled.items():
            try:
                preflight(platform, bundle, options)
                results[platform] = {
                    "status": "would_publish",
                    **(
                        {
                            "privacy": options.get("privacy", "private")
                            if options.get("audit_approved") is True
                            else "private"
                        }
                        if platform == "youtube"
                        else {}
                    ),
                }
            except Blocked as error:
                results[platform] = {"status": "blocked", "reason": str(error)}
        return {
            "status": "dry_run",
            "day": day,
            "sha256": bundle.sha256,
            "metadata_sha256": bundle.metadata_sha256,
            "platforms": results,
        }
    state = private_dir(config["state_dir"])
    lockpath = state / "publisher.lock"
    if lockpath.is_symlink():
        raise Blocked("invalid_lock_file")
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = Ledger(state)
        try:
            for platform, options in enabled.items():
                try:
                    previous = ledger.status(platform, bundle.sha256)
                    if previous == "posted":
                        results[platform] = {"status": "already_posted"}
                        continue
                    if previous is not None:
                        raise Blocked("previous_attempt_needs_reconciliation")
                    preflight(platform, bundle, options)
                    account = options["account_id"]
                    grant = Grant(config["authorization"], ledger, platform, account)
                    if (
                        platform == "x"
                        and grant.value.get("x_zero_cost_write_access_confirmed") is not True
                    ):
                        raise Blocked("x_zero_cost_access_not_authorized")
                    # Validate credentials before reserving; a missing secret is not a network attempt.
                    credentials = credential_factory(platform)
                    if not ledger.reserve(
                        platform, bundle, account, monthly_cap=options.get("monthly_post_cap", 25)
                    ):
                        results[platform] = {"status": "already_posted"}
                        continue
                    http = transport_factory(platform, grant)
                    token = credentials.token(http)

                    def checkpoint(stage, remote_id=None, p=platform):
                        ledger.checkpoint(p, bundle.sha256, stage, remote_id)

                    adapter_options = dict(options)
                    if platform == "tiktok":
                        adapter_options["media_url"] = MediaPool(state / "media").issue(bundle)
                    adapter = ADAPTERS[platform](
                        platform, http, token, account, adapter_options, checkpoint
                    )
                    remote = adapter.publish(bundle)
                    ledger.finish(platform, bundle.sha256, remote)
                    results[platform] = {"status": "posted", "remote_id": remote}
                except Blocked as error:
                    results[platform] = {"status": "blocked", "reason": str(error)}
                except Exception:
                    # No exception values, response bodies, headers, URLs or tokens in logs.
                    results[platform] = {
                        "status": "blocked",
                        "reason": "unexpected_adapter_failure_reconcile_ledger",
                    }
        finally:
            ledger.db.close()
    return {
        "status": "blocked"
        if any(v["status"] == "blocked" for v in results.values())
        else "complete",
        "day": day,
        "sha256": bundle.sha256,
        "platforms": results,
    }
