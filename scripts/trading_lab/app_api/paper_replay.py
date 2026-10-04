"""Bounded, read-only views of frozen replay evidence. Never opens a runtime DB."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.trading_lab.app_api.contracts import (
    AppApiError, ConflictError, DEFAULT_FILL_PAGE, MAX_FILL_PAGE, NotFoundError,
    SUPPORTED_PRODUCTS,
)
from scripts.trading_lab.app_api.pagination import decode_cursor, encode_cursor, require_limit

MAX_ARTIFACT_BYTES = 4 * 1024 * 1024


def _digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class PaperReplayViews:
    def __init__(self, data_root):
        self._root = Path(data_root) / "paper_replay_v2"

    def _read(self, name):
        path = self._root / name
        if not path.is_file():
            return None, None
        with path.open("rb") as source:
            raw = source.read(MAX_ARTIFACT_BYTES + 1)
        if len(raw) > MAX_ARTIFACT_BYTES:
            raise ConflictError("paper replay artifact exceeds its size bound")
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("replay artifacts must be objects")
            return payload, hashlib.sha256(raw).hexdigest()
        except (ValueError, UnicodeDecodeError) as error:
            raise ConflictError("invalid paper replay artifact") from error

    def _manifest(self):
        manifest, _ = self._read("manifest.json")
        if manifest is not None:
            if (_digest({k: v for k, v in manifest.items() if k != "manifest_hash"})
                    != manifest.get("manifest_hash")
                    or manifest.get("schema_version") != "trading-lab.paper-replay-manifest.v2"):
                raise ConflictError("paper replay manifest binding mismatch")
        return manifest

    def _result(self, product, manifest):
        if product not in SUPPORTED_PRODUCTS:
            raise NotFoundError("unknown replay product")
        if manifest is None:
            raise NotFoundError("no frozen paper replay available")
        result, file_hash = self._read(f"{product}.json")
        entry = manifest["products"].get(product)
        if result is None or entry is None:
            raise NotFoundError("no frozen paper replay for this product")
        if (file_hash != entry["file_sha256"] or result.get("product") != product
                or result.get("schema_version") != "trading-lab.paper-replay-result.v2"
                or _digest({k: v for k, v in result.items() if k != "result_hash"}) != entry["result_hash"]
                or result.get("result_hash") != entry["result_hash"]
                or result["hashes"]["corpus_content_hash"] != manifest["corpus_content_hash"]
                or result["hashes"]["paper_model_spec_hash"] != manifest["paper_model_spec_hash"]):
            raise ConflictError("paper replay result binding mismatch")
        return result

    def summary(self):
        manifest = self._manifest()
        if manifest is None:
            return {"available": False, "reason": "no frozen paper replay available", "products": []}
        return {
            "available": True, "reason": None,
            **{key: manifest[key] for key in (
                "experiment_type", "confirmatory", "optimized", "research_evidence",
                "shadow_only", "window", "limitations", "determinism", "manifest_hash")},
            "products": [
                {key: value for key, value in self._result(product, manifest).items()
                 if key not in ("equity_curve", "fills")}
                for product in SUPPORTED_PRODUCTS],
        }

    def page(self, product, leaf, *, limit=None, cursor=None):
        if leaf not in ("equity", "fills"):
            raise NotFoundError("no such replay endpoint")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (str, int))):
            raise AppApiError("limit must be an integer")
        if isinstance(cursor, str) and len(cursor) > 2048:
            raise AppApiError("cursor exceeds its size bound")
        size = require_limit(limit, default=DEFAULT_FILL_PAGE, maximum=MAX_FILL_PAGE)
        manifest = self._manifest()
        stored = self._result(product, manifest)
        field = "equity_curve" if leaf == "equity" else "fills"
        rows = stored[field]
        query = {"version": "v2", "result_hash": stored["result_hash"]}
        endpoint = f"paper_replay_{leaf}"
        if cursor:
            after = decode_cursor(cursor, endpoint=endpoint, product=product, query=query)
            rows = [row for row in rows if row["timestamp"] > after]
        page = rows[:size]
        has_more = len(rows) > size
        return {
            "product": product, "version": "v2", "result_hash": stored["result_hash"],
            "series" if leaf == "equity" else "fills": page,
            "page": {"returned": len(page), "total": len(stored[field]), "has_more": has_more,
                     "next_cursor": encode_cursor(endpoint=endpoint, product=product,
                                                  last_timestamp=page[-1]["timestamp"], query=query)
                     if has_more and page else None},
            **({"metadata": stored["equity_metadata"]} if leaf == "equity" else {}),
        }
