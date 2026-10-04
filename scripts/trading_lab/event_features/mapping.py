"""Closed, evidence-bound product/source mapping. Never infer an issuer."""

import json
from pathlib import Path

from scripts.trading_lab.instrument_registry import CATALOGUE_V1
from scripts.trading_lab.sources.canonical import sha256_canonical

MAPPING = json.loads(Path(__file__).with_name("mapping.json").read_text())
MAPPING_HASH = sha256_canonical(MAPPING)
PRODUCTS = tuple(row["product"] for row in MAPPING["products"])


def product_mapping(product: str) -> dict:
    if product in ("AAPL", "MSFT", "NVDA", "QQQ"):
        product = "xnas:" + product
    symbol = CATALOGUE_V1.resolve(product).instrument_id.symbol
    return next(row for row in MAPPING["products"] if row["product"] == symbol)


def edgar_state(mapping: dict) -> str | None:
    if mapping["edgar"] == "UNVERIFIED":
        return "UNKNOWN_MAPPING"
    if mapping["edgar"] == "NO_SEC_ISSUER" or mapping.get("edgar_8k") == "NOT_APPLICABLE":
        return "NOT_APPLICABLE"
    return None
