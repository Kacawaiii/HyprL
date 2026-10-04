"""Qualification of the EDGAR contract's unverified points (UV1-UV6) from a fixture store, offline and
read-only. Each point gets its evidence, its reach and a verdict that keeps three things apart: what the
official documentation says, what these fixtures show, and what stays unknown. A handful of responses
never proves a semantic, a universal absence or a behaviour that was not provoked; those rules stay.

    python -m scripts.trading_lab.edgar.qualify --store DIR [--out FILE]
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from scripts.trading_lab.edgar import spec
from scripts.trading_lab.edgar.store import EdgarStore
from scripts.trading_lab.sources.httpclock import parse_http_date, parse_iso

_ACCEPTANCE_SHAPES = (
    ("YYYY-MM-DDTHH:MM:SS.000Z", re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.000Z$")),
    ("YYYY-MM-DDTHH:MM:SS.fffZ", re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")),
    ("YYYY-MM-DDTHH:MM:SSZ", re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")),
    ("YYYY-MM-DDTHH:MM:SS+HH:MM", re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?[+-]\d{2}:\d{2}$")),
)


def _shape(value: str) -> str:
    return next((name for name, pattern in _ACCEPTANCE_SHAPES if pattern.match(value)), "OTHER")


def qualify(store: EdgarStore) -> dict:
    view = store.view()
    responses = view.rows("RESPONSE")
    attempts = view.rows("ATTEMPT_OUTCOME")
    listings, columns, types, forms, acceptance, amendments, ciks = [], Counter(), {}, Counter(), Counter(), 0, {}
    for resp in responses:
        doc = json.loads(store.read_raw(resp.body["raw_sha"]).decode("utf-8"))
        recent = doc.get("filings", {}).get("recent", {})
        listings.append(resp.seq)
        ciks[resp.body["cik"]] = {"listed_cik": doc.get("cik"), "name": doc.get("name"),
                                  "rows": len(recent.get("accessionNumber", [])),
                                  "older_pages": len(doc.get("filings", {}).get("files", []))}
        columns.update(recent.keys())
        for name, column in recent.items():
            kinds = sorted({type(v).__name__ for v in column}) if isinstance(column, list) else [type(column).__name__]
            types.setdefault(name, set()).update(kinds)
        forms.update(recent.get("form", []))
        acceptance.update(_shape(v) for v in recent.get("acceptanceDateTime", []))
        amendments += sum(1 for f in recent.get("form", []) if f.endswith("/A"))
    n = len(responses)
    statuses = Counter(a.body["outcome"] for a in attempts)
    verdicts = Counter(r.body["verdict"] for r in responses)
    offsets = []
    for r in responses:
        if len(r.body["date_lines"]) == 1 and parse_http_date(r.body["date_lines"][0]):
            offsets.append(round((parse_iso(r.body["wall_at_receipt"]) - parse_http_date(r.body["date_lines"][0])).total_seconds(), 3))
    header_names = Counter(name.lower() for r in responses for name, _value in r.body.get("header_lines", []))
    required_ok = all(columns[c] == n for c in spec.REQUIRED_COLUMNS) and n > 0
    expected_types = {**{c: ["str"] for c in spec.REQUIRED_COLUMNS + spec.TEXT_COLUMNS},
                      **{c: ["int"] for c in spec.INTEGER_COLUMNS}}
    type_mismatch = {c: sorted(types[c]) for c in expected_types if c in types and sorted(types[c]) != expected_types[c]}
    unknown_columns = sorted(set(columns) - set(spec.FIELD_OF))
    link_columns = [c for c in columns if re.search(r"amend|parent|original|related", c, re.I)]
    reach = f"{n} listing responses of {len(ciks)} CIK(s) ({', '.join(sorted(ciks))})"
    matrix = {
        "UV1": {"unknown": "column names and types of filings.recent",
                "documentation": "not documented (only the URL, the window and the files array are)",
                "observation": {"columns_in_every_listing": sorted(c for c in columns if columns[c] == n),
                                "columns_not_in_spec": unknown_columns, "type_mismatches": type_mismatch,
                                "required_present_in_all": required_ok},
                "reach": reach,
                "verdict": "OBSERVED_COMPATIBLE" if required_ok and not type_mismatch else "OBSERVED_INCOMPATIBLE",
                "rule_kept": "the parser still refuses any other shape (PARSER_FAILED); compatibility is observed, not guaranteed"},
        "UV2": {"unknown": "format and time zone of acceptanceDateTime",
                "documentation": "not documented",
                "observation": {"shapes": dict(acceptance)},
                "reach": reach,
                "verdict": "FORMAT_OBSERVED_SEMANTICS_UNKNOWN",
                "rule_kept": "verbatim text only, never parsed into an instant or used for availability or ordering: a 'Z' "
                             "suffix in the text does not prove the instant it names"},
        "UV3": {"unknown": "whether and when a removed filing leaves filings.recent",
                "documentation": "SEC staff may authorize post-acceptance removals and corrections (VF10); the listing "
                                 "behaviour is not described",
                "observation": {"absences_recorded": len(view.rows("FILING_ABSENCE")), "removal_provoked": False},
                "reach": reach,
                "verdict": "UNKNOWN_PERSISTS",
                "rule_kept": "only ABSENT_FROM_LISTING inside the documented window; never a confirmed removal or a deletion"},
        "UV4": {"unknown": "the response to a request above the fair-access rate",
                "documentation": "10 requests/second maximum; the SEC may limit rates (VF6, VF7); no status or duration given",
                "observation": {"attempt_outcomes": dict(statuses), "throttling_provoked": False},
                "reach": reach,
                "verdict": "UNKNOWN_PERSISTS",
                "rule_kept": "403/429 are SOURCE_THROTTLED; a trial stops; no request of any CIK for at least 600 s"},
        "UV5": {"unknown": "presence and accuracy of the HTTP Date header",
                "documentation": "not documented",
                "observation": {"clock_verdicts": dict(verdicts), "date_minus_receipt_seconds": offsets,
                                "responses_with_one_valid_date": len(offsets)},
                "reach": reach,
                "verdict": "OBSERVED_PRESENT_AND_WITHIN_TOLERANCE" if offsets and len(offsets) == n
                           and verdicts.get("CLOCK_VERIFIED", 0) == n else "OBSERVED_INCOMPLETE",
                "rule_kept": "a response without a valid Date or outside 90 s stays CLOCK_UNVERIFIED and attests nothing"},
        "UV6": {"unknown": "whether an amendment names the accession it amends in the listing",
                "documentation": "not documented",
                "observation": {"amendment_rows": amendments, "columns_that_could_link": link_columns},
                "reach": reach,
                "verdict": "NO_LINK_COLUMN_OBSERVED_UNIVERSALITY_UNKNOWN" if not link_columns else "LINK_COLUMN_OBSERVED",
                "rule_kept": "no parent is inferred: an 8-K/A stays its own filing (amends: null)"},
    }
    return {
        "spec_revision": spec.SPEC_REVISION, "spec_hash": spec.SPEC_HASH, "store_schema": spec.SCHEMA_VERSION,
        "listings": n, "ciks": ciks, "attempts": len(view.rows("TRANSPORT_INVOKED")), "attempt_outcomes": dict(statuses),
        "forms_seen": dict(forms.most_common()), "header_names": dict(sorted(header_names.items())),
        "fetch_seconds": [r.body.get("fetch_seconds") for r in responses],
        "raw_sha256": [r.body["raw_sha"] for r in responses], "matrix": matrix,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = qualify(EdgarStore(args.store, wall_clock=None, read_only=True))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
