"""Extract three real Task-2 score snapshots; keep frozen g8 separate.

The dynamic ledger is produced by task2/final_rules_v1_dyn/replay.py and is
controlled data. This script never estimates missing snapshots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


CUTOFFS = ("2026-06-21", "2026-07-11", "2026-07-31")
FIELDNAMES = (
    "gpsno",
    "score_20260621",
    "score_20260711",
    "score_20260731",
    "score_change_1_to_3",
    "evidence_flag",
    "rule_version",
    "teacher_version",
    "g8_risk_prob_asof_20260801",
    "g8_model_version",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_unique(path: Path, required: set[str]) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing columns {sorted(required - set(reader.fieldnames or []))}")
        result = {}
        for number, row in enumerate(reader, start=2):
            vehicle = row["gpsno"].strip()
            if not vehicle or vehicle in result:
                raise ValueError(f"{path}:{number}: empty or duplicate gpsno")
            result[vehicle] = row
    if len(result) != 500:
        raise ValueError(f"{path}: expected 500 vehicles, found {len(result)}")
    return result


def score(value: object, source: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid score {value!r}") from exc
    if not math.isfinite(number) or not 0 <= number <= 100:
        raise ValueError(f"{source}: score outside [0, 100]: {number}")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-ledger", type=Path, required=True)
    parser.add_argument("--final-scores", type=Path, required=True)
    parser.add_argument("--g8-forecast", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    static = read_unique(args.final_scores, {"gpsno", "S_final", "evidence_flag", "rule_version"})
    g8 = read_unique(args.g8_forecast, {"gpsno", "risk_prob"})
    if set(static) != set(g8):
        raise ValueError("g8 and Task-2 vehicle sets differ")

    ledger = json.loads(args.daily_ledger.read_text(encoding="utf-8-sig"))
    if ledger.get("policy_version") != "final-v1-dyn-policy-v1":
        raise ValueError("unexpected dynamic policy version")
    daily = ledger.get("daily")
    if not isinstance(daily, dict):
        raise ValueError("ledger has no daily score mapping")
    for cutoff in CUTOFFS:
        if cutoff not in daily:
            raise ValueError(f"ledger missing {cutoff}; replay through --end 2026-07-31")
        if set(daily[cutoff]) != set(static):
            raise ValueError(f"{cutoff}: vehicle set differs from frozen Task-2 score set")

    versions = {row["rule_version"].strip() for row in static.values()}
    if versions != {"task2-final-rules-v1"}:
        raise ValueError(f"unexpected Task-2 rule version(s): {versions}")

    rows = []
    for vehicle in sorted(static):
        values = [score(daily[cutoff][vehicle], f"{cutoff}/{vehicle}") for cutoff in CUTOFFS]
        anchor = score(static[vehicle]["S_final"], f"S_final/{vehicle}")
        if abs(values[0] - anchor) > 0.0005:
            raise ValueError(f"{vehicle}: dynamic anchor {values[0]} differs from frozen score {anchor}")
        probability = float(g8[vehicle]["risk_prob"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"{vehicle}: g8 probability outside [0, 1]")
        rows.append(
            {
                "gpsno": vehicle,
                "score_20260621": f"{values[0]:.3f}",
                "score_20260711": f"{values[1]:.3f}",
                "score_20260731": f"{values[2]:.3f}",
                "score_change_1_to_3": f"{values[2]-values[0]:+.3f}",
                "evidence_flag": static[vehicle]["evidence_flag"],
                "rule_version": "task2-final-rules-v1",
                "teacher_version": "sprint-g10-final",
                "g8_risk_prob_asof_20260801": f"{probability:.17g}",
                "g8_model_version": "g8v6-corrected",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "rows": len(rows),
        "cutoffs_exclusive": list(CUTOFFS),
        "score_windows": ["2026-06-01..2026-06-20", "2026-06-21..2026-07-10", "2026-07-11..2026-07-30"],
        "score_role": "frozen Task-2 dynamic replay; g10 teacher input",
        "g8_role": "2026-08-01 reference only; not used to calculate earlier scores",
        "inputs_sha256": {
            "daily_ledger": sha256(args.daily_ledger),
            "final_scores": sha256(args.final_scores),
            "g8_forecast": sha256(args.g8_forecast),
        },
        "output_sha256": sha256(args.output),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} vehicles to {args.output}")
    print(f"Provenance: {manifest_path}")


if __name__ == "__main__":
    main()
