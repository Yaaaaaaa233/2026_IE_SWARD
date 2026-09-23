"""Synthetic boundary checks for ADR-0007 proxy labels and fold generation."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from tools.build_task1_proxy_labels import build, count_label_events, make_label_rows, make_splits


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class ProxyLabelTests(unittest.TestCase):
    def test_same_day_occurrences_and_exact_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            events = Path(temp) / "events_clean.csv"
            base = {"event_type": "11804", "event_name": "near miss", "start_time": "2026-06-22 08:00:00",
                    "speed": "40", "lat": "1", "lng": "2"}
            rows = [
                {**base, "row_id": "e1", "gpsno": "A"},
                {**base, "row_id": "e2", "gpsno": "A", "start_time": "2026-06-22 09:00:00"},
                {**base, "row_id": "e3", "gpsno": "B"},
                {**base, "row_id": "e4", "gpsno": "B"},  # exact duplicate of e3
                {**base, "row_id": "e5", "gpsno": "C", "event_type": "11803"},
                {**base, "row_id": "e6", "gpsno": "D", "start_time": "2026-07-31 00:00:00"},
            ]
            write_csv(events, ["row_id", "gpsno", "event_type", "event_name", "start_time", "speed", "lat", "lng"], rows)
            samples = [(f"s-{key}", key) for key in "ABCD"]
            counts, duplicates = count_label_events(events, set("ABCD"), date(2026, 6, 21), date(2026, 7, 31))
            labels = make_label_rows(samples, counts, date(2026, 6, 21), date(2026, 7, 31), "2.0")
            self.assertEqual([row["y"] for row in labels], [1, 0, 1, 0])
            self.assertEqual(duplicates, 1)

    def test_split_is_deterministic_balanced_and_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            samples = [(f"sample-{key}", key) for key in "ABCDEFGHIJ"]
            labels = [{"y": int(index < 5)} for index in range(10)]
            first = make_splits(samples, labels)
            self.assertEqual(first, make_splits(samples, labels))
            self.assertEqual(sorted(row["fold"] for row in first[:5]), list(range(5)))
            self.assertEqual(sorted(row["fold"] for row in first[5:]), list(range(5)))

            features = root / "features.csv"
            vehicles = root / "target_vehicles.csv"
            events = root / "events_clean.csv"
            write_csv(features, ["sample_id", "gpsno", "as_of", "source_version"],
                      [{"sample_id": sid, "gpsno": gpsno, "as_of": "2026-06-21", "source_version": "2.0"}
                       for sid, gpsno in samples])
            write_csv(vehicles, ["gpsno"], [{"gpsno": gpsno} for _, gpsno in samples])
            write_csv(events, ["row_id", "gpsno", "event_type", "event_name", "start_time", "speed", "lat", "lng", "source_version"],
                      [{"row_id": f"e{i}", "gpsno": gpsno, "event_type": "11803", "event_name": "accident",
                        "start_time": "2026-06-23 00:00:00", "speed": "", "lat": "", "lng": "",
                        "source_version": "2.0"}
                       for i, (_, gpsno) in enumerate(samples[:5])])
            output = root / "new_protocol"
            build(events, features, vehicles, output, date(2026, 6, 21), date(2026, 7, 31), expected_vehicles=10)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["positive_vehicles"], 5)
            self.assertEqual({tuple(counts.values()) for counts in manifest["fold_counts"].values()}, {(1, 1)})
            with self.assertRaises(FileExistsError):
                build(events, features, vehicles, output, date(2026, 6, 21), date(2026, 7, 31), expected_vehicles=10)
            wrong_source = root / "wrong_source_events.csv"
            with events.open("r", encoding="utf-8", newline="") as handle:
                original = list(csv.DictReader(handle))
            original[0]["source_version"] = "1.0"
            write_csv(wrong_source, list(original[0]), original)
            with self.assertRaisesRegex(ValueError, "source_version"):
                build(wrong_source, features, vehicles, root / "should_not_exist",
                      date(2026, 6, 21), date(2026, 7, 31), expected_vehicles=10)


if __name__ == "__main__":
    unittest.main()
