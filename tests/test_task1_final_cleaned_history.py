import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from models.task1_final.extend_cleaned_history import (
    DATES,
    aggregate_sensor_stream,
    day_rows,
    event_segments,
    verify_control_day,
)


class CleanedHistoryTests(unittest.TestCase):
    def test_same_type_thirty_second_segments(self):
        events = pd.DataFrame(
            [
                ("car-a", "11804", "2026-07-31 01:00:00"),
                ("car-a", "11804", "2026-07-31 01:00:30"),
                ("car-a", "11804", "2026-07-31 01:01:01"),
                ("car-a", "11803", "2026-07-31 01:00:02"),
                ("car-b", "11804", "2026-07-31 01:00:00"),
            ],
            columns=["gpsno", "event_type", "start_time"],
        )
        result = event_segments(events).set_index("gpsno")
        self.assertEqual(result.loc["car-a", "evt_raw_count"], 4)
        self.assertEqual(result.loc["car-a", "evt_segments30"], 3)
        self.assertEqual(result.loc["car-b", "evt_segments30"], 1)

    def test_day_formula_and_control_rejects_bad_overlap(self):
        key = ("car-a", DATES[0])
        trajectory = {key: [12.345678, 2.125, 120, 1000, 2200]}
        imu = {key: [30, 1100, 2100]}
        counts = pd.DataFrame([{"gpsno": "car-a", "evt_raw_count": 3, "evt_segments30": 2}])
        day = day_rows(DATES[0], trajectory, imu, counts)
        self.assertEqual(day.iloc[0].traj_km, 12.346)
        self.assertEqual(day.iloc[0].traj_observed_seconds, 120)
        self.assertEqual(day.iloc[0].imu_observed_seconds, 60)
        self.assertEqual(day.iloc[0].imu_traj_overlap_seconds, 1000)
        self.assertEqual(verify_control_day(day, day.copy())["km_rounding_rows"], 0)
        near = day.copy()
        near.loc[0, "traj_km"] += 0.001
        self.assertEqual(verify_control_day(day, near)["km_rounding_rows"], 1)
        far = day.copy()
        far.loc[0, "traj_km"] += 0.002
        with self.assertRaisesRegex(ValueError, "traj_km"):
            verify_control_day(day, far)
        corrupt = day.copy()
        corrupt.loc[0, "imu_traj_overlap_seconds"] = 1001
        with self.assertRaisesRegex(ValueError, "imu_traj_overlap_seconds"):
            verify_control_day(day, corrupt)

    def test_sensor_scan_accepts_new_day_retained_with_old_window_flag(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "traj-part-00000"
            trajectory.write_text(
                "car-a\t100000\t3600\t2026-07-31 01:00:00\t20\t0\t30\t115\tok\tok\tok\tok\t0\t2.0\n"
                "car-a\t0\t0\t2026-07-31 01:00:03\t20\t0\t30\t115\tok\tok\tok\tok\t0\t2.0\n"
                "car-a\t0\t0\t2026-07-31 01:05:03\t0\t0\t30\t115\tok\tok\tok\tok\t0\t2.0\n"
            )
            imu = root / "imu-part-00000"
            imu.write_text(
                "car-a\tdevice\t2026-07-31 01:00:02\t20260731\t20\t20\t0\t0\t1\t0\t0\t0\tok\tok\tok\tok\t0\t2.0\n"
            )
            t = aggregate_sensor_stream([trajectory], "trajectory", {"car-a"})
            m = aggregate_sensor_stream([imu], "imu", {"car-a"})
            self.assertEqual(t[("car-a", DATES[1])][:3], [1.0, 1.0, 3])
            self.assertEqual(m[("car-a", DATES[1])][0], 1)
            trajectory.write_text(trajectory.read_text().replace("\t0\t2.0", "\t1\t2.0"))
            with self.assertRaisesRegex(RuntimeError, "window flags"):
                aggregate_sensor_stream([trajectory], "trajectory", {"car-a"})


if __name__ == "__main__":
    unittest.main()
