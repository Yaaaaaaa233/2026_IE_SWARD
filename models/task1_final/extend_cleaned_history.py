"""Extend the frozen task-1 model inputs through the official July 31 history.

This emits a *model-input projection*, not a replacement v4_assessed data-line
release.  The event output contains exactly the fields consumed by the frozen
g8 feature builders.  The vehicle-day output reproduces the existing v4 table
formula, checked against July 30 before July 31 is appended.  Source files are
read only; all output stays in a controlled, git-ignored directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


EVENT_COLS = ["row_id", "gpsno", "event_type", "start_time", "speed"]
DAY_COLS = [
    "gpsno", "date", "evt_raw_count", "evt_segments30", "evt_days_flag",
    "traj_km", "traj_hours", "traj_observed_seconds", "imu_observed_seconds",
    "imu_traj_overlap_seconds", "quality_flags", "source_version",
]
DATES = ("2026-07-30", "2026-07-31")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def event_segments(events: pd.DataFrame) -> pd.DataFrame:
    """Same-type alarms at most 30 seconds apart form one segment per day."""
    if events.empty:
        return pd.DataFrame(columns=["gpsno", "evt_raw_count", "evt_segments30"])
    e = events[["gpsno", "event_type", "start_time"]].copy()
    e["t"] = pd.to_datetime(e.start_time, errors="raise")
    e = e.sort_values(["gpsno", "event_type", "t"], kind="mergesort")
    gap = e.groupby(["gpsno", "event_type"], sort=False).t.diff().dt.total_seconds()
    e["new_segment"] = (gap.isna() | (gap > 30)).astype(int)
    return e.groupby("gpsno", as_index=False).agg(
        evt_raw_count=("start_time", "size"),
        evt_segments30=("new_segment", "sum"),
    )


def aggregate_sensor_stream(shards: list[Path], kind: str, roster: set[str]) -> dict:
    """Stream retained v1 records through rg/awk; keep only two daily aggregates."""
    if not shards:
        raise ValueError(f"no {kind} shards")
    if kind not in ("trajectory", "imu"):
        raise ValueError(f"unknown sensor kind: {kind}")
    time_col = 4 if kind == "trajectory" else 3
    flag_col = 13 if kind == "trajectory" else 17
    # Time-of-day seconds suffice because each key is one vehicle and one date.
    # The original v4 overlap is max(0, min(last)-max(first)) on that date.
    body = (
        'd=substr($%d,1,10); if(d!="2026-07-30" && d!="2026-07-31") next; '
        'if($%d != (d=="2026-07-30" ? "1" : "0")) {bad++; next}; '
        'k=$1 SUBSEP d; n[k]++; '
        's=substr($%d,12,2)*3600+substr($%d,15,2)*60+substr($%d,18,2); '
        'if(!(k in lo) || s<lo[k]) lo[k]=s; if(!(k in hi) || s>hi[k]) hi[k]=s; '
    ) % (time_col, flag_col, time_col, time_col, time_col)
    if kind == "trajectory":
        body += (
            'if(k in prev){gap=s-prev[k]; if(gap<0) disorder++; else if(gap<300) obs[k]+=gap} '
            'prev[k]=s; '
            'if($2!="" && $2!="\\\\N") dist[k]+=$2; '
            'if($3!="" && $3!="\\\\N") run[k]+=$3; '
        )
        ending = ('for(k in n){split(k,a,SUBSEP); '
                  'printf "%s\\t%s\\t%.12f\\t%.12f\\t%d\\t%d\\t%d\\n", '
                  'a[1],a[2],dist[k],run[k],obs[k],lo[k],hi[k]}')
    else:
        ending = ('for(k in n){split(k,a,SUBSEP); '
                  'printf "%s\\t%s\\t%d\\t%d\\t%d\\n", '
                  'a[1],a[2],n[k],lo[k],hi[k]}')
    program = ('BEGIN{FS="\\t"} {' + body + '} END{' + ending +
               'if(bad){print "bad window flags: " bad > "/dev/stderr"; exit 2} '
               'if(disorder){print "out-of-order sensor timestamps: " disorder > "/dev/stderr"; exit 3}}')
    rg_cmd = ["rg", "--no-filename", "-F", "-e", DATES[0], "-e", DATES[1], "--"] + [str(p) for p in shards]
    rg_proc = subprocess.Popen(rg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert rg_proc.stdout is not None
    awk_proc = subprocess.Popen(["awk", program], stdin=rg_proc.stdout,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rg_proc.stdout.close()
    stdout, awk_err = awk_proc.communicate()
    rg_err = rg_proc.stderr.read() if rg_proc.stderr else b""
    rg_code = rg_proc.wait()
    if rg_proc.stderr:
        rg_proc.stderr.close()
    if rg_code != 0 or awk_proc.returncode != 0:
        raise RuntimeError(
            f"{kind} scan failed: rg={rg_code}, awk={awk_proc.returncode}; "
            f"{(rg_err + awk_err).decode('utf-8', errors='replace')[:500]}"
        )
    out = {}
    for line in stdout.decode("utf-8").splitlines():
        fields = line.split("\t")
        gno, day = fields[:2]
        if gno not in roster:
            raise ValueError(f"{kind} day outside fixed roster")
        if kind == "trajectory":
            out[(gno, day)] = [
                float(fields[2]) / 100000.0, float(fields[3]) / 3600.0,
                int(fields[4]), int(fields[5]), int(fields[6]),
            ]
        else:
            out[(gno, day)] = [int(x) for x in fields[2:]]
    print(f"{kind} stream scanned: {len(shards)} shards, {len(out)} vehicle-days", flush=True)
    return out


def day_rows(
    day: str, trajectory: dict, imu: dict, event_counts: pd.DataFrame,
) -> pd.DataFrame:
    counts = event_counts.set_index("gpsno").to_dict("index") if len(event_counts) else {}
    keys = sorted(k for k in (set(trajectory) | set(imu)) if k[1] == day)
    rows = []
    for gno, _ in keys:
        t = trajectory.get((gno, day))
        m = imu.get((gno, day))
        evt = counts.get(gno, {})
        overlap = max(0, min(t[4], m[2]) - max(t[3], m[1])) if t and m else 0
        rows.append([
            gno, day, int(evt.get("evt_raw_count", 0)),
            int(evt.get("evt_segments30", 0)), 1,
            round(t[0], 3) if t else 0.0,
            round(t[1], 4) if t else 0.0,
            t[2] if t else 0, 2 * m[0] if m else 0, overlap,
            "" if m else "imu_not_observed", "2.0",
        ])
    return pd.DataFrame(rows, columns=DAY_COLS)


def verify_control_day(rebuilt: pd.DataFrame, old: pd.DataFrame) -> dict:
    reference = old[old.date == DATES[0]].sort_values("gpsno").reset_index(drop=True)
    actual = rebuilt.sort_values("gpsno").reset_index(drop=True)
    if len(actual) != len(reference) or actual.gpsno.tolist() != reference.gpsno.tolist():
        raise ValueError("July 30 vehicle-day keys do not reproduce v4")
    km_diff = (pd.to_numeric(actual.traj_km) - pd.to_numeric(reference.traj_km)).abs()
    # The archived v4 aggregator accumulated per-record km as binary floats.
    # The vectorized scan sums centimeters first; exact half-way values may
    # differ by one final displayed metre, with no underlying distance change.
    if (km_diff > 0.001000001).any():
        raise ValueError("July 30 vehicle-day field differs from v4: traj_km")
    for col in DAY_COLS[2:]:
        if col == "traj_km":
            continue
        if col == "quality_flags":
            ok = actual[col].fillna("").tolist() == reference[col].fillna("").tolist()
        elif col == "source_version":
            ok = set(actual[col].astype(str)) == {"2.0"}
        else:
            ok = ((pd.to_numeric(actual[col]) - pd.to_numeric(reference[col])).abs() < 1e-8).all()
        if not ok:
            raise ValueError(f"July 30 vehicle-day field differs from v4: {col}")
    return {"km_rounding_rows": int((km_diff > 1e-8).sum()),
            "max_km_difference": float(km_diff.max())}


def build(args: argparse.Namespace) -> None:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    old = pd.read_csv(args.vehicle_day, dtype={"gpsno": str, "source_version": str})
    if old.date.max() != DATES[0] or list(old.columns) != DAY_COLS:
        raise ValueError("expected frozen v4 vehicle_day ending July 30")
    roster = set(old.gpsno)
    if len(roster) != 500:
        raise ValueError("vehicle-day source must contain the fixed 500 vehicles")
    quarantine = pd.read_csv(
        args.quarantine, dtype={"gpsno": str, "event_type": str}, encoding="utf-8-sig"
    )
    new_events = quarantine[quarantine.exclude_reason == "out_of_window_0731"].copy()
    if new_events.empty or not set(new_events.gpsno).issubset(roster):
        raise ValueError("July 31 quarantine rows missing or outside target roster")
    if not new_events.start_time.str.startswith(DATES[1]).all():
        raise ValueError("quarantine contains events outside July 31")
    if new_events.drop(columns=["exclude_reason"]).duplicated().any():
        raise ValueError("July 31 quarantine has exact duplicate event rows")
    bad_speed = (new_events.speed < 0) | (new_events.speed > 200)
    new_events.loc[bad_speed, "speed"] = pd.NA
    counts31 = event_segments(new_events)

    # The July 30 event count/segment rule is independently verified against
    # existing v4 vehicle_day before extending the model input.
    prior = []
    old_count = 0
    max_id = 0
    for chunk in pd.read_csv(
        args.events, usecols=EVENT_COLS,
        dtype={"gpsno": str, "event_type": str, "row_id": str},
        encoding="utf-8-sig", chunksize=300_000,
    ):
        old_count += len(chunk)
        max_id = max(max_id, int(chunk.row_id.iloc[-1].split("_")[1]))
        prior.append(chunk[chunk.start_time.str.startswith(DATES[0])])
    if old_count != max_id:
        raise ValueError("frozen event IDs are not sequential through source end")
    counts30 = event_segments(pd.concat(prior, ignore_index=True))
    print(f"frozen events read: {old_count}; July 31 events: {len(new_events)}", flush=True)

    traj = aggregate_sensor_stream(sorted(args.trajectory_dir.glob("part-*")), "trajectory", roster)
    imu = aggregate_sensor_stream(sorted(args.imu_dir.glob("part-*")), "imu", roster)
    day30 = day_rows(DATES[0], traj, imu, counts30)
    control = verify_control_day(day30, old)
    print(f"July 30 control: {len(day30)} rows; km rounding rows {control['km_rounding_rows']}", flush=True)
    day31 = day_rows(DATES[1], traj, imu, counts31)
    if day31.empty:
        raise ValueError("no July 31 vehicle-day rows")
    days61 = pd.concat([old, day31], ignore_index=True)
    if days61.duplicated(["gpsno", "date"]).any():
        raise ValueError("vehicle-day duplicate key")
    days_path = output / "vehicle_day_61d.csv"
    days61.to_csv(days_path, index=False, lineterminator="\n")

    new_events = new_events.reset_index(drop=True)
    new_events.insert(0, "row_id", [f"evt_{i:08d}" for i in range(max_id + 1, max_id + 1 + len(new_events))])
    events_path = output / "events_model_61d.csv"
    with events_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(EVENT_COLS)
        for chunk in pd.read_csv(
            args.events, usecols=EVENT_COLS,
            dtype={"gpsno": str, "event_type": str, "row_id": str},
            encoding="utf-8-sig", chunksize=300_000,
        ):
            chunk.to_csv(f, index=False, header=False, lineterminator="\n")
        new_events[EVENT_COLS].to_csv(f, index=False, header=False, lineterminator="\n")

    manifest = {
        "role": "task1_model_input_projection",
        "full_v4_release": False,
        "history_start": "2026-06-01",
        "history_end_exclusive": "2026-08-01",
        "july30_control_rows": len(day30),
        "july30_control": "exact_except_at_most_0.001km_rounding",
        "july30_control_detail": control,
        "july31_event_rows": len(new_events),
        "july31_event_vehicles": int(new_events.gpsno.nunique()),
        "july31_vehicle_day_rows": len(day31),
        "july31_sensor_days": {
            "trajectory": sum(k[1] == DATES[1] for k in traj),
            "imu": sum(k[1] == DATES[1] for k in imu),
        },
        "input_sha256": {
            "v4_events": sha256(args.events),
            "v4_vehicle_day": sha256(args.vehicle_day),
            "quarantine": sha256(args.quarantine),
            "cleaning_manifest": sha256(args.cleaning_manifest),
        },
        "output_sha256": {events_path.name: sha256(events_path), days_path.name: sha256(days_path)},
        "limits": "Event cross-source v4 annotation columns are not reproduced; frozen g8 builders consume only the five emitted event columns.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("july31_event_rows", "july31_vehicle_day_rows", "july31_sensor_days")}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("events", "vehicle_day", "quarantine", "trajectory_dir", "imu_dir", "cleaning_manifest", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
