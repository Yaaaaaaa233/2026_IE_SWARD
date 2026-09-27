"""Public replay engine for task2-final-rules-v1_dyn (T2R-013).

Faithfully replays the FROZEN task2-final-rules-v1 five-step rule at any as-of date
and applies the dynamic policy of dynamics_policy_v1.json:
  L1  rolling replay, anchor-drift semantics (exact match to final_scores at main cutoff)
  D1  dual-warning-zone hysteresis door (exit<0.45, re-entry>0.55, armed from publication)
  D6  each day scored fresh from 100 (no static carryover)
  D3/D4 ship neutral/calendar per policy (learned candidates documented, not active)

Real data paths live in local controlled config; no real numbers or vehicle ids here.
Run:  python replay.py --events <events_clean.csv> --vehicle-day <vehicle_day.csv> \
        --targets <target_vehicles.csv> --teacher <model_score.csv> \
        --final-scores <final_scores.csv> --splits <splits.csv> --contract <classification_v1.json> \
        --out <daily_scores.json>
"""
from __future__ import annotations

import argparse
import csv, json, math, statistics
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from task2.deduction.contract_loader import load_contract, build_lookups

POLICY = json.loads(Path(__file__).with_name("dynamics_policy_v1.json").read_text(encoding="utf-8"))
P = POLICY["layers"]["L2_dynamic_policy"]
EXIT_T = P["D1_dual_warning_zone_door"]["exit_threshold"]
RE_T = P["D1_dual_warning_zone_door"]["reentry_threshold"]
EXTRA_CAP = 4.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--vehicle-day", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--final-scores", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--contract", type=Path, default=None)
    ap.add_argument("--start", type=date.fromisoformat, default=date(2026, 6, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 7, 30))
    ap.add_argument("--main-cutoff", type=date.fromisoformat, default=date(2026, 6, 21))
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    lk = build_lookups(load_contract(a.contract))
    ROLE, GROUP = lk["role_by_code"], lk["group_by_code"]
    SCORED = sorted(c for c in ROLE if ROLE[c] in ("behavior", "context_warning"))
    BLIND = {"60292", "60294"}

    folds = {r["gpsno"].strip(): r["fold"].strip() for r in csv.DictReader(a.splits.open(encoding="utf-8-sig"))}
    TARGET = sorted(folds)
    FOLD_IDS = sorted(set(folds.values()))
    MS = {r["gpsno"].strip(): float(r["prob"]) for r in csv.DictReader(a.teacher.open(encoding="utf-8-sig"))}
    FS = {r["gpsno"].strip(): float(r["S_final"]) for r in csv.DictReader(a.final_scores.open(encoding="utf-8-sig"))}
    equip = {}
    for row in csv.DictReader(a.targets.open(encoding="utf-8-sig")):
        equip[row["gpsno"].strip()] = row.get("equipped_bsd", "").strip() not in ("", "0", "False", "false")

    days = [a.start + timedelta(days=i) for i in range((a.end - a.start).days + 1)]
    IDX = {d: i for i, d in enumerate(days)}
    N = len(days)
    km_cum = {g: [0.0] * (N + 1) for g in TARGET}
    cnt = {g: {c: [0] * (N + 1) for c in SCORED} for g in TARGET}
    kmday = defaultdict(dict)
    for row in csv.DictReader(a.events.open(encoding="utf-8-sig", newline="")):
        g = row["gpsno"].strip()
        if g not in folds: continue
        code = row["event_type"].strip()
        if code not in ROLE or ROLE[code] not in ("behavior", "context_warning"): continue
        d = datetime.fromisoformat(row["start_time"]).date()
        if d not in IDX: continue
        cnt[g][code][IDX[d] + 1] += 1
    for row in csv.DictReader(a.vehicle_day.open(encoding="utf-8-sig")):
        g = row["gpsno"].strip()
        if g in folds:
            kmday[g][date.fromisoformat(row["date"].strip())] = float(row["traj_km"] or 0)
    for g in TARGET:
        run = 0.0
        for i, d in enumerate(days):
            run += kmday[g].get(d, 0.0)
            km_cum[g][i + 1] = run
        for c in SCORED:
            arr = cnt[g][c]
            for i in range(1, N + 1):
                arr[i] += arr[i - 1]

    def rates_at(g, c):
        i2 = IDX[c]; i1 = max(0, i2 - a.lookback)
        km = km_cum[g][i2] - km_cum[g][i1]
        if km <= 0: return {}, 0.0
        r = {}
        for code in SCORED:
            n = cnt[g][code][i2] - cnt[g][code][i1]
            if n > 0 and not (code in BLIND and not equip.get(g, False)):
                r[code] = n / km * 1000.0
        return r, km

    def frac(r, ks):
        if r >= ks[2]: return 1.0
        xs = (0.0, ks[0], ks[1], ks[2]); ys = (0.0, 0.5, 0.75, 1.0)
        j = 0
        while j < 3 and r > xs[j + 1]: j += 1
        x0, x1, y0, y1 = xs[j], xs[j + 1], ys[j], ys[j + 1]
        return y0 if x1 <= x0 else y0 + (y1 - y0) * (r - x0) / (x1 - x0)

    def score_core(rates, knots):
        if not rates: return 100.0
        grp = defaultdict(list)
        for code, r in rates.items():
            if code in knots: grp[GROUP[code]].append((code, r))
        tot = 0.0
        for gp, mem in grp.items():
            if gp == "G6" or not mem: continue
            vals = sorted(10.0 / len(mem) * frac(r, knots[c]) for c, r in mem)
            tot += vals[-1] + (0.25 * statistics.fmean(vals[:-1]) if len(vals) > 1 else 0.0)
        return 100.0 - tot

    def mk_pct(arr):
        arr = sorted(arr)
        def f(x):
            if not arr: return 0.5
            lo, hi = 0, len(arr)
            while lo < hi:
                m = (lo + hi) // 2
                if arr[m] < x: lo = m + 1
                else: hi = m
            return lo / max(1, len(arr))
        return f

    def fit(tr):
        rt = {g: rates_at(g, a.main_cutoff)[0] for g in tr}
        knots = {}
        for code in SCORED:
            arr = sorted(rt[g][code] for g in tr if code in rt[g])
            if len(arr) >= 5:
                knots[code] = tuple(arr[min(len(arr) - 1, int(round(p / 100 * len(arr))))] for p in (50, 75, 95))
        top = set(sorted(tr, key=lambda g: -MS[g])[:50])
        w = {}
        for code in SCORED:
            aa = [rt[g][code] for g in top if code in rt[g]]
            bb = [rt[g][code] for g in tr if code not in top and code in rt[g]]
            if len(aa) >= 3 and len(bb) >= 5:
                ex = statistics.fmean(math.log1p(x) for x in aa) - statistics.fmean(math.log1p(x) for x in bb)
                if ex > 0: w[code] = ex
        s = sum(w.values()); w = {c: v / s for c, v in w.items()} if s > 0 else {}
        return {"knots": knots, "w": w,
                "pct_rate": {c: mk_pct([rt[g][c] for g in tr if c in rt[g]]) for c in SCORED},
                "pct_v3": mk_pct([100.0 - score_core(rt[g], knots) for g in tr]),
                "pct_t": mk_pct([MS[g] for g in tr]), "tr": tr}

    MACH = {fid: fit([g for g in TARGET if folds[g] != fid]) for fid in FOLD_IDS}
    for fid, m in MACH.items():
        vals = []
        for g in m["tr"]:
            rates, km = rates_at(g, a.main_cutoff)
            s3 = score_core(rates, m["knots"])
            extra = 0.0
            if m["w"] and m["pct_t"](MS[g]) > 0.6 and m["pct_v3"](100.0 - s3) > 0.5:
                extra = min(EXTRA_CAP, sum(m["pct_rate"][c](rates[c]) * m["w"][c] for c in rates if c in m["w"]))
            vals.append(max(0.0, s3 - extra))
        m["anchor"] = statistics.median(vals)

    door = {}
    raw = {}
    for c in days:
        for g in TARGET:
            m = MACH[folds[g]]
            rates, km = rates_at(g, c)
            s3 = score_core(rates, m["knots"])
            pv3 = m["pct_v3"](100.0 - s3)
            if c < a.main_cutoff:
                in_zone = m["pct_t"](MS[g]) > 0.6 and pv3 >= 0.5
            else:
                st = door.get(g)
                if st is None:
                    st = "in" if (m["pct_t"](MS[g]) > 0.6 and pv3 >= 0.5) else "off"
                if st == "in" and pv3 < EXIT_T: st = "out"
                elif st == "out" and pv3 > RE_T and m["pct_t"](MS[g]) > 0.6: st = "in"
                door[g] = st
                in_zone = st == "in"
            extra = 0.0
            if in_zone and m["w"]:
                extra = min(EXTRA_CAP, sum(m["pct_rate"][c2](rates[c2]) * m["w"][c2] for c2 in rates if c2 in m["w"]))
            v = m["anchor"] if km <= 0 else (km / (km + 100.0)) * max(0.0, s3 - extra) + (100.0 - km / (km + 100.0)) * m["anchor"]
            raw.setdefault(g, {})[c.isoformat()] = v

    mains = a.main_cutoff.isoformat()
    daily = {d.isoformat(): {g: round(max(0.0, min(100.0, FS[g] + (raw[g][d.isoformat()] - raw[g][mains]))), 3)
                             for g in TARGET} for d in days}
    a.out.write_text(json.dumps({"policy_version": POLICY["policy_version"], "daily": daily},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"replay done: {len(days)} days x {len(TARGET)} vehicles -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
