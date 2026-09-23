# -*- coding: utf-8 -*-
"""F3 interface 合成测试：收尾剪枝确定性、ρ>0.95 配对去后者、近零方差剔除、契约断言报错路径、
合成端到端（合成分片→一遍扫描→interface 组装→契约断言）。

仅合成数据；不触及真实数据。必测红线：剔除清单同输入两次逐字节一致、|Spearman ρ|>0.95 成对
去后者（按声明列序保先者）、近零方差剔除（方差阈值口径，不按缺失率）、计数双轨只留 per_1000km 版、
sample_id 重复与 y 混入特征列的契约报错、事件时刻须在特征窗内、总列数＝基座保留＋新增保留可对账。
运行：cd feature_engineering && pytest tests -q
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from accident_pipeline_f3.core import to_epoch_s
from accident_pipeline_f3.interface import (META_COLUMNS, assert_contract, build_f3_interface,
                                            finalize_columns, is_count_column,
                                            load_base_columns, select_new_columns)
from accident_pipeline_f3.pipeline import load_f3_config, run_scan

DAY_S = 86400.0
AS_OF = "2026-06-21T00:00:00"
LOOKBACK_DAYS = 20.0


def _dump(entries: list[dict]) -> bytes:
    """剔除清单 JSON 序列化（与 build_f3_interface 同口径），返回字节串供逐字节比较。"""
    return (json.dumps(entries, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


# ------------------------------------------------------------- 清单读取与新增列选取

def test_load_base_columns_text_and_json(tmp_path):
    txt = tmp_path / "keep.txt"
    txt.write_text("# FEAT-006 lean 保留清单\ncol_b\n\ncol_a\n  col_c  \n", encoding="utf-8")
    assert load_base_columns(txt) == ["col_b", "col_a", "col_c"]  # 清单列序即声明列序
    js = tmp_path / "keep.json"
    js.write_text(json.dumps(["x1", "x2"]), encoding="utf-8")
    assert load_base_columns(js) == ["x1", "x2"]
    dup = tmp_path / "dup.txt"
    dup.write_text("a\nb\na\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复列名"):
        load_base_columns(dup)


def test_select_new_columns_family_quota_and_cap():
    cands = ["f3_hist_recurrence_gap_days", "f3_chain_event_count_per_1000km",
             "f3_score_fatigue", "f3_traj_speed_p50", "f3_copair_a__b_per_100h",
             "f3_score_night_chain", "f3_hist_decay_count_per_100h"]
    # r5 §2 族配额：score 恒优先整族先行，余族按预登记族序轮转（hist→speed→chain）
    assert select_new_columns(cands, max_new=4) == [
        "f3_score_fatigue", "f3_score_night_chain",
        "f3_hist_recurrence_gap_days", "f3_traj_speed_p50"]
    full = select_new_columns(cands)
    assert full == ["f3_score_fatigue", "f3_score_night_chain",
                    "f3_hist_recurrence_gap_days", "f3_traj_speed_p50",
                    "f3_chain_event_count_per_1000km"]
    assert is_count_column("f3_traj_gap_days_per_1000km")
    assert not is_count_column("f3_traj_speed_cv")


# ------------------------------------------------------------- 收尾通则（§3.0）

def test_finalize_near_zero_variance_removed():
    rng = np.random.default_rng(0)
    tiny = np.full(50, 1.0)
    tiny[0] += 1e-9  # 方差 ~1e-19 < 1e-12 阈值
    frame = pd.DataFrame({"c_const": np.ones(50), "c_tiny": tiny,
                          "c_ok": rng.normal(0, 1, 50)})
    kept, dropped = finalize_columns(frame, ["c_const", "c_tiny", "c_ok"])
    assert kept == ["c_ok"]
    by_col = {d["column"]: d for d in dropped}
    assert by_col["c_const"]["stage"] == "near_zero_variance"
    assert by_col["c_tiny"]["stage"] == "near_zero_variance"


def test_finalize_spearman_pair_drops_later_column():
    rng = np.random.default_rng(1)
    x = np.linspace(0.0, 1.0, 60)
    frame = pd.DataFrame({"col_x": x, "col_y": x ** 3,          # 单调变换：ρ=1.0
                          "col_neg": -x,                          # ρ=-1.0（绝对值口径）
                          "col_z": rng.normal(0, 1, 60)})
    kept, dropped = finalize_columns(frame, ["col_x", "col_y", "col_neg", "col_z"])
    assert kept == ["col_x", "col_z"]  # 保先者、去后者
    by_col = {d["column"]: d for d in dropped}
    assert by_col["col_y"]["stage"] == "spearman_rho"
    assert by_col["col_y"]["partner"] == "col_x"
    assert by_col["col_y"]["rho"] == pytest.approx(1.0, abs=1e-6)
    assert by_col["col_neg"]["partner"] == "col_x"
    # 声明列序反转 → 保前者反转（固定列序确定性）
    kept2, dropped2 = finalize_columns(frame, ["col_y", "col_x", "col_neg", "col_z"])
    assert kept2 == ["col_y", "col_z"]
    assert {d["column"] for d in dropped2} == {"col_x", "col_neg"}


def test_finalize_dual_track_keeps_per_1000km():
    rng = np.random.default_rng(2)
    v = rng.poisson(3, 40).astype(float)
    frame = pd.DataFrame({"f3_chain_event_count_per_1000km": v * 2.3,
                          "f3_chain_event_count_per_100h": v * 7.1,
                          "f3_hist_x_trend": rng.normal(0, 1, 40)})
    kept, dropped = finalize_columns(
        frame, ["f3_chain_event_count_per_1000km", "f3_chain_event_count_per_100h",
                "f3_hist_x_trend"])
    assert "f3_chain_event_count_per_1000km" in kept
    assert "f3_chain_event_count_per_100h" not in kept  # §3.0：同场景只保留 per_1000km 版
    by_col = {d["column"]: d for d in dropped}
    assert by_col["f3_chain_event_count_per_100h"]["stage"] == "dual_track"
    assert by_col["f3_chain_event_count_per_100h"]["partner"] == "f3_chain_event_count_per_1000km"


def test_finalize_dropped_list_byte_identical():
    """收尾剪枝确定性（§3.5/§5 L1 #3）：同输入跑两次剔除清单逐字节一致。"""
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 1.0, 80)
    frame = pd.DataFrame({"c_x": x, "c_y": 2.0 * x + 1.0, "c_const": np.zeros(80),
                          "c_a_per_1000km": rng.poisson(2, 80).astype(float),
                          "c_a_per_100h": rng.poisson(2, 80).astype(float) + 1.0,
                          "c_noise": rng.normal(0, 1, 80)})
    declared = list(frame.columns)
    kept_1, dropped_1 = finalize_columns(frame, declared)
    kept_2, dropped_2 = finalize_columns(frame, declared)
    assert kept_1 == kept_2
    assert _dump(dropped_1) == _dump(dropped_2)  # 逐字节一致


# ------------------------------------------------------------- 契约断言（L1 #4）

def _meta_frame(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame({"sample_id": [f"S{i}" for i in range(n)],
                         "gpsno": [f"V{i}" for i in range(n)],
                         "y": np.arange(n) % 2, "fold": np.arange(n) % 2})


def test_contract_duplicate_sample_id_raises():
    mi = _meta_frame()
    mi.loc[1, "sample_id"] = "S0"
    mi["b1"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="sample_id 不唯一"):
        assert_contract(mi, ["b1"], [])
    mi = _meta_frame()
    mi.loc[1, "gpsno"] = "V0"
    mi["b1"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="每车一行"):
        assert_contract(mi, ["b1"], [])


def test_contract_meta_leaks_into_feature_columns_raises():
    mi = _meta_frame()
    mi["b1"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="元数据混入特征列"):
        assert_contract(mi, ["y", "b1"], [])          # y 混入特征列
    with pytest.raises(ValueError, match="元数据混入特征列"):
        assert_contract(mi, ["b1"], ["fold"])          # fold 混入特征列


def test_contract_event_window_and_reconciliation():
    as_of_s = float(to_epoch_s(pd.Series([AS_OF]))[0])
    lookback_s = LOOKBACK_DAYS * DAY_S
    mi = _meta_frame()
    mi["b1"] = [1.0, 2.0]
    mi["n1"] = [3.0, 4.0]
    ok_events = pd.DataFrame({"event_id": ["E1", "E2"],
                              "gpsno": ["V0", "V1"],
                              "event_time": ["2026-06-05 08:00:00", "2026-06-20 23:00:00"]})
    assert_contract(mi, ["b1"], ["n1"], events=ok_events,
                    as_of_s=as_of_s, lookback_s=lookback_s)   # 正常路径不抛
    bad_events = pd.concat([ok_events, pd.DataFrame(
        {"event_id": ["E3"], "gpsno": ["V0"], "event_time": ["2026-06-25 08:00:00"]})],
        ignore_index=True)                                     # 落在标签窗（窗外）
    with pytest.raises(ValueError, match="事件时刻不在特征窗"):
        assert_contract(mi, ["b1"], ["n1"], events=bad_events,
                        as_of_s=as_of_s, lookback_s=lookback_s)
    dup_ev = pd.DataFrame({"event_id": ["E1", "E1"], "gpsno": ["V0", "V1"],
                           "event_time": ["2026-06-05 08:00:00", "2026-06-06 08:00:00"]})
    with pytest.raises(ValueError, match="事件表主键不唯一"):
        assert_contract(mi, ["b1"], ["n1"], events=dup_ev,
                        as_of_s=as_of_s, lookback_s=lookback_s)
    with pytest.raises(ValueError, match="不可对账"):            # 总列数 ≠ 基座保留＋新增保留
        assert_contract(mi, ["b1"], ["n2"], events=ok_events,
                        as_of_s=as_of_s, lookback_s=lookback_s)


# ------------------------------------------------------------- 合成端到端

def _write_e2e_inputs(root: Path) -> None:
    """合成轨迹分片/事件表/画像表/基座表/保留清单/本地配置（不触及真实数据）。"""
    rng = np.random.default_rng(2026)
    n_veh, n_pts, n_bursts = 24, 1400, 40
    as_of = pd.Timestamp(AS_OF)
    w0 = as_of - pd.Timedelta(days=LOOKBACK_DAYS)
    span_s = int(LOOKBACK_DAYS * DAY_S)
    spacing = span_s / n_pts

    (root / "traj").mkdir(parents=True, exist_ok=True)
    traj_rows = []
    events_rows = []
    profile_rows = []
    base_rows = []
    for i in range(n_veh):
        gps = f"V{i:03d}"
        off = float(rng.integers(0, int(spacing)))
        t = w0.value // 10**9 + off + np.arange(n_pts) * spacing
        speed = 30.0 + 35.0 * rng.random() + rng.normal(0, 3.0, n_pts)
        lat = 30.0 + i * 0.01 + np.cumsum(rng.normal(0, 0.0008, n_pts))
        lon = 114.0 + np.cumsum(rng.normal(0, 0.0008, n_pts))
        run = np.zeros(n_pts)  # 单行程累计：每 25 行重置（≈8h 行程）
        for s in range(0, n_pts, 25):
            run[s:s + 25] = np.arange(min(25, n_pts - s)) * spacing
        traj_rows.append(pd.DataFrame({
            "gpsno": gps, "data_time": pd.to_datetime(t, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
            "run_duration_s": run, "speed": speed, "lat": lat, "lon": lon,
            "ems_speed": speed + rng.normal(0, 1.0, n_pts), "gps_speed": speed}))
        # 事件：40 个连环爆发（每段 3 条、≤30min），热点坐标 2 簇＋零散噪声
        scen = ["hard_brake", "turn_l", "fatigue_alert", "collision"]
        ev_t, ev_sc, ev_sp, ev_la, ev_lo = [], [], [], [], []
        for b in range(n_bursts):
            base_t = float(rng.uniform(0, span_s * 0.99))
            for j in range(3):
                ev_t.append(base_t + j * 120.0)
                ev_sc.append(scen[(b + j) % len(scen)])
                ev_sp.append(float(rng.uniform(20, 90)))
        k_hot = 80 + (i * 3) % 38
        for k in range(len(ev_t)):
            if k < k_hot:
                c = (30.0 + i * 0.01, 114.0 + 0.005 * (k % 2))
                ev_la.append(c[0] + rng.normal(0, 0.0002))
                ev_lo.append(c[1] + rng.normal(0, 0.0002))
            else:
                ev_la.append(30.0 + i * 0.01 + rng.normal(0, 0.05))
                ev_lo.append(114.0 + rng.normal(0, 0.05))
        events_rows.append(pd.DataFrame({
            "event_id": [f"E{i:03d}{k:04d}" for k in range(len(ev_t))], "gpsno": gps,
            "event_time": pd.to_datetime(w0.value // 10**9 + np.array(ev_t), unit="s"
                                         ).strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": ev_sc, "speed": ev_sp, "lat": ev_la, "lon": ev_lo}))
        profile_rows.append({
            "gpsno": gps, "energy_type": ("diesel" if i % 2 == 0 else "electric"),
            "highway_ratio": float(rng.uniform(0, 1)),
            "profile_month_km": float(rng.uniform(2000, 8000)),
            "profile_month_hours": float(rng.uniform(40, 120)),
            "profile_month_night_share": float(rng.uniform(0.05, 0.30))})
        base_rows.append({
            "sample_id": f"S{i:03d}", "gpsno": gps, "y": int(i % 3 == 0), "fold": int(i % 2),
            "as_of": AS_OF, "window_start": str(w0), "window_end": AS_OF,
            "lookback_days": LOOKBACK_DAYS, "horizon_days": 30.0,
            "energy_type": ("diesel" if i % 2 == 0 else "electric"),
            "highway_ratio": float(rng.uniform(0, 1))})

    traj = pd.concat(traj_rows, ignore_index=True)
    veh = traj["gpsno"].unique()
    for shard_idx, half in enumerate(np.array_split(veh, 2)):
        part = traj[traj["gpsno"].isin(half)]
        part.to_csv(root / "traj" / f"part-{shard_idx:03d}.csv", index=False)

    base = pd.DataFrame(base_rows)
    rate = np.linspace(0.1, 3.0, n_veh) + np.random.default_rng(7).normal(0, 0.05, n_veh)
    base["base_evt_rate"] = rate
    base["base_evt_rate_dup"] = 2.0 * rate + 1.0          # 与 base_evt_rate ρ=1.0（去后者素材）
    base["base_imu_vol"] = np.random.default_rng(8).normal(0, 1, n_veh)
    base["base_night_share"] = np.random.default_rng(9).uniform(0, 1, n_veh)
    base["base_const"] = 1.0                              # 常数列（近零方差剔除素材）
    base.to_csv(root / "base.csv", index=False)
    pd.concat(events_rows, ignore_index=True).to_csv(root / "events.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(root / "profile.csv", index=False)
    (root / "keep_columns.txt").write_text(
        "# FEAT-006 lean 保留清单（占位合成）\nbase_evt_rate\nbase_evt_rate_dup\n"
        "base_imu_vol\nbase_night_share\nbase_const\nbase_missing_col\n", encoding="utf-8")
    (root / "config.yaml").write_text(f"""\
trajectory_dir: "traj"
events_path: "events.csv"
profile_path: "profile.csv"
base_input: "base.csv"
base_columns_file: "keep_columns.txt"
output_dir: "out"
window:
  as_of: "{AS_OF}"
  lookback_days: {LOOKBACK_DAYS}
  horizon_days: 30.0
shard_delimiter: ","
chunk_rows: 5000
exposure:
  min_km: 1.0
  min_hours: 0.5
cohort:
  energy_type_column: "energy_type"
  highway_share_column: "highway_ratio"
  source_columns: ["base_evt_rate", "base_imu_vol"]
events:
  copair_columns: ["f3_copair_collision__hard_brake"]
""", encoding="utf-8")


def test_e2e_synthetic_shards_to_interface(tmp_path):
    """合成端到端：合成分片 → 一遍扫描（分片读取→聚合）→ interface 组装 → 契约断言。"""
    _write_e2e_inputs(tmp_path)
    cfg = load_f3_config(tmp_path / "config.yaml")
    assert cfg.as_of_s == float(to_epoch_s(pd.Series([AS_OF]))[0])  # 窗口三元组来自配置（红线 6）
    new_path = run_scan(cfg)
    new_df = pd.read_csv(new_path)
    assert len(new_df) == 24                                # 每车一行
    assert {"f3_score_night_chain", "f3_score_fatigue"} <= set(new_df.columns)  # 综合分产出
    assert "f3_traj_speed_p50" in new_df.columns            # Tier 2 一遍带出
    assert "f3_copair_collision__hard_brake_per_1000km" in new_df.columns       # 计数双轨
    assert "f3_copair_collision__hard_brake_per_100h" in new_df.columns

    out = build_f3_interface(cfg)                           # 内含契约断言（违反即抛）
    assert out.exists()
    interface_dir = tmp_path / "out" / "model_interface"
    dropped_bytes = (interface_dir / "f3_dropped_columns.json").read_bytes()  # 第一次运行产物
    manifest = json.loads((interface_dir / "f3_manifest.json").read_text(encoding="utf-8"))
    dropped = json.loads((interface_dir / "f3_dropped_columns.json").read_text(encoding="utf-8"))

    # 总列数对账（§5 L1 #4）：总列数＝基座保留＋新增保留
    assert manifest["rows"] == 24
    assert manifest["feature_count"] == manifest["base_kept_count"] + manifest["new_kept_count"]
    assert manifest["new_kept_count"] <= 40               # §3.0 新增列上限
    assert manifest["code_commit"]
    assert manifest["input_fingerprint"] == "PENDING"     # 输入指纹留占位函数
    mi = pd.read_csv(out)
    actual = [c for c in mi.columns if c not in META_COLUMNS]
    assert actual == manifest["base_kept"] + manifest["new_kept"]
    assert {"sample_id", "gpsno", "y", "fold", "as_of", "window_end"} <= set(mi.columns)
    assert "f3_score_night_chain" in manifest["new_kept"]  # 综合分优先保留

    by_col = {d["column"]: d for d in dropped["dropped"]}
    assert by_col["base_evt_rate_dup"]["stage"] == "spearman_rho"   # ρ>0.95 去后者
    assert by_col["base_evt_rate_dup"]["partner"] == "base_evt_rate"
    assert by_col["base_const"]["stage"] == "near_zero_variance"    # 近零方差剔除
    assert by_col["base_missing_col"]["stage"] == "missing_in_base_input"
    assert all(not c.endswith("_per_100h") for c in manifest["new_kept"])   # 双轨只留 per_1000km
    # 剔除清单可对账：声明列＝基座保留＋新增保留＋收尾剔除（缺基座列另行登记）
    steps = [d for d in dropped["dropped"] if d["stage"] != "missing_in_base_input"]
    assert (len(dropped["declared_columns"])
            == len(dropped["kept_base"]) + len(dropped["kept_new"]) + len(steps))

    # 收尾剪枝确定性：同输入再跑一次，剔除清单逐字节一致
    build_f3_interface(cfg)
    again_bytes = (interface_dir / "f3_dropped_columns.json").read_bytes()
    assert again_bytes == dropped_bytes                      # 两次运行逐字节一致
