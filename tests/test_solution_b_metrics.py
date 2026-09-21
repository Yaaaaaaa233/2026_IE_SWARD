"""方案B指标库单元测试：全部使用合成数据（不依赖比赛数据与本机路径）。"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models" / "solution_b"))
spec = importlib.util.spec_from_file_location("sb_metrics", ROOT / "models/solution_b/metrics.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def synth(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n // 4), np.zeros(n - n // 4)].astype(int)
    rng.shuffle(y)
    p = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.1, n), 0.01, 0.99)
    sid = [f"s{i:03d}" for i in range(n)]
    return sid, y, p


class TestHardChecks(unittest.TestCase):
    def test_rejects_out_of_range_and_nan(self):
        sid, y, p = synth()
        bad = p.copy(); bad[0] = 1.5
        self.assertTrue(any("越界" in e for e in m.hard_check_predictions(sid, y, bad)))
        nan = p.copy(); nan[2] = np.nan
        self.assertTrue(any("缺失" in e for e in m.hard_check_predictions(sid, y, nan)))

    def test_rejects_duplicate_and_bad_label(self):
        sid, y, p = synth()
        sid2 = sid.copy(); sid2[1] = sid[0]
        self.assertTrue(any("重复" in e for e in m.hard_check_predictions(sid2, y, p)))
        y2 = y.copy(); y2[0] = 2
        self.assertTrue(any("标签" in e for e in m.hard_check_predictions(sid, y2, p)))

    def test_accepts_valid(self):
        sid, y, p = synth()
        self.assertEqual(m.hard_check_predictions(sid, y, p), [])


class TestMetrics(unittest.TestCase):
    def test_single_class_auc_none(self):
        _, _, p = synth()
        self.assertIsNone(m.auc_or_none(np.zeros(60), p))

    def test_perfect_ranking(self):
        y = np.r_[np.ones(10), np.zeros(50)]
        p = np.r_[np.linspace(0.9, 0.6, 10), np.linspace(0.4, 0.1, 50)]
        self.assertAlmostEqual(m.auc_or_none(y, p), 1.0)
        self.assertAlmostEqual(m.brier(y, p), np.mean((p - y) ** 2))

    def test_logloss_clip(self):
        y = np.array([1, 0])
        p = np.array([1.0, 0.0])
        ll = m.logloss(y, p, eps=1e-15)
        self.assertTrue(np.isfinite(ll))

    def test_topk_tie_break_by_sample_id(self):
        # 同分时按 sample_id 字典序取前 K；截止等分数量必须报告
        sid = ["b", "a", "c", "d"]
        y = [1, 0, 1, 0]
        p = [0.9, 0.9, 0.1, 0.1]  # b、a 同分：a 先入名单
        t = m.topk_table(y, p, sid, q_levels=[0.5])
        row = t.iloc[0]
        self.assertEqual(row["K"], 2)
        self.assertEqual(row["TP_K"], 1)      # 名单 {a, b}，命中 b
        self.assertEqual(row["ties_at_cut"], 2)
        self.assertAlmostEqual(row["Recall@K"], 0.5)

    def test_topk_quota_policy(self):
        sid, y, p = synth(n=50)
        t = m.topk_table(y, p, sid, q_levels=[0.2])
        self.assertEqual(t.iloc[0]["K"], 10)
        self.assertAlmostEqual(t.iloc[0]["Lift@K"],
                               (t.iloc[0]["TP_K"] / 10) / (sum(y) / 50))

    def test_all_metrics_shape(self):
        sid, y, p = synth()
        mm = m.all_metrics(y, p, sid)
        for k in ("AUC", "AP", "Recall@100", "Brier", "LogLoss"):
            self.assertIn(k, mm)
        self.assertEqual(mm["hard_check_errors"], [])


class TestBootstrap(unittest.TestCase):
    def test_paired_bootstrap_self_zero(self):
        sid, y, p = synth(seed=3)
        r = m.paired_bootstrap_auc(y, p, p, sid, n_resamples=100, seed=1)
        self.assertAlmostEqual(abs(r["delta_auc_point"]), 0.0)
        self.assertLessEqual(abs(r["ci_low"]), 1e-9)
        self.assertGreaterEqual(abs(r["ci_high"]), -1e-9)

    def test_paired_bootstrap_detects_better_model(self):
        rng = np.random.default_rng(5)
        y = np.r_[np.ones(20), np.zeros(80)]
        good = np.clip(0.7 + 0.25 * y + rng.normal(0, 0.05, 100), 0.01, 0.99)
        flat = np.full(100, 0.25)
        sid = [f"v{i:03d}" for i in range(100)]
        r = m.paired_bootstrap_auc(y, good, flat, sid, n_resamples=300, seed=2)
        self.assertGreater(r["delta_auc_point"], 0.3)
        self.assertGreater(r["ci_low"], 0.0)


if __name__ == "__main__":
    unittest.main()
