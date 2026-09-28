# 任务二三截点结果导出

本目录只放复算代码，不放逐车结果。真实输入、导出的 CSV、逐日台账和内置真实数据的 UI 保存在团队受控交付目录，遵守 `docs/DATA_POLICY.md`。

冻结动态层的 `task2/final_rules_v1_dyn/replay.py` 对 2026-06-21、2026-07-11、2026-07-31 三个截点回放前 20 天证据。默认结束日为 7 月 30 日；导出第三截点前，须在真实数据覆盖到 7 月 30 日且来源一致时设置 `--end 2026-07-31`。不得复制或插值早期分数补齐日期。

```sh
python task2/final_rules_v1_dyn/replay.py \
  --events <受控事件表> --vehicle-day <受控车辆日表> \
  --targets <受控500车清单> --teacher <冻结g10教师> \
  --final-scores <冻结静态分表> --splits <受控折分> \
  --contract task2/classification/classification_v1.json \
  --end 2026-07-31 --out <受控输出目录>/daily_scores.json

python task2/results/build_three_cutoff_csv.py \
  --daily-ledger <受控输出目录>/daily_scores.json \
  --final-scores <冻结静态分表> --g8-forecast <受控g8正式概率> \
  --output <受控输出目录>/task2_scores_500_three_cutoffs.csv
```

导出脚本验证 500 车集合一致、三个截点齐全、分数在 0--100、6 月 21 日动态分与冻结静态分逐车一致。`g8_risk_prob_asof_20260801` 是单列参照，**不参与**较早截点的扣分。冻结 v1 的教师仍为 `sprint-g10-final`；换教师须另开评分版本并重跑。

`merge_full_ui.py` 把真实三截点结果嵌入原有大 UI，保留总览、六维雷达、排名搜索/筛选和运营动作；原有雷达与逐车原因仍是清楚标注的结构演示值。它把原始 `gpsno` 换成新的展示编号，生成的 HTML 含逐车评分，不得提交到公开 Git：

```sh
python task2/results/merge_full_ui.py \
  --original-ui <受控原版大UI.html> --static-scores <冻结静态分表> \
  --three-scores <受控三截点CSV> --output <受控输出目录>/task2_full_dashboard_three_cutoffs.html
```

本轮真实结果与限制见 [`docs/report/task2/three-cutoff-results-20260928.md`](../../docs/report/task2/three-cutoff-results-20260928.md)。
