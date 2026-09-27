# 任务一正式提交适配器

## 冻结 g8v6 全量拟合

`run_g8v6_fullfit.py` 实现交付包中 `run_g8v6_corrected.py` 的固定四成员、三种子、固定权重全量拟合与预测部分，不做新选型。训练输入为同一冻结标签的 20 天特征表，推理输入须为真正截至 `2026-08-01` 的 61 天特征表。四组特征各 500 行，按 `gpsno` 对齐；训练有而推理缺失的列会拒绝，推理新增列仅记账不入模型。运行结果含模型 bundle、列清单、真实未来概率、来源 manifest，以及由下方适配器校验并生成的 `submission/forecast_result.csv`，均写入受控输出目录。

为避免误把 60 天窗当 61 天窗，必须随推理特征提供 `score_manifest.json`：`role=future_features`、`as_of=2026-08-01`、`feature_start=2026-06-01`、`feature_end_exclusive=2026-08-01`、四个成员表的 SHA-256，及 `source_coverage` 中 `events_clean`／`vehicle_day`／`trajectory`／`imu` 四源末次观测日期。事件和车辆日源必须覆盖 7 月 31 日。manifest 是可审计声明，仍须核验实际源数据及构建日志。

```sh
python3 models/task1_final/run_g8v6_fullfit.py \
  --labels <受控代理标签/labels.csv> --vehicles <受控500车清单> \
  --train-g1 <20天g1.csv> --score-g1 <61天g1.csv> \
  --train-g2 <20天g2.csv> --score-g2 <61天g2.csv> \
  --train-g3 <20天g3.csv> --score-g3 <61天g3.csv> \
  --train-m4 <20天成员4.csv> --score-m4 <61天成员4.csv> \
  --score-manifest <受控score_manifest.json> \
  --output-dir outputs/task1-final/<新run-id>
```

运行时需 Python 3.12、`interpret`、NumPy、pandas、scikit-learn、SciPy、joblib；交付包 `03_manifests/dependencies.txt` 的版本仅为其原设备锁定，本机环境差异须留在受控运行记录。若 61 天特征仍缺 7 月 31 日、列无法对齐或训练逻辑不能复现交付包 OOF，本入口不会被视为正式交付完成。

## 官方 CSV 打包

`prepare_submission.py` 只把**已经训练完成的模型对 2026-08-01 之后的真实推理概率**封装成官方提交文件，不训练模型，也不接受开发期 OOF。运行前先由所选模型线生成：

- `vehicles.csv`：一列 `gpsno`，赛事固定 500 车，一车一行；
- `predictions.csv`：两列 `gpsno,risk_prob`，同一 500 车的未来风险概率；
- `provenance.json`：训练模型、训练特征与标签、推理特征、预测文件的哈希和时间来源声明。

受控 `provenance.json` 的必填字段如下；哈希应由实际文件生成，不能用示例值或开发期 OOF 代替：

```json
{
  "prediction_role": "future_inference",
  "model_version": "<frozen-model-version>",
  "as_of": "2026-08-01",
  "feature_start": "2026-06-01",
  "feature_end_exclusive": "2026-08-01",
  "training_feature_end_exclusive": "2026-06-21",
  "training_label_start": "2026-06-21",
  "training_label_end_exclusive": "2026-07-31",
  "model_sha256": "<64-char-hex>",
  "inference_features_sha256": "<64-char-hex>",
  "training_features_sha256": "<64-char-hex>",
  "training_labels_sha256": "<64-char-hex>",
  "feature_schema_sha256": "<64-char-hex>",
  "predictions_sha256": "<64-char-hex>"
}
```

运行（路径均指本机受控文件）：

```sh
python3 models/task1_final/prepare_submission.py \
  --vehicles outputs/task1-final/vehicles.csv \
  --predictions outputs/task1-final/predictions.csv \
  --provenance outputs/task1-final/provenance.json \
  --output-dir outputs/task1-final/submission
```

成功后得到 `forecast_result.csv` 和 `submission_receipt.json`。`accident` 仅作格式字段：按概率降序、同分 `gpsno` 升序给前 100 车标 1；线上 AUC 以未降精度的 `risk_prob` 排序为准。输出保持 `gpsno,accident,risk_prob` 列序、500 行、UTF-8 无 BOM、LF。所有真实输入、模型、输出和哈希留在忽略目录。

适配器只校验 manifest 的内容和预测文件哈希；还须独立审计该 manifest 确实由冻结模型训练与 61 天特征构建流程生成，且训练／推理列和变换器同构。尤其不能把 6/21–7/31 的代理标签与覆盖该时段的 61 天特征直接配对训练。
