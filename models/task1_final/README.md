# 任务一正式提交适配器

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
