# FEAT-011 implementation

This directory implements the accepted [FEAT-011 plan](../../../docs/plans/feat-011-small-attribution-model-adaptation-plan.md). The S0 auditor reads the locked FEAT-009 v2 Y1 F3 table and Y0 visibility ledger, writes a schema/provenance ledger and checks into ignored `outputs/feat-011/`, and refuses to overwrite a non-empty run.

Use local paths only as command arguments; do not put machine paths or input hashes in Git:

```sh
python3 feature_engineering/experiments/feat011/audit_s0.py \
  --input outputs/feat-009/v2/y1/f3_model_input.csv \
  --manifest outputs/feat-009/v2/y1/input_manifest.json \
  --visibility outputs/feat-009/v2/y0/visibility.csv \
  --output-dir outputs/feat-011/s0/<run-id>
```

The audit has synthetic negative cases for future profile fields, duplicate vehicle keys, changed label/split versions, and column-set drift. Generate visual acceptance samples with `make_s0_synthetic_preview.py`; they contain no real records, counts, or results.

S0 completion does not authorize S1 preregistration or real OOF. That requires the human review gates and sequence in the accepted plan.

Run the S1 synthetic preflight in the FEAT-009 modeling environment. It trains A–D on generated examples, derives E with the fixed equal-logit rule, audits fold-scoped fitting, and writes synthetic-only results and review figures to ignored controlled outputs:

```sh
python3 feature_engineering/experiments/feat011/s1_synthetic_preflight.py \
  --output-dir outputs/feat-011/s1-synthetic/<run-id>
```

The generated pre-registration is a plumbing fixture, not a locked real-data registration. S1 figures and cost checks require human review. EVAL-001, EVAL-003 and FEAT-009 actual review gates still control whether real OOF may run.
