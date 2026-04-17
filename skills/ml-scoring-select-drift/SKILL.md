---
name: ml-scoring-select-drift
description: |
  Diagnose "KeyError: ['feat_a', 'feat_b', ...] not in index" in ML scoring/inference
  pipelines immediately after a feature-bundle retrain. Use when: (1) training ran with
  new features and the new model is promoted, (2) daily scoring (or batch inference) fails
  with a pandas KeyError naming the new features at the `df[feature_names]` slicing step,
  (3) the upstream feature table APPEARS to have been updated. Root cause is almost always
  that training loads features via `SELECT *` while scoring loads via a hardcoded column
  list that was never updated — compounded by `if col in df.columns` preprocessing guards
  that silently skip missing columns, so the failure only surfaces at model inference.
  Also covers the companion Dataform-incremental CREATE-IF-NOT-EXISTS silent-schema trap
  that often hides behind the same retrain.
author: Claude Code
version: 1.0.0
date: 2026-04-17
---

# ML Scoring SELECT Drift After Retrain

## Problem

A retrain bundle adds N new features. Training succeeds, the new model is promoted to
production, and the next scoring run fails with:

```
KeyError: ['new_feat_1', 'new_feat_2', ...] not in index
  File ".../daily_scoring.py", line XXXX, in generate_predictions
    base_feature_df = df[self.base_feature_names].copy()
```

The error lists exactly the features that were just added.

## Why It Happens (two cascading causes)

### Cause 1 — Asymmetric data loading between train and score

Training scripts commonly load via:

```python
query = f"SELECT * FROM `{self.training_table}` WHERE target_date BETWEEN ..."
```

Scoring scripts — to keep wire volume small and intent explicit — commonly load via:

```python
query = f"""
SELECT
    visitor_id,
    scoring_date,
    feat_1, feat_2, ..., feat_N  # hardcoded list
FROM `{self.serving_table}`
WHERE scoring_date = '{self.data_date}'
"""
```

When a PR adds new features:
- The Dataform/dbt SQL is updated → new columns materialize in both training & serving tables.
- The training script auto-picks them up (`SELECT *`) → model is trained with them.
- The scoring script's hardcoded SELECT is NOT updated → columns never enter the DataFrame.
- `base_feature_names` comes from the model artifact, which expects them.
- KeyError at the slicing step.

### Cause 2 — Silent `if col in df.columns` preprocessing guards

The same PR typically adds preprocessing like:

```python
if 'new_feat_1' in df.columns:
    df['new_feat_1'] = df['new_feat_1'].fillna(0).astype(int)
for _col in ('new_feat_a', 'new_feat_b'):
    if _col in df.columns:
        df[_col] = df[_col].fillna(0).astype(int)
```

These guards are a reasonable defensive pattern (useful during rollout / for backfills with
pre-feature data), but they actively hide the SELECT drift at preprocessing time. The
exception is deferred to model inference, where the stack trace points at pandas internals
rather than the real site of the bug (the SELECT).

### Cause 3 (companion, related) — Dataform incremental schema trap

Even when steps 1–2 are fixed, the upstream serving table may itself be missing the columns
because it is `type: "incremental"` and Dataform uses `CREATE TABLE IF NOT EXISTS`. Re-running
the workflow succeeds but silently APPENDS rows in the old schema. Verify via:

```sql
-- If last_modified_time didn't advance, the table wasn't rebuilt.
SELECT TIMESTAMP_SECONDS(CAST(last_modified_time/1000 AS INT64)) AS last_modified,
       row_count
FROM `project.dataset.__TABLES__`
WHERE table_id = 'your_serving_features_table';
```

Fix: `DROP TABLE IF EXISTS ...serving_features` → re-invoke the workflow. See also the
`dataform-incremental-schema-change` skill. Always back up first if you want to preserve
history (`CREATE TABLE ...backup_YYYYMMDD AS SELECT * FROM ...serving_features`).

## Context / Trigger Conditions

All of these together strongly indicate this bug:

- A retrain PR recently merged that added columns to a `.sqlx`/`.sql` + Python preprocessing.
- The KeyError column names match EXACTLY the new features from that PR.
- The scoring script's top-of-file load function has an explicit column list, not `SELECT *`.
- The serving table in BQ/warehouse DOES contain the new columns (after Dataform rebuild).
- Grep shows preprocessing code for the new features is guarded by
  `if col in df.columns` or `for _col in (...): if _col in df.columns`.

## Solution

### Step 1 — Confirm the asymmetry in ~1 minute

```bash
# 1. Scoring script — look for the hardcoded SELECT
grep -nE "SELECT|FROM.*serving" path/to/daily_scoring.py

# 2. Training script — look for SELECT *
grep -nE "SELECT|FROM.*training" path/to/monthly_retrain.py

# 3. Confirm the missing columns ARE in the serving table schema
bq show --schema --format=prettyjson project:dataset.serving_features_table \
  | python3 -c "import json,sys; names=[f['name'] for f in json.load(sys.stdin)]; \
    [print(t, 'PRESENT' if t in names else 'MISSING') for t in ['new_feat_1','new_feat_2']]"
```

If you see: training does `SELECT *`, scoring has a hardcoded list, and the columns are
`PRESENT` in the table → you have this bug.

### Step 2 — Fix the scoring SELECT

Add the missing columns to the scoring script's explicit SELECT list. Match names exactly
(including any `COALESCE(..., sentinel) AS col_name` aliasing from the `.sqlx`).

**Always add all features from the retrain bundle, not only the ones in the KeyError** —
some may not be in `base_feature_names` but are consumed by preprocessing helpers
(e.g., `apply_*_categorical(df)` functions that read a raw column and emit derived
features). If their source column is absent, they silently emit NaN categoricals and
you'll hit subtle prediction-quality drift instead of a loud KeyError.

### Step 3 — Rebuild + redeploy the scoring container

Cloud Run Job example:

```bash
cd path/to/scoring_service
# Copy shared modules into build context (Dockerfile COPY will fail otherwise)
cp -r ../_feature_common ./_feature_common
trap 'rm -rf ./_feature_common' EXIT
gcloud builds submit . --config cloudbuild.yaml --project=$PROJECT
gcloud run jobs update $SCORING_JOB \
  --image=gcr.io/$PROJECT/$SCORING_JOB:latest \
  --region=$REGION --project=$PROJECT
```

### Step 4 — Re-trigger the orchestration workflow (not just the job)

If a Cloud Workflow orchestrates `Dataform → scoring → enriched view`, execute the
workflow (not `gcloud run jobs execute`) so downstream steps (enriched predictions view,
SHAP job, dashboard tables) pick up today's scoring output.

### Step 5 — Verify end-to-end

```sql
SELECT scoring_date, COUNT(*), MIN(model_version)
FROM `project.ml_predictions.predictions_daily`
WHERE scoring_date = CURRENT_DATE()
GROUP BY scoring_date;
```

Row count should match your active-population size; `model_version` should be the
newly-promoted model.

## Defense (add once the hotfix is in)

Preferred permanent fix — pick one:

1. **Switch scoring to `SELECT *` + explicit column-order enforcement.** Load all columns,
   then reindex to `base_feature_names` order. Eliminates the drift class at the cost of
   a small amount of extra wire volume (usually negligible for a daily scoring run).

2. **Auto-generate the scoring SELECT list from the model artifact.** At container start,
   read `base_feature_names` from the loaded model and compose the SELECT from it (plus
   any raw columns consumed by `apply_*` helpers — enumerate those in one place). Fails
   LOUDLY at load time if a feature is missing in BQ, rather than silently at inference.

3. **Replace silent `if col in df.columns` guards with asserts** (at least in the scoring
   path). Training can keep them to support backfills; scoring should be strict.

4. **Parity test in CI.** For each retrain bundle PR, assert that every feature referenced
   by training preprocessing is also referenced by scoring's SELECT query (simple regex
   extraction + set diff).

## Verification

- The scoring job completes without KeyError.
- The predictions table has `COUNT(*) > 0` for today and the expected `model_version`.
- Spot-check a few rows: the new-feature values are populated (non-null) where expected.

## Notes

- This pattern is not Barry-specific. It appears in any ML pipeline where:
  training has a wide-open data loader (SELECT *, pandas `read_parquet` without column
  selection, etc.) and scoring has a narrowed explicit loader for throughput reasons.
- When the companion Dataform incremental trap is ALSO present, the table DROP + rebuild
  must happen BEFORE diagnosing the SELECT drift — otherwise the serving table itself is
  missing the columns and the signal is ambiguous. Do the Dataform rebuild first, verify
  the table schema, THEN investigate the scoring code.
- Back up the serving table before DROP + rebuild if any downstream consumer might read
  historical rows directly from it (most don't — they read from partitioned prediction
  outputs — but verify).

## See also

- `dataform-incremental-schema-change` — companion skill for Cause 3 (the CREATE TABLE
  IF NOT EXISTS silent-schema trap). Invoke first whenever a Dataform retrain adds
  columns and the downstream table appears stale.
- `gcp-cloudrun-image-stale-categorical-dtype` — related "silent schema drift between
  training and scoring" failure mode, but at the pandas-dtype level rather than the
  SELECT level.

## References

- Pandas KeyError on DataFrame slicing:
  https://pandas.pydata.org/docs/reference/api/pandas.errors.KeyError.html
- Dataform incremental semantics:
  https://cloud.google.com/dataform/docs/incremental-datasets
