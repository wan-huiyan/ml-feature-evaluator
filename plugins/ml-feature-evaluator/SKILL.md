---
name: ml-feature-evaluator
version: 2.3.0
date: 2026-04-16
author: wan-huiyan
# CHANGELOG 2.3.0 (2026-04-16) — Barry U S88 session frictions integrated:
#   - Q2: event-attendance leakage subpattern (post-outcome event filter)
#   - Q6-pre: stratify NULL bucket before rejecting on "0% in named bucket" finding
#   - Temporal Safety §1: dedup-query verification for upsert vs append mode
#   - Q8: multi-feature bundle recalibration table + per-feature SHAP rank expectations
#   - Cross-refs added to sister skills: null-bucket-hides-progressors-in-snapshot-training,
#     sf-bq-upsert-verify-before-createddate-gate, sentinel-real-disjunction-clamp
description: >
  Structured go/no-go evaluation for adding a new feature or data source to a production ML model.
  Triggers when the user asks any of these (or close variations):
  - "Should we add X to the model?"
  - "Is this feature/column/field worth integrating?"
  - "Evaluate whether adding [data source] improves our [model]"
  - "Run a go/no-go analysis on this candidate feature"
  - "Does this new table/CRM field/data feed have predictive value?"
  - "We have a coverage gap — would adding [source] help?"
  - "Is the incremental value of [feature] worth the engineering cost?"
  - "We want to expand our categorical from N to M buckets — worth it?"
  - "Compare these two data sources — which should we use for [signal]?"
  - "This column has lots of NULLs but looks predictive — should we use it?"
  - "Feature worth it?" / "Signal or noise?" / "Add this to the pipeline?"
  - "Assess whether [field] leaks or has real predictive power"
  Other natural-language variations that should trigger this skill:
  - "We got a new CRM/HubSpot/Salesforce field — is it useful for the model?"
  - "The model has a coverage gap for [population]. Would adding [table] help?"
  - "I just got access to a new engagement/demographic/financial table"
  - "Is it worth adding demographic/behavioral/credit data to the model?"
  - "Our top feature is X. Should we break it into sub-categories?"
  - "This new data feed overlaps with what we have — is there incremental value?"
  - "Would this real-time/streaming feature be worth the latency cost?"
  - "This text field has free-form notes — can we extract signal from it?"
  - "Is this high-cardinality ID field useful or just noise?"
  Does NOT trigger for: model selection, hyperparameter tuning, deployment,
  code review, pull request review, training window extension, unit testing,
  bug fixing, writing tests for pipelines, feature selection for a brand-new model
  from scratch, or general ML education questions. Specifically does NOT trigger
  for tasks about code quality, test coverage, or preprocessing bug fixes — even
  if those tasks mention features, pipelines, or NULLs.
scope: Evaluates ONE candidate feature or data source at a time against an existing (or planned) ML model. Not for bulk feature selection, model architecture, deployment, code review, testing, or training window decisions. For batch triage of multiple candidate signals, hand off to client-signal-triage first, which routes GO candidates here.
input: A candidate feature/column/table name + context about the target model (what it predicts, existing features)
output: >
  Structured diagnostic report with sections: Q0 (data quality), Q1-Q6 (core diagnostics),
  Q7-Q10 (contextual/advanced diagnostics), temporal safety assessment, go/no-go recommendation
  with confidence level, and (if GO) an implementation plan with monitoring spec and REVIEW_REQUEST block.
  Output is plain markdown suitable for parsing by downstream skills or review agents.
dependencies: Access to the data store (BigQuery, Postgres, Snowflake, pandas) containing the candidate feature and training data. Optionally, an existing trained model for Q7-Q10.
idempotent: true — rerunning on the same feature and data produces the same diagnostic results and recommendation
error_behavior: If data access fails, report which diagnostic step failed and which steps can still proceed. If Q0 reveals severe data quality issues (>80% NULL, <3 distinct values), halt and report before running Q1-Q10.
namespace: Outputs are scoped to the current conversation. No files written unless the user requests an implementation plan saved to disk.
version_compat: Works with any tabular ML model. Requires scikit-learn >= 1.0, shap >= 0.40. Supported versions include Python 3.8+.
composability:
  consumes_from: client-signal-triage (receives GO candidates with preliminary data availability assessment)
  hands_off_to:
    - ml-training-window-assessor (when evaluation reveals training window is the binding constraint, not the feature itself)
    - client-signal-triage (for batch evaluation of multiple candidate features — triage first, then evaluate GO candidates individually)
  output_contract: >
    Final output always contains: (1) a "## Recommendation" section with GO/NO-GO/DEFER verdict,
    (2) a "## Decision Framework" summary table with per-criterion pass/fail,
    (3) if GO, a "## Implementation Plan" section followed by a REVIEW_REQUEST block.
    Downstream skills can parse the verdict from the Recommendation header.
---

# ML Feature Evaluator

Assess whether a new data source or feature expansion justifies the engineering cost of integrating it into a production ML pipeline. The goal is to give a clear go/no-go recommendation backed by quantitative evidence, not just intuition.

## Workflow (follow in order)

1. Run Q0 data quality pre-check (fail fast on bad data)
2. Run the core diagnostic (Q1-Q6: distribution, outcome gradient, bucket decomposition, coverage gaps, entropy)
3. Run contextual diagnostics (Q7-Q8: conditional MI, incremental CV AUC) — these require the core results
4. Optionally run advanced diagnostics (Q9-Q10: SHAP interactions, permutation importance) when compute budget allows
5. Assess temporal safety using the Temporal Safety Checklist below
6. Apply the Decision Framework below → go/no-go
7. If GO: write implementation plan with monitoring spec
8. **MANDATORY: Spawn a fresh review agent** to critique the plan (see "Critical Review Step"). Do NOT present the plan to the user until the review is incorporated. If you cannot spawn a subagent (e.g., you are already running as a subagent), output a `## REVIEW NEEDED` section at the end listing what a reviewer should check — use the checklist from the Critical Review Step.
9. Incorporate review findings into the plan
10. Present final plan + diagnostic report to user

## Why this matters

Adding a new data source to a production pipeline has real costs: SQL changes, preprocessing parity across train/serve, testing, deployment risk. The worst outcome is spending a week integrating something that adds no predictive value. The second worst is *not* integrating something that would have been transformative. This skill helps you avoid both by running a structured diagnostic before committing to implementation.

## The Diagnostic Pattern

This is a reusable sequence that works for any categorical feature expansion. Adapt the specific SQL/queries to whatever data store the project uses (BigQuery, Postgres, Snowflake, pandas, etc.).

### Q0: Data Quality Pre-Check
**Purpose:** Fail fast on bad data before investing time in the full diagnostic.

Before evaluating predictive value, verify the candidate feature's data is sound:

- **NULL rate** — What percentage of rows have NULL/missing values? If >50%, the feature may not have enough signal to evaluate. Document the NULL rate for the decision framework.
- **Cardinality** — How many distinct values? A field with 1 value is useless; a field with 10,000 unique values may need bucketing before it's useful.
- **Distribution profile** — Is the distribution heavily skewed? If 95% of rows have the same value, the feature only distinguishes the remaining 5%.
- **Obvious data quality issues** — Impossible values (negative ages, future dates for historical events), encoding artifacts (mixed case, trailing whitespace), or sentinel values masquerading as real data (e.g., "N/A", "Unknown", "0000-00-00").

If Q0 reveals severe issues (>80% NULL, 1-2 distinct values, or widespread data corruption), stop and report the data quality problem before proceeding. A feature with bad data will produce misleading Q1-Q6 results.

This step is inspired by Evidently AI and Google's TFDV (TensorFlow Data Validation) — catching data problems early saves hours of wasted diagnostic work.

### Q1: Raw Distribution
**Purpose:** See what's actually in the data before making assumptions.

- Count distinct values in the new field(s)
- Show row counts per value
- Flag unexpected/undocumented codes
- Cross-reference against any client-provided documentation

This query often reveals surprises — codes you didn't know existed, NULL rates higher than expected, or values that don't match the documentation. Run this first so the rest of the analysis is grounded in reality.

### Q2: Outcome Gradient Across Proposed Categories
**Purpose:** The core value proposition — does the new data create meaningfully different outcome rates?

- Map raw values into proposed categories (the grouping you'd actually use as features)
- Compute the target outcome rate per category
- Include population counts per category

What you're looking for: a monotonic or near-monotonic gradient with at least 3x spread between the lowest and highest categories. If the spread is < 2x, the signal is probably too weak to justify integration.

**Leakage plausibility ceiling:** If the spread is >10x, or any single category has >95% outcome rate while the base rate is <30%, flag for leakage investigation before celebrating. Implausibly strong gradients often indicate the feature encodes information from after the label event (Kaufman, Rosset & Perlich, KDD 2011). Run the temporal safety checklist with extra scrutiny before proceeding. A legitimate >10x spread is possible (e.g., deposited vs. rejected students) but should be explainable by domain knowledge.

**Event-attendance / interaction-feature subpattern (common leakage trap):** When the candidate encodes attendance at an event or interaction (campaign_member, meeting attendance, RSVP, event_signup, workshop participation), explicitly filter for **post-outcome events** in Q1 before running Q2. Examples: a university predicting enrollment should exclude orientation, welcome, move-in, and new-student events — these are attended *after* enrollment by definition and retroactively "predict" the enrollment they came after. A SaaS product predicting churn should exclude post-churn offboarding-interview attendance. Run one query: `SELECT event_type, COUNT(*) FROM <source> WHERE status='Attended' GROUP BY event_type` and domain-check each type for temporal position relative to the label event. Apply the filter as a SQL `NOT REGEXP_CONTAINS` on the training-time CTE, not just in analysis — otherwise the filter won't exist in production. Typical impact: filters 1-3% of "Attended" rows but the rows removed are precisely the highest-outcome-rate ones; can turn a 6x apparent gradient into a true 3-4x.

### Q3: Bucket Decomposition
**Purpose:** Quantify how much information is hidden inside the current lumped bucket.

- Focus on the bucket that would be split by the expansion
- Show what % falls into each proposed sub-category
- Show outcome rates per sub-category

This answers: "How much are we losing by lumping these together?" If the sub-categories all have similar outcome rates, the current lumping is fine.

### Q4-Q5: Coverage Gap Analysis
**Purpose:** Determine whether the new source adds genuinely *new* information vs. what existing features already capture.

This is the most important and most often skipped step. For each signal the new source provides:

- Count entities where the new source has the signal
- Count entities where existing features already capture the same signal
- The gap = entities visible to the new source but invisible to existing features

Run one query per signal type (e.g., Q4 for acceptance status, Q5 for deposit status).

If existing features already cover >80% of what the new source provides, the incremental value may not justify the integration cost. If coverage gaps are >20%, the new source sees things existing features miss — that's strong evidence for integration.

### Q6-pre: Stratify NULL before rejecting on a "0% bucket" finding

**Purpose:** Avoid a false-negative verdict when temporal-gate SQL silently routes progressors to NULL.

Before concluding a candidate feature has no signal based on a 0% outcome rate in a named transitional bucket (In-Process, Pending, Lead-New, Quote-Sent), **always stratify the NULL bucket** of the gating column.

**Why:** Current-state-snapshot training tables with temporal gates of the form `WHEN decision_date IS NULL AND status IN (pre-decision codes) THEN status WHEN decision_date <= target_date THEN status ELSE NULL END` route students/customers **who were pre-decision at target_date and then decided after target_date** to NULL, not the named pre-decision bucket. The named bucket becomes self-selected for non-progressors (tautologically 0% outcome). The progressors — the population the feature is meant to predict — hide in NULL.

Run this sanity check:

```sql
SELECT
  stage_at_T,
  has_submitted,
  decision_happened_after_T,  -- derived from current decision_date > target_date
  COUNT(*) AS n,
  SUM(label) AS outcomes,
  AVG(label) * 100 AS rate_pct
FROM features
WHERE target_date = '<your_T>'
GROUP BY 1, 2, 3
ORDER BY rate_pct DESC;
```

If the NULL + submitted + decided-after-T bucket shows meaningful outcome rate, the candidate feature should be re-evaluated within that population, not the self-selected stuck population.

See sister skill: `null-bucket-hides-progressors-in-snapshot-training` for the full corrected-diagnostic pattern.

### Q6: Information Gain (Entropy) with Gain Ratio
**Purpose:** A single quantitative summary of how much the expansion helps, normalized for cardinality.

- Compute weighted binary entropy of the outcome within the bucket being split
- Compare: lumped (current) vs. split (proposed)
- Report the % entropy reduction
- **Also report gain ratio** = `information_gain / intrinsic_information(feature)` where `intrinsic_information = -Σ (|Si|/|S|) * log2(|Si|/|S|)` over the sub-categories

Entropy reduction >30% is a strong signal. >50% is exceptional. <10% suggests the split doesn't help much.

**Why gain ratio matters (Quinlan 1993):** Raw entropy reduction rewards more buckets — expanding from 4 to 20 categories will almost always show higher entropy reduction than 4 to 7, even if the extra categories add no useful signal. Gain ratio normalizes for this by dividing by the feature's own entropy (how many buckets, how evenly distributed). A high entropy reduction with a low gain ratio means the improvement comes from adding categories, not from the categories being informative. Report both numbers side by side.

### Q7: Conditional Mutual Information
**Purpose:** Does this feature add information *beyond what existing features already capture*?

This is the most common failure mode the original 6-query diagnostic missed: a feature looks great in isolation (high entropy reduction, strong gradient) but adds nothing because existing features already carry the same signal through different paths.

- Identify the top 3-5 existing features by importance (from the current model's feature importance or SHAP values)
- Compute `I(X_new; Y | X_existing)` — the mutual information between the candidate feature and the target, conditioned on the top existing features
- Compare against the unconditional `I(X_new; Y)` from Q6

**Interpretation:**
- If conditional MI is close to unconditional MI → the feature provides genuinely new information. Strong GO signal.
- If conditional MI is much lower than unconditional MI → existing features already capture most of this signal. The feature is redundant *in context*, even though it looks valuable *in isolation*.
- If conditional MI is higher than unconditional MI → the feature is synergistic with existing features (interaction effects). Very strong GO signal.

This step is inspired by Joint Mutual Information (JMI) from Brown et al. (JMLR 2012), which demonstrated that conditioning on existing features is strictly more informative than evaluating features independently. For practical computation, discretize continuous existing features into deciles and use empirical MI estimation.

### Q8: Incremental CV AUC (Ground Truth)
**Purpose:** The definitive answer — does adding this feature actually improve model performance?

Q0-Q7 are information-theoretic proxies. Q8 is the ground truth: train the model with and without the candidate feature and measure the difference.

- Using the existing model's training data and configuration, run k-fold cross-validation (k=5 is usually sufficient) in two configurations:
  - **Baseline:** Current feature set (no candidate)
  - **With candidate:** Current feature set + the proposed feature
- Report: mean AUC delta, standard deviation across folds, and whether the improvement is statistically significant (paired t-test or Wilcoxon signed-rank, p < 0.05)

**Interpretation:**
- Delta AUC > 0.005 with p < 0.05 → meaningful improvement. GO.
- Delta AUC 0.001-0.005 with p < 0.05 → marginal improvement. Weigh against integration cost.
- Delta AUC < 0.001 or p > 0.05 → no meaningful improvement. The information-theoretic proxies (Q2-Q7) may have been misleading. STOP and reassess.
- Delta AUC negative → the feature hurts performance (possible noise injection or feature collision). NO-GO.

**When to skip Q8:** If the candidate feature requires substantial pipeline work just to *create* the training data with the feature included (e.g., new SQL joins, new temporal guards, new preprocessing), Q8 may not be feasible as a quick diagnostic. In that case, rely on Q0-Q7 and note that Q8 was deferred. But if the data is readily available (e.g., you already have the column, you just haven't used it), always run Q8 — it takes minutes and provides definitive evidence.

**Multi-feature bundle recalibration** — when a retrain ships multiple features at once (cost-sensitive clients, batched releases), the single-feature AUC thresholds don't translate. For an N-feature bundle, use these adjustments:

| Bundle size | Expected ceiling (sum of single-feature Q8 estimates) | Promotion floor |
|---|---|---|
| 1 feature | Per-feature threshold (+0.005 typical) | +0.005 |
| 2 features | Σ single-feature Q8, conservatively discounted 20-30% for conditional-MI overlap | +0.003 per feature, or +0.004 aggregate |
| 3-5 features | Σ × 0.6-0.7 (heavier discount — overlap compounds) | +0.001-0.002 per feature aggregate |
| 6+ features | Use isolated ablation retrains post-hoc; aggregate Q8 cannot attribute per feature | Subjective — typically aggregate +0.005 minimum to justify the bundle cycle |

Always pair a multi-feature bundle with **per-feature SHAP rank expectations** for post-deploy attribution (e.g., "tenure top-20, A* variant top-10, ready_to_review top-40 or remove"). If per-feature SHAP rank falls below expectation, plan a lightweight pruning retrain (same model version, minor bump like v6.1.1) rather than a full rollback. This keeps the bundle economics intact while letting you retire features that didn't earn their feature-count slot.

This step is inspired by mlxtend's SequentialFeatureSelector (5.1k GitHub stars) and the Boruta algorithm (1.6k stars), both of which use model-in-the-loop evaluation as the gold standard for feature value.

### Q9: SHAP Interaction Detection
**Purpose:** Detect whether the candidate feature interacts synergistically with existing features, beyond what Q7 (conditional MI) captures.

- Use `shap.TreeExplainer.shap_interaction_values()` for pairwise interactions on tree models
- For any-order interactions, use `shapiq` library (702 stars, NeurIPS 2024 — Muschalik et al.)

```python
import shap
explainer = shap.TreeExplainer(model)
interaction_values = explainer.shap_interaction_values(X_test)
# Shape: (n_samples, n_features, n_features)
# Diagonal = main effects, off-diagonal = pairwise interactions
```

**Interpretation:** If the candidate feature shows strong off-diagonal interaction values with existing features, it provides synergistic signal that Q7's conditional MI might miss. Interaction strength > 10% of the main effect is meaningful.

**Warning:** Computationally expensive (quadratic in features). For large feature sets (>50 features), use `shapiq` with ProxySPEX approximator or limit to top-10 existing features by importance.

**References:** Lundberg & Lee, NeurIPS 2017; Muschalik et al., "shapiq: Shapley Interactions for ML", NeurIPS 2024

### Q10: Permutation Importance with Cross-Validation
**Purpose:** Model-agnostic validation that the feature's importance generalizes across data splits, not just the training set.

- ALWAYS compute on held-out data, not training data. Training-set importance reflects memorization, not generalization.
- Use sklearn's `permutation_importance` per CV fold with `n_repeats >= 10`

```python
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_importances = []
for train_idx, val_idx in kf.split(X):
    model.fit(X[train_idx], y[train_idx])
    result = permutation_importance(model, X[val_idx], y[val_idx], n_repeats=10, random_state=42)
    all_importances.append(result.importances_mean)
mean_importances = np.mean(all_importances, axis=0)
```

**Beware correlated features:** When two features are correlated, permuting one leaves the other intact, deflating both features' importance. Remedies: group correlated features, use SHAP-based methods (Q9), or use LOFO (`lofo-importance`, 863 stars).

**Interpretation:** If the candidate feature's mean permutation importance across folds is > 2 standard deviations from zero, it has robust predictive value. If importance is high on train but low on validation folds, the feature is overfitting.

**Alternative:** LOFO (Leave-One-Feature-Out) via `lofo-importance` library — retrains the model without the feature and measures performance drop. More expensive but handles correlated features correctly.

**When to skip:** If Q8 already showed definitive AUC results, Q10 adds confirmation but may not change the decision. Skip if compute budget is tight.

**References:** scikit-learn 1.8 Permutation Importance Guide; Kraev et al., "shap-select", arXiv 2024

## Temporal Safety Checklist

Before recommending integration, check for temporal leakage — using future information to predict present outcomes.

1. **Classify the source into one of three types** before designing temporal guards:

   | Type | Description | Temporal Guard Strategy |
   |------|-------------|------------------------|
   | **Current-state snapshot** | One row per entity, latest state only. CRM tables (Salesforce, HubSpot) are often this. | Gate with date columns (`decision_date <= target_date`). Conservative: intermediate states are lost. |
   | **Event log / append-only** | One row per event, immutable once written. Behavioral tracking, application logs. | Filter by `event_date <= target_date`. No state reconstruction needed. |
   | **Versioned history** | Multiple rows per entity with validity windows. SCD Type 2, CDC, changelogs, audit tables. | Range query: `valid_from <= target_date AND (valid_to IS NULL OR valid_to > target_date)`. Provides exact point-in-time state. |

   A field showing "Accepted" today may not have been "Accepted" at a historical training date. Snapshot tables cannot reconstruct intermediate states; versioned history tables can. **Always check whether a versioned history source exists for the same data before relying on snapshot temporal guards** — information gain measurements from snapshot data may be understated due to guard-induced information loss.

   **Verify the classification with a dedup query before trusting `created_date` / `modified_date` as a temporal gate.** Naming conventions are unreliable — the same ingestion tooling may produce upsert-mode for one SF entity and append-mode for another. Run:

   ```sql
   SELECT
     (SELECT COUNT(*) FROM table) AS total_rows,
     (SELECT COUNT(*) FROM (SELECT <pk_column>, COUNT(*) c FROM table GROUP BY 1 HAVING c > 1)) AS pk_dupes,
     (SELECT COUNT(*) FROM (SELECT <natural_key_cols>, COUNT(*) c FROM table GROUP BY <natural_key_cols> HAVING c > 1)) AS natural_key_dupes
   ```

   Interpret:
   - `pk_dupes = 0` AND low `natural_key_dupes` → **upsert-mode**; `created_date` = original record creation date (status updates overwrite in place). Retrospective status updates cause ADR-0009-class leakage but are consistent with other SF features.
   - `pk_dupes > 0` → **append-mode** (changelog); `created_date` on each row = change-event timestamp. Gate semantics are historically precise.
   - Document the finding in the feature SQL comment so the next reviewer doesn't re-ask.

   See sister skill: `sf-bq-upsert-verify-before-createddate-gate` for the full verification protocol with SQL templates and regression-guard assertion.

   When both a snapshot and versioned source exist for the same entity, consider a **dual-source strategy**: use the versioned source for training (exact historical state) and the snapshot for serving (real-time current state). This is safe when serving always uses `target_date = today`, where the snapshot's current state IS the correct point-in-time state. Document the asymmetry explicitly so future developers understand why training and serving SQL differ.

   When both a snapshot and versioned source exist for the same entity, consider a **dual-source strategy**: use the versioned source for training (exact historical state) and the snapshot for serving (real-time current state). This is safe when serving always uses `target_date = today`, where the snapshot's current state IS the correct point-in-time state. Document the asymmetry explicitly so future developers understand why training and serving SQL differ.

2. **Are there date columns that can serve as temporal guards?** Look for `decision_date`, `created_date`, `modified_date`, `status_change_date`. Gate feature visibility on `date_column <= target_date`. For versioned history sources, use the validity window columns instead (`valid_from`/`valid_to` or equivalent).

3. **Do the date columns mean what they say?** "Last modified" dates often record the most recent touch, not the initial event. Validate by checking ordering against related events. If anomalous ordering is common (>10%), treat as "last modified" and use conservatively.

4. **What happens when the date is NULL or in the future?** Default to the most conservative interpretation — treat as "unknown" or "not yet occurred."

5. **Leakage is relative to the label, not intermediate milestones.** An ordering anomaly vs. an *intermediate* event (e.g., `decision_date` AFTER `deposit_date`) is NOT leakage unless it also post-dates the *label* event.

6. **Validate co-occurrence before excluding fields.** Run `SELECT COUNT(*) WHERE field_x IS NOT NULL AND gating_field IS NULL` — if 0, `gating_field` safely bounds `field_x`.

7. **Observation point ≠ journey step.** `target_date` is where the model *observes*; the real-world process has its own timeline. Conflating these causes temporal guard design errors.

8. **Proxy leakage through causal structure.** A temporally safe feature can still leak if it's a near-deterministic proxy for the label (e.g., 99% of orientation attendees enroll). Check: does the feature's causal path go *through* the outcome? (Kapoor & Narayanan, 2023)

9. **Preprocessing leakage.** Full-dataset statistics (mean/std normalization, target encoding) before train/test split = invisible leakage. All preprocessing must use training-fold statistics only. (Yang et al., 2022)

## Decision Framework

The expansion is worth it if **most** of these hold:

| Criterion | Suggested Threshold | Notes |
|-----------|-------------------|-------|
| Data quality (Q0) | <50% NULL, >2 distinct values | Fail fast on bad data |
| Outcome gradient (Q2) | >3x spread | Between lowest and highest proposed categories |
| Leakage plausibility (Q2) | <10x spread | Flag >10x for leakage investigation |
| Coverage gaps (Q4-Q5) | >20% | New source sees entities existing features miss |
| Entropy reduction (Q6) | >30% | Within the bucket being split |
| Gain ratio (Q6) | >0.3 | Normalized for cardinality — penalizes many-bucket expansions |
| Conditional MI (Q7) | >50% of unconditional MI | Feature adds new info beyond existing features |
| Incremental AUC (Q8) | >0.005 with p<0.05 | Ground truth — definitive when feasible |
| SHAP interaction (Q9) | >10% of main effect | Strong interaction = synergistic value |
| Permutation importance (Q10) | >2σ from zero across folds | Robust generalization signal |
| Population per category | Hundreds+ | Each sub-category needs enough volume for the model to learn from |
| Temporal safety | Feasible guards exist | Date columns available and validated |

**Important:** All thresholds above are heuristic starting points derived from practitioner experience, not empirically validated universal cutoffs. They work well as defaults but should be calibrated to your domain, data scale, and pipeline complexity. A 2x outcome gradient may be transformative in a low-signal domain; a 5x gradient may be insufficient if integration cost is extreme. Adjust based on:
- **Pipeline complexity:** Higher integration cost → higher bar for evidence
- **Project timeline:** Wrapping up soon → only pursue if gains are exceptional
- **Feature count:** Already feature-rich → need stronger incremental evidence

## Implementation Planning

If the diagnostic says yes, produce an implementation plan covering:

1. **Data source changes** — What tables/columns to add, what joins are needed
2. **Temporal guards** — Exact SQL for date-based gating
3. **Feature encoding** — How raw values map to model features (categorical levels, groupings)
4. **Preprocessing parity** — Every file that needs updating (train, serve, config, data loader)
5. **Display/reporting** — Feature names, data dictionary, client-facing labels. **Full-document audit:** When a new data source is added, grep ALL client-facing documents (PDFs, slides, reports) for every section that references data sources, signal counts, or source lists — not just the section you're editing. Common miss: updating an appendix but leaving the executive summary saying "two data sources" when there are now three.
6. **Testing strategy** — How to validate before deploying (A/B, shadow scoring, offline eval)
7. **Rollback plan** — How to revert if something goes wrong
8. **Monitoring spec** — Post-deployment feature health contract (inspired by Uber's Model Excellence Scores and Google's TFDV):
   - **Expected NULL rate** — what % of NULLs is acceptable in production? Set a threshold and alert above it.
   - **Expected distribution** — define the expected value distribution (e.g., "60% In Process, 20% Accepted, 15% Rejected, 5% Deposited"). Alert on significant deviations.
   - **Drift threshold** — what level of distribution shift between training and serving triggers investigation? A common choice is PSI (Population Stability Index) > 0.2.
   - **Feature freshness** — does the serving pipeline compute this feature at the same granularity/recency as training? For batch pipelines, this is usually fine; for real-time, stale cache reads can introduce train/serve skew.
   - **Unknown value alerting** — for categoricals, log and alert when unrecognized values appear (new status codes, new tiers). This catches the "client added a code we didn't map" failure mode.

   **Concept drift monitoring** — Track distributional shifts post-deployment using PSI (>0.2 = significant), Wasserstein distance (continuous features), or KS test (p < 0.05, but oversensitive at >50k rows). Tools: Evidently AI for dashboards, alibi-detect for pipeline tests, NannyML CBPE for performance estimation without ground truth labels.

## Critical Review Step

This step is **mandatory** — do not skip it. The review catches blind spots that accumulate during the diagnostic (e.g., missing files in the update checklist, unsafe temporal guards, simpler alternatives). In testing, 0/3 evaluations triggered this step when it was optional. That's why it's now a required gate.

### Two-stage review pattern

The review works best when a **fresh agent** with no prior context reads the plan cold. Since skills often run as subagents (which cannot spawn nested subagents), use this two-stage approach:

**Stage 1 (you, the skill):** After producing the implementation plan, append a `REVIEW_REQUEST` block at the end of your response. This signals the caller (top-level agent or user) to spawn the reviewer on your behalf.

Format your output like this:

```
## Implementation Plan
[your plan here]

---
REVIEW_REQUEST: The implementation plan above requires a fresh-perspective review before presenting to the user. Please spawn a review agent with the following prompt:

You are a senior ML engineer reviewing an implementation plan for adding a new data source
to a production ML pipeline. Your job is to find flaws, gaps, and risks.

Review the following implementation plan critically:
[paste the plan above]

Cross-reference against these sources:
- The diagnostic results: [path to diagnostic report]
- The current codebase: [relevant file paths]
- Client documentation: [path to reference docs, if any]

Check for:
1. **Preprocessing parity gaps** — Does the plan update ALL files that need to stay in sync?
2. **Temporal leakage** — Are the proposed date guards actually safe? What edge cases are missed?
3. **Missing rollback plan** — What happens if this breaks production scoring?
4. **Display/label correctness** — Are all proposed display names validated against source documentation? Are ALL sections of client-facing documents consistent (executive summary, data sources, appendices)?
5. **Population edge cases** — What happens for NULL values, missing joins, new codes that appear after deployment?
6. **Testing gaps** — What isn't covered by the proposed testing strategy?
7. **Scope creep** — Is the plan doing more than necessary? Could it be simpler?

Output a structured review with: PASS (no issues), WARN (minor concerns), or BLOCK (must fix before proceeding) for each area.
```

**Stage 2 (the caller):** The top-level agent or user spawns the review agent using the prompt above. The reviewer reads the plan cold and produces a structured critique. The caller then either:
- Sends the review findings back to the skill for plan revision, or
- Presents both the plan and review to the user for their judgment.

**Why two stages?** Subagents cannot spawn nested subagents. The old approach (skill tries to spawn reviewer directly) silently fell back to self-review 100% of the time in practice, defeating the purpose. The two-stage pattern ensures a genuinely fresh perspective every time by delegating the spawn to the caller, which always has Agent tool access.

**If you ARE running as a top-level agent** (not a subagent) and can confirm you have Agent tool access, you may spawn the reviewer directly instead of outputting REVIEW_REQUEST. But default to the two-stage pattern — it works in all contexts.

## Implementation Plan Review — Common Bugs Checklist

Watch for these patterns while writing the plan. Each is a silent-breakage risk that reviewers should verify.

| Bug | Rule | Why It Matters |
|-----|------|---------------|
| **Backward-compat category name mismatch** | Fallback/`else` branch must emit the **old** category strings the existing model was trained on, not the new names | XGBoost silently produces wrong predictions on unseen categories |
| **Schema vs SELECT drift** | Only declare columns in the output schema that appear in the final SELECT | Dataform/dbt enforce alignment — intermediate CTE columns in the schema fail at compile time |
| **Dead code from temporal gating** | When temporal guards NULL out most records' status codes, the fallback path becomes the primary classification path — design it accordingly | Tests written against explicit code sets miss that 99% of records flow through the fallback |
| **Snapshot tables lack history** | Snapshot tables show terminal state only; check for versioned history sources (SCD, CDC, changelogs) that provide exact point-in-time state. Test ALL ID-like columns for join overlap — don't conclude "no bridge" from one column. | Affects populations where entities transition through states; versioned sources eliminate the gap entirely |
| **Explicit sets vs prefix matching** | Use explicit code sets but add monitoring for unrecognized values: `logger.warning(f"Unknown codes: {observed - known}")` | Explicit sets are safe today but silently misclassify future codes the client adds |
| **Fallback sentinel conflation** | Check `df['status_code'].isna()` instead of `df[stage_col] == 'No Application'` | Sentinel values conflate "genuinely missing" with "temporally gated away" — fragile if defaults change |
| **Category ordering** | Order categories least-to-most positive: No Application → Closed → In Process → Accepted → Deposited | Non-monotonic ordinal encoding confuses humans and can affect linear model components |
| **Re-export noise in versioned history** | Filter consecutive rows with identical state using `LAG()` when computing transition-based features | SCD/CDC tables often contain system re-exports that inflate transition counts |

## Source Priority When Two Sources Cover the Same Event

- **Coverage determines COALESCE order** — put the higher-coverage source first, not the "richer" one
- **CRM vs behavioral sources are usually complementary, not substitutable:** CRM captures state/coverage; events capture frequency/recency. Keep both unless they capture truly identical information
- When a CRM field covers 100% and a behavioral event covers 27% of the same real-world event, the CRM is the primary source

## Related Skills

- **`ml-training-window-assessor`**: When the question is "can we extend the training window?" rather than "should we add feature X?" — covers per-output label validity, lookforward bridging, and companion model vs extended training architecture decisions.
- **`client-signal-triage`**: When a client sends a batch of candidate signals (email, field list, meeting notes) — triages into GO/DEFER/NO-GO by data availability, then routes GO candidates here for full diagnostic. Use triage first when evaluating >3 candidates.

## Open-Source Tools and Benchmarks

| Tool | Stars | What It Does | URL |
|------|-------|-------------|-----|
| shap | 25.2k | SHAP values + pairwise interaction detection | github.com/shap/shap |
| shapiq | 702 | Any-order Shapley interaction computation | github.com/mmschlk/shapiq |
| scikit-learn | 62k+ | `permutation_importance` — model-agnostic feature importance | github.com/scikit-learn/scikit-learn |
| lofo-importance | 863 | Leave-One-Feature-Out importance with CV | github.com/aerdem4/lofo-importance |
| shap-select | 38 | Lightweight feature selection via SHAP regression | github.com/transferwise/shap-select |
| Evidently AI | 7.3k | ML observability; 20+ drift detection methods | github.com/evidentlyai/evidently |
| NannyML | 2.1k | Performance estimation without ground truth | github.com/NannyML/nannyml |
| alibi-detect | 2.5k | Outlier, adversarial, and drift detection | github.com/SeldonIO/alibi-detect |
| river | 5.8k | Online/streaming ML with drift detectors | github.com/online-ml/river |
| Boruta-Shap | 650 | Boruta + SHAP feature selection | github.com/Ekeany/Boruta-Shap |

## References

Lundberg & Lee, NeurIPS 2017 (SHAP) · Muschalik et al., NeurIPS 2024 (shapiq) · SHAP-IQ, NeurIPS 2023 · Quinlan, 1993 (gain ratio) · Brown et al., JMLR 2012 (JMI/conditional MI) · Kaufman et al., KDD 2011 (leakage) · Kapoor & Narayanan, Patterns 2023 (proxy leakage) · Yang et al., ASE 2022 (preprocessing leakage) · Kraev et al., arXiv 2024 (shap-select) · Molnar, "Interpretable ML" Ch. 18

## Anti-Patterns to Avoid

- **Skipping coverage gap analysis:** The most common mistake. A new source looks amazing in isolation but adds nothing over existing features.
- **Trusting field names:** "application_status" might mean current status, historical status, or something else entirely. Always validate against raw data and documentation.
- **Guessing code meanings from abbreviations:** AF doesn't mean "Application Filed" — it means "Accepted, Pending Final Transcript." Cross-reference against client documentation.
- **Assuming snapshot tables have history:** If the table gets upserted, yesterday's values are gone. Check sibling datasets for versioned history (SCD, CDC, changelogs, audit tables) before accepting the snapshot's temporal limitations. A "zero overlap" on one join key doesn't mean no bridge exists — test ALL ID-like columns before concluding.
- **Evaluating features from snapshots without checking for versioned alternatives:** Information gain and coverage gap measurements from snapshot data may be understated due to temporal guard information loss. If a versioned source exists, re-run the evaluation using exact point-in-time state — the results may be significantly different.
- **Re-export noise in versioned history tables:** Consecutive rows with identical state (but different validity timestamps) are often system re-exports, not real transitions. Filter with `LAG()` when computing transition-based features like status change counts.
- **Forgetting to monitor for new codes:** Explicit code sets are safe until the client adds a new status code that silently falls to the default. Always add runtime logging for unrecognized values.
- **Putting the lower-coverage source first in COALESCE:** Coverage determines priority. Don't default to "behavioral source primary, CRM fallback" when the CRM covers more students.
- **Updating appendix but not body of client docs:** When adding a new data source, grep the entire document generator for "data source", source counts ("two", "three"), and source names. Executive summaries, methodology sections, and appendices must all be consistent. A client seeing "two data sources" on page 1 and Snowflake on page 8 loses trust.
- **Flagging intermediate-milestone ordering anomalies as leakage:** Check ordering against the *label*, not against other intermediate events in the pipeline.
