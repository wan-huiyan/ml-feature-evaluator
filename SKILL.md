---
name: ml-feature-evaluator
description: Structured go/no-go evaluation for adding a new feature or data source to a production ML model. Use when the user asks "should we add X to the model?", discusses a new table/field/CRM source, wants to expand an existing categorical, or mentions coverage gaps. Runs a 10-step diagnostic (data quality pre-check → distribution → outcome gradient → bucket decomposition → coverage gaps × 2 → entropy with gain ratio → conditional mutual information → incremental CV AUC), checks temporal safety for snapshot tables, and produces a decision-backed implementation plan with monitoring spec. Also triggers on: feature expansion, new data feed, comparing data sources, assessing incremental value of a signal.
---

# ML Feature Evaluator

Assess whether a new data source or feature expansion justifies the engineering cost of integrating it into a production ML pipeline. The goal is to give a clear go/no-go recommendation backed by quantitative evidence, not just intuition.

## Workflow (follow in order)

1. Run Q0 data quality pre-check (fail fast on bad data)
2. Run the core diagnostic (Q1-Q6)
3. Run contextual diagnostics (Q7-Q8) — these require the core results
4. Assess temporal safety
5. Apply decision framework → go/no-go
6. If GO: write implementation plan with monitoring spec
5. **MANDATORY: Spawn a fresh review agent** to critique the plan (see "Critical Review Step"). Do NOT present the plan to the user until the review is incorporated. If you cannot spawn a subagent (e.g., you are already running as a subagent), output a `## REVIEW NEEDED` section at the end listing what a reviewer should check — use the checklist from the Critical Review Step.
6. Incorporate review findings into the plan
7. Present final plan + diagnostic report to user

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
- **Obvious data quality issues** — Impossible values (negative ages, future dates for historical events), encoding artifacts (mixed case, trailing whitespace), or placeholder values masquerading as real data (e.g., "N/A", "Unknown", "0000-00-00").

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

**Leakage plausibility ceiling:** If the spread is >10x, or any single category has >95% outcome rate while the base rate is <30%, flag for leakage investigation before celebrating. Implausibly strong gradients often indicate the feature encodes information from after the label event (Kaufman, Rosset & Perlich, KDD 2011). Run the temporal safety checklist with extra scrutiny before proceeding. A legitimate >10x spread is possible (e.g., short trips from Midtown vs long trips from outer boroughs) but should be explainable by domain knowledge.

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

Run one query per signal type (e.g., Q4 for short-trip vs long-trip classification, Q5 for peak vs off-peak patterns).

If existing features already cover >80% of what the new source provides, the incremental value may not justify the integration cost. If coverage gaps are >20%, the new source sees things existing features miss — that's strong evidence for integration.

### Q6: Information Gain (Entropy) with Gain Ratio
**Purpose:** A single quantitative summary of how much the expansion helps, normalized for cardinality.

- Compute weighted binary entropy of the outcome within the bucket being split
- Compare: lumped (current) vs. split (proposed)
- Report the % entropy reduction
- **Also report gain ratio** = `information_gain / intrinsic_information(feature)` where `intrinsic_information = -Σ (|Si|/|S|) * log2(|Si|/|S|)` over the sub-categories

Entropy reduction >30% is a strong signal. >50% is exceptional. <10% suggests the split
doesn't help much. These are practitioner heuristics, not published thresholds.

**Gain ratio interpretation:** Use gain ratio for **ranking**, not as a pass/fail test.
Following Quinlan (1993), first filter candidates whose information gain is below the
average across all candidates, then rank the survivors by gain ratio. The highest gain
ratio among above-average-info-gain candidates is preferred. No absolute cutoff exists
in the literature — a gain ratio of 0.2 may be excellent in one context and mediocre
in another.

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
- p < 0.05 (DeLong test for independent models, bootstrap for nested models) with
  positive delta AUC → statistically significant improvement. Report the delta and let
  the practitioner judge materiality in context. GO.
- p < 0.05 but very small delta (e.g., <0.001) → statistically significant but may not
  justify integration cost. Weigh against pipeline complexity.
- p > 0.05 → no statistically significant improvement. The information-theoretic proxies
  (Q2-Q7) may have been misleading. STOP and reassess.

**Note:** No fixed AUC delta threshold is scientifically justified (Cook 2007). AUC is
insensitive to many genuinely useful improvements. Consider supplementing with Net
Reclassification Improvement (NRI) or Integrated Discrimination Improvement (IDI) for
a fuller picture (Pencina et al. 2008).
- Delta AUC negative → the feature hurts performance (possible noise injection or feature collision). NO-GO.

**When to skip Q8:** If the candidate feature requires substantial pipeline work just to *create* the training data with the feature included (e.g., new SQL joins, new temporal guards, new preprocessing), Q8 may not be feasible as a quick diagnostic. In that case, rely on Q0-Q7 and note that Q8 was deferred. But if the data is readily available (e.g., you already have the column, you just haven't used it), always run Q8 — it takes minutes and provides definitive evidence.

This step is inspired by mlxtend's SequentialFeatureSelector (5.1k GitHub stars) and the Boruta algorithm (1.6k stars), both of which use model-in-the-loop evaluation as the gold standard for feature value.

## Temporal Safety Checklist

Before recommending integration, check for temporal leakage — using future information to predict present outcomes.

1. **Is the source table a snapshot or historical?** Some data sources (e.g., zone lookup tables, rate cards) are current-state snapshots, not event logs. A zone boundary showing "Midtown" today may not have had the same boundary at the historical training date.

2. **Are there date columns that can serve as temporal guards?** Look for `pickup_datetime`, `created_date`, `modified_date`, `effective_date`. Gate feature visibility on `date_column <= target_date`.

3. **Do the date columns mean what they say?** "Last modified" dates often record the most recent touch, not the initial event. Validate by checking ordering against related events. If anomalous ordering is common (>10%), treat as "last modified" and use conservatively.

4. **What happens when the date is NULL or in the future?** Default to the most conservative interpretation — treat as "unknown" or "not yet occurred."

5. **Temporal leakage is relative to the label, not intermediate milestones.** A field being anomalously ordered relative to another *intermediate* event (e.g., `pickup_datetime` AFTER `zone_boundary_update`) is NOT leakage unless the field also post-dates the *label* event. Before flagging an ordering anomaly as leakage, explicitly identify the label field and ask: "does this feature contain information from after the label, or just after some earlier step?" If the answer is the latter, it's conservative under-counting, not leakage.

6. **Validate field co-occurrence with a zero-count query before excluding a field.** If a field has no dedicated timestamp but you suspect another field implicitly bounds it (e.g., all zone-mapped trips have a `pickup_datetime`), run: `SELECT COUNT(*) WHERE field_x IS NOT NULL AND gating_field IS NULL`. If the result is 0, `gating_field` safely bounds `field_x`. This is cheap and resolves uncertainty.

7. **"Observation point" vs. "journey step" framing.** In training pipelines with a `target_date`, be explicit: `target_date` is where the model *observes*, not a step in the process being modeled. The real-world process has its own timeline (e.g., trip requested → pickup → en route → dropoff). At `target_date`, the model sees whatever state is available at that point. Conflating the two causes temporal guard design errors.

8. **Proxy leakage through causal structure.** A feature might not directly encode future information, but it could be a proxy for the label through a causal path. For example, "number of prior trips from this zone" might be safe temporally (all trips happened before the label date), but if 99% of high-frequency zones also have short median durations, the feature is essentially encoding the label through a near-deterministic proxy. Check: does the feature have a causal path that goes *through* the outcome? If so, is the correlation explained by the causal structure (legitimate signal) or by the feature being a disguised version of the label (leakage)? (Kapoor & Narayanan, *Patterns* 2023)

9. **Preprocessing leakage.** Feature engineering that uses statistics computed from the full dataset (train + test) introduces leakage even if the underlying data is clean. Common examples: normalizing with full-dataset mean/std before train/test split, target encoding using all rows, or imputing with population-level statistics. This type of leakage is invisible in the data — it only exists in the code. The implementation plan review should explicitly check that all preprocessing uses only training-fold statistics. (Yang et al., ASE 2022)

## Decision Framework

The expansion is worth it if **most** of these hold:

| Criterion | Suggested Threshold | Provenance |
|-----------|-------------------|------------|
| Data quality (Q0) | <50% NULL, >2 distinct values | Practitioner heuristic. Fail fast on bad data |
| Outcome gradient (Q2) | >3x spread | Practitioner heuristic. Between lowest and highest proposed categories |
| Leakage plausibility (Q2) | <10x spread | Practitioner heuristic. Flag >10x for leakage investigation; inspired by Kaufman et al. (2012) |
| Coverage gaps (Q4-Q5) | >20% | Practitioner heuristic. New source sees entities existing features miss |
| Entropy reduction (Q6) | >30% | Practitioner heuristic. Within the bucket being split |
| Gain ratio (Q6) | Rank-based: above-average info gain filter, then highest gain ratio | Quinlan (1993). Gain ratio is a ranking metric, not a pass/fail threshold. C4.5 filters candidates whose info gain is below the average, then ranks by gain ratio. No absolute cutoff exists in the literature. |
| Conditional MI (Q7) | Rank-based: JMI/CMIM score vs existing features | Brown et al. (2012). MI-based feature selection uses ranking (top-K or greedy forward selection), not percentage retention cutoffs. Report the conditional MI value and compare against the unconditional MI to show how much information survives conditioning — higher retention = less redundancy. |
| Incremental AUC (Q8) | p<0.05 via DeLong test or bootstrap | DeLong et al. (1988) for independent models; use bootstrap for nested models (Demler et al. 2012). No fixed AUC delta threshold is scientifically justified — report the delta and p-value, let the practitioner judge materiality in context. Cook (2007) showed AUC is insensitive to many meaningful improvements; consider supplementing with NRI/IDI. |
| Population per category | 20-300 per category (domain-dependent) | LightGBM default: 20 per leaf. Van der Ploeg et al. (2014): 200 events per variable for stable AUC in tree models. Cochran rule for chi-squared: >=5 expected per cell. Use 20+ as minimum, 200+ as comfortable. |
| Temporal safety | Feasible guards exist | Kapoor & Narayanan (2023). Date columns available and validated |

**Provenance note:** Thresholds marked "practitioner heuristic" are not derived from
published standards. They are starting points calibrated on production use cases. Adjust
based on:
- **Pipeline complexity:** Higher integration cost → higher bar for evidence
- **Project timeline:** Wrapping up soon → only pursue if gains are exceptional
- **Feature count:** Already feature-rich → need stronger incremental evidence
- **Sample size:** PSI thresholds are sample-size dependent (Yurdakul 2018); smaller
  datasets need wider confidence bands

## Implementation Planning

If the diagnostic says yes, produce an implementation plan covering:

1. **Data source changes** — What tables/columns to add, what joins are needed
2. **Temporal guards** — Exact SQL for date-based gating
3. **Feature encoding** — How raw values map to model features (categorical levels, groupings)
4. **Preprocessing parity** — Every file that needs updating (train, serve, config, data loader)
5. **Display/reporting** — Feature names, data dictionary, client-facing labels
6. **Testing strategy** — How to validate before deploying (A/B, shadow scoring, offline eval)
7. **Rollback plan** — How to revert if something goes wrong
8. **Monitoring spec** — Post-deployment feature health contract (inspired by Uber's Model Excellence Scores and Google's TFDV):
   - **Expected NULL rate** — what % of NULLs is acceptable in production? Set a threshold and alert above it.
   - **Expected distribution** — define the expected value distribution (e.g., "32% Manhattan, 25% Brooklyn, 20% Queens, 15% Bronx, 8% Staten Island"). Alert on significant deviations.
   - **Drift threshold** — what level of distribution shift between training and serving triggers investigation? A common choice is PSI (Population Stability Index) > 0.2.
   - **Feature freshness** — does the serving pipeline compute this feature at the same granularity/recency as training? For batch pipelines, this is usually fine; for real-time, stale cache reads can introduce train/serve skew.
   - **Unknown value alerting** — for categoricals, log and alert when unrecognized values appear (new status codes, new tiers). This catches the "client added a code we didn't map" failure mode.

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
4. **Display/label correctness** — Are all proposed display names validated against source documentation?
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

## Implementation Plan Review — Common Bugs

After producing a plan and having it reviewed (either by a fresh subagent or a human), these are the patterns most likely to surface. Build awareness of them while *writing* the plan so the reviewer catches fewer issues.

### Backward-Compat Category Name Mismatch

When expanding a categorical (e.g., 5 boroughs → 30 neighborhoods), the fallback/`else` branch for old data must use the **old** category names, not the new ones. If old models were trained on `"Manhattan"` and the fallback produces `"Manhattan – Midtown"`, XGBoost sees an unseen category and silently produces wrong predictions. This is the single most common silent-breakage bug in categorical expansions.

**Rule:** The fallback path should emit the exact category string the old model was trained on.

### Schema Declaration vs Output SELECT Drift

When adding intermediate columns (used for temporal gating inside a CTE but not needed in the final output), don't add them to the output schema declaration unless they're also in the output SELECT. Dataform and similar tools enforce alignment — listing a column in the schema that isn't selected will fail at compile time.

**Rule:** Only declare columns in the schema that appear in the final SELECT.

### Dead Code from Temporal Gating

When temporal guards gate on a date column that is NULL for most records in a category, the guard effectively nullifies the status code for those records. The explicit code set you wrote to match them becomes dead code — the records reach the correct bucket only through the fallback path.

**Concrete example:** You define `OUTER_BORO_ZONES = {'Fordham', 'Pelham', 'Riverdale'}` and write `df.loc[bronx_mask & zone.isin(OUTER_BORO_ZONES), zone_col] = 'Outer Bronx'`. But your SQL temporal guard does `CASE WHEN zone_effective_date IS NOT NULL AND zone_effective_date <= target_date THEN pickup_zone ELSE NULL END`. If 99% of Bronx trips have no `zone_effective_date`, their zone comes through as NULL. Python fills NULL with `""`. `""` isn't in `OUTER_BORO_ZONES`. Those records fall to the fallback: `df.loc[bronx_mask & (df[zone_col] == default), zone_col] = 'Outer Bronx'`. The result is correct — but the explicit zone set matched almost nobody. The fallback is doing all the work.

**Why this matters beyond code review:** When advising someone on implementation, proactively warn them that temporal gating + explicit code sets interact in surprising ways. The code sets handle the minority of records with date columns populated; the fallback handles the majority. If they don't understand this, they'll write tests against the code sets and miss that the fallback is the real classification path. If the fallback has a bug, it affects 99% of records, not 1%.

**Rule:** When writing code sets paired with temporal guards, trace the actual data flow end-to-end. Document which path is primary vs secondary. Design your fallback as carefully as your explicit matching — it's the main path, not the edge case.

### Snapshot Tables Can't Reconstruct Intermediate States

When a zone boundary transitions through revisions (e.g., "Zone A" → "Zone A-1" + "Zone A-2" after a boundary split), a snapshot lookup table only shows the current boundaries. At a target_date before the split, the temporal guard may hide the new zones, but the underlying zone IDs have already changed — there's no way to recover the pre-split zone boundary. Those trips get classified by whatever the fallback produces (often the borough-level zone or "Unknown").

**Rule:** Document populations affected by this gap. If the count is small and the fallback is conservative (under-counts, never over-counts), accept it. If large, consider alternative temporal anchors.

### Explicit Code Sets vs Wildcard/Prefix Matching

Data providers often define groupings by prefix (`M*` = Manhattan zones, `B*` = Brooklyn zones). Implementation plans often use explicit sets (`{'M01', 'M02', 'M03', ...}`). Explicit sets are safer (no false positives from new codes matching a prefix), but silently misclassify future zone codes the provider adds.

**Rule:** Use explicit sets for safety, but add a monitoring check that logs a warning when an unrecognized code appears. Example:
```python
unknown = observed_codes - known_accepted - known_closed - known_in_process
if unknown:
    logger.warning(f"Unknown status codes defaulting to fallback: {unknown}")
```

### Fallback Sentinel Conflation

When a fallback path checks `df[zone_col] == 'Unknown Zone'` to catch trips whose zone code was NULL (temporally gated away), it conflates two different meanings of `'Unknown Zone'`: (1) the trip genuinely has no mapped zone, and (2) the trip has a valid zone but it was nullified by the temporal guard. Both leave `zone_col` at its default value. The fallback works correctly (mapped trips are already filtered by a `has_zone` mask), but the logic is fragile — if the default value changes or the mask has a bug, the conflation can cause misclassification.

**Rule:** Prefer checking for NULL zone explicitly (`df['zone_code'].isna()`) over checking for a sentinel default value. This makes the intent clear and doesn't depend on upstream initialization.

### Category Ordering in Ordinal Encoding

`pd.Categorical` assigns integer codes by list position. If the category list puts an outer-borough zone at a lower ordinal position than a core-Manhattan zone, the ordinal encoding may not reflect geographic or trip-duration patterns. Tree models handle this fine (they split on thresholds, not direction), but it can confuse humans reading the encoding and may subtly affect linear components if present.

**Rule:** Order categories in a domain-meaningful sequence when possible: e.g., by median trip duration or geographic proximity.

## Source Priority When Two Sources Cover the Same Event

When two data sources both record the same real-world event (e.g., a trip pickup), **coverage determines which is the source of truth** — not which source is "richer" or "more granular."

**Rule:** Put the higher-coverage source first in COALESCE. If TLC trip records cover 100% of pickups and a GPS trace dataset covers 27%, the COALESCE should be `COALESCE(tlc_zone, gps_zone)` — not the reverse.

**Complementary vs. substitutable:** Tabular records and event-level data are usually *complementary*, not interchangeable:
- Trip records capture *state* and *coverage* (zone assignment for 100% of trips)
- GPS traces capture *granularity* and *trajectory* (second-by-second coordinates for instrumented vehicles)

Replace only when sources capture truly identical information. When they capture the same event differently (zone vs. coordinates), keep both.

**Concrete guidance:** If TLC records provide the official zone assignment (state), and a GPS trace dataset provides coordinate-level pickup locations only for instrumented vehicles (27% coverage), the GPS data is not a substitute — it's a complement for the 27% it covers, and a less authoritative one at that.

## Related Skills

- **`ml-training-window-assessor`**: When the question is "can we extend the training window?" rather than "should we add feature X?" — covers per-output label validity, lookforward bridging, and companion model vs extended training architecture decisions.

## Anti-Patterns to Avoid

- **Skipping coverage gap analysis:** The most common mistake. A new source looks amazing in isolation but adds nothing over existing features.
- **Trusting field names:** "pickup_zone" might mean the TLC taxi zone, the neighborhood name, or the borough. Always validate against raw data and documentation.
- **Guessing code meanings from abbreviations:** Zone ID 161 doesn't mean "Midtown Center" — it means "Midtown Center (TLC zone 161, north of 42nd St)." Cross-reference against official TLC zone lookup tables.
- **Assuming lookup tables have history:** If the zone lookup table gets upserted, yesterday's zone boundaries are gone. Check for versioned/history variants, and validate whether they actually preserve boundary history.
- **Forgetting to monitor for new codes:** Explicit zone sets are safe until the TLC adds a new zone code that silently falls to the default. Always add runtime logging for unrecognized values.
- **Putting the lower-coverage source first in COALESCE:** Coverage determines priority. Don't default to "GPS primary, TLC fallback" when TLC records cover more trips.
- **Flagging intermediate-milestone ordering anomalies as leakage:** Check ordering against the *label*, not against other intermediate events in the pipeline.
