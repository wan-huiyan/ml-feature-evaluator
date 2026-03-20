# Roadmap

Research-backed improvement plan for the ML Feature Evaluator skill. Based on a survey of 38 tools and 14 academic papers/industry blog posts.

## v1.1 (shipped)

Low-effort, high-impact prompt-only changes.

| Change | Source | What it fixes |
|--------|--------|---------------|
| **Q0: Data Quality Pre-Check** | Evidently AI (7.3k stars), Google TFDV (MLSys 2019) | Skill assumed clean data. Now checks NULL rate, cardinality, distribution profile, and obvious data issues before running the diagnostic. Fail fast on bad data. |
| **Q2+: Leakage plausibility ceiling** | Kaufman, Rosset & Perlich (KDD 2011) | No "too good to be true" check. Now flags >10x outcome gradient spread for leakage investigation. |
| **Q6+: Gain ratio normalization** | Quinlan (1993) | Raw entropy reduction rewarded more buckets. Now reports gain ratio alongside entropy reduction to penalize many-category expansions with little per-category value. |
| **Temporal+: Proxy & preprocessing leakage** | Kapoor & Narayanan (*Patterns* 2023), Yang et al. (ASE 2022) | Checklist missed proxy leakage (features correlated with label through causal structure) and preprocessing leakage (full-dataset statistics before train/test split). Added items 8 and 9 to the checklist. |
| **Q7: Conditional Mutual Information** | Brown et al. (JMLR 2012) — JMI framework | Feature evaluated in isolation could look great but add nothing over existing features. Now computes `I(X_new; Y | X_existing)` to detect redundancy in context. |
| **Q8: Incremental CV AUC** | mlxtend (5.1k stars), Boruta (1.6k stars) | All prior queries are proxies. Q8 is the ground truth — trains model with/without the feature, reports delta AUC with statistical significance. |
| **Monitoring spec in implementation plan** | Uber MES (2024), Google TFDV (2019) | Plan covered implementation but not post-deployment health. Now includes expected NULL rate, distribution, drift threshold, freshness, and unknown-value alerting. |

## v1.2 (planned)

Medium-effort improvements requiring more implementation work.

| Change | Source | What it would fix |
|--------|--------|-------------------|
| **SHAP S-R-I decomposition** | arXiv 2107.12436 (2021) | Q4-Q5 coverage gap is population-level overlap. S-R-I (Synergy-Redundancy-Independence) decomposition detects redundancy at the *model level* — two features can cover different populations yet carry the same signal when conditioned on other features. Requires prototype model training. |
| **Statistical significance via shadow features** | Boruta (scikit-learn-contrib), feature-engine ProbeFeatureSelection | Q6-Q7 have no confidence intervals. Shadow features (random permutations of the candidate) provide a null distribution for statistical testing. Report p-value for "is this feature better than random?" |
| **PSI (Population Stability Index) for train/serve drift** | feature-engine DropHighPSIFeatures (2.2k stars) | Current diagnostic only looks at training data. PSI compares feature distributions between training and serving environments to detect distribution shift before deployment. |

## v1.3 (future)

Interesting ideas that need more validation.

| Change | Source | What it would fix |
|--------|--------|-------------------|
| **Pipeline complexity cost score** | Sculley et al. (NeurIPS 2015) — "Hidden Technical Debt in ML Systems" | Decision framework weights signal strength but not engineering cost. A formal score (files touched, new failure modes, CACE entanglement risk) would balance signal vs. complexity. Hard to quantify programmatically. |
| **Transformation population audit** | Grafberger et al. (SIGMOD 2021) — mlinspect | SQL joins can silently drop populations (e.g., INNER JOIN dropping cookie-only events). Count rows before/after each join, flag drops >5%. Good idea, hard to generalize across data stores. |
| **Automated temporal leakage detection** | Timefence (GitHub), LeakageDetector (arXiv 2503.14723) | Current checklist is manual (9 items). Automated `feature_time < label_time` scanning would catch mechanical errors the checklist misses. Very new tools, not yet mature. |
| **Cross-method consensus** | Fidelity Selective (GitHub) | Run multiple selection methods (MI, chi-squared, tree importance, permutation) and compare whether they agree the feature is valuable. Diminishing returns for single-candidate evaluation. |
| **BigQuery-native MRMR** | mrmr library (624 stars) | Run redundancy scoring directly in BigQuery without extracting data. Useful for large-scale feature stores but overkill for single-feature evaluation. |

## Research References

### Papers
- Brown, Pocock, Zhao & Lujan. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 13, 2012.
- Sculley et al. "Hidden Technical Debt in Machine Learning Systems." NeurIPS 2015.
- Breck, Polyzotis, Roy, Whang & Zinkevich. "Data Validation for Machine Learning." MLSys 2019.
- Kaufman, Rosset & Perlich. "Leakage in Data Mining: Formulation, Detection, and Avoidance." KDD 2011 / ACM TKDD 2012.
- Kapoor & Narayanan. "Leakage and the Reproducibility Crisis in ML-based Science." Patterns (Cell Press), 2023.
- Yang et al. "Data Leakage in Notebooks: Static Detection and Better Processes." ASE 2022.
- Grafberger, Guha, Stoyanovich & Schelter. "mlinspect: A Data Distribution Debugger for ML Pipelines." SIGMOD 2021.
- Quinlan. "C4.5: Programs for Machine Learning." Morgan Kaufmann, 1993.
- "Feature Synergy, Redundancy, and Independence in Global Model Explanations using SHAP Vector Decomposition." arXiv 2107.12436, 2021.

### Tools
- [feature-engine](https://github.com/feature-engine/feature_engine) (2.2k stars) — 14 feature selection methods including ProbeFeatureSelection and PSI
- [mlxtend](https://github.com/rasbt/mlxtend) (5.1k stars) — Sequential feature selection with CV
- [boruta_py](https://github.com/scikit-learn-contrib/boruta_py) (1.6k stars) — All-relevant feature selection via shadow features
- [Boruta-Shap](https://github.com/Ekeany/Boruta-Shap) (650 stars) — Boruta + SHAP importance
- [mrmr](https://github.com/smazzanti/mrmr) (624 stars) — Minimum Redundancy Maximum Relevance with BQ support
- [featurewiz](https://github.com/AutoViML/featurewiz) (677 stars) — SULOV + Recursive XGBoost
- [Evidently AI](https://github.com/evidentlyai/evidently) (7.3k stars) — Data drift and quality monitoring
- [whylogs](https://github.com/whylabs/whylogs) (2.7k stars) — Statistical profiling and validation
- [Timefence](https://github.com/gauthierpiarrette/timefence) — Automated temporal leakage detection

### Industry
- Uber. "Model Excellence Scores: A Framework for Enhancing the Quality of Machine Learning Systems at Scale." Uber Engineering Blog, 2024.
- Nubank. "Dealing with Train-Serve Skew in Real-time ML Models: A Short Guide." Building Nubank Blog.
