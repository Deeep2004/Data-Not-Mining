Love it — v4 gave us exactly the clues we need for a fun, research-y v5. Here’s a tight, creative roadmap that stays practical and buildable on your current codebase (v3-lite + v4 tooling). I grouped ideas into 5 “bets,” each with what/why/how + a clean success check.

# V5 — Research-Grade Creativity (five bets)

## 1) Interaction-aware residuals (tiny model on top)

**What:** Keep your best CatBoost as stage-1. Train a very small stage-2 model (LightGBM or 2-layer MLP) **only** on a curated set: top features + the strongest crosses (e.g., `MSSubClass × {TotalSF, GrLivArea_to_TotalSF, Age, RemodAge, BathsPerRoom}`, plus 2–3 best ordinal qualities).
**Why:** v4’s interaction probe screams “structure by dwelling type (MSSubClass).” CatBoost catches a lot, but a tiny residual learner focused on those pairs often soaks up leftover nonlinearity.
**How (concrete):**

* Freeze v3-lite hyperparams from your best CV.
* Create explicit cross features: one-hot(MSSubClass) × each selected numeric (or bucket the numeric first by PDP thresholds).
* Train stage-1 → get out-of-fold residuals → train stage-2 on residuals (early-stop, tiny depth/hidden size).
* Final pred = stage-1 + stage-2.
  **Success:** ≥0.002 RMSLE drop vs v4 CatBoost alone in 5-fold CV (same folds and seed).

## 2) Learned blending via stacking (replace fixed alpha)

**What:** Blend 3 experts with a meta-learner rather than a fixed weight:

* Expert A: your CatBoost (v3-lite params).
* Expert B: Linear/Lasso on numeric-only (captures global trend).
* Expert C: “Leaf-index features” from CatBoost → Logistic/Linear reg (captures partition structure).
  Stage-2 meta: Ridge/ElasticNet on OOF predictions (and a few meta features: `log(TotalSF)`, `OverallQual`, `MSSubClass` buckets).
  **Why:** Fixed hybrid in v4 under-performed; a learned blender adapts weights by regime (e.g., large houses, specific MSSubClass).
  **How:** Generate OOF predictions for A/B/C on identical folds → stack them into a meta-frame → train a light linear meta-model with 5-fold OOF → test by nested CV or careful OOF assembly → refit all on full train for final.
  **Success:** Meta CV < min(CatBoost CV, Linear CV) by ≥0.0015 RMSLE.

## 3) Regime-aware multi-head CatBoost

**What:** One shared numeric preprocessing + **multi-head** scheme:

* Split training into 3–4 regimes (e.g., MSSubClass families: {20/60}, {70/75/80/85}, {90/190}, {120/150/160/180}; and/or by size threshold from PDP like `TotalSF > ~2500–3000`).
* Train a **small** CatBoost per regime + a router (simple rule: which head to use at inference).
  **Why:** Your interactions suggest the mapping changes with dwelling type/scale. A single model compromises; small heads specialize.
  **How:**
* Build deterministic router rules (no leakage): based only on raw features at inference.
* Keep per-head models tiny (depth 4–6) to avoid overfit; optionally share seed/params.
* Option B: single CatBoost with categorical “regime” feature and **monotone constraints** on `TotalSF`/`GrLivArea` (if your CatBoost version supports them) to stabilize extrapolation in luxury tail.
  **Success:** Either multi-head or single-with-regime beats baseline by ≥0.002 RMSLE and shows tighter residuals in the large-house tail.

## 4) Uncertainty + tail calibration (win leaderboard stability)

**What:** Train a **small ensemble** (e.g., 7 seeds) of the best single model; estimate mean ± std; apply **target-aware calibration** on the luxury tail.
**Why:** You saw skew/tail issues in v4; calibrated ensembles give more reliable public/private splits and improve log-error on big prices.
**How:**

* Bag seeds (different `random_state`, slight `rsm` jitter).
* For calibration: fit isotonic or a piecewise linear map on OOF `(pred → true)` specifically for upper-quantile (e.g., top 15–20% by predicted price).
* Optionally switch loss to **Huber** for head models used in the tail (robustness to outliers).
  **Success:** Lower CV RMSLE and visibly reduced under-prediction for expensive homes (QQ plot + top-quantile RMSLE).

## 5) Tabular embeddings + fusion (small MLP that complements trees)

**What:** Train a tiny MLP that learns **entity embeddings** for high-cardinality categoricals (Neighborhood, Exterior, KitchenQual, MSSubClass) and feeds numeric + embeddings → MLP. Fuse with CatBoost by stacking or residual learning.
**Why:** Trees are great; embeddings can capture “similar neighborhoods/materials” smoothly. Your v4 SHAP ranks quality/location features high — perfect for embeddings.
**How:**

* Preprocess: Ordinal encodings for qualities (as you already do), and trainable embedding layers for major categoricals; numeric standardized.
* Architecture: 2–3 dense layers (e.g., 128→64), dropout 0.1–0.2, Huber/MAE loss on log-target.
* Train with 5-fold OOF. Use either (a) residual on CatBoost, or (b) as a 4th expert in stacking.
  **Success:** Pure MLP may not beat CatBoost, but the **fused** model should give +0.001–0.003 RMSLE.



What each script does (outputs + tradeoffs are printed to console and saved)

v5_residual_interactions.py
Stage-1 CatBoost (v3-lite params) + explicit numeric×numeric crosses + a tiny residual model (LightGBM if available, else sklearn HistGBR).
Saves: outputs_v5/residual_interactions/summary.json, scoreboard.csv.
Pros: captures pairwise structure v4 hinted at; low overhead.
Cons: more features; must align folds for fairness.

v5_stack.py
Stacks three experts with a Ridge meta-learner:
A) CatBoost, B) Lasso on numeric-only (global trend), C) CatBoost leaf-indices → Ridge.
Saves: outputs_v5/stack/summary.json, scoreboard.csv.
Pros: learns when to trust each expert; robust across regimes.
Cons: OOF bookkeeping; extra training time.

v5_regime_multihead.py
Deterministic router (TotalSF bucket + MSSubClass family) → specialized CatBoost heads; also evaluates a single model with a “REGIME” categorical as a strong fallback.
Saves: outputs_v5/multihead/summary.json, scoreboard.csv.
Pros: specialists for structurally different homes; good fallback.
Cons: small buckets can overfit; a bit more orchestration.

v5_uncertainty_calib.py
7-seed CatBoost bagging (slight rsm jitter) → mean & std; isotonic calibration on the top 20% predicted prices to fix luxury-tail bias.
Saves: outputs_v5/uncertainty/summary.json, oof_mean_std.csv, scoreboard.csv.
Pros: better public/private stability; improved tail accuracy.
Cons: more compute; calibration must use OOF only (done).

v5_tabular_embed_fusion.py
Tiny MLP (sklearn MLPRegressor) on scaled numerics + one-hot of major categoricals, trained on residuals from CatBoost.
Saves: outputs_v5/emb_fusion/summary.json, scoreboard.csv.
Pros: neural residual complements tree splits with smoother category similarity.
Cons: modest gains; careful regularization needed.