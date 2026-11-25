# 0025 Other model families against Dixon-Coles: pre-registered challenge

Status: pre-registered 2026-11-25, before any of these models was fitted. Result appended.

## Why

Harsh's list: CatBoost / LightGBM, XGBoost, bivariate Poisson, a Bayesian dynamic model,
Elo / Glicko, a neural network (MLP), LSTM / Transformer. Only Elo and a dynamic model had been
tested (Elo lost; ADR 0022 dynamic ratings lost). The champion: the locked Dixon-Coles model,
5 leagues, weekly walk-forward (reports/evaluation/dixon_coles), raw (uncalibrated) chances.

## Challengers (fixed now)

1. **Bivariate Poisson**: the champion's team ratings and expected goals, but the score
   distribution is a bivariate Poisson (home = X1 + X3, away = X2 + X3, means kept) with the
   shared part X3 fitted on the training data, instead of the Dixon-Coles low-score correction.
   Weekly folds.
2. **LightGBM, XGBoost, CatBoost, each two ways**:
   * *direct*: three models predicting match result (3 classes), over 2.5 and both-score;
   * *goals*: two Poisson models predicting home and away goals, turned into a score grid
     (Dixon-Coles low-score correction with the current rho), so every market comes from one
     grid as in the live app.
   Features, all known before kickoff: the champion's walk-forward chances and expected goals,
   Elo and Glicko ratings before the match, each team's last 5 and 10 matches (goals, shots,
   shots on target, points, for and against), days since each team's last match, league,
   point in the season. Refitted before every test season on all earlier seasons from
   2018/19 (the first with out-of-sample champion predictions). Fixed settings, no tuning:
   400 trees, learning rate 0.03, depth 4 (LightGBM 15 leaves), at least 50 matches per leaf,
   80% row and column sampling.
3. **MLP** (scikit-learn, direct, same features, scaled; hidden layers 32 and 16, early
   stopping), as on the list as a later experiment; cheap enough to include.
4. **Elo and Glicko** as benchmarks: they give match result only, so they are compared on
   match-result log loss and cannot replace the champion.
5. **Bayesian dynamic**: already tested (ADR 0022, random-walk team ratings, lost). Not repeated.
6. **LSTM / Transformer**: not tested. Marked "not yet" on the list; about 20,000 matches is
   too few for them to learn more than the boosted models from the same information.

## Decision rule (fixed now)

Score per match: match result + over/under 2.5 + both-score log loss, 2020/21-2025/26 and
2026/27 so far, on the same matches. A challenger replaces the champion only if **both**:

1. on 2020/21-2025/26 the sum improves with the block-bootstrap interval (league x week)
   entirely below zero, at **99.4%** (95% Bonferroni-adjusted for 8 challengers);
2. on 2026/27 so far, the sum improves (point estimate).

If one passes, it still has to be built into the live app (features computed at prediction
time, calibration refitted) before going live; if several pass, the best on 2020-26 is taken.

## Result

Run 2026-11-25 (`challenge-all`, reports/challenger/model_families/verdict.md). Sum of the three
log losses per match, 2020/21-2025/26; champion (raw) 2.3537.

| Challenger | Sum | Difference | 99.4% interval | 2026/27 | Verdict |
|---|---|---|---|---|---|
| CatBoost (goals) | 2.3492 | -0.0045 | -0.0098 to +0.0008 | -0.0028 | FAIL |
| XGBoost (goals) | 2.3511 | -0.0026 | -0.0090 to +0.0034 | -0.0061 | FAIL |
| LightGBM (goals) | 2.3516 | -0.0021 | -0.0078 to +0.0041 | -0.0061 | FAIL |
| CatBoost (direct) | 2.3523 | -0.0014 | -0.0074 to +0.0048 | -0.0134 | FAIL |
| Bivariate Poisson | 2.3549 | +0.0012 | +0.0003 to +0.0022 | -0.0004 | FAIL |
| XGBoost (direct) | 2.3760 | +0.0223 | +0.0124 to +0.0317 | +0.0051 | FAIL |
| LightGBM (direct) | 2.3796 | +0.0259 | +0.0154 to +0.0356 | +0.0116 | FAIL |
| MLP (direct) | 2.3800 | +0.0264 | +0.0174 to +0.0358 | +0.0282 | FAIL |

Benchmarks, match result log loss 2020/21-2025/26: Elo 0.9921, Glicko 1.0058, Dixon-Coles
0.9887. Both worse.

**FAIL: Dixon-Coles stays.** Notes:

* The goals-mode boosted models were the only ones close, and none cleared the interval.
  Their best gain (-0.0045) is the size of what calibration (ADR 0020) already gives the live
  model (-0.0047: 2.3541 -> 2.3494 on almost the same matches). The boosted models, which are
  fed the champion's own chances, mostly re-learn that correction; they add nothing measurable
  on top of the calibrated model that is live.
* Direct classifiers and the MLP are clearly worse: predicting each market separately from
  ~10,000 training matches is noisier than one score grid.
* The bivariate Poisson shared part came out near zero (about 1e-5): goal counts carry no
  measurable positive covariance once team strengths are in, so it is the champion without the
  low-score correction, and slightly worse.
* LSTM / Transformer and a Bayesian dynamic model were not rerun (see above).
