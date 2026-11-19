# 0003 Pre-registered pass thresholds and kill rule
Status: accepted. Model must beat margin-removed closing odds by >= 0.005 nats/match log-loss with paired 95% CI excluding zero, positive stack coefficient beta (CI above zero), no material Brier/calibration loss, consistent across seasons and leagues. If it fails, stop. Thresholds are never lowered after seeing results.

## Addendum, 2026-11-20: operational definitions (written before the stack test was first run)

These make the criteria above precise. They were fixed before any stack-test result existed.

- **Forecasts compared:** market-only `p_m` (closing odds, margin removed with the proportional method) against the stacked blend `q ∝ p_m · (p_model / p_m)^β`.
- **Evaluation matches:** test seasons 2020/21-2025/26 (6,840 matches). 2018/19-2019/20 were used for model tuning and only serve to fit β for later seasons.
- **β per season:** for season `s`, β is fitted by maximum likelihood on predictions from 2018/19 up to `s-1` only.
- **Improvement Δ:** mean per-match log-loss of the market minus that of the blend. **Pass:** Δ ≥ 0.005 nats **and** the lower end of its 95% interval > 0.
- **Uncertainty:** block bootstrap, 2,000 resamples, blocks = league × weekly snapshot (matches in the same league and week are resampled together).
- **β:** pooled β fitted on 2018/19-2025/26; **pass** if the lower end of its bootstrap 95% interval > 0.
- **Brier:** blend Brier ≤ market Brier + 0.001.
- **Consistency:** Δ > 0 in at least 4 of the 6 evaluation seasons **and** in each of the 3 leagues.
- **All criteria must pass.** One model (dixon_coles) is tested; if further models are tested later, intervals are Bonferroni-adjusted for the number tested.
