# Other model families against Dixon-Coles (ADR 0025)

Log loss per match, 2020/21-2025/26 (lower is better). Interval: 99.4% (Bonferroni for 8), blocks = league x week.

| Challenger | Result | Over/under | Both score | Sum | Dixon-Coles sum | Difference | 99.4% interval | 2026/27 difference | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| catboost (goals) | 0.9866 | 0.6774 | 0.6851 | 2.3492 | 2.3537 | -0.0045 | -0.0098 to +0.0008 | -0.0028 | FAIL |
| xgboost (goals) | 0.9868 | 0.6783 | 0.6860 | 2.3511 | 2.3537 | -0.0026 | -0.0090 to +0.0034 | -0.0061 | FAIL |
| lightgbm (goals) | 0.9873 | 0.6784 | 0.6858 | 2.3516 | 2.3537 | -0.0021 | -0.0078 to +0.0041 | -0.0061 | FAIL |
| catboost (direct) | 0.9864 | 0.6787 | 0.6872 | 2.3523 | 2.3537 | -0.0014 | -0.0074 to +0.0048 | -0.0134 | FAIL |
| bivariate Poisson | 0.9894 | 0.6795 | 0.6859 | 2.3549 | 2.3537 | +0.0012 | +0.0003 to +0.0022 | -0.0004 | FAIL |
| xgboost (direct) | 0.9959 | 0.6860 | 0.6941 | 2.3760 | 2.3537 | +0.0223 | +0.0124 to +0.0317 | +0.0051 | FAIL |
| lightgbm (direct) | 0.9997 | 0.6857 | 0.6941 | 2.3796 | 2.3537 | +0.0259 | +0.0154 to +0.0356 | +0.0116 | FAIL |
| mlp (direct) | 0.9960 | 0.6898 | 0.6942 | 2.3800 | 2.3537 | +0.0264 | +0.0174 to +0.0358 | +0.0282 | FAIL |

Dixon-Coles alone: result 0.9887, over/under 0.6795, both score 0.6855.

Benchmarks (match result only, weekly refits):

| Model | Result log loss | Dixon-Coles |
|---|---|---|
| elo | 0.99209 | 0.98873 |
| glicko | 1.00579 | 0.98873 |

**FAIL: Dixon-Coles stays.**
