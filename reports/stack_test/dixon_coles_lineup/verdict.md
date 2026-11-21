# Stack test: dixon_coles_lineup, market 1x2

**Verdict: FAIL**  (all criteria must pass; pre-registered in docs/adr/0003 and 0005)

Evaluation: seasons 2020-2025, 6840 matches.

| Forecast | Log-loss |
|---|---|
| Model alone | 0.9810 |
| Market alone (closing, margin removed) | 0.9643 |
| Blend (market + beta x model) | 0.9641 |

Improvement of blend over market: +0.00018 per match (99.17% interval -0.00019 to +0.00054)
Pooled beta: -0.137 (99.17% interval -0.306 to 0.015)
Brier: market 0.5728, blend 0.5727

| Criterion | Result |
|---|---|
| improvement >= 0.005 | FAIL |
| improvement interval above 0 | FAIL |
| beta interval above 0 | FAIL |
| Brier not worse (+0.001) | PASS |
| positive in >= 4 of 6 seasons | FAIL |
| positive in every league | PASS |

| Season | beta used (fitted on earlier seasons) | Improvement |
|---|---|---|
| 2020 | 0.056 | -0.00017 |
| 2021 | 0.011 | -0.00000 |
| 2022 | 0.007 | -0.00005 |
| 2023 | -0.050 | +0.00046 |
| 2024 | -0.109 | +0.00070 |
| 2025 | -0.140 | +0.00013 |

| League | Improvement |
|---|---|
| E0 | +0.00004 |
| I1 | +0.00027 |
| SP1 | +0.00022 |
