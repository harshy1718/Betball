# Stack test: dixon_coles_trust, market 1x2

**Verdict: FAIL**  (all criteria must pass; pre-registered in docs/adr/0003 and 0005)

Evaluation: seasons 2020-2025, 6840 matches.

| Forecast | Log-loss |
|---|---|
| Model alone | 0.9811 |
| Market alone (closing, margin removed) | 0.9643 |
| Blend (market + beta x model) | 0.9640 |

Improvement of blend over market: +0.00023 per match (99.17% interval -0.00018 to +0.00066)
Pooled beta: -0.156 (99.17% interval -0.333 to 0.011)
Brier: market 0.5728, blend 0.5727

| Criterion | Result |
|---|---|
| improvement >= 0.005 | FAIL |
| improvement interval above 0 | FAIL |
| beta interval above 0 | FAIL |
| Brier not worse (+0.001) | PASS |
| positive in >= 4 of 6 seasons | PASS |
| positive in every league | PASS |

| Season | beta used (fitted on earlier seasons) | Improvement |
|---|---|---|
| 2020 | 0.043 | -0.00020 |
| 2021 | -0.025 | +0.00000 |
| 2022 | -0.023 | +0.00011 |
| 2023 | -0.059 | +0.00073 |
| 2024 | -0.142 | +0.00073 |
| 2025 | -0.164 | +0.00002 |

| League | Improvement |
|---|---|
| E0 | +0.00002 |
| I1 | +0.00031 |
| SP1 | +0.00037 |
