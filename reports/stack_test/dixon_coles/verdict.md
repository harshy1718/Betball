# Stack test: dixon_coles

**Verdict: FAIL**  (all criteria must pass; pre-registered in docs/adr/0003-kill-rule.md)

Evaluation: seasons 2020-2025, 6840 matches.

| Forecast | Log-loss |
|---|---|
| Model alone | 0.9829 |
| Market alone (closing, margin removed) | 0.9643 |
| Blend (market + beta x model) | 0.9640 |

Improvement of blend over market: +0.00029 per match (95% interval -0.00008 to +0.00066)
Pooled beta: -0.160 (95% interval -0.282 to -0.035)
Brier: market 0.5728, blend 0.5726

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
| 2020 | 0.049 | -0.00031 |
| 2021 | -0.037 | +0.00004 |
| 2022 | -0.042 | +0.00033 |
| 2023 | -0.098 | +0.00091 |
| 2024 | -0.151 | +0.00077 |
| 2025 | -0.169 | +0.00001 |

| League | Improvement |
|---|---|
| E0 | +0.00017 |
| I1 | +0.00032 |
| SP1 | +0.00039 |
