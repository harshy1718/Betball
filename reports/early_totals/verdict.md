# Early over/under 2.5 prices vs the model

Rules pre-registered in docs/adr/0009-early-totals-test.md. Seasons 2020/21-2025/26, 6,840 matches with model predictions and early prices. Intervals 98.75%.

## Test A: does the model add to the early Pinnacle price?

| Forecast | Log-loss |
|---|---|
| Early market alone (Pinnacle, margin removed) | 0.6740 |
| Model alone | 0.6809 |
| Blend (beta from earlier seasons) | 0.6742 |

Blend minus market: -0.00016 per match (interval -0.00050 to +0.00018), 5,140 matches.

## Test B: betting the model at early prices (flat 100)

| Variant | Bets | Staked | Profit | ROI | ROI interval | Win rate | Avg odds | CLV | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| b365_3 | 3,145 | 314,500 | -14,479 | -4.60% | -9.40% to -0.04% | 47.2% | 2.09 | -4.34% | FAIL |
| b365_6 | 2,132 | 213,200 | -12,916 | -6.06% | -11.45% to -0.73% | 45.2% | 2.15 | -4.32% | FAIL |
| max_3 | 4,640 | 464,000 | -4,633 | -1.00% | -4.81% to +2.72% | 48.1% | 2.13 | -0.90% | FAIL |
| max_6 | 3,350 | 335,000 | -4,820 | -1.44% | -6.04% to +3.20% | 47.1% | 2.17 | -0.75% | FAIL |

### b365_3

| Criterion | Result |
|---|---|
| ROI interval above 0 | FAIL |
| ROI > 0 in >= 4 of 6 seasons | FAIL |
| at least 200 bets | PASS |

| Season | ROI |
|---|---|
| 2020 | -2.51% |
| 2021 | -1.48% |
| 2022 | -10.45% |
| 2023 | -8.01% |
| 2024 | -5.15% |
| 2025 | +0.35% |

### b365_6

| Criterion | Result |
|---|---|
| ROI interval above 0 | FAIL |
| ROI > 0 in >= 4 of 6 seasons | FAIL |
| at least 200 bets | PASS |

| Season | ROI |
|---|---|
| 2020 | -2.23% |
| 2021 | -6.28% |
| 2022 | -10.51% |
| 2023 | -11.39% |
| 2024 | -6.72% |
| 2025 | +1.30% |

### max_3

| Criterion | Result |
|---|---|
| ROI interval above 0 | FAIL |
| ROI > 0 in >= 4 of 6 seasons | FAIL |
| at least 200 bets | PASS |

| Season | ROI |
|---|---|
| 2020 | +3.40% |
| 2021 | +1.55% |
| 2022 | -5.30% |
| 2023 | -3.71% |
| 2024 | -0.94% |
| 2025 | -1.11% |

### max_6

| Criterion | Result |
|---|---|
| ROI interval above 0 | FAIL |
| ROI > 0 in >= 4 of 6 seasons | FAIL |
| at least 200 bets | PASS |

| Season | ROI |
|---|---|
| 2020 | -0.02% |
| 2021 | +0.85% |
| 2022 | -7.27% |
| 2023 | -2.35% |
| 2024 | -0.10% |
| 2025 | +0.89% |
