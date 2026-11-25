# Promoted teams (ADR 0026) and goal spread (ADR 0027)

Per match: match result + over/under 2.5 + both-score log loss, lower is better. Champion = the live model (Dixon-Coles + five-number calibration). Interval: 98.3% (Bonferroni for 3), blocks = league x week.

| Challenger | Matches | Live model | Challenger | Difference | 98.3% interval | 2026/27 difference | Verdict |
|---|---|---|---|---|---|---|---|
| A: promoted stretch in the calibration (ADR 0026) | 10,734 | 2.34941 | 2.34927 | -0.00013 | -0.00086 to +0.00057 | -0.00100 (548) | FAIL |
| B: weaker pull to the promoted prior (ADR 0026) | 10,734 | 2.34941 | 2.35021 | +0.00080 | +0.00043 to +0.00119 | +0.00244 (548) | FAIL |
| Less spread, Conway-Maxwell-Poisson (ADR 0027) | 10,734 | 2.34941 | 2.34933 | -0.00007 | -0.00020 to +0.00004 | -0.00003 (548) | FAIL |

Secondary (2020/21 onwards, not part of the rule):

| Model | Promoted sides, matches 1-19: goal-difference residual | Spread home | Spread away | 0-0 predicted | 0-0 observed |
|---|---|---|---|---|---|
| champion | +0.103 | 0.942 | 0.948 | 0.0686 | 0.0640 |
| A | +0.016 | 0.943 | 0.948 | 0.0687 | 0.0640 |
| B | +0.092 | 0.943 | 0.948 | 0.0684 | 0.0640 |
| spread | +0.092 | 0.977 | 0.978 | 0.0661 | 0.0640 |

Fitted a (others) and a_p (promoted-early) per season: [{'season': 2019, 'a': 1.071, 'a_p': 0.998}, {'season': 2020, 'a': 1.102, 'a_p': 0.94}, {'season': 2021, 'a': 1.119, 'a_p': 0.889}, {'season': 2022, 'a': 1.104, 'a_p': 0.961}, {'season': 2023, 'a': 1.077, 'a_p': 0.964}, {'season': 2024, 'a': 1.1, 'a_p': 0.983}, {'season': 2025, 'a': 1.104, 'a_p': 0.984}, {'season': 2026, 'a': 1.111, 'a_p': 0.975}]
Fitted nu per season (1 = Poisson, above 1 = less spread): [{'season': 2019, 'nu': 1.041}, {'season': 2020, 'nu': 0.989}, {'season': 2021, 'nu': 1.035}, {'season': 2022, 'nu': 1.055}, {'season': 2023, 'nu': 1.064}, {'season': 2024, 'nu': 1.065}, {'season': 2025, 'nu': 1.074}, {'season': 2026, 'nu': 1.07}]

**FAIL: the live model stays.**
