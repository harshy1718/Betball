# dixon_coles_per_league against dixon_coles (ADR 0022)

Log loss per match, lower is better. Both models refitted on every weekly fold.

| | dixon_coles | dixon_coles_per_league |
|---|---|---|
| Match result | 0.98295 | 0.98296 |
| Over/under 2.5 | 0.68298 | 0.68278 |
| Both score | 0.68920 | 0.68975 |
| Sum of the three | 2.35512 | 2.35550 |

By season (sum of the three):

        champion  challenger
season                      
2020      2.3697      2.3679
2021      2.3517      2.3504
2022      2.3717      2.3712
2023      2.3117      2.3124
2024      2.3471      2.3527
2025      2.3789      2.3784

2020/21-2025/26: difference +0.00037 per match (95% interval -0.00134 to +0.00202; 6,840 matches).
2026/27 so far: 2.32774 -> 2.33007, difference +0.00232 (95% interval -0.00523 to +0.00967; 329 matches).

**FAIL: the locked model stays.**
