# dixon_coles_dynamic against dixon_coles (ADR 0022)

Log loss per match, lower is better. Both models refitted on every weekly fold.

| | dixon_coles | dixon_coles_dynamic |
|---|---|---|
| Match result | 0.98295 | 0.98237 |
| Over/under 2.5 | 0.68298 | 0.68516 |
| Both score | 0.68920 | 0.69082 |
| Sum of the three | 2.35512 | 2.35835 |

By season (sum of the three):

        champion  challenger
season                      
2020      2.3697      2.3733
2021      2.3517      2.3593
2022      2.3717      2.3681
2023      2.3117      2.3133
2024      2.3471      2.3521
2025      2.3789      2.3840

2020/21-2025/26: difference +0.00322 per match (95% interval +0.00093 to +0.00542; 6,840 matches).
2026/27 so far: 2.32774 -> 2.32364, difference -0.00411 (95% interval -0.01681 to +0.00908; 329 matches).

**FAIL: the locked model stays.**
