# 0012 Search every probability window: pre-registered rules

Status: accepted, 2026-11-22. Written before the search was first run.

## Question
Harsh: not round bars. Is there ANY window of model probability (for example 63%-71%) where betting
the model makes money? Markets with prices: match winner (home, draw and away each count) and
over/under 2.5 (both sides). Model only (`dixon_coles` walk-forward, locked); bets settled at Bet365
pre-match prices, flat 100.

## The trap, and how it is handled
Searching thousands of windows will always find some that made money by luck. So:
1. **Discovery** seasons 2018/19-2022/23: every window [lo, hi) with lo from 0.20 to 0.90 and width
   0.02 to 0.30, in steps of 0.01, per market. Windows with fewer than 200 bets are ignored.
2. **Pick** the window with the highest discovery ROI, per market. Also kept: the 10 best.
3. **Held-out test** seasons 2023/24-2025/26, never seen in the search. Verdict per market:
   **makes money** only if the picked window's held-out ROI > 0 with the lower end of its 97.5%
   block-bootstrap interval (league x week; 2 markets) above 0.
4. **Luck control (placebo):** the same search, with the model's probabilities shuffled between
   matches in the same league and season (so the model knows nothing). Reported: how many windows
   look profitable in discovery, real against placebo, and what the placebo's best window did.
