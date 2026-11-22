# 0013 Over/under windows by side and league: pre-registered rules (final search)

Status: accepted, 2026-11-22. Written before the search was run.

## Question
ADR 0012 found no over/under 2.5 window that held up on unseen seasons (over and under pooled,
all leagues). Harsh asks to try again. This is the last over/under search on this data: if it
fails, over/under probability bars are closed unless new data or a new model arrives.

## What is new
The same search as ADR 0012 (model only, Bet365 pre-match prices, flat 100, windows lo 0.20-0.90 x
width 0.02-0.30 in 0.01 steps, at least 200 bets in discovery... lowered to **100** for single-league
searches, which have a third of the matches) run separately for:
- side: **over only**, **under only**;
- league: **all three**, **Premier League**, **Serie A**, **La Liga**.
8 searches in total.

## Verdict
Discovery 2018/19-2022/23, held-out 2023/24-2025/26 (never seen by the search). For each of the 8,
the best discovery window is tested on the held-out seasons. A window **makes money** only if its
held-out ROI > 0 with the lower end of its block-bootstrap interval above 0 at **99.375%**
(Bonferroni for 8 searches). A placebo (model probabilities shuffled within league-season) is run
for each search and the number of "profitable" discovery windows is reported next to the real one.
