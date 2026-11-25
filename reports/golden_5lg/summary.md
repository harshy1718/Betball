# The full golden-bar search, repeated on five leagues (25 Nov 2026)

Same commands and rules as the original searches (ADRs 0011-0015 and the exploratory
searches of 22 Nov), on the 5-league model's walk-forward predictions, Bet365 prices, flat 100.
Search 2018/19-2022/23, unseen 2023/24-2025/26 (2020/21-2025/26 for 0011).

| Search | Folder | Result |
|---|---|---|
| 0011 one probability bar per market | 0011_prob_bar | not profitable (match winner, over/under) |
| 0012 every window | 0012_window_search | best windows fail on the unseen seasons |
| 0013 over/under by side and league | 0013_ou_split | all 12 fail |
| 0014 window per bet type and league | 0014_bar_by_league | no golden bar in 12 searches |
| 0015 the same, from 50% up | 0015_bar_above_50 | no golden bar in 12 searches |
| exploratory: profitable in both periods | window_both, window_both_above_50 | candidates below |
| exploratory: open-ended bars (X%+) | threshold_both | Premier League winner 72%+ only (2 / 0) |

Pre-registered: no golden bar is proven in any league, as before.

Exploratory candidates (profitable in both periods, real windows clearly above the luck count):

| Bet | League | Window | Return, 8 seasons | Real / luck | Status |
|---|---|---|---|---|---|
| Match winner | Premier League | 60-64% | +6.0% | 50 / 9 | live bar, kept |
| Match winner | Serie A | 63-68% | +7.0% | 36 / 4 | live bar, kept |
| Match winner | La Liga | 60-64% | +8.7% (search +10.8%, unseen +6.2%) | 97 / 7 | live bar, kept |
| Over/under | Premier League | 36-38% (backing the side the model calls less likely) | +14.3% | 25 / 8 | below 50%: outside the 50%+ rule (ADR 0015) |
| Over/under | Ligue 1 | 28-35% (same) | +5.2% | 28 / 5 | below 50%: outside the rule |
| Match winner | Premier League | 72% or more | +3.2% | 2 / 0 | too few windows to tell from luck |

Bundesliga (winner 1 / 31 from 50% up, 15 / 40 from 20%; over/under 7 / 24) and Ligue 1 (winner
1 / 70 and 1 / 97): luck finds as many windows as the model. No bars there.
