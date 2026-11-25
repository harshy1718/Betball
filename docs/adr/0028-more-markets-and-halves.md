# 0028 More markets on the desk, and a half-time model: pre-registered

Status: pre-registered 2026-11-25, before any of these markets was scored. Result appended.

## Why

Harsh listed the markets his site shows. Most full-time ones are sums over the score grid we
already make; half markets need a model of when goals come. All of this is paper only (logged
and graded like every other leg); none of it is a claim of an edge.

## Half model (fixed now)

Each side's first-half goals are a binomial share of its full-time goals:
P(HT h1-a1, FT h-a) = grid(h, a) x Bin(h1; h, s_home) x Bin(a1; a, s_away), with `grid` the live
calibrated full-time grid (so every full-time market is unchanged) and s_home / s_away the share
of that league's home / away goals scored before half-time, from all seasons before the one
predicted. Second half = full time minus first half.

Check done before this ADR (all 20,311 matches): the first-half share is flat across totals
(home 0.43-0.45 for totals 1-5, away 0.44-0.45) and the spread of first-half goals given the total
matches a binomial (variance ratio 0.95-1.00 for totals 1-3, the common ones).

## Markets added (fixed now)

* Full time, win or lose: total goals 5.5; team goals 3.5; handicap 0.5 and 3.5 (same as the
  existing 1.5 / 2.5); exact total goals 0-5 and 6+; goal range 0-1, 2-3, 4-5, 6+; winning margin
  (each side by 1, by 2, by 3+, draw); each team's exact goals 0, 1, 2, 3+; clean sheet each side
  yes / no; result & both score (6); result & total 2.5 (6); total 2.5 & both score (4).
* Full time with a refund: draw no bet (each side); Asian handicap on whole lines -3 to +3; Asian
  total on whole lines 2, 3, 4. The chance shown is the chance of winning when the bet is not
  refunded; a refunded bet is not graded. Quarter lines (e.g. -1.75) are half on each neighbouring
  line and are not listed. These are singles only (no parlays).
* Halves: first-half result, first-half total 0.5 / 1.5 / 2.5, first-half both score; the same for
  the second half; half-time / full-time (9); highest-scoring half (first, second, equal); each
  team to score in both halves.

## Rules (fixed now)

1. Every new market gets a walk-forward track record exactly like the existing extra goal bets
   (2020/21-2025/26, each match predicted only from earlier matches): per-band calibration, the
   High / Medium / Low label, and the skill flag (beats always predicting the overall rate).
   For the half model the shares are walk-forward too.
2. A market without skill is shown with "no skill" and never trusted.
3. The board's own picks (good singles, safe and risky parlays) are unchanged: new markets are
   not used by it until a desk replay with them has been run and holds up (ADR 0019 style).
   They can be checked, saved and are logged and graded every day.

## Result

Run 2026-11-25 (`build-confidence`; per selection in reports/confidence/new_markets.md, labels in
reports/confidence/track_record_goals.csv). 90 track-record markets, 10,734 matches 2020/21-2025/26.

* **Skill:** 87 of 90 beat always predicting the overall rate. No skill: exactly 3 goals, and
  "more goals in the 1st half" / "more goals in the 2nd half" (the model cannot tell which half
  will be busier; same-goals-in-each-half is barely above zero). These show "no skill".
* **Size of the skill:** small for anything about total goals (exact goals, ranges, half totals,
  both score in a half: 0-1.5% better than the average), bigger where the favourite matters
  (handicaps and draw no bet 14-16%, winning by 3+ 11-12%, team goals 3.5 about 11%, half-time /
  full-time home-home and away-away 7-9%). This is against a plain average, not the market.
* **Half model:** calibrated within about 1 point on half results and totals. Two misses: comebacks
  are under-predicted (leading at half-time and losing: predicted 1.9% / 2.3%, happened 2.4% /
  2.9%), and second-half goals are slightly over-predicted (over 2.5 in the 2nd half: 21.9% vs
  20.1%). Goals are not split independently of the score: teams behind push. Noted, not fixed.
* As fixed in rule 3, none of these is used by the board's own picks. They are shown under
  "All markets" on the desk, can be checked and saved, and are logged and graded every day.
