# The trader desk (private)

A page on the local Nox site, `/desk`, for Harsh only. Its API answers requests from this PC and
nowhere else, and it is never part of the public site. Settings: `trader:` in
`configs/default.yaml`. Decisions: ADR 0018.

## What it does

Nothing is hand-picked: Nox generates the whole board from the model.

**Handicaps on a three-way table.** Sites often show handicaps as "2:0 / Getafe" (Getafe starts
two goals up). Every handicap leg shows its table entry: Getafe +1.5 = 2:0 · Getafe, Getafe -1.5 =
0:1 · Getafe, Atletico +1.5 = 0:2 · Atletico, Atletico -1.5 = 1:0 · Atletico.

**Scorecard and posts.** The desk's slips are logged and graded automatically (docs/tracking.md);
the Scorecard tab shows how each slip type did against its promise.

**Pick a match.** The "Show" box under the header switches between all matches (the full board
below) and one match: then only that match's singles and two parlays are shown. The choice is
kept in the address (`/desk?match=<id>`), so a refresh or a bookmark keeps it. Picking a match
only filters the view; the slips are the same ones the full board shows.

1. **Top of the board.** The safest single on the board, and a safe and a risky parlay across
   matches (one strong leg per match).
2. **Golden singles.** Every golden call in the next 3 days, as singles, whatever the leg's
   confidence (they are unproven and tracked, and the desk still grades their risk).
3. **Match by match**, for every match in the next 3 days:
   - **Good singles**: the best leg of each bet family (result, total goals, team goals, both
     teams score, handicap, cards...) at 55-85%, strong legs only, High confidence first.
   - **Golden bets** (★): that match's golden calls, in their own box above the good singles,
     with the rule that made each one golden. When you pick one match, the box is always shown,
     and says so when the match has no golden call.
   - **Safe parlay**: the highest-paying combination of 2 to 4 strong legs that still comes in
     **60%+** of the time.
   - **Risky parlay**: a bigger payout that comes in **25%+** of the time, 2 to 5 legs, built
     only from **High-confidence** legs, so the risk is in the length, not in shaky legs.
   - In both, every leg cuts the chance by 5+ points (no passengers) and no two legs answer the
     same question. "Best" cannot mean "most likely" (each extra leg lowers the chance) and,
     without the site's prices, it cannot mean best value. The floors are in `trader.parlays`.
4. **Bet types**, all from our model: result, double chance, total goals (0.5 to 4.5), team
   goals (0.5, 1.5, 2.5), both teams score, handicap (1.5 and 2.5), win to nil, cards and corners
   (the site's lines: cards 3.5, 4.5, 5.5; corners 8.5, 9.5, 10.5; `trader.count_lines`), odd/even
   and correct score. Corners and odd/even have no model skill, so the board leaves them out.
   **A correct score is a single bet only**, never in a parlay.
5. **Every slip shows** its chance, our fair odds, a risk grade, the desk's call, what each leg
   costs ("without X the slip comes in 52%"), and a stake.
6. **Your site's odds.** Type the odds you are offered: the desk says whether they are above or
   below our fair price and what comes back per 100 staked on average.
7. **Ask the trader.** The LLM explains the slip like a risk manager. Code has already done every
   number and the call; the LLM only explains. Every number in its answer is checked.
8. **Save slip.** Saved to `bets/slips.csv` (never in git); saved slips settle themselves once
   the results are in, and the desk keeps a record by risk grade.

## How the numbers are made

| Number | How |
|---|---|
| Leg chance | our model only (`staking.confidence: model`) |
| Confidence | the leg's own track record at that chance (`reports/confidence/track_record.csv`; the extra goal bets in `track_record_goals.csv`, rebuilt by `football-edge build-confidence`) |
| Weak leg | no model skill on that kind of bet (corners and odd/even always), or Low confidence |
| Same-match goal legs | read exactly from the model's score grid (result, goals, team goals, handicap, win to nil, odd/even and correct score are all about the score) |
| Corners / cards with other legs | each stat becomes a range; how it moves with the goal legs is measured on real matches since 2020 (capped between 0.5x and 1.5x) |
| Different matches | chances multiply |
| Risk grade | slip chance 60%+ Low, 40-60% Medium, 20-40% High, under 20% Very high; one step up if any leg is weak |
| Stake | `trader.stake_by_risk` share of `staking.monthly_budget`, rounded; 0 once the month's stop-loss is hit |

## Honest limits

- Apart from the golden bars, which are still being tracked, no edge over the market is proven.
  A parlay pays the site's margin on every leg, so it needs a bigger edge than a single.
- Corners: the model is no better than the league average. Cards: some skill, but sites count
  cards differently (booking points), so check the rule before using a cards leg.
- The record by risk grade is the real test of the desk: judge it after 100+ settled slips.
