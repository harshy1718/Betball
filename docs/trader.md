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

**Build your own slip: good bets only (ADR 0029).** Under every match, "Build your own slip"
offers only the bets the model predicts well: on 2020-26 walk-forward matches each beat a plain
average by 5%+ in log loss and its chances held up (within 1.5 points). That is mostly bets on who
is stronger (result without the draw, handicaps, team goals); over/under, both score, the draw,
exact goals, cards and correct scores are not offered there. The list is rebuilt by
`build-confidence` (reports/confidence/best_markets.md). Nox's board uses its old markets plus
every best market, new ones included (`trader.add_best: true`; replay passed, ADR 0029).

**All markets (ADR 0028).** The desk can price every bet below,
grouped (result, handicaps, goals, team goals, both score & combos, correct score, 1st half,
2nd half, half-time / full-time, cards, corners), with our chance, fair odds (1 / chance) and
confidence. Tick legs and press Check to price your own slip. Markets marked **new** (exact goals,
goal ranges, winning margin, team exact goals, clean sheets, result & both score, result & total,
total & both score, draw no bet, Asian whole lines, all half markets) are logged and graded every
day but are not used by Nox's own slips until a replay with them holds up. **Refund** bets (draw no
bet, Asian whole lines) are singles only; their chance is for when the bet is not refunded, and
a refunded saved slip settles as "void" with no profit or loss. Half markets come from the half
model: each side's first-half goals are a share of its full-time goals (the league's own share,
about 44%), so full-time chances do not change. It under-rates comebacks a little. Quarter
Asian lines (e.g. -1.75) are half on each neighbouring line and are not listed.

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
   goals (0.5, 1.5, 2.5), both teams score, handicap (1.5 and 2.5), win to nil, cards (the site's
   lines 3.5, 4.5, 5.5; `trader.count_lines`), odd/even and correct score. Odd/even has no model
   skill, so the board leaves it out. Corners are no longer forecast at all (ADR 0020).
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
| Leg chance | our model only (`staking.confidence: model`), calibrated (ADR 0020): goals through one calibrated score grid, so singles and parlays agree; cards through a calibrated mean and spread |
| Confidence | the leg's own track record at that chance (`reports/confidence/track_record.csv`; the extra goal bets in `track_record_goals.csv`, rebuilt by `football-edge build-confidence`) |
| Weak leg | no model skill on that kind of bet (corners and odd/even always), or Low confidence |
| Same-match goal legs | read exactly from the model's score grid (result, goals, team goals, handicap, win to nil, odd/even and correct score are all about the score) |
| Corners / cards with other legs | each stat becomes a range; how it moves with the goal legs is measured on real matches since 2020 (capped between 0.5x and 1.5x) |
| Different matches | chances multiply |
| Risk grade | slip chance 60%+ Low, 40-60% Medium, 20-40% High, under 20% Very high; one step up if any leg is weak |
| Stake | `trader.stake_by_risk` share of `staking.monthly_budget`, rounded; 0 once the month's stop-loss is hit; cut so that open saved slips on any one match never exceed `staking.max_match_fraction` (4%) of the budget (a parlay's whole stake counts against every match in it) |

## Does it work? The replay

`football-edge replay-desk` replays the desk's own code on every match of 2020/21-2025/26 with
nothing from the future (calibrated chances, labels from earlier seasons only, lifts from
earlier matches only) and grades every slip. Results in `reports/replay/` and ADR 0020: singles
75.4% predicted, 75.8% landed; safe parlays 60.8% and 62.3%; risky parlays 33.2% and 33.4%.

## Honest limits

- Apart from the golden bars, which are still being tracked, no edge over the market is proven.
  A parlay pays the site's margin on every leg, so it needs a bigger edge than a single.
- Corners: the model is no better than the league average. Cards: some skill, but sites count
  cards differently (booking points), so check the rule before using a cards leg.
- The record by risk grade is the real test of the desk: judge it after 100+ settled slips.
