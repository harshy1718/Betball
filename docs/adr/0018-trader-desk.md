# 0018 The trader desk: code decides, the LLM explains

Status: accepted, 2026-11-22.

Harsh asked for "a brain with a trader mindset" over the predictions: is a slip risky, should a
leg (corners or any other) be added, and slips built on its own, for singles and parlays, on a
private page.

1. **All maths in code** (`trader/slip.py`): joint chance, risk grade, leg costs, stake, value at
   a price. The LLM never calculates; its review gets the facts and the desk's call
   (`trader/desk.py: desk_view`) and only explains. Every number it writes is checked against the
   facts (one rewrite, then flagged); a promise word ("guaranteed", "lock") replaces the review
   with the desk's call. One LLM call, no tools, so a review takes seconds, not a minute.
2. **Same-match legs are not independent.** Goal legs use the score grid exactly. Corners and
   cards come from separate models, so their link to the goal legs and to each other is measured
   on 2020-26 matches (a lift, kept within 0.5-1.5; without the goal condition when fewer than 300
   matches qualify). Measured before building: 10+ corners came in 50.1% overall and 48.9% in
   matches with 3+ goals, so the lift is small but real for cards (4.5+ cards and 10+ corners:
   20.8% together against 21.9% if independent).
3. **Fixed risk rules**, written before use: grade from the slip's chance, one step riskier for a
   weak leg. Corners are always weak (ADR 0008), whatever the track record's skill flag says.
4. **Nox's slips follow a fixed recipe** (55-85% legs, not weak, no corners by default). He picks
   for the highest chance, not value: without prices he cannot judge value, and says so.
5. **Private by construction.** The desk routes answer only requests from this PC; the public
   site, Ask Nox and any static export never include it.
6. **It is tracked.** Saved slips settle from results; the record by risk grade (came in vs
   expected) is how we will know whether the grades mean anything. Judge at 100+ settled slips.

## Addendum, 22 Nov 2026: more bet types, the site's lines, correct score

7. **More bet types from the same score grid**: total goals 0.5-4.5, team goals 0.5-2.5,
   handicaps 1.5 and 2.5 (no push lines, so no voids in a parlay), win to nil, odd/even. Each
   has its own track record (`trader/record.py`, `track_record_goals.csv`), built from the same
   2020/21-2025/26 walk-forward expected goals as the main record; rebuilding over/under 2.5 this
   way reproduces the saved probabilities exactly. Odd/even has no skill (always weak). Draw no
   bet and Asian quarter lines are left out: their voids and half-wins make parlay maths unclear.
8. **Corners and cards lines are the site's**: cards 3.5, 4.5, 5.5; corners 8.5, 9.5, 10.5
   (Harsh, 22 Nov), over and under, in `trader.count_lines`.
9. **Correct score is a single only** (Harsh, 22 Nov): `slip.check` refuses it with any other
   leg; the builder never uses it in a parlay.

## Addendum 2, 22 Nov 2026: generated board, no manual picking

10. **No slip builder** (Harsh: "do not give me a chance to choose the bets"). The desk
    generates the board: per match, the good singles (best strong leg per bet family) and the
    best parlay from that match; on top, the safest single and the best parlay across matches.
11. **"Best parlay" is defined, not judged**: the highest-paying combination (lowest chance,
    hence highest fair odds) of 2-4 strong legs that still comes in 40%+ of the time, with every
    leg cutting the chance by 5+ points and one leg per bet family. Searched exhaustively over
    the 12 strongest legs per match, so the answer does not depend on search order.
12. **Two parlays, safe and risky** (Harsh, 22 Nov), replacing the single 40% parlay: safe =
    highest payout at 60%+; risky = highest payout at 25%+, High-confidence legs only, up to 5
    legs. Same no-passenger and one-per-family rules. Floors in `trader.parlays`.
13. **Golden calls are always listed as singles** (Harsh, 22 Nov): in their match's singles
    (★) and in a golden strip at the top, even when the leg is weak (correct score).
