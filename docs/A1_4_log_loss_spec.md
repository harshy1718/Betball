# A1.4 Log-loss: how we score a forecast (Harsh writes this)

**You write:** `log_loss` in `src/football_edge/evaluation/metrics.py` (the signature is already there; `brier_score` underneath is a finished example to copy the style from).
**Done when:** `pytest tests/unit/test_metrics.py -v` shows 12 passed, 0 skipped.
**Expected size:** about 10-15 lines.

---

## 1. Why we need a score at all

Every model in this project gives three probabilities per match: home win, draw, away win. To compare models (team-blind vs Elo vs Dixon-Coles vs the bookmaker) we need **one number** that says how good those probabilities were, once we know what happened.

"Did the favourite win?" (accuracy) is not good enough. A model that says 51% and one that says 95% both "picked" the home side, but they are making very different claims. If the home side loses, the 95% model was badly wrong and the 51% model was barely wrong. We need a score that sees that difference.

## 2. The idea in one sentence

**Log-loss = how surprised you were by what actually happened, averaged over all matches.**

For one match: look at the probability you gave to the result that actually happened, and take `-ln` of it.

| You gave the actual result | -ln(p) | Meaning |
|---|---|---|
| 0.90 | 0.105 | barely surprised: good |
| 0.50 | 0.693 | |
| 0.33 | 1.099 | the "no idea" score |
| 0.10 | 2.303 | very surprised: bad |
| 0.01 | 4.605 | you were confidently wrong |
| 0.00 | infinity | you said it was impossible and it happened |

Lower is better. The penalty explodes as you approach 0, which is exactly what we want: **confident wrong answers are punished hard**. This is also why it matters for betting: a model that is overconfident will lose money even if it "picks" well.

(`ln` is the natural log, `np.log` in NumPy.)

## 3. A worked example

Two matches. The model gave both the same forecast: home 0.5, draw 0.3, away 0.2.

| Match | Forecast (H, D, A) | Result | Outcome index | Prob given to result | -ln |
|---|---|---|---|---|---|
| 1 | 0.5, 0.3, 0.2 | Home win | 0 | 0.5 | 0.693 |
| 2 | 0.5, 0.3, 0.2 | Draw | 1 | 0.3 | 1.204 |

Log-loss = average = (0.693 + 1.204) / 2 = **0.949**.

The outcome index is the column number: 0 = home, 1 = draw, 2 = away. That's the same order everywhere in the project (`OUTCOMES` in `models/base.py`).

## 4. Numbers to calibrate your sense of "good"

These are the real results on our 9,120 test matches (2018/19-2025/26), which you will reproduce once your function works:

| Forecast | Log-loss |
|---|---|
| Always (1/3, 1/3, 1/3) | 1.0986 (= ln 3) |
| Team-blind baseline (league H/D/A rates) | 1.0741 |
| Elo baseline | 0.9865 |
| Bookmaker closing odds (margin removed) | 0.9625 |

Two things to notice:
- The whole range from "know nothing" to "the market" is only about 0.14. Differences of **0.005** matter here. That is why the pass threshold in ADR 0003 is 0.005, not 0.05.
- Elo, a few lines of maths, gets about 80% of the way from team-blind to the market. The last 0.024 is where all the difficulty lives. Dixon-Coles has to close that gap to have any chance of beating the market.

## 5. What the function must do

```
log_loss(probs, outcomes, eps=1e-15) -> float
```

- `probs`: NumPy array, shape `(n, 3)`, one row per match, columns H/D/A.
- `outcomes`: NumPy integer array, shape `(n,)`, each 0, 1 or 2.
- Returns: one float, the mean of `-ln(probability given to the actual outcome)`.

Rules:

| # | Rule | Why |
|---|---|---|
| 1 | Raise `ValueError` if `probs` isn't 2-D with exactly 3 columns, or if `len(outcomes)` doesn't match the number of rows. | A silent shape mix-up would score the wrong matches. |
| 2 | Raise `ValueError` if any probability is below 0 or above 1, or a row doesn't sum to 1 (allow a tolerance of 1e-6). | These aren't probabilities; a bug upstream must fail loudly. |
| 3 | Raise `ValueError` if any outcome isn't 0, 1 or 2. | Same reason. |
| 4 | Clip the chosen probabilities to at least `eps` before taking the log. | `ln(0)` is minus infinity, and one such match would make the whole average infinite. With clipping it's a huge but finite penalty (about 34.5). |
| 5 | Don't modify the arrays you're given. | Same rule as `make_folds`: the caller still needs them. |
| 6 | Return a plain Python `float` (wrap the result in `float(...)`). | mypy expects a float, and NumPy scalars cause odd behaviour in reports. |

## 6. Suggested approach (in words)

1. Check the shapes (rule 1). `probs.ndim` gives the number of dimensions, `probs.shape` gives `(rows, columns)`.
2. Check the values (rules 2 and 3). Useful tools: `(probs < 0).any()`, `probs.sum(axis=1)` (one sum per row), `np.allclose(a, b, atol=1e-6)`, `np.isin(outcomes, [0, 1, 2]).all()`.
3. **Pick out the probability of the actual result for every match in one step.** This is the only new trick:
   `probs[np.arange(n), outcomes]`
   `np.arange(n)` is `[0, 1, ..., n-1]` (the row numbers). Pairing it with `outcomes` picks row 0's column `outcomes[0]`, row 1's column `outcomes[1]`, and so on. For the worked example: rows `[0, 1]`, columns `[0, 1]` → `[0.5, 0.3]`. This is called **fancy indexing**. Look at how `brier_score` uses the same trick.
4. Clip with `np.clip(values, eps, 1.0)` (rule 4). `np.clip` returns a new array, so rule 5 holds.
5. `-np.mean(np.log(...))`, wrapped in `float(...)`.

## 7. What each test checks

| Test | It fails if... |
|---|---|
| `test_log_loss_uniform_is_ln3` | (1/3, 1/3, 1/3) doesn't score ln 3 |
| `test_log_loss_worked_example` | section 3's example doesn't give 0.949 |
| `test_log_loss_perfect_is_near_zero` | giving the true result probability 1 doesn't score ~0 |
| `test_log_loss_zero_probability_is_finite` | a probability of 0 for the actual result gives infinity (rule 4 missing) |
| `test_log_loss_rejects_bad_input` (5 cases) | any of the bad inputs in rules 1-3 is accepted |
| `test_log_loss_does_not_modify_inputs` | the input array changed |

Until you write the function these tests show as **skipped**, not failed, so CI stays green. Run them with:
```
pytest tests/unit/test_metrics.py -v
```

## 8. Then see the baselines

Once the 12 tests pass:
```
football-edge evaluate --model team_blind
football-edge evaluate --model elo
```
Each prints a table of log-loss and Brier score overall, by league and by season, and saves it under `reports/evaluation/<model>/`. Your numbers should match section 4. Team-blind takes a few seconds, Elo about 10-20 seconds.

## 9. Question for your docstring

Answer in a sentence or two: **why is log-loss better than accuracy for a betting model?** (Hint: section 1, and think about what happens to your money when you're confidently wrong.)
