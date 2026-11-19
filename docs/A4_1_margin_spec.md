# A4.1 Margin removal: turning odds into fair probabilities (Harsh writes this)

**You write:** `overround` and `remove_margin` in `src/football_edge/markets/margin.py`.
**Done when:** `pytest tests/unit/test_margin.py -v` shows 9 passed, 0 skipped.
**Expected size:** about 10 lines in total.

## 1. Why

To compare our model with the bookmaker, we need the bookmaker's probabilities. Odds are not probabilities yet: they include the bookmaker's cut (the **margin**, or **overround**). Remove it and you get the bookmaker's honest opinion, which is what our model must compete with.

## 2. The idea, with real odds

Spain v England, from the betting site on 15 Nov 2026: home 1.96, draw 3.80, away 3.35.

**Step 1: odds to raw probabilities.** Raw probability = 1 ÷ odds.

| | Odds | 1 ÷ odds |
|---|---|---|
| Spain | 1.96 | 0.5102 |
| Draw | 3.80 | 0.2632 |
| England | 3.35 | 0.2985 |
| **Sum** | | **1.0719** |

**Step 2: spot the margin.** Real probabilities add up to 1. These add up to 1.0719. The extra **0.0719 (7.19%)** is the bookmaker's cut. That is `overround`.

**Step 3: remove it.** Divide each raw probability by the sum, so they add up to 1:

| | Fair probability |
|---|---|
| Spain | 0.5102 ÷ 1.0719 = **0.4760** |
| Draw | 0.2632 ÷ 1.0719 = **0.2455** |
| England | 0.2985 ÷ 1.0719 = **0.2785** |

That is `remove_margin`. This "divide by the sum" method is called **proportional** (or basic normalisation). It is the method agreed in the plan.

## 3. The two functions

```
overround(odds) -> array of n numbers      # sum of 1/odds per row, minus 1
remove_margin(odds) -> array of shape (n, 3)   # 1/odds divided by its row sum
```
`odds` has shape `(n, 3)`: one row per match, columns home/draw/away.

## 4. Rules

| # | Rule | Why |
|---|---|---|
| 1 | Raise `ValueError` if `odds` is not 2-D with exactly 3 columns. | Same reason as in `log_loss`: a shape mix-up must fail loudly. |
| 2 | Raise `ValueError` if any odds are missing (NaN) or not above 1.0. | Decimal odds of 1.0 or less are impossible (they pay nothing back). |
| 3 | Both functions apply the same checks. | Write the checks once in a small helper and call it from both. |
| 4 | Don't modify the input array. | As before. |

## 5. Hints (NumPy tools)

- `odds.ndim` and `odds.shape` for rule 1.
- `np.isnan(odds).any()` and `(odds <= 1).any()` for rule 2.
- `1 / odds` works on the whole array at once and returns a new array.
- `.sum(axis=1)` gives one sum per row (you used this in `log_loss`).
- To divide each row by its own sum, keep the sum as a column: `.sum(axis=1, keepdims=True)`. Without `keepdims=True`, NumPy can't line the sums up with the rows.
- A helper is just another function, e.g. `def _check(odds: Odds) -> None:` above the other two.

## 6. Tests

| Test | Fails if... |
|---|---|
| `test_overround_worked_example` | Spain v England's margin isn't 0.0719 |
| `test_remove_margin_worked_example` | its fair probabilities aren't 0.4760 / 0.2455 / 0.2785 |
| `test_rows_sum_to_one_and_keep_order` | rows don't add to 1, or the favourite stops being the likeliest |
| `test_fair_odds_have_no_margin` | odds with no cut get a margin |
| `test_bad_odds_rejected` (4 cases) | wrong shape, odds of 1.0, a missing price, or a 1-D array is accepted |
| `test_inputs_not_modified` | the input array changed |

They show as **skipped** until both functions exist, so CI stays green.
