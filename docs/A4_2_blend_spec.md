# A4.2 The blend: market plus a dose of our model (Harsh writes this)

**You write:** `blend` in `src/football_edge/markets/stack.py`.
**Done when:** `pytest tests/unit/test_stack.py -v` shows 7 passed, 0 skipped.
**Expected size:** about 5 lines.

## 1. Why

The stack test asks whether our model knows anything the bookmaker doesn't. It does that by mixing the two forecasts with a dial `beta` and finding the best setting. `blend` is the mixing step; everything around it (finding the best beta, confidence intervals, the verdict) is already written.

## 2. The formula

For each match and each outcome (home, draw, away):

```
raw  = p_market × (p_model ÷ p_market) ^ beta
blend = raw ÷ (sum of raw across the three outcomes)
```

- `beta = 0`: anything to the power 0 is 1, so `raw = p_market`. The market alone.
- `beta = 1`: `raw = p_market × p_model ÷ p_market = p_model`. Our model alone.
- In between: part of the way from the market towards our model.
- The second line rescales so the three probabilities add up to 1 again (the same trick as `remove_margin`).

## 3. Worked example: Spain v England, beta = 0.5

| | Market | Model | Model ÷ market | ^0.5 | raw = market × that |
|---|---|---|---|---|---|
| Spain | 0.4760 | 0.55 | 1.1555 | 1.0749 | 0.5117 |
| Draw | 0.2455 | 0.22 | 0.8961 | 0.9466 | 0.2324 |
| England | 0.2785 | 0.23 | 0.8259 | 0.9088 | 0.2531 |
| **Sum** | | | | | **0.9972** |

Divide each raw value by 0.9972: Spain **0.5131**, draw **0.2331**, England **0.2538**. Spain moved from the market's 47.6% about halfway towards our 55%.

## 4. Rules

| # | Rule |
|---|---|
| 1 | Raise `ValueError` if the two arrays have different shapes, or aren't shape `(n, 3)`. |
| 2 | Return an array of shape `(n, 3)` whose rows add up to 1. |
| 3 | Don't modify the inputs (`*`, `/` and `**` all return new arrays, so this is automatic). |

## 5. Hints

- `**` is Python's power operator: `x ** 0.5` is the square root. It works on whole arrays.
- `.sum(axis=1, keepdims=True)` again, exactly as in `remove_margin`.
- For rule 1: `p_market.shape != p_model.shape` compares both dimensions at once.

## 6. Tests

| Test | Fails if... |
|---|---|
| `test_beta_zero_is_market_and_one_is_model` | beta 0 doesn't give the market, or beta 1 doesn't give the model |
| `test_worked_example_beta_half` | section 3's example comes out differently |
| `test_rows_sum_to_one_for_any_beta` | a row doesn't add up to 1 (including negative beta) |
| `test_shape_mismatch_rejected` | mismatched shapes are accepted |
| `test_inputs_not_modified` | an input array changed |
| `test_fit_beta_recovers_informative_model` | on simulated data where the model really knows something, the fitted beta isn't near 1 |
| `test_stack_test_passes_informative_and_fails_noise` | the full test can't tell a useful model from a useless one |

The last two run the whole stack test on made-up data where we know the right answer, so they check that the machinery works, not just your function.

## 7. Then run the real thing

```
football-edge stack-test --model dixon_coles
```
It prints the verdict and saves it to `reports/stack_test/dixon_coles/verdict.md`. The criteria it applies were fixed in advance: `docs/adr/0003-kill-rule.md`, including the addendum.
