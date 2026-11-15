# A1.1 Walk-forward folds: what to build and why

**You write:** `make_folds` in `src/football_edge/evaluation/folds.py` (the `Fold` class and the function signature are already there).
**Done when:** `pytest -m integration tests/unit/test_folds.py` shows 6 passed.
**Expected size:** about 20-30 lines of pandas/NumPy.

---

## 1. The problem this solves

We want to know how well the model would have predicted matches **if we had been running it at the time**.

The trap is accidental cheating. If the model is trained on the whole 2018/19 season and then "predicts" a match from October 2018, it has already seen how that match ended. Its score will look brilliant and mean nothing. That is called **leakage**, and it is the most common way a betting model looks profitable on paper and then loses real money.

A **fold** is one honest "pretend it's this date" experiment:

- **snapshot**: the moment we pretend it is now
- **train**: every match that finished *before* the snapshot (the model may learn from these)
- **test**: the matches in the *next 7 days* after the snapshot (the model must predict these blind)

`make_folds` produces the full list of these experiments, one per week, from August 2018 to the end of 2025/26. Later, the model is fitted once per fold and scored on that fold's test matches. Every match in the test seasons gets predicted exactly once, using only the past.

## 2. Picture it

```
time ──────────────────────────────────────────────────────────────►

fold 1:  [====== train: everything before 6 Aug 2018 ======] | test: 6-12 Aug |
fold 2:  [====== train: everything before 13 Aug 2018 ========] | test: 13-19 Aug |
fold 3:  [====== train: everything before 20 Aug 2018 ==========] | test: 20-26 Aug |
  ...
```

The training window **grows** every week and never moves its start. That's called an *expanding window*. Old matches still count, but the model will weight them down itself with time decay. That's the model's job, not yours.

## 3. A worked example with the real data

Take the first fold.

1. **Snapshot** = Monday 6 Aug 2018, 00:00.
2. **Train** = every row with `date < 2018-08-06`. That is **3,420 matches** (all of 2015/16, 2016/17 and 2017/18, for all three leagues: 9 × 380 = 3,420).
3. **Test** = rows with `2018-08-06 <= date < 2018-08-13`. That is **10 matches**: the EPL opening weekend, from Man United v Leicester on Fri 10 Aug to Southampton v Burnley on Sun 12 Aug. Serie A and La Liga hadn't started yet.
4. 3,420 ≥ `min_train_matches` (1,500), so the fold is kept.

The next fold's snapshot is Monday 13 Aug, and so on. Use these numbers to sanity-check your first output.

## 4. The rules, precisely

| # | Rule | Why |
|---|---|---|
| 1 | Snapshots are **Mondays at 00:00**. One fold per calendar week (Mon-Sun). | Matches have a date but no kickoff time. Cutting on a Monday means a whole weekend round sits on one side of the line. |
| 2 | train = `date < snapshot` (strictly less than). | `<=` would let a match on the snapshot day leak into its own training set. |
| 3 | test = `snapshot <= date < snapshot + 7 days`. | Each match falls in exactly one week. |
| 4 | Only matches whose `season` is in `first_test_season..last_test_season` (2018..2025) go into **test**. Train may include anything earlier. | 2015/16-2017/18 are history only. 2026/27 is the live season, kept out of evaluation. |
| 5 | Skip a week with **no test matches**. | Summer and winter breaks, and the COVID gap (no matches from 9 Mar to 8 Jun 2020). |
| 6 | Skip a fold with fewer than `min_train_matches` training rows. | A model trained on almost nothing is noise. (With this data it never triggers, but keep it for other datasets.) |
| 7 | Folds come out **sorted by snapshot**. | The evaluator replays them in time order. |
| 8 | Don't sort, filter in place, or add columns to the `matches` DataFrame you're given. | The caller still needs it unchanged. Work on copies or on derived Series. |
| 9 | `train_idx` and `test_idx` are **positional row numbers** (NumPy integer arrays). | The model will do `matches.iloc[train_idx]`. The loaded table has a 0..N-1 index, so positions and labels match. Don't rely on that, though: return positions. |

## 5. Suggested approach (in words, no code)

1. Work out each match's week-start Monday. Hint: subtract the day-of-week (Monday = 0) as a number of days from `date`.
2. Choose which matches are eligible for test (rule 4) and list the distinct week-starts among those, sorted.
3. For each week-start (the snapshot): select the train positions (rule 2) and the test positions (rule 3 plus rule 4), apply the skip rules (5 and 6), and build a `Fold`.
4. Return the list.

A simple loop over about 300 weeks is fine. You don't need to be clever; correctness matters more than speed here. `np.flatnonzero(boolean_mask)` turns a True/False mask into positions.

## 6. What to expect when it works

- **294 folds** (one per week that has test matches)
- **9,120 test matches in total** (8 seasons × 3 leagues × 380): every test-season match exactly once
- typically ~31 test matches per fold (min 1, max 58)
- first snapshot 2018-08-06; training size grows from 3,420 upwards

If your counts differ, something is off.

## 7. What each test checks

| Test | It fails if... |
|---|---|
| `test_train_strictly_before_snapshot` | any training match is on or after its snapshot (leakage) |
| `test_test_window_is_one_week_from_snapshot` | a test match falls outside its 7-day window |
| `test_no_train_test_overlap` | a match is in both train and test of the same fold |
| `test_every_test_match_in_exactly_one_fold` | a match is predicted twice, or one from 2015-17 / 2026 slips into test |
| `test_folds_are_chronological` | folds aren't in time order |
| `test_input_not_mutated` | your function changed the DataFrame it was given |

Running them:
```
football-edge ingest --src "raw data"            # only needed once, builds the data
pytest -m integration tests/unit/test_folds.py -v
```

## 8. Three questions to answer in your docstring

Write a sentence or two on each in the function's docstring. I'll review these along with the code.

1. **COVID restart.** 2019/20 stopped in March 2020 and resumed in June to August. Does any week contain matches from two seasons? (Check it. With this data the answer is no, but say how you would know.) What does the first fold after the restart train on?
2. **Expanding window.** Why use all history rather than, say, only the last two seasons? (Hint: who handles "old matches matter less"?)
3. **Promoted teams.** A newly promoted team has no matches in train. Your function must *not* drop its test matches. Why is that the model's problem and not the fold's?

## 9. Common mistakes to avoid

- Using `<=` instead of `<` for train (rule 2).
- Building weeks from the test seasons only but forgetting that **train must include the earlier seasons** too.
- Returning DataFrame index labels instead of positions (rule 9).
- Adding a `week` column straight onto `matches` (that breaks rule 8; compute it as a separate Series).
- Filtering test by the *week's* season instead of each *match's* `season`.

When the 6 tests pass, send me the file. I'll review it adversarially for leakage before we build the model on top of it.
