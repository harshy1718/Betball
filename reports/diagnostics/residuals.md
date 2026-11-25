# Where the live model misses (diagnostics)

Walk-forward predictions with the live calibration, 2019/20 to 2026/27, 13,007 matches, 5 leagues. 'gap' = the interval excludes no effect.

## 1. Promoted teams

Goal-difference residual per match for the team (negative = the model rated it too high). 95% intervals, blocks = league x week.

| Group | Team-matches | Residual | 95% interval |
|---|---|---|---|
| promoted, matches 1-5 | 545 | +0.135 | +0.003 to +0.266 **gap** |
| promoted, matches 6-10 | 545 | +0.071 | -0.058 to +0.211 |
| promoted, matches 11-19 | 875 | +0.154 | +0.057 to +0.261 **gap** |
| promoted, matches 20-end | 1,693 | +0.027 | -0.048 to +0.103 |
| others, matches 1-5 | 3,335 | -0.020 | -0.042 to +0.002 |
| others, matches 6-10 | 3,335 | -0.013 | -0.036 to +0.008 |
| others, matches 11-19 | 5,381 | -0.027 | -0.043 to -0.010 **gap** |
| others, matches 20-end | 10,305 | -0.004 | -0.015 to +0.009 |

Promoted, first 10: scored +0.033, conceded -0.070 vs expected per match.

## 2. Season starts

Slope of actual on expected goal difference (1.00 = the model's spread between teams is right; below 1 = too sure, strengths should be pulled toward average).

| Part of season | Matches | Slope | 95% interval |
|---|---|---|---|
| team match 1-6 | 2,313 | 0.999 | 0.919 to 1.077 |
| team match 7-15 | 3,306 | 0.899 | 0.829 to 0.964 **gap** |
| team match 16-end | 7,388 | 0.940 | 0.894 to 0.985 **gap** |

## 3. Rest days

Days since the team's previous league match (cup and European games are not in the data, so a 'long' rest can hide a midweek game). Goal-difference residual.

| Rest | Team-matches | Residual | 95% interval |
|---|---|---|---|
| 3 days or less | 2,565 | +0.031 | -0.010 to +0.071 |
| 4-5 days | 3,556 | -0.032 | -0.070 to +0.007 |
| 6-8 days | 14,202 | +0.007 | -0.003 to +0.017 |
| 9-13 days | 2,095 | -0.018 | -0.074 to +0.035 |
| 14+ days (international break) | 3,491 | -0.009 | -0.034 to +0.015 |

## 4. Team-specific home advantage

Each team-season's home-minus-away residual against the same team's next season: correlation **-0.046** over 483 pairs (pure chance: -0.091 to +0.090). Spread of the gap: 0.555 goals per match.

A correlation inside the chance band means the gap does not repeat: it is noise, and a per-team home advantage would only fit noise.

## 5. Goal-count spread and low scores

| Event | Predicted | Observed | Observed - predicted | Standard errors |
|---|---|---|---|---|
| 0-0 | 0.0692 | 0.0638 | -0.0054 | -2.4 |
| 1-0 or 0-1 | 0.1545 | 0.1612 | +0.0067 | +2.1 |
| 1-1 | 0.1199 | 0.1206 | +0.0006 | +0.2 |
| any draw | 0.2554 | 0.2516 | -0.0038 | -1.0 |
| 0 goals in the match | 0.0692 | 0.0638 | -0.0054 | -2.4 |
| 1 goals in the match | 0.1545 | 0.1612 | +0.0067 | +2.1 |
| 2 goals in the match | 0.2418 | 0.2410 | -0.0008 | -0.2 |
| 3 goals in the match | 0.2133 | 0.2225 | +0.0092 | +2.6 **gap** |
| 4 goals in the match | 0.1583 | 0.1568 | -0.0015 | -0.5 |
| 5 goals in the match | 0.0889 | 0.0866 | -0.0022 | -0.9 |
| 6+ goals in the match | 0.0741 | 0.0680 | -0.0061 | -2.7 **gap** |
| home team 0 goals | 0.2327 | 0.2264 | -0.0063 | -1.7 |
| away team 0 goals | 0.2962 | 0.2922 | -0.0040 | -1.0 |
| home team 4+ goals | 0.0866 | 0.0808 | -0.0058 | -2.4 |
| away team 4+ goals | 0.0517 | 0.0466 | -0.0051 | -2.7 **gap** |

Spread check (squared error / predicted variance; 1.00 = the predicted spread is right, above 1 = more spread than the model allows): home 0.940, away 0.952 (about +-0.025 by chance).

## What this says (2026-11-25)

About 30 rows are checked, so one or two "gaps" are expected by chance; the ones below are
consistent patterns, not single rows.

1. **Promoted teams are underrated.** Over their first 19 matches they beat the model by
   +0.126 goals per match (95%: +0.055 to +0.196), mostly by conceding fewer than expected;
   the same for clubs coming back (+0.126) and clubs new to the data (+0.125). It fades by the
   second half of the season. The model already pulls promoted sides towards an average-promoted
   prior; that prior or its pull is too pessimistic. **Worth a pre-registered fix.**
2. **No case for pulling strengths toward average at season start.** Early-season slope is
   1.00 with calibration (raw 1.08: if anything the raw model is too cautious early). Mid and late
   season the calibrated slope is 0.90-0.94, i.e. the calibration's 8% stretch of goal
   difference overshoots on goal difference, but calibration was chosen on match-result,
   over/under and both-score log loss, where it does help. Season-transition test: skip.
3. **Rest days: nothing.** No bucket differs from zero. Caveat: cup and European games are
   missing, so true congestion is not measured. Skip until fixture data includes them.
4. **Team-specific home advantage is noise.** A team's home-minus-away residual does not repeat
   next season (correlation -0.046, chance band +-0.09). Skip.
5. **Goals are less spread out than Poisson.** Spread ratio 0.94-0.95 (chance +-0.025): fewer
   0-0s, fewer 6+ goal games and fewer 4+ goal teams than predicted; more 1-goal and 3-goal
   games. So negative binomial (more spread) and zero-inflation (more 0-0s) both point the wrong
   way. A goal distribution that allows *less* spread (e.g. Conway-Maxwell-Poisson, or a
   generalised Poisson with negative dispersion) is the right test. **Worth a pre-registered
   test**, smaller expected gain than 1.
