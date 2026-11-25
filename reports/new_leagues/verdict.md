# Adding Bundesliga and Ligue 1 (ADR 0024)

Log loss per match (result + over/under 2.5 + both-score), lower is better.

## 2020/21-2025/26

| League | Matches | Nox (5-league fit) | Nox (old fit) | League baseline |
|---|---|---|---|---|
| Premier League | 2,280 | 2.3517 | 2.3522 | 2.4518 |
| Serie A | 2,280 | 2.3591 | 2.3600 | 2.4746 |
| La Liga | 2,280 | 2.3525 | 2.3532 | 2.4538 |
| Bundesliga | 1,836 | 2.3181 | – | 2.4184 |
| Ligue 1 | 2,058 | 2.3828 | – | 2.4606 |

* Old leagues: 5-league fit minus old fit -0.00068 per match (rule: not worse than +0.00100) -> PASS
* Bundesliga: Nox minus baseline -0.10031 per match (95% interval -0.12219 to -0.07803) -> PASS
* Ligue 1: Nox minus baseline -0.07779 per match (95% interval -0.09599 to -0.05968) -> PASS

## 2026/27 so far

| League | Matches | Nox (5-league fit) | Nox (old fit) | League baseline |
|---|---|---|---|---|
| Premier League | 110 | 2.4252 | 2.4263 | 2.5105 |
| Serie A | 120 | 2.2828 | 2.2826 | 2.4538 |
| La Liga | 129 | 2.2707 | 2.2710 | 2.4572 |
| Bundesliga | 90 | 2.2744 | – | 2.3647 |
| Ligue 1 | 99 | 2.3401 | – | 2.4337 |

* Old leagues: 5-league fit minus old fit -0.00037 per match (rule: not worse than +0.00100) -> PASS
* Bundesliga: Nox minus baseline -0.09033 per match -> PASS
* Ligue 1: Nox minus baseline -0.09360 per match -> PASS

**PASS: add the leagues.**
