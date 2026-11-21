import numpy as np
import pandas as pd
import pytest

from football_edge.betting.staking import MonthStatus, StakingPolicy, select_bets

POLICY = {
    "monthly_budget": 10000,
    "currency": "INR",
    "tiers": {
        "1x2": [[0.50, 0.01], [0.60, 0.02], [0.70, 0.03]],
        "ou25": [[0.58, 0.01], [0.65, 0.02], [0.72, 0.03]],
        "btts": [[0.58, 0.01], [0.65, 0.02], [0.72, 0.03]],
    },
    "price_tolerance": 0.06,
    "max_match_fraction": 0.04,
    "stop_loss_fraction": 0.30,
    "stake_rounding": 10,
    "doubles": {"max_count": 2, "stake_fraction": 0.01, "min_leg_tier": "high"},
}
NOTHING_YET = MonthStatus(0.0, 0.0)


def _fc(p_h=0.75, p_d=0.15, over=0.5, btts=0.5, odds=(1.35, 5.0, 9.0), home="A", away="B"):
    return pd.DataFrame(
        {
            "league": ["E0"],
            "date": [pd.Timestamp("2026-11-22")],
            "home": [home],
            "away": [away],
            "p_h": [p_h],
            "p_d": [p_d],
            "p_a": [1 - p_h - p_d],
            "p_over25": [over],
            "p_under25": [1 - over],
            "p_btts_yes": [btts],
            "p_btts_no": [1 - btts],
            "odds_h": [odds[0]],
            "odds_d": [odds[1]],
            "odds_a": [odds[2]],
            "odds_over25": [np.nan],
            "odds_under25": [np.nan],
            "odds_btts_yes": [np.nan],
            "odds_btts_no": [np.nan],
        }
    )


def _pick(singles, market):
    return singles[singles["market"] == market].iloc[0]


def test_confidence_sets_stake():
    pol = StakingPolicy.from_dict(POLICY)
    singles, _, _ = select_bets(_fc(p_h=0.75), pol, NOTHING_YET)
    one = _pick(singles, "1x2")
    assert (one["selection"], one["tier"], one["stake"], one["status"]) == (
        "home",
        "high",
        300,
        "bet",
    )
    assert _pick(singles, "ou25")["status"] == "skip: probability too low"  # 50% is a coin flip


def test_price_guard_skips_short_prices():
    pol = StakingPolicy.from_dict(POLICY)
    singles, _, _ = select_bets(_fc(p_h=0.75, odds=(1.20, 6.0, 12.0)), pol, NOTHING_YET)
    one = _pick(singles, "1x2")  # 0.75 * 1.20 - 1 = -10%, worse than the 6% tolerance
    assert one["status"] == "skip: price too short" and one["stake"] == 0
    assert one["min_odds"] == pytest.approx(0.94 / 0.75)


def test_unpriced_selection_gets_min_odds_instruction():
    pol = StakingPolicy.from_dict(POLICY)
    singles, _, _ = select_bets(_fc(p_h=0.45, p_d=0.30, over=0.70), pol, NOTHING_YET)
    ou = _pick(singles, "ou25")
    assert ou["status"] == "bet if price >= min_odds" and ou["stake"] == 200


def test_match_cap_limits_exposure():
    pol = StakingPolicy.from_dict(POLICY)
    singles, _, _ = select_bets(_fc(p_h=0.75, over=0.75, btts=0.75), pol, NOTHING_YET)
    assert singles["stake"].sum() <= 400  # 3 x 300 asked, capped at 4% of 10,000


def test_stop_loss_and_budget():
    pol = StakingPolicy.from_dict(POLICY)
    singles, _, notes = select_bets(_fc(), pol, MonthStatus(-3000.0, 0.0))
    assert singles["stake"].sum() == 0 and "STOP-LOSS" in notes[0]
    singles, _, notes = select_bets(_fc(), pol, MonthStatus(-2000.0, 7900.0))
    assert singles["stake"].sum() <= 100  # only 10,000 - 2,000 - 7,900 left


def test_doubles_two_matches_only():
    pol = StakingPolicy.from_dict(POLICY)
    fc = pd.concat([_fc(home="A", away="B"), _fc(home="C", away="D")], ignore_index=True)
    _, doubles, _ = select_bets(fc, pol, NOTHING_YET)
    assert len(doubles) == 1  # one high-tier priced pick per match -> one double
    assert doubles["prob"].iloc[0] == pytest.approx(0.75 * 0.75)
    _, none, _ = select_bets(_fc(), pol, NOTHING_YET)
    assert none.empty


def test_bad_policy_rejected():
    bad = {**POLICY, "tiers": {**POLICY["tiers"], "1x2": [[0.6, 0.01], [0.5, 0.02], [0.7, 0.03]]}}
    with pytest.raises(ValueError, match="increase"):
        StakingPolicy.from_dict(bad)


def test_market_probability_sets_confidence_and_flags_disagreement():
    pol = StakingPolicy.from_dict(POLICY)
    fc = _fc(p_h=0.75, odds=(1.6, 4.5, 6.0)).assign(
        mkt_h=0.62, mkt_d=0.22, mkt_a=0.16, mkt_src_1x2="sharp"
    )
    one = _pick(select_bets(fc, pol, NOTHING_YET)[0], "1x2")
    assert one["prob"] == 0.62 and one["prob_source"] == "market (sharp)"
    assert one["model_prob"] == 0.75 and one["disagree"]  # 13 points apart
    assert one["tier"] == "normal" and one["stake"] == 200  # model alone would say high/300
    as_model = StakingPolicy.from_dict({**POLICY, "confidence": "model"})
    assert _pick(select_bets(fc, as_model, NOTHING_YET)[0], "1x2")["prob"] == 0.75
