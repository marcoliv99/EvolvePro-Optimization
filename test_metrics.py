"""
Simple tests for enrichment_factor() and apk() in metrics_functions.py

Each test builds a tiny DataFrame (already sorted by predicted score, which is
what both functions assume) and compares the returned value against a number
worked out by hand. This checks that the functions run without errors AND that
the math is correct.

Run with:   pytest -v test_metrics_functions.py
Or plainly: python test_metrics_functions.py
"""

import math
import pandas as pd
import pytest

from metrics import enrichment_factor, apk


def make_df(labels):
    """Helper: build a ranked DataFrame from a list of 0/1 activity labels.
    Index 0 = top-ranked prediction."""
    return pd.DataFrame({"activity_binary": labels})


# enrichment_factor()


def test_ef_perfect_ranking():
    # 100 variants, 10 active, and all 10 actives sit at the very top.
    # top 10% = 10 rows, all 10 are hits.
    # EF = (hits_top/n_top) / (n_active/n_total) = (10/10) / (10/100) = 10.0
    labels = [1] * 10 + [0] * 90
    df = make_df(labels)
    assert enrichment_factor(df, fraction=0.1) == pytest.approx(10.0)


def test_ef_random_like():
    # 100 compounds, 20 active. Top 10 rows contain exactly 2 actives.
    # EF = (2/10) / (20/100) = 0.2 / 0.2 = 1.0  -> no enrichment over random.
    labels = ([1, 1] + [0] * 8) + ([1] * 18 + [0] * 72)
    df = make_df(labels)
    assert df["activity_binary"].sum() == 20  # sanity check on the fixture
    assert enrichment_factor(df, fraction=0.1) == pytest.approx(1.0)


def test_ef_no_hits_in_top_returns_zero():
    # No actives in the selected top fraction -> function returns 0.0
    labels = [0] * 10 + [1] * 90
    df = make_df(labels)
    assert enrichment_factor(df, fraction=0.1) == 0.0


def test_ef_missing_column_raises():
    df = pd.DataFrame({"something_else": [1, 0, 1]})
    with pytest.raises(RuntimeError):
        enrichment_factor(df, fraction=0.1)


def test_ef_none_dataframe_raises():
    with pytest.raises(RuntimeError):
        enrichment_factor(None, fraction=0.1)


# apk()  (Average Precision at k)

def test_apk_all_top_k_active():
    # Top 3 are all active and there are exactly 3 actives in total.
    # precisions at each hit: 1/1, 2/2, 3/3 = 1 + 1 + 1 = 3
    # AP@k = 3 / (total actives = 3) = 1.0
    df = make_df([1, 1, 1, 0, 0])
    assert apk(df, k=3) == pytest.approx(1.0)


def test_apk_hand_computed_mixed_ranking():
    # ranking = [1,0,1,0,1], k=5, total actives = 3
    # hits at positions 1, 3, 5:
    #   pos 1 -> 1/1
    #   pos 3 -> 2/3
    #   pos 5 -> 3/5
    # sum = 1 + 0.6666... + 0.6 = 2.26666...
    # AP@k = 2.26666... / 3 = 0.755555...
    df = make_df([1, 0, 1, 0, 1])
    expected = (1 / 1 + 2 / 3 + 3 / 5) / 3
    assert apk(df, k=5) == pytest.approx(expected)


def test_apk_k_smaller_than_list():
    # Only the top k=2 rows are considered.
    # ranking top2 = [1, 0]; hit at pos 1 -> 1/1 = 1; sum = 1
    # total actives in whole df = 2  ->  AP@k = 1/2 = 0.5
    df = make_df([1, 0, 1, 0])
    assert apk(df, k=2) == pytest.approx(0.5)


def test_apk_no_actives_returns_zero():
    df = make_df([0, 0, 0, 0])
    assert apk(df, k=4) == 0.0


def test_apk_missing_column_raises():
    df = pd.DataFrame({"score": [0.9, 0.5, 0.1]})
    with pytest.raises(RuntimeError):
        apk(df, k=3)


def test_apk_none_dataframe_raises():
    with pytest.raises(RuntimeError):
        apk(None, k=3)


# Allow running without pytest installed as a runner.
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
