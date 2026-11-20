"""Tests for the reference greedy selector in coursework3_greedy.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coursework3_greedy import greedy_select  # noqa: E402

Y = np.array(["a", "a", "a", "b", "b", "b"])
CLASSES = np.unique(Y)


def test_starts_with_best_individual():
    preds = [
        ["a", "a", "a", "a", "a", "a"],  # 3/6
        ["a", "a", "a", "b", "b", "a"],  # 5/6
    ]
    selected, scores = greedy_select(preds, Y, CLASSES)
    assert selected[0] == 1
    assert scores[0] == pytest.approx(5 / 6)


def test_first_candidate_wins_tie_for_best_individual():
    preds = [
        ["a", "a", "a", "a", "b", "b"],  # 5/6
        ["a", "a", "b", "b", "b", "b"],  # 5/6
    ]
    selected, _ = greedy_select(preds, Y, CLASSES)
    assert selected[0] == 0


def test_stops_without_strict_improvement():
    # Adding either candidate to candidate 0 leaves accuracy at 5/6 or lower.
    preds = [
        ["a", "a", "a", "a", "b", "b"],
        ["a", "a", "a", "a", "b", "b"],
        ["b", "b", "b", "a", "a", "a"],
    ]
    selected, scores = greedy_select(preds, Y, CLASSES)
    assert selected == [0]
    assert scores == [pytest.approx(5 / 6)]


def test_adds_candidate_that_improves_vote():
    # Candidates 0 and 1 are each wrong on one "a" case. In the pair, each of
    # those cases is a tied vote, which goes to "a", so the pair is perfect.
    preds = [
        ["b", "a", "a", "b", "b", "b"],  # 5/6
        ["a", "b", "a", "b", "b", "b"],  # 5/6
        ["a", "a", "a", "a", "b", "b"],  # 5/6
    ]
    selected, scores = greedy_select(preds, Y, CLASSES)
    assert selected == [0, 1]
    assert scores == pytest.approx([5 / 6, 1.0])


def test_greedy_is_myopic():
    # Each candidate is right on 4 of 6 cases and the vote of all three is
    # right on every case, but no pair beats a single candidate, so forward
    # selection stops after the first.
    preds = [
        ["a", "a", "b", "b", "b", "a"],
        ["a", "b", "a", "a", "b", "b"],
        ["b", "a", "a", "b", "a", "b"],
    ]
    selected, scores = greedy_select(preds, Y, CLASSES)
    assert selected == [0]
    assert scores == [pytest.approx(4 / 6)]


def test_vote_ties_go_to_first_class():
    # Two candidates disagree on every case, so every vote is tied and goes
    # to "a": the pair scores 3/6, which is no better than candidate 0 alone.
    preds = [
        ["a", "a", "a", "b", "b", "b"],
        ["b", "b", "b", "a", "a", "a"],
    ]
    selected, scores = greedy_select(preds, Y, CLASSES)
    assert selected == [0]
    assert scores == [1.0]


def test_scores_follow_selection_and_strictly_increase():
    rng = np.random.default_rng(0)
    y = rng.choice(["x", "y", "z"], size=40)
    preds = np.where(rng.random((8, 40)) < 0.6, y, rng.choice(["x", "y", "z"], (8, 40)))
    selected, scores = greedy_select(preds, y, np.unique(y))
    assert len(selected) == len(set(selected)) == len(scores)
    assert all(b > a for a, b in zip(scores, scores[1:]))
    for i in range(len(selected)):
        sub = preds[selected[: i + 1]]
        votes = np.stack([(sub == c).sum(axis=0) for c in np.unique(y)], axis=1)
        vote_pred = np.unique(y)[np.argmax(votes, axis=1)]
        assert scores[i] == pytest.approx(np.mean(vote_pred == y))


def test_first_candidate_wins_tie_for_addition():
    # Candidates 1 and 2 both lift the pair to 6/6; candidate 1 is first.
    preds = [
        ["b", "a", "a", "b", "b", "b"],
        ["a", "b", "a", "b", "b", "b"],
        ["a", "a", "b", "b", "b", "b"],
    ]
    selected, scores = greedy_select(preds, Y, CLASSES)
    assert selected == [0, 1]
    assert scores == pytest.approx([5 / 6, 1.0])


def test_selects_every_candidate_when_each_addition_improves():
    preds = [["b", "a", "a", "b", "b", "b"], ["a", "b", "a", "b", "b", "b"]]
    selected, _ = greedy_select(preds, Y, CLASSES)
    assert selected == [0, 1]


def test_single_candidate():
    selected, scores = greedy_select([["a", "a", "a", "b", "b", "a"]], Y, CLASSES)
    assert selected == [0]
    assert scores == [pytest.approx(5 / 6)]


def test_integer_labels():
    y = np.array([0, 0, 1, 1])
    preds = [[0, 1, 1, 1], [0, 0, 1, 1]]
    selected, scores = greedy_select(preds, y, np.unique(y))
    assert selected == [1]
    assert scores == [1.0]


def test_rejects_unknown_label():
    with pytest.raises(ValueError):
        greedy_select([["a", "a", "a", "b", "b", "q"]], Y, CLASSES)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        greedy_select([["a", "b"]], Y, CLASSES)
