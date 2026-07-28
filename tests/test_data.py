import numpy as np

from sudoku_diffusion.data import (
    MASK,
    all_solutions,
    count_solutions,
    is_valid_solution,
    make_puzzle,
    orbit_split,
    solved,
    symmetry_group,
    validity_score,
)


def test_all_solutions_count_and_validity():
    sols = all_solutions()
    assert sols.shape == (288, 16)
    assert all(is_valid_solution(s) for s in sols)
    assert len({s.tobytes() for s in sols}) == 288


def test_symmetry_group_closed_and_valid():
    sols = all_solutions()
    group = symmetry_group()
    assert group.shape == (128, 16)
    keys = {s.tobytes() for s in sols}
    s = sols[7]
    for p in group:
        assert s[p].tobytes() in keys  # symmetries map solutions to solutions


def test_orbit_split_disjoint_and_orbit_closed():
    train, ev = orbit_split(np.random.default_rng(12345))
    assert len(train) + len(ev) == 288
    tk = {s.tobytes() for s in train}
    ek = {s.tobytes() for s in ev}
    assert not (tk & ek)
    group = symmetry_group()
    for s in ev:  # no symmetry of an eval grid appears in train
        for p in group:
            assert s[p].tobytes() not in tk


def test_make_puzzle_unique():
    rng = np.random.default_rng(0)
    sols = all_solutions()
    puz = make_puzzle(sols[0], 6, rng)
    assert puz is not None
    assert int((puz != MASK).sum()) == 6
    assert count_solutions(puz) == 1
    clue = puz != MASK
    assert np.all(puz[clue] == sols[0][clue])


def test_solved_and_validity_score():
    sols = all_solutions()
    s = sols[0]
    empty = np.zeros(16, dtype=s.dtype)
    assert solved(s, empty)  # any valid grid solves the empty puzzle
    assert validity_score(s) == 1.0
    bad = s.copy()
    bad[0] = bad[1]  # duplicate in row/box
    assert not solved(bad, empty)
    assert validity_score(bad) < 1.0
    puz = s.copy()
    puz[5:] = MASK
    other = sols[1]
    if not np.all(other[:5] == s[:5]):
        assert not solved(other, puz)  # clue-inconsistent full grid
