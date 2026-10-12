import numpy as np
import pytest

from core.lifespan import time_to_threshold
from core.model import out_of_domain


def test_reached_interpolation():
    t = np.array([0, 100, 200.0]); e = np.array([0, 0.1, 0.3])
    r = time_to_threshold(t, e, 0.2)
    assert r.status == "reached" and abs(r.t_star - 150) < 1e-9


def test_not_reached():
    r = time_to_threshold(np.array([0, 100.0]), np.array([0, 0.05]), 0.2)
    assert r.status == "not_reached" and r.t_star is None


def test_already_exceeded():
    t = np.array([0, 100, 200.0]); e = np.array([0, 0.3, 0.4])
    assert time_to_threshold(t, e, 0.2, current_age=150).status == "already_exceeded"


def test_nonpositive_threshold():
    with pytest.raises(ValueError):
        time_to_threshold(np.array([0, 1.0]), np.array([0, 1.0]), 0)


def test_out_of_domain():
    meta = {"domain": {"temp_c": [20, 60]}}
    assert out_of_domain({"temp_c": 80}, meta)
    assert not out_of_domain({"temp_c": 40}, meta)
