"""Der fortsetzbare Strom `Stream` und `mix_seed`: dieselben Zahlen wie der skalare SplitMix64, Fortsetzung ohne Lücke und Wiederholung."""

import numpy as np

from wbh_rng import SplitMix64
from wbh_stream import Stream, mix_seed


def test_stream_is_the_scalar_splitmix_stream_across_several_calls():
    s = Stream(2026)
    got = np.concatenate([s.uniform(5), s.uniform((2, 3)).ravel(), np.atleast_1d(s.uniform())])
    ref = SplitMix64(2026)
    assert np.array_equal(got, [ref.uniform() for _ in range(12)])


def test_integers_are_inside_the_range_and_roughly_uniform():
    x = Stream(7).integers(6, 60000)
    assert x.min() == 0 and x.max() == 5
    counts = np.bincount(x, minlength=6) / len(x)
    assert np.abs(counts - 1 / 6).max() < 0.01


def test_normal_has_mean_zero_and_unit_variance_and_the_right_shape():
    z = Stream(9).normal((200, 500))
    assert z.shape == (200, 500)
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01


def test_normal_with_odd_count_and_scalar():
    assert Stream(1).normal(7).shape == (7,)
    assert isinstance(float(Stream(1).normal()), float)


def test_mix_seed_separates_parts_and_order():
    seeds = {mix_seed(1, 2), mix_seed(2, 1), mix_seed(1, 2, 0), mix_seed(1), mix_seed(0, 1, 2)}
    assert len(seeds) == 5
    assert mix_seed(5, 6, 7) == mix_seed(5, 6, 7)
