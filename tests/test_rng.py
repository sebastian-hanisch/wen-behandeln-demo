"""SplitMix64 (kopiert aus ab-test-demo): Referenzwerte und vektorisierte Fassung gleich der skalaren."""

import numpy as np

from wbh_rng import SplitMix64, uniform_block


def test_known_reference_outputs_for_seed_zero():
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_block_equals_the_scalar_stream():
    for seed in (0, 1, 35, 123456789, 2**63 + 5):
        rng = SplitMix64(seed)
        scalar = np.array([rng.uniform() for _ in range(200)])
        assert np.array_equal(uniform_block(seed, 200), scalar)


def test_uniforms_are_inside_the_unit_interval_with_the_right_mean():
    u = uniform_block(99, 100000)
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.005
