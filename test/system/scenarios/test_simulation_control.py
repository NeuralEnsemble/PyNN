
from nose.tools import assert_almost_equal, assert_raises
from pyNN.utility import assert_arrays_equal, assert_arrays_almost_equal
from .registry import register


@register()
def test_reset(sim):
    """
    Run the same simulation n times without recreating the network,
    and check the results are the same each time.
    """
    repeats = 3
    dt      = 1
    sim.setup(timestep=dt, min_delay=dt)
    p = sim.Population(1, sim.IF_curr_exp(i_offset=0.1))
    p.record('v')

    for i in range(repeats):
        sim.run(10.0)
        sim.reset()
    data = p.get_data(clear=False)
    sim.end()

    assert len(data.segments) == repeats
    for segment in data.segments[1:]:
        assert_arrays_almost_equal(segment.analogsignalarrays[0],
                                   data.segments[0].analogsignalarrays[0], 1e-11)
test_reset.__test__ = False


@register()
def test_reset_with_clear(sim):
    """
    Run the same simulation n times without recreating the network,
    and check the results are the same each time.
    """
    repeats = 3
    dt      = 1
    sim.setup(timestep=dt, min_delay=dt)
    p = sim.Population(1, sim.IF_curr_exp(i_offset=0.1))
    p.record('v')

    data = []
    for i in range(repeats):
        sim.run(10.0)
        data.append(p.get_data(clear=True))
        sim.reset()

    sim.end()

    for rec in data:
        assert len(rec.segments) == 1
        assert_arrays_almost_equal(rec.segments[0].analogsignalarrays[0],
                                   data[0].segments[0].analogsignalarrays[0], 1e-11)
test_reset_with_clear.__test__ = False


@register()
def test_setup(sim):
    """
    Run the same simulation n times, recreating the network each time,
    and check the results are the same each time.
    """
    n = 3
    data = []
    dt   = 1

    for i in range(n):
        sim.setup(timestep=dt, min_delay=dt)
        p = sim.Population(1, sim.IF_curr_exp(i_offset=0.1))
        p.record('v')
        sim.run(10.0)
        data.append(p.get_data())
        sim.end()

    assert len(data) == n
    for block in data:
        assert len(block.segments) == 1
        signals = block.segments[0].analogsignalarrays
        assert len(signals) == 1
        assert_arrays_equal(signals[0], data[0].segments[0].analogsignalarrays[0])
test_setup.__test__ = False


@register()
def test_run_until(sim):
    sim.setup(timestep=0.1)
    p = sim.Population(1, sim.IF_cond_exp())
    sim.run_until(12.7)
    assert_almost_equal(sim.get_current_time(), 12.7, 10)
    sim.run_until(12.7)
    assert_almost_equal(sim.get_current_time(), 12.7, 10)
    sim.run_until(99.9)
    assert_almost_equal(sim.get_current_time(), 99.9, 10)
    assert_raises(ValueError, sim.run_until, 88.8)
    sim.end()
test_run_until.__test__ = False


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    test_reset(sim)
    test_reset_with_clear(sim)
    test_setup(sim)
    test_run_until(sim)
