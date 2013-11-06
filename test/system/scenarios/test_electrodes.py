

from nose.tools import assert_equal
import quantities as pq
from registry import register


@register(exclude=["pcsim", "nemo"])
def test_changing_electrode(sim):
    """
    Check that changing the values of the electrodes on the fly is taken into account
    """
    repeats = 2
    dt      = 0.1
    simtime = 100
    sim.setup(timestep=dt, min_delay=dt)
    p = sim.Population(1, sim.IF_curr_exp())
    c = sim.DCSource(amplitude=0.0)
    c.inject_into(p)
    p.record('v')

    for i in range(repeats):
        sim.run(simtime)
        c.amplitude += 0.1

    data = p.get_data().segments[0].analogsignalarrays[0]

    sim.end()

    # check that the value of v just before increasing the current is less than
    # the value at the end of the simulation
    assert data[int(simtime/dt), 0] < data[-1, 0]


@register()
def ticket226(sim):
    """
    Check that the start time of DCSources is correctly taken into account
    http://neuralensemble.org/trac/PyNN/ticket/226)
    """
    sim.setup(timestep=0.1)

    cell = sim.Population(1, sim.IF_curr_alpha(tau_m=20.0, cm=1.0, v_rest=-60.0,
                                               v_reset=-60.0))
    cell.initialize(v=-60.0)
    inj = sim.DCSource(amplitude=1.0, start=10.0, stop=20.0)
    cell.inject(inj)
    cell.record_v()
    sim.run(30.0)
    v = cell.get_data().segments[0].filter(name='v')[0][:, 0]
    v_10p0 = v[abs(v.times-10.0*pq.ms)<0.01*pq.ms][0]
    assert abs(v_10p0 - -60.0*pq.mV) < 1e-10
    v_10p1 = v[abs(v.times-10.1*pq.ms)<0.01*pq.ms][0]
    assert v_10p1 > -59.99*pq.mV, v_10p1


@register()
def issue165(sim):
    """Ensure that anonymous current sources are not lost."""
    sim.setup(time_step=0.1)
    p = sim.Population(1, sim.IF_cond_exp())
    p.inject(sim.DCSource(amplitude=1.0, start=10.0, stop=20.0))
    p.record('v')
    sim.run(20.0)
    data = p.get_data().segments[0].filter(name='v')[0]
    
    assert_equal(data[99, 0], -65.0)
    assert data[150, 0] > -65.0


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    test_changing_electrode(sim)
    ticket226(sim)
    issue165(sim)
