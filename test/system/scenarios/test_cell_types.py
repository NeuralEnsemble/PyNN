
import numpy
import quantities as pq
from registry import register


@register(exclude=['pcsim', 'moose', 'nemo'])
def test_EIF_cond_alpha_isfa_ista(sim):
    sim.setup(timestep=0.01, min_delay=0.1, max_delay=4.0)
    ifcell = sim.create(sim.EIF_cond_alpha_isfa_ista(
                            i_offset=1.0, tau_refrac=2.0, v_spike=-40))
    ifcell.record(['spikes', 'v'])
    ifcell.initialize(v=-65)
    sim.run(200.0)
    data = ifcell.get_data().segments[0]
    expected_spike_times = numpy.array([10.02, 25.52, 43.18, 63.42, 86.67,  113.13, 142.69, 174.79]) * pq.ms
    diff = (data.spiketrains[0] - expected_spike_times)/expected_spike_times
    assert abs(diff).max() < 0.001
    sim.end()
    return data


@register(exclude=['pcsim', 'nemo'])
def test_HH_cond_exp(sim):
    sim.setup(timestep=0.001, min_delay=0.1)
    cellparams = {
        'gbar_Na'   : 20.0,
        'gbar_K'    : 6.0,
        'g_leak'    : 0.01,
        'cm'        : 0.2,
        'v_offset'  : -63.0,
        'e_rev_Na'  : 50.0,
        'e_rev_K'   : -90.0,
        'e_rev_leak': -65.0,
        'e_rev_E'   : 0.0,
        'e_rev_I'   : -80.0,
        'tau_syn_E' : 0.2,
        'tau_syn_I' : 2.0,
        'i_offset'  : 1.0,
    }
    hhcell = sim.create(sim.HH_cond_exp(**cellparams))
    sim.initialize(hhcell, v=-64.0)
    hhcell.record('v')
    sim.run(20.0)
    v = hhcell.get_data().segments[0].filter(name='v')[0]
    first_spike = v.times[numpy.where(v>0)[0][0]]
    assert first_spike/pq.ms - 2.95 < 0.01


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    test_EIF_cond_alpha_isfa_ista(sim)
    test_HH_cond_exp(sim)
