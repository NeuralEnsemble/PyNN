
from __future__ import division
import numpy
import quantities as pq

from .registry import register

@register(exclude=['pcsim', 'moose', 'nemo'])
def test_EIF_cond_alpha_isfa_ista(sim, plot_figure=False):
    sim.setup(timestep=0.01, min_delay=0.1, max_delay=4.0)
    ifcell = sim.create(sim.EIF_cond_alpha_isfa_ista(
                            i_offset=1.0, tau_refrac=2.0, v_spike=-40))
    ifcell.record(['spikes', 'v', 'w'])
    ifcell.initialize(v=-65, w=0)
    sim.run(200.0)
    data = ifcell.get_data().segments[0]
    expected_spike_times = numpy.array([10.02, 25.52, 43.18, 63.42, 86.67,  113.13, 142.69, 174.79]) * pq.ms
    if plot_figure:
        import matplotlib.pyplot as plt
        vm = data.analogsignalarrays[0] 
        plt.plot(vm.times, vm)
        plt.plot(expected_spike_times, -40*numpy.ones_like(expected_spike_times), "ro")
        plt.savefig("test_EIF_cond_alpha_isfa_ista_%s.png" % sim.__name__)
    diff = (data.spiketrains[0] - expected_spike_times)/expected_spike_times
    assert abs(diff).max() < 0.01, abs(diff).max() 
    sim.end()
    return data
test_EIF_cond_alpha_isfa_ista.__test__ = False


@register(exclude=['pcsim', 'nemo'])
def test_HH_cond_exp(sim, plot_figure=False):
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
    sim.end()
    first_spike = v.times[numpy.where(v>0)[0][0]]
    assert first_spike/pq.ms - 2.95 < 0.01
test_HH_cond_exp.__test__ = False


@register(exclude=['pcsim', 'nemo', 'brian'])
def issue367(sim, plot_figure=False):
    # AdEx dynamics for delta_T=0
    sim.setup(timestep=0.001, min_delay=0.1, max_delay=4.0)
    v_thresh = -50
    ifcell = sim.create(sim.EIF_cond_exp_isfa_ista(
                        delta_T=0.0, i_offset=1.0, v_thresh=v_thresh, v_spike=-45))
    ifcell.record(('spikes', 'v'))
    ifcell.initialize(v=-70.6)
    sim.run(100.0)
    data = ifcell.get_data().segments[0]

    # we take the average membrane potential 0.1 ms before the spike and
    # compare it to the spike threshold
    spike_times = data.spiketrains[0]
    vm = data.analogsignalarrays[0]
    spike_bins = ((spike_times - 0.1*pq.ms)/vm.sampling_period).magnitude.astype(int)
    vm_before_spike = vm.magnitude[spike_bins]
    if plot_figure:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(vm.times, vm)
        plt.savefig("issue367_%s.png" % sim.__name__)
    print(sim.__name__, vm_before_spike)
    errmsg = "v_thresh = {0}, vm_before_spike.mean() = {1}".format(v_thresh, vm_before_spike.mean())
    assert abs((vm_before_spike.mean() - v_thresh)/v_thresh) < 0.01, errmsg
    sim.end()
    return data
issue367.__test__ = False


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator(("--plot-figure",
                               {"help": "generate a figure",
                                "action": "store_true"}))
    test_EIF_cond_alpha_isfa_ista(sim, plot_figure=args.plot_figure)
    test_HH_cond_exp(sim, plot_figure=args.plot_figure)
    issue367(sim, plot_figure=args.plot_figure)
