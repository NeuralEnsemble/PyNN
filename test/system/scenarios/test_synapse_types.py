
import sys
import numpy as np
from .fixtures import run_with_simulators


@run_with_simulators("nest", "neuron")
def test_simple_stochastic_synapse(sim, plot_figure=False):
    # in this test we connect
    sim.setup(min_delay=0.5)
    t_stop = 1000.0
    spike_times = np.arange(2.5, t_stop, 5.0)
    source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))
    neurons = sim.Population(4, sim.IF_cond_exp(tau_syn_E=1.0))
    synapse_type = sim.SimpleStochasticSynapse(weight=0.5,
                                               p=np.array([[0.0, 0.5, 0.5, 1.0]]))
    connections = sim.Projection(source, neurons, sim.AllToAllConnector(),
                                 synapse_type=synapse_type)
    source.record('spikes')
    neurons.record('gsyn_exc')
    sim.run(t_stop)

    data = neurons.get_data().segments[0]
    gsyn = data.analogsignals[0].rescale('uS')
    if plot_figure:
        import matplotlib.pyplot as plt
        for i in range(neurons.size):
            plt.subplot(neurons.size, 1, i+1)
            plt.plot(gsyn.times, gsyn[:, i])
        plt.savefig("test_simple_stochastic_synapse_%s.png" % sim.__name__)
    print(data.analogsignals[0].units)
    crossings = []
    for i in range(neurons.size):
        crossings.append(
            gsyn.times[:-1][np.logical_and(gsyn.magnitude[:-1, i] < 0.4, 0.4 < gsyn.magnitude[1:, i])])
    assert crossings[0].size == 0
    assert crossings[1].size < 0.6*spike_times.size
    assert crossings[1].size > 0.4*spike_times.size
    assert crossings[3].size == spike_times.size
    assert not np.array_equal(crossings[1], crossings[2])
    print(crossings[1].size / spike_times.size)
    if "pytest" not in sys.modules:
        return data


def _tsodyks_markram_releases(U, tau_rec, tau_facil, tau_syn, spike_times):
    """Fraction of resources released by each spike, from the exact solution of the
    Tsodyks-Markram model between spikes."""
    y = z = u = 0.0
    t_last = 0.0
    releases = []
    for t in spike_times:
        h = t - t_last
        z = (z * np.exp(-h / tau_rec)
             + y * (np.exp(-h / tau_syn) - np.exp(-h / tau_rec)) / (tau_syn / tau_rec - 1))
        y *= np.exp(-h / tau_syn)
        x = 1 - y - z
        if tau_facil > 0:
            u *= np.exp(-h / tau_facil)
            u += U * (1 - u)
        else:
            u = U
        releases.append(x * u)
        y += x * u
        t_last = t
    return np.array(releases)


@run_with_simulators("nest", "neuron", "brian2")
def test_tsodyks_markram_synapse(sim):
    """The peak postsynaptic conductance after each presynaptic spike matches the exact
    solution of the Tsodyks-Markram model, for a depressing synapse (tau_facil = 0)
    and a facilitating one."""
    dt, tau_syn, weight = 0.1, 5.0, 0.01
    spike_times = np.arange(10.0, 100.0, 10.0)
    parameters = {
        "depressing": dict(U=0.5, tau_rec=800.0, tau_facil=0.0),
        "facilitating": dict(U=0.04, tau_rec=100.0, tau_facil=1000.0),
    }
    sim.setup(timestep=dt)
    source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))
    targets = {}
    for label, p in parameters.items():
        targets[label] = sim.Population(1, sim.IF_cond_exp(tau_syn_I=tau_syn))
        targets[label].record("gsyn_inh")
        sim.Projection(source, targets[label], sim.AllToAllConnector(),
                       sim.TsodyksMarkramSynapse(weight=weight, delay=0.5, **p),
                       receptor_type="inhibitory")
    sim.run(100.0)
    for label, p in parameters.items():
        gsyn = targets[label].get_data().segments[0].filter(name="gsyn_inh")[0].magnitude[:, 0]
        peaks = np.array([gsyn[int(round(t / dt)):int(round((t + 10.0) / dt))].max()
                          for t in spike_times])
        releases = _tsodyks_markram_releases(tau_syn=tau_syn, spike_times=spike_times, **p)
        expected = np.array([np.sum(weight * releases[:k + 1]
                                    * np.exp(-(spike_times[k] - spike_times[:k + 1]) / tau_syn))
                             for k in range(spike_times.size)])
        # NEURON samples the peak one time step after it occurs, hence the tolerance
        assert np.allclose(peaks, expected, rtol=0.03), (label, peaks, expected)
    sim.end()


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator(("--plot-figure",
                               {"help": "generate a figure",
                                "action": "store_true"}))
    test_simple_stochastic_synapse(sim, plot_figure=args.plot_figure)
