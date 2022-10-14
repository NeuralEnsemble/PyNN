
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
    try:
        assert crossings[1] != crossings[2]
    except ValueError:
        assert not (crossings[1] == crossings[2]).all()
    print(crossings[1].size / spike_times.size)
    return data




if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator(("--plot-figure",
                               {"help": "generate a figure",
                                "action": "store_true"}))
    test_simple_stochastic_synapse(sim, plot_figure=args.plot_figure)
