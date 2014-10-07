
import numpy
from .registry import register


@register(exclude=["nemo"])
def ticket166(sim, plot_figure=False):
    """
    Check that changing the spike_times of a SpikeSourceArray mid-simulation
    works (see http://neuralensemble.org/trac/PyNN/ticket/166)
    """
    dt = 0.1 # ms
    t_step = 100.0 # ms
    lag = 3.0 # ms

    sim.setup(timestep=dt, min_delay=dt)

    spikesources = sim.Population(2, sim.SpikeSourceArray())
    cells = sim.Population(2, sim.IF_cond_exp())
    syn = sim.StaticSynapse(weight=0.01)
    conn = sim.Projection(spikesources, cells, sim.OneToOneConnector(), syn)
    cells.record('v')

    spiketimes = numpy.arange(2.0, t_step, t_step/13.0)
    spikesources[0].spike_times = spiketimes
    spikesources[1].spike_times = spiketimes + lag

    t = sim.run(t_step) # both neurons depolarized by synaptic input
    t = sim.run(t_step) # no more synaptic input, neurons decay

    spiketimes += 2*t_step
    spikesources[0].spike_times = spiketimes
    # note we add no new spikes to the second source
    t = sim.run(t_step) # first neuron gets depolarized again

    vm = cells.get_data().segments[0].analogsignalarrays[0]
    final_v_0 = vm[-1, 0]
    final_v_1 = vm[-1, 1]

    sim.end()

    if plot_figure:
        import matplotlib.pyplot as plt
        plt.plot(vm.times, vm[:, 0])
        plt.plot(vm.times, vm[:, 1])
        plt.savefig("ticket166_%s.png" % sim.__name__)

    assert final_v_0 > -60.0  # first neuron has been depolarized again
    assert final_v_1 < -64.99 # second neuron has decayed back towards rest


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator(("--plot-figure",
                               {"help": "generate a figure",
                                "action": "store_true"}))
    ticket166(sim, plot_figure=args.plot_figure)
