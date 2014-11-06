
import numpy
from nose.tools import assert_equal
from .registry import register

@register(exclude=["moose", "nemo"])
def scenario2(sim):
    """
    Array of neurons, each injected with a different current.

    firing period of a IF neuron injected with a current I:

    T = tau_m*log(I*tau_m/(I*tau_m - v_thresh*cm))

    (if v_rest = v_reset = 0.0)

    we set the refractory period to be very large, so each neuron fires only
    once (except neuron[0], which never reaches threshold).
    """
    n = 83
    t_start = 25.0
    duration = 100.0
    t_stop = 150.0
    tau_m = 20.0
    v_thresh = 10.0
    cm = 1.0
    cell_params = {"tau_m": tau_m, "v_rest": 0.0, "v_reset": 0.0,
                   "tau_refrac": 100.0, "v_thresh": v_thresh, "cm": cm}
    I0 = (v_thresh*cm)/tau_m
    sim.setup(timestep=0.01, min_delay=0.1, spike_precision="off_grid")
    neurons = sim.Population(n, sim.IF_curr_exp(**cell_params))
    neurons.initialize(v=0.0)
    I = numpy.arange(I0, I0+1.0, 1.0/n)
    currents = [sim.DCSource(start=t_start, stop=t_start+duration, amplitude=amp)
                for amp in I]
    for j, (neuron, current) in enumerate(zip(neurons, currents)):
        if j%2 == 0:                      # these should
            neuron.inject(current)        # be entirely
        else:                             # equivalent
            current.inject_into([neuron])
    neurons.record(['spikes', 'v'])

    sim.run(t_stop)

    spiketrains = neurons.get_data().segments[0].spiketrains
    assert_equal(len(spiketrains), n)
    assert_equal(len(spiketrains[0]), 0) # first cell does not fire
    assert_equal(len(spiketrains[1]), 1) # other cells fire once
    assert_equal(len(spiketrains[-1]), 1) # other cells fire once
    expected_spike_times = t_start + tau_m*numpy.log(I*tau_m/(I*tau_m - v_thresh*cm))
    a = spike_times = [numpy.array(st)[0] for st in spiketrains[1:]]
    b = expected_spike_times[1:]
    max_error = abs((a-b)/b).max()
    print("max error =", max_error)
    assert max_error < 0.005, max_error
    sim.end()
    return a,b, spike_times


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    scenario2(sim)
