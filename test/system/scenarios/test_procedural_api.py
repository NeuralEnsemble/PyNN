
import numpy as np
from numpy.testing import assert_allclose
from pyNN.utility import init_logging
from .registry import register


@register()
def ticket195(sim):
    """
    Check that the `connect()` function works correctly with single IDs (see
    http://neuralensemble.org/trac/PyNN/ticket/195)
    """
    init_logging(None, debug=True)
    sim.setup(timestep=0.01)
    pre = sim.Population(10, sim.SpikeSourceArray(spike_times=range(1, 10)))
    post = sim.Population(10, sim.IF_cond_exp())
    #sim.connect(pre[0], post[0], weight=0.01, delay=0.1, p=1)
    sim.connect(pre[0:1], post[0:1], weight=0.01, delay=0.1, p=1)
    #prj = sim.Projection(pre, post, sim.FromListConnector([(0, 0, 0.01, 0.1)]))
    post.record(['spikes', 'v'])
    sim.run(100.0)
    assert_allclose(post.get_data().segments[0].spiketrains[0].magnitude, np.array([13.4]), 0.5)
    sim.end()

if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    ticket195(sim)
