
import numpy as np

try:
    import pyNN.neuroml
    have_neuroml = True
except ImportError:
    have_neuroml = False

import pytest


def test_save_validate_network():
    if not have_neuroml:
        pytest.skip("neuroml module not available")
    sim = pyNN.neuroml
    reference='Test0'

    sim.setup(reference=reference)
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=np.arange(10, 100, 10)))
    neurons = sim.Population(5, sim.IF_cond_exp(e_rev_I=-75))
    sim.end()

    from neuroml.utils import validate_neuroml2

    validate_neuroml2('%s.net.nml'%reference)
