
import numpy as np
from numpy.testing import assert_array_equal

try:
    import pyNN.brian2
    have_brian2 = True
except ImportError:
    have_brian2 = False

from pyNN import space
import pytest


def test_ticket235():
    if not have_brian2:
        pytest.skip("brian2 not available")
    pynnn = pyNN.brian2
    pynnn.setup()
    p1 = pynnn.Population(9, pynnn.IF_curr_alpha(), structure=space.Grid2D())
    p2 = pynnn.Population(9, pynnn.IF_curr_alpha(), structure=space.Grid2D())
    p1.record('spikes', to_file=False)
    p2.record('spikes', to_file=False)
    prj1_2 = pynnn.Projection(p1, p2, pynnn.OneToOneConnector(
    ), pynnn.StaticSynapse(weight=10.0), receptor_type='excitatory')
    # we note that the connectivity is as expected: a uniform diagonal
    prj1_2.get('weight', format='array')
    src = pynnn.DCSource(amplitude=70)
    src.inject_into(p1[:])
    pynnn.run(50)
    n_spikes_p1 = p1.get_spike_counts()
    # We see that both populations have fired uniformly as expected:
    n_spikes_p2 = p2.get_spike_counts()
    for n in n_spikes_p1.values():
        assert n == n_spikes_p1[p1[1]]
    for n in n_spikes_p2.values():
        assert n == n_spikes_p2[p2[1]]
    # With this new setup, only the second p2 unit should fire:
    # prj1_2.set(weight=[0, 20, 0, 0, 0, 0, 0, 0, 0])
    new_weights = np.where(np.eye(9), 0, np.nan)
    new_weights[1, 1] = 20.0
    prj1_2.set(weight=new_weights)
    # This looks good:
    prj1_2.get('weight', format='array')
    pynnn.run(50)
    n_spikes_p1 = p1.get_spike_counts()
    for n in n_spikes_p1.values():
        assert n == n_spikes_p1[p1[1]]
    # p2[1] should be ahead in spikes count, and others should not have
    # fired more. It is not what I observe:
    n_spikes_p2 = p2.get_spike_counts()
    assert n_spikes_p2[p2[1]] > n_spikes_p2[p2[0]]


def test_tsodyks_markram_synapse():
    if not have_brian2:
        pytest.skip("brian2 not available")
    sim = pyNN.brian2
    sim.setup()
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=np.arange(10, 100, 10)))
    neurons = sim.Population(5, sim.IF_cond_exp(
        e_rev_I=-75, tau_syn_I=np.arange(0.2, 0.7, 0.1)))
    synapse_type = sim.TsodyksMarkramSynapse(U=0.04, tau_rec=100.0,
                                             tau_facil=1000.0, weight=0.01,
                                             delay=0.5)
    connector = sim.AllToAllConnector()
    prj = sim.Projection(spike_source, neurons, connector,
                         receptor_type='inhibitory',
                         synapse_type=synapse_type)
    neurons.record('gsyn_inh')
    sim.run(100.0)
    tau_psc = prj._brian2_synapses[0][0].tau_syn_ * 1e3  # s --> ms
    assert_array_equal(tau_psc, np.arange(0.2, 0.7, 0.1))


def test_issue648():
    """
    https://github.com/NeuralEnsemble/PyNN/issues/648

    For a population of size 2:

      cells[0].inject(dc_source)
      cells[1].inject(dc_source)

    should give the same results as:

      cells.inject(dc_source)
    """
    if not have_brian2:
        pytest.skip("brian2 not available")
    sim = pyNN.brian2
    sim.setup()
    cells = sim.Population(2, sim.IF_curr_exp(v_rest = -65.0, v_thresh=-55.0,
                                              tau_refrac=5.0, i_offset=-1.0))
    dc_source = sim.DCSource(amplitude=0.5, start=25, stop=50)
    cells[0].inject(dc_source)
    cells[1].inject(dc_source)
    cells.record(['v'])
    sim.run(100)
