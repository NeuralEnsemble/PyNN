
import glob, os
from nose.tools import assert_equal
from pyNN.random import NumpyRNG, RandomDistribution
from .registry import register


@register(exclude=["nemo"])
def scenario1(sim):
    """
    Balanced network of integrate-and-fire neurons.
    """
    cell_params = {
        'tau_m': 20.0, 'tau_syn_E': 5.0, 'tau_syn_I': 10.0, 'v_rest': -60.0,
        'v_reset': -60.0, 'v_thresh': -50.0, 'cm': 1.0, 'tau_refrac': 5.0,
        'e_rev_E': 0.0, 'e_rev_I': -80.0
    }
    stimulation_params = {'rate' : 100.0, 'duration' : 50.0}
    n_exc = 80
    n_inh = 20
    n_input = 20
    rngseed = 98765
    parallel_safe = True
    n_threads = 1
    pconn_recurr = 0.02
    pconn_input = 0.01
    tstop = 900.0
    delay = 0.2
    dt    = 0.1
    weights = {
        'excitatory': 4.0e-3,
        'inhibitory': 51.0e-3,
        'input': 0.1,
    }

    sim.setup(timestep=dt, min_delay=dt, threads=n_threads)
    all_cells = sim.Population(n_exc+n_inh, sim.IF_cond_exp(**cell_params), label="All cells")
    cells = {
        'excitatory': all_cells[:n_exc],
        'inhibitory': all_cells[n_exc:],
        'input': sim.Population(n_input, sim.SpikeSourcePoisson(**stimulation_params), label="Input")
    }

    rng = NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)
    uniform_distr = RandomDistribution('uniform',
                                       low=cell_params['v_reset'],
                                       high=cell_params['v_thresh'], rng=rng)
    all_cells.initialize(v=uniform_distr)

    connections = {}
    for name, pconn, receptor_type in (
        ('excitatory', pconn_recurr, 'excitatory'),
        ('inhibitory', pconn_recurr, 'inhibitory'),
        ('input',      pconn_input,  'excitatory'),
    ):
        connector = sim.FixedProbabilityConnector(pconn, rng=rng)
        syn = sim.StaticSynapse(weight=weights[name], delay=delay)
        connections[name] = sim.Projection(cells[name], all_cells, connector,
                                           syn, receptor_type=receptor_type,
                                           label=name)

    all_cells.record('spikes')
    cells['excitatory'][0:2].record('v')
    assert_equal(cells['excitatory'][0:2].grandparent, all_cells)

    sim.run(tstop)

    E_count = cells['excitatory'].mean_spike_count()
    I_count = cells['inhibitory'].mean_spike_count()
    print("Excitatory rate        : %g Hz" % (E_count*1000.0/tstop,))
    print("Inhibitory rate        : %g Hz" % (I_count*1000.0/tstop,))
    sim.end()



@register(exclude=["brian", "nemo"])
def scenario1a(sim):
    """
    Balanced network of integrate-and-fire neurons, built with the "low-level"
    API.
    """
    cell_params = {
        'tau_m': 10.0, 'tau_syn_E': 2.0, 'tau_syn_I': 5.0, 'v_rest': -60.0,
        'v_reset': -65.0, 'v_thresh': -55.0, 'cm': 0.5, 'tau_refrac': 2.5,
        'e_rev_E': 0.0, 'e_rev_I': -75.0
    }
    stimulation_params = {'rate': 80.0, 'duration': 50.0}
    n_exc = 80
    n_inh = 20
    n_input = 20
    rngseed = 87546
    parallel_safe = True
    n_threads = 1
    pconn_recurr = 0.03
    pconn_input = 0.01
    tstop = 1100.0
    delay = 1
    w_exc = 3.0e-3
    w_inh = 45.0e-3
    w_input = 0.12
    dt      = 0.1

    sim.setup(timestep=dt, min_delay=dt, threads=n_threads)
    iaf_neuron = sim.IF_cond_alpha(**cell_params)
    excitatory_cells = sim.create(iaf_neuron, n=n_exc)
    inhibitory_cells = sim.create(iaf_neuron, n=n_inh)
    inputs = sim.create(sim.SpikeSourcePoisson(**stimulation_params), n=n_input)
    all_cells = excitatory_cells + inhibitory_cells
    sim.initialize(all_cells, v=cell_params['v_rest'])

    sim.connect(excitatory_cells, all_cells, weight=w_exc, delay=delay,
                receptor_type='excitatory', p=pconn_recurr)
    sim.connect(inhibitory_cells, all_cells, weight=w_exc, delay=delay,
                receptor_type='inhibitory', p=pconn_recurr)
    sim.connect(inputs, all_cells, weight=w_input, delay=delay,
                receptor_type='excitatory', p=pconn_input)
    sim.record('spikes', all_cells, "scenario1a_%s_spikes.pkl" % sim.__name__)
    sim.record('v', excitatory_cells[0:2], "scenario1a_%s_v.pkl" % sim.__name__)

    sim.run(tstop)

    E_count = excitatory_cells.mean_spike_count()
    I_count = inhibitory_cells.mean_spike_count()
    print("Excitatory rate        : %g Hz" % (E_count*1000.0/tstop,))
    print("Inhibitory rate        : %g Hz" % (I_count*1000.0/tstop,))
    sim.end()
    for filename in glob.glob("scenario1a_*"):
        os.remove(filename)


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    scenario1(sim)
    scenario1a(sim)
