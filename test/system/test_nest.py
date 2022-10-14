import numpy as np
from numpy.testing import assert_array_equal

try:
    import pyNN.nest
    have_nest = True
except ImportError:
    have_nest = False

from pyNN.utility import init_logging
from pyNN.random import RandomDistribution
import pytest


def test_record_native_model():
    if not have_nest:
        pytest.skip("nest not available")
    nest = pyNN.nest
    from pyNN.random import RandomDistribution

    init_logging(logfile=None, debug=True)

    nest.setup()

    parameters = {'tau_m': 17.0}
    n_cells = 10
    p1 = nest.Population(n_cells, nest.native_cell_type("ht_neuron")(**parameters))
    p1.initialize(V_m=-70.0, Theta=-50.0)
    p1.set(theta_eq=-51.5)
    #assert_array_equal(p1.get('theta_eq'), -51.5*np.ones((10,)))
    assert p1.get('theta_eq') == -51.5
    print(p1.get('tau_m'))
    p1.set(tau_m=RandomDistribution('uniform', low=15.0, high=20.0))
    print(p1.get('tau_m'))

    current_source = nest.StepCurrentSource(times=[50.0, 110.0, 150.0, 210.0],
                                            amplitudes=[0.01, 0.02, -0.02, 0.01])
    p1.inject(current_source)

    p2 = nest.Population(1, nest.native_cell_type("poisson_generator")(rate=200.0))

    print("Setting up recording")
    p2.record('spikes')
    p1.record('V_m')

    connector = nest.AllToAllConnector()
    syn = nest.StaticSynapse(weight=0.001)

    prj_ampa = nest.Projection(p2, p1, connector, syn, receptor_type='AMPA')

    tstop = 250.0
    nest.run(tstop)

    vm = p1.get_data().segments[0].analogsignals[0]
    n_points = int(tstop / nest.get_time_step()) + 1
    assert vm.shape == (n_points, n_cells)
    assert vm.max() > 0.0  # should have some spikes


def test_native_stdp_model():
    if not have_nest:
        pytest.skip("nest not available")
    nest = pyNN.nest
    from pyNN.utility import init_logging

    init_logging(logfile=None, debug=True)

    nest.setup()

    p1 = nest.Population(10, nest.IF_cond_exp())
    p2 = nest.Population(10, nest.SpikeSourcePoisson())

    stdp_params = {'Wmax': 50.0, 'lambda': 0.015, 'weight': 0.001}
    stdp = nest.native_synapse_type("stdp_synapse")(**stdp_params)

    connector = nest.AllToAllConnector()

    prj = nest.Projection(p2, p1, connector, receptor_type='excitatory',
                          synapse_type=stdp)


def test_ticket240():
    if not have_nest:
        pytest.skip("nest not available")
    nest = pyNN.nest
    nest.setup(threads=4)
    parameters = {'tau_m': 17.0}
    p1 = nest.Population(4, nest.IF_curr_exp())
    p2 = nest.Population(5, nest.native_cell_type("ht_neuron")(**parameters))
    conn = nest.AllToAllConnector()
    syn = nest.StaticSynapse(weight=1.0)
    # This should be a nonstandard receptor type but I don't know of one to use.
    prj = nest.Projection(p1, p2, conn, syn, receptor_type='AMPA')
    connections = prj.get(('weight',), format='list')
    assert len(connections) > 0


def test_ticket244():
    if not have_nest:
        pytest.skip("nest not available")
    nest = pyNN.nest
    nest.setup(threads=4)
    p1 = nest.Population(4, nest.IF_curr_exp())
    p1.record('spikes')
    poisson_generator = nest.Population(3, nest.SpikeSourcePoisson(rate=1000.0))
    conn = nest.OneToOneConnector()
    syn = nest.StaticSynapse(weight=1.0)
    nest.Projection(poisson_generator, p1.sample(3), conn, syn, receptor_type="excitatory")
    nest.run(15)
    p1.get_data()


def test_ticket236():
    """Calling get_spike_counts() in the middle of a run should not stop spike recording"""
    if not have_nest:
        pytest.skip("nest not available")
    pynnn = pyNN.nest
    pynnn.setup()
    p1 = pynnn.Population(2, pynnn.IF_curr_alpha(), structure=pynnn.space.Grid2D())
    p1.record('spikes', to_file=False)
    src = pynnn.DCSource(amplitude=70)
    src.inject_into(p1[:])
    pynnn.run(50)
    s1 = p1.get_spike_counts()  # as expected, {1: 124, 2: 124}
    pynnn.run(50)
    s2 = p1.get_spike_counts()  # unexpectedly, still {1: 124, 2: 124}
    assert s1[p1[0]] < s2[p1[0]]


def test_issue237():
    if not have_nest:
        pytest.skip("nest not available")
    sim = pyNN.nest
    n_exc = 10
    sim.setup()
    exc_noise_in_exc = sim.Population(n_exc, sim.SpikeSourcePoisson, {'rate': 1000.})
    exc_cells = sim.Population(n_exc, sim.IF_cond_exp())
    exc_noise_connector = sim.OneToOneConnector()
    noise_ee_prj = sim.Projection(exc_noise_in_exc, exc_cells,
                                  exc_noise_connector, receptor_type="excitatory")
    noise_ee_prj.set(weight=1e-3)


def test_random_seeds():
    if not have_nest:
        pytest.skip("nest not available")
    sim = pyNN.nest
    data = []
    for seed in (854947309, 470924491):
        sim.setup(threads=1, rng_seed=seed)
        p = sim.Population(3, sim.SpikeSourcePoisson(rate=100.0))
        p.record('spikes')
        sim.run(100)
        data.append(p.get_data().segments[0].spiketrains)
    assert data[0] != data[1]


def test_tsodyks_markram_synapse():
    if not have_nest:
        pytest.skip("nest not available")
    import nest
    sim = pyNN.nest
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
    connections = nest.GetConnections(nest.NodeCollection(list(prj._sources)),
                                      synapse_model=prj.nest_synapse_model)
    tau_psc = np.array(nest.GetStatus(connections, 'tau_psc'))
    assert_array_equal(tau_psc, np.arange(0.2, 0.7, 0.1))


def test_native_electrode_types():
    """ Test of NativeElectrodeType class. (See issue #506)"""
    if not have_nest:
        pytest.skip("nest not available")
    sim = pyNN.nest
    dt = 0.1
    sim.setup(timestep=0.1, min_delay=0.1)
    current_sources = [sim.DCSource(amplitude=0.5, start=50.0, stop=400.0),
                       sim.native_electrode_type('dc_generator')(
                           amplitude=500.0, start=50.0 - dt, stop=400.0 - dt),
                       sim.StepCurrentSource(times=[50.0, 210.0, 250.0, 410.0],
                                             amplitudes=[0.4, 0.6, -0.2, 0.2]),
                       sim.native_electrode_type('step_current_generator')(
                           amplitude_times=[50.0 - dt, 210.0 - dt, 250.0 - dt, 410.0 - dt],
                           amplitude_values=[400.0, 600.0, -200.0, 200.0]),
                       sim.ACSource(start=50.0, stop=450.0, amplitude=0.2,
                                    offset=0.1, frequency=10.0, phase=180.0),
                       sim.native_electrode_type('ac_generator')(
                           start=50.0 - dt, stop=450.0 - dt, amplitude=200.0,
                           offset=100.0, frequency=10.0, phase=180.0),
                       sim.NoisyCurrentSource(mean=0.5, stdev=0.2, start=50.0,
                                              stop=450.0, dt=1.0),
                       sim.native_electrode_type('noise_generator')(
                           mean=500.0, std=200.0, start=50.0 - dt,
                           stop=450.0 - dt, dt=1.0), ]

    n_cells = len(current_sources)
    cells = sim.Population(n_cells, sim.IF_curr_exp(v_thresh=-55.0, tau_refrac=5.0, tau_m=10.0))

    for cell, current_source in zip(cells, current_sources):
        cell.inject(current_source)

    cells.record('v')
    sim.run(500)

    vm = cells.get_data().segments[0].filter(name="v")[0]
    assert_array_equal(vm[:, 0].magnitude, vm[:, 1].magnitude)
    assert_array_equal(vm[:, 2].magnitude, vm[:, 3].magnitude)


def test_issue529():
    # A combination of NEST Common synapse properties and FromListConnector doesn't work
    if not have_nest:
        pytest.skip("nest not available")
    import nest
    sim = pyNN.nest

    sim.setup()

    iaf_neuron = sim.native_cell_type('iaf_psc_exp')
    poisson = sim.native_cell_type('poisson_generator')

    p1 = sim.Population(10, iaf_neuron(tau_m=20.0, tau_syn_ex=3., tau_syn_in=3.))
    p2 = sim.Population(10, iaf_neuron(tau_m=20.0, tau_syn_ex=3., tau_syn_in=3.))

    nest.SetStatus(p2.node_collection, {'tau_minus': 20.})

    stdp = sim.native_synapse_type("stdp_synapse_hom")(**{
        'lambda': 0.005,
        'mu_plus': 0.,
        'mu_minus': 0.,
        'alpha': 1.1,
        'tau_plus': 20.,
        'Wmax': 10.,
    })

    W = np.random.rand(5)

    connections = [
        (0, 0, W[0]),
        (0, 1, W[1]),
        (0, 2, W[2]),
        (1, 5, W[3]),
        (6, 1, W[4]),
    ]

    ee_connector = sim.FromListConnector(connections, column_names=["weight"])

    prj_plastic = sim.Projection(
        p1, p2, ee_connector, receptor_type='excitatory', synapse_type=stdp)


def test_issue662a():
    """Setting tau_minus to a random distribution fails..."""
    if not have_nest:
        pytest.skip("nest not available")
    import nest
    sim = pyNN.nest

    sim.setup()
    p1 = sim.Population(5, sim.SpikeSourcePoisson(rate=100.0))
    p2 = sim.Population(10, sim.IF_cond_exp())

    syn = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            A_plus=0.2,
            A_minus=0.1,
            tau_minus=RandomDistribution('uniform', (20, 40)),
            tau_plus=RandomDistribution('uniform', (10, 20))
        ),
        weight_dependence=sim.AdditiveWeightDependence(w_min=0.0, w_max=0.01)
    )

    with pytest.raises(ValueError):
        sim.Projection(p1, p2, sim.AllToAllConnector(),
                       synapse_type=syn, receptor_type='excitatory')


def test_issue662b():
    """Setting tau_minus to a random distribution fails..."""
    if not have_nest:
        pytest.skip("nest not available")
    import nest
    sim = pyNN.nest

    sim.setup(min_delay=0.5)
    p1 = sim.Population(5, sim.SpikeSourcePoisson(rate=100.0))
    p2 = sim.Population(10, sim.IF_cond_exp())

    syn = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            A_plus=0.2,
            A_minus=0.1,
            tau_minus=30,
            tau_plus=RandomDistribution('uniform', (10, 20))
        ),
        weight_dependence=sim.AdditiveWeightDependence(w_min=0.0, w_max=0.01),
        weight=0.005
    )

    connections = sim.Projection(p1, p2, sim.AllToAllConnector(),
                                 synapse_type=syn,
                                 receptor_type='inhibitory')

    connections.set(tau_minus=25)  # RandomDistribution('uniform', (20,40)))
    # todo: check this worked
    with pytest.raises(ValueError):
        connections.set(tau_minus=RandomDistribution('uniform', (20, 40)))


if __name__ == '__main__':
    #data = test_random_seeds()
    test_native_electrode_types()
