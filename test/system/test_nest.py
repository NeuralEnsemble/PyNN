from nose.plugins.skip import SkipTest
from .scenarios.registry import registry
from nose.tools import assert_equal, assert_not_equal
from pyNN.utility import init_logging, assert_arrays_equal
import numpy

try:
    import pyNN.nest
    have_nest = True
except ImportError:
    have_nest = False


def test_scenarios():
    for scenario in registry:
        if "nest" not in scenario.exclude:
            scenario.description = scenario.__name__
            if have_nest:
                yield scenario, pyNN.nest
            else:
                raise SkipTest


def test_record_native_model():
    nest = pyNN.nest
    from pyNN.random import RandomDistribution

    init_logging(logfile=None, debug=True)

    nest.setup()

    parameters = {'Tau_m': 17.0}
    n_cells = 10
    p1 = nest.Population(n_cells, nest.native_cell_type("ht_neuron")(**parameters))
    p1.initialize(V_m=-70.0, Theta=-50.0)
    p1.set(Theta_eq=-51.5)
    #assert_arrays_equal(p1.get('Theta_eq'), -51.5*numpy.ones((10,)))
    assert_equal(p1.get('Theta_eq'), -51.5)
    print(p1.get('Tau_m'))
    p1.set(Tau_m=RandomDistribution('uniform', low=15.0, high=20.0))
    print(p1.get('Tau_m'))

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

    vm = p1.get_data().segments[0].analogsignalarrays[0]
    n_points = int(tstop/nest.get_time_step()) + 1
    assert_equal(vm.shape, (n_points, n_cells))
    assert vm.max() > 0.0 # should have some spikes


def test_native_stdp_model():
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
    nest = pyNN.nest
    nest.setup(threads=4)
    parameters = {'Tau_m': 17.0}
    p1 = nest.Population(4, nest.IF_curr_exp())
    p2 = nest.Population(5, nest.native_cell_type("ht_neuron")(**parameters))
    conn = nest.AllToAllConnector()
    syn = nest.StaticSynapse(weight=1.0)
    prj = nest.Projection(p1, p2, conn, syn, receptor_type='AMPA') # This should be a nonstandard receptor type but I don't know of one to use.
    connections = prj.get(('weight',), format='list')
    assert len(connections) > 0


def test_ticket244():
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
    pynnn = pyNN.nest
    pynnn.setup()
    p1 = pynnn.Population(2, pynnn.IF_curr_alpha(), structure=pynnn.space.Grid2D())
    p1.record('spikes', to_file=False)
    src = pynnn.DCSource(amplitude=70)
    src.inject_into(p1[:])
    pynnn.run(50)
    s1 = p1.get_spike_counts() # as expected, {1: 124, 2: 124}
    pynnn.run(50)
    s2 = p1.get_spike_counts() # unexpectedly, still {1: 124, 2: 124}
    assert s1[p1[0]] < s2[p1[0]]


def test_issue237():
    sim = pyNN.nest
    n_exc = 10
    sim.setup()
    exc_noise_in_exc = sim.Population(n_exc, sim.SpikeSourcePoisson, {'rate' : 1000.})
    exc_cells = sim.Population(n_exc, sim.IF_cond_exp())
    exc_noise_connector = sim.OneToOneConnector()
    noise_ee_prj = sim.Projection(exc_noise_in_exc, exc_cells, exc_noise_connector, receptor_type="excitatory")
    noise_ee_prj.set(weight=1e-3)


def test_random_seeds():
    sim = pyNN.nest
    data = []
    for seed in (854947309, 470924491):
        sim.setup(threads=1, rng_seeds=[seed])
        p = sim.Population(3, sim.SpikeSourcePoisson(rate=100.0))
        p.record('spikes')
        sim.run(100)
        data.append(p.get_data().segments[0].spiketrains)
    assert_not_equal(*data)


def test_tsodyks_markram_synapse():
    import nest
    sim = pyNN.nest
    sim.setup()
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=numpy.arange(10, 100, 10)))
    neurons = sim.Population(5, sim.IF_cond_exp(e_rev_I=-75, tau_syn_I=numpy.arange(0.2, 0.7, 0.1)))
    synapse_type = sim.TsodyksMarkramSynapse(U=0.04, tau_rec=100.0,
                                             tau_facil=1000.0, weight=0.01,
                                             delay=0.5)
    connector = sim.AllToAllConnector()
    prj = sim.Projection(spike_source, neurons, connector,
                         receptor_type='inhibitory',
                         synapse_type=synapse_type)
    neurons.record('gsyn_inh')
    sim.run(100.0)
    connections = nest.GetConnections(prj._sources.tolist(), synapse_model=prj.nest_synapse_model)
    tau_psc = numpy.array(nest.GetStatus(connections, 'tau_psc'))
    assert_arrays_equal(tau_psc, numpy.arange(0.2, 0.7, 0.1))


if __name__ == '__main__':
    data = test_random_seeds()
