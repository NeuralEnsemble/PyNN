from nose.plugins.skip import SkipTest
from scenarios import scenarios
from nose.tools import assert_equal

try:
    import pyNN.nest
    have_nest = True
except ImportError:
    have_nest = False

def test_scenarios():
    for scenario in scenarios:
        if "nest" not in scenario.exclude:
            scenario.description = scenario.__name__
            if have_nest:
                yield scenario, pyNN.nest
            else:
                raise SkipTest
        
        
def test_record_native_model():
    nest = pyNN.nest
    from pyNN.random import RandomDistribution
    from pyNN.utility import init_logging

    init_logging(logfile=None, debug=True)
    
    nest.setup()
    
    parameters = {'Tau_m': 17.0}
    n_cells = 10
    p1 = nest.Population(n_cells, nest.native_cell_type("ht_neuron"), parameters)
    p1.initialize('V_m', -70.0)
    p1.initialize('Theta', -50.0)
    p1.set('Theta_eq', -51.5)
    assert_equal(p1.get('Theta_eq'), [-51.5]*10)
    print p1.get('Tau_m')
    p1.rset('Tau_m', RandomDistribution('uniform', [15.0, 20.0]))
    print p1.get('Tau_m')
    
    current_source = nest.StepCurrentSource([50.0, 110.0, 150.0, 210.0],
                                            [0.01, 0.02, -0.02, 0.01])
    p1.inject(current_source)
    
    p2 = nest.Population(1, nest.native_cell_type("poisson_generator"), {'rate': 200.0})
    
    print "Setting up recording"
    p2.record()
    p1._record('V_m')
    
    connector = nest.AllToAllConnector(weights=0.001)
    
    prj_ampa = nest.Projection(p2, p1, connector, target='AMPA')
    
    tstop = 250.0
    nest.run(tstop)
    
    n_points = int(tstop/nest.get_time_step()) + 1
    assert_equal(p1.recorders['V_m'].get().shape, (n_points*n_cells, 3))
    id, t, v = p1.recorders['V_m'].get().T
    assert v.max() > 0.0 # should have some spikes

 
def test_native_stdp_model():
    nest = pyNN.nest
    from pyNN.utility import init_logging

    init_logging(logfile=None, debug=True)
    
    nest.setup()
    
    p1 = nest.Population(10, nest.IF_cond_exp)
    p2 = nest.Population(10, nest.SpikeSourcePoisson)
    
    stdp_params = {'Wmax': 50.0, 'lambda': 0.015}
    stdp = nest.NativeSynapseDynamics("stdp_synapse", stdp_params)
    
    connector = nest.AllToAllConnector(weights=0.001)
    
    prj = nest.Projection(p2, p1, connector, target='excitatory',
                          synapse_dynamics=stdp)

def test_ticket244():
    nest = pyNN.nest
    nest.setup(threads=4)
    p1 = nest.Population(4, nest.IF_curr_exp)
    p1.record()
    poisson_generator = nest.Population(3, nest.SpikeSourcePoisson, {'rate': 1000.})
    conn = nest.OneToOneConnector(weights=1.)
    nest.Projection(poisson_generator, p1.sample(3), conn, target="excitatory")
    nest.run(15)
    p1.getSpikes()

def test_ticket236():
    """Calling get_spike_counts() in the middle of a run should not stop spike recording"""
    pynnn = pyNN.nest
    pynnn.setup()
    p1 = pynnn.Population(2, pynnn.IF_curr_alpha, structure=pynnn.space.Grid2D())
    p1.record(to_file=False)
    src = pynnn.DCSource(amplitude=70)
    src.inject_into(p1[:])
    pynnn.run(50)
    s1 = p1.get_spike_counts() # as expected, {1: 124, 2: 124}
    pynnn.run(50)
    s2 = p1.get_spike_counts() # unexpectedly, still {1: 124, 2: 124}
    assert s1[p1[0]] < s2[p1[0]]
