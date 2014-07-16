from .. import test_population
from registry import registry
from sys import modules

try:
    import unittest2 as unittest
except ImportError:
    import unittest
    
try:
    import pyNN.mock
    sim = pyNN.mock
    have_mock = True
except ImportError:
    have_mock = False

def setUp():
    for t_c in registry:
        m = modules[t_c.__module__]
        m.alias_cell_types(
            alias_IF_cond_exp=sim.IF_cond_exp,
            alias_IF_cond_alpha=sim.IF_cond_alpha,
            alias_HH_cond_exp=sim.HH_cond_exp,
            alias_IF_curr_exp=sim.IF_curr_exp,
            alias_IF_curr_alpha=sim.IF_curr_alpha,
            alias_EIF_cond_exp_isfa_ista=sim.EIF_cond_exp_isfa_ista,
            alias_SpikeSourceArray=sim.SpikeSourceArray,
            alias_SpikeSourcePoisson=sim.SpikeSourcePoisson
            )
    
def tearDown():
    assert True
    
def func_setup():
    sim.setup()
    
def func_teardown():
    sim.end()
    
def test_scenarios():
    print registry[0].__module__
    for t_c in registry:
        c = t_c()
        for scenario in c.registry:
            if "mock" not in scenario.exclude:
                scenario.description = scenario.__name__
                if have_mock:
                    func_setup()
                    yield scenario, c, sim
                    func_teardown()
                else:
                    raise SkipTest
                
if __name__ == "__main__":
    unittest.main()