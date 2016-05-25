# --------------------------------------------------
# get common imports from __init__.py
# --------------------------------------------------

from .glob_import import *
import unittest

# --------------------------------------------------
# CHANGE the name below for a new simulator backend
# --------------------------------------------------

sim_name = "hardware.brainscales"
 
# --------------------------------------------------
# CHANGE below only for a new hardware simulator backend
# -------------------------------------------------- 
 
try:
    import pyNN.hardware.brainscales as sim
    have_sim = True
except ImportError:
    have_sim = False


def setUp():
    pass


def tearDown():
    pass


extra = {
    'loglevel': 0, 
    'ignoreHWParameterRanges': True, 
    'useSystemSim': True, 
    'hardware': sim.hardwareSetup['one-hicann'],
    'allowStandardTypeSubstitution': True
    }


class PopulationViewTest(unittest.TestCase):

    def setUp(self):
        sim.setup(**extra)
        
    def tearDown(self):
        sim.end()
        
    def test_can_record_populationview(self):
        pv = sim.Population(17, sim.EIF_cond_exp_isfa_ista())[::2]
        assert pv.can_record('v')
        assert not pv.can_record('w')
        assert not pv.can_record('gsyn_inh')
        assert pv.can_record('spikes')
        assert not pv.can_record('foo')
        
    def test_can_record_population(self, sim=sim):
        p = sim.Population(17, sim.EIF_cond_exp_isfa_ista())
        assert p.can_record('v')
        assert not p.can_record('w')
        assert not p.can_record('gsyn_inh')
        assert p.can_record('spikes')
        assert not p.can_record('foo')
        
# --------------------------------------------------
# DON'T CHANGE below this line
# -------------------------------------------------- 
    

def test_scenarios(sim_name=sim_name, have_sim=have_sim):
    for TestClass in registry:
        module_name = TestClass.__module__
        test_class = TestClass()
        for scenario in test_class.registry:
            if is_included(sim_name=sim_name, scenario=scenario, module_name=module_name):
                scenario.description = scenario.__name__
                if have_sim:
                    test_class.setUp(sim, **extra)
                    yield scenario, test_class, sim
                    test_class.tearDown(sim)
                else:
                    yield skip
            else:
                yield skip
                
if __name__ == "__main__":
    unittest.main()
