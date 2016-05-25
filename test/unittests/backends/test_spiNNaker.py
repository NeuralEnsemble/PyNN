# --------------------------------------------------
# get common imports from __init__.py
# --------------------------------------------------

from .glob_import import *
   
# --------------------------------------------------
# CHANGE the name below for a new simulator backend
# --------------------------------------------------

sim_name = "spiNNaker"
 
# --------------------------------------------------
# CHANGE below only for a new hardware simulator backend
# -------------------------------------------------- 
 
try:
    import pyNN.spiNNaker as sim
    have_sim = True
except ImportError:
    have_sim = False


def setUp():
    pass
    

def tearDown():
    pass

extra = {}

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
    import unittest
    unittest.main()
