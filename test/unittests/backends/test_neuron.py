# --------------------------------------------------
# get common imports from __init__.py
# --------------------------------------------------

from .glob_import import *
   
# --------------------------------------------------
# CHANGE the name below for a new simulator backend
# --------------------------------------------------

sim_name = "neuron"
 
 # --------------------------------------------------
# CHANGE below only for a new hardware simulator backend
# -------------------------------------------------- 
 
try:
    exec("import pyNN.%s" % sim_name)
    exec("sim = pyNN.%s" % sim_name)
    have_sim = True
except ImportError:
    have_sim = False

def setUp():
    for TestClass in registry:
        m = modules[TestClass.__module__]
        alias_cell_types(m, **take_all_cell_classes(sim))
    
def tearDown():
    assert True
    
def func_setup():
    sim.setup()
    
def func_teardown():
    sim.end()

# --------------------------------------------------
# DON'T CHANGE below this line
# -------------------------------------------------- 
    
def test_scenarios(sim_name=sim_name, have_sim=have_sim, func_setup=func_setup, func_teardown=func_teardown):
    for TestClass in registry:
        test_class = TestClass()
        for scenario in test_class.registry:
            if is_included(sim_name=sim_name, scenario=scenario):
                scenario.description = scenario.__name__
                if have_sim:
                    func_setup()
                    yield scenario, test_class, sim
                    func_teardown()
                else:
                    raise SkipTest  