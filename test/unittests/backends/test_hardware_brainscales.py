# --------------------------------------------------
# get common imports from __init__.py
# --------------------------------------------------

from .glob_import import *
   
# --------------------------------------------------
# CHANGE the name below for a new simulator backend
# --------------------------------------------------

sim_name = "hardware.brainscales"
 
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
    if have_sim:
        for TestClass in registry:
            m = modules[TestClass.__module__]
            alias_cell_types(m, 
                            IF_cond_exp = sim.Hardware_IF_cond_exp,
                            EIF_cond_exp_isfa_ista = sim.Hardware_EIF_cond_exp_isfa_ista,
                            SpikeSourceArray = sim.SpikeSourceArray,
                            SpikeSourcePoisson = sim.SpikeSourcePoisson,
                            EIF_cond_alpha_isfa_ista = sim.Hardware_EIF_cond_exp_isfa_ista,
                            IF_curr_alpha = sim.Hardware_IF_cond_exp,
                            IF_curr_exp = sim.Hardware_IF_cond_exp,
                            IF_cond_alpha = sim.Hardware_IF_cond_exp,
                            IF_cond_exp_gsfa_grr = sim.Hardware_IF_cond_exp,
                            HH_cond_exp = sim.Hardware_IF_cond_exp
                            )
    
def tearDown():
    pass

extra = {'loglevel':0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}

# --------------------------------------------------
# DON'T CHANGE below this line
# -------------------------------------------------- 
    
def test_scenarios(sim_name=sim_name, have_sim=have_sim):
    for TestClass in registry:
        test_class = TestClass()
        for scenario in test_class.registry:
            if is_included(sim_name=sim_name, scenario=scenario):
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