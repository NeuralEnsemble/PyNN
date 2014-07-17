# --------------------------------------------------
# get common imports from __init__.py
# --------------------------------------------------

from .glob_import import *
   
# --------------------------------------------------
# CHANGE the name below for a new simulator backend
# --------------------------------------------------

sim_name = "nest"
 
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
    for c in registry:
        m = modules[c.__module__]
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

# --------------------------------------------------
# DON'T CHANGE below this line
# --------------------------------------------------

def test_scenarios(sim_name=sim_name, have_sim=have_sim):
    print registry[0].__module__
    for t_c in registry:
        c = t_c()
        for scenario in c.registry:
            if sim_name not in scenario.exclude:
                scenario.description = scenario.__name__
                if have_sim:
                    func_setup()
                    yield scenario, c, sim
                    func_teardown()
                else:
                    raise SkipTest

if __name__ == "__main__":
    unittest.main()