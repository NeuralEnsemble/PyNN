from nose.plugins.skip import SkipTest
from scenarios.registry import registry
from nose.tools import assert_equal, assert_not_equal
from pyNN.utility import init_logging, assert_arrays_equal
import numpy
import logging
try:
    import unittest2 as unittest
except ImportError:
    import unittest

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

try:
    import pyNN.hardware.brainscales as sim
    have_hardware_brainscales = True
except ImportError:
    have_hardware_brainscales = False

class HardwareTest(unittest.TestCase):

    def setUp(self):
        extra = {'loglevel':0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}
        sim.setup(**extra)

    def test_IF_cond_exp_default_values(self):
        ifcell  = sim.IF_cond_exp()
        
    def test_IF_cond_exp_default_values2(self):
        ifcell  = sim.IF_cond_exp()
    
    #def test_scenarios():
        #sim=pyNN.hardware.brainscales
        #extra = {'loglevel':0, 'useSystemSim': True}
        #if sim.__name__ == "pyNN.hardware.brainscales":
            #extra['hardware'] = sim.hardwareSetup['small']
        #sim.setup(**extra)
        #for scenario in registry:
            #if (scenario.include_only and "hardware.brainscales" in scenario.include_only):
                ##if "hardware.brainscales" not in scenario.exclude:
                #scenario.description = scenario.__name__
                #print scenario.description
                #if have_hardware_brainscales:
                    #yield scenario, sim
                #else:
                    #raise SkipTest
        #sim.end()

    #def test_restart_loop():
        #sim = pyNN.hardware.brainscales
        #extra = {'loglevel':0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}
        #sim.setup(**extra)
        #sim.end()
        #sim.setup(**extra)
        #sim.end()
        #sim.setup(**extra)
        #sim.run(10.0)
        #sim.end()
        #sim.setup(**extra)
        #sim.run(10.0)
        #sim.end()
        
    #def test_several_runs():
        #sim = pyNN.hardware.brainscales
        #extra = {'loglevel':0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}
        #sim.setup(**extra)
        #sim.run(10.0)
        #sim.run(10.0)
        #sim.end()

    #def test_sim_without_clearing():
        #sim = pyNN.hardware.brainscales
        #extra = {'loglevel':0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}
        #sim.setup(**extra)    
        
    #def test_sim_without_setup():
        #sim = pyNN.hardware.brainscales
        #sim.end()   
  
if __name__ == '__main__':
    #test_scenarios()
    #test_restart_loop()
    #test_sim_without_clearing()
    #test_sim_without_setup()
    test_several_runs()
