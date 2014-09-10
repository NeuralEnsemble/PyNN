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
        extra = {
            'loglevel':0, 
            'ignoreHWParameterRanges': True, 
            'useSystemSim': True, 
            'hardware': sim.hardwareSetup['one-hicann']
            }
        sim.setup(**extra)

    def test_IF_cond_exp_default_values(self):
        ifcell  = sim.IF_cond_exp()
        
    def test_IF_cond_exp_default_values2(self):
        ifcell  = sim.IF_cond_exp()


def test_scenarios():
    extra = {'loglevel':0, 'useSystemSim': True}
    extra['hardware'] = sim.hardwareSetup['small']
    
    for scenario in registry:
        if "hardware.brainscales" not in scenario.exclude:
            if have_hardware_brainscales:
                sim.setup(**extra)
                yield scenario, sim
                sim.end() 
            else:
                raise SkipTest
            
def test_restart_loop():
    extra = {'loglevel':0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}
    sim.setup(**extra)
    sim.end()
    sim.setup(**extra)
    sim.end()
    sim.setup(**extra)
    sim.run(10.0)
    sim.end()
    sim.setup(**extra)
    sim.run(10.0)
    sim.end()
    
#def test_several_runs():
    #extra = {'loglevel':0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}
    #sim.setup(**extra)
    #sim.run(10.0)
    #sim.run(10.0)
    #sim.end()

def test_sim_without_clearing():
    extra = {'loglevel':0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}
    sim.setup(**extra)    
    
def test_sim_without_setup():
    sim.end()   
    
 
if __name__ == '__main__':
    #test_scenarios()
    #test_restart_loop()
    #test_sim_without_clearing()
    #test_sim_without_setup()
    #test_several_runs()
