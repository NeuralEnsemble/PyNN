from nose.plugins.skip import SkipTest
from scenarios.registry import registry
from nose.tools import assert_equal, assert_not_equal
from pyNN.utility import init_logging, assert_arrays_equal
import numpy
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

try:
    import pyNN.hardware.brainscales
    have_hardware_brainscales = True
except ImportError:
    have_hardware_brainscales = False

def test_scenarios():
    for scenario in registry:
        if (scenario.include_only and "hardware.brainscales" in scenario.include_only):
            #if "hardware.brainscales" not in scenario.exclude:
            scenario.description = scenario.__name__
            print scenario.description
            if have_hardware_brainscales:
                yield scenario, pyNN.hardware.brainscales
            else:
                raise SkipTest

def test_restart_loop():
    sim = pyNN.hardware.brainscales
    extra = {'loglevel':0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}
    sim.setup(**extra)
    sim.end()
    sim.setup(**extra)
    sim.end()

def test_sim_without_clearing():
    sim = pyNN.hardware.brainscales
    extra = {'loglevel':0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}
    sim.setup(**extra)    
  
if __name__ == '__main__':
    data = test_scenarios()
    test_restart_loop()
    test_sim_without_clearing()
