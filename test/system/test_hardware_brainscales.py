from nose.plugins.skip import SkipTest
from scenarios.registry import registry
from nose.tools import assert_equal, assert_not_equal
from pyNN.utility import init_logging, assert_arrays_equal
import numpy

try:
    import pyNN.hardware.brainscales
    have_hardware_brainscales = True
except ImportError:
    have_hardware_brainscales = False

def test_scenarios():
    for scenario in registry:
        if "hardware_brainscales" not in scenario.exclude:
            scenario.description = scenario.__name__
            if have_hardware_brainscales:
                yield scenario, pyNN.hardware.brainscales
            else:
                raise SkipTest

if __name__ == '__main__':
    data = test_scenarios()
