# encoding: utf-8

from .. import test_simulation_control
from .. import test_population
from .. import test_populationview
from .. import test_assembly
from .. import test_connectors_parallel
from .. import test_connectors_serial
from .. import test_projection

exclude_modules = ['test_connectors_parallel']

try:
    import unittest2 as unittest
except ImportError:
    import unittest
    
from nose.plugins.skip import SkipTest
from registry import registry
from sys import modules 


def is_included(sim_name, scenario, module_name):
    """
    Checks if the simulator sim_name is included in the test called scenario
    """
    included = False
    short_module_name = module_name.split('.')[-1]
    if short_module_name in exclude_modules:
        included = False
    elif sim_name not in scenario.exclude:
        included = True
    return included


def skip():
    raise SkipTest
