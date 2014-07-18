# encoding: utf-8

from ..alias_cell_types import alias_cell_types, take_all_cell_classes
from .. import test_simulation_control
from .. import test_population
from .. import test_populationview
from .. import test_assembly
try:
    import unittest2 as unittest
except ImportError:
    import unittest
    
from registry import registry
from sys import modules 

def is_included(sim_name, scenario):
    """
    Checks if the simulator sim_name is included in the test called scenario
    """
    included = False
    if scenario.include_only and sim_name == scenario.include_only:
        included = True
    elif sim_name not in scenario.exclude:
        included = True
    return included