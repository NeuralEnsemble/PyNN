# encoding: utf-8

from .. import test_population
try:
    import unittest2 as unittest
except ImportError:
    import unittest
    
from registry import registry
from sys import modules 

def is_included(sim_name, scenario):
    included = False
    if scenario.include_only and sim_name == scenario.include_only:
        included = True
    elif sim_name not in scenario.exclude:
        included = True
    
    return included