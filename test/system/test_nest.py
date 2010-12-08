from nose.plugins.skip import SkipTest
from scenarios import *

try:
    import pyNN.nest
    have_nest = True
except ImportError:
    have_nest = False

def test_all():
    for scenario in (scenario1, scenario2):
        if have_nest:
            yield scenario, pyNN.nest
        else:
            raise SkipTest