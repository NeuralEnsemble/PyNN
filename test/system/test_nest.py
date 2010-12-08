from nose.plugins.skip import SkipTest
from scenarios import scenarios

try:
    import pyNN.nest
    have_nest = True
except ImportError:
    have_nest = False

def test_all():
    for scenario in scenarios:
        scenario.description = scenario.__name__
        if have_nest:
            yield scenario, pyNN.nest
        else:
            raise SkipTest