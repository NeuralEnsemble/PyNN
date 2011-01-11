from nose.plugins.skip import SkipTest
from scenarios import scenarios

try:
    import pyNN.brian
    have_brian = True
except ImportError:
    have_brian = False

def test_scenarios():
    for scenario in scenarios:
        if "brian" not in scenario.exclude:
            scenario.description = scenario.__name__
            if have_brian:
                yield scenario, pyNN.brian
            else:
                raise SkipTest