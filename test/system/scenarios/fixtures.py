import functools
import pytest

available_modules = {}

try:
    import pyNN.neuron
    available_modules["neuron"] = pyNN.neuron
except ImportError:
    pass

try:
    import pyNN.nest
    available_modules["nest"] = pyNN.nest
except ImportError:
    pass

try:
    import pyNN.brian2
    available_modules["brian2"] = pyNN.brian2
except ImportError:
    pass


class SimulatorNotAvailable:

    def __init__(self, sim_name):
        self.sim_name = sim_name

    def setup(self, *args, **kwargs):
        pytest.skip(f"{self.sim_name} not available")


def get_simulator(sim_name):
    if sim_name in available_modules:
        return pytest.param(available_modules[sim_name], id=sim_name)
    else:
        return pytest.param(SimulatorNotAvailable(sim_name), id=sim_name)


def run_with_simulators(*sim_names):
    sim_modules = (get_simulator(sim_name) for sim_name in sim_names)

    return pytest.mark.parametrize("sim", sim_modules)
