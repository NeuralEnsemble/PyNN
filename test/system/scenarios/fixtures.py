
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


def get_simulator(sim_name):
    if sim_name in available_modules:
        return available_modules[sim_name]
    else:
        pytest.skip(f"{sim_name} not available")
