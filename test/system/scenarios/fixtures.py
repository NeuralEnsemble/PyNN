import importlib
import pytest

_loaded_modules = {}


class LazySimulator:

    def __init__(self, sim_name):
        self._sim_name = sim_name

    @property
    def __name__(self):
        return f"pyNN.{self._sim_name}"

    def __str__(self):
        return f"pyNN.{self._sim_name}"

    def __getattr__(self, name):
        module = _loaded_modules.get(self._sim_name)
        if module is None:
            try:
                module = importlib.import_module(f"pyNN.{self._sim_name}")
                _loaded_modules[self._sim_name] = module
            except ImportError:
                pytest.skip(f"{self._sim_name} not available")
        return getattr(module, name)


def run_with_simulators(*sim_names):
    params = [pytest.param(LazySimulator(name), id=name) for name in sim_names]
    return pytest.mark.parametrize("sim", params)
