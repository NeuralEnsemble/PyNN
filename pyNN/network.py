"""


"""

import sys
import inspect
from itertools import chain
from neo.io import get_io
from .common import Population, PopulationView, Projection, Assembly


class Network(object):
    """
    docstring
    """

    def __init__(self, *components):
        self._populations = set([])
        self._views = set([])
        self._assemblies = set([])
        self._projections = set([])
        self.add(*components)

    @property
    def populations(self):
        return frozenset(self._populations)

    @property
    def views(self):
        return frozenset(self._views)

    @property
    def assemblies(self):
        return frozenset(self._assemblies)

    @property
    def projections(self):
        return frozenset(self._projections)

    @property
    def sim(self):
        """Figure out which PyNN backend module this Network is using."""
        # we assume there is only one. Could be mixed if using multiple simulators
        # at once.
        populations_module = inspect.getmodule(list(self.populations)[0].__class__)
        return sys.modules[".".join(populations_module.__name__.split(".")[:-1])]

    def count_neurons(self):
        return sum(population.size for population in chain(self.populations))

    def count_connections(self):
        return sum(projection.size() for projection in chain(self.projections))

    def add(self, *components):
        for component in components:
            if isinstance(component, Population):
                self._populations.add(component)
            elif isinstance(component, PopulationView):
                self._views.add(component)
                self._populations.add(component.parent)
            elif isinstance(component, Assembly):
                self._assemblies.add(component)
                self._populations.update(component.populations)
            elif isinstance(component, Projection):
                self._projections.add(component)
                # todo: check that pre and post populations/views/assemblies have been added
            else:
                raise TypeError()

    def get_component(self, label):
        for obj in chain(self.populations, self.views, self.assemblies, self.projections):
            if obj.label == label:
                return obj
        return None

    def filter(self, cell_types=None):
        """Return an Assembly of all components that have a cell type in the list"""
        if cell_types is None:
            raise NotImplementedError()
        else:
            if cell_types == "all":
                return self.sim.Assembly(*(pop for pop in self.populations
                                           if pop.celltype.injectable))
            else:
                return self.sim.Assembly(*(pop for pop in self.populations
                                           if pop.celltype.__class__ in cell_types))

    def record(self, variables, to_file=None, sampling_interval=None, include_spike_source=True):
        for obj in chain(self.populations, self.assemblies):
            if include_spike_source or obj.injectable:  # spike sources are not injectable
                obj.record(variables, to_file=to_file, sampling_interval=sampling_interval)

    def get_data(self, variables='all', gather=True, clear=False, annotations=None):
        return [assembly.get_data(variables, gather, clear, annotations)
                for assembly in self.assemblies]

    def write_data(self, io, variables='all', gather=True, clear=False, annotations=None):
        if isinstance(io, str):
            io = get_io(io)
        data = self.get_data(variables, gather, clear, annotations)
        # if self._simulator.state.mpi_rank == 0 or gather is False:
        if True:  # tmp. Need to handle MPI
            io.write(data)
