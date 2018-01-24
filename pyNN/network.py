"""


"""

from itertools import chain
from pyNN.common import Population, PopulationView, Projection, Assembly


class Network(object):
    """
    docstring
    """

    def __init__(self, *components):
        self.populations = set([])
        self.views = set([])
        self.assemblies = set([])
        self.projections = set([])
        for component in components:
            if isinstance(component, Population):
                self.populations.add(component)
            elif isinstance(component, PopulationView):
                self.views.add(component)
                self.populations.add(component.parent)
            elif isinstance(component, Assembly):
                self.assemblies.add(component)
                self.populations.update(component.populations)
            elif isinstance(component, Projection):
                self.projections.add(component)
                # todo: check that pre and post populations/views/assemblies have been added
            else:
                raise TypeError()

    def count_neurons(self):
        return sum(population.size for population in chain(self.populations))
    
    def count_connections(self):
        return sum(projection.size() for projection in chain(self.projections))