# encoding: utf-8
"""
NOT an implementation of PyNN API
"""

import arbor

class model_recipe (arbor.recipe):

    def __init__(self, n=1, population):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)
        self.ncells = n
        self.params = arbor.cell_parameters()

    # The num_cells method that returns the total number of cells in the model
    # must be implemented.
    def num_cells(self):
        return self.ncells

    # The cell_description method returns a cell.
    def cell_description(self, gid):
        # Cell should have a synapse labeled "syn"
        # and a detector labeled "detector"
        return make_cable_cell(gid, self.params)

    # The kind method returns the type of cell with gid.
    # Note: this must agree with the type returned by cell_description.
    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    # Make a ring network
    def connections_on(self, gid):
        src = (gid-1)%self.ncells
        w = 0.01
        d = 10
        return [arbor.connection((src,"detector"), "syn", w, d)]

    # Attach a generator to the first cell in the ring.
    def event_generators(self, gid):
        if gid==0:
            sched = arbor.explicit_schedule([1])
            return [arbor.event_generator("syn", 0.1, sched)]
        return []

    def get_probes(self, id):
        # Probe just the membrane voltage at a location on the soma.
        return [arbor.cable_probe_membrane_voltage('(location 0 0)')]