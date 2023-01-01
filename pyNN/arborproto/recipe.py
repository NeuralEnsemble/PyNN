import arbor


class ring_recipe(arbor.recipe):
    def __init__(self, ncells, cell_in_pop):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)
        self.ncells = ncells
        self.celltemplate = cell_in_pop
        self.props = arbor.neuron_cable_properties()

    # (6) The num_cells method that returns the total number of cells in the model
    # must be implemented.
    def num_cells(self):
        return self.ncells

    # (7) The cell_description method returns a cell
    # def cell_description(self, gid):
    #     return make_cable_cell(gid)
    def cell_description(self, gid):
        return arbor.cable_cell(self.celltemplate._cell._arbor_morphology,
                                self.celltemplate._cell._arbor_labels,
                                self.celltemplate._cell._decor)

    # The kind method returns the type of cell with gid.
    # Note: this must agree with the type returned by cell_description.
    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    # (8) Make a ring network. For each gid, provide a list of incoming connections.
    def connections_on(self, gid):
        src = (gid - 1) % self.ncells
        w = 0.01  # 0.01 μS on expsyn
        d = 5  # ms delay
        return [arbor.connection((src, "detector"), "syn", w, d)]

    # (9) Attach a generator to the first cell in the ring.
    def event_generators(self, gid):
        if gid == 0:
            sched = arbor.explicit_schedule([1])  # one event at 1 ms
            weight = 0.1  # 0.1 μS on expsyn
            return [arbor.event_generator("syn", weight, sched)]
        return []

    # (10) Place a probe at the root of each cell.
    def probes(self, gid):
        return [arbor.cable_probe_membrane_voltage('"root"')]

    def global_properties(self, kind):
        return self.props


class network(arbor.recipe):
    def __init__(self, ncells, cell_in_pop):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)
        self.ncells = ncells
        self.celltemplate = cell_in_pop
        self.props = arbor.neuron_cable_properties()

    # (6) The num_cells method that returns the total number of cells in the model
    # must be implemented.
    def num_cells(self):
        return self.ncells

    # (7) The cell_description method returns a cell
    # def cell_description(self, gid):
    #     return make_cable_cell(gid)
    def cell_description(self):
        return arbor.cable_cell(self.celltemplate._cell._arbor_morphology,
                                self.celltemplate._cell._arbor_labels,
                                self.celltemplate._cell._decor)

    # The kind method returns the type of cell with gid.
    # Note: this must agree with the type returned by cell_description.
    def cell_kind(self, gid):
        return arbor.cell_kind.cable
