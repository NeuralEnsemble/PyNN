# encoding: utf8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API, for the Arbor simulator.

Classes and attributes useable by the common implementation:

Classes:
    ID
    Connection

Attributes:
    state -- a singleton instance of the _State class.

All other functions and classes are private, and should not be used by other
modules.

:copyright: Copyright 2006-2020 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN import common
import arbor
from pyNN.arborproto.recipe import ring_recipe

name = "Arbor"


class ID(int, common.IDMixin):

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)

    def _build_cell(self, cell_model, cell_parameters):
        """
        Create a cell in Arbor, and register its global ID.

        `cell_model` -- one of the cell classes defined in the
                        `arbor.cells` module (more generally, any class that
                        implements a certain interface, but I haven't
                        explicitly described that yet).
        `cell_parameters` -- a ParameterSpace containing the parameters used to
                             initialise the cell model.
        """
        gid = int(self)
        self._cell = cell_model(**cell_parameters)  # create the cell object
        # state.register_gid(gid, self._cell.source, section=self._cell.source_section)
        # if hasattr(self._cell, "get_threshold"):            # this is not adequate, since the threshold may be changed after cell creation
        #     state.parallel_context.threshold(int(self), self._cell.get_threshold())  # the problem is that self._cell does not know its own gid


class State(common.control.BaseState):

    def __init__(self):
        common.control.BaseState.__init__(self)
        self.mpi_rank = 0
        self.num_processes = 1
        self.clear()
        self.dt = 0.1
        self.native_rng_baseseed = 0

    # def run(self, simtime):
    #     self.t += simtime
    #     self.running = True
    def run(self, simtime, populn, proj=None):
        self.model = self.__prerun(populn, proj)
        self.model.run(tfinal=simtime)
        self.t += simtime
        self.running = True

    def run_until(self, tstop):
        self.t = tstop
        self.running = True

    def clear(self):
        self.recorders = set([])
        self.gid_counter = 42
        self.segment_counter = -1
        self.reset()

    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.t = 0
        self.t_start = 0
        self.segment_counter += 1

    def __prerun(self, populn, proj):
        # NOTE: this hack is only for single cell population
        if populn.all_cells.size == 1:
            a_cell = arbor.cable_cell(populn.all_cells[0]._cell._arbor_morphology,
                                      populn.all_cells[0]._cell._arbor_labels,
                                      populn.all_cells[0]._cell._decor)
            model = arbor.single_cell_model(a_cell)
            model.probe("voltage", '"root"', frequency=10)  # 10 kHz sampling (i.e. every 0.1 ms)
        elif populn.all_cells.size > 1:
            model = arbor.simulation(proj.recipe)
            model.record(arbor.spike_recording.all)
            # sample probe
            probeset_id = arbor.cell_member(0, 0)
            self.probehandle = model.sample(probeset_id, arbor.regular_schedule(0.02))
        return model


state = State()
