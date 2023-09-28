"""

"""

from collections import defaultdict
import logging
import os.path
import subprocess
import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    pass
import arbor
from .. import common
from ..core import find

logger = logging.getLogger("PyNN")
name = "Arbor"


def build_mechanisms():
    # run `arbor-build-catalogue <name> <path/to/nmodl>`
    mech_path = os.path.join(os.path.dirname(__file__), "nmodl")
    if not os.path.exists(os.path.join(mech_path, "PyNN-catalogue.so")):
        cat_builder = find("arbor-build-catalogue")
        if not cat_builder:
            raise Exception("Unable to find arbor-build-catalogue. Please ensure Arbor is correctly installed.")
        proc = subprocess.run([cat_builder, "PyNN", mech_path], cwd=mech_path)
        if proc.returncode != 0:
            err_msg = "\n  ".join(proc.stdout)
            raise Exception(f"Unable to compile Arbor mechanisms. Output was:\n  {err_msg}")
        else:
            logger.info("Successfully compiled Arbor mechanisms.")
        return mech_path


class Cell(int):  # (, common.IDMixin):

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        #int.__init__(n)
        #common.IDMixin.__init__(self)
        self.gid = n
        #self.morph = None  #morph
        #self.decor = None  # decor
        #self.labels = None  # labels
        self.local = True
        #self.decor.place('"root"', arbor.threshold_detector(-10), f"detector-{self.gid}")

    # def __lt__(self, other):
    #     return self.gid < other.gid

    # def __lte__(self, other):
    #     return self.gid <= other.gid

    # def __gt__(self, other):
    #     return self.gid > other.gid

    # def __gte__(self, other):
    #     return self.gid >= other.gid

    # def __eq__(self, other):
    #     return self.gid == other.gid

    # def __ne__(self, other):
    #     return self.gid != other.gid


class NetworkRecipe(arbor.recipe):
    """

    """

    def __init__(self):
        arbor.recipe.__init__(self)
        self._populations = []
        self._gid_lookup = []
        self._event_generators = {}
        self._projection_map = defaultdict(list)

    def add_population(self, population):
        # unclear if spike sources should be added
        self._populations.append(population)
        if self._gid_lookup:
            n_gids_current = self._gid_lookup[-1]
        else:
            n_gids_current = 0
        self._gid_lookup.append(n_gids_current + population.size)

    def add_projection(self, projection):
        if hasattr(projection.post, "parent"):
            postsynaptic_population = projection.post.parent
        else:
            postsynaptic_population = projection.post
        self._projection_map[id(postsynaptic_population)].append(projection)

    def _get_population_by_gid(self, gid):
        population_index = np.argmax(np.array(self._gid_lookup) > gid)
        return self._populations[population_index]

    def num_cells(self):
        """The number of cells in the model."""
        return sum(population.size for population in self._populations)

    def cell_kind(self, gid):
        """The cell kind of the cell with global identifier gid (return type: arbor.cell_kind)."""
        return self._get_population_by_gid(gid).arbor_cell_kind

    def cell_description(self, gid):
        """
        A high level description of the cell with global identifier gid, for example the morphology,
        synapses and ion channels required to build a multi-compartment neuron.
        """
        return self._get_population_by_gid(gid).arbor_cell_description(gid)

    def connections_on(self, gid):
        """
        Returns a list of all the incoming connections to gid.

        Each connection should have a valid synapse label connection.dest on the post-synaptic
        target gid, and a valid source label connection.source.label on the pre-synaptic
        source connection.source.gid. See connection.

        By default returns an empty list.
        """
        postsynaptic_population = self._get_population_by_gid(gid)
        projections = self._projection_map.get(id(postsynaptic_population), [])
        connections = sum((proj.arbor_connections(gid) for proj in projections), start=[])
        return connections

    def event_generators(self, gid):
        """
        A list of all the event_generators that are attached to gid.

        By default returns an empty list.
        """
        return self._event_generators.get(gid, [])

    def probes(self, gid):
        """
        Returns a list specifying the probesets describing probes on the cell gid.
        Each element in the list is an opaque object of type probe produced by
        cell kind-specific probeset functions.
        Each probeset in the list has a corresponding probeset id of type cell_member:
        an id (gid, i) refers to the probes described by the ith entry in the list
        returned by get_probes(gid).

        By default returns an empty list.
        """
        return self._get_population_by_gid(gid).recorder._get_arbor_probes(gid)

    def global_properties(self, kind):
        """
        The global properties of a model.

        This method needs to be implemented for arbor.cell_kind.cable,
        where the properties include ion concentrations and reversal potentials;
        initial membrane voltage; temperature; axial resistivity; membrane capacitance;
        cv_policy; and a pointer to the mechanism catalogue.

        By default returns an empty object.
        """
        if kind == arbor.cell_kind.cable:
            props = arbor.neuron_cable_properties()
            catalogue_path = os.path.join(
                os.path.dirname(__file__), "nmodl", "PyNN-catalogue.so"
            )
            props.catalogue = arbor.load_catalogue(catalogue_path)
            return props
        # Spike source cells have nothing to report.
        return None


class State(common.control.BaseState):

    def __init__(self):
        common.control.BaseState.__init__(self)
        self.num_threads = 1
        self.rng_seed = 42
        self.clear()
        self.dt = 0.1
        alloc = arbor.proc_allocation(threads=self.num_threads)
        config = arbor.config()
        if config["mpi4py"]:
            comm = arbor.mpi_comm(MPI.COMM_WORLD)
        else:
            comm = None
        self.arbor_context = arbor.context(alloc, comm)
        # unclear if we can create the recipe now, or if we have to
        # construct it only when we've assembled the whole network
        self.network = NetworkRecipe()
        self.arbor_sim = None  # for debugging

    @property
    def mpi_rank(self):
        return self.arbor_context.rank

    @property
    def num_processes(self):
        return self.arbor_context.ranks

    def run(self, simtime):
        recipe = self.network
        if not self.running:
            hints = {}
            decomp = arbor.partition_load_balance(recipe, self.arbor_context, hints)
            self.arbor_sim = arbor.simulation(recipe, self.arbor_context, decomp, self.rng_seed)
            self.arbor_sim.record(arbor.spike_recording.all)  # todo: for now record all, but should be controlled by population.record()
            for recorder in self.recorders:
                recorder._set_arbor_sim(self.arbor_sim)
        self.t += simtime
        self.arbor_sim.run(self.t, self.dt)
        self.running = True

    def run_until(self, tstop):
        return self.run(tstop - self.t)

    def clear(self):
        self.recorders = set([])
        self.id_counter = 0
        self.segment_counter = -1
        self.reset()

    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.t = 0
        self.t_start = 0
        self.segment_counter += 1


build_mechanisms()
state = State()
