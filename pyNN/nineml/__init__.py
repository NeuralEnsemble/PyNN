"""
Export of PyNN scripts as NineML.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""
import logging
import nineml.user as ul
from neo.io import get_io

from .. import common, random, standardmodels as std
from . import simulator
from .standardmodels import *
from .populations import Population, PopulationView, Assembly
from .projections import Projection
from .connectors import *

logger = logging.getLogger("PyNN")
random.get_mpi_config = lambda: (0, 1)


def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    return [obj.__name__ for obj in globals().values() if isinstance(obj, type) and issubclass(obj, std.StandardCellType)]


def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, **extra_params):
    common.setup(timestep, min_delay, max_delay, **extra_params)
    simulator.state.clear()
    simulator.state.dt = timestep  # move to common.setup?
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    simulator.state.mpi_rank = extra_params.get('rank', 0)
    simulator.state.num_processes = extra_params.get('num_processes', 1)
    simulator.state.output_filename = extra_params.get("filename", "PyNN29ML.xml")
    simulator.state.net = Network(label=extra_params.get("label", "PyNN29ML"))
    return rank()


def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for (population, variables, filename) in simulator.state.write_on_end:
        io = get_io(filename)
        population.write_data(io, variables)
    simulator.state.write_on_end = []
    # should have common implementation of end()
    simulator.state.net.to_nineml().write(simulator.state.output_filename)


run = common.build_run(simulator)

reset = common.build_reset(simulator)

initialize = common.initialize

get_current_time, get_time_step, get_min_delay, get_max_delay, \
                    num_processes, rank = common.build_state_queries(simulator)

create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector, StaticSynapse)

set = common.set

record = common.build_record(simulator)

record_v = lambda source, filename: record(['v'], source, filename)

record_gsyn = lambda source, filename: record(['gsyn_exc', 'gsyn_inh'], source, filename)


class Network(object):  # move to .simulator ?

    def __init__(self, label):
        self.label = label
        self.populations = []
        self.projections = []
        self.current_sources = []
        self.assemblies = []

    def to_xml(self):
        return self.to_nineml().to_xml()

    def to_nineml(self):
        model = ul.Model(name=self.label)
        for cs in self.current_sources:
            model.add_component(cs.to_nineml())
            # needToDefineWhichCellsTheCurrentIsInjectedInto
            # doWeJustReuseThePopulationProjectionIdiom="?"
        main_group = ul.Network(name="Network")
        _populations = self.populations[:]
        _projections = self.projections[:]
        for a in self.assemblies:
            group = a.to_nineml()
            for p in a.populations:
                _populations.remove(p)
                group.add(p.to_nineml())
            for prj in self.projections:
                if (prj.pre is a or prj.pre in a.populations) and \
                   (prj.post is a or prj.post in a.populations):
                    _projections.remove(prj)
                    group.add(prj.to_nineml())
            model.add_group(group)
        for p in _populations:
            main_group.add(p.to_nineml())
        for prj in _projections:
            main_group.add(prj.to_nineml())
        model.add_group(main_group)
        model.check()
        return model
