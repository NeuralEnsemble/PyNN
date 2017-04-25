"""

Export of PyNN models to NeuroML 2

Contact Padraig Gleeson for more details

:copyright: Copyright 2006-2017 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
from pyNN import common
from pyNN.common.control import DEFAULT_MAX_DELAY, DEFAULT_TIMESTEP, DEFAULT_MIN_DELAY
from pyNN.connectors import *
from pyNN.recording import *
from pyNN.neuroml import simulator
from pyNN.neuroml.standardmodels import *
from pyNN.neuroml.standardmodels.synapses import *
from pyNN.neuroml.standardmodels.cells import *
from pyNN.neuroml.standardmodels.electrodes import *
from pyNN.neuroml.populations import Population, PopulationView, Assembly
from pyNN.neuroml.projections import Projection

from neo.io import get_io

import neuroml

logger = logging.getLogger("PyNN_NeuroML")

save_format = 'xml'


def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    return [obj.__name__ for obj in globals().values() if isinstance(obj, type) and issubclass(obj, StandardCellType)]


def setup(timestep=DEFAULT_TIMESTEP, min_delay=DEFAULT_MIN_DELAY,
          max_delay=DEFAULT_MAX_DELAY, **extra_params):
    """ Set up for saving cell models and network structure to NeuroML """
    common.setup(timestep, min_delay, max_delay, **extra_params)
    simulator.state.clear()
    simulator.state.dt = timestep  # move to common.setup?
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    simulator.state.mpi_rank = extra_params.get('rank', 0)
    simulator.state.num_processes = extra_params.get('num_processes', 1)

    logger.debug("Creating network in NeuroML document to store structure")
    nml_doc = simulator._get_nml_doc(extra_params.get('reference', "PyNN_NeuroML2_Export"),reset=True)
    global save_format
    save_format = extra_params.get('save_format', "xml")
    
    # Create network
    net = neuroml.Network(id="network")
    nml_doc.networks.append(net)
    
    lems_sim = simulator._get_lems_sim(reset=True)
    lems_sim.dt = '%s'%timestep

    return rank()


def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for (population, variables, filename) in simulator.state.write_on_end:
        io = get_io(filename)
        population.write_data(io, variables)
    simulator.state.write_on_end = []
    
    nml_doc = simulator._get_nml_doc()

    import neuroml.writers as writers
    if save_format == 'xml':
        nml_file = '%s.net.nml'%nml_doc.id
        writers.NeuroMLWriter.write(nml_doc, nml_file)
    elif save_format == 'hdf5':
        nml_file = '%s.net.nml.h5'%nml_doc.id
        writers.NeuroMLHdf5Writer.write(nml_doc, nml_file)
        
    logger.info("Written NeuroML 2 file out to: "+nml_file)
    
    lems_sim = simulator._get_lems_sim()
    lems_sim.include_neuroml2_file("PyNN.xml", include_included=False)
    lems_sim.include_neuroml2_file(nml_file)
    lems_file = lems_sim.save_to_file()
    logger.info("Written LEMS file (to simulate NeuroML file) to: "+lems_file)
    
    # should have common implementation of end()


run, run_until = common.build_run(simulator)
run_for = run

reset = common.build_reset(simulator)

initialize = common.initialize

get_current_time, get_time_step, get_min_delay, get_max_delay, \
                    num_processes, rank = common.build_state_queries(simulator)


create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector, StaticSynapse)

#set = common.set

record = common.build_record(simulator)

record_v = lambda source, filename: record(['v'], source, filename)

record_gsyn = lambda source, filename: record(['gsyn_exc', 'gsyn_inh'], source, filename)
