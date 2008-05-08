"""Interfacing MOOSE to PyNN"""
import moose
import Neuron
from cells import *
from pyNN import common

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, debug=False, **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    common.setup(timestep, min_delay, max_delay, debug, **extra_params)
    ctx = moose.PyMooseBase.getContext()
    ctx.setClock(0, timestep, 0)
    return 0

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    moose.PyMooseBase.endSimulation()
 
def create(cellclass, param_dict=None, n=1):
    """
    Create n cells all of the same type.
    If n > 1, return a list of cell ids/references.
    If n==1, return just the single id.
    """
    assert n > 0, 'n must be a positive integer'
    cell_gids = []
    if isinstance(cellclass, type):
	for i in range(n):
		print cellclass, param_dict
		cell_type = cellclass(param_dict)
                cell = eval("Neuron."+cell_type.moose_name)(**cell_type.parameters)
		cell_gids.append(cell.id)
        cell_gids = [ID(gid) for gid in cell_gids]
    elif isinstance(cellclass, str):  # celltype is not a standard cell
	cellclass = eval(cellclass)
	for i in range(n):
		cell = cellclass(**param_dict)
		cell_gids.append(cell.id)
        cell_gids = [ID(gid) for gid in cell_gids]
    else:
        raise Exception("Invalid cell type")
    for id in cell_gids:
        id.cellclass = cellclass
    if n == 1:
        return cell_gids[0]
    else:
        return cell_gids

def record(source, filename):
	dataDir = Neutral("/data")
	probe = Table(source.name, dataDir)
	probe.step_mode = 3
	probe.connect("inputRequest", source, "Vm")
	
def run(simtime):
	"""Run the simulation for simtime"""
	moose.step(simtime)
