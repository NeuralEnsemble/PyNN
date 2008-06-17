# encoding: utf-8
"""
nrnpython implementation of the PyNN API.

This is an attempt at a parallel-enabled implementation.                                                                
$Id:__init__.py 188 2008-01-29 10:03:59Z apdavison $
"""
__version__ = "$Rev$"

from neuron import hoc, Vector
h = hoc.HocObject()
from pyNN import __path__ as pyNN_path
from pyNN.random import *
from math import *
from pyNN import common
from pyNN.neuron.cells import *
from pyNN.neuron.connectors import *
from pyNN.neuron.synapses import *
import os.path
import types
import sys
import numpy
import logging
import platform
Set = set

gid           = 0
ncid          = 0
gidlist       = []
recorder_list = []
running       = False
initialised   = False
nrn_dll_loaded = []
quit_on_end   = True
RECORDING_VECTOR_NAMES = {'spikes': 'spiketimes',
                          'v': 'vtrace'}

# ==============================================================================
#   Utility classes and functions
# ==============================================================================

class ID(int, common.IDMixin):
    """
    Instead of storing ids as integers, we store them as ID objects,
    which allows a syntax like:
        p[3,4].tau_m = 20.0
    where p is a Population object. The question is, how big a memory/performance
    hit is it to replace integers with ID objects?
    """

    def __init__(self, n):
        int.__init__(n)
        common.IDMixin.__init__(self)
    
    def __getattr__(self, name):
        # Need to override the version from common due to the problem of not
        # being able to get a list of all the parameters in a native model
        if self.is_standard_cell():
            return self.get_parameters()[name]
        else:
            cell = self._hoc_cell()
            return self._get_hoc_parameter(cell, name)
    
    def _hoc_cell(self):
        """Returns the hoc object corresponding to the cell with this id."""
        assert self in gidlist, "Cell %d does not exist on this node" % self
        if self.parent:
            hoc_cell_list = getattr(h, self.parent.label)
            try:
                list_index = self.parent.gidlist.index(self)
                cell = hoc_cell_list.object(list_index)
            except RuntimeError:
                print "id:", self
                print "parent.gid_start:", self.parent.gid_start
                print "len(parent):", len(self.parent)
                print "hoc_cell_list.count():", hoc_cell_list.count()
                print "parent.gidlist.index(id):", self.parent.gidlist.index(self)
                print "id.hocname:", self.hocname
                raise
        else:
            cell_name = "cell%d" % int(self)
            cell = getattr(h, cell_name)
        return cell
    
    def _get_hoc_parameter(self, cell, name):
        try:
            val = getattr(cell, name)
        except HocError:
            val = getattr(cell.source, name)
        return val
    
    def get_native_parameters(self):
        # Construct the list of hoc parameter names to get
        if self.is_standard_cell():
            parameter_names = [D['translated_name'] for D in self.cellclass.translations.values()]
        else:
            parameter_names = [] # for native cells, don't have a way to get their list of parameters 
        # Obtain the hoc object whose parameters we are going to get
        cell = self._hoc_cell()
        # Get the values from hoc
        parameters = {}
        for name in parameter_names:
            val = self._get_hoc_parameter(cell, name)
            if isinstance(val, hoc.HocObject):
                val = [val.x[i] for i in range(int(val.size()))]
            parameters[name] = val
        return parameters

    def set_native_parameters(self, parameters):
        cell = self._hoc_cell()
        logging.debug("Setting %s in %s" % (parameters, cell))
        for name, val in parameters.items():
            if hasattr(val, '__len__'):
                setattr(cell, name, Vector(val).hoc_obj)
            else:
                setattr(cell, name, val)
            cell.param_update()
        
def list_standard_models():
    return [obj for obj in globals().values() if isinstance(obj, type) and issubclass(obj, common.StandardCellType)]

def load_mechanisms(path=pyNN_path[0]):
    global nrn_dll_loaded
    if path not in nrn_dll_loaded:
        arch_list = [platform.machine(), 'i686', 'x86_64', 'powerpc']
        # in case NEURON is assuming a different architecture to Python, we try multiple possibilities
        for arch in arch_list:
            lib_path = os.path.join(path, 'hoc', arch, '.libs', 'libnrnmech.so')
            if os.path.exists(lib_path):
                h.nrn_load_dll(lib_path)
                nrn_dll_loaded.append(path)
                return
        raise Exception("NEURON mechanisms not found in %s." % os.path.join(path, 'hoc'))

# ==============================================================================
#   Module-specific functions and classes (not part of the common API)
# ==============================================================================

class HocError(Exception): pass

def hoc_execute(hoc_commands, comment=None):
    assert isinstance(hoc_commands, list)
    if comment:
        logging.debug(comment)
    for cmd in hoc_commands:
        logging.debug(cmd)
        success = hoc.execute(cmd)
        if not success:
            raise HocError('Error produced by hoc command "%s"' % cmd)

def hoc_comment(comment):
    logging.debug(comment)

def _hoc_arglist(paramlist):
    """Convert a list of Python objects to a list of hoc commands which will
       generate equivalent hoc objects."""
    hoc_commands = []
    argstr = ""
    nvec = 0; nstr = 0; nvar = 0; ndict = 0; nmat = 0
    for item in paramlist:
        if type(item) == types.ListType:
            hoc_commands += ['objref argvec%d' % nvec,
                             'argvec%d = new Vector(%d)' % (nvec, len(item))]
            argstr += 'argvec%d, ' % nvec
            for i in xrange(len(item)):
                hoc_commands.append('argvec%d.x[%d] = %g' % (nvec, i, item[i])) # assume only numerical values
            nvec += 1
        elif type(item) == types.StringType:
            hoc_commands += ['strdef argstr%d' % nstr,
                             'argstr%d = "%s"' % (nstr, item)]
            argstr += 'argstr%d, ' % nstr
            nstr += 1
        elif type(item) == types.DictType:
            dict_init_list = []
            for k, v in item.items():
                if type(v) == types.StringType:
                    dict_init_list += ['"%s", "%s"' % (k, v)]
                elif type(v) == types.ListType:
                    hoc_commands += ['objref argvec%d' % nvec,
                                     'argvec%d = new Vector(%d)' % (nvec, len(v))]
                    dict_init_list += ['"%s", argvec%d' % (k, nvec)]
                    for i in xrange(len(v)):
                        hoc_commands.append('argvec%d.x[%d] = %g' % (nvec, i, v[i])) # assume only numerical values
                    nvec += 1
                else: # assume number
                    dict_init_list += ['"%s", %g' % (k, float(v))]
            hoc_commands += ['objref argdict%d' % ndict,
                             'argdict%d = new Dict(%s)' % (ndict,", ".join(dict_init_list))]
            argstr += 'argdict%d, ' % ndict
            ndict += 1
        elif isinstance(item, numpy.ndarray):
            ndim = len(item.shape)
            if ndim == 1:  # this has not been tested yet
                cmd, argstr1 = _hoc_arglist([list(item)]) # convert to a list and call the current function recursively
                hoc_commands += cmd
                argstr += argstr1
            elif ndim == 2:
                argstr += 'argmat%s,' % nmat
                hoc_commands += ['objref argmat%d' % nmat,
                                 'argmat%d = new Matrix(%d,%d)' % (nmat, item.shape[0], item.shape[1])]
                for i in xrange(item.shape[0]):
                    for j in xrange(item.shape[1]):
                        try:
                          hoc_commands += ['argmat%d.x[%d][%d] = %g' % (nmat, i, j, item[i, j])]
                        except TypeError:
                          raise common.InvalidParameterValueError
                nmat += 1
            else:
                raise common.InvalidDimensionsError, 'number of dimensions must be 1 or 2'
        elif item is None:
            pass
        else:
            hoc_commands += ['argvar%d = %f' % (nvar, item)]
            argstr += 'argvar%d, ' % nvar
            nvar += 1
    return hoc_commands, argstr.strip().strip(',')

def _translate_synapse_type(synapse_type, weight=None, extra_mechanism=None):
    """
    If synapse_type is given (not None), it is used to determine whether the
    synapse is excitatory or inhibitory.
    Otherwise, the synapse type is inferred from the sign of the weight.
    Much testing needed to check if this behaviour matches nest and pcsim.
    """
    if synapse_type:
        if synapse_type == 'excitatory':
            syn_objref = "esyn"
        elif synapse_type == 'inhibitory':
            syn_objref = "isyn"
        else:
            # More sophisticated treatment needed once we have more sophisticated synapse
            # models, e.g. NMDA...
            #raise common.InvalidParameterValueError, synapse_type, "valid types are 'excitatory' or 'inhibitory'"
            syn_objref = synapse_type
    else:
        if weight is None or weight >= 0.0:
            syn_objref = "esyn"
        else:
            syn_objref = "isyn"
    if extra_mechanism == 'tsodkys-markram':
            syn_objref += "_tm"
    return syn_objref

#def checkParams(param, val=None):
#    """Check parameters are of valid types, normalise the different ways of
#       specifying parameters and values by putting everything in a dict.
#       Called by set() and Population.set()."""
#    if isinstance(param, str):
#        if isinstance(val, float) or isinstance(val, int):
#            param_dict = {param:float(val)}
#        elif isinstance(val,(str, list)):
#            param_dict = {param:val}
#        else:
#            raise common.InvalidParameterValueError
#    elif isinstance(param, dict):
#        param_dict = param
#    else:
#        raise common.InvalidParameterValueError
#    return param_dict

class HocToPy:
    """Static class to simplify getting variables from hoc."""
    
    fmt_dict = {'int' : '%d', 'integer' : '%d', 'float' : '%f', 'double' : '%f',
                'string' : '\\"%s\\"', 'str' : '\\"%s\\"'}
    
    @staticmethod
    def get(name, return_type='float'):
        """Return a variable from hoc.
           name can be a hoc variable (int, float, string) or a function/method
           that returns such a variable.
        """
        # We execute some commands here to avoid too much outputs in the log file
        errorstr = '"raise HocError(\'caused by HocToPy.get(%s, return_type=\\"%s\\")\')"' % (name, return_type)
        hoc_commands = ['success = sprint(cmd,"HocToPy.hocvar = %s",%s)' % (HocToPy.fmt_dict[return_type], name),
                        'if (success) { nrnpython(cmd) } else { nrnpython(%s) }' % errorstr ]
        hoc_execute(hoc_commands)
        return HocToPy.hocvar
    
    @staticmethod
    def bool(condition):
        """Evaluate the condition in hoc and return True or False."""
        HocToPy.hocvar = None
        hoc.execute('if (%s) { nrnpython("HocToPy.hocvar = True") } \
                     else { nrnpython("HocToPy.hocvar = False") }' % condition)
        if HocToPy.hocvar is None:
            raise HocError("caused by HocToPy.bool('%s')" % condition)
        return HocToPy.hocvar


class Recorder(object):
    """Encapsulates data and functions related to recording model variables."""
    
    def __init__(self, variable, population=None, file=None):
        """
        `file` should be one of:
            a file-name,
            `None` (write to a temporary file)
            `False` (write to memory).
        """
        assert variable in RECORDING_VECTOR_NAMES
        self.variable = variable
        self.filename = file or None
        self.population = population # needed for writing header information
        self.recorded = Set([])        

    def record(self, ids):
        """Add the cells in `ids` to the set of recorded cells."""
        if self.population:
            ids = Set([id for id in ids if id in self.population.gidlist])
        else:
            ids = Set(ids) # what about non-local cells?
        new_ids = list( ids.difference(self.recorded) )
        self.recorded = self.recorded.union(ids)
        
        if self.variable == 'spikes':
            for cell in new_ids:
                cell._hoc_cell().record(1)
        elif self.variable == 'v':
            dt = get_time_step()
            for cell in new_ids:
                cell._hoc_cell().record_v(1, dt)    
        
    def get(self, gather=False):
        """Returns the recorded data."""
        # not implemented yet
        data = recording.readArray(filename, sepchar=None)
        data = recording.convert_compatible_output(data, self.population, variable)
        return data
    
    def write(self, file=None, gather=False, compatible_output=True):
        hoc_execute(['objref gathered_vec_list',
                     'gathered_vec_list =  new List()'])
        vector_operation = ''
        if self.variable == 'spikes':
            vector_operation = '.where("<=", tstop)'
        header = "# dt = %g\\n# n = %d\\n" % (get_time_step(), int(h.tstop/get_time_step()))
        if self.population is None:
            cell_template = "cell%d"
            post_label = "node%d: post cellX.%s" % (myid, self.variable)
            id_list = gidlist
            padding = 0
        else:
            cell_template = "%s.object(%s)" % (self.population.hoc_label, "%d")
            post_label = 'node%d: post_%s.%s' % (myid, self.population.hoc_label, self.variable)
            id_list = self.population.gidlist
            padding = self.population.gid_start
            
        def post_data():
            pack_template = 'tmp = pc.pack(%s.%s%s)' % (cell_template,
                                                        RECORDING_VECTOR_NAME[self.variable],
                                                        vector_operation)
            for cell in self.recorded:
                hoc_commands += ['tmp = pc.pack(%d)' % id_list.index(cell),
                                 pack_template % id_list.index(cell)]
            hoc_commands += ['tmp = pc.post("%s")' % post_label]
            hoc_execute(hoc_commands,"--- Population[%s].__print()__ --- [Post objects to master]" %self.label)
        def take_data():
            hoc_commands = ['tmp = pc.take(post_label)']
            for node in range(1, num_processes()):
                hoc_commands += ['gathered_vec_list.append(pc.upkscalar())',
                                 'gathered_vec_list.append(pc.upkvec())']
        def write_data():
            if self.population is None:
                header = "# first_id = %d\\n# last_id = %d\\n" % (min(self.recorded), max(self.recorded))
            else:
                header = "# %d" % self.population.dim[0]
                for dimension in list(self.population.dim)[1:]:
                    header = "%s\t%d" % (header, dimension)
                header += "\\n# first_id = %d\\n# last_id = %d\\n" % (self.population.gid_start, self.population.gid_start+self.population.size-1)
            
            if self.variable == 'v':
                header += "# dt = %g\\n# n = %d\\n" % (get_time_step(), int(h.tstop/get_time_step()))
                num_format = "%.6g"
            elif self.variable == 'spikes':
                header += "# dt = %g\\n"% get_time_step()
                num_format = "%.2f"
            filename = file or self.filename
            hoc_commands = ['objref fileobj',
                            'fileobj = new File()',
                            'tmp = fileobj.wopen("%s")' % filename,
                            'tmp = fileobj.printf("%s")' % header,
                            'i = 0']
            write_template = 'tmp = %s.%s%s.printf(fileobj, fmt)' % (cell_template,
                                                                     RECORDING_VECTOR_NAMES[self.variable],
                                                                     vector_operation)
            for cell in self.recorded:
                hoc_commands += ['fmt = "%s\\t%d\\n"' % (num_format, cell-padding),
                                 write_template % id_list.index(cell)]
            # writing gathered data is currently broken
            #hoc_commands += ['while i < gathered_vec_list.count()-2 { gathered_vec_list.o(i+1).printf(fileobj, "%s broken") ' % num_format]
            hoc_commands += ['tmp = fileobj.close()']
            hoc_execute(hoc_commands, "Recorder.write()")
            
        if gather:
            if myid != 0: # on slave nodes, post data
                post_data()
            else:
                take_data()
                write_data()
        else:
            self.filename += ".%d" % myid    
            write_data()
                
# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, debug=False,**extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    global nhost, myid, logger, initialised, quit_on_end, running
    load_mechanisms()
    if 'quit_on_end' in extra_params:
        quit_on_end = extra_params['quit_on_end']
    # Initialisation of the log module. To write in the logfile, simply enter
    # logging.critical(), logging.debug(), logging.info(), logging.warning() 
    if debug:
        logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='neuron.log',
                    filemode='w')
    else:
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='neuron.log',
                    filemode='w')
        
    logging.info("Initialization of NEURON (use setup(.., debug=True) to see a full logfile)")
    
    # All the objects that will be used frequently in the hoc code are declared in the setup
    
    if initialised:
        h.dt = timestep
        h.min_delay = min_delay
        running = False
    else:
        tmp = h.xopen(os.path.join(pyNN_path[0],'hoc','standardCells.hoc'))
        tmp = h.xopen(os.path.join(pyNN_path[0],'hoc','odict.hoc'))
        h('objref pc')
        h('pc = new ParallelContext()')
        h.dt = timestep
        h('tstop = 0')
        h('min_delay = %f' % min_delay)
        #'create dummy_section',
        #    'access dummy_section',
        h('objref netconlist, nil')
        h('netconlist = new List()') 
        h('strdef cmd')
        h('strdef fmt') 
        h('objref nc') 
        h('objref rng')
        h('objref cell')    
        #---Experimental--- Optimize the simulation time ? / Reduce inter-processors exchanges ?
        tmp = h.pc.spike_compress(1,0)
        if extra_params.has_key('use_cvode') and extra_params['use_cvode'] == True:
            h('objref cvode')
            h('cvode = new CVode()')
            h.cvode.active(1)
        
    nhost = int(h.pc.nhost())
    if nhost < 2:
        nhost = 1; myid = 0
    else:
        myid = int(h.pc.id())
    
    initialised = True
    return int(myid)

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    global logfile, myid #, vfilelist, spikefilelist
    
    for recorder in recorder_list:
        recorder.write(gather=False, compatible_output=compatible_output)
    h.pc.runworker()
    h.pc.done()
    if quit_on_end:
        h.quit()
        logging.info("Finishing up with NEURON.")
        sys.exit(0)

def run(simtime):
    """Run the simulation for simtime ms."""
    global running

    if not running:
        running = True
        h('local_minimum_delay = pc.set_maxstep(10)')
        h.finitialize()
        h.tstop = 0
        logging.debug("local_minimum_delay on host #%d = %g" % (myid, h.local_minimum_delay))
        if nhost > 1:
            assert h.local_minimum_delay >= get_min_delay(),\
                   "There are connections with delays (%g) shorter than the minimum delay (%g)" % (h.local_minimum_delay, get_min_delay())
    h.tstop = simtime
    h.pc.psolve(h.tstop)
    return get_current_time()

def get_current_time():
    """Return the current time in the simulation."""
    return h.t

def get_time_step():
    return h.dt
common.get_time_step = get_time_step

def get_min_delay():
    return h.min_delay
common.get_min_delay = get_min_delay

def num_processes():
    return int(h.pc.nhost())

def rank():
    """Return the MPI rank."""
    myid = int(h.pc.id())
    return myid

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

def create(cellclass, param_dict=None, n=1):
    """
    Create n cells all of the same type.
    If n > 1, return a list of cell ids/references.
    If n==1, return just the single id.
    """
    global gid, gidlist, nhost, myid
    
    assert n > 0, 'n must be a positive integer'
    if isinstance(cellclass, type):
        celltype = cellclass(param_dict)
        hoc_name = celltype.hoc_name
        hoc_commands, argstr = _hoc_arglist([celltype.parameters])
    elif isinstance(cellclass, str):
        hoc_name = cellclass
        hoc_commands, argstr = _hoc_arglist([param_dict])
    argstr = argstr.strip().strip(',')
 
    # round-robin partitioning
    newgidlist = [i+myid for i in range(gid, gid+n, nhost) if i < gid+n-myid]
    logging.debug("Creating cells %s on host %d" % (newgidlist, myid))
    for cell_id in newgidlist:
        hoc_commands += ['tmp = pc.set_gid2node(%d,%d)' % (cell_id, myid),
                         'objref cell%d' % cell_id,
                         'cell%d = new %s(%s)' % (cell_id, hoc_name, argstr),
                         'tmp = cell%d.connect2target(nil, nc)' % cell_id,
                         #'nc = new NetCon(cell%d.source, nil)' % cell_id,
                         'tmp = pc.cell(%d, nc)' % cell_id]
        #h('tmp = pc.set_gid2node(%d,%d)' % (cell_id, myid))
        #h('objref cell%d' % cell_id)
        #h('cell%d = new %s(%s)' % (cell_id, hoc_name, argstr))
        #h('tmp = cell%d.connect2target(nil, nc)' % cell_id)
        #h('tmp = pc.cell(%d, nc)' % cell_id)
    hoc_execute(hoc_commands, "--- create() ---")

    gidlist.extend(newgidlist)
    cell_list = [ID(i) for i in range(gid, gid+n)]
    for id in cell_list:
        id.cellclass = cellclass
    gid = gid+n
    if n == 1:
        cell_list = cell_list[0]
    return cell_list

def connect(source, target, weight=None, delay=None, synapse_type=None, p=1, rng=None):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise.
    Weights should be in nA or µS."""
    global ncid, gid, gidlist, myid
    if type(source) != types.ListType:
        source = [source]
    if type(target) != types.ListType:
        target = [target]
    if weight is None:  weight = 0.0
    if delay  is None:  delay = get_min_delay()
    syn_objref = _translate_synapse_type(synapse_type, weight)
    nc_start = ncid
    hoc_commands = []
    logging.debug("connecting %s to %s on host %d" % (source, target, myid))
    for tgt in target:
        if tgt > gid or tgt < 0 or not isinstance(tgt, int):
            raise common.ConnectionError, "Postsynaptic cell id %s does not exist." % str(tgt)
        if "cond" in tgt.cellclass.__name__:
            weight = abs(weight) # weights must be positive for conductance-based synapses
        else:
            if tgt in gidlist: # only create connections to cells that exist on this machine
                if p < 1:
                    if rng: # use the supplied RNG
                        rarr = self.rng.uniform(0,1, len(source))
                    else:   # use the default RNG
                        rarr = numpy.random.uniform(0,1, len(source))
                for j, src in enumerate(source):
                    if src > gid or src < 0 or not isinstance(src, int):
                        raise common.ConnectionError, "Presynaptic cell id %s does not exist." % str(src)
                    else:
                        if p >= 1.0 or rarr[j] < p: # might be more efficient to vectorise the latter comparison
                            hoc_commands += [#'nc = pc.gid_connect(%d, pc.gid2cell(%d).%s)' % (src, tgt, syn_objref),
                                             'nc = pc.gid_connect(%d, cell%d.%s)' % (src, tgt, syn_objref),
                                             'nc.delay = %g' % delay,
                                             'nc.weight = %g' % weight,
                                             'tmp = netconlist.append(nc)']
                            ncid += 1
            else:
                for j, src in enumerate(source):
                    if src > gid or src < 0 or not isinstance(src, int):
                        raise common.ConnectionError, "Presynaptic cell id %s does not exist." % str(src)
    hoc_execute(hoc_commands, "--- connect(%s,%s) ---" % (str(source), str(target)))
    return range(nc_start, ncid)

def set(cells, param, val=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value.
    cellclass must be supplied for doing translation of parameter names."""
    if val:
        param = {param:val}
    if not hasattr(cells, '__len__'):
        cells = [cells]
    # see comment in Population.set() below about the efficiency of the
    # following
    cells = [cell for cell in cells if cell in gidlist]
    for cell in cells:
        cell.set_parameters(**param)

def record(source, filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    if not hasattr(source, '__len__'):
        source = [source]
    recorder = Recorder('spikes', file=filename)
    recorder.record(source)
    recorder_list.append(recorder)

def record_v(source, filename):
    """
    Record membrane potential to a file. source can be an individual cell or
    a list of cells."""
    # would actually like to be able to record to an array and
    # choose later whether to write to a file.
    if not hasattr(source, '__len__'):
        source = [source]
    recorder = Recorder('spikes', file=filename)
    recorder.record(source)
    recorder_list.append(recorder)


# ==============================================================================
#   High-level API for creating, connecting and recording from populations of
#   neurons.
# ==============================================================================

class Population(common.Population):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    All cells have both an address (a tuple) and an id (an integer). If p is a
    Population object, the address and id can be inter-converted using :
    id = p[address]
    address = p.locate(id)
    """
    nPop = 0
    
    def __init__(self, dims, cellclass, cellparams=None, label=None):
        """
        dims should be a tuple containing the population dimensions, or a single
          integer, for a one-dimensional population.
          e.g., (10,10) will create a two-dimensional population of size 10x10.
        cellclass should either be a standardized cell class (a class inheriting
        from common.StandardCellType) or a string giving the name of the
        simulator-specific model that makes up the population.
        cellparams should be a dict which is passed to the neuron model
          constructor
        label is an optional name for the population.
        """
        global gid, myid, nhost, gidlist, fullgidlist
        
        common.Population.__init__(self, dims, cellclass, cellparams, label)

        # set the steps list, used by the __getitem__() method.
        self.steps = [1]*self.ndim
        for i in xrange(self.ndim-1):
            for j in range(i+1, self.ndim):
                self.steps[i] *= self.dim[j]

        if isinstance(cellclass, type):
            self.celltype = cellclass(cellparams)
            self.cellparams = self.celltype.parameters
            hoc_name = self.celltype.hoc_name
        elif isinstance(cellclass, str): # not a standard model
            hoc_name = cellclass
        
        if self.cellparams is not None:
            hoc_commands, argstr = _hoc_arglist([self.cellparams])
            argstr = argstr.strip().strip(',')
        else:
            hoc_commands = []
            argstr = ''
    
        if not self.label:
            self.label = 'population%d' % Population.nPop
        self.hoc_label = self.label.replace(" ","_")
        
        self.recorders = {}
        for variable in RECORDING_VECTOR_NAMES:
            self.recorders[variable] = Recorder(variable, population=self)        
        
        # Now the gid and cellclass are stored as instance of the ID class, which will allow a syntax like
        # p[i, j].set(param, val). But we have also to deal with positions : a population needs to know ALL the positions
        # of its cells, and not only those of the cells located on a particular node (i.e in self.gidlist). So
        # each population should store what we call a "fullgidlist" with the ID of all the cells in the populations 
        # (and therefore their positions)
        self.fullgidlist = numpy.array([ID(i) for i in range(gid, gid+self.size) if i < gid+self.size], ID)
        self.cell = self.fullgidlist
        
        # self.gidlist is now derived from self.fullgidlist since it contains only the cells of the population located on
        # the node
        self.gidlist     = [self.fullgidlist[i+myid] for i in range(0, len(self.fullgidlist), nhost) if i < len(self.fullgidlist)-myid]
        self.gid_start   = gid

        # Write hoc commands
        hoc_commands += ['objref %s' % self.hoc_label,
                         '%s = new List()' % self.hoc_label]

        for cell_id in self.gidlist:
            hoc_commands += ['tmp = pc.set_gid2node(%d,%d)' % (cell_id, myid),
                             'cell = new %s(%s)' % (hoc_name, argstr),
                             #'nc = new NetCon(cell.source, nil)',
                             'tmp = cell.connect2target(nil, nc)',
                             'tmp = pc.cell(%d, nc)' % cell_id,
                             'tmp = %s.append(cell)' %(self.hoc_label)]
        hoc_execute(hoc_commands, "--- Population[%s].__init__() ---" %self.label)
        Population.nPop += 1
        gid = gid+self.size

        # We add the gidlist of the population to the global gidlist
        gidlist += self.gidlist
        
        # By default, the positions of the cells are their coordinates, given by the locate()
        # method. Note that each node needs to know all the positions of all the cells 
        # in the population
        for cell_id in self.fullgidlist:
            cell_id.parent = self
            #cell_id.setPosition(self.locate(cell_id))
                    
        # On the opposite, each node has to know only the precise hocname of its cells, if we
        # want to be able to use the low level set() function
        for cell_id in self.gidlist:
            cell_id.hocname = "%s.o(%d)" % (self.hoc_label, self.gidlist.index(cell_id))

    def __getitem__(self, addr):
        """Return a representation of the cell with coordinates given by addr,
           suitable for being passed to other methods that require a cell id.
           Note that __getitem__ is called when using [] access, e.g.
             p = Population(...)
             p[2,3] is equivalent to p.__getitem__((2,3)).
        """

        global gidlist

        # What we actually pass around are gids.
        if isinstance(addr, int):
            addr = (addr,)
        if len(addr) != len(self.dim):
            raise common.InvalidDimensionsError, "Population has %d dimensions. Address was %s" % (self.ndim, str(addr))
        index = 0
        for i, s in zip(addr, self.steps):
            index += i*s
        id = index + self.gid_start
        assert addr == self.locate(id), 'index=%s addr=%s id=%s locate(id)=%s' % (index, addr, id, self.locate(id))
        # We return the gid as an ID object. Note that each instance of Populations
        # distributed on several node can give the ID object, because fullgidlist is duplicated
        # and common to all the node (not the case of global gidlist, or self.gidlist)
        return self.fullgidlist[index]

    def __iter__(self):
        return self.__gid_gen()

    def __address_gen(self):
        """
        Generator to produce an iterator over all cells on this node,
        returning addresses.
        """
        for i in self.gidlist:
            yield self.locate(i)
    
    def __gid_gen(self):
        """
        Generator to produce an iterator over all cells on this node,
        returning gids.
        """
        for i in self.gidlist:
            yield i
        
    def addresses(self):
        return self.__address_gen()
    
    def ids(self):
        return self.__gid_gen()
        
    def locate(self, id):
        """Given an element id in a Population, return the coordinates.
               e.g. for  4 6  , element 2 has coordinates (1,0) and value 7
                         7 9
        """
        # id should be a gid
        assert isinstance(id, int), "id is %s, not int" % type(id)
        id -= self.gid_start
        if self.ndim == 3:
            rows = self.dim[1]; cols = self.dim[2]
            i = id/(rows*cols); remainder = id%(rows*cols)
            j = remainder/cols; k = remainder%cols
            coords = (i, j, k)
        elif self.ndim == 2:
            cols = self.dim[1]
            i = id/cols; j = id%cols
            coords = (i, j)
        elif self.ndim == 1:
            coords = (id,)
        else:
            raise common.InvalidDimensionsError
        return coords

    def index(self, n):
        """Return the nth cell in the population (Indexing starts at 0)."""
        if hasattr(n, '__len__'):
            n = numpy.array(n)
        return self.fullgidlist[n]

    def get(self, parameter_name, as_array=False):
        """
        Get the values of a parameter for every cell in the population.
        """
        # Arguably we should reshape to the shape of the Population
        values = [getattr(cell, parameter_name) for cell in self.gidlist]
        if as_array:
            values = numpy.array(values)
        return values

    def set(self, param, val=None):
        """
        Set one or more parameters for every cell in the population. param
        can be a dict, in which case val should not be supplied, or a string
        giving the parameter name, in which case val is the parameter value.
        val can be a numeric value, or list of such (e.g. for setting spike times).
        e.g. p.set("tau_m",20.0).
             p.set({'tau_m':20,'v_rest':-65})
        """

        if isinstance(param, str):
            if isinstance(val, (str, float, int)):
                param_dict = {param: float(val)}
            elif isinstance(val, (list, numpy.ndarray)):
                param_dict = {param: val}
            else:
                raise common.InvalidParameterValueError
        elif isinstance(param, dict):
            param_dict = param
        else:
            raise common.InvalidParameterValueError
        logging.debug("Setting %s in %s" % (param_dict, self.label))
        for cell in self.gidlist:
            cell.set_parameters(**param_dict)

    def tset(self, parametername, value_array):
        """
        'Topographic' set. Set the value of parametername to the values in
        value_array, which must have the same dimensions as the Population.
        """
        # Convert everything to 1D arrays
        if self.dim == value_array.shape: # the values are numbers or non-array objects
            values = value_array.flatten()
        elif len(value_array.shape) == len(self.dim)+1: # the values are themselves 1D arrays
            values = numpy.reshape(value_array, (self.dim, value_array.size/self.cell.size))
        else:
            raise common.InvalidDimensionsError, "Population: %s, value_array: %s" % (str(self.dim),
                                                                                      str(value_array.shape))
        values = values.take(numpy.array(self.gidlist)-self.gid_start) # take just the values for cells on this machine
        assert len(values) == len(self.gidlist)
        
        # Set the values for each cell
        for cell, val in zip(self.gidlist, values):
            if not isinstance(val, str) and hasattr(val, "__len__"):
                # tuples, arrays are all converted to lists, since this is
                # what SpikeSourceArray expects. This is not very robust
                # though - we might want to add things that do accept arrays.
                val = list(val)
            if cell in self.gidlist: # this is not necessary, surely?
                setattr(cell, parametername, val)

    def rset(self, parametername, rand_distr):
        """
        'Random' set. Set the value of parametername to a value taken from
        rand_distr, which should be a RandomDistribution object.
        """
        if isinstance(rand_distr.rng, NativeRNG):
            if isinstance(self.celltype, common.StandardCellType):
                parametername = self.celltype.__class__.translations[parametername]['translated_name']
                if parametername in self.celltype.__class__.computed_parameters():
                    raise Exception("rset() with NativeRNG not (yet) supported for computed parameters.")
            paramfmt = "%g,"*len(rand_distr.parameters); paramfmt = paramfmt.strip(',')
            distr_params = paramfmt % tuple(rand_distr.parameters)
            hoc_commands = ['rng = new Random(%d)' % 0 or distribution.rng.seed,
                            'tmp = rng.%s(%s)' % (rand_distr.name, distr_params)]
            # We do the loop in hoc, to speed up the code
            loop = "for tmp = 0, %d" %(len(self.gidlist)-1)
            cmd = '%s.object(tmp).%s = rng.repick()' % (self.hoc_label, parametername)
            hoc_commands += ['cmd="%s { %s success = %s.object(tmp).param_update()}"' %(loop, cmd, self.hoc_label),
                             'success = execute1(cmd)']
            hoc_execute(hoc_commands, "--- Population[%s].__rset()__ ---" %self.label)   
        else:
            rarr = rand_distr.next(n=self.size)
            hoc_comment("--- Population[%s].__rset()__ --- " %self.label)
            for cell,val in zip(self.gidlist, rarr):
                setattr(cell, parametername, val)

    def _call(self, methodname, arguments):
        """
        Calls the method methodname(arguments) for every cell in the population.
        e.g. p.call("set_background","0.1") if the cell class has a method
        set_background().
        """
        raise Exception("Method not yet implemented")
        ## Not sure this belongs in the API, because cell classes only have
        ## parameters/attributes, not methods.

    def _tcall(self, methodname, objarr):
        """
        `Topographic' call. Calls the method methodname() for every cell in the 
        population. The argument to the method depends on the coordinates of the
        cell. objarr is an array with the same dimensions as the Population.
        e.g. p.tcall("memb_init", vinitArray) calls
        p.cell[i][j].memb_init(vInitArray[i][j]) for all i, j.
        """
        raise Exception("Method not yet implemented")    

    def __record(self, record_what, record_from=None, rng=None):
        """
        Private method called by record() and record_v().
        """
        global myid
        fixed_list=False
        if isinstance(record_from, list): #record from the fixed list specified by user
            fixed_list=True
        elif record_from is None: # record from all cells:
            record_from = self.gidlist
        elif isinstance(record_from, int): # record from a number of cells, selected at random  
            # Each node will record N/nhost cells...
            nrec = int(record_from/nhost)
            if not rng:
                rng = numpy.random
            record_from = rng.permutation(self.gidlist)[0:nrec] # need ID objects, permutation returns integers
            # need ID objects, permutation returns integers
            record_from = [self.gidlist.index(i) for i in record_from]
        else:
            raise Exception("record_from must be either a list of cells or the number of cells to record from")
        # record_from is now a list or numpy array
        self.recorders[record_what].record(record_from)

    def record(self, record_from=None, rng=None):
        """
        If record_from is not given, record spikes from all cells in the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        hoc_comment("--- Population[%s].__record()__ ---" %self.label)
        self.__record('spikes', record_from, rng)

    def record_v(self, record_from=None, rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        hoc_comment("--- Population[%s].__record_v()__ ---" %self.label)
        self.__record('v', record_from, rng)

    def printSpikes(self, filename, gather=True, compatible_output=True):
        """
        Write spike times to file.
        
        If compatible_output is True, the format is "spiketime cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        This allows easy plotting of a `raster' plot of spiketimes, with one
        line for each cell.
        The timestep, first id, last id, and number of data points per cell are
        written in a header, indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        spike files.
        
        For parallel simulators, if gather is True, all data will be gathered
        to the master node and a single output file created there. Otherwise, a
        file will be written on each node, containing only the cells simulated
        on that node.
        """        
        hoc_comment("--- Population[%s].__printSpikes()__ ---" %self.label)
        self.recorders['spikes'].write(filename, gather, compatible_output)

    def print_v(self, filename, gather=True, compatible_output=True):
        """
        Write membrane potential traces to file.
        
        If compatible_output is True, the format is "v cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        The timestep, first id, last id, and number of data points per cell are
        written in a header, indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        voltage files.
        
        For parallel simulators, if gather is True, all data will be gathered
        to the master node and a single output file created there. Otherwise, a
        file will be written on each node, containing only the cells simulated
        on that node.
        """
        hoc_comment("--- Population[%s].__print_v()__ ---" %self.label)
        self.recorders['v'].write(filename, gather, compatible_output)

    def getSpikes(self, gather=True):
        """
        Return a 2-column numpy array containing cell ids and spike times for
        recorded cells.

        Useful for small populations, for example for single neuron Monte-Carlo.
        """
        # This is a bit of a hack implemetation
        tmpfile = "neuron_tmpfile" # should really use tempfile module
        self.recorders['spikes'].write(tmpfile, gather, compatible_output=False)
        if not gather:
            tmpfile += '%d' % myid
        if myid==0 or not gather:
            f = open(tmpfile, 'r')
            lines = [line for line in f.read().split('\n') if line and line[0]!='#'] # remove blank and comment lines
            line2spike = lambda s: (int(s[1]), float(s[0]))
            spikes = numpy.array([line2spike(line.split()) for line in lines])
            f.close()
            #os.remove(tmpfile)
            return spikes
        else:
            return numpy.empty((0,2))
        
    def meanSpikeCount(self, gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        global myid
        # If gathering, each node posts the number of spikes and
        # the number of cells to the master node (myid == 0)
        if gather and myid != 0:
            hoc_commands = []
            nspikes = 0;ncells  = 0
            for id in self.recorders['spikes'].recorded:
                if id in self.gidlist:
                    #nspikes += HocToPy.get('%s.object(%d).spiketimes.size()' %(self.hoc_label, self.gidlist.index(id)),'int')
                    nspikes += getattr(h, self.hoc_label).object(self.gidlist.index(id)).spiketimes.size()
                    ncells  += 1
            hoc_commands += ['tmp = pc.post("%s.node[%d].nspikes",%d)' % (self.hoc_label, myid, nspikes)]
            hoc_commands += ['tmp = pc.post("%s.node[%d].ncells",%d)' % (self.hoc_label, myid, ncells)]    
            hoc_execute(hoc_commands,"--- Population[%s].__meanSpikeCount()__ --- [Post spike count to master]" %self.label)
            return 0

        if myid==0 or not gather:
            nspikes = 0.0; ncells = 0.0
            hoc_execute(["nspikes = 0", "ncells = 0"])
            for id in self.recorders['spikes'].recorded:
                if id in self.gidlist:
                    nspikes += getattr(h, self.hoc_label).object(self.gidlist.index(id)).spiketimes.size()
                    ncells  += 1
            if gather:
                for id in range(1, nhost):
                    hoc_execute(['tmp = pc.take("%s.node[%d].nspikes",&nspikes)' % (self.hoc_label, id)])
                    #nspikes += HocToPy.get('nspikes','int')
                    nspikes += int(h.nspikes)
                    hoc_execute(['tmp = pc.take("%s.node[%d].ncells",&ncells)' % (self.hoc_label, id)])
                    #ncells  += HocToPy.get('ncells','int')
                    ncells += int(h.ncells)
            return float(nspikes)/ncells

    def randomInit(self, rand_distr):
        """
        Set initial membrane potentials for all the cells in the population to
        random values.
        """
        hoc_comment("--- Population[%s].__randomInit()__ ---" %self.label)
        self.rset("v_init", rand_distr)
    
    def describe(self):
        """
        Return a human readable description of the population"
        """
        print "\n------- Population description -------"
        print "Population called %s is made of %d cells [%d being local]" %(self.label, len(self.fullgidlist), len(self.gidlist))
        print "-> Cells are aranged on a %dD grid of size %s" %(len(self.dim), self.dim)
        print "-> Celltype is %s" %self.celltype
        print "-> Cell Parameters used for cell[0] (during initialization and now) are: " 
        for key, value in self.cellparams.items():
          print "\t|", key, "\t: ", "init->", value, "\t now->", getattr(self.cell[0],key)
        print "--- End of Population description ----"


class Projection(common.Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    
    nProj = 0
    
    def __init__(self, presynaptic_population, postsynaptic_population, method='allToAll',
                 method_parameters=None, source=None, target=None,
                 synapse_dynamics=None, label=None, rng=None):
        """
        presynaptic_population and postsynaptic_population - Population objects.
        
        source - string specifying which attribute of the presynaptic cell
                 signals action potentials
                 
        target - string specifying which synapse on the postsynaptic cell to
                 connect to
                 
        If source and/or target are not given, default values are used.
        
        method - string indicating which algorithm to use in determining
                 connections.
        Allowed methods are 'allToAll', 'oneToOne', 'fixedProbability',
        'distanceDependentProbability', 'fixedNumberPre', 'fixedNumberPost',
        'fromFile', 'fromList'.
        
        method_parameters - dict containing parameters needed by the connection
        method, although we should allow this to be a number or string if there
        is only one parameter.
        
        synapse_dynamics - a `SynapseDynamics` object specifying which
        synaptic plasticity mechanisms to use.
        
        rng - since most of the connection methods need uniform random numbers,
        it is probably more convenient to specify a RNG object here rather
        than within method_parameters, particularly since some methods also use
        random numbers to give variability in the number of connections per cell.
        """
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population, method,
                                   method_parameters, source, target, synapse_dynamics, label, rng)
        self.connections = []
        if not label:
            self.label = 'projection%d' % Projection.nProj
        self.hoc_label = self.label.replace(" ","_")
        if not rng:
            self.rng = NumpyRNG()
        hoc_commands = ['objref %s' % self.hoc_label,
                        '%s = new List()' % self.hoc_label]
        self.synapse_type = target
        
        ## Deal with short-term synaptic plasticity
        if self.short_term_plasticity_mechanism:
            U = self._short_term_plasticity_parameters['U']
            tau_rec = self._short_term_plasticity_parameters['tau_rec']
            tau_facil = self._short_term_plasticity_parameters['tau_facil']
            u0 = self._short_term_plasticity_parameters['u0']
            syn_code = {None: 1,
                        'excitatory': 1,
                        'inhibitory' :2}
            for cell in self.post:
                hoc_cell = cell._hoc_cell()
                hoc_cell.use_Tsodyks_Markram_synapses(syn_code[self.synapse_type], U, tau_rec, tau_facil, u0)
                
        self._syn_objref = _translate_synapse_type(self.synapse_type, extra_mechanism=self.short_term_plasticity_mechanism)

        ## Create connections
        if isinstance(method, str):
            connection_method = getattr(self,'_%s' % method)   
            hoc_commands += connection_method(method_parameters)
        elif isinstance(method, common.Connector):
            hoc_commands += method.connect(self)
            # delays should already be set to min_delay
        hoc_execute(hoc_commands, "--- Projection[%s].__init__() ---" %self.label)
        
        # By defaut, we set all the delays to min_delay, except if
        # the Projection data have been loaded from a file or a list.
        # This should already have been done if using a Connector object
        if isinstance(method, str) and (method != 'fromList') and (method != 'fromFile'):
            self.setDelays(get_min_delay())
                
        ## Deal with long-term synaptic plasticity
        if self.long_term_plasticity_mechanism:
            self._setupSTDP(self.long_term_plasticity_mechanism, self._stdp_parameters)
            
        Projection.nProj += 1

    def __len__(self):
        """Return the total number of connections."""
        return len(self.connections)

    # --- Connection methods ---------------------------------------------------
    
    def __connect(self, src, tgt):
        """
        Write hoc commands to connect a single pair of neurons.
        """
        cmdlist = ['nc = pc.gid_connect(%d,%s.object(%d).%s)' % (src,
                                                                 self.post.hoc_label,
                                                                 self.post.gidlist.index(tgt),
                                                                 self._syn_objref),
                'tmp = %s.append(nc)' % self.hoc_label]
        self.connections.append((src, tgt))
        return cmdlist
    
    def _allToAll(self, parameters=None):
        """
        Connect all cells in the presynaptic population to all cells in the
        postsynaptic population.
        """
        allow_self_connections = True # when pre- and post- are the same population,
                                      # is a cell allowed to connect to itself?
        if parameters and parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
        c = AllToAllConnector(allow_self_connections)
        return c.connect(self)
        
    def _oneToOne(self, parameters=None):
        """
        Where the pre- and postsynaptic populations have the same size, connect
        cell i in the presynaptic population to cell i in the postsynaptic
        population for all i.
        In fact, despite the name, this should probably be generalised to the
        case where the pre and post populations have different dimensions, e.g.,
        cell i in a 1D pre population of size n should connect to all cells
        in row i of a 2D post population of size (n,m).
        """
        c = OneToOneConnector()
        return c.connect(self)
    
    def _fixedProbability(self, parameters):
        """
        For each pair of pre-post cells, the connection probability is constant.
        """
        allow_self_connections = True
        try:
            p_connect = float(parameters)
        except TypeError:
            p_connect = parameters['p_connect']
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']

        c = FixedProbabilityConnector(p_connect=p_connect,
                                      allow_self_connections=allow_self_connections)
        return c.connect(self)    

    def _distanceDependentProbability(self, parameters):
        """
        For each pair of pre-post cells, the connection probability depends on distance.
        d_expression should be the right-hand side of a valid python expression
        for probability, involving 'd', e.g. "exp(-abs(d))", or "float(d<3)"
        """
        allow_self_connections = True
        if type(parameters) == types.StringType:
            d_expression = parameters
        else:
            d_expression = parameters['d_expression']
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']

        c = DistanceDependentProbabilityConnector(d_expression=d_expression,
                                                  allow_self_connections=allow_self_connections)
        return c.connect(self)
    
    def _fixedNumber(self, parameters, connector_class):
        allow_self_connections = True
        if type(parameters) == types.IntType:
            n = parameters
            assert n > 0
        elif type(parameters) == types.DictType:
            if parameters.has_key('n'): # all cells have same number of connections
                n = int(parameters['n'])
                assert n > 0
            elif parameters.has_key('rand_distr'): # number of connections per cell follows a distribution
                n = parameters['rand_distr']
                assert isinstance(n, RandomDistribution)
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
        elif isinstance(parameters, RandomDistribution):
            n = parameters
        else:
            raise Exception("Invalid argument type: should be an integer, dictionary or RandomDistribution object.")
        c = connector_class(n=n, allow_self_connections=allow_self_connections)
        return c.connect(self)
    
    def _fixedNumberPre(self, parameters):
        """Each presynaptic cell makes a fixed number of connections."""
        return self._fixedNumber(parameters, FixedNumberPreConnector)
            
    def _fixedNumberPost(self, parameters):
        """Each postsynaptic cell receives a fixed number of connections."""
        return self._fixedNumber(parameters, FixedNumberPostConnector)
    
    def _fromFile(self, parameters):
        """
        Load connections from a file.
        """
        lines =[]
        if type(parameters) == types.FileType:
            fileobj = parameters
            # should check here that fileobj is already open for reading
            lines = fileobj.readlines()
        elif type(parameters) == types.StringType:
            filename = parameters
            # now open the file...
            f = open(filename,'r',10000)
            lines = f.readlines()
        elif type(parameters) == types.DictType:
            # dict could have 'filename' key or 'file' key
            # implement this...
            raise "Argument type not yet implemented"
        
        # We read the file and gather all the data in a list of tuples (one per line)
        input_tuples = []
        for line in lines:
            single_line = line.rstrip()
            src, tgt, w, d = single_line.split("\t", 4)
            src = "[%s" % src.split("[",1)[1]
            tgt = "[%s" % tgt.split("[",1)[1]
            input_tuples.append((eval(src), eval(tgt), float(w), float(d)))
        f.close()
        return self._fromList(input_tuples)
    
    def _fromList(self, conn_list):
        """
        Read connections from a list of tuples,
        containing [pre_addr, post_addr, weight, delay]
        where pre_addr and post_addr are both neuron addresses, i.e. tuples or
        lists containing the neuron array coordinates.
        """
        hoc_commands = []
        
        # Then we go through those tuple and extract the fields
        for i in xrange(len(conn_list)):
            src, tgt, weight, delay = conn_list[i][:]
            src = self.pre[tuple(src)]
            tgt = self.post[tuple(tgt)]
            hoc_commands += self.__connect(src, tgt)
            hoc_commands += ['%s.object(%d).weight = %f' % (self.hoc_label, i, float(weight)), 
                             '%s.object(%d).delay = %f'  % (self.hoc_label, i, float(delay))]
        return hoc_commands
    
    # --- Methods for setting connection parameters ----------------------------
    
    def setWeights(self, w):
        """
        w can be a single number, in which case all weights are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        Weights should be in nA for current-based and µS for conductance-based
        synapses.
        """
        if isinstance(w, float) or isinstance(w, int):
            loop = ['for tmp = 0, %d {' % (len(self)-1), 
                        '%s.object(tmp).weight = %f ' % (self.hoc_label, float(w)),
                    '}']
            hoc_code = "".join(loop)
            hoc_commands = [ 'cmd = "%s"' % hoc_code,
                             'success = execute1(cmd)']
        elif isinstance(w, list) or isinstance(w, numpy.ndarray):
            hoc_commands = []
            assert len(w) == len(self), "List of weights has length %d, Projection %s has length %d" % (len(w), self.label, len(self))
            for i, weight in enumerate(w):
                hoc_commands += ['%s.object(%d).weight = %f' % (self.hoc_label, i, weight)]
        else:
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")
        hoc_execute(hoc_commands, "--- Projection[%s].__setWeights__() ---" % self.label)
        
    def randomizeWeights(self, rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        # If we have a native rng, we do the loops in hoc. Otherwise, we do the loops in
        # Python
        if isinstance(rand_distr.rng, NativeRNG):
            paramfmt = "%f,"*len(rand_distr.parameters); paramfmt = paramfmt.strip(',')
            distr_params = paramfmt % tuple(rand_distr.parameters)
            hoc_commands = ['rng = new Random(%d)' % 0 or distribution.rng.seed,
                            'tmp = rng.%s(%s)' % (rand_distr.name, distr_params)]
                            
            loop = ['for tmp = 0, %d {' %(len(self)-1), 
                        '%s.object(tmp).weight = rng.repick() ' %(self.hoc_label),
                    '}']
            hoc_code = "".join(loop)
            hoc_commands += ['cmd = "%s"' %hoc_code,
                             'success = execute1(cmd)']
        else:       
            hoc_commands = []
            for i in xrange(len(self)):
                hoc_commands += ['%s.object(%d).weight = %f' % (self.hoc_label, i, float(rand_distr.next()))]  
        hoc_execute(hoc_commands, "--- Projection[%s].__randomizeWeights__() ---" %self.label)
        
    def setDelays(self, d):
        """
        d can be a single number, in which case all delays are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        """
        if isinstance(d, float) or isinstance(d, int):
            if d < get_min_delay():
                raise Exception("Delays must be greater than or equal to the minimum delay, currently %g ms" % get_min_delay())
            loop = ['for tmp = 0, %d {' %(len(self)-1), 
                        '%s.object(tmp).delay = %f ' % (self.hoc_label, float(d)),
                    '}']
            hoc_code = "".join(loop)
            hoc_commands = [ 'cmd = "%s"' %hoc_code,
                             'success = execute1(cmd)']
            # if we have STDP, need to update pre2wa and post2wa delays as well
            if self.synapse_dynamics and self.synapse_dynamics.slow:
                ddf = self.synapse_dynamics.slow.dendritic_delay_fraction
                loop = ['for i = 0, %d {' %(len(self)-1), 
                            '%s_pre2wa[i].delay = %f ' % (self.hoc_label, float(d)*(1-ddf)),
                            '%s_post2wa[i].delay = %f ' % (self.hoc_label, float(d)*ddf),
                        '}']
            hoc_commands = [ 'cmd = "%s"' % "".join(loop),
                             'success = execute1(cmd)']
        elif isinstance(d, list) or isinstance(d, numpy.ndarray):
            # need check for min_delay here
            hoc_commands = []
            assert len(d) == len(self), "List of delays has length %d, Projection %s has length %d" % (len(d), self.label, len(self))
            for i, delay in enumerate(d):
                hoc_commands += ['%s.object(%d).delay = %f' % (self.hoc_label, i, delay)]
            # if we have STDP, need to update pre2wa and post2wa delays as well
            if self.synapse_dynamics and self.synapse_dynamics.slow:
                ddf = self.synapse_dynamics.slow.dendritic_delay_fraction
                for i, delay in enumerate(d):
                    hoc_commands += ['%s_pre2wa[%d].delay = %f' % (self.hoc_label, i, delay*(1-ddf)),
                                     '%s_post2wa[%d].delay = %f' % (self.hoc_label, i, delay*ddf)]
        else:
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")
        hoc_execute(hoc_commands, "--- Projection[%s].__setDelays__() ---" %self.label)
        
    def randomizeDelays(self, rand_distr):
        """
        Set delays to random values taken from rand_distr.
        """   
        # If we have a native rng, we do the loops in hoc. Otherwise, we do the loops in
        # Python
        # if we have STDP, need to update pre2wa and post2wa delays as well
        if isinstance(rand_distr.rng, NativeRNG):
            paramfmt = "%f,"*len(rand_distr.parameters); paramfmt = paramfmt.strip(',')
            distr_params = paramfmt % tuple(rand_distr.parameters)
            hoc_commands = ['rng = new Random(%d)' % 0 or distribution.rng.seed,
                            'tmp = rng.%s(%s)' % (rand_distr.name, distr_params)]
            if self.synapse_dynamics and self.synapse_dynamics.slow:
                ddf = self.synapse_dynamics.slow.dendritic_delay_fraction
                hoc_commands += ['ddf = %g' % ddf]
                loop = ['for i = 0, %d {' % (len(self)-1),
                            'rr = rng.repick()',
                            '%s.object(i).delay = rr ' % (self.hoc_label),
                            '%s_pre2wa[i].delay = rr*(1-ddf)' % (self.hoc_label),
                            '%s_post2wa[i].delay = rr*ddf' % (self.hoc_label),
                        '}']    
            else:
                loop = ['for tmp = 0, %d {' % (len(self)-1), 
                            '%s.object(tmp).delay = rng.repick() ' %(self.hoc_label),
                        '}']
            hoc_code = "".join(loop)
            hoc_commands += ['cmd = "%s"' % hoc_code,
                             'success = execute1(cmd)']    
        else:
            hoc_commands = []
            if self.synapse_dynamics and self.synapse_dynamics.slow:
                ddf = self.synapse_dynamics.slow.dendritic_delay_fraction
                for i in xrange(len(self)):
                    rr = float(rand_distr.next())
                    hoc_commands += ['%s.object(%d).delay = %f' % (self.hoc_label, i, rr),
                                     '%s_pre2wa[%d].delay = %f' % (self.hoc_label, i, rr*(1-ddf)),
                                     '%s_post2wa[%d].delay = %f' % (self.hoc_label, i, rr*ddf)]
            else:
                for i in xrange(len(self)):
                    hoc_commands += ['%s.object(%d).delay = %f' % (self.hoc_label, i, float(rand_distr.next()))]
        hoc_execute(hoc_commands, "--- Projection[%s].__randomizeDelays__() ---" %self.label)
    
    def setSynapseDynamics(self, param, value):
        """
        Set parameters of the synapse dynamics linked with the projection
        """
        raise Exception("Method not yet implemented !")
    
    def randomizeSynapseDynamics(self, param, rand_distr):
        """
        Set parameters of the synapse dynamics to values taken from rand_distr
        """
        raise Exception("Method not yet implemented !")
        
    def setTopographicDelays(self, delay_rule, rand_distr=None, mask=None, scale_factor=1.0):
        """
        Set delays according to a connection rule expressed in delay_rule, based
        on the delay distance 'd' and an (optional) rng 'rng'. For example,
        the rule can be "rng*d + 0.5", with "a" extracted from the rng and
        d being the distance.
        """
        # if we have STDP, need to update pre2wa and post2wa delays as well
        if self.synapse_dynamics and self.synapse_dynamics.slow:
            raise Exception("setTopographicDelays() does not currently work with STDP")
        hoc_commands = []
        
        if rand_distr==None:
            for i in xrange(len(self)):
                src = self.connections[i][0]
                tgt = self.connections[i][1]
                # calculate the distance between the two cells
                idx_src = numpy.where(self.pre.fullgidlist == src)[0][0]
                idx_tgt = numpy.where(self.post.fullgidlist == tgt)[0][0]
                dist = common.distance(self.pre.fullgidlist[idx_src], self.post.fullgidlist[idx_tgt],
                                       mask, scale_factor)
                # then evaluate the delay according to the delay rule
                delay = eval(delay_rule.replace('d', '%f' %dist))
                hoc_commands += ['%s.object(%d).delay = %f' % (self.hoc_label, i, float(delay))]
        else:
            if isinstance(rand_distr.rng, NativeRNG):
                paramfmt = "%f,"*len(rand_distr.parameters); paramfmt = paramfmt.strip(',')
                distr_params = paramfmt % tuple(rand_distr.parameters)
                hoc_commands += ['rng = new Random(%d)' % 0 or distribution.rng.seed,
                            'tmp = rng.%s(%s)' % (rand_distr.name, distr_params)]
                for i in xrange(len(self)):
                    src = self.connections[i][0]
                    tgt = self.connections[i][1]
                    # calculate the distance between the two cells
                    idx_src = self.pre.fullgidlist.index(src)
                    idx_tgt = self.post.fullgidlist.index(tgt)
                    dist = common.distance(self.pre.fullgidlist[idx_src], self.post.fullgidlist[idx_tgt],
                                           mask, scale_factor)
                    # then evaluate the delay according to the delay rule
                    delay = delay_rule.replace('d', '%f' % dist)
                    #delay = eval(delay.replace('rng', '%f' % HocToPy.get('rng.repick()', 'float')))
                    delay = eval(delay.replace('rng', '%f' % h.rng.repick()))
                    hoc_commands += ['%s.object(%d).delay = %f' % (self.hoc_label, i, float(delay))]   
            else:
                for i in xrange(len(self)):
                    src = self.connections[i][0]
                    tgt = self.connections[i][1]    
                    # calculate the distance between the 2 cells :
                    idx_src = self.pre.fullgidlist.index(src)
                    idx_tgt = self.post.fullgidlist.index(tgt)
                    dist = common.distance(self.pre.fullgidlist[idx_src], self.post.fullgidlist[idx_tgt],
                                           mask, scale_factor)
                    # then evaluate the delay according to the delay rule :
                    delay = delay_rule.replace('d', '%f' %dist)
                    delay = eval(delay.replace('rng', '%f' %rand_distr.next()))
                    hoc_commands += ['%s.object(%d).delay = %f' % (self.hoc_label, i, float(delay))]        
        
        hoc_execute(hoc_commands, "--- Projection[%s].__setTopographicDelays__() ---" %self.label)
    
    # --- Methods relating to synaptic plasticity ------------------------------
    
    def _setupSTDP(self, stdp_model, parameterDict):
        """Set-up STDP."""
        ddf = self.synapse_dynamics.slow.dendritic_delay_fraction
        if ddf > 0.5 and nhost > 1:
            # depending on delays, can run into problems with the delay from the
            # pre-synaptic neuron to the weight-adjuster mechanism being zero.
            # The best (only?) solution would be to create connections on the
            # node with the pre-synaptic neurons for ddf>0.5 and on the node
            # with the post-synaptic neuron (as is done now) for ddf<0.5
            raise Exception("STDP with dendritic_delay_fraction > 0.5 is not yet supported for parallel computation.")
        # Define the objref to handle plasticity
        hoc_commands =  ['objref %s_wa[%d]'      %(self.hoc_label, len(self)),
                         'objref %s_pre2wa[%d]'  %(self.hoc_label, len(self)),
                         'objref %s_post2wa[%d]' %(self.hoc_label, len(self))]
        # For each connection
        for i in xrange(len(self)):
            src = self.connections[i][0]
            tgt = self.connections[i][1]
            # we reproduce the structure of STDP that can be found in layerConn.hoc
            hoc_commands += [
                '%s_wa[%d] = new %s(0.5)' %(self.hoc_label, i, stdp_model),
                '%s_pre2wa[%d] = pc.gid_connect(%d, %s_wa[%d])' % (self.hoc_label, i, src, self.hoc_label, i),
                '%s_pre2wa[%d].threshold = %s.object(%d).threshold' %(self.hoc_label, i, self.hoc_label, i),
                '%s_pre2wa[%d].delay = %s.object(%d).delay * %g' % (self.hoc_label, i, self.hoc_label, i, (1-ddf)),
                '%s_pre2wa[%d].weight = 1' %(self.hoc_label, i),
                #'%s_post2wa[%d] = pc.gid_connect(%d, %s_wa[%d])' %(self.hoc_label, i, tgt, self.hoc_label, i),
                # directly create NetCon as wa is on the same machine as the post-synaptic cell
                '%s_post2wa[%d] = new NetCon(%s.object(%d).source, %s_wa[%d])' % (self.hoc_label, i, self.post.hoc_label, self.post.gidlist.index(tgt), self.hoc_label,i),
                '%s_post2wa[%d].threshold = 1' %(self.hoc_label, i),
                '%s_post2wa[%d].delay = %s.object(%d).delay * %g' % (self.hoc_label, i, self.hoc_label, i, ddf),
                '%s_post2wa[%d].weight = -1' % (self.hoc_label, i),
                'setpointer %s_wa[%d].wsyn, %s.object(%d).weight' %(self.hoc_label, i, self.hoc_label, i)]   
        # then update the parameters
        for param, val in parameterDict.items():
            hoc_commands += ['%s_wa[%d].%s = %f' % (self.hoc_label, i, param, val)]
        hoc_execute(hoc_commands, "--- Projection[%s].__setupSTDP__() ---" %self.label)  
        # debugging
        #pre2wa_array = getattr(h, "%s_pre2wa" % self.hoc_label)
        #for i in xrange(len(self)):
        #    print pre2wa_array[i].delay,
        #print
        #post2wa_array = getattr(h, "%s_post2wa" % self.hoc_label)
        #for i in xrange(len(self)):
        #    print post2wa_array[i].delay,
    
    # --- Methods for writing/reading information to/from file. ----------------
    
    def getWeights(self, format='list', gather=True):
        """
        Possible formats are: a list of length equal to the number of connections
        in the projection, a 2D weight array (with zero or None for non-existent
        connections).
        """
        assert format in ('list', 'array'), "`format` is '%s', should be one of 'list', 'array'" % format
        if format == 'list':
            values = [getattr(h, self.hoc_label).object(i).weight[0] for i in range(len(self))]
        elif format == 'array':
            values = numpy.zeros((len(self.pre), len(self.post)), 'float')
            for i in xrange(len(self)):
                weight = getattr(h, self.hoc_label).object(i).weight[0]
                values[self.connections[i][0]-self.pre.gid_start,
                       self.connections[i][1]-self.post.gid_start] = weight
        return values
        
    def getDelays(self, format='list', gather=True):
        """
        Possible formats are: a list of length equal to the number of connections
        in the projection, a 2D delay array (with None or 1e12 for non-existent
        connections).
        """
        assert format in ('list', 'array'), "`format` is '%s', should be one of 'list', 'array'" % format
        if format == 'list':
            values = [getattr(h, self.hoc_label).object(i).delay for i in range(len(self))]
        elif format == 'array':
            raise Exception("Not yet implemented")
        return values
    
    def saveConnections(self, filename, gather=False):
        """Save connections to file in a format suitable for reading in with the
        'fromFile' method."""
        if gather:
            raise Exception("saveConnections() with gather=True not yet implemented")
        elif num_processes() > 1:
            filename += '.%d' % rank()
        hoc_comment("--- Projection[%s].__saveConnections__() ---" % self.label)
        f = open(filename, 'w', 10000)
        for i in xrange(len(self)):
            src = self.connections[i][0]
            tgt = self.connections[i][1]
            line = "%s%s\t%s%s\t%g\t%g\n" % (self.pre.hoc_label,
                                     self.pre.locate(src),
                                     self.post.hoc_label,
                                     self.post.locate(tgt),
                                     getattr(h, self.hoc_label).object(i).weight[0],
                                     getattr(h, self.hoc_label).object(i).delay)
            line = line.replace('(','[').replace(')',']')
            f.write(line)
        f.close()
    
    def printWeights(self, filename, format='list', gather=True):
        """Print synaptic weights to file."""
        global myid
        
        hoc_execute(['objref weight_list'])
        hoc_commands = [] 
        hoc_comment("--- Projection[%s].__printWeights__() ---" %self.label)
        
        # Here we have to deal with the gather options. If we gather, then each
        # slave node posts its list of weights to the master node.
        if gather and myid !=0:
            if format == 'array': raise Exception("Gather not implemented for 'array'.")
            hoc_commands += ['weight_list = new Vector()']
            for i in xrange(len(self)):
                #weight = HocToPy.get('%s.object(%d).weight' % (self.hoc_label, i),'float')
                weight = getattr(h, self.hoc_label).object(i).weight[0]
                hoc_commands += ['weight_list = weight_list.append(%f)' % weight]
            hoc_commands += ['tmp = pc.post("%s.weight_list.node[%d]", weight_list)' %(self.hoc_label, myid)]
            hoc_execute(hoc_commands, "--- [Posting weights list to master] ---")

        if not gather or myid == 0:
            if hasattr(filename, 'write'): # filename should be renamed to file, to allow open file objects to be used
                f = filename
            else:
                f = open(filename,'w',10000)
            if format == 'list':
                for i in xrange(len(self)):
                    #weight = "%f\n" %HocToPy.get('%s.object(%d).weight' % (self.hoc_label, i),'float')
                    weight = getattr(h, self.hoc_label).object(i).weight[0]
                    f.write("%f\n" % weight)
            elif format == 'array':
                weights = numpy.zeros((len(self.pre), len(self.post)), 'float')
                fmt = "%g "*len(self.post) + "\n"
                for i in xrange(len(self)):
                    weight = getattr(h, self.hoc_label).object(i).weight[0]
                    weights[self.connections[i][0]-self.pre.gid_start,
                            self.connections[i][1]-self.post.gid_start] = weight
                for row in weights:
                    f.write(fmt % tuple(row))
            else:
                raise Exception("Valid formats are 'list' and 'array'")
            if gather:
                if format == 'array' and nhost > 1: raise Exception("Gather not implemented for array format.")
                for id in range (1, nhost):
                    hoc_commands = ['weight_list = new Vector()']       
                    hoc_commands += ['tmp = pc.take("%s.weight_list.node[%d]", weight_list)' %(self.hoc_label, id)]
                    hoc_execute(hoc_commands)                
                    #for j in xrange(HocToPy.get('weight_list.size()', 'int')):
                    for j in xrange(int(h.weight_list.size)):
                        #weight = "%f\n" %HocToPy.get('weight_list.x[%d]' %j, 'float')
                        weight = h.weight_list.x[j]
                        f.write("%f\n" % weight)
            if not hasattr(filename, 'write'):
                f.close()
  
    def weightHistogram(self, min=None, max=None, nbins=10):
        """
        Return a histogram of synaptic weights.
        If min and max are not given, the minimum and maximum weights are
        calculated automatically.
        """
        # it is arguable whether functions operating on the set of weights
        # should be put here or in an external module.
        bins = numpy.arange(min, max, (max-min)/nbins)
        return numpy.histogram(self.getWeights(), bins) # returns n, bins
    
# ==============================================================================
#   Utility classes
# ==============================================================================

Timer = common.Timer
    
# ==============================================================================
