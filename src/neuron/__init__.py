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
Set = set

gid           = 0
ncid          = 0
gidlist       = []
vfilelist     = {}
spikefilelist = {}
dt            = 0.1
running       = False
initialised   = False
_min_delay    = 0.0
nrn_dll_loaded = False

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
    
    #def __init__(self,n):
    #    common.ID.__init__(self,n)
    #    self.hocname = None
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
        if self.parent:
            hoc_cell_list = getattr(h, self.parent.label)
            cell = hoc_cell_list.object(self - self.parent.gid_start)
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
            parameter_names = None # for native cells, don't have a way to get their list of parameters
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
        for name, val in parameters.items():
            if hasattr(val, '__len__'):
                setattr(cell, name, Vector(val).hoc_obj)
            else:
                setattr(cell, name, val)
            cell.param_update()
        
def list_standard_models():
    return [obj for obj in globals().values() if isinstance(obj, type) and issubclass(obj, common.StandardCellType)]

def load_mechanisms():
    global nrn_dll_loaded
    if not nrn_dll_loaded:
        hoc.execute('nrn_load_dll("%s/hoc/i686/.libs/libnrnmech.so")' % pyNN_path[0])
        nrn_dll_loaded = True

# ==============================================================================
#   Module-specific functions and classes (not part of the common API)
# ==============================================================================

class HocError(Exception): pass

def hoc_execute(hoc_commands, comment=None):
    assert isinstance(hoc_commands,list)
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
                             'argvec%d = new Vector(%d)' % (nvec,len(item))]
            argstr += 'argvec%d, ' % nvec
            for i in xrange(len(item)):
                hoc_commands.append('argvec%d.x[%d] = %g' % (nvec,i,item[i])) # assume only numerical values
            nvec += 1
        elif type(item) == types.StringType:
            hoc_commands += ['strdef argstr%d' % nstr,
                             'argstr%d = "%s"' % (nstr,item)]
            argstr += 'argstr%d, ' % nstr
            nstr += 1
        elif type(item) == types.DictType:
            dict_init_list = []
            for k,v in item.items():
                if type(v) == types.StringType:
                    dict_init_list += ['"%s", "%s"' % (k,v)]
                elif type(v) == types.ListType:
                    hoc_commands += ['objref argvec%d' % nvec,
                                     'argvec%d = new Vector(%d)' % (nvec,len(v))]
                    dict_init_list += ['"%s", argvec%d' % (k,nvec)]
                    for i in xrange(len(v)):
                        hoc_commands.append('argvec%d.x[%d] = %g' % (nvec,i,v[i])) # assume only numerical values
                    nvec += 1
                else: # assume number
                    dict_init_list += ['"%s", %g' % (k,float(v))]
            hoc_commands += ['objref argdict%d' % ndict,
                             'argdict%d = new Dict(%s)' % (ndict,", ".join(dict_init_list))]
            argstr += 'argdict%d, ' % ndict
            ndict += 1
        elif isinstance(item,numpy.ndarray):
            ndim = len(item.shape)
            if ndim == 1:  # this has not been tested yet
                cmd, argstr1 = _hoc_arglist([list(item)]) # convert to a list and call the current function recursively
                hoc_commands += cmd
                argstr += argstr1
            elif ndim == 2:
                argstr += 'argmat%s,' % nmat
                hoc_commands += ['objref argmat%d' % nmat,
                                 'argmat%d = new Matrix(%d,%d)' % (nmat,item.shape[0],item.shape[1])]
                for i in xrange(item.shape[0]):
                    for j in xrange(item.shape[1]):
                        try:
                          hoc_commands += ['argmat%d.x[%d][%d] = %g' % (nmat,i,j,item[i,j])]
                        except TypeError:
                          raise common.InvalidParameterValueError
                nmat += 1
            else:
                raise common.InvalidDimensionsError, 'number of dimensions must be 1 or 2'
        elif item is None:
            pass
        else:
            hoc_commands += ['argvar%d = %f' % (nvar,item)]
            argstr += 'argvar%d, ' % nvar
            nvar += 1
    return hoc_commands, argstr.strip().strip(',')

def get_min_delay():
    global _min_delay
    return _min_delay

def _translate_synapse_type(synapse_type,weight=None):
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
    return syn_objref

def checkParams(param,val=None):
    """Check parameters are of valid types, normalise the different ways of
       specifying parameters and values by putting everything in a dict.
       Called by set() and Population.set()."""
    if isinstance(param,str):
        if isinstance(val,float) or isinstance(val,int):
            param_dict = {param:float(val)}
        elif isinstance(val,(str, list)):
            param_dict = {param:val}
        else:
            raise common.InvalidParameterValueError
    elif isinstance(param,dict):
        param_dict = param
    else:
        raise common.InvalidParameterValueError
    return param_dict

class HocToPy:
    """Static class to simplify getting variables from hoc."""
    
    fmt_dict = {'int' : '%d', 'integer' : '%d', 'float' : '%f', 'double' : '%f',
                'string' : '\\"%s\\"', 'str' : '\\"%s\\"'}
    
    @staticmethod
    def get(name,return_type='float'):
        """Return a variable from hoc.
           name can be a hoc variable (int, float, string) or a function/method
           that returns such a variable.
        """
        # We execute some commands here to avoid too much outputs in the log file
        errorstr = '"raise HocError(\'caused by HocToPy.get(%s,return_type=\\"%s\\")\')"' % (name,return_type)
        hoc_commands = ['success = sprint(cmd,"HocToPy.hocvar = %s",%s)' % (HocToPy.fmt_dict[return_type],name),
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
                
# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1,min_delay=0.1,max_delay=0.1,debug=False,**extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    global dt, nhost, myid, _min_delay, logger, initialised
    load_mechanisms()
    dt = timestep
    _min_delay = min_delay
    print "min delay=", _min_delay
        
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
        
    logging.info("Initialization of NEURON (use setup(..,debug=True) to see a full logfile)")
    
    # All the objects that will be used frequently in the hoc code are declared in the setup
    if initialised:
        hoc_commands = ['dt = %f' % dt]    
    else:
        hoc_commands = [
            'tmp = xopen("%s")' % os.path.join(pyNN_path[0],'hoc','standardCells.hoc'),
            'tmp = xopen("%s")' % os.path.join(pyNN_path[0],'hoc','odict.hoc'),
            'objref pc',
            'pc = new ParallelContext()',
            'dt = %f' % dt,
            'create dummy_section',
            'access dummy_section',
            'objref netconlist, nil',
            'netconlist = new List()', 
            'strdef cmd',
            'strdef fmt', 
            'objref nc', 
            'objref rng',
            'objref cell']    
        #---Experimental--- Optimize the simulation time ? / Reduce inter-processors exchanges ?
        hoc_commands += [
            'tmp   = pc.spike_compress(1,0)']
        if extra_params.has_key('use_cvode') and extra_params['use_cvode'] == True:
            hoc_commands += [
                'objref cvode',
                'cvode = new CVode()',
                'cvode.active(1)']   
        
    hoc_execute(hoc_commands,"--- setup() ---")
    #nhost = HocToPy.get('pc.nhost()','int')
    nhost = int(h.pc.nhost())
    if nhost < 2:
        nhost = 1; myid = 0
    else:
        #myid = HocToPy.get('pc.id()','int')
        myid = int(h.pc.id())
    print "\nHost #%d of %d" % (myid+1, nhost)
    
    initialised = True
    return int(myid)

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    global logfile, myid #, vfilelist, spikefilelist
    hoc_commands = []
    if len(vfilelist) > 0:
        hoc_commands = ['objref fileobj',
                        'fileobj = new File()']
        while len(vfilelist):
            filename, cell_list = vfilelist.popitem()
            #tstop = HocToPy.get('tstop','float')
            tstop = h.tstop
            header = "# dt = %g\\n# n = %d\\n" % (dt,int(tstop/dt))
            hoc_commands += ['tmp = fileobj.wopen("%s")' % filename,
                             'tmp = fileobj.printf("%s")' % header]
            for cell in cell_list:
                hoc_commands += ['fmt = "%s\\t%d\\n"' % ("%.6g",cell),
                                 'tmp = cell%d.vtrace.printf(fileobj,fmt)' % cell]
            hoc_commands += ['tmp = fileobj.close()']
    if len(spikefilelist) > 0:
        hoc_commands += ['objref fileobj',
                        'fileobj = new File()']
        header = "# dt = %g\\n"% dt
        while len(spikefilelist):
            filename, cell_list = spikefilelist.popitem()
            hoc_commands += ['tmp = fileobj.wopen("%s")' % filename,
                             'tmp = fileobj.printf("%s")' % header]
            for cell in cell_list:
                hoc_commands += ['fmt = "%s\\t%d\\n"' % ("%.2f",cell),
                                 #'tmp = fileobj.printf("# cell%d\\n")' % cell,
                                 'tmp = cell%d.spiketimes.where("<=",tstop).printf(fileobj,fmt)' % cell]
            hoc_commands += ['tmp = fileobj.close()']
    hoc_commands += ['tmp = pc.runworker()',
                     'tmp = pc.done()']
    hoc_execute(hoc_commands,"--- end() ---")
    #hoc.execute('tmp = quit()')
    #logging.info("Finishing up with NEURON.")
    #sys.exit(0)

def run(simtime):
    """Run the simulation for simtime ms."""
    global running
    hoc_commands = []
    if not running:
        running = True
        hoc_commands += ['tstop = 0',
                         'local_minimum_delay = pc.set_maxstep(100)',
                         'tmp = finitialize()',]
    hoc_commands += ['tstop += %f' %simtime,
                     #'print "tstop     = ", tstop',
                     'tmp = pc.psolve(tstop)']
    hoc_execute(hoc_commands,"--- run() ---")
    return current_time()

def current_time():
    """Return the current time in the simulation."""
    return h.t

def get_time_step():
    return h.dt

def setRNGseeds(seedList):
    """Globally set rng seeds."""
    pass # not applicable to NEURON?

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

def create(cellclass,param_dict=None,n=1):
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
    elif isinstance(cellclass,str):
        hoc_name = cellclass
        hoc_commands, argstr = _hoc_arglist([param_dict])
    argstr = argstr.strip().strip(',')
 
    # round-robin partitioning
    newgidlist = [i+myid for i in range(gid,gid+n,nhost) if i < gid+n-myid]
    for cell_id in newgidlist:
        hoc_commands += ['tmp = pc.set_gid2node(%d,%d)' % (cell_id,myid),
                         'objref cell%d' % cell_id,
                         'cell%d = new %s(%s)' % (cell_id,hoc_name,argstr),
                         'tmp = cell%d.connect2target(nil,nc)' % cell_id,
                         #'nc = new NetCon(cell%d.source,nil)' % cell_id,
                         'tmp = pc.cell(%d,nc)' % cell_id]
    hoc_execute(hoc_commands, "--- create() ---")

    gidlist.extend(newgidlist)
    cell_list = [ID(i) for i in range(gid,gid+n)]
    for id in cell_list:
        id.cellclass = cellclass
    gid = gid+n
    if n == 1:
        cell_list = cell_list[0]
    return cell_list

def connect(source,target,weight=None,delay=None,synapse_type=None,p=1,rng=None):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise.
    Weights should be in nA or uS."""
    global ncid, gid, gidlist, _min_delay
    if type(source) != types.ListType:
        source = [source]
    if type(target) != types.ListType:
        target = [target]
    if weight is None:  weight = 0.0
    if delay  is None:  delay = _min_delay
    syn_objref = _translate_synapse_type(synapse_type,weight)
    nc_start = ncid
    hoc_commands = []
    for tgt in target:
        if tgt > gid or tgt < 0 or not isinstance(tgt,int):
            raise common.ConnectionError, "Postsynaptic cell id %s does not exist." % str(tgt)
        else:
            if tgt in gidlist: # only create connections to cells that exist on this machine
                if p < 1:
                    if rng: # use the supplied RNG
                        rarr = self.rng.uniform(0,1,len(source))
                    else:   # use the default RNG
                        rarr = numpy.random.uniform(0,1,len(source))
                for j,src in enumerate(source):
                    if src > gid or src < 0 or not isinstance(src,int):
                        raise common.ConnectionError, "Presynaptic cell id %s does not exist." % str(src)
                    else:
                        if p >= 1.0 or rarr[j] < p: # might be more efficient to vectorise the latter comparison
                            hoc_commands += [#'nc = pc.gid_connect(%d,pc.gid2cell(%d).%s)' % (src,tgt,syn_objref),
                                             'nc = pc.gid_connect(%d, cell%d.%s)' % (src,tgt,syn_objref),
                                             'nc.delay = %g' % delay,
                                             'nc.weight = %g' % weight,
                                             'tmp = netconlist.append(nc)']
                            ncid += 1
    hoc_execute(hoc_commands, "--- connect(%s,%s) ---" % (str(source),str(target)))
    return range(nc_start,ncid)

def set(cells,cellclass,param,val=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value.
    cellclass must be supplied for doing translation of parameter names."""
    global gidlist
    
    param_dict = checkParams(param,val)

    if type(cellclass) == type and issubclass(cellclass, common.StandardCellType):
        param_dict = cellclass({}).translate(param_dict)
    if not isinstance(cells,list):
        cells = [cells]    
    hoc_commands = []
    if cells[0].parent:
        prefix = "%s.object(%s)" % (cells[0].parent.hoc_label, "%d")
    else:
        prefix = "cell%d"
    for param,val in param_dict.items():
        if isinstance(val,str):
            #fmt = 'pc.gid2cell(%d).%s = "%s"'
            fmt = prefix + '.%s = "%s"'
        elif isinstance(val,list):
            cmds,argstr = _hoc_arglist([val])
            hoc_commands += cmds
            #fmt = 'pc.gid2cell(%d).%s = %s'
            fmt = prefix + '.%s = %s'
            val = argstr
        else:
            #fmt = 'pc.gid2cell(%d).%s = %g'
            fmt = prefix + '.%s = %g'
        for cell in cells:
            if cell in gidlist:
                id = int(cell)
                if cell.parent:
                    id = list(cell.parent.cell).index(id) # id=cell.parent.locate(cell)[0]) ??
                hoc_commands += [fmt % (id, param, val),
                                 "tmp = %s.param_update()" % prefix % id,]
                                 #'if (pc.gid2cell(%d) != pc.gid2obj(%d)) { tmp = pc.gid2cell(%d).param_update() }' % (cell,cell,cell)]
    hoc_execute(hoc_commands, "--- set() ---")

def record(source,filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    global spikefilelist, gidlist
    if type(source) != types.ListType:
        source = [source]
    hoc_commands = []
    if not spikefilelist.has_key(filename):
        spikefilelist[filename] = []
    for src in source:
        if src in gidlist:
            hoc_commands += ['tmp = cell%d.record(1)' % src]
            spikefilelist[filename] += [src] # writing to file is done in end()
    hoc_execute(hoc_commands, "---record() ---")

def record_v(source,filename):
    """
    Record membrane potential to a file. source can be an individual cell or
    a list of cells."""
    # would actually like to be able to record to an array and
    # choose later whether to write to a file.
    global vfilelist, gidlist
    if type(source) != types.ListType:
        source = [source]
    hoc_commands = []
    if not vfilelist.has_key(filename):
        vfilelist[filename] = []
    for src in source:
        if src in gidlist:
            hoc_commands += ['tmp = cell%d.record_v(1,%g)' % (src,dt)]
            vfilelist[filename] += [src] # writing to file is done in end()
    hoc_execute(hoc_commands, "---record_v() ---")

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
        
        common.Population.__init__(self,dims,cellclass,cellparams,label)

        # set the steps list, used by the __getitem__() method.
        self.steps = [1]*self.ndim
        for i in xrange(self.ndim-1):
            for j in range(i+1,self.ndim):
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
        
        self.record_from = { 'spiketimes': Set(), 'vtrace': Set() }
        
        
        # Now the gid and cellclass are stored as instance of the ID class, which will allow a syntax like
        # p[i,j].set(param, val). But we have also to deal with positions : a population needs to know ALL the positions
        # of its cells, and not only those of the cells located on a particular node (i.e in self.gidlist). So
        # each population should store what we call a "fullgidlist" with the ID of all the cells in the populations 
        # (and therefore their positions)
        self.fullgidlist = numpy.array([ID(i) for i in range(gid, gid+self.size) if i < gid+self.size], ID)
        self.cell = self.fullgidlist
        
        # self.gidlist is now derived from self.fullgidlist since it contains only the cells of the population located on
        # the node
        self.gidlist     = [self.fullgidlist[i+myid] for i in range(0, len(self.fullgidlist),nhost) if i < len(self.fullgidlist)-myid]
        self.gid_start   = gid

        # Write hoc commands
        hoc_commands += ['objref %s' % self.hoc_label,
                         '%s = new List()' % self.hoc_label]

        for cell_id in self.gidlist:
            hoc_commands += ['tmp = pc.set_gid2node(%d,%d)' % (cell_id,myid),
                             'cell = new %s(%s)' % (hoc_name,argstr),
                             #'nc = new NetCon(cell.source,nil)',
                             'tmp = cell.connect2target(nil,nc)',
                             'tmp = pc.cell(%d,nc)' % cell_id,
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

    def __getitem__(self,addr):
        """Returns a representation of the cell with coordinates given by addr,
           suitable for being passed to other methods that require a cell id.
           Note that __getitem__ is called when using [] access, e.g.
             p = Population(...)
             p[2,3] is equivalent to p.__getitem__((2,3)).
        """

        global gidlist

        # What we actually pass around are gids.
        if isinstance(addr,int):
            addr = (addr,)
        if len(addr) != len(self.dim):
            raise common.InvalidDimensionsError, "Population has %d dimensions. Address was %s" % (self.ndim,str(addr))
        index = 0
        for i,s in zip(addr,self.steps):
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
        
    def locate(self,id):
        """Given an element id in a Population, return the coordinates.
               e.g. for  4 6  , element 2 has coordinates (1,0) and value 7
                         7 9
        """
        # id should be a gid
        assert isinstance(id,int), "id is %s, not int" % type(id)
        id -= self.gid_start
        if self.ndim == 3:
            rows = self.dim[1]; cols = self.dim[2]
            i = id/(rows*cols); remainder = id%(rows*cols)
            j = remainder/cols; k = remainder%cols
            coords = (i,j,k)
        elif self.ndim == 2:
            cols = self.dim[1]
            i = id/cols; j = id%cols
            coords = (i,j)
        elif self.ndim == 1:
            coords = (id,)
        else:
            raise common.InvalidDimensionsError
        return coords

    def __len__(self):
        """Returns the total number of cells in the population."""
        return self.size

    def set(self,param,val=None):
        """
        Set one or more parameters for every cell in the population. param
        can be a dict, in which case val should not be supplied, or a string
        giving the parameter name, in which case val is the parameter value.
        val can be a numeric value, or list of such (e.g. for setting spike times).
        e.g. p.set("tau_m",20.0).
             p.set({'tau_m':20,'v_rest':-65})
        """
        param_dict = checkParams(param,val)
        if isinstance(self.celltype, common.StandardCellType):
            param_dict = self.celltype.translate(param_dict)

        strfmt  = '%s.object(tmp).%s = "%s"' % (self.hoc_label,"%s","%s")
        numfmt  = '%s.object(tmp).%s = %s' % (self.hoc_label,"%s","%g")
        listfmt = '%s.object(tmp).%s = %s' % (self.hoc_label,"%s","%s")
        for param,val in param_dict.items():
            if isinstance(val,str):
                fmt = strfmt
            elif isinstance(val,list):
                cmds,argstr = _hoc_arglist([val])
                hoc_commands += cmds
                fmt = listfmt
                val = argstr
            else:
                fmt = numfmt
            # We do the loop in hoc, to speed up the code
            loop = "for tmp = 0, %d" %(len(self.gidlist)-1)
            cmd  = fmt % (param,val)
            hoc_commands = ['cmd="%s { %s success = %s.object(tmp).param_update()}"' %(loop, cmd, self.hoc_label),
                            'success = execute1(cmd)']
        hoc_execute(hoc_commands, "--- Population[%s].__set()__ ---" %self.label)

    def tset(self,parametername,valueArray):
        """
        'Topographic' set. Sets the value of parametername to the values in
        valueArray, which must have the same dimensions as the Population.
        """
        if self.dim == valueArray.shape:
            values = numpy.reshape(valueArray,valueArray.size)
            values = values.take(numpy.array(self.gidlist)-self.gid_start) # take just the values for cells on this machine
            assert len(values) == len(self.gidlist)
            if isinstance(self.celltype, common.StandardCellType):
                parametername = self.celltype.translate({parametername: values[0]}).keys()[0]
            hoc_commands = []
            fmt = '%s.object(%s).%s = %s' % (self.hoc_label, "%d", parametername, "%g")
            for i,val in enumerate(values):
                try:
                    hoc_commands += [fmt % (i,val),
                                     'success = %s.object(%d).param_update()' % (self.hoc_label, i)]
                except TypeError:
                    raise common.InvalidParameterValueError, "%s is not a numeric value" % str(val)
            hoc_execute(hoc_commands, "--- Population[%s].__tset()__ ---" %self.label)
        else:
            raise common.InvalidDimensionsError

    def rset(self,parametername,rand_distr):
        """
        'Random' set. Sets the value of parametername to a value taken from
        rand_distr, which should be a RandomDistribution object.
        """
        if isinstance(rand_distr.rng, NativeRNG):
            if isinstance(self.celltype, common.StandardCellType):
                parametername = self.celltype.translate({parametername: 0}).keys()[0]
            paramfmt = "%g,"*len(rand_distr.parameters); paramfmt = paramfmt.strip(',')
            distr_params = paramfmt % tuple(rand_distr.parameters)
            hoc_commands = ['rng = new Random(%d)' % 0 or distribution.rng.seed,
                            'tmp = rng.%s(%s)' % (rand_distr.name,distr_params)]
            # We do the loop in hoc, to speed up the code
            loop = "for tmp = 0, %d" %(len(self.gidlist)-1)
            cmd = '%s.object(tmp).%s = rng.repick()' % (self.hoc_label, parametername)
            hoc_commands += ['cmd="%s { %s success = %s.object(tmp).param_update()}"' %(loop, cmd, self.hoc_label),
                             'success = execute1(cmd)']
            hoc_execute(hoc_commands, "--- Population[%s].__rset()__ ---" %self.label)   
        else:
            rarr = rand_distr.next(n=self.size)
            rarr = rarr.reshape(self.dim)
            hoc_comment("--- Population[%s].__rset()__ --- " %self.label)
            self.tset(parametername, rarr)

    def _call(self,methodname,arguments):
        """
        Calls the method methodname(arguments) for every cell in the population.
        e.g. p.call("set_background","0.1") if the cell class has a method
        set_background().
        """
        raise Exception("Method not yet implemented")
        ## Not sure this belongs in the API, because cell classes only have
        ## parameters/attributes, not methods.

    def _tcall(self,methodname,objarr):
        """
        `Topographic' call. Calls the method methodname() for every cell in the 
        population. The argument to the method depends on the coordinates of the
        cell. objarr is an array with the same dimensions as the Population.
        e.g. p.tcall("memb_init",vinitArray) calls
        p.cell[i][j].memb_init(vInitArray[i][j]) for all i,j.
        """
        raise Exception("Method not yet implemented")    

    def __record(self,record_what,record_from=None,rng=None):
        """
        Private method called by record() and record_v().
        """
        global myid
        hoc_commands = []
        fixed_list=False

        if isinstance(record_from,list): #record from the fixed list specified by user
            fixed_list=True
        elif record_from is None: # record from all cells:
            record_from = self.gidlist
        elif isinstance(record_from,int): # record from a number of cells, selected at random  
            # Each node will record N/nhost cells...
            nrec = int(record_from/nhost)
            if rng:
                record_from = rng.permutation(self.gidlist)
            else:
                record_from = numpy.random.permutation(self.gidlist)
            # Taken as random in self.gidlist
            record_from = record_from[0:nrec]
            record_from = numpy.array(record_from) # is this line necessary?
        else:
            raise Exception("record_from must be either a list of cells or the number of cells to record from")
        # record_from is now a list or numpy array

        suffix = ''*(record_what=='spiketimes') + '_v'*(record_what=='vtrace')
        for id in record_from:
            if id in self.gidlist:
                hoc_commands += ['tmp = %s.object(%d).record%s(1)' % (self.hoc_label,self.gidlist.index(id),suffix)]

        # note that self.record_from is not the same on all nodes, like self.gidlist, for example.
        self.record_from[record_what].update(Set(record_from))
        hoc_commands += ['objref record_from']
        hoc_execute(hoc_commands)

        # Then we have to send the lists of local recorded objects to the master node,
        # but only if the list has not been specified by the user.
        if fixed_list is False:
            if myid != 0:  # on slave nodes
                hoc_commands = ['record_from = new Vector()']
                for id in self.record_from[record_what]:
                    if id in self.gidlist:
                        hoc_commands += ['record_from = record_from.append(%d)' %id]
                hoc_commands += ['tmp = pc.post("%s.record_from[%s].node[%d]", record_from)' %(self.hoc_label, record_what, myid)]
                hoc_execute(hoc_commands, "   (Posting recorded cells)")
            else:          # on the master node
                for id in range (1, nhost):
                    hoc_commands = ['record_from = new Vector()']
                    hoc_commands += ['tmp = pc.take("%s.record_from[%s].node[%d]", record_from)' %(self.hoc_label, record_what, id)]
                    hoc_execute(hoc_commands)
                    for j in xrange(h.record_from.size()):
                        self.record_from[record_what].add(h.record_from.x[j])

    def record(self,record_from=None,rng=None):
        """
        If record_from is not given, record spikes from all cells in the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        hoc_comment("--- Population[%s].__record()__ ---" %self.label)
        self.__record('spiketimes',record_from,rng)

    def record_v(self,record_from=None,rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        hoc_comment("--- Population[%s].__record_v()__ ---" %self.label)
        self.__record('vtrace',record_from,rng)

    def __print(self,print_what,filename,num_format,gather,header=None):
        """Private method used by printSpikes() and print_v()."""
        global myid
        vector_operation = ''
        if print_what == 'spiketimes':
            vector_operation = '.where("<=", tstop)'
        if gather and myid != 0: # on slave nodes, post data
            hoc_commands = []
            for id in self.record_from[print_what]:
                if id in self.gidlist:
                    hoc_commands += ['tmp = pc.post("%s[%d].%s",%s.object(%d).%s%s)' % (self.hoc_label,id,
                                                                                        print_what,
                                                                                        self.hoc_label,
                                                                                        self.gidlist.index(id),
                                                                                        print_what,
                                                                                        vector_operation)]
            hoc_execute(hoc_commands,"--- Population[%s].__print()__ --- [Post objects to master]" %self.label)

        if myid==0 or not gather:
            hoc_commands = ['objref fileobj',
                            'fileobj = new File()',
                            'tmp = fileobj.wopen("%s")' % filename]
            if header:
                hoc_commands += ['tmp = fileobj.printf("%s\\n")' % header]
            if gather:
                hoc_commands += ['objref gatheredvec']
            padding = self.fullgidlist[0]
            for id in self.record_from[print_what]:
                addr = self.locate(id)
                #hoc_commands += ['fmt = "%s\\t%s\\n"' % (num_format, "\\t".join([str(j) for j in addr]))]
                hoc_commands += ['fmt = "%s\\t%d\\n"' % (num_format, id-padding)]
                if id in self.gidlist:
                    hoc_commands += ['tmp = %s.object(%d).%s%s.printf(fileobj,fmt)' % (self.hoc_label,
                                                                                       self.gidlist.index(id),
                                                                                       print_what,
                                                                                       vector_operation)]
                elif gather: 
                    hoc_commands += ['gatheredvec = new Vector()']
                    hoc_commands += ['tmp = pc.take("%s[%d].%s",gatheredvec)' % (self.hoc_label,id,print_what),
                                     'tmp = gatheredvec.printf(fileobj,fmt)']
            hoc_commands += ['tmp = fileobj.close()']
            hoc_execute(hoc_commands,"--- Population[%s].__print()__ ---" %self.label)

    def printSpikes(self,filename,gather=True, compatible_output=True):
        """
        Writes spike times to file.
        If compatible_output is True, the format is "spiketime cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        This allows easy plotting of a `raster' plot of spiketimes, with one
        line for each cell.
        The timestep and number of data points per cell is written as a header,
        indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        spike files.
        
        If gather is True, the file will only be created on the master node,
        otherwise, a file will be written on each node.
        """        
        hoc_comment("--- Population[%s].__printSpikes()__ ---" %self.label)
        header = "# %d" %self.dim[0]
        for dimension in list(self.dim)[1:]:
                header = "%s\t%d" %(header,dimension)
        self.__print('spiketimes',filename,"%.2f",gather, header)

    def print_v(self,filename, gather=True, compatible_output=True):
        """
        Write membrane potential traces to file.
        If compatible_output is True, the format is "v cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        This allows easy plotting of a `raster' plot of spiketimes, with one
        line for each cell.
        The timestep and number of data points per cell is written as a header,
        indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        voltage files.
        """
        #tstop = HocToPy.get('tstop','float')
        tstop = h.tstop
        header = "# dt = %f\\n# n = %d\\n" % (dt,int(tstop/dt))
        header = "%s# %d" %(header,self.dim[0])
        for dimension in list(self.dim)[1:]:
                header = "%s\t%d" %(header,dimension)
        hoc_comment("--- Population[%s].__print_v()__ ---" %self.label)
        self.__print('vtrace',filename,"%.4g",gather,header)

    def getSpikes(self, gather=True):
        """
        Returns a numpy array of the spikes of the population

        Useful for small populations, for example for single neuron Monte-Carlo.

        NOTE: getSpikes or printSpikes should be called only once per run,
        because they mangle simulator recorder files.
        """
        # This is a bit of a hack implemetation
        tmpfile = "neuron_tmpfile" # should really use tempfile module
        self.__print('spiketimes', tmpfile, "%.2f", gather)
        f = open(tmpfile, 'r')
        lines = [line for line in f.read().split('\n') if line] # remove blank lines
        line2spike = lambda s: (int(s[1]), float(s[0]))
        spikes = numpy.array([line2spike(line.split()) for line in lines])
        f.close()
        os.remove("neuron_tmpfile")
        return spikes
        
    def meanSpikeCount(self,gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        global myid
        # If gathering, each node posts the number of spikes and
        # the number of cells to the master node (myid == 0)
        if gather and myid != 0:
            hoc_commands = []
            nspikes = 0;ncells  = 0
            for id in self.record_from['spiketimes']:
                if id in self.gidlist:
                    #nspikes += HocToPy.get('%s.object(%d).spiketimes.size()' %(self.hoc_label, self.gidlist.index(id)),'int')
                    nspikes += getattr(h, self.hoc_label).object(self.gidlist.index(id)).spiketimes.size()
                    ncells  += 1
            hoc_commands += ['tmp = pc.post("%s.node[%d].nspikes",%d)' % (self.hoc_label,myid,nspikes)]
            hoc_commands += ['tmp = pc.post("%s.node[%d].ncells",%d)' % (self.hoc_label,myid,ncells)]    
            hoc_execute(hoc_commands,"--- Population[%s].__meanSpikeCount()__ --- [Post spike count to master]" %self.label)
            return 0

        if myid==0 or not gather:
            nspikes = 0.0; ncells = 0.0
            hoc_execute(["nspikes = 0", "ncells = 0"])
            for id in self.record_from['spiketimes']:
                if id in self.gidlist:
                    nspikes += getattr(h, self.hoc_label).object(self.gidlist.index(id)).spiketimes.size()
                    ncells  += 1
            if gather:
                for id in range(1,nhost):
                    hoc_execute(['tmp = pc.take("%s.node[%d].nspikes",&nspikes)' % (self.hoc_label,id)])
                    #nspikes += HocToPy.get('nspikes','int')
                    nspikes += int(h.nspikes)
                    hoc_execute(['tmp = pc.take("%s.node[%d].ncells",&ncells)' % (self.hoc_label,id)])
                    #ncells  += HocToPy.get('ncells','int')
                    ncells += int(h.ncells)
            return float(nspikes/ncells)

    def randomInit(self,rand_distr):
        """
        Sets initial membrane potentials for all the cells in the population to
        random values.
        """
        hoc_comment("--- Population[%s].__randomInit()__ ---" %self.label)
        self.rset("v_init",rand_distr)


class Projection(common.Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    
    nProj = 0
    
    def __init__(self,presynaptic_population,postsynaptic_population,method='allToAll',
                 method_parameters=None,source=None,target=None,
                 synapse_dynamics=None, label=None,rng=None):
        """
        presynaptic_population and postsynaptic_population - Population objects.
        
        source - string specifying which attribute of the presynaptic cell signals action potentials
        
        target - string specifying which synapse on the postsynaptic cell to connect to
        If source and/or target are not given, default values are used.
        
        method - string indicating which algorithm to use in determining connections.
        Allowed methods are 'allToAll', 'oneToOne', 'fixedProbability',
        'distanceDependentProbability', 'fixedNumberPre', 'fixedNumberPost',
        'fromFile', 'fromList'
        
        method_parameters - dict containing parameters needed by the connection method,
        although we should allow this to be a number or string if there is only
        one parameter.
        
        synapse_dynamics - ...
        
        rng - since most of the connection methods need uniform random numbers,
        it is probably more convenient to specify a RNG object here rather
        than within method_parameters, particularly since some methods also use
        random numbers to give variability in the number of connections per cell.
        """
        global _min_delay
        common.Projection.__init__(self,presynaptic_population,postsynaptic_population,method,
                                   method_parameters,source,target,synapse_dynamics,label,rng)
        self.connections = []
        if not label:
            self.label = 'projection%d' % Projection.nProj
        self.hoc_label = self.label.replace(" ","_")
        if not rng:
            self.rng = numpy.random.RandomState()
        hoc_commands = ['objref %s' % self.hoc_label,
                        '%s = new List()' % self.hoc_label]
        self.synapse_type = target
        self._syn_objref = _translate_synapse_type(self.synapse_type)

        if isinstance(method, str):
            connection_method = getattr(self,'_%s' % method)   
            hoc_commands += connection_method(method_parameters)
        elif isinstance(method,common.Connector):
            hoc_commands += method.connect(self)
            # delays should already be set to min_delay
        hoc_execute(hoc_commands, "--- Projection[%s].__init__() ---" %self.label)
        
        # By defaut, we set all the delays to min_delay, except if
        # the Projection data have been loaded from a file or a list.
        # This should already have been done if using a Connector object
        if isinstance(method, str) and (method != 'fromList') and (method != 'fromFile'):
            self.setDelays(_min_delay)
        
        ## Deal with synaptic plasticity
        if self.long_term_plasticity_mechanism is not None:
            self._setupSTDP(self._plasticity_model, self._stdp_parameters)
            
        Projection.nProj += 1

    def __len__(self):
        """Return the total number of connections."""
        return len(self.connections)

    # --- Connection methods ---------------------------------------------------
    
    def __connect(self,src,tgt):
        """
        Write hoc commands to connect a single pair of neurons.
        """
        cmdlist = ['nc = pc.gid_connect(%d,%s.object(%d).%s)' % (src,
                                                                 self.post.hoc_label,
                                                                 self.post.gidlist.index(tgt),
                                                                 self._syn_objref),
                'tmp = %s.append(nc)' % self.hoc_label]
        self.connections.append((src,tgt))
        return cmdlist
    
    def _allToAll(self,parameters=None):
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
        
    def _oneToOne(self,parameters=None):
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
    
    def _fixedProbability(self,parameters):
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

    def _distanceDependentProbability(self,parameters):
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
    
    def _fixedNumberPre(self,parameters):
        """Each presynaptic cell makes a fixed number of connections."""
        allow_self_connections = True
        if type(parameters) == types.IntType:
            n = parameters
            assert n > 0
            fixed = True
        elif type(parameters) == types.DictType:
            if parameters.has_key('n'): # all cells have same number of connections
                n = int(parameters['n'])
                assert n > 0
                fixed = True
            elif parameters.has_key('rand_distr'): # number of connections per cell follows a distribution
                rand_distr = parameters['rand_distr']
                assert isinstance(rand_distr,RandomDistribution)
                fixed = False
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
        elif isinstance(parameters, RandomDistribution):
            rand_distr = parameters
            fixed = False
        else:
            raise Exception("Invalid argument type: should be an integer, dictionary or RandomDistribution object.")
        hoc_commands = []
        
        if self.rng:
            rng = self.rng
        else:
            rng = numpy.random
        for src in self.pre.gidlist:            
            # pick n neurons at random
            if not fixed:
                n = rand_distr.next()
            for tgt in rng.permutation(self.post.gidlist)[0:n]:
                if allow_self_connections or (src != tgt):
                    hoc_commands += self.__connect(src,tgt)
        return hoc_commands
            
    def _fixedNumberPost(self,parameters):
        """Each postsynaptic cell receives a fixed number of connections."""
        allow_self_connections = True
        if type(parameters) == types.IntType:
            n = parameters
            assert n > 0
            fixed = True
        elif type(parameters) == types.DictType:
            if parameters.has_key('n'): # all cells have same number of connections
                n = int(parameters['n'])
                assert n > 0
                fixed = True
            elif parameters.has_key('rand_distr'): # number of connections per cell follows a distribution
                rand_distr = parameters['rand_distr']
                assert isinstance(rand_distr,RandomDistribution)
                fixed = False
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
        elif isinstance(parameters, RandomDistribution):
            rand_distr = parameters
            fixed = False
        else:
            raise Exception("Invalid argument type: should be an integer, dictionary or RandomDistribution object.")
        hoc_commands = []
        
        if self.rng:
            rng = self.rng
        else:
            rng = numpy.random
        for tgt in self.post.gidlist:            
            # pick n neurons at random
            if not fixed:
                n = rand_distr.next()
            for src in rng.permutation(self.pre.gidlist)[0:n]:
                if allow_self_connections or (src != tgt):
                    hoc_commands += self.__connect(src,tgt)
        return hoc_commands
    
    def _fromFile(self,parameters):
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
            input_tuples.append((eval(src),eval(tgt),float(w),float(d)))
        f.close()
        return self._fromList(input_tuples)
    
    def _fromList(self,conn_list):
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
            hoc_commands += self.__connect(src,tgt)
            hoc_commands += ['%s.object(%d).weight = %f' % (self.hoc_label, i, float(weight)), 
                             '%s.object(%d).delay = %f'  % (self.hoc_label, i, float(delay))]
        return hoc_commands
    
    # --- Methods for setting connection parameters ----------------------------
    
    def setWeights(self,w):
        """
        w can be a single number, in which case all weights are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        Weights should be in nA for current-based and µS for conductance-based
        synapses.
        """
        if isinstance(w,float) or isinstance(w,int):
            loop = ['for tmp = 0, %d {' % (len(self)-1), 
                        '%s.object(tmp).weight = %f ' % (self.hoc_label, float(w)),
                    '}']
            hoc_code = "".join(loop)
            hoc_commands = [ 'cmd = "%s"' % hoc_code,
                             'success = execute1(cmd)']
        elif isinstance(w,list) or isinstance(w,numpy.ndarray):
            hoc_commands = []
            assert len(w) == len(self), "List of weights has length %d, Projection %s has length %d" % (len(w),self.label,len(self))
            for i,weight in enumerate(w):
                hoc_commands += ['%s.object(%d).weight = %f' % (self.hoc_label, i, weight)]
        else:
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")
        hoc_execute(hoc_commands, "--- Projection[%s].__setWeights__() ---" % self.label)
        
    def randomizeWeights(self,rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        # If we have a native rng, we do the loops in hoc. Otherwise, we do the loops in
        # Python
        if isinstance(rand_distr.rng, NativeRNG):
            paramfmt = "%f,"*len(rand_distr.parameters); paramfmt = paramfmt.strip(',')
            distr_params = paramfmt % tuple(rand_distr.parameters)
            hoc_commands = ['rng = new Random(%d)' % 0 or distribution.rng.seed,
                            'tmp = rng.%s(%s)' % (rand_distr.name,distr_params)]
                            
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
        
    def setDelays(self,d):
        """
        d can be a single number, in which case all delays are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        """
        if isinstance(d,float) or isinstance(d,int):
            if d < get_min_delay():
                raise Exception("Delays must be greater than or equal to the minimum delay, currently %g ms" % get_min_delay())
            loop = ['for tmp = 0, %d {' %(len(self)-1), 
                        '%s.object(tmp).delay = %f ' %(self.hoc_label, float(d)),
                    '}']
            hoc_code = "".join(loop)
            hoc_commands = [ 'cmd = "%s"' %hoc_code,
                             'success = execute1(cmd)']
        elif isinstance(d,list) or isinstance(d,numpy.ndarray):
            # need check for min_delay here
            hoc_commands = []
            assert len(d) == len(self), "List of delays has length %d, Projection %s has length %d" % (len(d),self.label,len(self))
            for i,delay in enumerate(d):
                hoc_commands += ['%s.object(tmp).delay = %f' % (self.hoc_label,delay)]
        else:
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")
        hoc_execute(hoc_commands, "--- Projection[%s].__setDelays__() ---" %self.label)
        
    def randomizeDelays(self,rand_distr):
        """
        Set delays to random values taken from rand_distr.
        """   
        # If we have a native rng, we do the loops in hoc. Otherwise, we do the loops in
        # Python  
        if isinstance(rand_distr.rng, NativeRNG):
            paramfmt = "%f,"*len(rand_distr.parameters); paramfmt = paramfmt.strip(',')
            distr_params = paramfmt % tuple(rand_distr.parameters)
            hoc_commands = ['rng = new Random(%d)' % 0 or distribution.rng.seed,
                            'tmp = rng.%s(%s)' % (rand_distr.name,distr_params)]
            loop = ['for tmp = 0, %d {' %(len(self)-1), 
                        '%s.object(tmp).delay = rng.repick() ' %(self.hoc_label),
                    '}']
            hoc_code = "".join(loop)
            hoc_commands += ['cmd = "%s"' %hoc_code,
                             'success = execute1(cmd)']    
        else:
            hoc_commands = [] 
            for i in xrange(len(self)):
                hoc_commands += ['%s.object(%d).delay = %f' % (self.hoc_label, i, float(rand_distr.next()))]
        hoc_execute(hoc_commands, "--- Projection[%s].__randomizeDelays__() ---" %self.label)
        
    def setTopographicDelays(self,delay_rule,rand_distr=None,mask=None,scale_factor=1.0):
        """
        Set delays according to a connection rule expressed in delay_rule, based
        on the delay distance 'd' and an (optional) rng 'rng'. For example,
        the rule can be "rng*d + 0.5", with "a" extracted from the rng and
        d being the distance.
        """  
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
                            'tmp = rng.%s(%s)' % (rand_distr.name,distr_params)]
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
        
    def setThreshold(self,threshold):
        """
        Where the emission of a spike is determined by watching for a
        threshold crossing, set the value of this threshold.
        """
        # This is a bit tricky, because in NEST the spike threshold is a
        # property of the cell model, whereas in NEURON it is a property of the
        # connection (NetCon).
        raise Exception("Method not yet implemented")
    
    
    # --- Methods relating to synaptic plasticity ------------------------------
    
    def _setupSTDP(self, stdp_model, parameterDict):
        """Set-up STDP."""
        ddf = self.long_term_plasticity_mechanism.dendritic_delay_fraction
        # Define the objref to handle plasticity
        hoc_commands =  ['objref %s_wa[%d]'      %(self.hoc_label,len(self)),
                         'objref %s_pre2wa[%d]'  %(self.hoc_label,len(self)),
                         'objref %s_post2wa[%d]' %(self.hoc_label,len(self))]
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
                '%s_post2wa[%d] = pc.gid_connect(%d, %s_wa[%d])' %(self.hoc_label, i, tgt, self.hoc_label, i),
                '%s_post2wa[%d].threshold = 1' %(self.hoc_label, i),
                '%s_post2wa[%d].delay = %s.object(%d).delay * %g' % (self.hoc_label, i, self.hoc_label, i, ddf),
                '%s_post2wa[%d].weight = -1' % (self.hoc_label, i),
                'setpointer %s_wa[%d].wsyn, %s.object(%d).weight' %(self.hoc_label, i,self.hoc_label,i)]
            # then update the parameters
            for param,val in parameterDict.items():
                hoc_commands += ['%s_wa[%d].%s = %f' % (self.hoc_label, i, param, val)]
            
        hoc_execute(hoc_commands, "--- Projection[%s].__setupSTDP__() ---" %self.label)  
    
    
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
            raise Exception("Not yet implemented")
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
    
    def saveConnections(self,filename,gather=False):
        """Save connections to file in a format suitable for reading in with the
        'fromFile' method."""
        hoc_comment("--- Projection[%s].__saveConnections__() ---" %self.label)  
        f = open(filename,'w',10000)
        for i in xrange(len(self)):
            src = self.connections[i][0]
            tgt = self.connections[i][1]
            line = "%s%s\t%s%s\t%g\t%g\n" % (self.pre.hoc_label,
                                     self.pre.locate(src),
                                     self.post.hoc_label,
                                     self.post.locate(tgt),
                                     getattr(h,self.hoc_label).object(i).weight[0],
                                     getattr(h,self.hoc_label).object(i).delay)
            line = line.replace('(','[').replace(')',']')
            f.write(line)
        f.close()
    
    def printWeights(self,filename,format='list',gather=True):
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
                #weight = HocToPy.get('%s.object(%d).weight' % (self.hoc_label,i),'float')
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
                    #weight = "%f\n" %HocToPy.get('%s.object(%d).weight' % (self.hoc_label,i),'float')
                    weight = getattr(h, self.hoc_label).object(i).weight[0]
                    f.write("%f\n" % weight)
            elif format == 'array':
                weights = numpy.zeros((len(self.pre),len(self.post)), 'float')
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
