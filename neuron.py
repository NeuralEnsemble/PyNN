"""
nrnpython implementation of the PyNN API.

This is an attempt at a parallel-enabled implementation.                                                                
$Id$
"""
__version__ = "$Revision$"

import hoc
from pyNN.random import *
from math import *
from pyNN import __path__, common
import os.path
import types
import sys
import numpy
import logging

gid           = 0
ncid          = 0
gidlist       = []
vfilelist     = {}
spikefilelist = {}
dt            = 0.1

# ==============================================================================
#   Utility classes
# ==============================================================================

class ID(common.ID):
    """
    This class is experimental. The idea is that instead of storing ids as
    integers, we store them as ID objects, which allows a syntax like:
      p[3,4].set('tau_m',20.0)
    where p is a Population object. The question is, how big a memory/performance
    hit is it to replace integers with ID objects?
    """
    
    def set(self,param,val=None):
        # We perform a call to the low-level function set() of the API.
        # If the cellclass is not defined in the ID object, we have an error (?) :
        if (self._cellclass == None):
            raise Exception("Unknown cellclass")
        else:
            #Otherwise we use the ID one. Nevertheless, here we have a small problem in the
            #parallel framework. Suppose a population is created, distributed among
            #several nodes. Then a call like cell[i,j].set() should be performed only on the
            #node who owns the cell. To do that, if the node doesn't have the cell, a call to set()
            #do nothing...
            ##if self._hocname != None:
            ##    set(self,self._cellclass,param,val, self._hocname)
            set(self,self._cellclass,param,val)
    
    def get(self,param):
        #This function should be improved, with some test to translate
        #the parameter according to the cellclass
        #We have here the same problem that with set() in the parallel framework
        if self._hocname != None:
            return HocToPy.get('%s.%s' %(self._hocname, param),'float')
    
    # Fonctions used only by the neuron version of pyNN, to optimize the
    # creation of networks
    def setHocName(self, name):
    	self._hocname = name

    def getHocName(self):
    	return self._hocname
    

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
        else:
            hoc_commands += ['argvar%d = %f' % (nvar,item)]
            argstr += 'argvar%d, ' % nvar
            nvar += 1
    return hoc_commands, argstr.strip().strip(',')

def _translate_synapse_type(synapse_type):
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
        syn_objref = "esyn"
    return syn_objref

def checkParams(param,val=None):
    """Check parameters are of valid types, normalise the different ways of
       specifying parameters and values by putting everything in a dict.
       Called by set() and Population.set()."""
    if isinstance(param,str):
        if isinstance(val,float) or isinstance(val,int):
            paramDict = {param:float(val)}
        elif isinstance(val,(str, list)):
            paramDict = {param:val}
        else:
            raise common.InvalidParameterValueError
    elif isinstance(param,dict):
        paramDict = param
    else:
        raise common.InvalidParameterValueError
    return paramDict



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
#   Standard cells
# ==============================================================================
 
class IF_curr_alpha(common.IF_curr_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
    
    translations = {
        'tau_m'     : ('tau_m'    , "parameters['tau_m']"),
        'cm'        : ('CM'       , "parameters['cm']"),
        'v_rest'    : ('v_rest'   , "parameters['v_rest']"),
        'v_thresh'  : ('v_thresh' , "parameters['v_thresh']"),
        'v_reset'   : ('v_reset'  , "parameters['v_reset']"),
        'tau_refrac': ('t_refrac' , "parameters['tau_refrac']"),
        'i_offset'  : ('i_offset' , "parameters['i_offset']"),
        'tau_syn'   : ('tau_syn'  , "parameters['tau_syn']"),
        'v_init'    : ('v_init'   , "parameters['v_init']"),
    }
    hoc_name = "StandardIF"
    
    def __init__(self,parameters):
        common.IF_curr_alpha.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate(self.parameters)
        self.parameters['syn_type']  = 'current'
        self.parameters['syn_shape'] = 'alpha'

class IF_curr_exp(common.IF_curr_exp):
    """Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses."""
    
    translations = {
        'tau_m'     : ('tau_m'    , "parameters['tau_m']"),
        'cm'        : ('CM'       , "parameters['cm']"),
        'v_rest'    : ('v_rest'   , "parameters['v_rest']"),
        'v_thresh'  : ('v_thresh' , "parameters['v_thresh']"),
        'v_reset'   : ('v_reset'  , "parameters['v_reset']"),
        'tau_refrac': ('t_refrac' , "parameters['tau_refrac']"),
        'i_offset'  : ('i_offset' , "parameters['i_offset']"),
        'tau_syn_E' : ('tau_e'    , "parameters['tau_syn_E']"),
        'tau_syn_I' : ('tau_i'    , "parameters['tau_syn_I']"),
        'v_init'    : ('v_init'   , "parameters['v_init']"),
    }
    hoc_name = "StandardIF"
    
    def __init__(self,parameters):
        common.IF_curr_exp.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)
        self.parameters['syn_type']  = 'current'
        self.parameters['syn_shape'] = 'exp'


class IF_cond_alpha(common.IF_cond_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
    
    translations = {
        'tau_m'     : ('tau_m'    , "parameters['tau_m']"),
        'cm'        : ('CM'       , "parameters['cm']"),
        'v_rest'    : ('v_rest'   , "parameters['v_rest']"),
        'v_thresh'  : ('v_thresh' , "parameters['v_thresh']"),
        'v_reset'   : ('v_reset'  , "parameters['v_reset']"),
        'tau_refrac': ('t_refrac' , "parameters['tau_refrac']"),
        'i_offset'  : ('i_offset' , "parameters['i_offset']"),
        'tau_syn_E' : ('tau_e'    , "parameters['tau_syn_E']"),
        'tau_syn_I' : ('tau_i'    , "parameters['tau_syn_I']"),
        'v_init'    : ('v_init'   , "parameters['v_init']"),
        'e_rev_E'   : ('e_e'      , "parameters['e_rev_E']"),
        'e_rev_I'   : ('e_i'      , "parameters['e_rev_I']")
    }
    hoc_name = "StandardIF"
    
    def __init__(self,parameters):
        common.IF_cond_alpha.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate(self.parameters)
        self.parameters['syn_type']  = 'conductance'
        self.parameters['syn_shape'] = 'alpha'

class SpikeSourcePoisson(common.SpikeSourcePoisson):
    """Spike source, generating spikes according to a Poisson process."""

    translations = {
        'start'    : ('start'  , "parameters['start']"),
        'rate'     : ('number' , "int((parameters['rate']/1000.0)*parameters['duration'])"),
        'duration' : ('number' , "int((parameters['rate']/1000.0)*parameters['duration'])")
    }
    hoc_name = 'SpikeSource'
   
    def __init__(self,parameters):
        common.SpikeSourcePoisson.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)
        self.parameters['source_type'] = 'NetStim'    
        self.parameters['noise'] = 1

    def translate(self,parameters):
        translated_parameters = common.SpikeSourcePoisson.translate(self,parameters)
        if parameters.has_key('rate') and parameters['rate'] != 0:
            translated_parameters['interval'] = 1000.0/parameters['rate']
        return translated_parameters

class SpikeSourceArray(common.SpikeSourceArray):
    """Spike source generating spikes at the times given in the spike_times array."""

    translations = {
        'spike_times' : ('spiketimes' , "parameters['spike_times']"),
    }
    hoc_name = 'SpikeSource'
    
    def __init__(self,parameters):
        common.SpikeSourceArray.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)  
        self.parameters['source_type'] = 'VecStim'
        
                        
# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1,min_delay=0.1,max_delay=0.1,debug=False):
    """Should be called at the very beginning of a script."""
    global dt, nhost, myid, _min_delay, logger
    dt = timestep
    _min_delay = min_delay
    
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
    hoc_commands = [
        'tmp = xopen("%s")' % os.path.join(__path__[0],'hoc','standardCells.hoc'),
        'tmp = xopen("%s")' % os.path.join(__path__[0],'hoc','odict.hoc'),
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
        
    hoc_execute(hoc_commands,"--- setup() ---")
    nhost = HocToPy.get('pc.nhost()','int')
    if nhost < 2:
        nhost = 1; myid = 0
    else:
        myid = HocToPy.get('pc.id()','int')
    print "\nHost #%d of %d" % (myid+1, nhost)
    
    return int(myid)

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    global logfile, myid #, vfilelist, spikefilelist
    hoc_commands = []
    if len(vfilelist) > 0:
        hoc_commands = ['objref fileobj',
                        'fileobj = new File()']
        for filename,cell_list in vfilelist.items():
            hoc_commands += ['tmp = fileobj.wopen("%s")' % filename]
            tstop = HocToPy.get('tstop','float')
            header = "# dt = %f\\n# n = %d\\n" % (dt,int(tstop/dt))
            for cell in cell_list:
                hoc_commands += ['fmt = "%s\\t%d\\n"' % ("%.6g",cell),
                                 'tmp = fileobj.printf("%s")' % header,
                                 'tmp = cell%d.vtrace.printf(fileobj,fmt)' % cell]
            hoc_commands += ['tmp = fileobj.close()']
    if len(spikefilelist) > 0:
        hoc_commands += ['objref fileobj',
                        'fileobj = new File()']
        for filename,cell_list in spikefilelist.items():
            hoc_commands += ['tmp = fileobj.wopen("%s")' % filename]
            for cell in cell_list:
                hoc_commands += ['fmt = "%s\\t%d\\n"' % ("%.2f",cell),
                                 #'tmp = fileobj.printf("# cell%d\\n")' % cell,
                                 'pc.cell%d.spiketimes.printf(fileobj,fmt)' % cell]
            hoc_commands += ['tmp = fileobj.close()']
    hoc_commands += ['tmp = pc.runworker()',
                     'tmp = pc.done()']
    hoc_execute(hoc_commands,"--- end() ---")
    hoc.execute('tmp = quit()')
    logging.info("Finishing up with NEURON.")
    sys.exit(0)

def run(simtime):
    """Run the simulation for simtime ms."""
    hoc_commands = ['tstop = %f' %simtime,
                    'print "dt        = %f"' %dt,
                    'print "tstop     = %f"' % simtime,
                    'print "min delay = ", pc.set_maxstep(100)',
                    'tmp = finitialize()',
                    'tmp = pc.psolve(%f)' %simtime]
    hoc_execute(hoc_commands,"--- run() ---")

def setRNGseeds(seedList):
    """Globally set rng seeds."""
    pass # not applicable to NEURON?

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

def create(cellclass,paramDict=None,n=1):
    """
    Create n cells all of the same type.
    If n > 1, return a list of cell ids/references.
    If n==1, return just the single id.
    """
    global gid, gidlist, nhost, myid
    
    assert n > 0, 'n must be a positive integer'
    if isinstance(cellclass, type):
        celltype = cellclass(paramDict)
        hoc_name = celltype.hoc_name
        hoc_commands, argstr = _hoc_arglist([celltype.parameters])
    elif isinstance(cellclass,str):
        hoc_name = cellclass
        hoc_commands, argstr = _hoc_arglist([paramDict])
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
    cell_list = range(gid,gid+n)
    gid = gid+n
    if n == 1:
        cell_list = cell_list[0]
    return cell_list

def connect(source,target,weight=None,delay=None,synapse_type=None,p=1,rng=None):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise."""
    global ncid, gid, gidlist, _min_delay
    if type(source) != types.ListType:
        source = [source]
    if type(target) != types.ListType:
        target = [target]
    if weight is None:  weight = 0.0
    if delay  is None:  delay = _min_delay
    syn_objref = _translate_synapse_type(synapse_type)
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
                            hoc_commands += ['nc = pc.gid_connect(%d,pc.gid2cell(%d).%s)' % (src,tgt,syn_objref),
                                             'nc.delay = %g' % delay,
                                             'nc.weight = %g' % weight,
                                             'tmp = netconlist.append(nc)']
                            ncid += 1
    hoc_execute(hoc_commands, "--- connect(%s,%s) ---" % (str(source),str(target)))
    return range(nc_start,ncid)

def set(cells,cellclass,param,val=None): #,hocname=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value.
    cellclass must be supplied for doing translation of parameter names."""
    global gidlist
    
    paramDict = checkParams(param,val)

    if type(cellclass) == type and issubclass(cellclass, common.StandardCellType):
        paramDict = cellclass({}).translate(paramDict)
    if not isinstance(cells,list):
        cells = [cells]    
    hoc_commands = []
    for param,val in paramDict.items():
        if isinstance(val,str):
            ## If we know the hoc name of the object (set() applied to a population object), we use it
            #if (hocname != None):
            #    fmt = '%s.%s = "%s"'
            #else:
            #    fmt = 'cell%d.%s = "%s"'
            fmt = 'pc.gid2cell(%d).%s = "%s"'
        elif isinstance(val,list):
            cmds,argstr = _hoc_arglist([val])
            hoc_commands += cmds
            ## If we know the hoc name of the object (set() applied to a population object), we use it
            #if (hocname != None):
            #    fmt = '%s.%s = %s'
            #else:
            #    fmt = 'cell%d.%s = %s'
            fmt = 'pc.gid2cell(%d).%s = %s'
            val = argstr
        else:
            ## If we know the hoc name of the object (set() applied to a population object), we use it
            #if (hocname != None):
            #    fmt = '%s.%s = %g'
            #else:
            #    fmt = 'cell%d.%s = %g'
            fmt = 'pc.gid2cell(%d).%s = %g'
        for cell in cells:
            if cell in gidlist:
                ## If we know the hoc name of the object (set() applied to a population object), we use it
                #if (hocname != None):
                #    hoc_commands += [fmt % (hocname,param,val),
                #                     'tmp = %s.param_update()' %hocname]
                #else:
                #    hoc_commands += [fmt % (cell,param,val),
                #                     'tmp = cell%d.param_update()' %cell]
                hoc_commands += [fmt % (cell,param,val),
                                 'tmp = pc.gid2cell(%d).param_update()' % cell]
    hoc_execute(hoc_commands, "--- set() ---")

def record(source,filename, compatible_output=True):
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

def record_v(source,filename, compatible_output=True):
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
            hoc_commands += ['tmp = cell%d.record_v(1)' % src]
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
    
    def __init__(self,dims,cellclass,cellparams=None,label=None):
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
        #if self.ndim > 1:
        #    for i in range(1,self.ndim):
        #        if self.dim[i] != self.dim[0]:
        #            raise common.InvalidDimensionsError, "All dimensions must be the same size (temporary restriction)."

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
        self.record_from = { 'spiketimes': [], 'vtrace': [] }
        
        
        # Now the gid and cellclass are stored as instance of the ID class, which will allow a syntax like
        # p[i,j].set(param, val). But we have also to deal with positions : a population needs to know ALL the positions
        # of its cells, and not only those of the cells located on a particular node (i.e in self.gidlist). So
        # each population should store what we call a "fullgidlist" with the ID of all the cells in the populations 
        # (and therefore their positions)
        self.fullgidlist = [ID(i) for i in range(gid, gid+self.size) if i < gid+self.size]
        
        # self.gidlist is now derived from self.fullgidlist since it contains only the cells of the population located on
        # the node
        self.gidlist     = [self.fullgidlist[i+myid] for i in range(0, len(self.fullgidlist),nhost) if i < len(self.fullgidlist)-myid]
        self.gid_start   = gid

        # Write hoc commands
        hoc_commands += ['objref %s' % self.label,
                         '%s = new List()' % self.label]

        for cell_id in self.gidlist:
            hoc_commands += ['tmp = pc.set_gid2node(%d,%d)' % (cell_id,myid),
                             'cell = new %s(%s)' % (hoc_name,argstr),
                             #'nc = new NetCon(cell.source,nil)',
                             'tmp = cell.connect2target(nil,nc)',
                             'tmp = pc.cell(%d,nc)' % cell_id,
                             'tmp = %s.append(cell)' %(self.label)]       
        hoc_execute(hoc_commands, "--- Population[%s].__init__() ---" %self.label)
        Population.nPop += 1
        gid = gid+self.size

        # We add the gidlist of the population to the global gidlist
        gidlist += self.gidlist
        
        # By default, the positions of the cells are their coordinates, given by the locate()
        # method. Note that each node needs to know all the positions of all the cells 
        # in the population
        for cell_id in self.fullgidlist:
            cell_id.setCellClass(cellclass)
            cell_id.setPosition(self.locate(cell_id))
                    
        # On the opposite, each node has to know only the precise hocname of its cells, if we
        # want to be able to use the low level set() function
        for cell_id in self.gidlist:
            cell_id.setHocName("%s.o(%d)" %(self.label, self.gidlist.index(cell_id)))

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
        paramDict = checkParams(param,val)
        if isinstance(self.celltype, common.StandardCellType):
            paramDict = self.celltype.translate(paramDict)

        strfmt  = '%s.object(tmp).%s = "%s"' % (self.label,"%s","%s")
        numfmt  = '%s.object(tmp).%s = %s' % (self.label,"%s","%g")
        listfmt = '%s.object(tmp).%s = %s' % (self.label,"%s","%s")
        for param,val in paramDict.items():
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
            hoc_commands = ['cmd="%s { %s success = %s.object(tmp).param_update()}"' %(loop, cmd, self.label),
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
            fmt = '%s.object(%s).%s = %s' % (self.label, "%d", parametername, "%g")
            for i,val in enumerate(values):
                try:
                    hoc_commands += [fmt % (i,val),
                                     'success = %s.object(%d).param_update()' % (self.label, i)]
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
            cmd = '%s.object(tmp).%s = rng.repick()' % (self.label, parametername)
            hoc_commands += ['cmd="%s { %s success = %s.object(tmp).param_update()}"' %(loop, cmd, self.label),
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
                hoc_commands += ['tmp = %s.object(%d).record%s(1)' % (self.label,self.gidlist.index(id),suffix)]

        # note that self.record_from is not the same on all nodes, like self.gidlist, for example.
        self.record_from[record_what] += list(record_from)
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
                hoc_commands += ['tmp = pc.post("%s.record_from[%s].node[%d]", record_from)' %(self.label, record_what, myid)]
                hoc_execute(hoc_commands, "   (Posting recorded cells)")
            else:          # on the master node
                for id in range (1, nhost):
                    hoc_commands = ['record_from = new Vector()']
                    hoc_commands += ['tmp = pc.take("%s.record_from[%s].node[%d]", record_from)' %(self.label, record_what, id)]
                    hoc_execute(hoc_commands)
                    for j in xrange(HocToPy.get('record_from.size()', 'int')):
                        self.record_from[record_what] += [HocToPy.get('record_from.x[%d]' %j, 'int')]

    def record(self,record_from=None,rng=None):
        """
        If record_from is not given, record spikes from all cells in the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids (e.g., (i,j,k) tuple for a 3D population)
        of the cells to record.
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
        if gather and myid != 0: # on slave nodes, post data
            hoc_commands = []
            for id in self.record_from[print_what]:
                if id in self.gidlist:
                    hoc_commands += ['tmp = pc.post("%s[%d].%s",%s.object(%d).%s)' % (self.label,id,print_what,
                                                                                      self.label,
                                                                                      self.gidlist.index(id),
                                                                                      print_what)]
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
                    hoc_commands += ['tmp = %s.object(%d).%s.printf(fileobj,fmt)' % (self.label,self.gidlist.index(id),print_what)]
                elif gather: 
                    hoc_commands += ['gatheredvec = new Vector()']
                    hoc_commands += ['tmp = pc.take("%s[%d].%s",gatheredvec)' %(self.label,id,print_what),
                                     'tmp = gatheredvec.printf(fileobj,fmt)']
            hoc_commands += ['tmp = fileobj.close()']
            hoc_execute(hoc_commands,"--- Population[%s].__print()__ ---" %self.label)

    def printSpikes(self,filename,gather=True, compatible_output=True):
        """
        Prints spike times to file in the two-column format
        "spiketime cell_id" where cell_id is the index of the cell counting
        along rows and down columns (and the extension of that for 3-D).
        This allows easy plotting of a `raster' plot of spiketimes, with one
        line for each cell. This method requires that the cell class records
        spikes in a vector spiketimes.
        If gather is True, the file will only be created on the master node,
        otherwise, a file will be written on each node.
        """
        hoc_comment("--- Population[%s].__printSpikes()__ ---" %self.label)
	header = "# %d" %self.dim[0]
	for dimension in list(self.dim)[1:]:
	        header = "%s\t%d" %(header,dimension)
        self.__print('spiketimes',filename,"%.2f",gather, header)

    def print_v(self,filename,gather=True, compatible_output=True):
        """
        Write membrane potential traces to file.
        """
        tstop = HocToPy.get('tstop','float')
        header = "# dt = %f\\n# n = %d\\n" % (dt,int(tstop/dt))
        header = "%s# %d" %(header,self.dim[0])
        for dimension in list(self.dim)[1:]:
	        header = "%s\t%d" %(header,dimension)
        hoc_comment("--- Population[%s].__print_v()__ ---" %self.label)
        self.__print('vtrace',filename,"%.4g",gather,header)

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
                    nspikes += HocToPy.get('%s.object(%d).spiketimes.size()' %(self.label, self.gidlist.index(id)),'int')
                    ncells  += 1
            hoc_commands += ['tmp = pc.post("%s.node[%d].nspikes",%d)' % (self.label,myid,nspikes)]
            hoc_commands += ['tmp = pc.post("%s.node[%d].ncells",%d)' % (self.label,myid,ncells)]    
            hoc_execute(hoc_commands,"--- Population[%s].__meanSpikeCount()__ --- [Post spike count to master]" %self.label)
            return 0

        if myid==0 or not gather:
            nspikes = 0.0; ncells = 0.0
            hoc_execute(["nspikes = 0", "ncells = 0"])
            for id in self.record_from['spiketimes']:
                if id in self.gidlist:
                    nspikes += HocToPy.get('%s.object(%d).spiketimes.size()' % (self.label, self.gidlist.index(id)),'int')
                    ncells  += 1
            if gather:
                for id in range(1,nhost):
                    hoc_execute(['tmp = pc.take("%s.node[%d].nspikes",&nspikes)' % (self.label,id)])
                    nspikes += HocToPy.get('nspikes','int')
                    hoc_execute(['tmp = pc.take("%s.node[%d].ncells",&ncells)' % (self.label,id)])
                    ncells  += HocToPy.get('ncells','int')
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
    A container for all the connections between two populations, together with
    methods to set parameters of those connections, including of plasticity
    mechanisms.
    """
    
    nProj = 0
    
    def __init__(self,presynaptic_population,postsynaptic_population,method='allToAll',methodParameters=None,source=None,target=None,label=None,rng=None):
        """
        presynaptic_population and postsynaptic_population - Population objects.
        
        source - string specifying which attribute of the presynaptic cell signals action potentials
        
        target - string specifying which synapse on the postsynaptic cell to connect to
        If source and/or target are not given, default values are used.
        
        method - string indicating which algorithm to use in determining connections.
        Allowed methods are 'allToAll', 'oneToOne', 'fixedProbability',
        'distanceDependentProbability', 'fixedNumberPre', 'fixedNumberPost',
        'fromFile', 'fromList'
        
        methodParameters - dict containing parameters needed by the connection method,
        although we should allow this to be a number or string if there is only
        one parameter.
        
        rng - since most of the connection methods need uniform random numbers,
        it is probably more convenient to specify a RNG object here rather
        than within methodParameters, particularly since some methods also use
        random numbers to give variability in the number of connections per cell.
        """
        global _min_delay
        common.Projection.__init__(self,presynaptic_population,postsynaptic_population,method,methodParameters,source,target,label,rng)
        self.connections = []
        if not label:
            self.label = 'projection%d' % Projection.nProj
        if not rng:
            self.rng = numpy.random.RandomState()
        hoc_commands = ['objref %s' % self.label,
                        '%s = new List()' % self.label]
        connection_method = getattr(self,'_%s' % method)
        
        if target:
            hoc_commands += connection_method(methodParameters,synapse_type=target)
        else:
            hoc_commands += connection_method(methodParameters)
        hoc_execute(hoc_commands, "--- Projection[%s].__init__() ---" %self.label)
        
        # By defaut, we set all the delays to min_delay, except if
        # the Projection data have been loaded from a file or a list.
        if (method != 'fromList') and (method != 'fromFile'):
            self.setDelays(_min_delay)
        
        Projection.nProj += 1

    def __len__(self):
        """Return the total number of connections."""
        return len(self.connections)
     
    def _distance(self, presynaptic_population, postsynaptic_population, src, tgt):
        """
        Return the Euclidian distance between two cells. For the moment, we do
        a scaling between the two dimensions of the populations: the target
        population is scaled to the size of the source population."""
        dist = 0.0
        src_position = src.getPosition()
        tgt_position = tgt.getPosition()
        if (len(src_position) == len(tgt_position)):
            for i in xrange(len(src_position)):
                # We normalize the positions in each population and calculate the
                # Euclidian distance :
                #scaling = float(presynaptic_population.dim[i])/float(postsynaptic_population.dim[i])
                src_coord = float(src_position[i])
                tgt_coord = float(tgt_position[i])
            
                dist += float(src_coord-tgt_coord)*float(src_coord-tgt_coord)
        else:    
            raise Exception("Method _distance() not yet implemented for Populations with different sizes.")
        return sqrt(dist)
    

    # --- Connection methods ---------------------------------------------------
    
    def _allToAll(self,parameters=None,synapse_type=None):
        """
        Connect all cells in the presynaptic population to all cells in the
        postsynaptic population.
        """
        allow_self_connections = True # when pre- and post- are the same population,
                                      # is a cell allowed to connect to itself?
        if parameters and parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
        syn_objref = _translate_synapse_type(synapse_type)
        hoc_commands = []
        for tgt in self.post.gidlist:
            for src in self.pre.fullgidlist:
                if allow_self_connections or tgt != src:
                    hoc_commands += ['nc = pc.gid_connect(%d,%s.object(%d).%s)' % (src,
                                                                                   self.post.label,
                                                                                   self.post.gidlist.index(tgt),
                                                                                   syn_objref),
                                     'tmp = %s.append(nc)' % self.label]
                self.connections.append((src,tgt))
        return hoc_commands
        
    def _oneToOne(self,synapse_type=None):
        """
        Where the pre- and postsynaptic populations have the same size, connect
        cell i in the presynaptic population to cell i in the postsynaptic
        population for all i.
        In fact, despite the name, this should probably be generalised to the
        case where the pre and post populations have different dimensions, e.g.,
        cell i in a 1D pre population of size n should connect to all cells
        in row i of a 2D post population of size (n,m).
        """   
        if self.pre.dim == self.post.dim:
            syn_objref = _translate_synapse_type(synapse_type)
            hoc_commands = []
            for tgt in self.post.gidlist:
                src = tgt - self.post.gid_start + self.pre.gid_start
                hoc_commands += ['nc = pc.gid_connect(%d,%s.object(%d).%s)' % (src,
                                                                                self.post.label,
                                                                                self.post.gidlist.index(tgt),
                                                                                syn_objref),
                                 'tmp = %s.append(nc)' % self.label]
                self.connections.append((src,tgt))
        else:
            raise "Method '%s' not yet implemented for the case where presynaptic \
                    and postsynaptic Populations have different sizes." % sys._getframe().f_code.co_name
        return hoc_commands
    
    def _fixedProbability(self,parameters,synapse_type=None):
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
            
        syn_objref = _translate_synapse_type(synapse_type)
        hoc_commands = []
        if isinstance(self.rng, NativeRNG): # use hoc Random object
            hoc_commands = ['rng = new Random(%d)' % 0 or self.rng.seed,
                            'tmp = rng.uniform(0,1)']
            # Here we are forced to execute the commands on line to be able to
            # catch the connections from NEURON.
            hoc_execute(hoc_commands)
            hoc_commands = []
            #Then we do the loop
            for tgt in self.post.gidlist:
                for src in self.pre.fullgidlist:
                    if HocToPy.get('rng.repick()','float') < p_connect:
                        if allow_self_connections or tgt != src:
                            hoc_commands += ['nc = pc.gid_connect(%d,%s.object(%d).%s)' % (src,
                                                                                           self.post.label,
                                                                                           self.post.gidlist.index(tgt),
                                                                                           syn_objref),
                                             'tmp = %s.append(nc)' % self.label]
                            self.connections.append((src,tgt))
            return hoc_commands
        else: # use Python RNG
            for tgt in self.post.gidlist:
                rarr = self.rng.uniform(0, 1, self.pre.size)
                for j in xrange(self.pre.size):
                    src = j + self.pre.gid_start
                    if rarr[j] < p_connect:
                        if allow_self_connections or tgt != src:
                            hoc_commands += ['nc = pc.gid_connect(%d,%s.object(%d).%s)' % (src,
                                                                                       self.post.label,
                                                                                       self.post.gidlist.index(tgt),
                                                                                       syn_objref),
                                         'tmp = %s.append(nc)' % self.label]
                            self.connections.append((src,tgt))
        return hoc_commands

    def _distanceDependentProbability(self,parameters,synapse_type=None):
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
        syn_objref = _translate_synapse_type(synapse_type)
        hoc_commands = []
        
        # Here we observe the connectivity rule: if it is a probability function
        # like "exp(-d^2/2s^2)" then distance_expression should have only
        # alphanumeric characters. Otherwise, if we have characters
        # like >,<, = the connectivity rule is by itself a test.
        alphanum = True
        operators = ['<', '>', '=']
        for i in xrange(len(operators)):
            if not d_expression.find(operators[i])==-1:
                alphanum = False
        
        if isinstance(self.rng, NativeRNG):
            hoc_commands = ['rng = new Random(%d)' % 0 or distribution.rng.seed,
                            'tmp = rng.uniform(0,1)']
            # Here we are forced to execute the commands on line to be able to
            # catch the connections from Neuron
            hoc_execute(hoc_commands)
            hoc_commands = []
            # We need to use the gid stored as ID, so we should modify the loop to scan the global gidlist (containing ID)
            for tgt in self.post.gidlist:
                for src in self.pre.fullgidlist:
                    if allow_self_connections or tgt != src: 
                        # calculate the distance between the two cells :
                        dist = self._distance(self.pre, self.post, src, tgt)
                        distance_expression = d_expression.replace('d', '%f' %dist)
                        if alphanum:
                            if HocToPy('rng.repick()','float') < eval(distance_expression):
                                hoc_commands += ['nc = pc.gid_connect(%d,%s.object(%d).%s)' % (src,
                                                                                          self.post.label,
                                                                                          self.post.gidlist.index(tgt),
                                                                                          syn_objref),
                                             'tmp = %s.append(nc)' % self.label]
                                self.connections.append((src,tgt))
                        elif eval(distance_expression):
                            hoc_commands += ['nc = pc.gid_connect(%d,%s.object(%d).%s)' % (src,
                                                                                          self.post.label,
                                                                                          self.post.gidlist.index(tgt),
                                                                                          syn_objref),
                                             'tmp = %s.append(nc)' % self.label]
                            self.connections.append((src,tgt))
            return hoc_commands
        else: # use a python RNG
            for tgt in self.post.gidlist:
                rarr = self.rng.uniform(0,1,self.pre.size)
                for j in xrange(self.pre.size):
                    # Again, we should have an ID (stored in the global gidlist) instead
                    # of a simple int.
                    src = self.pre.fullgidlist[j]
                    if allow_self_connections or tgt != src:
                        # calculate the distance between the two cells :
                        dist = self._distance(self.pre, self.post, src, tgt)
                        distance_expression = d_expression.replace('d', '%f' %dist)                      
                        if alphanum:
                            if rarr[j] < eval(distance_expression):
                                hoc_commands += ['nc = pc.gid_connect(%d,%s.object(%d).%s)' % (src,
                                                                                          self.post.label,
                                                                                          self.post.gidlist.index(tgt),
                                                                                          syn_objref),
                                             'tmp = %s.append(nc)' % self.label]
                                self.connections.append((src,tgt))
                        elif eval(distance_expression):
                            hoc_commands += ['nc = pc.gid_connect(%d,%s.object(%d).%s)' % (src,
                                                                                          self.post.label,
                                                                                          self.post.gidlist.index(tgt),
                                                                                          syn_objref),
                                             'tmp = %s.append(nc)' % self.label]
                            self.connections.append((src,tgt))
        return hoc_commands
    
    def _fixedNumberPre(self,parameters,synapse_type=None):
        """Each presynaptic cell makes a fixed number of connections."""
        raise Exception("Method not yet implemented")
        allow_self_connections = True
        if type(parameters) == types.IntType:
            n = parameters
        elif type(parameters) == types.DictType:
            if parameters.has_key['n']: # all cells have same number of connections
                n = parameters['n']
            elif parameters.has_key['rng']: # number of connections per cell follows a distribution
                rng = parameters['rng']
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
        else : # assume parameters is a rng
            rng = parameters
            
    def _fixedNumberPost(self,parameters,synapse_type=None):
        """Each postsynaptic cell receives a fixed number of connections."""
        raise Exception("Method not yet implemented")
        allow_self_connections = True
        if type(parameters) == types.IntType:
            n = parameters
        elif type(parameters) == types.DictType:
            if parameters.has_key['n']: # all cells have same number of connections
                n = parameters['n']
            elif parameters.has_key['rng']: # number of connections per cell follows a distribution
                rng = parameters['rng']
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
        else : # assume parameters is a rng
            rng = parameters
        syn_objref = _translate_synapse_type(synapse_type)
    
    def _fromFile(self,parameters,synapse_type=None):
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
            f = open(filename,'r')
            lines = f.readlines()
        elif type(parameters) == types.DictType:
            # dict could have 'filename' key or 'file' key
            # implement this...
            raise "Argument type not yet implemented"
        
        # We read the file and gather all the data in a list of tuples (one per line)
        input_tuples = []
        for line in lines:
            single_line = line.rstrip()
            single_line = single_line.split("\t", 4)
            input_tuples.append(single_line)    
        f.close()
        
        return self._fromList(input_tuples, synapse_type)
    
    def _fromList(self,conn_list,synapse_type=None):
        """
        Read connections from a list of tuples,
        containing ['src[x,y]', 'tgt[x,y]', 'weight', 'delay']
        """
        hoc_commands = []
        syn_objref = _translate_synapse_type(synapse_type)
        
        # Then we go through those tuple and extract the fields
        for i in xrange(len(conn_list)):
            src    = conn_list[i][0]
            tgt    = conn_list[i][1]
            weight = eval(conn_list[i][2])
            delay  = eval(conn_list[i][3])
            src = "[%s" %src.split("[",1)[1]
            tgt = "[%s" %tgt.split("[",1)[1]
            src  = eval("self.pre%s" % src)
            tgt  = eval("self.post%s" % tgt)
            hoc_commands += ['nc = pc.gid_connect(%d,%s.object(%d).%s)' % (src,
                                                                           self.post.label,
                                                                           self.post.gidlist.index(tgt),
                                                                           syn_objref),
                             'tmp = %s.append(nc)' % self.label]
            hoc_commands += ['%s.object(%d).weight = %f' % (self.label, i, float(weight)), 
                             '%s.object(%d).delay = %f'  % (self.label, i, float(delay))]
            self.connections.append((src,tgt))
        return hoc_commands
    
    # --- Methods for setting connection parameters ----------------------------
    
    def setWeights(self,w):
        """
        w can be a single number, in which case all weights are set to this
        value, or an array with the same dimensions as the Projection array.
        """
        if isinstance(w,float) or isinstance(w,int):
            loop = ['for tmp = 0, %d {' %(len(self)-1), 
                        '%s.object(tmp).weight = %f ' %(self.label, float(w)),
                    '}']
            hoc_code = "".join(loop)
            hoc_commands = [ 'cmd = "%s"' %hoc_code,
                             'success = execute1(cmd)']
        else:
            raise Exception("Population.setWeights() not yet implemented for weight arrays.")
        hoc_execute(hoc_commands, "--- Projection[%s].__setWeights__() ---" %self.label)
        
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
                        '%s.object(tmp).weight = rng.repick() ' %(self.label),
                    '}']
            hoc_code = "".join(loop)
            hoc_commands += ['cmd = "%s"' %hoc_code,
                             'success = execute1(cmd)']
        else:       
            hoc_commands = []
            for i in xrange(len(self)):
                hoc_commands += ['%s.object(%d).weight = %f' % (self.label, i, float(rand_distr.next()))]  
        hoc_execute(hoc_commands, "--- Projection[%s].__randomizeWeights__() ---" %self.label)
        
    def setDelays(self,d):
        """
        d can be a single number, in which case all delays are set to this
        value, or an array with the same dimensions as the Projection array.
        """
        if isinstance(d,float) or isinstance(d,int):
            loop = ['for tmp = 0, %d {' %(len(self)-1), 
                        '%s.object(tmp).delay = %f ' %(self.label, float(d)),
                    '}']
            hoc_code = "".join(loop)
            hoc_commands = [ 'cmd = "%s"' %hoc_code,
                             'success = execute1(cmd)']
        else:
            raise Exception("Population.setDelays() not yet implemented for delay arrays.")
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
                        '%s.object(tmp).delay = rng.repick() ' %(self.label),
                    '}']
            hoc_code = "".join(loop)
            hoc_commands += ['cmd = "%s"' %hoc_code,
                             'success = execute1(cmd)']    
        else:
            hoc_commands = [] 
            for i in xrange(len(self)):
                hoc_commands += ['%s.object(%d).delay = %f' % (self.label, i, float(rand_distr.next()))]
        hoc_execute(hoc_commands, "--- Projection[%s].__randomizeDelays__() ---" %self.label)
        
    def setTopographicDelays(self,delay_rule,rand_distr=None):
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
                idx_src = self.pre.fullgidlist.index(src)
                idx_tgt = self.post.fullgidlist.index(tgt)
                dist = self._distance(self.pre, self.post, self.pre.fullgidlist[idx_src], self.post.fullgidlist[idx_tgt])
                # then evaluate the delay according to the delay rule
                delay = eval(delay_rule.replace('d', '%f' %dist))
                hoc_commands += ['%s.object(%d).delay = %f' % (self.label, i, float(delay))]
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
                    dist = self._distance(self.pre, self.post, self.pre.fullgidlist[idx_src], self.post.fullgidlist[idx_tgt])
                    # then evaluate the delay according to the delay rule
                    delay = delay_rule.replace('d', '%f' % dist)
                    delay = eval(delay.replace('rng', '%f' % HocToPy.get('rng.repick()', 'float')))
                    hoc_commands += ['%s.object(%d).delay = %f' % (self.label, i, float(delay))]   
            else:
                for i in xrange(len(self)):
                    src = self.connections[i][0]
                    tgt = self.connections[i][1]    
                    # calculate the distance between the 2 cells :
                    idx_src = self.pre.fullgidlist.index(src)
                    idx_tgt = self.post.fullgidlist.index(tgt)
                    dist = self._distance(self.pre, self.post, self.pre.fullgidlist[idx_src], self.post.fullgidlist[idx_tgt])
                    # then evaluate the delay according to the delay rule :
                    delay = delay_rule.replace('d', '%f' %dist)
                    delay = eval(delay.replace('rng', '%f' %rand_distr.next()))
                    hoc_commands += ['%s.object(%d).delay = %f' % (self.label, i, float(delay))]        
        
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
    
    def setupSTDP(self,stdp_model,parameterDict):
        """Set-up STDP."""
        
        # Define the objref to handle plasticity
        hoc_commands =  ['objref %s_wa[%d]'      %(self.label,len(self)),
                         'objref %s_pre2wa[%d]'  %(self.label,len(self)),
                         'objref %s_post2wa[%d]' %(self.label,len(self))]
        # For each connection
        for i in xrange(len(self)):
            src = self.connections[i][0]
            tgt = self.connections[i][1]
            # we reproduce the structure of STDP that can be found in layerConn.hoc
            hoc_commands += ['%s_wa[%d]     = new %s(0.5)' %(self.label, i, stdp_model),
                             '%s_pre2wa[%d] = pc.gid_connect(%d, %s_wa[%d])' % (self.label, i, src, self.label, i),  
                             '%s_pre2wa[%d].threshold = %s.object(%d).threshold' %(self.label, i, self.label, i),
                             '%s_pre2wa[%d].delay = %s.object(%d).delay' % (self.label, i, self.label, i),
                             '%s_pre2wa[%d].weight = 1' %(self.label, i),
                             '%s_post2wa[%d] = pc.gid_connect(%d, %s_wa[%d])' %(self.label, i, tgt, self.label, i),
                             '%s_post2wa[%d].threshold = 1' %(self.label, i),
                             '%s_post2wa[%d].delay = 0' % (self.label, i),
                             '%s_post2wa[%d].weight = -1' % (self.label, i),
                             'setpointer %s_wa[%d].wsyn, %s.object(%d).weight' %(self.label, i,self.label,i)]
            # then update the parameters
            for param,val in parameterDict.items():
                hoc_commands += ['%s_wa[%d].%s = %f' % (self.label, i, param, val)]
            
        hoc_execute(hoc_commands, "--- Projection[%s].__setupSTDP__() ---" %self.label)  
    
    def toggleSTDP(self,onoff):
        """Turn plasticity on or off. 
        onoff = True => ON  and onoff = False => OFF. By defaut, it is on."""
        # We do the loop in hoc, to speed up the code
        loop = ['for tmp = 0, %d {' %(len(self)-1), 
                    '{ %s_wa[tmp].on = %d ' %(loop, self.label, onoff),
                '}']
        hoc_code = "".join(loop)      
        hoc_commands = [ 'cmd="%s"' %hoc_code,
                         'success = execute1(cmd)']
        hoc_execute(hoc_commands, "--- Projection[%s].__toggleSTDP__() ---" %self.label)  
    
    def setMaxWeight(self,wmax):
        """Note that not all STDP models have maximum or minimum weights."""
        # We do the loop in hoc, to speed up the code
        loop = ['for tmp = 0, %d {' %(len(self)-1), 
                    '{ %s_wa[tmp].wmax = %d ' %(loop, self.label, wmax),
                '}']
        hoc_code = "".join(loop)        
        hoc_commands = [ 'cmd="%s"' %hoc_code,
                         'success = execute1(cmd)']
        hoc_execute(hoc_commands, "--- Projection[%s].__setMaxWeight__() ---" %self.label)  
    
    def setMinWeight(self,wmin):
        """Note that not all STDP models have maximum or minimum weights."""
        # We do the loop in hoc, to speed up the code
        loop = ['for tmp = 0, %d {' %(len(self)-1), 
                    '{ %s_wa[tmp].wmin = %d ' %(loop, self.label, wmin),
                '}']
        hoc_code = "".join(loop)
        hoc_commands = [ 'cmd="%s"' %hoc_code,
                         'success = execute1(cmd)']
        hoc_execute(hoc_commands, "--- Projection[%s].__setMinWeight__() ---" %self.label) 
    
    # --- Methods for writing/reading information to/from file. ----------------
    
    def saveConnections(self,filename,gather=False):
        """Save connections to file in a format suitable for reading in with the
        'fromFile' method."""
        hoc_comment("--- Projection[%s].__saveConnections__() ---" %self.label)  
        f = open(filename,'w')
        for i in xrange(len(self)):
            src = self.connections[i][0]
            tgt = self.connections[i][1]
            line = "%s%s\t%s%s\t%g\t%g\n" % (self.pre.label,
                                     self.pre.locate(src),
                                     self.post.label,
                                     self.post.locate(tgt),
                                     HocToPy.get('%s.object(%d).weight' % (self.label,i),'float'),
                                     HocToPy.get('%s.object(%d).delay' % (self.label,i),'float'))
            line = line.replace('(','[').replace(')',']')
            f.write(line)
        f.close()
    
    def printWeights(self,filename,format=None,gather=True):
        """Print synaptic weights to file."""
        global myid
        
        hoc_execute(['objref weight_list'])
        hoc_commands = [] 
        hoc_comment("--- Projection[%s].__printWeights__() ---" %self.label)
        
        # Here we have to deal with the gather options. If we gather, then each
        # slave node posts its list of weights to the master node.
        if gather and myid !=0:
            hoc_commands += ['weight_list = new Vector()']
            for i in xrange(len(self)):
                weight = HocToPy.get('%s.object(%d).weight' % (self.label,i),'float')
                hoc_commands += ['weight_list = weight_list.append(%f)' %float(weight)]
            hoc_commands += ['tmp = pc.post("%s.weight_list.node[%d]", weight_list)' %(self.label, myid)]
            hoc_execute(hoc_commands, "--- [Posting weights list to master] ---")

        if not gather or myid == 0:
            f = open(filename,'w')
            for i in xrange(len(self)):
                weight = "%f\n" %HocToPy.get('%s.object(%d).weight' % (self.label,i),'float')
                f.write(weight)
            if gather:
                for id in range (1, nhost):
                    hoc_commands = ['weight_list = new Vector()']       
                    hoc_commands += ['tmp = pc.take("%s.weight_list.node[%d]", weight_list)' %(self.label, id)]
                    hoc_execute(hoc_commands)                
                    for j in xrange(HocToPy.get('weight_list.size()', 'int')):
                        weight = "%f\n" %HocToPy.get('weight_list.x[%d]' %j, 'float')
                        f.write(weight)
            f.close()
  
    def weightHistogram(self,min=None,max=None,nbins=10):
        """
        Return a histogram of synaptic weights.
        If min and max are not given, the minimum and maximum weights are
        calculated automatically.
        """
        # it is arguable whether functions operating on the set of weights
        # should be put here or in an external module.
        raise Exception("Method not yet implemented")

# ==============================================================================
#   Utility classes
# ==============================================================================

Timer = common.Timer
    
# ==============================================================================
