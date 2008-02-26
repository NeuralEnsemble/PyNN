# encoding=utf-8
"""
nrnpy implementation of the PyNN API.
$Id:oldneuron.py 188 2008-01-29 10:03:59Z apdavison $
"""
__version__ = "$Revision:188 $"

import hoc
from pyNN.random import *
from pyNN import __path__, common
import os.path, types, time, sys
import numpy

hoc_cells = 0
hoc_netcons = 0
vfilelist = {}
spikefilelist = {}
dt = 0.1

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
            if self._hocname != None:
                set(self,self._cellclass,param,val, self._hocname)
    
    def get(self,param):
        #This function should be improved, with some test to translate
        #the parameter according to the cellclass
        #We have here the same problem that with set() in the parallel framework
        if self._hocname != None:
            return HocToPy.get('%s.%s' %(self._hocname, param),'float')
    
    # Functions used only by the neuron module of pyNN, to optimize the
    # creation of networks
    def setHocName(self, name):
        self._hocname = name

    def getHocName(self):
        return self._hocname

def list_standard_models():
    return [obj for obj in globals().values() if isinstance(obj, type) and issubclass(obj, common.StandardCellType)]

# ==============================================================================
#   Module-specific functions and classes (not part of the common API)
# ==============================================================================

def hoc_execute(hoc_commands, comment=None):
    assert isinstance(hoc_commands,list)
    if comment:
        logfile.write("//" + comment + "\n")    
    for cmd in hoc_commands:
        logfile.write(cmd + "\n")
        hoc.execute(cmd)

def hoc_comment(comment):
    logfile.write("//" + str(comment) + "\n")

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
            for i in range(len(item)):
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
                    for i in range(len(v)):
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
                for i in range(item.shape[0]):
                    for j in range(item.shape[1]):
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

def log(logstr):
    global logfile
    logfile.write(logstr + '\n')

def checkParams(param,val=None):
    """Check parameters are of valid types, normalise the different ways of
       specifying parameters and values by putting everything in a dict.
       Called by set() and Population.set()."""
    if isinstance(param,str):
        if isinstance(val,float) or isinstance(val,int):
            param_dict = {param:float(val)}
        elif isinstance(val,(list, str)):
            param_dict = {param:val}
        else:
            raise common.InvalidParameterValueError
    elif isinstance(param,dict):
        param_dict = param
    else:
        raise common.InvalidParameterValueError
    return param_dict

class HocError(Exception): pass

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
        errorstr = '"raise HocError(\'caused by HocToPy.get(%s,return_type=\\"%s\\")\')"' % (name,return_type)
        hoc.execute('success = 0')
        hoc.execute('strdef cmd')
        hoc.execute('success = sprint(cmd,"HocToPy.hocvar = %s",%s)' % (HocToPy.fmt_dict[return_type],name))        
        
        hoc_commands =  ['if (success) { nrnpython(cmd) } else { nrnpython(%s) }' % errorstr ]
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
    
    translations = common.build_translations(
        ('tau_m',      'tau_m'),
        ('cm',         'CM'),
        ('v_rest',     'v_rest'),
        ('v_thresh',   'v_thresh'),
        ('v_reset',    'v_reset'),
        ('tau_refrac', 't_refrac'),
        ('i_offset',   'i_offset'),
        ('tau_syn_E',  'tau_e'),
        ('tau_syn_I',  'tau_i'),
        ('v_init',     'v_init'),
    )
    hoc_name = "StandardIF"
    
    def __init__(self,parameters):
        common.IF_curr_alpha.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters['syn_type']  = 'current'
        self.parameters['syn_shape'] = 'alpha'

class IF_curr_exp(common.IF_curr_exp):
    """Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses."""
    
    translations = common.build_translations(
        ('tau_m',      'tau_m'),
        ('cm',         'CM'),
        ('v_rest',     'v_rest'),
        ('v_thresh',   'v_thresh'),
        ('v_reset',    'v_reset'),
        ('tau_refrac', 't_refrac'),
        ('i_offset',   'i_offset'),
        ('tau_syn_E',  'tau_e'),
        ('tau_syn_I',  'tau_i'),
        ('v_init',     'v_init'),
    )
    hoc_name = "StandardIF"
    
    def __init__(self,parameters):
        common.IF_curr_exp.__init__(self,parameters)
        self.parameters['syn_type']  = 'current'
        self.parameters['syn_shape'] = 'exp'

class IF_cond_alpha(common.IF_cond_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""
    
    translations = common.build_translations(
        ('tau_m',      'tau_m'),
        ('cm',         'CM'),
        ('v_rest',     'v_rest'),
        ('v_thresh',   'v_thresh'),
        ('v_reset',    'v_reset'),
        ('tau_refrac', 't_refrac'),
        ('i_offset',   'i_offset'),
        ('tau_syn_E',  'tau_e'),
        ('tau_syn_I',  'tau_i'),
        ('v_init',     'v_init'),
        ('e_rev_E',    'e_e'),
        ('e_rev_I',    'e_i')
    )
    hoc_name = "StandardIF"
    
    def __init__(self,parameters):
        common.IF_cond_alpha.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters['syn_type']  = 'conductance'
        self.parameters['syn_shape'] = 'alpha'

class IF_cond_exp(common.IF_cond_exp):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""
    
    translations = common.build_translations(
        ('tau_m',      'tau_m'),
        ('cm',         'CM'),
        ('v_rest',     'v_rest'),
        ('v_thresh',   'v_thresh'),
        ('v_reset',    'v_reset'),
        ('tau_refrac', 't_refrac'),
        ('i_offset',   'i_offset'),
        ('tau_syn_E',  'tau_e'),
        ('tau_syn_I',  'tau_i'),
        ('v_init',     'v_init'),
        ('e_rev_E',    'e_e'),
        ('e_rev_I',    'e_i')
    )
    hoc_name = "StandardIF"
    
    def __init__(self,parameters):
        common.IF_cond_exp.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters['syn_type']  = 'conductance'
        self.parameters['syn_shape'] = 'exp'

class SpikeSourcePoisson(common.SpikeSourcePoisson):
    """Spike source, generating spikes according to a Poisson process."""

    translations = common.build_translations(
        ('start',    'start'),
        ('rate',     'interval',  "1000.0/rate",  "1000.0/interval"),
        ('duration', 'number',    "int(rate/1000.0*duration)", "number*interval"), # should there be a +/1 here?
    )
    hoc_name = 'SpikeSource'
   
    def __init__(self,parameters):
        common.SpikeSourcePoisson.__init__(self,parameters)
        self.parameters['source_type'] = 'NetStim'    
        self.parameters['noise'] = 1

    def translate(self,parameters):
        translated_parameters = common.SpikeSourcePoisson.translate(self,parameters)
        if parameters.has_key('rate') and parameters['rate'] != 0:
            translated_parameters['interval'] = 1000.0/parameters['rate']
        return translated_parameters

class SpikeSourceArray(common.SpikeSourceArray):
    """Spike source generating spikes at the times given in the spike_times array."""

    translations = common.build_translations(
        ('spike_times', 'input_spiketimes'),
    )
    hoc_name = 'SpikeSource'
    
    def __init__(self,parameters):
        common.SpikeSourceArray.__init__(self,parameters) 
        self.parameters['source_type'] = 'VecStim'
        
                        
# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1,min_delay=0.1,max_delay=0.1,debug=False,**extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    global logfile, dt, _min_delay
    dt = timestep
    _min_delay = min_delay
    logfile = open('nrn.log','w')
    hoc_commands = [
        'xopen("%s")' % os.path.join(__path__[0],'hoc','netLayer.hoc'),
        'xopen("%s")' % os.path.join(__path__[0],'hoc','layerConn.hoc'),
        'xopen("%s")' % os.path.join(__path__[0],'hoc','standardCells.hoc'),
        'xopen("%s")' % os.path.join(__path__[0],'hoc','odict.hoc'),
        'dt = %f' % dt,
        #'objref cvode',
        #'cvode = new CVode(1)',
        #'cvode.condition_order(2)',
        'create dummy_section',
        'access dummy_section']
    hoc_execute(hoc_commands,"--- setup() ---")
    return 0

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    global logfile, vfilelist, spikefilelist
    hoc_commands = []
    if len(vfilelist) > 0:
        hoc_commands = ['objref fileobj',
                        'fileobj = new File()']
        for filename,cell_list in vfilelist.items():
            hoc_commands += ['fileobj.wopen("%s")' % filename]
            for cell in cell_list:
                hoc_commands += ['fileobj.printf("# %s\\n")' % (cell),
                                 '%s.vtrace.printf(fileobj)' % cell]
            hoc_commands += ['fileobj.close()']
    if len(spikefilelist) > 0:
        hoc_commands += ['objref fileobj',
                        'fileobj = new File()']
        for filename,cell_list in spikefilelist.items():
            hoc_commands += ['fileobj.wopen("%s")' % filename]
            for cell in cell_list:
                hoc_commands += ['fileobj.printf("# %s\\n")' % (cell),
                                 '%s.spiketimes.printf(fileobj)' % cell]
            hoc_commands += ['fileobj.close()']
    hoc_execute(hoc_commands,"--- end() ---")
    logfile.close()
    sys.exit(0)

def run(simtime):
    """Run the simulation for simtime ms."""
    hoc_commands = ['finitialize()',
                    'tstop = %f' % simtime,
                    'while (t < tstop) { fadvance() }']
    hoc_execute(hoc_commands,"--- run() ---")

def setRNGseeds(seedList):
    """Globally set rng seeds. Not applicable to NEURON?"""
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
    global hoc_cells
    
    assert n > 0, 'n must be a positive integer'
    if isinstance(cellclass, type):
        celltype = cellclass(param_dict)
        hoc_name = celltype.hoc_name
        hoc_commands, argstr = _hoc_arglist([celltype.parameters])
    elif isinstance(cellclass,str):
        hoc_name = cellclass
        hoc_commands, argstr = _hoc_arglist([param_dict])
 
    retval = []
    for i in range(0,n):
        hoc_commands += ['objref cell%d' % hoc_cells,
                         'cell%d = new %s(%s)' % (hoc_cells,hoc_name,argstr.strip().strip(','))]
        retval.append('cell%d' % hoc_cells)
        hoc_cells += 1
    hoc_execute(hoc_commands, "--- create() ---")
    if n == 1:
        retval = retval[0]
    return retval

def connect(source,target,weight=None,delay=None,synapse_type=None,p=1,rng=None):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise.
    Weights should be in nA or uS."""
    global hoc_netcons, _min_delay
    if type(source) != types.ListType:
        source = [source]
    if type(target) != types.ListType:
        target = [target]
    if weight is None:  weight = 0.0
    if delay  is None:  delay = _min_delay
    syn_objref = _translate_synapse_type(synapse_type, weight)
    nc_start = hoc_netcons
    hoc_commands = []
    for src in source:
        assert isinstance(src, str)
        if int(src[4:]) >= hoc_cells:         # } crude and error-prone (but fast) way of 
            raise common.ConnectionError      # } checking if a cell exists. Needs improving
        if p < 1:
            if rng: # use the supplied RNG
                rarr = self.rng.uniform(0,1,len(target))
            else:   # use the default RNG
                rarr = numpy.random.uniform(0,1,len(target))
        for j,tgt in enumerate(target):
            assert isinstance(tgt, str)
            if int(tgt[4:]) >= hoc_cells:     # } crude and error-prone (but fast) way of 
                raise common.ConnectionError  # } checking if a cell exists. Needs improving
            if p >= 1.0 or rarr[j] < p: # might be more efficient to vectorise the latter comparison
                hoc_commands += ['objref nc%d' % hoc_netcons,
                                 'nc%d = new NetCon(%s.source,%s.%s,1.0,%g,%g)' % (hoc_netcons,src,tgt,syn_objref,delay,weight)]
                hoc_netcons += 1
    hoc_execute(hoc_commands, "--- connect() ---")
    return range(nc_start,hoc_netcons)

def set(cells,cellclass,param,val=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value.
    cellclass must be supplied for doing translation of parameter names."""
    
    param_dict = checkParams(param,val)
    if isinstance(cellclass, common.StandardCellType):
        param_dict = cellclass.translate(param_dict)
    if not isinstance(cells,list):
        cells = [cells]    
    hoc_commands = []
    for param,val in param_dict.items():
        if isinstance(val,str):
            fmt = '%s.%s = "%s"'
        elif isinstance(val,list):
            cmds,argstr = _hoc_arglist([val])
            hoc_commands += cmds
            fmt = '%s.%s = %s'
            val = argstr
        else:
            fmt = '%s.%s = %g'
        for cell in cells:
            hoc_commands += [fmt % (cell,param,val),
                             '%s.param_update()' % cell] 
    hoc_execute(hoc_commands, "--- set() ---")

def record(source,filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    global spikefilelist
    if type(source) != types.ListType:
        source = [source]
    hoc_commands = []
    for src in source:
        hoc_commands += ['%s.record(1)' % src]
    spikefilelist[filename] = source # writing to file is done in end()
    hoc_execute(hoc_commands, "---record() ---")

def record_v(source,filename):
    """
    Record membrane potential to a file. source can be an individual cell or
    a list of cells."""
    # would actually like to be able to record to an array and
    # choose later whether to write to a file.
    global vfilelist
    if type(source) != types.ListType:
        source = [source]
    hoc_commands = []
    for src in source:
        hoc_commands += ['%s.record_v(1)' % src]
    vfilelist[filename] = source # writing to file is done in end()
    hoc_execute(hoc_commands, "---record_v() ---")

# ==============================================================================
#   High-level API for creating, connecting and recording from populations of
#   neurons.
# ==============================================================================

class Population(common.Population):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
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
        
        common.Population.__init__(self,dims,cellclass,cellparams,label)
        if self.ndim > 1:
            for i in range(1,self.ndim):
                if self.dim[i] != self.dim[0]:
                    raise common.InvalidDimensionsError, "All dimensions must be the same size (temporary restriction)."
        
        if isinstance(cellclass, type):
            self.celltype = cellclass(cellparams)
            self.cellparams = self.celltype.parameters
            hoc_name = self.celltype.hoc_name
        elif isinstance(cellclass, str): # not a standard model
            hoc_name = cellclass
        
        hoc_commands, argstr = _hoc_arglist([self.cellparams])
        argstr = argstr.strip().strip(',')
    
        if not self.label:
            self.label = 'population%d' % Population.nPop
        
        # Declare the objref ...    
        hoc_commands += ['objref %s' % self.label]
        
        # Create the object
        hoc_commands += ['%s = new NetLayer(%d,%d,"%s",%s,"%s")' % (self.label,
                                                                    self.ndim,
                                                                    self.dim[0],
                                                                    hoc_name,
                                                                    argstr,
                                                                    self.label)]        
        hoc_execute(hoc_commands, "--- Population.__init__() ---")
        Population.nPop += 1       
    
    def __getitem__(self,addr):
        """Returns a representation of the cell with coordinates given by addr,
           suitable for being passed to other methods that require a cell id.
           Note that __getitem__ is called when using [] access, e.g.
             p = Population(...)
             p[2,3] is equivalent to p.__getitem__((2,3)).
        """
        if isinstance(addr,int):
            addr = (addr,)
        format = "[%d]"*self.ndim
        return format % addr
    
    def __len__(self):
        """Returns the total number of cells in the population."""
        return self.size
    
    def __locate(self,n):
            """Given the order n of an element in a Population, return the coordinates.
               e.g. for  4 6  , element 2 has coordinates (1,0) and value 7
                         7 9
            """
            assert isinstance(n,int)
            if self.ndim == 3:
                rows = self.dim[0]; cols = self.dim[1]
                i = n/(rows*cols); remainder = n%(rows*cols)
                j = remainder/cols; k = remainder%cols
                coords = (i,j,k)
            elif self.ndim == 2:
                cols = self.dim[1]
                i = n/cols; j = n%cols
                coords = (i,j)
            elif self.ndim == 1:
                coords = (n,)
            else:
                raise common.InvalidDimensionsError
            return coords
        
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
        
        hoc_commands = []
        for param,val in param_dict.items():
            if type(val) == types.StringType:
                hoc_commands += ['%s.set("%s","%s")' % (self.label,param,val)]
            else:
                hoc_commands += ['%s.set("%s",%s)' % (self.label,param,float(val))]
                
        hoc_execute(hoc_commands, "--- Population.set() ---")
        
    def tset(self,parametername,valueArray):
        """
        'Topographic' set. Sets the value of parametername to the values in
        valueArray, which must have the same dimensions as the Population.
        """
        
        if self.dim == valueArray.shape:
            values = numpy.reshape(valueArray,valueArray.size)
            if isinstance(self.celltype, common.StandardCellType):
                parametername = self.celltype.translate({parametername: values[0]}).keys()[0]
            hoc_commands, argstr = _hoc_arglist([valueArray])
            hoc_commands += ['%s.tset("%s","%s")' % (self.label,parametername,argstr)]
            hoc_execute(hoc_commands, "--- Population.tset() ---")
        else:
            raise common.InvalidDimensionsError
        
    def rset(self,parametername,rand_distr):
        """
        'Random' set. Sets the value of parametername to a value taken from
        rand_distr, which should be a RandomDistribution object.
        """
        if isinstance(self.celltype, common.StandardCellType):
            parametername = self.celltype.translate({parametername: 0.0}).keys()[0]
        if isinstance(rand_distr.rng, NativeRNG):
            paramfmt = "%g,"*len(rand_distr.parameters); paramfmt = paramfmt.strip(',')
            distr_params = paramfmt % tuple(rand_distr.parameters)
            hoc_commands = ['objref rng',
                            'rng = new Random(%d)' % 0 or distribution.rng.seed,
                            'rng.%s(%s)' % (rand_distr.name,distr_params),
                            '%s.rset("%s",rng)' % (self.label,parametername),
                            '%s.call("param_update","")' % self.label]
        else:
            rarr = rand_distr.next(n=self.size)
            rarr = rarr.reshape(self.dim)
            hoc_commands, argstr = _hoc_arglist([rarr])
            hoc_commands += ['%s.tset("%s","%s")' % (self.label,parametername,argstr),
                             '%s.call("param_update","")' % self.label]
            
        hoc_execute(hoc_commands, "--- Population.rset() ---")
    
    def _call(self,methodname,arguments):
        """
        Calls the method methodname(arguments) for every cell in the population.
        e.g. p.call("set_background","0.1") if the cell class has a method
        set_background().
        """
        # Not sure this belongs in the API, because cell classes only have
        # parameters/attributes, not methods.
        hoc_commands, argstr = _hoc_arglist(arguments)
        hoc_commands += ['%s.call(%s)' % (self.label,argstr)]
        hoc_execute(hoc_commands, "--- Population._call() ---")
    
    def _tcall(self,methodname,objarr):
        """
        `Topographic' call. Calls the method methodname() for every cell in the 
        population. The argument to the method depends on the coordinates of the
        cell. objarr is an array with the same dimensions as the Population.
        e.g. p.tcall("memb_init",vinitArray) calls
        p.cell[i][j].memb_init(vInitArray[i][j]) for all i,j.
        """
        if self.dim == objarr.shape:
            hoc_commands, argstr = _hoc_arglist([objarr])
            hoc_commands += ['%s.tcall("%s","%s")' % (self.label,methodname,argstr)]
            hoc_execute(hoc_commands, "--- Population._tcall() ---")
        else:
            raise common.InvalidDimensionsError      

    def __record(self,record_what,record_from=None,rng=None):
        """
        Private method called by record() and record_v().
        """
        hoc_commands = ['objref id_list', 'id_list = new List()']
        if isinstance(record_from,int): # record from a number of cells, selected at random
            nrec = record_from
            if rng:
                record_from = rng.permutation(self.size)
            else:
                record_from = numpy.random.permutation(self.size)
            record_from = record_from[0:nrec]
            record_from = [self[self.__locate(id)] for id in record_from]
            self.__record(record_what,record_from) # call again, this time with the random list.
        elif isinstance(record_from,list):
            for addr in record_from:
                hoc_commands += ['id_list.append(new String("%s"))' % addr]
            hoc_commands += ['%s.lcall(id_list,"record%s","1")' % (self.label,record_what)]
        elif record_from is None: # record all cells
            hoc_commands = ['%s.call("record%s","1")' % (self.label,record_what)]
        
        hoc_execute(hoc_commands, "--- Population.record%s() ---" % record_what)

    def record(self,record_from=None,rng=None):
        """
        If record_from is not given, record spikes from all cells in the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        self.__record('',record_from,rng)
        
    def record_v(self,record_from=None,rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        self.__record('_v',record_from,rng)
        
    def printSpikes(self,filename,gather=True,compatible_output=True):
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
        # Note that 'gather' has no effect, since this module is not
        # parallelised.
        hoc_commands = ['objref spikefile',
                        'spikefile = new File()',
                        'spikefile.wopen("%s")' % filename,
                        '%s.print_spikes(spikefile)' % self.label,
                        'spikefile.close()']
        hoc_execute(hoc_commands, "--- Population.printSpikes() ---")

    def meanSpikeCount(self,gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        # gather is not relevant, but is needed for API consistency
        hoc_comment("--- Population.meanSpikesCount() ---")
        self.nspikes = HocToPy.get('%s.mean_spike_count()' % self.label,'float')
        return self.nspikes

    def randomInit(self,rand_distr):
        """
        Sets initial membrane potentials for all the cells in the population to
        random values.
        """
        hoc_comment("--- Population.randomInit() ---")
        #rvals = getattr(numpy.random,distribution)(size=self.dim,*params)
        rvals = rand_distr.next(self.dim)
        self._tcall('memb_init',rvals)
        
    def print_v(self,filename,gather=True,compatible_output=True):
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
        hoc_commands = ['objref vfile',
                        'vfile = new File()',
                        'vfile.wopen("%s")' % filename,
                        'vfile.printf("# dt = %f\\n",dt)',
                        'vfile.printf("# n = %d\\n",int(tstop/dt))',
                        '%s.print_v(vfile)' % self.label,
                        'vfile.close()']
        hoc_execute(hoc_commands, "--- Population.print_v() ---")
    
    
class Projection(common.Projection):
    """
    A container for all the connections between two populations, together with
    methods to set parameters of those connections, including of plasticity
    mechanisms.
    """
    
    nProj = 0
    
    class ConnectionDict:
            
        def __init__(self,parent):
            self.parent = parent
            #self.ndim = len(parent.shape)
    
        def __locate(self,n,population):
            """Given the order n of an element in a Population, return the coordinates.
               e.g. for  4 6  , element 2 has coordinates (1,0) and value 7
                         7 9
            """
            assert isinstance(n,int)
            if population.ndim == 3:
                rows = population.dim[0]; cols = population.dim[1]
                i = n/(rows*cols); remainder = n%(rows*cols)
                j = remainder/cols; k = remainder%cols
                coords = (i,j,k)
            elif population.ndim == 2:
                cols = population.dim[1]
                i = n/cols; j = n%cols
                coords = (i,j)
            elif population.ndim == 1:
                coords = (n,)
            else:
                raise common.InvalidDimensionsError
            return coords
            
        def __getitem__(self,id):
            """Returns a connection id.
            Suppose we have a 2D Population (5x3) projecting to a 3D Population (4x5x7).
            Total number of possible connections is 5x3x4x5x7 = 2100.
            Therefore valid calls are:
            connection[2099] - 2099th connection
            connection[14,139] - connection between 14th pre- and 139th postsynaptic neuron (may not exist)
            connection[(4,2),(3,4,6)] - connection between presynaptic neuron with address (4,2)
            and post-synaptic neuron with address (3,4,6) (may not exist)
            Assuming all connections exist, all the above would return:
            "[4][2][3][4][6]", which is the LayerConn.nc address in hoc."""
            if isinstance(id, int): # linear mapping
                preID = id/self.parent.post.size; postID = id%self.parent.post.size
                return self.__getitem__((preID,postID))
            elif isinstance(id, tuple): # (pre,post)
                if len(id) == 2:
                    pre = id[0]
                    post = id[1]
                    if isinstance(pre,int) and isinstance(post,int):
                        pre_coords = self.__locate(pre, self.parent.pre)
                        post_coords = self.__locate(post, self.parent.post)
                        return self.__getitem__((pre_coords,post_coords))
                    elif isinstance(pre,tuple) and isinstance(post,tuple): # should also allow lists
                        if len(pre) == self.parent.pre.ndim and len(post) == self.parent.post.ndim:
                            fmt = "[%d]"*(len(pre)+len(post))
                            address = fmt % (pre+post)
                        else:
                            raise common.InvalidDimensionsError
                    else:
                        raise KeyError
                else:
                    raise common.nvalidDimensionsError
            else:
                raise KeyError #most appropriate?
            
            return address
    
    
    def __init__(self,presynaptic_population,postsynaptic_population,method='allToAll',method_parameters=None,source=None,target=None,label=None,rng=None):
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
        
        rng - since most of the connection methods need uniform random numbers,
        it is probably more convenient to specify a RNG object here rather
        than within method_parameters, particularly since some methods also use
        random numbers to give variability in the number of connections per cell.
        """
        global _min_delay
        common.Projection.__init__(self,presynaptic_population,postsynaptic_population,method,method_parameters,source,target,label,rng)
        if not label:
            self.label = 'projection%d' % Projection.nProj
        connection_method = getattr(self,'_%s' % method)
        if target:
            hoc_commands = connection_method(method_parameters,synapse_type=target)
        else:
            hoc_commands = connection_method(method_parameters)
        hoc_execute(hoc_commands, "--- Projection.__init__() ---")
        self.connection = Projection.ConnectionDict(self)
        self.nconn = HocToPy.get('%s.count()' % self.label,'int')
        self.setDelays(_min_delay)
        Projection.nProj += 1

    def __len__(self):
        """Return the total number of connections."""
        return self.nconn
    
    def connections(self):
        """for conn in prj.connections(): ...
        This is equivalent to: for conn in prn.connection: ..."""
        for i in range(len(self)):
            yield self.connection[i]

    # --- Connection methods ---------------------------------------------------
    
    def _allToAll(self,parameters=None,synapse_type=None):
        """
        Connect all cells in the presynaptic population to all cells in the postsynaptic population.
        """
        allow_self_connections = True # when pre- and post- are the same population,
                                      # is a cell allowed to connect to itself?
        if parameters and parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
        syn_objref = _translate_synapse_type(synapse_type)
        hoc_commands = ["objref %s" % self.label,
                        '%s = new LayerConn(%s,"source",%s,"%s",1)' % (
                            self.label, self.pre.label,self.post.label,syn_objref)]
        if not allow_self_connections:
            hoc_commands += ['%s.remove_self_connections()' % self.label]
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
            hoc_commands = ["objref %s" % self.label,
                        '%s = new LayerConn(%s,"source",%s,"%s",0)' % (
                            self.label, self.pre.label,self.post.label,syn_objref)]
            #self.nconn = HocToPy.get('%s.count()' % self.label,'int')
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

        if self.rng is None or isinstance(self.rng, NativeRNG):
            hoc_commands = ["objref rng",  # actually, we ought not to create a new RNG if one already
                            "rng = new Random()", # exists with the same label.
                            "rng.uniform(0,1)",
                            "objref %s" % self.label,
                            '%s = new LayerConn(%s,"source",%s,"%s",1,%f,rng)' % (
                                self.label, self.pre.label,self.post.label,syn_objref,p_connect)
                            ]
        else:
            npre = self.pre.size
            npost = self.post.size
            hoc_commands = ["objref %s" % self.label,
                            '%s = new LayerConn(%s,"source",%s,"%s",2)' % (
                                self.label, self.pre.label,self.post.label,syn_objref)
                           ]
            for i in range(0,npre): # this is a temporary cheat, since it assumes we only have 1D Populations
                rarr = self.rng.uniform(0,1,npost)
                for j in range(0,npost):
                    if rarr[j] < p_connect:
                        hoc_commands += ['%s.nc[%d][%d] = new NetCon(%s.prelayer.cell[%d].source,%s.postlayer.cell[%d].%s,0,0,0)' % (self.label,i,j,self.label,i,self.label,j,syn_objref)]
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
        
        raise Exception("Method not yet implemented")
    
    def _fixedNumberPre(self,parameters,synapse_type=None):
        """Each presynaptic cell makes a fixed number of connections."""
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
        raise Exception("Method not yet implemented")
    
    def _fixedNumberPost(self,parameters,synapse_type=None):
        """Each postsynaptic cell receives a fixed number of connections."""
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
        # cheating here, and ignoring the rng supplied as a parameter for now.
        hoc_commands = ["objref rng",
                        "rng = new Random()",
                        "rng.uniform(0,1)",
                        "objref %s" % self.label,
                        '%s = new LayerConn(%s,"source",%s,"%s",1,%f,rng)' % (
                            self.label, self.pre.label,self.post.label,syn_objref,n/float(self.pre.size))]
        # ought to use method 3, but I haven't implemented it, so I'm cheating for now and using method 1
        return hoc_commands
    
    def _fromFile(self,parameters,synapse_type=None):
        """
        Load connections from a file.
        """
        if type(parameters) == types.FileType:
            fileobj = parameters
            # check fileobj is already open for reading
        elif type(parameters) == types.StringType:
            filename = parameters
            # now open the file...
        elif type(parameters) == types.DictType:
            # dict could have 'filename' key or 'file' key
            # implement this...
            raise "Argument type not yet implemented"
        raise Exception("Method not yet implemented")
    
    def _fromList(self,conn_list,synapse_type=None):
        """
        Read connections from a list of tuples,
        containing [pre_addr, post_addr, weight, delay]
        where pre_addr and post_addr are both neuron addresses, i.e. tuples or
        lists containing the neuron array coordinates.
        """
        # Need to implement parameter parsing here...
        raise Exception("Method not yet implemented")
    
    # --- Methods for setting connection parameters ----------------------------
    
    def setWeights(self,w):
        """
        w can be a single number, in which case all weights are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        Weights should be in nA for current-based and µS for conductance-based
        synapses.
        """
        cmd = "%s.set_weights(%f)" % (self.label,float(w))
        hoc_execute([cmd], "--- Projection.__setWeights__() ---")
    
    def randomizeWeights(self,rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        # Arguably, we could merge this with set_weights just by detecting the
        # argument type. It could make for easier-to-read simulation code to
        # give it a separate name, though. Comments?
        raise Exception("Method not yet implemented")
    
    def setDelays(self,d):
        """
        d can be a single number, in which case all delays are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        """
        cmd = "%s.set_delays(%f)" % (self.label,float(d))
        hoc_execute([cmd], "--- Projection.__setDelays__() ---")
    
    def randomizeDelays(self,rand_distr):
        """
        Set delays to random values taken from rand_distr.
        """
        raise Exception("Method not yet implemented")
    
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
        raise Exception("Method not yet implemented")
    
    def toggleSTDP(self,onoff):
        """Turn plasticity on or off."""
        raise Exception("Method not yet implemented")
    
    def setMaxWeight(self,wmax):
        """Note that not all STDP models have maximum or minimum weights."""
        raise Exception("Method not yet implemented")
    
    def setMinWeight(self,wmin):
        """Note that not all STDP models have maximum or minimum weights."""
        raise Exception("Method not yet implemented")
    
    # --- Methods for writing/reading information to/from file. ----------------
    
    def saveConnections(self,filename,gather=False):
        """Save connections to file in a format suitable for reading in with the
        'fromFile' method."""
        raise Exception("Method not yet implemented")
    
    def printWeights(self,filename,format=None,gather=True):
        """Print synaptic weights to file."""
        raise Exception("Method not yet implemented")
    
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
