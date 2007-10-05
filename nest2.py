# -*- coding: utf-8 -*-
"""
PyNEST implementation of the PyNN API.
$Id$
"""
__version__ = "$Revision:5 $"

# temporary fix to import nest rather than pyNN.nest
import imp
mod_search = imp.find_module('nest', ['/usr/lib/python/site-packages','/usr/local/lib/python2.5/site-packages'])
nest = imp.load_module('nest',*mod_search)
from pyNN import common
from pyNN.random import *
import numpy, types, sys, shutil, os, logging, copy, tempfile
from math import *

#ll_spike_files = []
#ll_v_files     = {}
#hl_spike_files = {}
#hl_v_files     = {}
#hl_c_files     = {}
recorder_dict = {}
tempdirs       = []
dt             = 0.1
recording_device_names = {'spikes': 'spike_detector',
                          'v': 'voltmeter',
                          'conductance': 'conductancemeter'}

# ==============================================================================
#   Utility classes
# ==============================================================================

class ID(common.ID):
    """
    Instead of storing ids as integers, we store them as ID objects,
    which allows a syntax like:
        p[3,4].tau_m = 20.0
    where p is a Population object. The question is, how big a memory/performance
    hit is it to replace integers with ID objects?
    """
    
    def __getattr__(self,name):
        """Note that this currently does not translate units."""
        if issubclass(self.cellclass, common.StandardCellType):
            translated_name = self.cellclass.translations[name][0]
        elif isinstance(self.cellclass, str) or self.cellclass is None:
            translated_name = name
        else:
            raise Exception("ID object has invalid cell class %s" % str(self.cellclass))
        return nest.GetStatus([int(self)])[0][translated_name]
    
    def setParameters(self,**parameters):
        # We perform a call to the low-level function set() of the API.
        # If the cellclass is not defined in the ID object :
        #if (self.cellclass == None):
        #    raise Exception("Unknown cellclass")
        #else:
        #    # We use the one given by the user
        set(self, self.cellclass, parameters) 

    def getParameters(self):
        """Note that this currently does not translate units."""
        nest_params = nest.GetStatus([int(self)])[0]
        params = {}
        for k,v in self.cellclass.translations.items():
            params[k] = nest_params[v[0]]
        return params
            

class Connection(object):
    
    def __init__(self, pre, post):
        self.pre = pre
        self.post = post
        conn_dict = nest.GetConnections([pre], 'static_synapse')[0]
        if conn_dict:
            self.port = len(conn_dict['targets'])-1
        else:
            raise Exception("Could not get port number for connection between %s and %s" % (pre, post))

    def _set_weight(self, w):
        pass

    def _get_weight(self):
        # this needs to be modified to take account of threads
        # also see nest.GetConnection (was nest.GetSynapseStatus)
        conn_dict = nest.GetConnections([self.pre],'static_synapse')[0]
        if conn_dict:
            return conn_dict['weights'][self.port]
        else:
            return None
        
    weight = property(_get_weight, _set_weight)

# ==============================================================================
#   Standard cells
# ==============================================================================
 
class IF_curr_alpha(common.IF_curr_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
    
    translations = {
            'v_rest'    : ('E_L'    , "parameters['v_rest']"),
            'v_reset'   : ('V_reset', "parameters['v_reset']"),
            'cm'        : ('C_m'    , "parameters['cm']*1000.0"), # C_m is in pF, cm in nF
            'tau_m'     : ('tau_m'  , "parameters['tau_m']"),
            'tau_refrac': ('tau_ref', "max(dt,parameters['tau_refrac'])"),
            'tau_syn_E' : ('tau_ex' , "parameters['tau_syn_E']"),
            'tau_syn_I' : ('tau_in' , "parameters['tau_syn_I']"),
            'v_thresh'  : ('V_th'   , "parameters['v_thresh']"),
            'i_offset'  : ('I_e'    , "parameters['i_offset']*1000.0"), # I_e is in pA, i_offset in nA
            'v_init'    : ('V_m'    , "parameters['v_init']"),
    }
    nest_name = "iaf_psc_alpha"
    
    def __init__(self,parameters):
        common.IF_curr_alpha.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate(self.parameters)

class IF_curr_exp(common.IF_curr_exp):
    """Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses."""
    
    translations = {
        'v_rest'    : ('E_L'        , "parameters['v_rest']"),
        'v_reset'   : ('V_reset'    , "parameters['v_reset']"),
        'cm'        : ('C_m'        , "parameters['cm']*1000.0"), # C is in pF, cm in nF
        'tau_m'     : ('tau_m'      , "parameters['tau_m']"),
        'tau_refrac': ('tau_ref_abs', "max(dt,parameters['tau_refrac'])"),
        'tau_syn_E' : ('tau_ex'     , "parameters['tau_syn_E']"),
        'tau_syn_I' : ('tau_in'     , "parameters['tau_syn_I']"),
        'v_thresh'  : ('V_th'       , "parameters['v_thresh']"),
        'i_offset'  : ('I_e'        , "parameters['i_offset']*1000.0"), # I0 is in pA, i_offset in nA
        'v_init'    : ('V_m'        , "parameters['v_init']"),
    }
    nest_name = 'iaf_psc_exp'
    
    def __init__(self,parameters):
        common.IF_curr_exp.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)

class IF_cond_alpha(common.IF_cond_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""
    
    translations = {
            'v_rest'    : ('E_L'       , "parameters['v_rest']"),
            'v_reset'   : ('V_reset'   , "parameters['v_reset']"),
            'cm'        : ('C_m'       , "parameters['cm']*1000.0"), # C is in pF, cm in nF
            'tau_m'     : ('g_L'       , "parameters['cm']/parameters['tau_m']*1000.0"),
            'tau_refrac': ('t_ref'     , "max(dt,parameters['tau_refrac'])"),
            'tau_syn_E' : ('tau_syn_ex', "parameters['tau_syn_E']"),
            'tau_syn_I' : ('tau_syn_in', "parameters['tau_syn_I']"),
            'v_thresh'  : ('V_th'      , "parameters['v_thresh']"),
            #'i_offset'  : ('Istim'    , "parameters['i_offset']*1000.0"), # I0 is in pA, i_offset in nA
            'e_rev_E'   : ('E_ex'      , "parameters['e_rev_E']"),
            'e_rev_I'   : ('E_in'      , "parameters['e_rev_I']"),
            'v_init'    : ('V_m'       , "parameters['v_init']"),
    }
    nest_name = "iaf_cond_alpha"
    
    def __init__(self,parameters):
        common.IF_cond_alpha.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate(self.parameters)
        

class IF_cond_exp(common.IF_cond_exp):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""
    
    translations = {
            'v_rest'    : ('E_L'          , "parameters['v_rest']"),
            'v_reset'   : ('V_reset'      , "parameters['v_reset']"),
            'cm'        : ('C_m'           , "parameters['cm']*1000.0"), # C is in pF, cm in nF
            'tau_m'     : ('g_L'         , "parameters['cm']/parameters['tau_m']*1000.0"),
            'tau_refrac': ('t_ref'        , "max(dt,parameters['tau_refrac'])"),
            'tau_syn_E' : ('tau_syn_ex'    , "parameters['tau_syn_E']"),
            'tau_syn_I' : ('tau_syn_in'    , "parameters['tau_syn_I']"),
            'v_thresh'  : ('V_th'       , "parameters['v_thresh']"),
            #'i_offset'  : ('Istim'       , "parameters['i_offset']*1000.0"), # I0 is in pA, i_offset in nA
            'e_rev_E'   : ('E_ex', "parameters['e_rev_E']"),
            'e_rev_I'   : ('E_in', "parameters['e_rev_I']"),
            'v_init'    : ('V_m'           , "parameters['v_init']"),
    }
    nest_name = "iaf_cond_exp"
    
    def __init__(self,parameters):
        common.IF_cond_exp.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate(self.parameters)
        

class HH_cond_exp(common.HH_cond_exp):
    """docstring needed here."""
    
    translations = {
        'gbar_Na'   : ('g_Na',   "parameters['gbar_Na']"),   
        'gbar_K'    : ('g_K',    "parameters['gbar_K']"),    
        'g_leak'    : ('g_L',    "parameters['g_leak']"),    
        'cm'        : ('C_m',    "parameters['cm']*1000.0"),  
        'v_offset'  : ('U_tr',   "parameters['v_offset']"),
        'e_rev_Na'  : ('E_Na',   "parameters['e_rev_Na']"),
        'e_rev_K'   : ('E_K',    "parameters['e_rev_K']"), 
        'e_rev_leak': ('E_L',    "parameters['e_rev_leak']"),
        'e_rev_E'   : ('E_ex',   "parameters['e_rev_E']"),
        'e_rev_I'   : ('E_in',   "parameters['e_rev_I']"),
        'tau_syn_E' : ('tau_ex', "parameters['tau_syn_E']"),
        'tau_syn_I' : ('tau_in', "parameters['tau_syn_I']"),
        'i_offset'  : ('I_stim', "parameters['i_offset']*1000.0"),
        'v_init'    : ('V_m',    "parameters['v_init']"),
    }
    nest_name = "hh_cond_exp_traub"
    
    def __init__(self,parameters):
        common.HH_cond_exp.__init__(self,parameters) # checks supplied parameters and adds default
                                                     # values for not-specified parameters.
        self.parameters = self.translate(self.parameters)
        
class AdaptiveExponentialIF_alpha(common.AdaptiveExponentialIF_alpha):
    """adaptive exponential integrate and fire neuron according to Brette and Gerstner (2005)"""
    
    translations = {
        'v_init'    : ('V_m',        "parameters['v_init']"),
        'w_init'    : ('w',          "parameters['w_init']*1000.0"), # nA -> pA
        'cm'        : ('C_m',        "parameters['cm']*1000.0"),     # nF -> pF
        'tau_refrac': ('t_ref',      "parameters['tau_refrac']"), 
        'v_spike'   : ('V_peak',     "parameters['v_spike']"),
        'v_reset'   : ('V_reset',    "parameters['v_reset']"),
        'v_rest'    : ('E_L',        "parameters['v_rest']"),
        'tau_m'     : ('g_L',        "parameters['cm']/parameters['tau_m']*1000.0"),
        'i_offset'  : ('I_e',        "parameters['i_offset']*1000.0"), # nA -> pA
        'a'         : ('a',          "parameters['a']"),       
        'b'         : ('b',          "parameters['b']*1000.0"),  # nA -> pA.
        'delta_T'   : ('Delta_T',    "parameters['delta_T']"), 
        'tau_w'     : ('tau_w',      "parameters['tau_w']"), 
        'v_thresh'  : ('V_th',       "parameters['v_thresh']"), 
        'e_rev_E'   : ('E_ex',       "parameters['e_rev_E']"),
        'tau_syn_E' : ('tau_syn_ex', "parameters['tau_syn_E']"), 
        'e_rev_I'   : ('E_in',       "parameters['e_rev_I']"), 
        'tau_syn_I' : ('tau_syn_in', "parameters['tau_syn_I']"),
    }
    nest_name = "aeif_cond_alpha"
    
    def __init__(self,parameters):
        common.AdaptiveExponentialIF_alpha.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)
        
class SpikeSourcePoisson(common.SpikeSourcePoisson):
    """Spike source, generating spikes according to a Poisson process."""

    translations = {
        'rate'     : ('rate'   , "parameters['rate']"),
        'start'    : ('start'  , "parameters['start']"),
        'duration' : ('stop'   , "parameters['duration']+parameters['start']")
    }
    nest_name = 'poisson_generator'
    
    
    def __init__(self,parameters):
        common.SpikeSourcePoisson.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)
        self.parameters['origin'] = 1.0
    
class SpikeSourceArray(common.SpikeSourceArray):
    """Spike source generating spikes at the times given in the spike_times array."""

    translations = {
        'spike_times' : ('spike_times' , "parameters['spike_times']"),
    }
    nest_name = 'spike_generator'
    
    def __init__(self,parameters):
        common.SpikeSourceArray.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)  
    

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1,min_delay=0.1,max_delay=0.1,debug=False,**extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    #if min_delay > max_delay:
    #    raise Exception("min_delay has to be less than or equal to max_delay.")
    global dt
    global tempdir
    global _min_delay
    #global hl_spike_files, hl_v_files
    dt = timestep
    _min_delay = min_delay
    
    # reset the simulation kernel
    nest.ResetKernel()
    # clear the sli stack, if this is not done --> memory leak cause the stack increases
    nest.sr('clear')

#    # check if hl_spike_files , hl_v_files are empty
#    if not len(hl_spike_files) == 0:
#        print 'hl_spike_files still contained files, please close all open files before setup'
#        hl_spike_files = {}
#    if not len(hl_v_files) == 0:
#        print 'hl_v_files still contained files, please close all open files before setup'
#        hl_v_files = {}
    
    tempdir = tempfile.mkdtemp()
    tempdirs.append(tempdir) # append tempdir to tempdirs list
    
    # set tempdir
    nest.SetStatus([0], {'device_prefix':tempdir})
    # set resolution
    nest.SetStatus([0], {'resolution': dt})
    
    if extra_params.has_key('threads'):
        if extra_params.has_key('kernelseeds'):
            print 'params has kernelseed ', extra_params['kernelseeds']
            kernelseeds = extra_params['kernelseeds']
        else:
            # default kernelseeds, for each thread one, to ensure same for each sim we get the rng with seed 42
            rng = NumpyRNG(42)
            num_processes = nest.GetStatus([0])[0]['num_processes']
            kernelseeds = (rng.rng.uniform(size=extra_params['threads']*num_processes)*100).astype('int').tolist()
            print 'params has not kernelseed ',kernelseeds
            
        nest.SetStatus([0],[{'local_num_threads'     : extra_params['threads'],
                            'rng_seeds'   : kernelseeds}])

    
    # Initialisation of the log module. To write in the logfile, simply enter
    # logging.critical(), logging.debug(), logging.info(), logging.warning()
    if debug:
        if isinstance(debug, basestring):
            filename = debug
        else:
            filename = "nest.log"
        logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename=filename,
                    filemode='w')
    else:
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='nest.log',
                    filemode='w')

    logging.info("Initialization of Nest")    
    return nest.Rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    # We close the high level files opened by populations objects
    # that may have not been written.

    # NEST will soon close all its output files after the simulate function is over, therefore this step is not necessary
    global tempdirs
    
    # And we postprocess the low level files opened by record()
    # and record_v() method
    #for file in ll_spike_files:
    #    _printSpikes(file, compatible_output)
    #for file, nest_file in ll_v_files:
    #    _print_v(file, nest_file, compatible_output)
    print "Saving the following files:", recorder_dict.keys()
    for filename in recorder_dict.keys():
        _print(filename, gather=False, compatible_output=compatible_output)
    
    for tempdir in tempdirs:
        os.system("rm -rf %s" %tempdir)

def run(simtime):
    """Run the simulation for simtime ms."""
    nest.Simulate(simtime)

def setRNGseeds(seedList):
    """Globally set rng seeds."""
    nest.SetStatus([0],{'rng_seeds': seedList})

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

def create(cellclass,paramDict=None,n=1):
    """
    Create n cells all of the same type.
    If n > 1, return a list of cell ids/references.
    If n==1, return just the single id.
    """
    assert n > 0, 'n must be a positive integer'
    if isinstance(cellclass, type):
        celltype = cellclass(paramDict)
        cell_gids = nest.Create(celltype.nest_name,n)
        cell_gids = [ID(gid) for gid in cell_gids]
        nest.SetStatus(cell_gids, [celltype.parameters])
    elif isinstance(cellclass, str):  # celltype is not a standard cell
        cell_gids = nest.Create(cellclass,n)
        cell_gids = [ID(gid) for gid in cell_gids]
        if paramDict:
            nest.SetStatus(cell_gids, [paramDict])
    else:
        raise "Invalid cell type"
    for id in cell_gids:
    #    #id.setCellClass(cellclass)
        id.cellclass = cellclass
    if n == 1:
        return cell_gids[0]
    else:
        return cell_gids

def connect(source,target,weight=None,delay=None,synapse_type=None,p=1,rng=None):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise.
    Weights should be in nA or uS."""
    global dt
    if weight is None:
        weight = 0.0
    if delay is None:
        delay = _min_delay
    weight = weight*1000 # weights should be in nA or uS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
                         # Using convention in this way is not ideal. We should be able to look up the units used by each model somewhere.
    if synapse_type == 'inhibitory' and weight > 0:
        weight *= -1
    try:
        if type(source) != types.ListType and type(target) != types.ListType:
            nest.ConnectWD([source],[target],[weight],[delay])
            connect_id = Connection(source, target)
        else:
            connect_id = []
            if type(source) != types.ListType:
                source = [source]
            if type(target) != types.ListType:
                target = [target]
            for src in source:
                if p < 1:
                    if rng: # use the supplied RNG
                        rarr = rng.rng.uniform(0,1,len(target))
                    else:   # use the default RNG
                        rarr = numpy.random.uniform(0,1,len(target))
                for j,tgt in enumerate(target):
                    if p >= 1 or rarr[j] < p:
                        nest.ConnectWD([src],[tgt],[weight],[delay])
                        connect_id += [Connection(src,tgt)]
    except nest.SLIError:
        raise common.ConnectionError
    return connect_id

def set(cells,cellclass,param,val=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value.
    cellclass must be supplied for doing translation of parameter names."""
    # we should just assume that cellclass has been defined and raise an Exception if it has not
    if val:
        param = {param:val}
    try:
        i = cells[0]
    except TypeError:
        cells = [cells]
    if not (isinstance(cellclass,str) or cellclass is None):
        if issubclass(cellclass, common.StandardCellType):
            param = cellclass({}).translate(param)
        else:
            raise TypeError, "cellclass must be a string, None, or derived from commonStandardCellType"
    nest.SetStatus(cells,[param])

def _connect_recording_device(recorder, record_from=None):
    print "Connecting recorder %s to cell(s) %s" % (recorder, record_from)
    device = nest.GetStatus(recorder, "model")[0]
    if device == "spike_detector":
        nest.ConvergentConnect(record_from, recorder)
    elif device in ('voltmeter', 'conductancemeter'):        
        nest.DivergentConnect(recorder, record_from)
    else:
        raise Exception("Not a valid recording device")

def _record(variable, source, filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    device_name = recording_device_names[variable]
    
    recording_device = nest.Create(device_name)
    nest.SetStatus(recording_device,
                   {"to_file" : True, "withgid" : True, "withtime" : True,
                    "interval": nest.GetStatus([0], "resolution")[0]})
    print "Trying to record %s from cell %s using %s %s (filename=%s)" % (variable, source, device_name, recording_device, filename)
            
    if type(source) != types.ListType:
        source = [source]
    _connect_recording_device(recording_device, record_from=source)
    if filename is not None:
        recorder_dict[filename] = recording_device 

def record(source, filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    _record('spikes', source, filename) 
    
def record_v(source, filename):
    """
    Record membrane potential to a file. source can be an individual cell or
    a list of cells."""
    # would actually like to be able to record to an array and
    # choose later whether to write to a file.
    _record('v', source, filename) 

def _print(user_filename, gather=True, compatible_output=True, population=None, variable=None):
    global recorder_dict
    
    if population is None:
        recorder = recorder_dict[user_filename]
    else:
        assert variable in ['spikes', 'v', 'conductance']
        recorder = population.recorders[variable]
    
    print "Printing to %s from recorder %s (compatible_output=%s)" % (user_filename, recorder,
                                                                      compatible_output)
    
    nest.FlushDevice(recorder) 
    status = nest.GetStatus([0])[0]
    local_num_threads = status['local_num_threads']
    node_list = range(nest.GetStatus([0], "num_processes")[0])
    
    # First combine data from different threads
    os.system("rm -f %s" % user_filename)
    for nest_thread in range(local_num_threads):
        nest.sps(recorder[0])
        nest.sr("%i GetAddress %i append" % (recorder[0], nest_thread))
        nest.sr("GetStatus /filename get")
        nest_filename = nest.spp() #nest.GetStatus(recorder, "filename")
        ###os.system("cat %s" % nest_filename)
        ##system_line = 'cat %s >> %s' % (nest_filename, "%s_%d" % (user_filename, nest.Rank()))
        merged_filename = "%s/%s" % (os.path.dirname(nest_filename), user_filename)
        system_line = 'cat %s >> %s' % (nest_filename, merged_filename) # will fail if writing to a common directory, e.g. using NFS
        print system_line
        os.system(system_line)
    if gather and len(node_list) > 1:
        raise Warning("'gather' not currently supported.")
        if nest.Rank() == 0: # only on the master node (?)
            for node in node_list:
                pass # not a good way to do it at the moment
    
    if compatible_output:
        if gather == False or nest.Rank() == 0: # if we gather, only do this on the master node
            logging.info("Writing %s in compatible format." % user_filename)
            
            # Here we postprocess the file to have effectively the
            # desired format: spiketime (in ms) cell_id-min(cell_id)
            #if not os.path.exists(user_filename):
            result = open(user_filename,'w',1000)
            #else:
            #    result = open(user_filename,'a',1000)
                
            ## Writing header info (e.g., dimensions of the population)
            if population is not None:
                result.write("# " + "\t".join([str(d) for d in population.dim]) + "\n")
                padding = population.cell.flatten()[0]
            else:
                padding = 0
            result.write("# dt = %g\n" % nest.GetStatus([0], "resolution")[0])
                            
            # Writing spiketimes, cell_id-min(cell_id)
                 
            # (Pylab has a great load() function, but it is not necessary to import
            # it into pyNN. The fromfile() function of numpy has trouble on several
            # machine with Python 2.5, so that's why a dedicated _readArray function
            # has been created to load from file the raster or the membrane potentials
            # saved by NEST).
                    
            # open file
            if int(os.path.getsize(merged_filename)) > 0:
                data = _readArray(merged_filename, sepchar=None)
                result.write("# n = %d\n" % len(data))
                data[:,0] = data[:,0] - padding
                # sort
                indx = data.argsort(axis=0, kind='mergesort')[:,0] # will quicksort (not stable) work?
                data = data[indx]
                if data.shape[1] == 4: # conductance files
                    raise Exception("Not yet implemented")
                elif data.shape[1] == 3: # voltage files
                    for idx in xrange(len(data)):
                        result.write("%g\t%d\n" % (data[idx][2], data[idx][0])) # v id
                elif data.shape[1] == 2: # spike files
                    for idx in xrange(len(data)):
                        result.write("%g\t%d\n" % (data[idx][1], data[idx][0])) # time id
                else:
                    raise Exception("Data file should have 2,3 or 4 columns, actually has %d" % data.shape[1])
            else:
                logging.info("%s is empty" % merged_filename)
            result.close()

    if population is None:
        recorder_dict.pop(user_filename)    

def _readArray(filename, sepchar=None, skipchar='#'):
    logging.debug(filename)
    myfile = open(filename, "r")
    contents = myfile.readlines()
    myfile.close()
    logging.debug(contents)
    data = []
    for line in contents:
        stripped_line = line.lstrip()
        if (len(stripped_line) != 0):
            if (stripped_line[0] != skipchar):
                items = stripped_line.split(sepchar)
                #if len(items) != 3:
                #    print stripped_line
                #    print items
                #    raise Exception()
                data.append(map(float, items))
    #try :
    a = numpy.array(data)
    #except Exception:
    #    raise
        # The last line has just a gid, so we has to remove it
        #a = numpy.array(data[0:len(data)-2])
    (Nrow,Ncol) = a.shape
    logging.debug(str(a.shape))
    if ((Nrow == 1) or (Ncol == 1)): a = ravel(a)
    return(a)

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
        
        common.Population.__init__(self,dims,cellclass,cellparams,label)  # move this to common.Population.__init__()
        
        # Should perhaps use "LayoutNetwork"?
        
        if isinstance(cellclass, type):
            self.celltype = cellclass(cellparams)
            self.cell = nest.Create(self.celltype.nest_name, self.size)
            self.cellparams = self.celltype.parameters
        elif isinstance(cellclass, str):
            self.cell = nest.Create(cellclass, self.size)

        
        self.cell = numpy.array([ ID(GID) for GID in self.cell ], ID)
        self.id_start = self.cell.reshape(self.size,)[0]
        
        for id in self.cell:
            id.parent = self
            #id.setCellClass(cellclass)
            #id.setPosition(self.locate(id))
            
        if self.cellparams:
            nest.SetStatus(self.cell, [self.cellparams])
            
        self.cell = numpy.reshape(self.cell, self.dim)    
        
        if not self.label:
            self.label = 'population%d' % Population.nPop
        self.recorders = {'spikes': None, 'v': None, 'conductance': None}
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
        if len(addr) == self.ndim:
            id = self.cell[addr]
        else:
            raise common.InvalidDimensionsError, "Population has %d dimensions. Address was %s" % (self.ndim,str(addr))
        if addr != self.locate(id):
            raise IndexError, 'Invalid cell address %s' % str(addr)
        return id
    
    def __len__(self):
        """Returns the total number of cells in the population."""
        return self.size
    
    def __iter__(self):
        return self.cell.flat

    def __address_gen(self):
        """
        Generator to produce an iterator over all cells on this node,
        returning addresses.
        """
        for i in self.__iter__():
            yield self.locate(i)
        
    def addresses(self):
        return self.__address_gen()
    
    def ids(self):
        return self.__iter__()
    
    def locate(self, id):
        """Given an element id in a Population, return the coordinates.
               e.g. for  4 6  , element 2 has coordinates (1,0) and value 7
                         7 9
        """
        # The top two lines (commented out) are the original implementation,
        # which does not scale well when the population size gets large.
        # The next lines are the neuron implementation of the same method. This
        # assumes that the id values in self.cell are consecutive. This should
        # always be the case, I think? A unit test is needed to check this.
    
        ###assert isinstance(id,int)
        ###return tuple([a.tolist()[0] for a in numpy.where(self.cell == id)])
        
        id -= self.id_start
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
    
    def set(self,param,val=None):
        """
        Set one or more parameters for every cell in the population. param
        can be a dict, in which case val should not be supplied, or a string
        giving the parameter name, in which case val is the parameter value.
        val can be a numeric value, or list of such (e.g. for setting spike times).
        e.g. p.set("tau_m",20.0).
             p.set({'tau_m':20,'v_rest':-65})
        """
        if isinstance(param,str):
            if isinstance(val,str) or isinstance(val,float) or isinstance(val,int):
                paramDict = {param:float(val)}
            else:
                raise common.InvalidParameterValueError
        elif isinstance(param,dict):
            paramDict = param
        else:
            raise common.InvalidParameterValueError
        if isinstance(self.celltype, common.StandardCellType):
            paramDict = self.celltype.translate(paramDict)
        nest.SetStatus(numpy.reshape(self.cell,(self.size,)), [paramDict])
        

    def tset(self,parametername,valueArray):
        """
        'Topographic' set. Sets the value of parametername to the values in
        valueArray, which must have the same dimensions as the Population.
        """
        # Convert everything to 1D arrays
        cells = numpy.reshape(self.cell,self.cell.size)
        if self.cell.shape == valueArray.shape: # the values are numbers or non-array objects
            values = numpy.reshape(valueArray,self.cell.size)
        elif len(valueArray.shape) == len(self.cell.shape)+1: # the values are themselves 1D arrays
            values = numpy.reshape(valueArray,(self.cell.size,valueArray.size/self.cell.size))
        else:
            raise common.InvalidDimensionsError, "Population: %s, valueArray: %s" % (str(cells.shape), str(valueArray.shape))
        # Translate the parameter name
        if isinstance(self.celltype, common.StandardCellType):
            parametername = self.celltype.translate({parametername: values[0]}).keys()[0]
        # Set the values for each cell
        if len(cells) == len(values):
            for cell,val in zip(cells,values):
                try:
                    if not isinstance(val,str) and hasattr(val,"__len__"):
                        val = list(val) # tuples, arrays are all converted to lists, since this is what SpikeSourceArray expects. This is not very robust though - we might want to add things that do accept arrays.
                    else:
                        nest.SetStatus([cell],[{parametername: val}])
                except nest.SLIError:
                    raise common.InvalidParameterValueError, "Error from SLI"
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
            raise Exception('rset() not yet implemented for NativeRNG')
        else:
            rarr = rand_distr.next(n=self.size)
            cells = numpy.reshape(self.cell,self.cell.size)
            assert len(rarr) == len(cells)
            for cell,val in zip(cells,rarr):
                try:
                    nest.SetStatus([cell],{parametername: val})
                except nest.SLIError:
                    raise common.InvalidParameterValueError
            
    def _call(self,methodname,arguments):
        """
        Calls the method methodname(arguments) for every cell in the population.
        e.g. p.call("set_background","0.1") if the cell class has a method
        set_background().
        """
        raise Exception("Method not yet implemented")
    
    def _tcall(self,methodname,objarr):
        """
        `Topographic' call. Calls the method methodname() for every cell in the 
        population. The argument to the method depends on the coordinates of the
        cell. objarr is an array with the same dimensions as the Population.
        e.g. p.tcall("memb_init",vinitArray) calls
        p.cell[i][j].memb_init(vInitArray[i][j]) for all i,j.
        """
        raise Exception("Method not yet implemented")

    def _record(self, variable, record_from=None, rng=None):
        assert variable in ('spikes', 'v', 'conductance')
    
        # create device
        device_name = recording_device_names[variable]
        if self.recorders[variable] is None:
            self.recorders[variable] = nest.Create(device_name)
            nest.SetStatus(self.recorders[variable],
                           {"to_file" : True, "withgid" : True, "withtime" : True,
                            "interval": nest.GetStatus([0], "resolution")[0]})      
        
        # create list of neurons        
        fixed_list = False
        if record_from:
            if type(record_from) == types.ListType:
                fixed_list = True
                n_rec = len(record_from)
            elif type(record_from) == types.IntType:
                n_rec = record_from
            else:
                raise "record_from must be a list or an integer"
        else:
            n_rec = self.size
            
        if variable == 'spikes':
            self.n_rec = n_rec
        
        tmp_list = []
        if (fixed_list == True):
            for neuron in record_from:
                tmp_list = [neuron for neuron in record_from]
        else:
            # should use `rng` here, if provided
            for neuron in numpy.random.permutation(numpy.reshape(self.cell,(self.cell.size,)))[0:n_rec]:
                tmp_list.append(neuron)
                
        # connect device to neurons
        _connect_recording_device(self.recorders[variable], record_from=tmp_list)

    def record(self, record_from=None, rng=None):
        """
        If record_from is not given, record spikes from all cells in the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids
        of the cells to record.
        """
        self._record('spikes', record_from, rng)
        
    def record_v(self,record_from=None,rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        self._record('v', record_from, rng)
    
    def record_c(self,record_from=None,rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        self._record('conductance', record_from, rng)
    
    def printSpikes(self, filename, gather=True, compatible_output=True):
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
        is used. This may be faster.
        If gather is True, the file will only be created on the master node,
        otherwise, a file will be written on each node.
        """
        _print(filename, gather=gather, compatible_output=compatible_output,
               population=self, variable="spikes")
       
    def meanSpikeCount(self, gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        # gather is not relevant, but is needed for API consistency
        n_spikes = nest.GetStatus(self.recorders['spikes'], "events")[0]
        return float(n_spikes)/self.n_rec

    def randomInit(self, rand_distr):
        """
        Sets initial membrane potentials for all the cells in the population to
        random values.
        """
        self.rset('v_init',rand_distr)

    def print_v(self, filename, gather=True, compatible_output=True):
        """
        Write membrane potential traces to file.
        If compatible_output is True, the format is "t v cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        The timestep and number of data points per cell is written as a header,
        indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        voltage files.
        """
        _print(filename, gather=gather, compatible_output=compatible_output,
               population=self, variable="v")
               
    def print_c(self,filename,gather=True, compatible_output=True):
        """
        Write conductance traces to file.
        If compatible_output is True, the format is "t g cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        The timestep and number of data points per cell is written as a header,
        indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        voltage files.
        """
        _print(filename, gather=gather, compatible_output=compatible_output,
               population=self, variable="conductance")

    
class Projection(common.Projection):
    """
    A container for all the connections between two populations, together with
    methods to set parameters of those connections, including of plasticity
    mechanisms.
    """
    
    class ConnectionDict:
            
            def __init__(self,parent):
                self.parent = parent
    
            def __getitem__(self,id):
                """Returns a (source address,target port number) tuple."""
                assert isinstance(id, int)
                return (self.parent._sources[id], self.parent._targetPorts[id])
    
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
        common.Projection.__init__(self,presynaptic_population,postsynaptic_population,method,methodParameters,source,target,label,rng)
        
        self._targetPorts = [] # holds port numbers
        self._targets = []     # holds gids
        self._sources = []     # holds gids
        self.synapse_type = target
        
        if isinstance(method, str):
            connection_method = getattr(self,'_%s' % method)   
            self.nconn = connection_method(methodParameters)
        elif isinstance(method,common.Connector):
            self.nconn = method.connect(self)

        #assert len(self._sources) == len(self._targets) == len(self._targetPorts), "Connection error. Source and target lists are of different lengths."
        self.connection = Projection.ConnectionDict(self)
    
    def __len__(self):
        """Return the total number of connections."""
        return len(self._sources)
    
    def connections(self):
        """for conn in prj.connections()..."""
        for i in xrange(len(self)):
            yield self.connection[i]
    
    # --- Connection methods ---------------------------------------------------
    
    def _allToAll(self,parameters=None):
        """
        Connect all cells in the presynaptic population to all cells in the postsynaptic population.
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
        c = FixedProbabilityConnector(p_connect, allow_self_connections)
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
        c = DistanceDependentProbabilityConnector(d_expression, allow_self_connections=allow_self_connections)
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
         
        postsynaptic_neurons = numpy.reshape(self.post.cell,(self.post.cell.size,))
        presynaptic_neurons  = numpy.reshape(self.pre.cell,(self.pre.cell.size,))
        if self.rng:
            rng = self.rng
        else:
            rng = numpy.random
        for pre in presynaptic_neurons:
            pre_addr = nest.getAddress(pre)
            # Reserve space for connections
            if not fixed:
                n = rand_distr.next()
            nest.resCons(pre_addr,n)                
            # pick n neurons at random
            for post in rng.permutation(postsynaptic_neurons)[0:n]:
                if allow_self_connections or (pre != post):
                    self._sources.append(pre)
                    self._targets.append(post)
                    self._targetPorts.append(nest.connect(pre_addr,nest.getAddress(post)))
    
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
         
        postsynaptic_neurons = numpy.reshape(self.post.cell,(self.post.cell.size,))
        presynaptic_neurons  = numpy.reshape(self.pre.cell,(self.pre.cell.size,))
        if self.rng:
            rng = self.rng
        else:
            rng = numpy.random
        for post in postsynaptic_neurons:
            post_addr = nest.getAddress(post)
            # Reserve space for connections
            if not fixed:
                n = rand_distr.next()
            nest.resCons(post_addr,n)                
            # pick n neurons at random
            for pre in rng.permutation(presynaptic_neurons)[0:n]:
                if allow_self_connections or (pre != post):
                    self._sources.append(pre)
                    self._targets.append(post)
                    self._targetPorts.append(nest.connect(nest.getAddress(pre),post_addr))
    
    def _fromFile(self,parameters):
        """
        Load connections from a file.
        """
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
        
        self._fromList(input_tuples)
        
    def _fromList(self,conn_list):
        """
        Read connections from a list of tuples,
        containing [pre_addr, post_addr, weight, delay]
        where pre_addr and post_addr are both neuron addresses, i.e. tuples or
        lists containing the neuron array coordinates.
        """
        for i in xrange(len(conn_list)):
            src, tgt, weight, delay = conn_list[i][:]
            src = self.pre[tuple(src)]
            tgt = self.post[tuple(tgt)]
            pre_addr = nest.getAddress(src)
            post_addr = nest.getAddress(tgt)
            self._sources.append(src)
            self._targets.append(tgt)
            self._targetPorts.append(nest.connectWD(pre_addr,post_addr, 1000*weight, delay))

    def _2D_Gauss(self,parameters):
        """
        Source neuron is connected to a 2D targetd population with a spatial profile (Gauss).
        parameters should have:
        rng:
        source_position: x,y of source neuron mapped to target populatio.
        source_id: source id
        n: number of synpases
        sigma: sigma of the Gauss
        """
        def rcf_2D(parameters):
            rng = parameters['rng']
            pre_id = parameters['pre_id']
            pre_position = parameters['pre_position']
            n = parameters['n']
            sigma = parameters['sigma']
            weight = parameters['weight']
            delay = parameters['delay']
            
            phi = rng.uniform(size=n)*(2.0*pi)
            r = rng.normal(scale=sigma,size=n)
            target_position_x = numpy.floor(pre_position[1]+r*numpy.cos(phi))
            target_position_y = numpy.floor(pre_position[0]+r*numpy.sin(phi))
            target_id = []
            for syn_nr in range(len(target_position_x)):
                #print syn_nr
                try:
                    # print target_position_x[syn_nr]
                    target_id.append(self.post[(target_position_x[syn_nr],target_position_y[syn_nr])])
                    # print target_id
                except IndexError:
                    target_id.append(False)
            
            nest.divConnect(pre_id,target_id,[weight],[delay])
        
        
        n = parameters['n']
                
        if n > 0:
            ratio_dim_pre_post = ((1.*self.pre.dim[0])/(1.*self.post.dim[0]))
            print 'ratio_dim_pre_post',ratio_dim_pre_post
            run_id = 0

            for pre in numpy.reshape(self.pre.cell,(self.pre.cell.size)):
                #print 'pre',pre
                run_id +=1
                #print 'run_id',run_id
                if numpy.mod(run_id,500) == 0:
                    print 'run_id',run_id
                
                pre_position_tmp = self.pre.locate(pre)
                parameters['pre_position'] = numpy.divide(pre_position_tmp,ratio_dim_pre_post)
                parameters['pre_id'] = pre
                #a=Projection(self.pre,self.post,'rcf_2D',parameters)
                rcf_2D(parameters)

    def _test_delay(self,params):
        # debug get delays from outside
        #delay_array = parameters['delays_array']
        #weight_array = parameters['weights_array']
        #target_id = parameters['target_id']
        #pre_id = parameters['pre_id']
        print 'inside test_delay'
        print 'delays ',params['delays_array']
        print 'weights ',params['weights_array']
        print 'pre_id ',params['pre_id']
        print 'target_id ',params['target_id']
        eval(params['eval_string'])
        #cons=nest.divConnect(params['pre_id'],params['target_id'],params['weights_array'].tolist(),params['delays_array'].tolist())
        #nest.divConnect(pre_id,target_id,weight_array.tolist(),delay_array.tolist())
        print 'leaving test_delay'
        
    def _3D_Gauss(self,parameters):
        """
        Source neuron is connected to a 3D targetd population with a spatial profile (Gauss).
        parameters should have:
        rng:
        source_position: x,y of source neuron mapped to target populatio.
        source_id: source id
        n: number of synpases
        sigma: sigma of the Gauss
        """
        #def get_ids(self,parameters):
            #ids = []
            #if len(addrs) == self.ndim:
        #
        #for addr in range(len(parameters['x'])):
        #    try:
        #        ids = numpy.append(ids,post.cell[addr])
        #    except IndexError:
        #        pass
        #else:
        #    raise common.InvalidDimensionsError, "Population has %d dimensions. Address was %s" % (self.ndim,str(addrs))
        #return ids.astype('int')

        
        def rcf_3D(parameters):
            rng = parameters['rng']
            rng_params = parameters['rng_params']
            pre_id = parameters['pre_id']
            pre_position = parameters['pre_position']
            n = parameters['n']
            sigma = parameters['sigma']
            weight = parameters['weight']
            weight = weight*1000 # weights should be in nA or µS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
            # Using convention in this way is not ideal. We should be able to look up the units used by each model somewhere.
            
            min_delay_offset = parameters['min_delay_offset']
            post_dim = parameters['post_dim']
            params_dist = parameters['params_dist']
            size_in_mm = parameters['size_in_mm']
            #architecture = parameters['architecture']
            conduction_speed = parameters['conduction_speed']
            #min_delay = parameters['min_delay']
            
            phi = rng.uniform(size=n)*(2.0*pi)
            r = rng.normal(scale=sigma,size=n)
            # for z 
            #h = rng.uniform(size=n)*post_dim[2] # here post dim because it does not metter where it comes from in pre dim
            
            target_position_x = numpy.floor(pre_position[1]+r*numpy.cos(phi)).astype('int')
            target_position_y = numpy.floor(pre_position[0]+r*numpy.sin(phi)).astype('int')
            

            # because array[-1] gives you the last entrie, we have to get rid of the negative values, in either x or y
            valid_positions= eval('target_position_x >= 0')*eval('target_position_y >= 0')
            target_position_x = target_position_x[valid_positions]
            target_position_y = target_position_y[valid_positions]
            r = r[valid_positions]#  this is needed for the distant dependant delay 
            # new n
            n = len(target_position_x)
            
            # for z dim, here we dont have to remove unvalid pos, because the values can not be unvalid since it is unfiorm dis between the limits
            # however, n is reduced
            h = rng.uniform(size=n)*post_dim[2] # here post dim because it does not metter where it comes from in pre dim
            target_position_z = numpy.floor(h).astype('int')
            
            target_id = []
            # an array of bool, will be filled with True if synpase is on the grid, with False when outside
            target_id_bool = numpy.array([],dtype='bool')
            # __getitems__ version
            
            
            for syn_nr in range(len(target_position_x)):
                try:
                    target_id.append(self.post.cell[(target_position_x[syn_nr],target_position_y[syn_nr],target_position_z[syn_nr])])
                    #target_id.append(self.post[(target_position_x[syn_nr],target_position_y[syn_nr],target_position_z[syn_nr])])
                    target_id_bool = numpy.append(target_id_bool,True)
                    #target_id_bool.append(True)
                except IndexError:
                    target_id_bool = numpy.append(target_id_bool,False)
                    #target_id_bool.append(False)
                    #pass
                # some synapses fall outside the grid, they are lost
                    #target_id.append(False)
                    
            # number of synapses that are actually made
            n_syn = len(target_id) 
            # r will be used to calculate the distant dependent delay, but since some synapses have not been made, they have to be removed
            r_syn = numpy.abs(r[target_id_bool])
            # print 'min r_syn: ',r_syn.min()
            # print 'max r_syn: ',r_syn.max()
            # from r_syn we calculate the delay, with 0.1 m/s --> 0.1 mm/ms
            # r_syn is in units of population size, meaning in grid size, since sigma was around 0.4 of p.dim
            # we have to convert it to mm to get delay in mm/ms
            # we do this by: r_syn * architecture['a']/post_dim[0] , the x dim of the post synaptic population
            # should give us: r_syn in mm, since r_syn [in grid units], post_dim[0] [max grid units],   architecture['a'] is in mm
            # print 'size_in_mm: ',size_in_mm
            # print 'post_dim[0]: ',post_dim[0]
            # print 'conduction_speed: ',conduction_speed
            # print 'params dist: ',params_dist
            # units*mm/units
            r_syn2 = r_syn*size_in_mm/numpy.float(post_dim[0])
            #print 'type r_syn',type(r_syn)
            # there is a min dely, which is around 1.0 ms, params['min_delay'] see Markam 97
            min_delay_offset_array = rng_params.normal(loc=min_delay_offset,scale=abs(min_delay_offset*params_dist),size=n_syn)
            #print 'type min_delay',type(min_delay_array)
            # print 'min min_delay_arrayr_syn: ',min_delay_array.min()
            # print 'max min_delay_array: ',min_delay_array.max()
            delay_array = numpy.add(r_syn2/conduction_speed,min_delay_offset_array)
            #delay_array=delay_array.round(decimals=2)
            #print 'type delay_array',type(delay_array)
            #print 'min delay_arrayr: ',delay_array.min()
            #print 'max delay_array: ',delay_array.max()
            #print 'min delay_arrayr.tolsit(): ',min(delay_array.tolist())
            #print 'max delay_array.tolist(): ',max(delay_array.tolist())
            #print 'params dist', params_dist
            #if params_dist >0:
                #print 'with dist'
            # print 'n_syn',n_syn
            # print 'abs(weight*params_dist) ',abs(weight*params_dist)
            # print 'weight ', weight
            
            
            weight_array = rng_params.normal(loc=weight,scale=abs(weight*params_dist),size=n_syn)
            #weight_array=weight_array.round(decimals=2)
                #delay_list = rng_params.normal(loc=delay,scale=delay/params_dist,size=n_syn)
            #delay_list = r_syn/transmission_speed

            
            # debug get delays from outside
            #delay_array = parameters['delays_array']
            #weight_array = parameters['weights_array']
            #target_id = parameters['target_id']
            #if pre_id==100:
            #    print '#############################################'
            #    print 'This is the data in 3D Gauss'
            #    print '#############################################'
            #    print 'preneuron id',pre_id
                #print 'r_syn: ',r_syn
                #print 'r_syn2',r_syn2
                #print 'min_delay_offset_array ',min_delay_offset_array
            #    print 'delay_array ',delay_array
            #    print 'type first element of delay ',type(delay_array[0])
            #    print 'weight ',weight_array
            #    print 'type first element of weight ',type(weight_array[0])
            #    print 'now we dive into nest.hl_api.... yeah'
            #    print '#############################################'
            #    print '\n'
                #print 'size_in_mm: ',size_in_mm
                #print 'post_dim[0]: ',post_dim[0]
                #print 'conduction_speed: ',conduction_speed
                #print 'params dist: ',params_dist
            #    printed = True
            #    done = True
            #print 'len delay == len weigth',len(delay_array.tolist())==len(weight_array.tolist())
                #print 'len weigth ', len(weight_array.tolist())
                #print 'delay list: ',type(delay_array.tolist())

            #nest.divConnect(pre_id,target_id,weight_array.tolist(),delay_array.tolist())
            #delays_array = rng.normal(loc=nest.getDict([0])[0]['max_delay']/2.,scale=abs(nest.getDict([0])[0]['max_delay']/2.*params_dist),size=n)
            #weights_array = delays_array
            
            nest.divConnect(pre_id,target_id,weight_array.tolist(),delay_array.tolist())
            
            #return delay_array
            #else:
                #print 'no dist'
            #    nest.divConnect(pre_id,target_id,[weight],[delay])
        
        
        n = parameters['n']
        #global printed
        #printed = False
        #done = False
        if n > 0:
            ratio_dim_pre_post = ((1.*self.pre.dim[0])/(1.*self.post.dim[0]))
            #print 'ratio_dim_pre_post',ratio_dim_pre_post
            run_id = 0
            
            for pre in numpy.reshape(self.pre.cell,(self.pre.cell.size)):
                #if done:
                #    return
                #print 'pre',pre
                run_id +=1
                #print 'run_id',run_id
                if numpy.mod(run_id,500) == 0:
                    print 'run_id',run_id
                
                pre_position_tmp = self.pre.locate(pre)
                parameters['pre_position'] = numpy.divide(pre_position_tmp,ratio_dim_pre_post)
                parameters['pre_id'] = pre
                parameters['post_dim'] = self.post.dim
                #a=Projection(self.pre,self.post,'rcf_2D',parameters)
                rcf_3D(parameters)
                #if done:
                #    return
        

    
    # --- Methods for setting connection parameters ----------------------------
    
    def setWeights(self, w):
        """
        w can be a single number, in which case all weights are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        Weights should be in nA for current-based and µS for conductance-based
        synapses.
        """
        w = w*1000.0 # weights should be in nA or µS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
                     # Using convention in this way is not ideal. We should be able to look up the units used by each model somewhere.
        if self.synapse_type == 'inhibitory' and w > 0:
            w *= -1
        if type(w) == types.FloatType or type(w) == types.IntType or type(w) == numpy.float64 :
            # set all the weights from a given node at once
            for src in self.pre.cell.flat:
                conn_dict = nest.GetConnections([src], 'static_synapse')[0]
                if conn_dict:
                    n = len(conn_dict['weights'])
                nest.SetConnections([src], 'static_synapse', [{'weights': [w]*n}])
        elif isinstance(w,list) or isinstance(w,numpy.ndarray):
            raise Exception("Not yet implemented")
        else:
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")
    
    def randomizeWeights(self,rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        for src in self.pre.cell.flat:
            conn_dict = nest.GetConnections([src], 'static_synapse')[0]
            n = len(conn_dict['weights'])
            weights = 1000.0*rand_distr.next(n)
            if n == 1:
                weights = [weights]
            else:
                weights = weights.tolist()    
            # if self.synapse_type == 'inhibitory', should we *= -1 ???
            nest.SetConnections([src], 'static_synapse', [{'weights': weights}])
    
    def setDelays(self,d):
        """
        d can be a single number, in which case all delays are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        """
        if type(d) == types.FloatType or type(d) == types.IntType or type(d) == numpy.float64:
            d = float(d)
            # set all the weights from a given node at once
            for src in self.pre.cell.flat:
                conn_dict = nest.GetConnections([src], 'static_synapse')[0]
                if conn_dict:
                    n = len(conn_dict['delays'])
                nest.SetConnections([src], 'static_synapse', [{'delay': [d]*n}])
        elif isinstance(d,list) or isinstance(d,numpy.ndarray):
            raise Exception("Not yet implemented")
        else:
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")
    
    def randomizeDelays(self,rand_distr):
        """
        Set delays to random values taken from rand_distr.
        """
        for src in self.pre.cell.flat:
            conn_dict = nest.GetConnections([src], 'static_synapse')[0]
            n = len(conn_dict['delays'])
            delays = 1.0*rand_distr.next(n)
            if n == 1:
                delays = [delays]
            else:
                delays = delays.tolist()
            nest.SetConnections([src], 'static_synapse', [{'delays': delays}])
    
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
        f = open(filename,'w',1000)
        # Note unit change from pA to nA or nS to uS, depending on synapse type
        weights = [0.001*nest.getWeight(src,port) for (src,port) in self.connections()]
        delays = [nest.getDelay(src,port) for (src,port) in self.connections()] 
        fmt = "%s%s\t%s%s\t%s\t%s\n" % (self.pre.label,"%s",self.post.label,"%s","%g","%g")
        for i in xrange(len(self)):
            line = fmt  % (self.pre.locate(self._sources[i]),
                           self.post.locate(self._targets[i]),
                           weights[i],
                           delays[i])
            line = line.replace('(','[').replace(')',']')
            f.write(line)
        f.close()
    
    def printWeights(self,filename,format=None,gather=True):
        """Print synaptic weights to file."""
        file = open(filename,'w',1000)
        postsynaptic_neurons = numpy.reshape(self.post.cell,(self.post.cell.size,)).tolist()
        presynaptic_neurons  = numpy.reshape(self.pre.cell,(self.pre.cell.size,)).tolist()
        weightArray = numpy.zeros((self.pre.size,self.post.size),dtype=float)
        for src in self._sources:
            src_addr = nest.getAddress(src)
            nest.sps(src_addr)
            nest.sr('GetTargets')
            targetList = [nest.getGID(tgt) for tgt in nest.spp()]
            nest.sps(src_addr)
            nest.sr('GetWeights')
            weightList = nest.spp()
            
            i = presynaptic_neurons.index(src)
            for tgt,w in zip(targetList,weightList):
                try:
                    j = postsynaptic_neurons.index(tgt)
                    weightArray[i][j] = w
                except ValueError: # tgt is in a different population to the current postsynaptic population
                    pass
        fmt = "%g "*len(postsynaptic_neurons) + "\n"
        for i in xrange(weightArray.shape[0]):
            file.write(fmt % tuple(weightArray[i]))
        file.close()
            
    
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
#   Connection method classes
# ==============================================================================

class AllToAllConnector(common.AllToAllConnector):    
    
    def connect(self, projection):
        postsynaptic_neurons  = projection.post.cell.flatten()
        target_list = postsynaptic_neurons.tolist()
        for pre in projection.pre.cell.flat:
            # if self connections are not allowed, check whether pre and post are the same
            if not self.allow_self_connections:
                target_list = postsynaptic_neurons.tolist()
                if pre in target_list:
                    target_list.remove(pre)
            projection._targets += target_list
            projection._sources += [pre]*len(target_list) 
            conn_dict = nest.GetConnections([pre], 'static_synapse')[0]
            if conn_dict:
                first_port = len(conn_dict['targets'])
            else:
                first_port = 0
            projection._targetPorts += range(first_port, first_port+len(target_list))
            nest.DivergentConnectWD([pre], target_list, [1000.0], [_min_delay])
        return len(projection._targets)

class OneToOneConnector(common.OneToOneConnector):
    
    def connect(self, projection):
        if projection.pre.dim == projection.post.dim:
            projection._sources = projection.pre.cell.flatten()
            projection._targets = projection.post.cell.flatten()
            for pre,post in zip(projection._sources,projection._targets):
                #projection._targetPorts.append(nest.connect(pre_addr,post_addr))
                nest.ConnectWD([pre], [post], [1000.0], [_min_delay])
            return projection.pre.size
        else:
            raise Exception("Connection method not yet implemented for the case where presynaptic and postsynaptic Populations have different sizes.")
    
class FixedProbabilityConnector(common.FixedProbabilityConnector):
    
    def connect(self, projection):
        #postsynaptic_neurons = numpy.reshape(projection.post.cell,(projection.post.cell.size,))
        presynaptic_neurons  = projection.pre.cell.flatten()
        npre = projection.pre.size
        for post in projection.post.cell.flat:
            if projection.rng:
                rarr = projection.rng.uniform(0,1,(npre,)) # what about NativeRNG?
            else:
                rarr = numpy.random.uniform(0,1,(npre,))
            source_list = numpy.compress(numpy.less(rarr,self.p_connect),presynaptic_neurons).tolist()
            # if self connections are not allowed, check whether pre and post are the same
            if not self.allow_self_connections and post in source_list:
                source_list.remove(post)
            projection._targets += [post]*len(source_list)
            projection._sources += source_list
            #projection._targetPorts += nest.convergentConnect(source_list,post,[1.0],[0.1])
            nest.convergentConnect(source_list, [post], [1000.0], [_min_delay])
        return len(projection._sources)
    
class DistanceDependentProbabilityConnector(common.DistanceDependentProbabilityConnector):
    
    def connect(self, projection):                  
        postsynaptic_neurons = projection.post.cell.flat # iterator
        presynaptic_neurons  = projection.pre.cell.flatten() # array
        # what about NativeRNG?
        if projection.rng:
            if isinstance(projection.rng, NativeRNG):
                print "Warning: use of NativeRNG not implemented. Using NumpyRNG"
                rarr = numpy.random.uniform(0,1,(projection.pre.size*projection.post.size,))
            else:
                rarr = projection.rng.uniform(0,1,(projection.pre.size*projection.post.size,))
        else:
            rarr = numpy.random.uniform(0,1,(projection.pre.size*projection.post.size,))
        j = 0
        for post in postsynaptic_neurons:
            for pre in presynaptic_neurons:
                if self.allow_self_connections or pre != post: 
                    # calculate the distance between the two cells :
                    d = common.distance(pre, post, self.mask, self.scale_factor)
                    p = eval(self.d_expression)
                    if p >= 1 or (0 < p < 1 and rarr[j] < p):
                        projection._sources.append(pre)
                        projection._targets.append(post)
                        #projection._targetPorts.append(nest.connect(pre_addr,post_addr))
                        nest.Connect(pre,post, [1000.0], [_min_delay])
                j += 1
        return len(projection._sources)


# ==============================================================================
#   Utility classes
# ==============================================================================
   
Timer = common.Timer

# ==============================================================================
