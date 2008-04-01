# global pypcsim objects used throughout simulation
class PyPCSIM_GLOBALS:    
    net = None
    dt = None
    minDelay = None
    maxDelay = None
    constructRNGSeed = None
    simulationRNGSeed = None
    spikes_multi_rec = {}
    vm_multi_rec = {}
    pass

pcsim_globals = PyPCSIM_GLOBALS()