
spikes_multi_rec = {}
vm_multi_rec = {}

class _State(object):
    """Represent the simulator state."""
    
    def __init__(self):
        self.initialized = False
        self.t = 0.0
        self.dt = None
        self.min_delay = None
        self.max_delay = None
        self.constructRNGSeed = None
        self.simulationRNGSeed = None
    
    @property
    def num_processes(self):
        return net.mpi_size()
    
    @property
    def mpi_rank(self):
        return net.mpi_rank()


class Connection(object):
    
    def __init__(self, pcsim_connection, weight_unit_factor):
        self.pcsim_connection = pcsim_connection
        self.weight_unit_factor = weight_unit_factor
        
    @property
    def weight(self):
        return self.weight_unit_factor*self.pcsim_connection.W
    
    @property
    def delay(self):
        return 1000.0*self.pcsim_connection.delay # s --> ms
    

class ConnectionManager(object):
    """docstring needed."""

    def __init__(self, synapse_model='static_synapse', parent=None):
        self.parent = parent

    def __getitem__(self, i):
        """Returns a Connection object."""
        if self.parent.is_conductance:
            A = 1e6 # S --> uS
        else:
            A = 1e9 # A --> nA
        return Connection(self.parent.pcsim_projection.object(i), A)
    
    def __len__(self):
        return self.parent.pcsim_projection.size()
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
            
            
net = None
state = _State()
del _State