from pyNN import common

name = "MockSimulator"

class ID(int, common.IDMixin):
    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)

class State(common.control.BaseState):
    def __init__(self):
        common.control.BaseState.__init__(self)
        self.mpi_rank = 0
        self.num_processes = 1
        self.clear()
        self.dt = 0.1
    def run(self, simtime):
        self.t += simtime
        self.running = True
    def run_until(self, tstop):
        self.t = tstop
        self.running = True
    def clear(self):
        self.recorders = set([])
        self.id_counter = 42
        self.segment_counter = -1
        self.reset()
    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.t = 0
        self.t_start = 0
        self.segment_counter += 1

state = State()
