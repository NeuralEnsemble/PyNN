


class MultiSim(object):
    
    def __init__(self, sim_list, net, parameters):
        self.sim_list = sim_list
        self.nets = {}
        for sim in sim_list:
            self.nets[sim.__name__] = net(sim, parameters)
            
    def __iter__(self):
        return self.nets.itervalues()
    
    def __getattr__(self, name):
        def iterate_over_nets(*args, **kwargs):
            retvals = {}
            for sim_name, net in self.nets.items():
                retvals[sim_name] = getattr(net, name)(*args, **kwargs)
            return retvals
        return iterate_over_nets
            
    def run(self, simtime, steps=1, *callbacks):
        dt = float(simtime)/steps
        for i in range(steps):
            for sim in self.sim_list:
                sim.run(dt)
            for func in callbacks:
                func()