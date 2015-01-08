"""
A small framework to make it easier to run the same model on multiple
simulators.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from multiprocessing import Process, Queue

def run_simulation(network_model, sim, parameters, input_queue, output_queue):
    """
    Build the model defined in the class `network_model`, with parameters
    `parameters`, and then consume tasks from `input_queue` until receiving the
    command 'STOP'.
    """
    print("Running simulation with %s" % sim.__name__)
    network = network_model(sim, parameters)
    print("Network constructed with %s." % sim.__name__)
    for obj_name, attr, args, kwargs in iter(input_queue.get, 'STOP'):
        print("%s processing command %s.%s(%s, %s)" % (sim.__name__, obj_name, attr, args, kwargs))
        obj = eval(obj_name)
        result = getattr(obj, attr)(*args, **kwargs)
        output_queue.put(result)
    print("Simulation with %s complete" % sim.__name__)
    #sim.end()


class MultiSim(object):
    """
    Interface that runs a network model on different simulators, with each
    simulation in a separate process.
    """
    
    def __init__(self, sim_list, network_model, parameters):
        """
        Build the model defined in the class `network_model`, with parameters
        `parameters`, for each of the simulator modules specified in `sim_list`.
        
        The `network_model` constructor takes arguments `sim` and `parameters`.
        """
        self.processes = {}
        self.task_queues = {}
        self.result_queues = {}
        for sim in sim_list:
            task_queue = Queue()
            result_queue = Queue()
            p = Process(target=run_simulation,
                        args=(network_model, sim, parameters, task_queue, result_queue))
            p.start()
            self.processes[sim.__name__] = p
            self.task_queues[sim.__name__] = task_queue
            self.result_queues[sim.__name__] = result_queue
            
    def __iter__(self):
        return self.processes.itervalues()
    
    def __getattr__(self, name):
        """
        Assumes `name` is a method of the `network_model` model.
        Return a function that runs `net.name()` for all the simulators.
        """
        def iterate_over_nets(*args, **kwargs):
            retvals = {}
            for sim_name in self.processes:
                self.task_queues[sim_name].put(('network', name, args, kwargs))
            for sim_name in self.processes:
                retvals[sim_name] = self.result_queues[sim_name].get()
            return retvals
        return iterate_over_nets
            
    def run(self, simtime, steps=1): #, *callbacks):
        """
        Run the model for a time `simtime` in all simulators.
        
        The run may be broken into a number of steps (each of equal duration).
        #Any functions in `callbacks` will be called after each step.
        """
        dt = float(simtime)/steps
        for i in range(steps):
            for sim_name in self.processes:
                self.task_queues[sim_name].put(('sim', 'run', [dt], {}))
            for sim_name in self.processes:
                t = self.result_queues[sim_name].get()
                
    def end(self):
        for sim_name in self.processes:
            self.task_queues[sim_name].put('STOP')
            self.processes[sim_name].join()
