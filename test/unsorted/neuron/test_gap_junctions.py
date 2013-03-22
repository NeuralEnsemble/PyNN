import sys
import os.path
from neuron import h, nrn
from pyNN.neuron.cells import SingleCompartmentNeuron
import pyNN.neuron as sim

class TestCell(SingleCompartmentNeuron):
    parameter_names = ['syn_type', 'syn_shape', "c_m", "i_offset",
                  "tau_e", "tau_i", "e_e", "e_i", "source_section"]
    parameters = {'syn_type': 'conductance', 'syn_shape':'exp', "c_m":1.0, "i_offset":0,
                  "tau_e":2, "tau_i":2, "e_e":0, "e_i":-75}
    recordable = ['spikes', 'v']
    default_initial_values = {
        'v':-70.0, #'v_rest',
    }
    conductance_based = True

    def __init__(self, params):
        SingleCompartmentNeuron.__init__(self, 'conductance', 'exp', params['c_m'], params['i_offset'],
                                         params['tau_e'], params['tau_i'], params['e_e'], params['e_i'])
        self.source = self.seg._ref_v
        self.model = TestCellModel # What is the advantage of separating the model from the Model class?
        self.insert('pas')

    def get_threshold(self):
        return 10.0

    def record(self, active):
        if active:
            rec = h.NetCon(self.source, None, sec=self)
            rec.record(self.spike_times)

    def has_parameter(self, param):
        return param in self.parameters.keys()

    def get_parameter_names(self):
        return self.parameter_names

class TestCellModel(TestCell):

    def __init__(self, **params):
        TestCell.__init__(self, params)

cell_params = {
    "c_m": 1.0,
    "i_offset": 0.0,
    "v_init":-65.0,
    "tau_e": 2.0,
    "tau_i": 2.0,
    "e_e": 0.0,
    "e_i":-75.0,
}

sim.setup(timestep=0.1, min_delay=0.2, max_delay=10.0)
spike_input = sim.Population(1, sim.SpikeSourcePoisson, {'rate': 100.0})
source = sim.Population(10, TestCell, cell_params)
target1 = sim.Population(10, TestCell, cell_params)
target2 = sim.Population(10, TestCell, cell_params)
target3 = sim.Population(10, TestCell, cell_params)
target4 = sim.Population(10, TestCell, cell_params)
target5 = sim.Population(10, TestCell, cell_params)
#target6 = sim.Population(10, TestCell, cell_params)
#target7 = sim.Population(10, TestCell, cell_params)
source.record_v()
target1.record_v()
target2.record_v()
target3.record_v()
target4.record_v()
target5.record_v()
#target6.record_v()
#target7.record_v()
spike_proj = sim.Projection(spike_input, source, sim.AllToAllConnector(weights=1.0))
# The 'AllToAllConnector' is based on ProbabilisticConnector so all connectors based on 
# ProbabilisticConnector should work if it does
gap1 = sim.GapJunctionProjection(source, target1, sim.AllToAllConnector())
gap2 = sim.GapJunctionProjection(source, target2, sim.OneToOneConnector())
connections_list = [(0, 9, 3.0, 0.0),
                    (0, 8, 3.0, 0.0),
                    (0, 7, 3.0, 0.0),
                    (0, 6, 3.0, 0.0),
                    (1, 8, 3.0, 0.0),
                    (2, 7, 3.0, 0.0),
                    (3, 6, 3.0, 0.0),
                    (4, 5, 3.0, 0.0),
                    (5, 4, 3.0, 0.0),
                    (6, 3, 3.0, 0.0),
                    (7, 2, 3.0, 0.0),
                    (8, 1, 3.0, 0.0),
                    (9, 0, 3.0, 0.0)]
gap3 = sim.GapJunctionProjection(source, target3, sim.FromListConnector(connections_list))
gap4 = sim.GapJunctionProjection(source, target4, sim.FixedNumberPostConnector(3))
gap5 = sim.GapJunctionProjection(source, target5, sim.FixedNumberPreConnector(4))
#gap6 = sim.GapJunctionProjection(source, target6, sim.SmallWorldConnector(1, 0.5)) # This actually works on a single node but not over MPI
#gap7 = sim.GapJunctionProjection(source, target7, sim.CSAConnector())
gap1.set('weight', 1.0)
gap2.set('weight', 2.0)
gap4.set('weight', 4.0)
gap5.set('weight', 5.0)
#gap6.set('weight', 6.0)
#gap7.set('weight', 7.0)
sim.run(1000.0)
if len(sys.argv) > 1 and sys.argv[1] == '--plot':
    from matplotlib import pyplot as plt
    t, source_v = source.get_v()[:, 1:3].transpose()
    t, target1_v = target1.get_v()[:, 1:3].transpose()
    t, target2_v = target2.get_v()[:, 1:3].transpose()
    t, target3_v = target3.get_v()[:, 1:3].transpose()
    t, target4_v = target4.get_v()[:, 1:3].transpose()
    t, target5_v = target5.get_v()[:, 1:3].transpose()
    t, target6_v = target2.get_v()[:, 1:3].transpose()
    t, target7_v = target2.get_v()[:, 1:3].transpose()
    plt.figure(1)
    plt.plot(t, source_v)
    plt.plot(t, target1_v)
    plt.plot(t, target2_v)
    plt.plot(t, target3_v)
    plt.plot(t, target4_v)
    plt.plot(t, target5_v)
    plt.plot(t, target6_v)
    plt.plot(t, target7_v)
    plt.legend()
    plt.show()
else:
    if not os.path.exists('gap_junction_output'):
        raise Exception("Please create the folder 'gap_junction_output' in your current working "
                        "directory to store the output files")
    source.print_v('gap_junction_output/source.v')
    target1.print_v('gap_junction_output/target1.v')
    target2.print_v('gap_junction_output/target2.v')
    target3.print_v('gap_junction_output/target3.v')
    target4.print_v('gap_junction_output/target4.v')
    target5.print_v('gap_junction_output/target5.v')
#    target6.print_v('gap_junction_output/target6.v')
#    target2.print_v('gap_junction_output/target7.v')
print ("All tests ran successfully. Please check the output folder '{}' to see if "
       "the gap junctions are working effectively"
       .format(os.path.abspath(os.path.join(os.getcwd(), 'gap_junction_output'))))
h.quit()

