import logging
import unittest


from numpy import nan_to_num

try:
    import pyNN.hardware.brainscales as sim
    have_hardware_brainscales = True
except ImportError:
    have_hardware_brainscales = False

import pytest


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class HardwareTest(unittest.TestCase):

    def setUp(self):
        if not have_hardware_brainscales:
            pytest.skip("BrainScaleS module not available")
        extra = {
            'loglevel': 0,
            'ignoreHWParameterRanges': True,
            'useSystemSim': True,
            'hardware': sim.hardwareSetup['one-hicann']
        }
        sim.setup(**extra)

    def test_IF_cond_exp_default_values(self):
        ifcell = sim.IF_cond_exp()

    def test_IF_cond_exp_default_values2(self):
        ifcell = sim.IF_cond_exp()

    def test_SpikeSourceArray(self):
        from pyNN.utility.plotting import Figure, Panel
        spike_times = [50.]
        p = sim.Population(3, sim.SpikeSourceArray(spike_times=spike_times))
        p2 = sim.Population(3, sim.Hardware_IF_cond_exp())
        syn = sim.StaticSynapse(weight=0.012)
        con = sim.Projection(p, p2, connector=sim.OneToOneConnector(),
                             synapse_type=syn, receptor_type='excitatory')
        spike_times_g = p.get('spike_times')
        p2.record('v')
        sim.run(100.0)
        weights = nan_to_num(con.get('weight', format="array"))
        print(weights)
        data = p2.get_data().segments[0]
        vm = data.filter(name="v")[0]
        print(vm)
        Figure(
            Panel(weights, data_labels=[
                  "ext->cell"], line_properties=[{'xticks': True, 'yticks': True, 'cmap': 'Greys'}]),
            Panel(vm, ylabel="Membrane potential (mV)", data_labels=[
                  "excitatory", "excitatory"], line_properties=[{'xticks': True, 'yticks': True}]),
        ).save("result")

    # def test_set_parameters(self):
        #p = sim.Population(3, sim.SpikeSourceArray())
        #p2 = sim.Population(3, sim.Hardware_IF_cond_exp())
        #syn = sim.StaticSynapse(weight=0.012)
        #con = sim.Projection(p, p2, connector = sim.OneToOneConnector(), synapse_type=syn,receptor_type='excitatory')
        #p[0].set_parameters(spike_times=Sequence([1., 2., 3., 40.]))
        #p[1].set_parameters(spike_times=Sequence([2., 3., 4., 50.]))
        #p[2].set_parameters(spike_times=Sequence([3., 4., 5., 50.]))
        #spike_times = p.get('spike_times')
        #self.assertEqual(spike_times.size, 3)
        #assert_array_equal(spike_times[1], Sequence([2, 3, 4, 50]))
        # p2.record('v')
        # sim.run(100.0)
        #weights = nan_to_num(con.get('weight', format="array"))
        # print weights
        #data = p2.get_data().segments[0]
        #vm = data.filter(name="v")[0]
        # print vm
        # Figure(
        #Panel(weights,data_labels=["ext->cell"], line_properties=[{'xticks':True, 'yticks':True, 'cmap':'Greys'}]),
        #Panel(vm, ylabel="Membrane potential (mV)", data_labels=["excitatory", "excitatory"], line_properties=[{'xticks': True, 'yticks':True}]),
        # ).save("result")



def test_restart_loop():
    if not have_hardware_brainscales:
        pytest.skip("BrainScaleS module not available")
    extra = {'loglevel': 0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}
    sim.setup(**extra)
    sim.end()
    sim.setup(**extra)
    sim.end()
    sim.setup(**extra)
    sim.run(10.0)
    sim.end()
    sim.setup(**extra)
    sim.run(10.0)
    sim.end()

# def test_several_runs():
    if not have_hardware_brainscales:
        pytest.skip("BrainScaleS module not available")
    #extra = {'loglevel':0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}
    # sim.setup(**extra)
    # sim.run(10.0)
    # sim.run(10.0)
    # sim.end()


def test_sim_without_clearing():
    if not have_hardware_brainscales:
        pytest.skip("BrainScaleS module not available")
    extra = {'loglevel': 0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}
    sim.setup(**extra)


def test_sim_without_setup():
    if not have_hardware_brainscales:
        pytest.skip("BrainScaleS module not available")
    sim.end()


if __name__ == '__main__':
    # test_scenarios()
    # test_restart_loop()
    # test_sim_without_clearing()
    test_sim_without_setup()
    # test_several_runs()
