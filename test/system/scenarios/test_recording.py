
import os
import pickle
import numpy as np
import quantities as pq
from numpy.testing import assert_array_equal, assert_allclose
from neo.io import get_io
from pyNN.utility import normalized_filename
from .fixtures import run_with_simulators


@run_with_simulators("neuron", "brian2")
def test_reset_recording(sim):
    """
    Check that record(None) resets the list of things to record.

    This test injects different levels of current into two neurons. In the
    first run, we record one of the neurons, in the second we record the other.
    The main point is to check that the first neuron is not recorded in the
    second run.
    """
    sim.setup()
    p = sim.Population(7, sim.IF_cond_exp())
    p[3].i_offset = 0.1
    p[4].i_offset = 0.2
    p[3:4].record('v')
    sim.run(10.0)
    sim.reset()
    p.record(None)
    p[4:5].record('v')
    sim.run(10.0)
    data = p.get_data()
    sim.end()
    ti = lambda i: data.segments[i].analogsignals[0].times
    assert_array_equal(ti(0), ti(1))
    assert_array_equal(data.segments[0].analogsignals[0].array_annotations["channel_index"], np.array([3]))
    assert_array_equal(data.segments[1].analogsignals[0].array_annotations["channel_index"], np.array([4]))
    vi = lambda i: data.segments[i].analogsignals[0]
    assert vi(0).shape == vi(1).shape == (101, 1)
    assert vi(0)[0, 0] == vi(1)[0, 0] == p.initial_values['v'].evaluate(simplify=True) * pq.mV  # the first value should be the same
    assert not (vi(0)[1:, 0] == vi(1)[1:, 0]).any()            # none of the others should be, because of different i_offset


@run_with_simulators("nest", "neuron", "brian2")
def test_record_vm_and_gsyn_from_assembly(sim):
    from pyNN.utility import init_logging
    init_logging(logfile=None, debug=True)
    dt = 0.1
    tstop = 100.0
    sim.setup(timestep=dt, min_delay=dt)
    cells = sim.Population(5, sim.IF_cond_exp()) + sim.Population(6, sim.EIF_cond_exp_isfa_ista())
    inputs = sim.Population(5, sim.SpikeSourcePoisson(rate=50.0))
    sim.connect(inputs, cells, weight=0.1, delay=0.5, receptor_type='inhibitory')
    sim.connect(inputs, cells, weight=0.1, delay=0.3, receptor_type='excitatory')
    cells.record('v')
    cells[2:9].record(['gsyn_exc', 'gsyn_inh'])
#    for p in cells.populations:
#        assert p.recorders['v'].recorded, set(p.all_cells))

#    assert cells.populations[0].recorders['gsyn'].recorded, set(cells.populations[0].all_cells[2:5]))
#    assert cells.populations[1].recorders['gsyn'].recorded, set(cells.populations[1].all_cells[0:4]))
    sim.run(tstop)
    data0 = cells.populations[0].get_data().segments[0]
    data1 = cells.populations[1].get_data().segments[0]
    data_all = cells.get_data().segments[0]
    vm_p0 = data0.filter(name='v')[0]
    vm_p1 = data1.filter(name='v')[0]
    vm_all = data_all.filter(name='v')[0]
    gsyn_p0 = data0.filter(name='gsyn_exc')[0]
    gsyn_p1 = data1.filter(name='gsyn_exc')[0]
    gsyn_all = data_all.filter(name='gsyn_exc')[0]

    n_points = int(tstop / dt) + 1
    assert vm_p0.shape == (n_points, 5)
    assert vm_p1.shape == (n_points, 6)
    assert vm_all.shape == (n_points, 11)
    assert gsyn_p0.shape == (n_points, 3)
    assert gsyn_p1.shape == (n_points, 4)
    assert gsyn_all.shape == (n_points, 7)

    assert_array_equal(vm_p1[:, 3], vm_all[:, 8])

    assert_array_equal(vm_p0.array_annotations["channel_index"], np.arange(5))
    assert_array_equal(vm_p1.array_annotations["channel_index"], np.arange(6))
    #assert_array_equal(vm_all.array_annotations["channel_index"], np.arange(11))
    assert_array_equal(gsyn_p0.array_annotations["channel_index"], np.array([2, 3, 4]))
    assert_array_equal(gsyn_p1.array_annotations["channel_index"], np.arange(4))
    #assert_array_equal(gsyn_all.array_annotations["channel_index"], np.arange(2, 9))

    sim.end()


@run_with_simulators("nest", "neuron")
def test_issue259(sim):
    """
    A test that retrieving data with "clear=True" gives correct spike trains.
    """
    sim.setup(timestep=0.05, spike_precision="off_grid")
    p = sim.Population(1, sim.SpikeSourceArray(spike_times=[0.075, 10.025, 12.34, 1000.025]))
    p.record('spikes')
    sim.run(10.0)
    spiketrains0 = p.get_data('spikes', clear=True).segments[0].spiketrains
    print(spiketrains0[0])
    sim.run(10.0)
    spiketrains1 = p.get_data('spikes', clear=True).segments[0].spiketrains
    print(spiketrains1[0])
    sim.run(10.0)
    spiketrains2 = p.get_data('spikes', clear=True).segments[0].spiketrains
    print(spiketrains2[0])
    sim.end()

    assert_allclose(spiketrains0[0].rescale(pq.ms).magnitude, np.array([0.075]), 1e-17)
    assert_allclose(spiketrains1[0].rescale(pq.ms).magnitude, np.array([10.025, 12.34]), 1e-14)
    assert spiketrains2[0].size == 0


@run_with_simulators("nest", "neuron", "brian2")
def test_sampling_interval(sim):
    """
    A test of the sampling_interval argument.
    """
    sim.setup(0.1)
    p1 = sim.Population(3, sim.IF_cond_exp())
    p2 = sim.Population(4, sim.IF_cond_exp())
    p1.record('v', sampling_interval=1.0)
    p2.record('v', sampling_interval=0.5)
    sim.run(10.0)
    d1 = p1.get_data().segments[0].analogsignals[0]
    d2 = p2.get_data().segments[0].analogsignals[0]
    assert d1.sampling_period == 1.0 * pq.ms
    assert d1.shape == (11, 3)
    assert d2.sampling_period == 0.5 * pq.ms
    assert d2.shape == (21, 4)
    sim.end()


@run_with_simulators("nest", "neuron", "brian2")
def test_mix_procedural_and_oo(sim):
    # cf Issues #217, #234
    fn_proc = "test_write_procedural.pkl"
    fn_oo = "test_write_oo.pkl"
    sim.setup(timestep=0.1, min_delay=0.1)
    cells = sim.Population(5, sim.IF_cond_exp(i_offset=0.2))
    sim.record('v', cells, fn_proc)
    sim.run(10.0)
    cells.write_data(fn_oo)   # explicitly write data
    sim.end()                 # implicitly write data using filename provided previously

    data_proc = get_io(fn_proc).read()[0]
    data_oo = get_io(fn_oo).read()[0]
    assert_array_equal(data_proc.segments[0].analogsignals[0],
                       data_oo.segments[0].analogsignals[0])

    os.remove(fn_proc)
    os.remove(fn_oo)


@run_with_simulators("nest", "neuron")
def test_record_with_filename(sim):
    """
    Test to ensure that Simulator and Population recording work properly
    The following 12 scenarios are explored:
        Note: var1 = "spikes", var2 = "v"
        1) sim.record()
            i) cell[0]
                a) 2 parameters (2vars)         (scenario 1)
                b) parameter1 (var1)            (scenario 2)
                c) parameter2 (var2)            (scenario 3)
            ii) cell[1]
                a) 2 parameters (2vars)         (scenario 4)
                b) parameter1 (var1)            (scenario 5)
                c) parameter2 (var2)            (scenario 6)
            iii) population
                a) 2 parameters (2vars)         (scenario 7)
                b) parameter1 (var1)            (scenario 8)
                c) parameter2 (var2)            (scenario 9)
        2) pop.record() - always records for a population; not a single cell
            a) 2 parameters (2vars)             (scenario 10)
            b) parameter1 (var1)                (scenario 11)
            c) parameter2 (var2)                (scenario 12)

    cf Issues #449, #490, #491
    """
    # START ***** defining methods needed for test *****

    def get_file_data(filename):
        # method to access pickled file and retrieve data
        data = []
        with (open(filename, "rb")) as openfile:
            while True:
                try:
                    data.append(pickle.load(openfile))
                except EOFError:
                    break
        return data

    def eval_num_cells(data):
        # scan data object to evaluate number of cells; returns 4 values
        # nCells  :  # of cells in analogsignals (if "v" recorded)
        # nspikes1:  # of spikes in first recorded cell
        # nspikes2:  # of spikes in second recorded cell (if exists)
        # -- if any parameter absent, return -1 as its value
        # annot_bool # true if specified annotation exists; false otherwise

        try:
            nCells = data[0].segments[0].analogsignals[0].shape[1]
        except:
            nCells = -1

        try:
            nspikes1 = data[0].segments[0].spiketrains[0].shape[0]
        except:
            nspikes1 = -1

        try:
            nspikes2 = data[0].segments[0].spiketrains[1].shape[0]
        except:
            nspikes2 = -1

        if 'script_name' in data[0].annotations.keys():
            annot_bool = True
        else:
            annot_bool = False

        return (nCells, nspikes1, nspikes2, annot_bool)

    # END ***** defining methods needed for test *****

    sim_dt = 0.1
    sim.setup(min_delay=1.0, timestep = sim_dt)

    # creating a population of two cells; only cell[0] gets stimulus
    # hence only cell[0] will have entries for spiketrains
    cells = sim.Population(2, sim.IF_curr_exp(v_thresh=-55.0, tau_refrac=5.0))
    steady = sim.DCSource(amplitude=2.5, start=25.0, stop=75.0)
    cells[0].inject(steady)

    # specify appropriate filenames for output files
    filename_sim_cell1_2vars = normalized_filename("Results", "sim_cell1_2vars", "pkl", sim.__name__)
    filename_sim_cell1_var1  = normalized_filename("Results", "sim_cell1_var1", "pkl", sim.__name__)
    filename_sim_cell1_var2  = normalized_filename("Results", "sim_cell1_var2", "pkl", sim.__name__)
    filename_sim_cell2_2vars = normalized_filename("Results", "sim_cell2_2vars", "pkl", sim.__name__)
    filename_sim_cell2_var1  = normalized_filename("Results", "sim_cell2_var1", "pkl", sim.__name__)
    filename_sim_cell2_var2  = normalized_filename("Results", "sim_cell2_var2", "pkl", sim.__name__)
    filename_sim_popl_2vars  = normalized_filename("Results", "sim_popl_2vars", "pkl", sim.__name__)
    filename_sim_popl_var1   = normalized_filename("Results", "sim_popl_var1", "pkl", sim.__name__)
    filename_sim_popl_var2   = normalized_filename("Results", "sim_popl_var2", "pkl", sim.__name__)
    filename_rec_2vars = normalized_filename("Results", "rec_2vars", "pkl", sim.__name__)
    filename_rec_var1  = normalized_filename("Results", "rec_var1", "pkl", sim.__name__)
    filename_rec_var2  = normalized_filename("Results", "rec_var2", "pkl", sim.__name__)

    # instruct pynn to record as per above scenarios
    sim.record(["spikes", "v"], cells[0], filename_sim_cell1_2vars, annotations={'script_name': __file__})
    sim.record(["spikes"], cells[0], filename_sim_cell1_var1, annotations={'script_name': __file__})
    sim.record(["v"], cells[0], filename_sim_cell1_var2, annotations={'script_name': __file__})
    sim.record(["spikes", "v"], cells[1], filename_sim_cell2_2vars, annotations={'script_name': __file__})
    sim.record(["spikes"], cells[1], filename_sim_cell2_var1, annotations={'script_name': __file__})
    sim.record(["v"], cells[1], filename_sim_cell2_var2, annotations={'script_name': __file__})
    sim.record(["spikes", "v"], cells, filename_sim_popl_2vars, annotations={'script_name': __file__})
    sim.record(["spikes"], cells, filename_sim_popl_var1, annotations={'script_name': __file__})
    sim.record(["v"], cells, filename_sim_popl_var2, annotations={'script_name': __file__})
    cells.record(["spikes", "v"], to_file=filename_rec_2vars)
    cells.record(["spikes"], to_file=filename_rec_var1)
    cells.record(["v"], to_file=filename_rec_var2)

    sim.run(100.0)
    sim.end()

    # retrieve data from the created files, and perform appropriate checks
    # scenario 1
    nCells, nspikes1, nspikes2, annot_bool = eval_num_cells(get_file_data(filename_sim_cell1_2vars))
    assert (nCells == 1)
    assert (nspikes1 > 0)
    assert (nspikes2 == -1)
    assert (annot_bool)

    # scenario 2
    nCells, nspikes1, nspikes2, annot_bool = eval_num_cells(get_file_data(filename_sim_cell1_var1))
    assert (nCells == -1)
    assert (nspikes1 > 0)
    assert (nspikes2 == -1)
    assert (annot_bool)

    # scenario 3
    nCells, nspikes1, nspikes2, annot_bool = eval_num_cells(get_file_data(filename_sim_cell1_var2))
    assert (nCells == 1)
    assert (nspikes1 == -1)
    assert (nspikes2 == -1)
    assert (annot_bool)

    # scenario 4
    nCells, nspikes1, nspikes2, annot_bool = eval_num_cells(get_file_data(filename_sim_cell2_2vars))
    assert (nCells == 1)
    assert (nspikes1 == 0)
    assert (nspikes2 == -1)
    assert (annot_bool)

    # scenario 5
    nCells, nspikes1, nspikes2, annot_bool = eval_num_cells(get_file_data(filename_sim_cell2_var1))
    assert (nCells == -1)
    assert (nspikes1 == 0)
    assert (nspikes2 == -1)
    assert (annot_bool)

    # scenario 6
    nCells, nspikes1, nspikes2, annot_bool = eval_num_cells(get_file_data(filename_sim_cell2_var2))
    assert (nCells == 1)
    assert (nspikes1 == -1)
    assert (nspikes2 == -1)
    assert (annot_bool)

    # scenario 7
    nCells, nspikes1, nspikes2, annot_bool = eval_num_cells(get_file_data(filename_sim_popl_2vars))
    assert (nCells == 2)
    assert (nspikes1 > 0)
    assert (nspikes2 == 0)
    assert (annot_bool)

    # scenario 8
    nCells, nspikes1, nspikes2, annot_bool = eval_num_cells(get_file_data(filename_sim_popl_var1))
    assert (nCells == -1)
    assert (nspikes1 > 0)
    assert (nspikes2 == 0)
    assert (annot_bool)

    # scenario 9
    nCells, nspikes1, nspikes2, annot_bool = eval_num_cells(get_file_data(filename_sim_popl_var2))
    assert (nCells == 2)
    assert (nspikes1 == -1)
    assert (nspikes2 == -1)
    assert (annot_bool)

    # scenario 10
    nCells, nspikes1, nspikes2, annot_bool = eval_num_cells(get_file_data(filename_rec_2vars))
    assert (nCells == 2)
    assert (nspikes1 > 0)
    assert (nspikes2 == 0)
    assert (annot_bool)

    # scenario 11
    nCells, nspikes1, nspikes2, annot_bool = eval_num_cells(get_file_data(filename_rec_var1))
    assert (nCells == -1)
    assert (nspikes1 > 0)
    assert (nspikes2 == 0)
    assert (annot_bool)

    # scenario 12
    nCells, nspikes1, nspikes2, annot_bool = eval_num_cells(get_file_data(filename_rec_var2))
    assert (nCells == 2)
    assert (nspikes1 == -1)
    assert (nspikes2 == -1)
    assert (annot_bool)


@run_with_simulators("nest", "neuron")
def test_issue499(sim):
    """
    Test to check that sim.end() does not erase the recorded data
    """
    sim.setup(min_delay=1.0, timestep = 0.1)
    cells = sim.Population(1, sim.IF_curr_exp())
    dcsource = sim.DCSource(amplitude=0.5, start=20, stop=80)
    cells[0].inject(dcsource)
    cells.record('v')

    sim.run(50.0)
    sim.end()
    vm = cells.get_data().segments[0].filter(name="v")[0]
    v_dc = vm[:, 0]
    assert (len(v_dc)!=0)


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    test_reset_recording(sim)
    test_record_vm_and_gsyn_from_assembly(sim)
    test_issue259(sim)
    test_sampling_interval(sim)
    test_mix_procedural_and_oo(sim)
    test_record_with_filename(sim)
    test_issue499(sim)
