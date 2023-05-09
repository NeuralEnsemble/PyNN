

from numpy.testing import assert_array_equal
import quantities as pq
import numpy as np

from .fixtures import run_with_simulators
import pytest

try:
    import scipy
    have_scipy = True
except ImportError:
    have_scipy = False


@run_with_simulators("nest", "neuron", "brian2")
def test_changing_electrode(sim):
    """
    Check that changing the values of the electrodes on the fly is taken into account
    """
    repeats = 2
    dt = 0.1
    simtime = 100
    sim.setup(timestep=dt, min_delay=dt)
    p = sim.Population(1, sim.IF_curr_exp())
    c = sim.DCSource(amplitude=0.0)
    c.inject_into(p)
    p.record('v')

    for i in range(repeats):
        sim.run(simtime)
        c.amplitude += 0.1

    data = p.get_data().segments[0].analogsignals[0]

    sim.end()

    # check that the value of v just before increasing the current is less than
    # the value at the end of the simulation
    assert data[int(simtime / dt), 0] < data[-1, 0]


@run_with_simulators("nest", "neuron", "brian2")
def test_ticket226(sim):
    """
    Check that the start time of DCSources is correctly taken into account
    http://neuralensemble.org/trac/PyNN/ticket/226)
    """
    sim.setup(timestep=0.1, min_delay=0.1)

    cell = sim.Population(1, sim.IF_curr_alpha(tau_m=20.0, cm=1.0, v_rest=-60.0,
                                               v_reset=-60.0))
    cell.initialize(v=-60.0)
    inj = sim.DCSource(amplitude=1.0, start=10.0, stop=20.0)
    cell.inject(inj)
    cell.record_v()
    sim.run(30.0)
    v = cell.get_data().segments[0].filter(name='v')[0][:, 0]
    sim.end()
    v_10p0 = v.magnitude[abs(v.times - 10.0 * pq.ms) < 0.01 * pq.ms, 0][0]
    assert abs(v_10p0 - -60.0) < 1e-10
    v_10p1 = v.magnitude[abs(v.times - 10.1 * pq.ms) < 0.01 * pq.ms, 0][0]
    assert v_10p1 > -59.99, v_10p1


@run_with_simulators("nest", "neuron", "brian2")
def test_issue165(sim):
    """Ensure that anonymous current sources are not lost."""
    sim.setup(timestep=0.1)
    p = sim.Population(1, sim.IF_cond_exp())
    p.inject(sim.DCSource(amplitude=1.0, start=10.0, stop=20.0))
    p.record('v')
    sim.run(20.0)
    data = p.get_data().segments[0].filter(name='v')[0]
    sim.end()
    assert data[99, 0] == -65.0
    assert data[150, 0] > -65.0


@run_with_simulators("nest", "neuron", "brian2")
def test_issue321(sim):
    """Check that non-zero currents at t=0 are taken into account."""
    sim.setup(timestep=0.1, min_delay=0.1)
    cells = sim.Population(3, sim.IF_curr_alpha(tau_m=20.0, cm=1.0, v_rest=-60.0,
                                                v_reset=-60.0))
    cells.initialize(v=-60.0)
    cells[0].i_offset = 1.0
    inj1 = sim.DCSource(amplitude=1.0, start=0.0)
    inj2 = sim.StepCurrentSource(times=[0.0], amplitudes=[1.0])
    cells[1].inject(inj1)
    cells[2].inject(inj2)
    cells.record_v()
    sim.run(20.0)
    v = cells.get_data().segments[0].filter(name='v')[0]
    sim.end()
    # the DCSource and StepCurrentSource should be equivalent
    assert_array_equal(v[:, 1], v[:, 2])
    # Ideally, the three cells should have identical traces, but in
    # practice there is always a delay with NEST until the current from
    # a current generator kicks in
    assert abs((v[-3:, 1] - v[-3:, 0]).max()) < 0.2


@run_with_simulators("nest", "neuron", "brian2")
def test_issue437(sim):
    """
    Checks whether NoisyCurrentSource works properly, by verifying that:
    1) no change in vm before start time
    2) change in vm at dt after start time
    3) monotonic decay of vm after stop time
    4) noise.dt is properly implemented
    Note: On rare occasions this test might fail as the signal is stochastic.
    Test implementation makes use of certain approximations for thresholding.
    If fails, run the test again to confirm. Passes 9/10 times on first attempt.
    """
    if not have_scipy:
        pytest.skip("scipy not available")
    v_rest = -60.0  # for this test keep v_rest < v_reset
    sim.setup(timestep=0.1, min_delay=0.1)
    cells = sim.Population(2, sim.IF_curr_alpha(tau_m=20.0, cm=1.0, v_rest=v_rest,
                                                v_reset=-55.0, tau_refrac=5.0))
    cells.initialize(v=-60.0)

    # We test two cases: dt = simulator.state.dt and dt != simulator.state.dt
    t_start = 25.0
    t_stop = 150.0
    dt_0 = 0.1
    dt_1 = 1.0
    noise_0 = sim.NoisyCurrentSource(mean=0.5, stdev=0.25, start=t_start, stop=t_stop, dt=dt_0)
    noise_1 = sim.NoisyCurrentSource(mean=0.5, stdev=0.25, start=t_start, stop=t_stop, dt=dt_1)
    cells[0].inject(noise_0)
    cells[1].inject(noise_1)

    cells.record('v')
    sim.run(200.0)
    v = cells.get_data().segments[0].filter(name="v")[0]
    v0 = v[:, 0]
    v1 = v[:, 1]
    t = v.times
    sim.end()

    t_start_ind = int(np.argmax(t >= t_start))
    t_stop_ind = int(np.argmax(t >= t_stop))

    # test for no change in vm before start time
    # note: exact matches not appropriate owing to floating point rounding errors
    assert (all(abs(val0 - v_rest*pq.mV) < 1e-9 and abs(val1 - v_rest*pq.mV) <
                    1e-9 for val0, val1 in zip(v0[:t_start_ind+1], v1[:t_start_ind+1])))

    # test for change in vm at dt after start time
    assert (abs(v0[t_start_ind+1] - v_rest*pq.mV) >=
                1e-9 and abs(v1[t_start_ind+1] - v_rest*pq.mV) >= 1e-9)

    # test for monotonic decay of vm after stop time
    assert (all(val0 >= val0_next and val1 >= val1_next for val0, val0_next, val1, val1_next in zip(
        v0[t_stop_ind:], v0[t_stop_ind+1:], v1[t_stop_ind:], v1[t_stop_ind+1:])))

    # test for ensuring noise.dt is properly implemented; checking first instance for each
    #   recording current profiles not implemented currently, thus using double derivative of vm
    #   necessary to upsample signal with noise of dt; else fails in certain scenarios
    #   Test implementation makes use of certain approximations for thresholding.
    #   Note: there can be a much simpler check for this once recording current profiles enabled (for all simulators).
    #   Test implementation makes use of certain approximations for thresholding; hence taking mode of initial values
    t_up = np.arange(float(min(t)), float(max(t))+dt_0/10.0, dt_0/10.0)
    v0_up = np.interp(t_up, t, v0.magnitude.flat)
    v1_up = np.interp(t_up, t, v1.magnitude.flat)
    d2_v0_up = np.diff(v0_up, n=2)
    d2_v1_up = np.diff(v1_up, n=2)
    dt_0_list = [j for (i, j) in zip(d2_v0_up, t_up) if abs(i) >= 0.00005]
    dt_1_list = [j for (i, j) in zip(d2_v1_up, t_up) if abs(i) >= 0.00005]
    dt_0_list_diff = np.diff(dt_0_list, n=1)
    dt_1_list_diff = np.diff(dt_1_list, n=1)
    dt_0_mode = scipy.stats.mode(dt_0_list_diff[0:10], keepdims=False)[0]
    dt_1_mode = scipy.stats.mode(dt_1_list_diff[0:10], keepdims=False)[0]
    assert (abs(dt_0_mode - dt_0) < 1e-9 or abs(dt_1_mode - dt_1) < 1e-9)


@run_with_simulators("nest", "neuron", "brian2")
def test_issue442(sim):
    """
    Checks whether ACSource works properly, by verifying that:
    1) no change in vm before start time
    2) change in vm at dt after start time
    3) monotonic decay of vm after stop time
    4) accurate frequency of output signal
    5) offset included in output signal
    """
    v_rest = -60.0
    sim.setup(timestep=0.1, min_delay=0.1)
    cells = sim.Population(1, sim.IF_curr_alpha(tau_m=20.0, cm=1.0, v_rest=v_rest,
                                                v_reset=-65.0, tau_refrac=5.0))
    cells.initialize(v=v_rest)

    # set t_start, t_stop and freq such that
    # "freq*1e-3*(t_stop-t_start)" is an integral value
    t_start = 22.5
    t_stop = 122.5
    freq = 100.0
    acsource = sim.ACSource(start=t_start, stop=t_stop, amplitude=0.5,
                            offset=0.1, frequency=freq, phase=0.0)
    cells[0].inject(acsource)

    cells.record('v')
    sim.run(150.0)
    v = cells.get_data().segments[0].filter(name="v")[0]
    v0 = v[:, 0]
    t = v.times
    sim.end()

    t_start_ind = int(np.argmax(t >= t_start))
    t_stop_ind = int(np.argmax(t >= t_stop))

    # test for no change in vm before start time
    # note: exact matches not appropriate owing to floating point rounding errors
    assert (all(abs(val0 - v_rest*pq.mV) < 1e-9 for val0 in v0[:t_start_ind+1]))

    # test for change in vm at dt after start time
    assert (abs(v0[t_start_ind+1] - v0[t_start_ind]) >= 1e-9)

    # test for monotonic decay of vm after stop time
    assert (all(val0 >= val0_next for val0, val0_next in zip(
        v0[t_stop_ind:], v0[t_stop_ind+1:])))

    # test for accurate frequency; simply counts peaks
    peak_ctr = 0
    peak_ind = []
    for i in range(t_stop_ind-t_start_ind):
        if v0[t_start_ind+i-1] < v0[t_start_ind+i] and v0[t_start_ind+i] >= v0[t_start_ind+i+1]:
            peak_ctr += 1
            peak_ind.append(t_start_ind+i)
    assert peak_ctr == freq*1e-3*(t_stop-t_start)
    # also test for offset; peaks initially increase in magnitude
    assert (v0[peak_ind[0]] < v0[peak_ind[1]] and v0[peak_ind[1]] < v0[peak_ind[2]])


@run_with_simulators("nest", "neuron", "brian2")
def test_issue445(sim):
    """
    This test basically checks if a new value of current is calculated at every
    time step, and that the total number of time steps is as expected theoretically
    Note: NEST excluded as recording of electrode currents still to be implemented
    """
    sim_dt = 0.1
    simtime = 200.0
    sim.setup(timestep=sim_dt, min_delay=1.0)
    cells = sim.Population(1, sim.IF_curr_exp(v_thresh=-55.0, tau_refrac=5.0))
    t_start = 50.0
    t_stop = 125.0
    acsource = sim.ACSource(start=t_start, stop=t_stop, amplitude=0.5,
                            offset=0.0, frequency=100.0, phase=0.0)
    cells[0].inject(acsource)
    acsource.record()

    sim.run(simtime)
    sim.end()

    i_ac = acsource.get_data()
    i_t_ac = i_ac.times.magnitude
    t_start_ind = np.argmax(i_t_ac >= t_start)
    t_stop_ind = np.argmax(i_t_ac >= t_stop)
    assert (all(val != val_next for val, val_next in zip(
        i_t_ac[t_start_ind:t_stop_ind-1], i_t_ac[t_start_ind+1:t_stop_ind])))
    # note: exact matches not appropriate owing to floating point rounding errors
    assert ((len(i_t_ac) - ((max(i_t_ac)-min(i_t_ac))/sim_dt + 1)) < 1e-9)


@run_with_simulators("nest", "neuron", "brian2")
def test_issue451(sim):
    """
    Modification of test: test_changing_electrode
    Difference: incorporates a start and stop time for stimulus
    Check that changing the values of the electrodes on the fly is taken into account
    """
    repeats = 2
    dt = 0.1
    simtime = 100
    sim.setup(timestep=dt, min_delay=dt)
    v_rest = -60.0
    p = sim.Population(1, sim.IF_curr_exp(v_rest=v_rest))
    p.initialize(v=v_rest)
    c = sim.DCSource(amplitude=0.0, start=25.0, stop=50.0)
    c.inject_into(p)
    p.record('v')

    for i in range(repeats):
        sim.run(simtime)
        c.amplitude += 0.1

    v = p.get_data().segments[0].filter(name="v")[0]
    sim.end()
    # check that the value of v is equal to v_rest throughout the simulation
    # note: exact matches not appropriate owing to floating point rounding errors
    assert (all((val.item()-v_rest) < 1e-9 for val in v[:, 0]))


@run_with_simulators("nest", "neuron", "brian2")
def test_issue483(sim):
    """
    Test to ensure that length of recorded voltage vector is as expected
    (checks for the specific scenario that failed earlier)
    """
    dt = 0.1
    sim.setup(timestep=dt, min_delay=dt)
    p = sim.Population(1, sim.IF_curr_exp())
    c = sim.DCSource(amplitude=0.5)
    c.inject_into(p)
    p.record('v')

    simtime = 200.0
    sim.run(100.0)
    sim.run(100.0)

    v = p.get_data().segments[0].filter(name="v")[0]

    # check that the length of vm vector is as expected theoretically
    assert (len(v) == (int(simtime/dt) + 1))


@run_with_simulators("nest", "neuron", "brian2")
def test_issue487(sim):
    """
    Test to ensure that DCSource and StepCurrentSource work properly
    for repeated runs. Problem existed under "neuron".
    Following sub-tests performed:
    1) DCSource active across two runs
    2) StepCurrentSource active across two runs
    3) DCSource active only during second run (earlier resulted in no current input)
    4) StepCurrentSource active only during second run (earlier resulted in current initiation at end of first run)
    """
    dt = 0.1
    sim.setup(timestep=dt, min_delay=dt)

    v_rest = -60.0
    cells = sim.Population(4, sim.IF_curr_exp(v_thresh=-55.0, tau_refrac=5.0, v_rest=v_rest))
    cells.initialize(v=v_rest)
    cells.record('v')

    dcsource = sim.DCSource(amplitude=0.15, start=25.0, stop=115.0)
    cells[0].inject(dcsource)

    step = sim.StepCurrentSource(times=[25.0, 75.0, 115.0], amplitudes=[0.05, 0.10, 0.20])
    cells[1].inject(step)

    dcsource_2 = sim.DCSource(amplitude=0.15, start=115.0, stop=145.0)
    cells[2].inject(dcsource_2)

    step_2 = sim.StepCurrentSource(times=[125.0, 175.0, 215.0], amplitudes=[0.05, 0.10, 0.20])
    cells[3].inject(step_2)

    simtime = 100.0
    sim.run(simtime)
    sim.run(simtime)

    v = cells.get_data().segments[0].filter(name="v")[0]
    sim.end()
    v_dc = v[:, 0]
    v_step = v[:, 1]
    v_dc_2 = v[:, 2]
    v_step_2 = v[:, 3]

    # check that membrane potential does not fall after end of first run
    # Test 1
    assert (v_dc[int(simtime/dt)] < v_dc[int(simtime/dt)+1])
    # Test 2
    assert (v_step[int(simtime/dt)] < v_step[int(simtime/dt)+1])
    # check that membrane potential of cell undergoes a change
    # Test 3
    v_dc_2_arr = np.squeeze(np.array(v_dc_2))
    assert not (np.isclose(v_dc_2_arr, v_rest).all())
    # check that membrane potential of cell undergoes no change till start of current injection
    # Test 4
    v_step_2_arr = np.squeeze(np.array(v_step_2))
    assert (np.isclose(v_step_2_arr[0:int(step_2.times[0]/dt)], v_rest).all())


@run_with_simulators("nest", "neuron", "brian2")
def test_issue_465_474_630(sim):
    """
    Checks the current traces recorded for each of the four types of
    electrodes in pyNN, and verifies that:
    1) Length of the current traces are as expected
    2) Values at t = t_start and t = t_stop present
    3) Changes in current value occur at the expected time instant
    4) Change in Vm begins at the immediate next time instant following current injection
    """
    sim_dt = 0.1
    sim.setup(min_delay=1.0, timestep=sim_dt)

    v_rest = -60.0
    cells = sim.Population(4, sim.IF_curr_exp(v_thresh=-55.0, tau_refrac=5.0, v_rest=v_rest))
    cells.initialize(v=v_rest)

    amp = 0.5
    offset = 0.1
    start = 50.0
    stop = 125.0

    acsource = sim.ACSource(start=start, stop=stop, amplitude=amp,
                            offset=offset, frequency=100.0, phase=0.0)
    cells[0].inject(acsource)
    acsource.record()

    dcsource = sim.DCSource(amplitude=amp, start=start, stop=stop)
    cells[1].inject(dcsource)
    dcsource.record()

    noise = sim.NoisyCurrentSource(mean=amp, stdev=0.05, start=start, stop=stop, dt=sim_dt)
    cells[2].inject(noise)
    noise.record()

    step = sim.StepCurrentSource(times=[start, (start+stop)/2, stop], amplitudes=[0.4, 0.6, 0.2])
    cells[3].inject(step)
    step.record()

    cells.record('v')
    runtime = 100.0
    simtime = 0
    # testing for repeated runs
    sim.run(runtime)
    simtime += runtime
    sim.run(runtime)
    simtime += runtime

    vm = cells.get_data().segments[0].filter(name="v")[0]
    sim.end()

    v_ac = vm[:, 0]
    v_dc = vm[:, 1]
    v_noise = vm[:, 2]
    v_step = vm[:, 3]

    i_ac = acsource.get_data()
    i_dc = dcsource.get_data()
    i_noise = noise.get_data()
    i_step = step.get_data()

    # test for length of recorded current traces
    assert (len(i_ac) == (int(simtime/sim_dt)+1) == len(v_ac))
    assert (len(i_dc) == int(simtime/sim_dt)+1 == len(v_dc))
    assert (len(i_noise) == int(simtime/sim_dt)+1 == len(v_noise))
    assert (len(i_step) == int(simtime/sim_dt)+1 == len(v_step))

    # test to check values exist at start and end of simulation
    assert (i_ac.t_start == 0.0 * pq.ms and np.isclose(float(i_ac.times[-1]), simtime))
    assert (i_dc.t_start == 0.0 * pq.ms and np.isclose(float(i_dc.times[-1]), simtime))
    assert (i_noise.t_start == 0.0 *
                pq.ms and np.isclose(float(i_noise.times[-1]), simtime))
    assert (i_step.t_start == 0.0 * pq.ms and np.isclose(float(i_step.times[-1]), simtime))

    # test to check current changes at start time instant
    assert (i_ac[(int(start / sim_dt)) - 1, 0] == 0 *
                pq.nA and i_ac[int(start / sim_dt), 0] != 0 * pq.nA)
    assert (i_dc[int(start / sim_dt) - 1, 0] == 0 *
                pq.nA and i_dc[int(start / sim_dt), 0] != 0 * pq.nA)
    assert (i_noise[int(start / sim_dt) - 1, 0] == 0 *
                pq.nA and i_noise[int(start / sim_dt), 0] != 0 * pq.nA)
    assert (i_step[int(start / sim_dt) - 1, 0] == 0 *
                pq.nA and i_step[int(start / sim_dt), 0] != 0 * pq.nA)

    # test to check current changes appropriately at stop time instant - issue #630
    assert (i_ac[(int(stop / sim_dt)) - 1, 0] != 0.0 *
                pq.nA and i_ac[(int(stop / sim_dt)), 0] == 0.0 * pq.nA)
    assert (i_dc[(int(stop / sim_dt)) - 1, 0] != 0.0 *
                pq.nA and i_ac[(int(stop / sim_dt)), 0] == 0.0 * pq.nA)
    assert (i_noise[(int(stop / sim_dt)) - 1, 0] != 0.0 *
                pq.nA and i_ac[(int(stop / sim_dt)), 0] == 0.0 * pq.nA)
    assert (i_step[(int(stop / sim_dt)) - 1, 0] != 0.2 *
                pq.nA and i_ac[(int(stop / sim_dt)), 0] == 0.0 * pq.nA)

    # test to check vm changes at the time step following current initiation
    assert (np.isclose(float(v_ac[int(start / sim_dt), 0].item()),
                              v_rest) and v_ac[int(start / sim_dt) + 1] != v_rest * pq.mV)
    assert (np.isclose(float(v_dc[int(start / sim_dt), 0].item()),
                              v_rest) and v_dc[int(start / sim_dt) + 1] != v_rest * pq.mV)
    assert (np.isclose(float(v_noise[int(start / sim_dt), 0].item()),
                              v_rest) and v_noise[int(start / sim_dt) + 1] != v_rest * pq.mV)
    assert (np.isclose(float(v_step[int(start / sim_dt), 0].item()),
                              v_rest) and v_step[int(start / sim_dt) + 1] != v_rest * pq.mV)


@run_with_simulators("nest", "neuron", "brian2")
def test_issue497(sim):
    """
    This is a test to check that the specified phase for the ACSource is valid
    at the specified start time (and not, for example, at t=0 as NEST currently does)

    Approach:
    > Two signals with different initial specified phases
    > 'start' of one signal updated on the fly
    > 'frequency' of other signal updated on the fly
    > Test to ensure that initial specified phases applicable at t = start
    """
    sim_dt = 0.1
    sim.setup(min_delay=1.0, timestep=sim_dt)

    start1 = 5.0
    freq1 = 100.0
    phase1 = 0.0
    start2 = 5.0
    freq2 = 100.0
    phase2 = 90.0
    amplitude = 1.0

    cells = sim.Population(2, sim.IF_curr_exp(v_thresh=-55.0, tau_refrac=5.0))

    acsource1 = sim.ACSource(start=start1, stop=20.0, amplitude=amplitude, offset=0.0,
                             frequency=freq1, phase=phase1)
    cells[0].inject(acsource1)
    acsource1.record()
    acsource2 = sim.ACSource(start=start2, stop=20.0, amplitude=amplitude, offset=0.0,
                             frequency=freq2, phase=phase2)
    cells[1].inject(acsource2)
    acsource2.record()

    # cannot directly assign/read from electrode variables, as each
    # simulator currently returns parameters in different units (see #452)
    start1 = 10.0
    acsource1.start = start1
    freq2 = 20.0
    acsource2.frequency = freq2

    cells.record('v')
    sim.run(25.0)
    vm = cells.get_data().segments[0].filter(name="v")[0]
    sim.end()
    i_ac1 = acsource1.get_data()
    i_ac2 = acsource2.get_data()

    # verify that acsource1 has value at t = start as 0 and as non-zero at next dt
    assert (abs(i_ac1[int(start1 / sim_dt), 0]) < 1e-9)
    assert (abs(i_ac1[int(start1 / sim_dt) + 1, 0]) > 1e-9)
    # verify that acsources has value at t = start as 'amplitude'
    assert (abs(i_ac2[int(start2 / sim_dt), 0] - amplitude * pq.nA) < 1e-9)


@run_with_simulators("nest", "neuron", "brian2")
def test_issue512(sim):
    """
    Test to ensure that StepCurrentSource times are handled similarly across
    all simulators. Multiple combinations of step times tested for:
    1) dt = 0.1 ms, min_delay = 0.1 ms
    2) dt = 0.01 ms, min_delay = 0.01 ms
    Note: exact matches of times not appropriate owing to floating point
    rounding errors. If absolute difference <1e-9, then considered equal.
    """

    def get_len(data):
        if hasattr(data, "evaluate"):
            return len(data.evaluate())
        else:
            return len(data)

    # 1) dt = 0.1 ms, min_delay = 0.1 ms
    dt = 0.1
    sim.setup(timestep=dt, min_delay=dt)
    cells = sim.Population(1, sim.IF_curr_exp(v_thresh=-55.0, tau_refrac=5.0, v_rest=-60.0))
    # 1.1) Negative time value
    with pytest.raises(ValueError):
        sim.StepCurrentSource(times=[0.4, -0.6, 0.8], amplitudes=[0.5, -0.5, 0.5])
    # 1.2) Time values not monotonically increasing
    with pytest.raises(ValueError):
        sim.StepCurrentSource(times=[0.4, 0.2, 0.8], amplitudes=[0.5, -0.5, 0.5])
    # 1.3) Check mapping of time values and removal of duplicates
    step = sim.StepCurrentSource(times=[0.41, 0.42, 0.86],  # should be mapped to [0.4, 0.9]
                                 amplitudes=[0.5, -0.5, 0.5])
    assert get_len(step.times) == 2
    assert get_len(step.amplitudes) == 2
    # if "brian" in str(sim):
    #     # Brian requires time in seconds (s)
    #     assert (abs(step.times[0]-0.4*1e-3) < 1e-9)
    #     assert (abs(step.times[1]-0.9*1e-3) < 1e-9)
    #     # Brain requires amplitudes in amperes (A)
    #     assert (step.amplitudes[0] == -0.5*1e-9)
    #     assert (step.amplitudes[1] == 0.5*1e-9)
    # else:
    # NEST requires amplitudes in picoamperes (pA) but stored
    # as LazyArray and so needn't manually adjust; use nA
    # NEURON requires amplitudes in nanoamperes (nA)
    assert (step.amplitudes[0] == -0.5)
    assert (step.amplitudes[1] == 0.5)
    # NEST has time stamps reduced by min_delay
    if "nest" in str(sim):
        assert (abs(step.times[0]-0.3) < 1e-9)
        assert (abs(step.times[1]-0.8) < 1e-9)
    else:  # neuron, brian
        assert (abs(step.times[0]-0.4) < 1e-9)
        assert (abs(step.times[1]-0.9) < 1e-9)

    # 2) dt = 0.01 ms, min_delay = 0.01 ms
    dt = 0.01
    sim.setup(timestep=dt, min_delay=dt)
    cells = sim.Population(1, sim.IF_curr_exp(v_thresh=-55.0, tau_refrac=5.0, v_rest=-60.0))
    # 2.1) Negative time value
    with pytest.raises(ValueError):
        sim.StepCurrentSource(times=[0.4, -0.6, 0.8], amplitudes=[0.5, -0.5, 0.5])
    # 2.2) Time values not monotonically increasing
    with pytest.raises(ValueError):
        sim.StepCurrentSource(times=[0.5, 0.4999, 0.8], amplitudes=[0.5, -0.5, 0.5])
    # 2.3) Check mapping of time values and removal of duplicates
    step = sim.StepCurrentSource(times=[0.451, 0.452, 0.86],
                                 amplitudes=[0.5, -0.5, 0.5])
    assert get_len(step.times) == 2
    assert get_len(step.amplitudes) == 2
    # NEST requires amplitudes in picoamperes (pA) but stored
    # as LazyArray and so needn't manually adjust; use nA
    # NEURON requires amplitudes in nanoamperes (nA)
    assert (step.amplitudes[0] == -0.5)
    assert (step.amplitudes[1] == 0.5)
    # NEST has time stamps reduced by min_delay
    if "nest" in str(sim):
        assert (abs(step.times[0]-0.44) < 1e-9)
        assert (abs(step.times[1]-0.85) < 1e-9)
    else:  # neuron, brian
        assert (abs(step.times[0]-0.45) < 1e-9)
        assert (abs(step.times[1]-0.86) < 1e-9)


@run_with_simulators("nest", "neuron", "brian2")
def test_issue631(sim):
    """
    Test to ensure that recording of multiple electrode currents do not
    interfere with one another.
    """
    sim_dt = 0.1
    sim.setup(timestep=sim_dt, min_delay=sim_dt)

    cells = sim.Population(1, sim.IF_curr_exp(v_rest=-65.0, v_thresh=-
                                              55.0, tau_refrac=5.0))  # , i_offset=-1.0*amp))
    dc_source = sim.DCSource(amplitude=0.5, start=25, stop=50)
    ac_source = sim.ACSource(start=75, stop=125, amplitude=0.5,
                             offset=0.25, frequency=100.0, phase=0.0)
    noisy_source = sim.NoisyCurrentSource(mean=0.5, stdev=0.05, start=150, stop=175, dt=1.0)
    step_source = sim.StepCurrentSource(times=[200, 225, 250], amplitudes=[0.4, 0.6, 0.2])

    cells[0].inject(dc_source)
    cells[0].inject(ac_source)
    cells[0].inject(noisy_source)
    cells[0].inject(step_source)

    dc_source.record()
    ac_source.record()
    noisy_source.record()
    step_source.record()

    sim.run(275.0)

    i_dc = dc_source.get_data()
    i_ac = ac_source.get_data()
    i_noisy = noisy_source.get_data()
    i_step = step_source.get_data()

    assert (np.all(i_dc.magnitude[:int(25.0 / sim_dt) - 1:] == 0)
                and np.all(i_dc.magnitude[int(50.0 / sim_dt):] == 0))
    assert (np.all(i_ac.magnitude[:int(75.0 / sim_dt) - 1:] == 0)
                and np.all(i_ac.magnitude[int(125.0 / sim_dt):] == 0))
    assert (np.all(i_noisy.magnitude[:int(150.0 / sim_dt) - 1:] == 0)
                and np.all(i_noisy.magnitude[int(175.0 / sim_dt):] == 0))
    assert (np.all(i_step.magnitude[:int(200.0 / sim_dt) - 1:] == 0))


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    test_changing_electrode(sim)
    test_ticket226(sim)
    test_issue165(sim)
    test_issue321(sim)
    test_issue437(sim)
    test_issue442(sim)
    test_issue445(sim)
    test_issue451(sim)
    test_issue483(sim)
    test_issue487(sim)
    test_issue_465_474_630(sim)
    test_issue497(sim)
    test_issue512(sim)
    test_issue631(sim)
