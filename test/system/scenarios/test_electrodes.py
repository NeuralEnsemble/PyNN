

from nose.tools import assert_equal, assert_true, assert_false
from numpy.testing import assert_array_equal
import quantities as pq
import numpy
from .registry import register

try:
    import scipy
    have_scipy = True
except ImportError:
    have_scipy = False
from nose.plugins.skip import SkipTest


@register(exclude=["nemo"])
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
test_changing_electrode.__test__ = False


@register()
def ticket226(sim):
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


@register()
def issue165(sim):
    """Ensure that anonymous current sources are not lost."""
    sim.setup(timestep=0.1)
    p = sim.Population(1, sim.IF_cond_exp())
    p.inject(sim.DCSource(amplitude=1.0, start=10.0, stop=20.0))
    p.record('v')
    sim.run(20.0)
    data = p.get_data().segments[0].filter(name='v')[0]
    sim.end()
    assert_equal(data[99, 0], -65.0)
    assert data[150, 0] > -65.0


@register()
def issue321(sim):
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


@register()
def issue437(sim):
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
        raise SkipTest

    v_rest = -60.0  # for this test keep v_rest < v_reset
    sim.setup(timestep=0.1, min_delay=0.1)
    cells = sim.Population(2, sim.IF_curr_alpha(tau_m=20.0, cm=1.0, v_rest=v_rest,
                                               v_reset=-55.0, tau_refrac=5.0))
    cells.initialize(v=-60.0)

    #We test two cases: dt = simulator.state.dt and dt != simulator.state.dt
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

    t_start_ind = int(numpy.argmax(t >= t_start))
    t_stop_ind = int(numpy.argmax(t >= t_stop))

    # test for no change in vm before start time
    # note: exact matches not appropriate owing to floating point rounding errors
    assert_true (all(abs(val0 - v_rest*pq.mV) < 1e-9 and abs(val1 - v_rest*pq.mV) < 1e-9 for val0, val1 in zip(v0[:t_start_ind+1], v1[:t_start_ind+1])))

    # test for change in vm at dt after start time
    assert_true (abs(v0[t_start_ind+1] - v_rest*pq.mV) >= 1e-9 and abs(v1[t_start_ind+1] - v_rest*pq.mV) >= 1e-9)

    # test for monotonic decay of vm after stop time
    assert_true (all(val0 >= val0_next and val1 >= val1_next for val0, val0_next, val1, val1_next in zip(v0[t_stop_ind:], v0[t_stop_ind+1:], v1[t_stop_ind:], v1[t_stop_ind+1:])))

    # test for ensuring noise.dt is properly implemented; checking first instance for each
    #   recording current profiles not implemented currently, thus using double derivative of vm
    #   necessary to upsample signal with noise of dt; else fails in certain scenarios
    #   Test implementation makes use of certain approximations for thresholding.
    #   Note: there can be a much simpler check for this once recording current profiles enabled (for all simulators).
    #   Test implementation makes use of certain approximations for thresholding; hence taking mode of initial values
    t_up = numpy.arange(float(min(t)), float(max(t))+dt_0/10.0, dt_0/10.0)
    v0_up = numpy.interp(t_up, t, v0.magnitude.flat)
    v1_up = numpy.interp(t_up, t, v1.magnitude.flat)
    d2_v0_up = numpy.diff(v0_up, n=2)
    d2_v1_up = numpy.diff(v1_up, n=2)
    dt_0_list = [ j for (i,j) in zip(d2_v0_up, t_up) if abs(i) >= 0.00005 ]
    dt_1_list = [ j for (i,j) in zip(d2_v1_up, t_up) if abs(i) >= 0.00005 ]
    dt_0_list_diff = numpy.diff(dt_0_list, n=1)
    dt_1_list_diff = numpy.diff(dt_1_list, n=1)
    dt_0_mode = scipy.stats.mode(dt_0_list_diff[0:10])[0][0]
    dt_1_mode = scipy.stats.mode(dt_1_list_diff[0:10])[0][0]
    assert_true (abs(dt_0_mode - dt_0) < 1e-9 or abs(dt_1_mode - dt_1) < 1e-9)


@register()
def issue442(sim):
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
    acsource = sim.ACSource(start=t_start, stop=t_stop, amplitude=0.5, offset=0.1, frequency=freq, phase=0.0)
    cells[0].inject(acsource)

    cells.record('v')
    sim.run(150.0)
    v = cells.get_data().segments[0].filter(name="v")[0]
    v0 = v[:, 0]
    t = v.times
    sim.end()

    t_start_ind = int(numpy.argmax(t >= t_start))
    t_stop_ind = int(numpy.argmax(t >= t_stop))

    # test for no change in vm before start time
    # note: exact matches not appropriate owing to floating point rounding errors
    assert_true(all(abs(val0 - v_rest*pq.mV) < 1e-9 for val0 in v0[:t_start_ind+1]))

    # test for change in vm at dt after start time
    assert_true(abs(v0[t_start_ind+1] - v0[t_start_ind]) >= 1e-9)

    # test for monotonic decay of vm after stop time
    assert_true(all(val0 >= val0_next for val0, val0_next in zip(v0[t_stop_ind:], v0[t_stop_ind+1:])))

    # test for accurate frequency; simply counts peaks
    peak_ctr = 0
    peak_ind = []
    for i in range(t_stop_ind-t_start_ind):
        if v0[t_start_ind+i-1] < v0[t_start_ind+i] and v0[t_start_ind+i] >= v0[t_start_ind+i+1]:
            peak_ctr+=1
            peak_ind.append(t_start_ind+i)
    assert_equal(peak_ctr, freq*1e-3*(t_stop-t_start))
    # also test for offset; peaks initially increase in magnitude
    assert_true(v0[peak_ind[0]] < v0[peak_ind[1]] and v0[peak_ind[1]] < v0[peak_ind[2]])


@register(exclude=["nest"])
def issue445(sim):
    """
    This test basically checks if a new value of current is calculated at every
    time step, and that the total number of time steps is as expected theoretically
    Note: NEST excluded as recording of electrode currents still to be implemented
    """
    sim_dt = 0.1
    simtime = 200.0
    sim.setup(timestep=sim_dt, min_delay=1.5)
    cells = sim.Population(1, sim.IF_curr_exp(v_thresh=-55.0, tau_refrac=5.0))
    t_start=50.0
    t_stop=125.0
    acsource = sim.ACSource(start=t_start, stop=t_stop, amplitude=0.5, offset=0.0, frequency=100.0, phase=0.0)
    cells[0].inject(acsource)
    acsource._record()

    sim.run(simtime)
    sim.end()

    i_t_ac, i_amp_ac = acsource._get_data()
    t_start_ind = numpy.argmax(i_t_ac >= t_start)
    t_stop_ind = numpy.argmax(i_t_ac >= t_stop)
    assert_true (all(val != val_next for val, val_next in zip(i_t_ac[t_start_ind:t_stop_ind-1], i_t_ac[t_start_ind+1:t_stop_ind])))
    # note: exact matches not appropriate owing to floating point rounding errors
    assert_true (( len(i_t_ac) - ((max(i_t_ac)-min(i_t_ac))/sim_dt + 1) )< 1e-9)


@register()
def issue451(sim):
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
    assert_true (all( (val.item()-v_rest)<1e-9 for val in v[:, 0]))


@register()
def issue483(sim):
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


@register()
def issue487(sim):
    """
    Test to ensure that DCSource and StepCurrentSource work properly
    for repeated runs. Problem existed under pyNN.neuron.
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
    assert_true (v_dc[int(simtime/dt)] < v_dc[int(simtime/dt)+1])
    # Test 2
    assert_true (v_step[int(simtime/dt)] < v_step[int(simtime/dt)+1])
    # check that membrane potential of cell undergoes a change
    # Test 3
    v_dc_2_arr = numpy.squeeze(numpy.array(v_dc_2))
    assert_false (numpy.isclose(v_dc_2_arr, v_rest).all())
    # check that membrane potential of cell undergoes no change till start of current injection
    # Test 4
    v_step_2_arr = numpy.squeeze(numpy.array(v_step_2))
    assert_true (numpy.isclose(v_step_2_arr[0:int(step_2.times[0]/dt)], v_rest).all())


@register()
def issue_465_474(sim):
    """
    Checks the current traces recorded for each of the four types of
    electrodes in pyNN, and verifies that:
    1) Length of the current traces are as expected
    2) Values at t = t_start and t = t_stop present
    3) Changes in current value occur at the expected time instant
    4) Change in Vm begins at the immediate next time instant following current injection
    """
    sim_dt = 0.1
    sim.setup(min_delay=1.0, timestep = sim_dt)

    v_rest = -60.0
    cells = sim.Population(4, sim.IF_curr_exp(v_thresh=-55.0, tau_refrac=5.0, v_rest=v_rest))
    cells.initialize(v=v_rest)

    amp=0.5
    offset = 0.1
    start=50.0
    stop=125.0

    acsource = sim.ACSource(start=start, stop=stop, amplitude=amp, offset=offset, frequency=100.0, phase=0.0)
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
    sim.end()

    vm = cells.get_data().segments[0].filter(name="v")[0]
    v_ac = vm[:, 0]
    v_dc = vm[:, 1]
    v_noise = vm[:, 2]
    v_step = vm[:, 3]

    i_t_ac, i_amp_ac = acsource.get_data()
    i_t_dc, i_amp_dc = dcsource.get_data()
    i_t_noise, i_amp_noise = noise.get_data()
    i_t_step, i_amp_step = step.get_data()

    # test for length of recorded current traces
    assert_true (len(i_t_ac) == len(i_amp_ac) == int(simtime/sim_dt)+1 == len(v_ac))
    assert_true (len(i_t_dc) == len(i_amp_dc) == int(simtime/sim_dt)+1 == len(v_dc))
    assert_true (len(i_t_noise) == len(i_amp_noise) == int(simtime/sim_dt)+1 == len(v_noise))
    assert_true (len(i_t_step) == len(i_amp_step) == int(simtime/sim_dt)+1 == len(v_step))

    # test to check values exist at start and end of simulation
    assert_true (i_t_ac[0]==0.0 and numpy.isclose(i_t_ac[-1],simtime))
    assert_true (i_t_dc[0]==0.0 and numpy.isclose(i_t_dc[-1],simtime))
    assert_true (i_t_noise[0]==0.0 and numpy.isclose(i_t_noise[-1],simtime))
    assert_true (i_t_step[0]==0.0 and numpy.isclose(i_t_step[-1],simtime))

    # test to check current changes at the expected time instant
    assert_true (i_amp_ac[(int(start/sim_dt))-1]==0 and i_amp_ac[int(start/sim_dt)]!=0)
    assert_true (i_amp_dc[int(start/sim_dt)-1]==0 and i_amp_dc[int(start/sim_dt)]!=0)
    assert_true (i_amp_noise[int(start/sim_dt)-1]==0 and i_amp_noise[int(start/sim_dt)]!=0)
    assert_true (i_amp_step[int(start/sim_dt)-1]==0 and i_amp_step[int(start/sim_dt)]!=0)

    # test to check vm changes at the time step following current initiation
    assert_true (numpy.isclose(v_ac[int(start/sim_dt)].item(),v_rest) and v_ac[int(start/sim_dt)+1]!=v_rest)
    assert_true (numpy.isclose(v_dc[int(start/sim_dt)].item(),v_rest) and v_dc[int(start/sim_dt)+1]!=v_rest)
    assert_true (numpy.isclose(v_noise[int(start/sim_dt)].item(),v_rest) and v_noise[int(start/sim_dt)+1]!=v_rest)
    assert_true (numpy.isclose(v_step[int(start/sim_dt)].item(),v_rest) and v_step[int(start/sim_dt)+1]!=v_rest)


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    test_changing_electrode(sim)
    ticket226(sim)
    issue165(sim)
    issue321(sim)
    issue437(sim)
    issue442(sim)
    issue445(sim)
    issue451(sim)
    issue483(sim)
    issue487(sim)
    issue_465_474(sim)
