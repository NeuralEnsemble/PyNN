
from collections import defaultdict
import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pyNN.connectors import Connector

try:
    import pyNN.brian2 as sim
    import brian2
except ImportError:
    brian2 = False

import pytest


class MockConnector(Connector):

    def connect(self, projection):
        pass


@unittest.skipUnless(brian2, "Requires Brian")
class TestProjection(unittest.TestCase):

    def setUp(self):
        sim.setup()
        self.syn = sim.StaticSynapse(weight=0.123, delay=0.5)

    def test_partitioning(self):
        p1 = sim.Population(5, sim.IF_cond_exp())
        p2 = sim.Population(7, sim.IF_cond_exp())
        a = p1 + p2[1:4]
        # [0 2 3 4 5][x 1 2 3 x x x]
        prj = sim.Projection(a, a, MockConnector(), synapse_type=self.syn)
        presynaptic_indices = np.array([0, 3, 4, 6, 7])
        partitions = prj._partition(presynaptic_indices)
        self.assertEqual(len(partitions), 2)
        assert_array_equal(partitions[0], np.array([0, 3, 4]))
        assert_array_equal(partitions[1], np.array([2, 3]))

        # [0 1 2 3 4][x 1 2 3 x]
        self.assertEqual(prj._localize_index(0), (0, 0))
        self.assertEqual(prj._localize_index(3), (0, 3))
        self.assertEqual(prj._localize_index(5), (1, 1))
        self.assertEqual(prj._localize_index(7), (1, 3))


class MockSimulatorState:

    def __init__(self, dt=0.1):
        self.dt = dt
        self.t = 0
        self.current_sources = []
        self._steps_so_far = 0

    def run_until(self, t_stop):
        # todo: pregenerate the times, to minimize floating point errors from adding small values
        n_steps = int(round(t_stop / self.dt))
        for i in range(self._steps_so_far, n_steps):
            self.t = i * self.dt
            for cs in self.current_sources:
                cs._update_current()
        self._steps_so_far = n_steps
        self.t = t_stop


class RecordingInlineAdder:

    def __init__(self):
        self.values = []
        self.current_value = 0.0

    def __iadd__(self, value):
        self.values.append(value)
        self.current_value += value
        return self


class MockPopulation:
    pass


class MockBrianGroup:

    def __init__(self):
        self.i_inj = {33: RecordingInlineAdder()}


class MockCell:

    def __init__(self):
        self.parent = MockPopulation()
        self.parent.brian2_group = MockBrianGroup()


@unittest.skipUnless(brian2, "Requires Brian")
class TestCurrentSources(unittest.TestCase):

    def test_step_current_source(self):
        parameters = {
            "times": [1.0, 2.5, 3.0, 5.0],
            "amplitudes": [5.0, 10.0, 7.0, 0.0]
        }
        simulator_state = MockSimulatorState()
        with patch("pyNN.brian2.simulator.state", simulator_state):
            assert isinstance(sim.simulator.state, MockSimulatorState)
            current_source = sim.StepCurrentSource(**parameters)

            mock_cell = MockCell()
            current_source.cell_list.append(mock_cell)
            current_source.indices.append(33)

            current_source.record()
            assert_array_equal(
                current_source._brian_parameters["times"],
                np.array(parameters["times"]) * brian2.ms
            )
            assert_array_equal(
                current_source._brian_parameters["amplitudes"],
                np.array(parameters["amplitudes"]) * brian2.nA
            )
            assert_array_equal(
                current_source._times,
                current_source._brian_parameters["times"]
            )
            simulator_state.run_until(0.5)
            assert mock_cell.parent.brian2_group.i_inj[33].current_value == 0.0
            simulator_state.run_until(1.5)
            assert mock_cell.parent.brian2_group.i_inj[33].current_value == 5.0 * brian2.nA
            simulator_state.run_until(2.6)
            assert mock_cell.parent.brian2_group.i_inj[33].current_value == 10.0 * brian2.nA
            simulator_state.run_until(6.0)
            assert mock_cell.parent.brian2_group.i_inj[33].current_value == 0.0 * brian2.nA

            injected_deltas = np.array([x / brian2.nA for x in mock_cell.parent.brian2_group.i_inj[33].values])
            expected_deltas = np.array([5, 5, -3, -7])
            assert_array_almost_equal(injected_deltas, expected_deltas)

            recorded_times, recorded_amplitudes = current_source._get_data()
            expected_times = np.arange(0, 6.01, 0.1)
            expected_amplitudes = np.hstack((
                np.zeros(10),
                5.0 * np.ones(15),
                10.0 * np.ones(5),
                7.0 * np.ones(20),
                np.zeros(11)
            ))

            assert_array_equal(recorded_times, expected_times)
            assert_array_equal(recorded_amplitudes, expected_amplitudes)

    def test_dc_source(self):
        parameters = {
            "start": 2.0,
            "stop": 3.5,
            "amplitude": 12.3
        }
        simulator_state = MockSimulatorState()
        with patch("pyNN.brian2.simulator.state", simulator_state):
            assert isinstance(sim.simulator.state, MockSimulatorState)
            current_source = sim.DCSource(**parameters)

            mock_cell = MockCell()
            current_source.cell_list.append(mock_cell)
            current_source.indices.append(33)

            current_source.record()
            assert current_source._brian_parameters["start"] == parameters["start"] * brian2.ms
            assert current_source._brian_parameters["amplitude"] == parameters["amplitude"] * brian2.nA

            simulator_state.run_until(1.9)
            assert mock_cell.parent.brian2_group.i_inj[33].current_value == 0.0
            simulator_state.run_until(2.1)
            assert mock_cell.parent.brian2_group.i_inj[33].current_value == 12.3 * brian2.nA
            simulator_state.run_until(3.4)
            assert mock_cell.parent.brian2_group.i_inj[33].current_value == 12.3 * brian2.nA
            simulator_state.run_until(3.6)
            assert mock_cell.parent.brian2_group.i_inj[33].current_value == 0.0 * brian2.nA
            simulator_state.run_until(4.0)
            assert mock_cell.parent.brian2_group.i_inj[33].current_value == 0.0 * brian2.nA

            injected_deltas = np.array([x / brian2.nA for x in mock_cell.parent.brian2_group.i_inj[33].values])
            expected_deltas = np.array([0, 12.3, -12.3])
            assert_array_almost_equal(injected_deltas, expected_deltas)

            recorded_times, recorded_amplitudes = current_source._get_data()
            expected_times = np.arange(0, 4.01, 0.1)
            expected_amplitudes = np.hstack((
                np.zeros(20),
                12.3 * np.ones(15),
                np.zeros(6)
            ))

            assert_array_equal(recorded_times, expected_times)
            assert_array_equal(recorded_amplitudes, expected_amplitudes)

    def test_noisy_current_source(self):
        parameters = {
            "mean": 10.0,
            "stdev": 2.0,
            "start": 1.0,
            "stop": 2.0,
            "dt": 0.1
        }
        simulator_state = MockSimulatorState()
        with patch("pyNN.brian2.simulator.state", simulator_state):
            assert isinstance(sim.simulator.state, MockSimulatorState)
            current_source = sim.NoisyCurrentSource(**parameters)

            mock_cell = MockCell()
            current_source.cell_list.append(mock_cell)
            current_source.indices.append(33)

            current_source.record()
            assert current_source._brian_parameters["mean"] == parameters["mean"] * brian2.nA
            assert current_source._brian_parameters["stop"] == parameters["stop"] * brian2.ms

            simulator_state.run_until(0.5)
            assert mock_cell.parent.brian2_group.i_inj[33].current_value == 0.0
            simulator_state.run_until(1.5)
            assert mock_cell.parent.brian2_group.i_inj[33].current_value != 0.0
            simulator_state.run_until(3.0)
            # without the following "approx", sometimes fails with current values in units of yoctoamps!
            assert mock_cell.parent.brian2_group.i_inj[33].current_value == pytest.approx(0.0)

            injected_deltas = np.array([x / brian2.nA for x in mock_cell.parent.brian2_group.i_inj[33].values])
            recorded_times, recorded_amplitudes = current_source._get_data()
            expected_times = np.arange(0, 3.01, 0.1)
            assert_array_almost_equal(recorded_times, expected_times)

    def test_ac_source(self):
        parameters = {
            "amplitude": 0.23,
            "offset": 1.23,
            "frequency": 100.0,
            "phase": 0.5,
            "start": 1.0,
            "stop": 11.0,
        }
        simulator_state = MockSimulatorState()
        with patch("pyNN.brian2.simulator.state", simulator_state):
            assert isinstance(sim.simulator.state, MockSimulatorState)
            current_source = sim.ACSource(**parameters)

            mock_cell = MockCell()
            current_source.cell_list.append(mock_cell)
            current_source.indices.append(33)

            current_source.record()
            assert current_source._brian_parameters["offset"] == parameters["offset"] * brian2.nA
            assert current_source._brian_parameters["phase"] == parameters["phase"]

            simulator_state.run_until(0.5)
            assert mock_cell.parent.brian2_group.i_inj[33].current_value == 0.0
            simulator_state.run_until(1.5)
            assert mock_cell.parent.brian2_group.i_inj[33].current_value != 0.0
            # todo: check the sine wave in the current_value
            simulator_state.run_until(12.0)
            assert mock_cell.parent.brian2_group.i_inj[33].current_value == 0.0

            injected_deltas = np.array([x / brian2.nA for x in mock_cell.parent.brian2_group.i_inj[33].values])
            recorded_times, recorded_amplitudes = current_source._get_data()
            expected_times = np.arange(0, 12.01, 0.1)
            assert_array_almost_equal(recorded_times, expected_times)
            # todo: check the recorded sine wave



        # todo: add tests for changing parameters part-way through a run


if __name__ == '__main__':
    unittest.main()
