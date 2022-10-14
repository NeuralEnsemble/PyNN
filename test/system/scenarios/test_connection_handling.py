"""
Tests of the fine-grained connection API

"""
from numpy.testing import assert_array_equal
import numpy as np
from pyNN import common
from .fixtures import run_with_simulators
import pytest


@run_with_simulators("nest", "neuron", "brian2")
def test_connections_attribute(sim):
    sim.setup()
    p1 = sim.Population(4, sim.SpikeSourceArray())
    p2 = sim.Population(3, sim.IF_cond_exp())
    prj = sim.Projection(p1, p2, sim.FixedProbabilityConnector(0.7),
                         sim.StaticSynapse(weight=0.05, delay=0.5))

    connections = list(prj.connections)
    assert len(connections) == len(prj)
    assert isinstance(connections[0], common.Connection)


@run_with_simulators("nest", "neuron", "brian2")
def test_connection_access_weight_and_delay(sim):
    sim.setup()
    p1 = sim.Population(4, sim.SpikeSourceArray())
    p2 = sim.Population(3, sim.IF_cond_exp())
    prj = sim.Projection(p1, p2, sim.FixedProbabilityConnector(0.8),
                         sim.StaticSynapse(weight=0.05, delay=0.5))

    connections = list(prj.connections)
    assert connections[2].weight == pytest.approx(0.05)  # places=9)
    assert connections[2].delay == pytest.approx(0.5)  # places=9)
    connections[2].weight = 0.0123
    connections[2].delay = 1.0
    target = np.empty((prj.size(), 2))
    target[:, 0] = 0.05
    target[:, 1] = 0.5
    target[2, 0] = 0.0123
    target[2, 1] = 1.0
    assert_array_equal(np.array(prj.get(('weight', 'delay'), format='list', with_address=False)),
                       target)


@run_with_simulators("nest", "neuron", "brian2")
def test_issue672(sim):
    """
    Check that creating new Projections does not mess up existing ones.
    """
    sim.setup(verbosity="error")

    p1 = sim.Population(5, sim.IF_curr_exp())
    p2 = sim.Population(4, sim.IF_curr_exp())
    p3 = sim.Population(6, sim.IF_curr_exp())

    prj1 = sim.Projection(p2, p3, sim.AllToAllConnector(), sim.StaticSynapse(weight=lambda d: d))
    # Get weight array of first Projection
    wA = prj1.get("weight", format="array")
    # Create a new Projection
    prj2 = sim.Projection(p2, p3, sim.AllToAllConnector(), sim.StaticSynapse(weight=lambda w: 1))
    # Get weight array of first Projection again
    #   - incorrect use of caching could lead to this giving different results
    wB = prj1.get("weight", format="array")

    assert_array_equal(wA, wB)


# @run_with_simulators("nest", "neuron", "brian2")
# def test_update_synaptic_plasticity_parameters(sim):
# #     sim.setup()
#     p1 = sim.Population(3, sim.IF_cond_exp(), label="presynaptic")
#     p2 = sim.Population(2, sim.IF_cond_exp(), label="postsynaptic")

#     stdp_model = sim.STDPMechanism(
#         timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,
#                                             A_plus=0.011, A_minus=0.012),
#         weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.0000001),
#         weight=0.00000005,
#         delay=0.2)
#     connections = sim.Projection(p1, p2, sim.AllToAllConnector(), stdp_model)

#     assert (connections.get("A_minus", format="array") == 0.012).all()
#     connections.set(A_minus=0.013)
#     assert (connections.get("A_minus", format="array") == 0.013).all()
#     connections.set(A_minus=np.array([0.01, 0.011, 0.012, 0.013, 0.014, 0.015]))
#     assert_array_equal(connections.get("A_minus", format="array"),
#                        np.array([[0.01, 0.011], [0.012, 0.013], [0.014, 0.015]]))


@run_with_simulators("nest", "neuron")
def test_issue652(sim):
    """Correctly handle A_plus = 0 in SpikePairRule."""
    sim.setup()

    neural_population1 = sim.Population(10, sim.IF_cond_exp())
    neural_population2 = sim.Population(10, sim.IF_cond_exp())

    amount_of_neurons_to_connect_to = 5
    synaptic_weight = 0.5

    synapse_type = sim.STDPMechanism(
        weight=synaptic_weight,
        timing_dependence=sim.SpikePairRule(
            tau_plus=20,
            tau_minus=20,
            A_plus=0.0,
            A_minus=0.0,
        ),
        weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=1)
    )

    connection_to_input = sim.Projection(
        neural_population1, neural_population2,
        sim.FixedNumberPreConnector(amount_of_neurons_to_connect_to), synapse_type
    )

    a_plus, a_minus = connection_to_input.get(["A_plus", "A_minus"], format="array")
    assert a_plus[~np.isnan(a_plus)][0] == 0.0
    assert a_minus[~np.isnan(a_minus)][0] == 0.0

    synapse_type = sim.STDPMechanism(
        weight=synaptic_weight,
        timing_dependence=sim.SpikePairRule(
            tau_plus=20,
            tau_minus=20,
            A_plus=0.0,
            A_minus=0.5,
        ),
        weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=1)
    )

    connection_to_input = sim.Projection(
        neural_population1, neural_population2,
        sim.FixedNumberPreConnector(amount_of_neurons_to_connect_to), synapse_type
    )

    a_plus, a_minus = connection_to_input.get(["A_plus", "A_minus"], format="array")
    assert a_plus[~np.isnan(a_plus)][0] == 0.0
    assert a_minus[~np.isnan(a_minus)][0] == 0.5


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    test_connections_attribute(sim)
