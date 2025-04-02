from pyNN.space import Grid2D
from pyNN.parameters import IonicSpecies
from pyNN.utility import init_logging
from pyNN.morphology import load_morphology

import pytest

#from .fixtures import run_with_simulators

#@run_with_simulators("arbor", "neuron")
def test_scenario6(sim):
    """
    Small feed-forward network of pyramidal neurons
    """
    init_logging(logfile=None, debug=True)

    m = sim.morphology
    rp = m.random_placement

    sim.setup()

    pyr_morph = load_morphology("../test_data/oi15rpy4-1.CNG.swc", replace_axon=None)

    pyramidal_cell_class = sim.MultiCompartmentNeuron
    pyramidal_cell_class.label = "PyramidalNeuron"
    pyramidal_cell_class.ion_channels = {
        'pas': sim.PassiveLeak,
        'na': sim.NaChannel,
        'kdr': sim.KdrChannel
    }
    pyramidal_cell_class.post_synaptic_entities = {'AMPA': sim.CondExpPostSynapticResponse,
                                                   'GABA_A': sim.CondExpPostSynapticResponse}

    pyramidal_cell = pyramidal_cell_class(
                        morphology=pyr_morph,
                        pas={"conductance_density": m.uniform('all', 0.0003)},
                        na={"conductance_density": m.uniform('soma', 0.120)},
                        kdr={"conductance_density": m.by_distance(m.apical_dendrites(), lambda d: 0.05*d/200.0)},
                        ionic_species={
                            "na": IonicSpecies("na", reversal_potential=50.0),
                            "k": IonicSpecies("k", reversal_potential=-77.0)
                        },
                        cm=1.0,
                        Ra=500.0,
                        AMPA={"locations": rp(m.uniform('all', 0.05)),  # number per µm
                              "e_syn": 0.0,
                              "tau_syn": 2.0},
                        GABA_A={"locations": rp(m.by_distance(m.dendrites(), lambda d: 0.05 * (d < 50.0))),  # number per µm
                                "e_syn": -70.0,
                                "tau_syn": 5.0})

    pyramidal_cells = sim.Population(2, pyramidal_cell, initial_values={'v': -60.0}, structure=Grid2D())
    inputs = sim.Population(1000, sim.SpikeSourcePoisson(rate=1000.0))

    pyramidal_cells.record('spikes')
    pyramidal_cells[:1].record('v', locations=m.centre(m.soma()))
    pyramidal_cells[:1].record('v', locations=m.centre(m.apical_dendrites()))

    i2p = sim.Projection(inputs, pyramidal_cells,
                        connector=sim.AllToAllConnector(location_selector=m.random_section(m.apical_dendrites())),
                        synapse_type=sim.StaticSynapse(weight=0.5, delay=0.5),
                        receptor_type="AMPA"
                        )

    sim.run(10.0)
    data = pyramidal_cells.get_data().segments[0]
    assert len(data.analogsignals) == 2
    assert set(sig.name for sig in data.analogsignals) == set(("soma.centre.v", "apical_dendrites.centre.v"))


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    test_scenario6(sim)
