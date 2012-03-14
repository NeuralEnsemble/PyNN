"""
Example script to help explore what needs to be done to integrate MUSIC with PyNN

Script does not work at present.

"""

from pyNN import music

vizapp = music.Config("vizapp", 1, "/path/to/vizapp", "args")

music.setup(music.Config("neuron", 10), music.Config("nest", 20), vizapp)

sim1, sim2 = music.get_simulators("neuron", "nest")
sim1.setup(timestep=0.025)
sim2.setup(timestep=0.1)
cell_parameters = {"tau_m": 12.0, "cm": 0.8, "v_thresh": -50.0, "v_reset": -65.0}
pE = sim1.Population((100,100), sim.IF_cond_exp, cell_parameters, label="excitatory neurons")
pI = sim2.Population((50,50), sim.IF_cond_exp, cell_parameters, label="inhibitory neurons")
all = pE + pI
def connector(sim):
    DDPC = getattr(sim, "DistanceDependentProbabilityConnector")
    return DDPC("exp(-d**2/400.0)", weights=0.05, delays="0.5+0.01d")
e2e = sim1.Projection(pE, pE, connector(sim1), target="excitatory")
e2i = music.Projection(pE, pI, connector(music), target="excitatory")
i2i = sim2.Projection(pI, pI, connector(sim2), target="inhibitory")

output = music.Port(pE, "spikes", vizapp, "pE_spikes_viz")

music.run(1000.0)
