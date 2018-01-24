


import pyNN.nest as sim
from pyNN.random import RandomDistribution as RD
from pyNN.network import Network
from pyNN.serialization import export_to_abc

sim.setup()

p1 = sim.Population(10, 
                    sim.IF_cond_exp(tau_m=lambda i: 10 + 0.1*i,
                                    cm=RD('normal', (0.5, 0.05))),
                    label="population_one")
p2 = sim.Population(20, 
                    sim.IF_curr_alpha(tau_m=lambda i: 10 + 0.1*i),
                    label="population_two")

prj = sim.Projection(p1, p2,
                     sim.FixedProbabilityConnector(p_connect=0.5),
                     synapse_type=sim.StaticSynapse(weight=RD('uniform', [0.0, 0.1]),
                                                    delay=0.5),
                     receptor_type='excitatory')

net = Network(p1, p2, prj)

export_to_abc(net, "tmp_serialization_test")
