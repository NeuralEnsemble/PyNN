from NeuroTools.parameters import ParameterSet
import sys
from math import sqrt
from pyNN.space import Space, Grid2D

P = ParameterSet(sys.argv[1])

exec("import pyNN.%s as sim" % P.simulator)

sim.setup()

dx1 = dy1 = 500.0 / sqrt(P.n1)
dx2 = dy2 = 500.0 / sqrt(P.n2)
struct1 = Grid2D(dx=dx1, dy=dy1)
struct2 = Grid2D(dx=dx2, dy=dy2)

p1 = sim.Population(P.n1, sim.IF_cond_exp, structure=struct1)
p2 = sim.Population(P.n2, sim.IF_cond_exp, structure=struct2)

space = Space()
DDPC = sim.DistanceDependentProbabilityConnector
c = DDPC(P.d_expression, P.allow_self_connections, P.weights, P.delays, space, P.safe)

prj = sim.Projection(p1, p2, c)

sys.stdout.write(p1.describe().encode('utf-8'))
sys.stdout.write(p2.describe().encode('utf-8'))
sys.stdout.write(prj.describe().encode('utf-8'))


sim.end()
