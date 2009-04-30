import pyNN.brian as sim
import unittest
from pyNN import random

# ==============================================================================
class ProjectionInitTest(unittest.TestCase):
    """Tests of the __init__() method of the Projection class."""
        
    def setUp(self):
        sim.setup()
        sim.Population.nPop = 0
        sim.Projection.nProj = 0
        self.target33    = sim.Population((3,3), sim.IF_curr_alpha)
        self.target6     = sim.Population((6,), sim.IF_curr_alpha)
        self.target1     = sim.Population((1,), sim.IF_cond_exp)
        self.source5     = sim.Population((5,), sim.SpikeSourcePoisson)
        self.source22    = sim.Population((2,2), sim.SpikeSourcePoisson)
        self.source33    = sim.Population((3,3), sim.SpikeSourcePoisson)
        self.expoisson33 = sim.Population((3,3), sim.SpikeSourcePoisson,{'rate': 100})
        
    def testAllToAll(self):
        for srcP in [self.source5, self.source22, self.target33]:
            for tgtP in [self.target6, self.target33]:
                if srcP == tgtP:
                    prj = sim.Projection(srcP, tgtP, sim.AllToAllConnector(allow_self_connections=False,
                                                                           weights=1.234))
                else:
                    prj = sim.Projection(srcP, tgtP, sim.AllToAllConnector(weights=1.234))
                weights = prj._connections.W.toarray().flatten().tolist()
                self.assertEqual(weights, [1.234]*len(prj))
        
    def testFixedProbability(self):
        """For all connections created with "fixedProbability"..."""
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target1, self.target6, self.target33]:
                prj1 = sim.Projection(srcP, tgtP, sim.FixedProbabilityConnector(0.5), rng=random.NumpyRNG(12345))
                prj2 = sim.Projection(srcP, tgtP, sim.FixedProbabilityConnector(0.5), rng=random.NativeRNG(12345))
                for prj in prj1, prj2:
                    assert (0 < len(prj) < len(srcP)*len(tgtP)), 'len(prj) = %d, len(srcP)*len(tgtP) = %d' % (len(prj), len(srcP)*len(tgtP))
                
    def testOneToOne(self):
        """For all connections created with "OneToOne" ..."""
        prj = sim.Projection(self.source33, self.target33, sim.OneToOneConnector(weights=0.5))
        self.assertEqual(prj._connections.W.getnnz(), self.target33.cell.size)
     
    def testDistanceDependentProbability(self):
        """For all connections created with "distanceDependentProbability"..."""
        # Test should be improved..."
        for rngclass in (random.NumpyRNG, random.NativeRNG):
            for expr in ('exp(-d)', 'd < 0.5'):
        #rngclass = random.NumpyRNG
        #expr = 'exp(-d)'
                prj = sim.Projection(self.source33, self.target33,
                                        sim.DistanceDependentProbabilityConnector(d_expression=expr),
                                        rng=rngclass(12345))
                assert (0 < len(prj) <= len(self.source33)*len(self.target33)), len(prj)
        self.assertRaises(ZeroDivisionError, sim.DistanceDependentProbabilityConnector, d_expression="d/0.0")



if __name__ == "__main__":
    unittest.main()