"""
Unit tests for pyNN.neuroml package


"""

from pyNN.neuroml import *
import unittest

class PopulationInitTest(unittest.TestCase):
    """Tests of the __init__() method of the Population class."""

    def setUp(self):
        setup(file="testneuroml1.xml")


    def tearDown(self):
        end()

    def testInit(self):
        source22 = Population((2,2), IF_cond_exp, label="cellGroupA")
        #target33 = Population((10,10), IF_cond_exp, label="cellGroupB")

# ==============================================================================
#class ProjectionInitTest(unittest.TestCase):
#    """Tests of the __init__() method of the Projection class."""
#
#    def setUp(self):
#        setup(file="testneuroml2.xml")
#
#
#    def tearDown(self):
#        end()
#
#    def testInit(self):
#        #to be sure source33 and target33 have well been created
#        source33 = Population((3,3),"IF_cond_alpha",label="cellGroupA")
#        target33 = Population((10,10),"IF_cond_exp",label="cellGroupB")
#        proj1 = Projection(source33, target33, "fixed_probability",{'probability': '0.5'},label="aProjection")

# ==============================================================================        
if __name__ == "__main__":
    unittest.main()
