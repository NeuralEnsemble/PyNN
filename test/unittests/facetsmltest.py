"""
Unit tests for pyNN.facetsml package


"""

from pyNN.facetsml import *
import unittest

class PopulationInitTest(unittest.TestCase):
    """Tests of the __init__() method of the Population class."""
    
    def setUp(self):
        setup()
        
    
    def tearDown(self):
        pass
        
    def testInit(self):
        source33 = Population((3,3),"IF_curr_alpha",label="cellGroupA")
        target33 = Population((10,10),"LifNeuron",label="cellGroupB")
	writeDocument('testfacetsml.xml')
        
# ==============================================================================
class ProjectionInitTest(unittest.TestCase):
    """Tests of the __init__() method of the Population class."""
    
    def setUp(self):
        setup()
        
    
    def tearDown(self):
        pass
        
    def testInit(self):
	#to be sure source33 and target33 have well been created
	source33 = Population((3,3),"IF_curr_alpha",label="cellGroupA")
        target33 = Population((10,10),"LifNeuron",label="cellGroupB")
        proj1 = Projection(source33, target33, "fixed_probability",{'probability': '0.5'},label="aProjection")
	writeDocument('testfacetsml.xml')