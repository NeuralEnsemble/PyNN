
from scenarios import *
#from pyNN import common
import pyNN.nest

def test_all():
    #common.simulator = pyNN.nest.simulator
    #common.recording = pyNN.nest.recording
    scenario1(pyNN.nest)