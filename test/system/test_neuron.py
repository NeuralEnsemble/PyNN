
from scenarios import *
#from pyNN import common
import pyNN.neuron

def test_all():
    #common.simulator = pyNN.neuron.simulator
    #common.recording = pyNN.neuron.recording
    scenario1(pyNN.neuron)