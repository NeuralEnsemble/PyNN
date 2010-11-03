
from scenarios import *
#from pyNN import common
import pyNN.brian

def test_all():
    #common.simulator = pyNN.brian.simulator
    #common.recording = pyNN.brian.recording
    scenario1(pyNN.brian)