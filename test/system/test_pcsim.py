
from scenarios import *
#from pyNN import common
import pyNN.pcsim

def test_all():
    #common.simulator = pyNN.pcsim.simulator
    #common.recording = pyNN.pcsim.recording
    scenario1(pyNN.pcsim)