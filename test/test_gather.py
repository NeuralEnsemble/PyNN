from pyNN.recording import gather
import numpy
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD

for x in range(7):
    N = pow(10, x)
    local_data = numpy.empty((N,2))
    local_data[:,0] = numpy.ones(N, dtype=float)*comm.rank
    local_data[:,1] = numpy.random.rand(N)
    
    start_time = time.time()
    all_data = gather(local_data)
    #print comm.rank, "local", local_data
    if comm.rank == 0:
    #    print "all", all_data
        print N, time.time()-start_time