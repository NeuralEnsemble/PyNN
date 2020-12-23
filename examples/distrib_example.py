
from mpi4py import MPI
from pyNN.utility import get_script_args

import sys
import numpy as np

simulator = get_script_args(1)[0]
exec("import pyNN.%s as sim" % simulator)


comm = MPI.COMM_WORLD

sim.setup(debug=True)

print("\nThis is node %d (%d of %d)" % (sim.rank(), sim.rank() + 1, sim.num_processes()))
assert comm.rank == sim.rank()
assert comm.size == sim.num_processes()

data1 = np.empty(100, dtype=float)
if comm.rank == 0:
    data1 = np.arange(100, dtype=float)
else:
    pass
comm.Bcast([data1, MPI.DOUBLE], root=0)
print(comm.rank, data1)

data2 = np.arange(comm.rank, 10 + comm.rank, dtype=float)
print(comm.rank, data2)
data2g = np.empty(10 * comm.size)
comm.Gather([data2, MPI.DOUBLE], [data2g, MPI.DOUBLE], root=0)
if comm.rank == 0:
    print("gathered (2):", data2g)

data3 = np.arange(0, 5 * (comm.rank + 1), dtype=float)
print(comm.rank, data3)
if comm.rank == 0:
    sizes = range(5, 5 * comm.size + 1, 5)
    disp = [size - 5 for size in sizes]
    data3g = np.empty(sum(sizes))
else:
    sizes = disp = []
    data3g = np.empty([])
comm.Gatherv([data3, data3.size, MPI.DOUBLE], [data3g, (sizes, disp), MPI.DOUBLE], root=0)
if comm.rank == 0:
    print("gathered (3):", data3g)


def gather(data):
    assert isinstance(data, np.ndarray)
    # first we pass the data size
    size = data.size
    sizes = comm.gather(size, root=0) or []
    # now we pass the data
    displacements = [sum(sizes[:i]) for i in range(len(sizes))]
    print(comm.rank, "sizes=", sizes, "displacements=", displacements)
    gdata = np.empty(sum(sizes))
    comm.Gatherv([data, size, MPI.DOUBLE], [gdata, (sizes, displacements), MPI.DOUBLE], root=0)
    return gdata


data3g = gather(data3)
if comm.rank == 0:
    print("gathered (3, again):", data3g)


sim.end()
