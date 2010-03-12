from pylab import *
from pyNN.nest import *
from pyNN.utility import Timer
import os

N     = 50
setup(timestep=0.1, min_delay=0.1, max_delay=10.)
x     = Population((N,N), IF_curr_exp)
timer = Timer()
timer.start()

for i,dx in enumerate(numpy.linspace(0, 1, N)):
    for j,dy in enumerate(numpy.linspace(0, 1, N)):
        x[i,j].position = (numpy.random.rand(), numpy.random.rand(), 0)

#w        = RandomDistribution('uniform', (0,1))
w        = "0.2 + d/0.2"
#w        = 0.1
d        = RandomDistribution('uniform', (0.1,5.))
#d        = "0.2 + d/0.2"
#d        = 0.1
sp       = Space(periodic_boundaries=((0,1), (0,1), None))
safe     = False
verbose  = True
autapse  = False

##### Parameter to test the appropriate connector ####
case     = 4 
######################################################

if case is 1:
    conn  = DistanceDependentProbabilityConnector("d < 0.1", delays=d, weights=w, space=sp, safe=safe, verbose=verbose, allow_self_connections=autapse)
    fig_name = "DistanceDependent.png"
elif case is 2:
    conn  = FixedProbabilityConnector(0.05, weights=w, delays=d, space=sp, safe=safe, verbose=verbose, allow_self_connections=autapse)
    fig_name = "FixedProbability.png"
elif case is 3:
    conn  = AllToAllConnector(delays=d, weights=w, space=sp, safe=safe, verbose=verbose, allow_self_connections=autapse)
    fig_name = "AllToAll.png"
elif case is 4:
    conn  = FixedNumberPostConnector(100, weights=w, delays=d, space=sp, safe=safe, verbose=verbose, allow_self_connections=autapse)
    fig_name = "FixedNumberPost.png"
elif case is 5:
    conn  = FixedNumberPreConnector(100, weights=w, delays=d, space=sp, safe=safe, verbose=verbose, allow_self_connections=autapse)
    fig_name = "FixedNumberPre.png"
elif case is 6:
    conn  = OneToOneConnector(safe=safe, verbose=verbose)
    fig_name = "OneToOne.png"
elif case is 7:
    conn  = FromFileConnector('connections.dat', safe=safe, verbose=verbose)
    fig_name = "FromFile.png"

print "Generating data for %s" %fig_name
rng   = NumpyRNG(23434, num_processes=num_processes(), parallel_safe=False)
prj   = Projection(x, x, conn, rng=rng)

simulation_time = timer.elapsedTime()
print "Building time", simulation_time
print "Nb synapses built", len(prj)

print "Saving Positions...."
x.savePositions('positions.dat')

print "Saving Connections...."
prj.saveConnections('connections.dat', compatible_output=False)


def draw_rf(cell, positions, connections, color='k'):
    idx     = numpy.where(connections[:,1] == cell)[0]
    targets = connections[idx] 
    idx     = numpy.where(positions == cell)[0][0]
    source  = positions[idx]    
    for tgt in targets:        
        idx    = numpy.where(positions == tgt)[0][0]
        target = positions[idx]
        plot([source[1], target[1]], [source[2], target[2]], c=color)

def distances(pos_1, pos_2, N):
    # Since we deal with a toroidal space, we have to take the min distance
    # on the torus.
    dx = abs(pos_1[:,0]-pos_2[:,0])
    dy = abs(pos_1[:,1]-pos_2[:,1])
    dx = numpy.minimum(dx, N-dx)
    dy = numpy.minimum(dy, N-dy)
    return sqrt(dx*dx + dy*dy)

if rank() == 0:
    print "Generating and saving %s" %fig_name
    positions   = numpy.loadtxt('positions.dat')
    connections = numpy.loadtxt('connections.dat')
    N           = 1
    positions   = positions[numpy.argsort(positions[:,0])]
    idx_pre     = (connections[:,0] - 1).astype(int)
    idx_post    = (connections[:,1] - 1).astype(int)
    d           = distances(positions[idx_pre,1:3], positions[idx_post,1:3], N)

    subplot(231)
    title('Cells positions')
    plot(positions[:,1], positions[:,2], '.')
    subplot(232)
    title('Weights distribution')
    hist(connections[:,2], 50)
    subplot(233)
    title('Delay distribution')
    hist(connections[:,3], 50)
    subplot(234)
    ids   = numpy.random.permutation(numpy.unique(positions[:,0]))[0:6]
    colors = ['k', 'r', 'b', 'g', 'c', 'y'] 
    for count, cell in enumerate(ids):
        draw_rf(cell, positions, connections, colors[count])
    subplot(235)
    plot(d, connections[:,2], '.')

    subplot(236)
    plot(d, connections[:,3], '.')
    savefig(fig_name)
    os.remove('connections.dat')
    os.remove('positions.dat')