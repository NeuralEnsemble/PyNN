from pylab import *
from pyNN.utility import get_script_args, Timer
from pyNN.common import rank
from pyNN.space import *
import os

simulator_name = get_script_args(1)[0]
exec("from pyNN.%s import *" % simulator_name)

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
    dx = abs(pos_1[:,0]-pos_2[:,0])
    dy = abs(pos_1[:,1]-pos_2[:,1])
    dx = numpy.minimum(dx, N-dx)
    dy = numpy.minimum(dy, N-dy)
    return sqrt(dx*dx + dy*dy)


node_id = setup(timestep=0.1, min_delay=0.1, max_delay=5.)    
print "Creating cells population..."
N       = 20
x       = Population((N, N), IF_curr_exp)
x.set_positions(RandomPositions([(0, 1), (0, 1), None], seed=34295))
#x.set_positions(GridPositions([(0, 1), (0, 1), None]))


def test(cases=[1]):    
    
    sp            = Space(periodic_boundaries=((0,1), (0,1), None))
    safe          = False
    verbose       = True
    autapse       = False
    parallel_safe = True    
    render        = True
        
    for case in cases:
        #w = RandomDistribution('uniform', (0,1))
        w = "0.2 + d/0.2"
        #w = 0.1
        #w = lambda dist : 0.1 + numpy.random.rand(len(dist[0]))*sqrt(dist[0]**2 + dist[1]**2) 
        
        #delay = RandomDistribution('uniform', (0.1,5.))
        delay = "0.1 + d/0.2"
        #delay = 0.1    
        #delay = lambda distances : 0.1 + numpy.random.rand(len(distances))*distances 
    
        d_expression = "d < 0.1"
        #d_expression = "(d[0] < 0.05) & (d[1] < 0.05)"
        #d_expression = "(d[0]/(0.05**2) + d[1]/(0.1**2)) < 100*numpy.random.rand()"
    
        timer   = Timer()
        np      = num_processes()
        timer.start()    
        if case is 1:
            conn  = DistanceDependentProbabilityConnector(d_expression, delays=delay, weights=w, space=sp, safe=safe, verbose=verbose, allow_self_connections=autapse)
            fig_name = "DistanceDependent_%s_np_%d.png" %(simulator_name, np)
        elif case is 2:
            conn  = FixedProbabilityConnector(0.05, weights=w, delays=delay, space=sp, safe=safe, verbose=verbose, allow_self_connections=autapse)
            fig_name = "FixedProbability_%s_np_%d.png" %(simulator_name, np)
        elif case is 3:
            conn  = AllToAllConnector(delays=delay, weights=w, space=sp, safe=safe, verbose=verbose, allow_self_connections=autapse)
            fig_name = "AllToAll_%s_np_%d.png" %(simulator_name, np)
        elif case is 4:
            conn  = FixedNumberPostConnector(50, weights=w, delays=delay, space=sp, safe=safe, verbose=verbose, allow_self_connections=autapse)
            fig_name = "FixedNumberPost_%s_np_%d.png" %(simulator_name, np)
        elif case is 5:
            conn  = FixedNumberPreConnector(50, weights=w, delays=delay, space=sp, safe=safe, verbose=verbose, allow_self_connections=autapse)
            fig_name = "FixedNumberPre_%s_np_%d.png" %(simulator_name, np)
        elif case is 6:
            conn  = OneToOneConnector(safe=safe, weights=w, delays=delay, verbose=verbose)
            fig_name = "OneToOne_%s_np_%d.png" %(simulator_name, np)
        elif case is 7:
            conn  = FromFileConnector('connections.dat', safe=safe, verbose=verbose)
            fig_name = "FromFile_%s_np_%d.png" %(simulator_name, np)
        elif case is 8:
            conn  = SmallWorldConnector(degree=0.1, rewiring=0., weights=w, delays=delay, safe=safe, verbose=verbose, allow_self_connections=autapse, space=sp)
            fig_name = "SmallWorld_%s_np_%d.png" %(simulator_name, np)
        
        
        print "Generating data for %s" %fig_name
        rng   = NumpyRNG(23434, num_processes=np, parallel_safe=parallel_safe)
        prj   = Projection(x, x, conn, rng=rng)

        simulation_time = timer.elapsedTime()
        print "Building time", simulation_time
        print "Nb synapses built", len(prj)

        if render : 
            if not(os.path.isdir('Results')):
                os.mkdir('Results')

            print "Saving Positions...."
            x.savePositions('Results/positions.dat')

            print "Saving Connections...."
            prj.saveConnections('Results/connections.dat', compatible_output=False)
            
        if node_id == 0 and render:
            figure()
            print "Generating and saving %s" %fig_name
            positions   = numpy.loadtxt('Results/positions.dat')
            connections = numpy.loadtxt('Results/connections.dat')
            positions   = positions[numpy.argsort(positions[:,0])]
            idx_pre     = (connections[:,0] - x.first_id).astype(int)
            idx_post    = (connections[:,1] - x.first_id).astype(int)
            d           = distances(positions[idx_pre,1:3], positions[idx_post,1:3], 1)
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
            savefig("Results/" + fig_name)            
            os.remove('Results/connections.dat')
            os.remove('Results/positions.dat')
    
if __name__ == '__main__':
    #import hotshot, os
    #prof = hotshot.Profile("hotshot_edi_stats")
    #prof.runcall(test)
    #prof.close()
    #from hotshot import stats
    #s = stats.load("hotshot_edi_stats")
    #s.sort_stats("time").print_stats()
    test()