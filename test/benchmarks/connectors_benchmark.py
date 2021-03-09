from pylab import *
from pyNN.utility import Timer, init_logging, ProgressBar
import os

simulator_name = sys.argv[1]
exec("from pyNN.%s import *" % simulator_name)
test_cases = [int(x) for x in sys.argv[2:]]

from pyNN.recording import files
from pyNN.space import *

timer = Timer()
progress_bar = ProgressBar(mode='fixed', width=20)
init_logging("connectors_benchmark_%s.log" % simulator_name, debug=True)


def draw_rf(cell, positions, connections, color='k'):
    idx = np.where(connections[:, 1] == cell)[0]
    sources = connections[idx, 0]
    for src in sources:
        plot([positions[cell, 1], positions[src, 1]], [positions[cell, 2], positions[src, 2]], c=color)


def distances(pos_1, pos_2, N):
    dx = abs(pos_1[:, 0] - pos_2[:, 0])
    dy = abs(pos_1[:, 1] - pos_2[:, 1])
    dx = np.minimum(dx, N - dx)
    dy = np.minimum(dy, N - dy)
    return sqrt(dx * dx + dy * dy)

timer.start()
node_id = setup(timestep=0.1, min_delay=0.1, max_delay=4.)
print("Creating cells population...")
N = 30

structure = RandomStructure(Cuboid(1, 1, 1), origin=(0.5, 0.5, 0.5), rng=NumpyRNG(2652))
#structure = Grid2D(dx=1/float(N), dy=1/float(N))

x = Population(N**2, IF_curr_exp(), structure=structure)
mytime = timer.diff()
print("Time to build the cell population:", mytime, 's')


def test(cases=[1]):

    sp = Space(periodic_boundaries=((0, 1), (0, 1), None), axes='xy')
    safe = False
    callback = progress_bar.set_level
    autapse = False
    parallel_safe = True
    render = True
    to_file = True

    for case in cases:
        #w = RandomDistribution('uniform', (0,1))
        w = "0.2 + d/0.2"
        #w = 0.1
        #w = lambda dist : 0.1 + np.random.rand(len(dist[0]))*sqrt(dist[0]**2 + dist[1]**2)

        #delay = RandomDistribution('uniform', (0.1,5.))
        #delay = "0.1 + d/0.2"
        delay = 0.1
        #delay = lambda distances : 0.1 + np.random.rand(len(distances))*distances

        d_expression = "exp(-d**2/(2*0.1**2))"
        #d_expression = "(d[0] < 0.05) & (d[1] < 0.05)"
        #d_expression = "(d[0]/(0.05**2) + d[1]/(0.1**2)) < 100*np.random.rand()"

        timer = Timer()
        np = num_processes()
        timer.start()

        synapse = StaticSynapse(weight=w, delay=delay)
        rng = NumpyRNG(23434, parallel_safe=parallel_safe)

        if case is 1:
            conn = DistanceDependentProbabilityConnector(d_expression, safe=safe, callback=callback, allow_self_connections=autapse, rng=rng)
            fig_name = "DistanceDependent_%s_np_%d.png" % (simulator_name, np)
        elif case is 2:
            conn = FixedProbabilityConnector(0.02, safe=safe, callback=callback, allow_self_connections=autapse, rng=rng)
            fig_name = "FixedProbability_%s_np_%d.png" % (simulator_name, np)
        elif case is 3:
            conn = AllToAllConnector(delays=delay, safe=safe, callback=callback, allow_self_connections=autapse)
            fig_name = "AllToAll_%s_np_%d.png" % (simulator_name, np)
        elif case is 4:
            conn = FixedNumberPostConnector(50, safe=safe, callback=callback, allow_self_connections=autapse, rng=rng)
            fig_name = "FixedNumberPost_%s_np_%d.png" % (simulator_name, np)
        elif case is 5:
            conn = FixedNumberPreConnector(50, safe=safe, callback=callback, allow_self_connections=autapse, rng=rng)
            fig_name = "FixedNumberPre_%s_np_%d.png" % (simulator_name, np)
        elif case is 6:
            conn = OneToOneConnector(safe=safe, callback=callback)
            fig_name = "OneToOne_%s_np_%d.png" % (simulator_name, np)
        elif case is 7:
            conn = FromFileConnector(files.NumpyBinaryFile('Results/connections.dat', mode='r'), safe=safe, callback=callback, distributed=True)
            fig_name = "FromFile_%s_np_%d.png" % (simulator_name, np)
        elif case is 8:
            conn = SmallWorldConnector(degree=0.1, rewiring=0., safe=safe, callback=callback, allow_self_connections=autapse)
            fig_name = "SmallWorld_%s_np_%d.png" % (simulator_name, np)

        print("Generating data for %s" % fig_name)

        prj = Projection(x, x, conn, synapse, space=sp)

        mytime = timer.diff()
        print("Time to connect the cell population:", mytime, 's')
        print("Nb synapses built", prj.size())

        if to_file:
            if not(os.path.isdir('Results')):
                os.mkdir('Results')
            print("Saving Connections....")
            prj.save('all', files.NumpyBinaryFile('Results/connections.dat', mode='w'), gather=True)

        mytime = timer.diff()
        print("Time to save the projection:", mytime, 's')

        if render and to_file:
            print("Saving Positions....")
            x.save_positions('Results/positions.dat')
        end()

        if node_id == 0 and render and to_file:
            figure()
            print("Generating and saving %s" % fig_name)
            positions = np.loadtxt('Results/positions.dat')

            positions[:, 0] -= positions[:, 0].min()
            connections = files.NumpyBinaryFile('Results/connections.dat', mode='r').read()
            print(positions.shape, connections.shape)
            connections[:, 0] -= connections[:, 0].min()
            connections[:, 1] -= connections[:, 1].min()
            idx_pre = connections[:, 0].astype(int)
            idx_post = connections[:, 1].astype(int)
            d = distances(positions[idx_pre, 1:3], positions[idx_post, 1:3], 1)
            subplot(231)
            title('Cells positions')
            plot(positions[:, 1], positions[:, 2], '.')
            subplot(232)
            title('Weights distribution')
            hist(connections[:, 2], 50)
            subplot(233)
            title('Delay distribution')
            hist(connections[:, 3], 50)
            subplot(234)
            np.random.seed(74562)
            ids = np.random.permutation(positions[:, 0])[0:6]
            colors = ['k', 'r', 'b', 'g', 'c', 'y']
            for count, cell in enumerate(ids):
                draw_rf(cell, positions, connections, colors[count])
            subplot(235)
            plot(d, connections[:, 2], '.')

            subplot(236)
            plot(d, connections[:, 3], '.')
            savefig("Results/" + fig_name)
            #os.remove('Results/connections.dat')
            #os.remove('Results/positions.dat')
            show()

if __name__ == '__main__':
    #import hotshot, os
    #prof = hotshot.Profile("hotshot_edi_stats")
    #prof.runcall(test)
    #prof.close()
    #from hotshot import stats
    #s = stats.load("hotshot_edi_stats")
    #s.sort_stats("time").print_stats()
    test(test_cases)
