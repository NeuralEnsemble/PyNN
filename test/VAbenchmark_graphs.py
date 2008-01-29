import pylab, sys
from NeuroTools import nstats

if len(sys.argv) < 2:
    print "Usage: python VAbenchmark_graphs.py <benchmark>\n\nwhere <benchmark> is either CUBA or COBA."
    sys.exit(1)
benchmark = sys.argv[1]

simulators = ('nest2','nest1','neuron','pcsim')
v_thresh = -50.0
#pylab.rcParams['backend'] = 'PS'
CM=1/2.54
pylab.rcParams['figure.figsize'] = [60*CM,40*CM] # inches

ny = 4
dy = 1.0/ny; dx = 1.0/len(simulators);
h = 0.8*dy; w = 0.8*dx
y0 = (1-ny*h)/(ny+1);
x0 = 0.05

def get_header(filename):
    cmd = ''
    f = open(filename, 'r')
    for line in f.readlines():
        if line[0] == '#':
            cmd += line[1:].strip() + ';'
    f.close()
    return cmd

def population_isis(spiketimes,ids):
    """Calculate the interspike intervals for each cell in the population,
    starting with a 1D array of spiketimes and a corresponding array of IDS.
    """
    population_spiketimes = nstats.sort_by_id(spiketimes,ids)
    population_isis = [nstats.isi(s) for s in population_spiketimes]
    return population_isis

def isi_hist(population_isis,bins):
    isis = pylab.array([])
    for isi in population_isis:
        isis = pylab.concatenate((isis, isi))
    isihist = nstats.histc(isis,bins)
    return isihist

def plot_hist(axes,hist,bins,width,xlabel=None,ylabel="n in bin",
                     xticks=None, xticklabels=None, xmin=None, ymax=None):
    pylab.axes(axes)
    if xlabel: pylab.xlabel(xlabel)
    if ylabel: pylab.ylabel(ylabel)
    for t,n in zip(bins,hist):
        pylab.bar(t,n,width=width)
    if xmin: pylab.axis(xmin=xmin)
    if ymax: pylab.axis(ymax=ymax)
    if xticks is not None:
        if xticklabels:
            pylab.xticks(xticks,xticklabels)
        else:
            pylab.xticks(xticks)



x = x0; 
for simulator in simulators:

    col = 1
    pylab.axes([x,y0+2.9*dy,w,h])
    pylab.title(simulator[:6].upper(),fontsize='x-large')
    pylab.ylabel("Membrane potential (mV)")
    
    # Get info about dataset from header of .v file
    exec(get_header("VAbenchmark_%s_exc_%s.v" % (benchmark,simulator)))
    
    # Plot membrane potential trace
    allvdata = pylab.load("VAbenchmark_%s_exc_%s.v" % (benchmark,simulator), comments='#')
    cell_ids = allvdata[:,1].astype(int)
    allvdata = allvdata[:,0]
    sortmap = pylab.argsort(cell_ids, kind='mergesort')
    cell_ids = pylab.take(cell_ids,sortmap)
    allvdata = pylab.take(allvdata,sortmap)
    for i in 0,1:
        tdata = pylab.arange(0,(n+1)*dt,dt)
        vdata = allvdata.compress(cell_ids==i)
        vdata = pylab.where(vdata>=v_thresh-0.05,0.0,vdata) # add fake APs for plotting
        if len(tdata) > len(vdata):
            print "Warning. Shortening tdata from %d to %d elements (%s)" % (len(tdata),len(vdata),simulator)
            tdata = tdata[0:len(vdata)]
        assert len(tdata)==len(vdata), "%d != %d (%s)" % (len(tdata),len(vdata),simulator)
    
        #for i in 0,1:
        #    pylab.plot(tdata,vdata[i*n:(i+1)*n])
        pylab.plot(tdata,vdata)
        #pylab.axis(ymin=-70,ymax=0.0)
    
    # Plot spike rasters
    exc_spikedata = pylab.load("VAbenchmark_%s_exc_%s.ras" % (benchmark,simulator))
    inh_spikedata = pylab.load("VAbenchmark_%s_inh_%s.ras" % (benchmark,simulator))

    exc_spikeids   = exc_spikedata[:,1].astype(int)
    inh_spikeids   = inh_spikedata[:,1].astype(int)
    exc_spiketimes = exc_spikedata[:,0]
    inh_spiketimes = inh_spikedata[:,0]

    pylab.axes([x,y0+2*dy,w,h])
    pylab.xlabel("Time (ms)")
    pylab.ylabel("cell #")
    pylab.plot(exc_spiketimes,exc_spikeids, markersize=1, marker='.', linestyle='')
    pylab.axis([0, n*dt, 0, 320])

    # Inter-spike-interval histograms
    bins = pylab.exp(pylab.arange(0,8,0.2))
    exc_pop_isis = population_isis(exc_spiketimes,exc_spikeids)
    isihist = isi_hist(exc_pop_isis,bins)
    plot_hist([x,y0+dy,0.4*w,h],isihist,pylab.arange(0,8,0.2),0.2,
        xlabel="Inter-spike interval (ms)",xticks=pylab.log([3,10,30,100,1000]),
        xticklabels=['3','10','30','100','1000'],xmin=pylab.log(2),ymax=1.0e4)
    pylab.title('Exc')

    inh_pop_isis = population_isis(inh_spiketimes,inh_spikeids)
    isihist = isi_hist(inh_pop_isis,bins)
    plot_hist([x+0.45*dx,y0+dy,0.4*w,h],isihist,pylab.arange(0,8,0.2),0.2,
        xlabel="Inter-spike interval (ms)",xticks=pylab.log([3,10,30,100,1000]),
        xticklabels=['3','10','30','100','1000'],xmin=pylab.log(2),ymax=0.2e4)
    pylab.title('Inh')

    # Histograms of coefficients of variation of ISI
    for pop_isis,xoffset,ymax in zip([exc_pop_isis, inh_pop_isis],[0.0,0.45*dx],[800,200]):
        cvs = []
        for isi in pop_isis:
            if len(isi) > 1:        # need at least two spikes to have a CV
                m = pylab.mean(isi)
                if m > 0: cvs.append(pylab.std(isi)/m)
        bins = pylab.arange(0,3,0.1)
        cvhist = nstats.histc(cvs,bins)
        plot_hist([x+xoffset,y0,0.4*w,h], cvhist, bins, 0.1, xlabel="ISI CV", ymax=ymax)
    
    x += dx
    
pylab.savefig("VAbenchmark_%s_exc.png" % benchmark)
pylab.show()
