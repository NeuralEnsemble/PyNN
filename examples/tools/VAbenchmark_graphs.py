"""
Plot graphs showing the results of running the VAbenchmarks.py script.
"""

import pylab, sys
import numpy
from NeuroTools import signals, plotting
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

if len(sys.argv) < 2:
    print "Usage: python VAbenchmark_graphs.py <benchmark>\n\nwhere <benchmark> is either CUBA or COBA."
    sys.exit(1)
benchmark = sys.argv[1]

simulators = ('neuron', 'nest', 'pcsim', 'brian')
#simulators = ['nest']
nodes = (1,)
#nodes = (1,2,4)
v_thresh = -50.0
#pylab.rcParams['backend'] = 'PS'
CM=1/2.54
pylab.rcParams['figure.figsize'] = [60*CM,40*CM] # inches

ny = 4
dy = 1.0/ny; dx = 1.0/(len(simulators)*len(nodes));
h = 0.8*dy; w = 0.8*dx
y0 = (1-ny*h)/(ny+1);
x0 = 0.05

def get_header(filename):
    metadata = {}
    f = open(filename, 'r')
    for line in f.readlines():
        if line[0] == '#':
            key, value = line[1:].strip().split("=")
            key = key.strip()
            try:
                metadata[key] = eval(value)
            except (NameError, SyntaxError):
                metadata[key] = value.strip()
    f.close()
    return metadata

def population_isis(spiketimes,ids):
    """Calculate the interspike intervals for each cell in the population,
    starting with a 1D array of spiketimes and a corresponding array of IDS.
    """
    population_spiketimes = nstats.sort_by_id(spiketimes,ids)
    population_isis = [nstats.isi(s) for s in population_spiketimes]
    return population_isis

def plot_hist(subplot, hist, bins, width, xlabel=None, ylabel="n in bin",
              xticks=None, xticklabels=None, xmin=None, ymax=None):
    if xlabel: subplot.set_xlabel(xlabel)
    if ylabel: subplot.set_ylabel(ylabel)
    for t,n in zip(bins,hist):
        subplot.bar(t,n,width=width)
    if xmin: subplot.set_xlim(xmin=xmin)
    if ymax: subplot.set_ylim(ymax=ymax)
    if xticks is not None: subplot.set_xticks(xticks)
    if xticklabels: subplot.set_xticklabels(xticklabels)
            

x = x0;
figure = pylab.Figure()
for simulator in simulators:
    for num_nodes in nodes:
        col = 1
        subplot = figure.add_axes([x,y0+2.9*dy,w,h])
        subplot.set_title("%s (np%d)" % (simulator[:6].upper(),num_nodes), fontsize='x-large')
        subplot.set_ylabel("Membrane potential (mV)")
        
        # Get info about dataset from header of .v file
        metadata = get_header("Results/VAbenchmark_%s_exc_%s_np%d.v" % (benchmark, simulator, num_nodes))
        n = metadata['n']
        dt = metadata['dt']
        
        # Plot membrane potential trace
        allvdata = numpy.loadtxt("Results/VAbenchmark_%s_exc_%s_np%d.v" % (benchmark, simulator, num_nodes), comments='#')
        cell_ids = allvdata[:,1].astype(int)
        allvdata = allvdata[:,0]
        sortmap = pylab.argsort(cell_ids, kind='mergesort')
        cell_ids = pylab.take(cell_ids,sortmap)
        allvdata = pylab.take(allvdata,sortmap)
        for i in 0,1:
            tdata = pylab.arange(0,(n+1)*dt,dt)
            vdata = allvdata.compress(cell_ids==i)
            vdata = pylab.where(vdata>=v_thresh-0.05,0.0, vdata) # add fake APs for plotting
            if len(tdata) > len(vdata):
                print "Warning. Shortening tdata from %d to %d elements (%s)" % (len(tdata),len(vdata),simulator)
                tdata = tdata[0:len(vdata)]
            assert len(tdata)==len(vdata), "%d != %d (%s)" % (len(tdata),len(vdata),simulator)
            subplot.plot(tdata,vdata)
        
        # Plot spike rasters
        subplot = figure.add_axes([x,y0+2*dy,w,h])
        exc_spikedata = signals.load_spikelist("Results/VAbenchmark_%s_exc_%s_np%d.ras" % (benchmark, simulator, num_nodes))
        inh_spikedata = signals.load_spikelist("Results/VAbenchmark_%s_inh_%s_np%d.ras" % (benchmark, simulator, num_nodes))
        exc_spikedata.raster_plot(display=subplot)

        # Inter-spike-interval histograms
        bins = pylab.exp(pylab.arange(0, 8, 0.2))
        isihist, bins = exc_spikedata.isi_hist(bins)
        subplot = figure.add_axes([x,y0+dy,0.4*w,h])
        plot_hist(subplot, isihist, pylab.arange(0, 8, 0.2), 0.2,
            xlabel="Inter-spike interval (ms)", xticks=pylab.log([3,10,30,100,1000]),
            xticklabels=['3','10','30','100','1000'], xmin=pylab.log(2), ymax=0.006)
        subplot.set_title('Exc')
        
        isihist, bins = inh_spikedata.isi_hist(bins)
        subplot = figure.add_axes([x+0.45*dx,y0+dy,0.4*w,h])
        plot_hist(subplot, isihist, pylab.arange(0,8,0.2),0.2,
            xlabel="Inter-spike interval (ms)", xticks=pylab.log([3,10,30,100,1000]),
            xticklabels=['3','10','30','100','1000'], xmin=pylab.log(2), ymax=0.006)
        subplot.set_title('Inh')
        
        # Histograms of coefficients of variation of ISI
        bins = pylab.arange(0, 3, 0.1)
        for dataset, xoffset, ymax in zip([exc_spikedata, inh_spikedata], [0.0, 0.45*dx], [2.5,2.5]):
            cvhist, bins = dataset.cv_isi_hist(bins)
        
            #cvhist = nstats.histc(cvs,bins)
            subplot = figure.add_axes([x+xoffset, y0, 0.4*w, h])
            plot_hist(subplot, cvhist, bins, 0.1, xlabel="ISI CV", ymax=ymax)
        
        x += dx

figure.set_canvas(FigureCanvas(figure))
figure.savefig("Results/VAbenchmark_%s.png" % benchmark)

