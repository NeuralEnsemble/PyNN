""" Master script for running tests.
$Id$
"""

import subprocess, sys, glob, os, re
import numpy as N
import logging
import sys
try:
    import pylab
    plotting = True
except ImportError:
    plotting = False

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='runtests.log',
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger('').addHandler(console)

nrnheader = re.compile(r'(?P<header>NEURON.*Additional mechanisms from files(\s+\w+.mod)+\s+)(?P<body>.*)', re.S)

def run(cmd,engine):
    #print 'Running "', cmd, '" with', engine.upper()
    fail = False
    logfile = open("Results/%s_%s.log" % (cmd,engine), 'w')
    if engine in ('nest1', 'pcsim', 'nest2', 'neuron', 'oldneuron', 'brian'):
        cmd = sys.executable + ' ' + cmd + '.py ' + engine
    else:
        logging.error('Invalid simulation engine "%s". Valid values are "nest1", "nest2", "pcsim", "neuron", "oldneuron", "brian"' % engine)
        
    p = subprocess.Popen(cmd, shell=True, stdout=logfile, stderr=subprocess.PIPE, close_fds=True)
    p.wait()
    logfile.close()
    errorMsg = p.stderr.read()
    #errorMsg = errorMsg.strip(nrnheader)
    match = nrnheader.match(errorMsg)
    if match:
        errorMsg = match.groupdict()['body']
    if len(errorMsg) > 0:
        logging.error("\n=== %s Error =======================" % engine.upper())
        logging.error("  " + errorMsg.replace("\n","\n   "))
        logging.error("=======================================")
        raise Exception()

def sortTracesByCells(traces, gids):
    # First, we see what are the gid present in the recorded file:
    cells_id = []
    for gid in gids:
        if not gid in cells_id:
            cells_id.append(gid)
    # Then we return a list of tuples containing the Vm traces sorted
    # by the cell_ids:
    cells_id = N.sort(cells_id)
    result = N.array([])
    for id in cells_id:
        idx = N.where(gids == id)
        result = N.concatenate((result, traces[idx]))
    return result


def compare_traces(script, mse_threshold, engines):
    """For scripts that write a voltage trace to file."""
    
    #print script, ": ",
    traces = {}
    fail = False
    fail_message = ""
    
    if plotting:
        fig = pylab.figure()
        pylab.title(script)
    
    for engine in engines:
        logging.debug("Running %s with %s" % (script, engine))
        traces[engine] = []
        try:
            run(script, engine)
            pattern = 'Results/%s_*_%s.v' % (script, engine)
            filenames = glob.glob(pattern)
            if len(filenames) == 0:
                pattern = 'Results/%s_%s.v' % (script, engine)
                filenames = glob.glob(pattern)
            if filenames:
                for filename in filenames:
                    f = open(filename,'r')
                    trace = [line.strip() for line in f.readlines() if line[0] != "#"]
                    f.close()
                    trace = [line for line in trace if line] # ignore blank lines
                    try:
                        position = N.array([int(line.split()[1]) for line in trace]) # take second column
                    except IndexError:
                        print engine
                        print line
                        raise
                    trace = N.array([float(line.split()[0]) for line in trace]) # take first column 
                    trace = sortTracesByCells(trace, position)
                    traces[engine].append(trace)
                    if plotting:
                        pylab.plot(trace, label=engine)
            else:
                fail = True; fail_message += "No files match glob pattern. %s" % pattern
        except Exception:
            fail = True
            fail_message += "Exception raised in %s. " % engine.upper()

    if plotting:
        pylab.legend()
        pylab.savefig("Results/%s.png" % script)

    if not fail:
        mse = 0.0
        engine1 = engines[0] # compare all against the first engine in the list.
        for engine2 in engines[1:]:
            for trace1,trace2 in zip(traces[engine1],traces[engine2]):
                l1 = len(trace1); l2 = len(trace2)
                if l1 > 0 and l2 > 0 :
                    if l1 > l2:
                        trace1 = trace1[:l2]
                    elif l2 > l1:
                        trace2 = trace2[:l1]
                    
                    diff = trace1 - trace2
                    mse += N.sqrt(N.mean(N.square(diff)))
                else:
                    fail = True;
                    if (l1 == 0):
                        fail_message = "%s produce an empty file for %s" %(engine1.upper(), script)
                    if (l2 == 0):
                        fail_message = "%s produce an empty file for %s" %(engine2.upper(), script)
    if not fail:
        if mse < mse_threshold:
            logging.info("%s: Pass (mse = %f, threshold = %f)" % (script, mse, mse_threshold))
        else:
            logging.info("%s:  Fail (mse = %f, threshold = %f)" % (script, mse, mse_threshold))
    else:
        logging.info("%s: Fail - %s" % (script, fail_message))
    return fail

def compare_rasters(script,mse_threshold,engines):
    """For scripts that write a voltage trace to file."""
    
    #print script, ": ",
    rasters = {}
    fail = False
    fail_message = ""
    
    for engine in engines:
        rasters[engine] = []
        try:
        #if (True):
            run(script, engine)
            pattern = 'Results/%s_*_%s.ras' % (script, engine)
            filenames = glob.glob(pattern)
            if len(filenames) == 0:
                pattern = 'Results/%s_%s.ras' % (script, engine)
                filenames = glob.glob(pattern)
            if filenames:
                for filename in filenames:
                    f = open(filename,'r')
                    raster = [line.strip() for line in f.readlines() if line[0] != "#"]
                    f.close()
                    raster = [line for line in raster if line] # ignore blank lines
                    try:
                        position = N.array([int(line.split()[1]) for line in raster]) # take second column
                    except IndexError:
                        print engine
                        print line
                        raise
                    raster = N.array([float(line.split()[0]) for line in raster]) # take first column 
                    raster = sortTracesByCells(raster, position)
                    rasters[engine].append(raster)
            else:
                fail = True; fail_message += "No files match glob pattern %s. " % pattern
        except Exception:
            fail = True
            fail_message += "Exception raised in %s. " % engine.upper()

    if not fail:
        mse  = 0.0
        engine1 = engines[0] # compare all against the first engine in the list.
        for engine2 in engines[1:]:
            for raster1,raster2 in zip(rasters[engine1],rasters[engine2]):
                l1 = len(raster1); l2 = len(raster2)
                if l1 > 0 and l2 > 0 :
                    diff = []
                    for idx in xrange(len(raster1)):
                        diff.append(N.abs(raster1[idx]-raster2).min())
                    mse += N.sqrt(N.mean(N.square(N.array(diff))))
                else:
                    fail = True;
                    if (l1 == 0):
                        fail_message = "%s produce an empty file for %s" %(engine1.upper(), script)
                    if (l2 == 0):
                        fail_message = "%s produce an empty file for %s" %(engine2.upper(), script)

    if not fail:
        if mse < mse_threshold:
            logging.info("%s: Pass (mse = %f, threshold = %f)" % (script, mse, mse_threshold))
        else:
            logging.info("%s:  Fail (mse = %f, threshold = %f)" % (script, mse, mse_threshold))
    else:
        logging.info("%s: Fail - %s" % (script, fail_message))
        
if __name__ == "__main__":
    
    engine_list = ["nest1", "neuron", "pcsim", "nest2", "oldneuron", "brian"]
    for engine in engine_list:
        try:
            exec("import pyNN.%s" % engine)
        except ImportError, errmsg:
            engine_list.remove(engine)
            logging.warning("Unable to use %s: %s" % (engine, errmsg))
    if len(engine_list) < 2:
        logging.error("Need at least two simulators to run tests.")
        sys.exit(1)
    logging.debug("Using the following simulators: %s" % ", ".join(engine_list))
            
    thresholds_v = {"IF_curr_alpha"  : 0.26,
                    "IF_curr_exp"    : 0.25, 
                    "IF_cond_alpha"  : 0.25,
                    "IF_cond_exp"    : 0.25,
                    "simpleNetworkL" : 0.5,
                    "simpleNetwork"  : 0.7,
                    "IF_curr_alpha2" : 5.0,
                    "small_network"  : 5.0,
                    "IF_curr_exp2"   : 0.6}
    
    thresholds_ras = {"SpikeSourcePoisson" : 50.,
                      "IF_curr_exp2": 0.25,}
    
    scripts_v   = []
    scripts_ras = []
    
    if len(sys.argv) > 1:
        for item in sys.argv[1:]:
            if item in thresholds_v.keys():
                scripts_v.append(item)
            if item in thresholds_ras.keys():
                scripts_ras.append(item)
    else:
        scripts_v   = thresholds_v.keys()
        scripts_ras = thresholds_ras.keys()

    logging.debug("Running the following tests: %s,%s using engines %s" % (scripts_v, scripts_ras,engine_list))
    
    for script in scripts_v:
        compare_traces(script, thresholds_v[script]*(len(engine_list)-1), engine_list)

    for script in scripts_ras:
        compare_rasters(script, thresholds_ras[script]*(len(engine_list)-1), engine_list)
    
    if len(scripts_v)+len(scripts_ras) == 0:
        logging.error("Invalid test name(s).")