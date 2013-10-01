# coding: utf-8
"""
A simple network model for benchmarking purposes.


Usage: python simple_network.py [-h] parameter_file data_store

positional arguments:
  parameter_file  Parameter file (for format see
                  http://parameters.readthedocs.org/)
  data_store      filename for output data file

optional arguments:
  -h, --help      show this help message and exit
"""

from importlib import import_module
from parameters import ParameterSet
from pyNN.utility import Timer


def main_pyNN(parameters):
    P = parameters
    timer = Timer()
    sim = import_module(P.simulator)
    timer.mark("import")

    sim.setup()
    timer.mark("setup")

    p = sim.Population(P.n, sim.IF_cond_exp(i_offset=1.0))
    timer.mark("build")

    if P.recording:
        for var_name, n_record in P.recording:
            p.sample(n_record).record(var_name)
        timer.mark("record")

    sim.run(P.sim_time)
    timer.mark("run")

    if P.recording:
        block = p.get_data()  # perhaps include some summary statistics in the data returned?
        timer.mark("get_data")

    mpi_rank = sim.rank()
    num_processes = sim.num_processes()
    sim.end()
    
    data = dict(timer.marks)
    data.update(num_processes=num_processes)
    return mpi_rank, data


def main_pynest(parameters):
    P = parameters
    assert P.sim_name == "pynest"
    timer = Timer()
    import nest
    timer.mark("import")

    nest.SetKernelStatus({"resolution": 0.1})
    timer.mark("setup")

    p = nest.Create("iaf_neuron", n=P.n, params={"I_e": 1000.0})
    timer.mark("build")

    # todo: add recording and data retrieval
    nest.Simulate(P.sim_time)
    timer.mark("run")

    mpi_rank = nest.Rank()
    num_processes = nest.NumProcesses()
    
    data = P.as_dict()
    data.update(num_processes=num_processes,
                timings=timer.marks)
    return mpi_rank, data


if __name__ == "__main__":
    from datetime import datetime
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("parameter_file", help="Parameter file (for format see http://parameters.readthedocs.org/)")
    parser.add_argument("data_store", help="filename for output data file")
    args = parser.parse_args()
    
    parameters = ParameterSet(args.parameter_file)
    
    print parameters.pretty()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if parameters.simulator == "pynest":
        main = main_pynest
    else:
        main = main_pyNN
    mpi_rank, data = main(parameters)
    
    if mpi_rank == 0:
        #import shelve
        #shelf = shelve.open(args.data_store)
        #shelf[timestamp] = data
        #shelf.close()
        import os, csv
        fieldnames = ["timestamp"] + parameters.keys() + data.keys()
        data.update(timestamp=timestamp)
        data.update(parameters.as_dict())
        write_header = True
        if os.path.exists(args.data_store):
            write_header = False
        with open(args.data_store, "a+b") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames, quoting=csv.QUOTE_NONNUMERIC)
            if write_header:
                writer.writeheader()
            writer.writerow(data)

