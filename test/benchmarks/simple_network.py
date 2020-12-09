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
    timer = Timer()
    sim = import_module(parameters.simulator)
    timer.mark("import")

    sim.setup(threads=parameters.threads)
    timer.mark("setup")

    populations = {}
    for name, P in parameters.populations.parameters():
        populations[name] = sim.Population(P.n, getattr(sim, P.celltype)(**P.params), label=name)
    timer.mark("build")

    if parameters.projections:
        projections = {}
        for name, P in parameters.projections.parameters():
            connector = getattr(sim, P.connector.type)(**P.connector.params)
            synapse_type = getattr(sim, P.synapse_type.type)(**P.synapse_type.params)
            projections[name] = sim.Projection(populations[P.pre],
                                               populations[P.post],
                                               connector,
                                               synapse_type,
                                               receptor_type=P.receptor_type,
                                               label=name)
        timer.mark("connect")

    if parameters.recording:
        for pop_name, to_record in parameters.recording.parameters():
            for var_name, n_record in to_record.items():
                populations[pop_name].sample(n_record).record(var_name)
        timer.mark("record")

    sim.run(parameters.sim_time)
    timer.mark("run")

    spike_counts = {}
    if parameters.recording:
        for pop_name in parameters.recording.names():
            block = populations[pop_name].get_data()  # perhaps include some summary statistics in the data returned?
            spike_counts["spikes_%s" % pop_name] = populations[pop_name].mean_spike_count()
        timer.mark("get_data")

    mpi_rank = sim.rank()
    num_processes = sim.num_processes()
    sim.end()
    
    data = dict(timer.marks)
    data.update(num_processes=num_processes)
    data.update(spike_counts)
    return mpi_rank, data


def main_pynest(parameters):
    P = parameters
    assert P.sim_name == "pynest"
    timer = Timer()
    import nest
    timer.mark("import")

    nest.SetKernelStatus({"resolution": 0.1})
    timer.mark("setup")

    p = nest.Create("iaf_psc_alpha", n=P.n, params={"I_e": 1000.0})
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
    
    #print(parameters.pretty())
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
        parameters_flat = parameters.flatten()
        fieldnames = ["timestamp"] + parameters_flat.keys() + data.keys()
        data.update(timestamp=timestamp)
        data.update(parameters_flat)
        write_header = True
        if os.path.exists(args.data_store):
            write_header = False
        with open(args.data_store, "a+b") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames, quoting=csv.QUOTE_NONNUMERIC)
            if write_header:
                writer.writeheader()
            writer.writerow(data)

