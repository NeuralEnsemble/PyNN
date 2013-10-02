=====================
Benchmarking strategy
=====================

The aim of benchmarking is to provide measurements that

1. allow us to determine the effect of code changes on performance;
2. tell us how much overhead PyNN has compared to the underlying simulators;
3. show how well PyNN scales with the number of MPI processes.

Point 2 means that where possible, we should provide both a PyNN version and
a PyNEST version of benchmarks. PyNEURON, Brian and NEST-SLI versions would also
be interesting.

We should use the PyNN Timer object to separate timings for different phases, i.e. network
construction, recording specification, simulation, data retrieval.

We should start with a fresh kernel every time, except where we specifically test the effect of reset.
Each simulation should write to a common database, from which we can generate reports. Ideally, we would
use Sumatra for this, but it might be simpler to start with, e.g., shelve.

Ideas for benchmarks:

* neurons only (construction and simulation), as a function of network size. No recording. Three different neuron models.
* entire network, as a function of number of neurons and number of synapses. No recording. Three different synapse types. All connection methods.
* neurons only, with recording, as a function of number of neurons recorded (spikes, vm, gysn). Also time data retrieval.
* incremental simulation, with and without clearing recorders.


For at least the first three of these, we could probably make it a single script,
with separate parameter files (the same parameter files could be the inputs
to the plotting script).