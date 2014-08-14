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

We should use the PyNN Timer object to separate timings for different phases, i.e.

  * module imports
  * setup
  * network construction
    - creating neurons
    - creating connections
  * recording specification
  * simulation
  * data retrieval

We should start with a fresh kernel every time, except where we specifically test the effect of reset.
Each simulation should write to a common database, from which we can generate reports. Ideally, we would
use Sumatra or Mozaik for this, but it might be simpler to start with, e.g., shelve or csv

Ideas for benchmarks:

  * a simple network with two connected populations, varying:
    - number of neurons
    - number of connections
    - neuron type (in particular, spike sources are treated very differently in NEST to other neuron models)
    - synapse type
    - connection method
    - number of neurons recorded
  * incremental simulation, with and without clearing recorders.
