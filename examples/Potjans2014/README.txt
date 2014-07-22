/*
 *  README.txt
 *
 */

Cortical microcircuit simulation: PyNN version

This is an implementation of the multi-layer microcircuit model of early
sensory cortex published by Potjans and Diesmann (2014) The cell-type specific
cortical microcircuit: relating structure and activity in a full-scale spiking
network model. Cerebral Cortex 24 (3): 785-806, doi:10.1093/cercor/bhs358.

It has only been tested with the NEST back-end.

Files:
	- network_params.py
	Script containing model parameters

        - sim_params.py
        Script containing simulation and system parameters

	- microcircuit.py
	Simulation script - can be left unchanged

        - network.py
        In which the network is set up

        - connectivity.py
        Definition of connection function

        - scaling.py
        Functions for computing numbers of synapses in full-scale and down-scaled networks

	- run_microcircuit.py
	Creates output directory, copies all scripts to this directory, 
        creates sim_script.sh and submits it to the queue
        Takes all parameters from sim_params.sli and can be left unchanged

	- plotting.py
	Python script to create raster and firing rate plot


Instructions:

1. Download and install your desired back-end. 
   For NEST, see http://www.nest-initiative.org/index.php/Software:Download
   and to enable full-scale simulation, compile it with MPI support 
   (use the --with-mpi option when configuring) according to the instructions on
   http://www.nest-initiative.org/index.php/Software:Installation

2. Install PyNN 0.8 according to the instructions on 
   http://neuralensemble.org/docs/PyNN/installation.html

4. In sim_params.py adjust the following parameters:

   - Set the simulation time via 'sim_duration'
   - the number of compute nodes 'n_nodes'
   - the number of processes per node 'n_procs_per_node'
   - queuing system parameters 'walltime' and 'memory'
   - Adjust 'output_path', 'mpi_path', 'nest_path', and 'pyNN_path' to your system

5. In network_params.py:

   - Add dictionary to params_dict for the back-end you wish to use
   - Choose the network size via 'N_scaling' and 'K_scaling', 
     which scales the numbers of neurons and in-degrees, respectively
   - Choose the external input via 'input_type'
   - Optionally activate thalamic input via 'thalamic_input' 
     and set any thalamic input parameters 
   
6. Run the simulation by typing 'python run_microcircuit.py' in your terminal
   (microcircuit.py and the parameter files need to be in the same folder)

7. Output files and basic analysis:
   
   - Spikes are written to .txt files containing IDs of the recorded neurons
     and corresponding spike times in ms.
     Separate files are written out for each population and virtual process.
     File names are formed as 'spikes'+ layer + population + MPI process + .txt
   - Voltages are written to .dat files containing GIDs, times in ms, and the
     corresponding membrane potentials in mV. File names are formed as
     voltmeter label + layer index + population index + spike detector GID +
     virtual process + .dat

   - If 'create_raster_plot' is set to True, a raster plot is saved as 'result.png'
    

The simulation was successfully tested with NEST 2.4.1 and MPI 1.4.3.
Plotting works with Python 2.6.6 including packages numpy 1.3.0,
matplotlib 0.99.1.1, and glob.

---------------------------------------------------

Simulation on a single process:

1. Go to the folder that includes microcircuit.py and the parameter files

2. Adjust 'N_scaling' and 'K_scaling' in network_params.py such that the network
   is small enough to fit on your system 

3. Ensure that the output directory exists, as it is not created via the bash
   script anymore

4. Type 'python microcircuit.py' to start the simulation on a single process


