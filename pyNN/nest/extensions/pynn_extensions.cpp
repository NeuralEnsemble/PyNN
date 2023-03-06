/*

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

*/

#include "pynn_extensions.h"

// Generated includes:
#include "config.h"

// include headers with your own stuff
#include "simple_stochastic_synapse.h"
#include "stochastic_stp_synapse.h"
#include "stochastic_stp_synapse_impl.h"

// Includes from nestkernel:
#include "connection_manager_impl.h"
#include "connector_model_impl.h"
#include "dynamicloader.h"
#include "exceptions.h"
#include "genericmodel.h"
#include "kernel_manager.h"
#include "model.h"
#include "model_manager_impl.h"
#include "nest.h"
#include "nest_impl.h"
#include "nestmodule.h"
#include "target_identifier.h"

// Includes from sli:
#include "booldatum.h"
#include "integerdatum.h"
#include "sliexceptions.h"
#include "tokenarray.h"

// -- Interface to dynamic module loader ---------------------------------------

/*
 * There are three scenarios, in which PyNNExtensions can be loaded by NEST:
 *
 * 1) When loading your module with `Install`, the dynamic module loader must
 * be able to find your module. You make the module known to the loader by
 * defining an instance of your module class in global scope. (LTX_MODULE is
 * defined) This instance must have the name
 *
 * <modulename>_LTX_mod
 *
 * The dynamicloader can then load modulename and search for symbol "mod" in it.
 *
 * 2) When you link the library dynamically with NEST during compilation, a new
 * object has to be created. In the constructor the DynamicLoaderModule will
 * register your module. (LINKED_MODULE is defined)
 *
 * 3) When you link the library statically with NEST during compilation, the
 * registration will take place in the file `static_modules.h`, which is
 * generated by cmake.
 */
#if defined( LTX_MODULE ) | defined( LINKED_MODULE )
pynn::PyNNExtensions pynn_extensions_LTX_mod;
#endif
// -- DynModule functions ------------------------------------------------------

pynn::PyNNExtensions::PyNNExtensions()
{
#ifdef LINKED_MODULE
  // register this module at the dynamic loader
  // this is needed to allow for linking in this module at compile time
  // all registered modules will be initialized by the main app's dynamic loader
  nest::DynamicLoaderModule::registerLinkedModule( this );
#endif
}

pynn::PyNNExtensions::~PyNNExtensions()
{
}

const std::string
pynn::PyNNExtensions::name( void ) const
{
  return std::string( "PyNN extensions for NEST" ); // Return name of the module
}

const std::string
pynn::PyNNExtensions::commandstring( void ) const
{
  // Instruct the interpreter to load pynn_extensions-init.sli
  return std::string( "(pynn_extensions-init) run" );
}

//-------------------------------------------------------------------------------------

void
pynn::PyNNExtensions::init( SLIInterpreter* i )
{
  /* Register a neuron or device model.
     Give node type as template argument and the name as second argument.
  */
  /* nest::kernel().model_manager.register_node_model< pif_psc_alpha >(
    "pif_psc_alpha" );
  */

  /* Register a synapse type.
     Give synapse type as template argument and the name as second argument.

     There are two choices for the template argument:
         - nest::TargetIdentifierPtrRport
         - nest::TargetIdentifierIndex
     The first is the standard and you should usually stick to it.
     nest::TargetIdentifierIndex reduces the memory requirement of synapses
     even further, but limits the number of available rports. Please see
     Kunkel et al, Front Neurofinfom 8:78 (2014), Sec 3.3.2, for details.
  */
  nest::register_connection_model< simple_stochastic_synapse >( "simple_stochastic_synapse" );
  nest::register_connection_model< stochastic_stp_synapse >( "stochastic_stp_synapse" );

} // PyNNExtensions::init()
