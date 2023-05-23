/*
 *  :copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
 *  :license: CeCILL, see LICENSE for details.
 *
 */

#ifndef SIMPLE_STOCHASTIC_SYNAPSE_H
#define SIMPLE_STOCHASTIC_SYNAPSE_H

// Includes from nestkernel:
#include "connection.h"


/* BeginUserDocs: synapse, short-term plasticity

Short description
+++++++++++++++++

Synapse dropping spikes stochastically.

Description
+++++++++++

This synapse will deliver spikes with probability p.

Parameters
++++++++++

= ==== =========================================================================================
p real probability that a spike is transmitted, default = 1.0 (i.e. spike is always transmitted)
= ==== =========================================================================================

Transmits
+++++++++

SpikeEvent

SeeAlso
+++++++

static_synapse, synapsedict
EndUserDocs */

namespace pynn
{

template < typename targetidentifierT >
class simple_stochastic_synapse : public nest::Connection< targetidentifierT >
{
private:
  double weight_; //!< Synaptic weight
  double p_;      //!< Probability of spike transmission

public:
  //! Type to use for representing common synapse properties
  typedef nest::CommonSynapseProperties CommonPropertiesType;

  //! Shortcut for base class
  typedef nest::Connection< targetidentifierT > ConnectionBase;

  /**
   * Default Constructor.
   * Sets default values for all parameters. Needed by GenericConnectorModel.
   */
  simple_stochastic_synapse()
    : ConnectionBase()
    , weight_( 1.0 )
    , p_( 1.0 )
  {
  }

  //! Default Destructor.
  ~simple_stochastic_synapse()
  {
  }

  /**
   * Helper class defining which types of events can be transmitted.
   *
   * These methods are only used to test whether a certain type of connection
   * can be created.
   *
   * `handles_test_event()` should be added for all event types that the
   * synapse can transmit. The methods shall return `invalid_port`; the
   * return value will be ignored.
   *
   * Since this is a synapse model dropping spikes, it is only for spikes,
   * therefore we only implement `handles_test_event()` only for spike
   * events.
   *
   * See Kunkel et al (2014), Sec 3.3.1, for background information.
   */
  class ConnTestDummyNode : public nest::ConnTestDummyNodeBase
  {
  public:
    using nest::ConnTestDummyNodeBase::handles_test_event;
    nest::port
    handles_test_event( nest::SpikeEvent&, nest::rport )
    {
      return nest::invalid_port;
    }

    nest::port
    handles_test_event( nest::DSSpikeEvent&, nest::rport )
    {
      return nest::invalid_port;
    }
  };

  /**
   * Check that requested connection can be created.
   *
   * This function is a boilerplate function that should be included unchanged
   * in all synapse models. It is called before a connection is added to check
   * that the connection is legal. It is a wrapper that allows us to call
   * the "real" `check_connection_()` method with the `ConnTestDummyNode
   * dummy_target;` class for this connection type. This avoids a virtual
   * function call for better performance.
   *
   * @param s  Source node for connection
   * @param t  Target node for connection
   * @param receptor_type  Receptor type for connection
   * @param lastspike Time of most recent spike of presynaptic (sender) neuron,
   *                  not used here
   */
  void
  check_connection( nest::Node& s,
    nest::Node& t,
    nest::rport receptor_type,
    const CommonPropertiesType& )
  {
    ConnTestDummyNode dummy_target;
    ConnectionBase::check_connection_( dummy_target, s, t, receptor_type );
  }

  /**
   * Send an event to the receiver of this connection.
   * @param e The event to send
   * @param t Thread
   * @param cp Common properties to all synapses.
   */
  void send( nest::Event& e, nest::thread t, const CommonPropertiesType& cp );

  // The following methods contain mostly fixed code to forward the
  // corresponding tasks to corresponding methods in the base class and the w_
  // data member holding the weight.

  //! Store connection status information in dictionary
  void get_status( DictionaryDatum& d ) const;

  /**
   * Set connection status.
   *
   * @param d Dictionary with new parameter values
   * @param cm ConnectorModel is passed along to validate new delay values
   */
  void set_status( const DictionaryDatum& d, nest::ConnectorModel& cm );

  //! Allows efficient initialization on construction
  void
  set_weight( double w )
  {
    weight_ = w;
  }
};


template < typename targetidentifierT >
inline void
simple_stochastic_synapse< targetidentifierT >::send( nest::Event& e,
  nest::thread t,
  const CommonPropertiesType& props )
{
  if ( nest::get_vp_specific_rng( t )->drand() < (1 - p_) )  // drop spike
    return;

  // Even time stamp, we send the spike using the normal sending mechanism
  // send the spike to the target
  e.set_weight( weight_ );
  e.set_delay_steps( ConnectionBase::get_delay_steps() );
  e.set_receiver( *ConnectionBase::get_target( t ) );
  e.set_rport( ConnectionBase::get_rport() );
  e(); // this sends the event
}

template < typename targetidentifierT >
void
simple_stochastic_synapse< targetidentifierT >::get_status(
  DictionaryDatum& d ) const
{
  ConnectionBase::get_status( d );
  def< double >( d, nest::names::weight, weight_ );
  def< double >( d, nest::names::p, p_ );
  def< long >( d, nest::names::size_of, sizeof( *this ) );
}

template < typename targetidentifierT >
void
simple_stochastic_synapse< targetidentifierT >::set_status(
  const DictionaryDatum& d,
  nest::ConnectorModel& cm )
{
  ConnectionBase::set_status( d, cm );
  updateValue< double >( d, nest::names::weight, weight_ );
  updateValue< double >( d, nest::names::p, p_ );
}

} // namespace

#endif // simple_stochastic_synapse.h
