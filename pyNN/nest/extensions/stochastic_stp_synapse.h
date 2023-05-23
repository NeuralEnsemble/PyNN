/*
 *  stochastic_stp_synapse.h
 *
 *  :copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
 *  :license: CeCILL, see LICENSE for details.
 *
 */

#ifndef STOCHASTIC_STP_SYNAPSE_H
#define STOCHASTIC_STP_SYNAPSE_H

// Includes from nestkernel:
#include "connection.h"

/* BeginUserDocs: synapse, short-term plasticity

Short description
+++++++++++++++++

Probabilistic synapse model with short term plasticity.

Description
+++++++++++

This synapse model implements synaptic short-term depression and
short-term facilitation according to an algorithm developed by
the Blue Brain Project.

The implementation is based on quantal_stp_synapse and the NMODL
file ProbGABAAB_EMS.mod from the Blue Brain Project.

Parameters
++++++++++

The following parameters can be set in the status dictionary:

======= ==== =======================================================
U       real Maximal fraction of available resources [0,1],
                default=0.5
u       real release probability, default=0.5
p       real probability that a vesicle is available, default = 1.0
R       real recovered state {0=unrecovered, 1=recovered}, default=1
tau_rec real time constant for depression in ms, default=800 ms
tau_fac real time constant for facilitation in ms, default=0 (off)
t_surv  real time since last evaluation of survival in ms, default=0
======= ==== =======================================================

Transmits
+++++++++

SpikeEvent

SeeAlso
+++++++

tsodyks2_synapse, synapsedict, quantal_stp_synapse, static_synapse

EndUserDocs */

namespace pynn
{

template < typename targetidentifierT >
class stochastic_stp_synapse : public nest::Connection< targetidentifierT >
{
public:
  typedef nest::CommonSynapseProperties CommonPropertiesType;
  typedef nest::Connection< targetidentifierT > ConnectionBase;

  /**
   * Default Constructor.
   * Sets default values for all parameters. Needed by GenericConnectorModel.
   */
  stochastic_stp_synapse();
  /**
   * Copy constructor to propagate common properties.
   */
  stochastic_stp_synapse( const stochastic_stp_synapse& );

  // Explicitly declare all methods inherited from the dependent base
  // ConnectionBase. This avoids explicit name prefixes in all places these
  // functions are used. Since ConnectionBase depends on the template parameter,
  // they are not automatically found in the base class.
  using ConnectionBase::get_delay_steps;
  using ConnectionBase::get_delay;
  using ConnectionBase::get_rport;
  using ConnectionBase::get_target;

  /**
   * Get all properties of this connection and put them into a dictionary.
   */
  void get_status( DictionaryDatum& d ) const;

  /**
   * Set default properties of this connection from the values given in
   * dictionary.
   */
  void set_status( const DictionaryDatum& d, nest::ConnectorModel& cm );

  /**
   * Send an event to the receiver of this connection.
   * \param e The event to send
   * \param cp Common properties to all synapses (empty).
   */
  void send( nest::Event& e, nest::thread t, const CommonPropertiesType& cp );

  class ConnTestDummyNode : public nest::ConnTestDummyNodeBase
  {
  public:
    // Ensure proper overriding of overloaded virtual functions.
    // Return values from functions are ignored.
    using nest::ConnTestDummyNodeBase::handles_test_event;
    nest::port
    handles_test_event( nest::SpikeEvent&, nest::rport )
    {
      return nest::invalid_port;
    }
  };

  void
  check_connection( nest::Node& s,
    nest::Node& t,
    nest::rport receptor_type,
    const CommonPropertiesType& )
  {
    ConnTestDummyNode dummy_target;
    ConnectionBase::check_connection_( dummy_target, s, t, receptor_type );
  }

  void
  set_weight( double w )
  {
    weight_ = w;
  }

private:
  double weight_;  //!< synaptic weight
  double U_;       //!< unit increment of a facilitating synapse (U)
  double u_;       //!< dynamic value of probability of release
  double tau_rec_; //!< [ms] time constant for recovery from depression (D)
  double tau_fac_; //!< [ms] time constant for facilitation (F)
  double R_;       //!< recovered state {0=unrecovered, 1=recovered}
  double t_surv_;  //!< time since last evaluation of survival
  double t_lastspike_; //!< Time point of last spike emitted
};


/**
 * Send an event to the receiver of this connection.
 * \param e The event to send
 * \param t The thread on which this connection is stored.
 * \param t_lastspike Time point of last spike emitted
 * \param cp Common properties object, containing the stochastic_stp parameters.
 */
template < typename targetidentifierT >
inline void
stochastic_stp_synapse< targetidentifierT >::send( nest::Event& e,
  nest::thread thr,
  const CommonPropertiesType& )
{

  double t_spike = e.get_stamp().get_ms();

  // calculation of u
  if ( tau_fac_ > 1.0e-10 ) {
    u_ *= std::exp( -(t_spike - t_lastspike_) / tau_fac_ );
    u_ += U_ * ( 1 - u_ );
  } else {
    u_ = U_;
  }

  // check for recovery

  bool release = false;
  double p_surv = 0.0;  // survival probability of unrecovered state

  if ( R_ == 0 ) {
    release = false;
    // probability of survival of unrecovered state based on Poisson recovery with rate 1/tau_rec
    p_surv = std::exp( -(t_spike - t_surv_) / tau_rec_ );
    if ( nest::get_vp_specific_rng( thr )->drand() > p_surv ) {
      R_ = 1;                           // recovered
    } else {
      t_surv_ = t_spike; // failed to recover
    }
  }

  // check for release
  if ( R_ == 1 ) {
    if ( nest::get_vp_specific_rng( thr )->drand() < u_ ) {    // release
      release = true;
      R_ = 0;
      t_surv_ = t_spike;
    } else {
      release = false;
    }
  }

  if ( release )
  {
    e.set_receiver( *get_target( thr ) );
    e.set_weight( weight_ );
    e.set_delay_steps( get_delay_steps() );
    e.set_rport( get_rport() );
    e();
  }

  t_lastspike_ = t_spike;
}

} // namespace

#endif // STOCHASTIC_STP_SYNAPSE_H
