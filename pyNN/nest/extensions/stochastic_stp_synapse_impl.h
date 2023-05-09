/*
 *  stochastic_stp_synapse_impl.h
 *
 *  :copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
 *  :license: CeCILL, see LICENSE for details.
 *
 */

#ifndef STOCHASTIC_STP_SYNAPSE_IMPL_H
#define STOCHASTIC_STP_SYNAPSE_IMPL_H

#include "stochastic_stp_synapse.h"

// Includes from nestkernel:
#include "connection.h"
#include "connector_model.h"
#include "nest_names.h"

// Includes from sli:
#include "dictutils.h"

namespace pynn
{

template < typename targetidentifierT >
stochastic_stp_synapse< targetidentifierT >::stochastic_stp_synapse()
  : ConnectionBase()
  , weight_( 1.0 )
  , U_( 0.5 )
  , u_( 0.0 )
  , tau_rec_( 800.0 )
  , tau_fac_( 10.0 )
  , R_( 1.0 )
  , t_surv_( 0.0 )
  , t_lastspike_( 0.0 )
{
}

template < typename targetidentifierT >
stochastic_stp_synapse< targetidentifierT >::stochastic_stp_synapse(
  const stochastic_stp_synapse& rhs )
  : ConnectionBase( rhs )
  , weight_( rhs.weight_ )
  , U_( rhs.U_ )
  , u_( rhs.u_ )
  , tau_rec_( rhs.tau_rec_ )
  , tau_fac_( rhs.tau_fac_ )
  , R_( rhs.R_ )
  , t_surv_( rhs.t_surv_ )
  , t_lastspike_( rhs.t_lastspike_ )
{
}


template < typename targetidentifierT >
void
stochastic_stp_synapse< targetidentifierT >::get_status(
  DictionaryDatum& d ) const
{
  ConnectionBase::get_status( d );
  def< double >( d, nest::names::weight, weight_ );
  def< double >( d, nest::names::dU, U_ );
  def< double >( d, nest::names::u, u_ );
  def< double >( d, nest::names::tau_rec, tau_rec_ );
  def< double >( d, nest::names::tau_fac, tau_fac_ );
}


template < typename targetidentifierT >
void
stochastic_stp_synapse< targetidentifierT >::set_status(
  const DictionaryDatum& d,
  nest::ConnectorModel& cm )
{
  ConnectionBase::set_status( d, cm );
  updateValue< double >( d, nest::names::weight, weight_ );

  updateValue< double >( d, nest::names::dU, U_ );
  updateValue< double >( d, nest::names::u, u_ );
  updateValue< double >( d, nest::names::tau_rec, tau_rec_ );
  updateValue< double >( d, nest::names::tau_fac, tau_fac_ );
}

} // of namespace pynn

#endif // #ifndef STOCHASTIC_STP_SYNAPSE_IMPL_H
