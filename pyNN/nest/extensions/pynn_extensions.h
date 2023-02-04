/*

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

*/

#ifndef PYNNEXTENSIONS_H
#define PYNNEXTENSIONS_H

// Includes from sli:
#include "slifunction.h"
#include "slimodule.h"

// Put your stuff into your own namespace.
namespace pynn
{

/**
 * Class defining your model.
 * @note For each model, you must define one such class, with a unique name.
 */
class PyNNExtensions : public SLIModule
{
public:
  // Interface functions ------------------------------------------

  /**
   * @note The constructor registers the module with the dynamic loader.
   *       Initialization proper is performed by the init() method.
   */
  PyNNExtensions();

  /**
   * @note The destructor does not do much in modules.
   */
  ~PyNNExtensions();

  /**
   * Initialize module.
   * @param SLIInterpreter* SLI interpreter
   */
  void init( SLIInterpreter* );

  /**
   * Return the name of your model.
   */
  const std::string name( void ) const;

  /**
   * Return the name of a sli file to execute when the module is loaded.
   * This mechanism can be used to define SLI commands associated with your
   * module, in particular, set up type tries for functions you have defined.
   */
  const std::string commandstring( void ) const;
};
} // namespace pynn

#endif
