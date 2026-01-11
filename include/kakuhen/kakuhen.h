/*!
 * @file kakuhen.h
 * @brief Main umbrella header for the kakuhen numerical integration library.
 *
 * This header includes all public-facing components of the kakuhen library,
 * making it convenient for users to include all necessary definitions for
 * utilizing the integrators (Basin, Plain, Vegas) and their associated
 * functionalities.
 */
#pragma once

#include "kakuhen/integrator/basin.h"
#include "kakuhen/integrator/integrator_base.h"
#include "kakuhen/integrator/plain.h"
#include "kakuhen/integrator/vegas.h"

namespace kakuhen {}  // namespace kakuhen
