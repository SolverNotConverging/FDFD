#pragma once

#include "model.hpp"

namespace tl {

// Return the audited calculator defaults for one geometry.  Parameters keeps
// Microstrip defaults so it remains convenient as a value type; callers that
// switch line types should use this function.
[[nodiscard]] Parameters defaultParameters(LineType type);

// Build a dedicated conforming scalar P1 mesh, solve the dielectric and
// vacuum electrostatic problems, and extract the forward quasi-TEM RLGC mode.
// Invalid user input throws std::invalid_argument; numerical failures throw
// std::runtime_error.
[[nodiscard]] Result solve(const Parameters& parameters);

}  // namespace tl
