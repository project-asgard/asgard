#pragma once

#include "asgard_tools.hpp"

namespace asgard
{

/*!
 * \brief Default precision to use, double if enabled and float otherwise.
 */
#ifdef ASGARD_ENABLE_DOUBLE
using default_precision = double;
#else
using default_precision = float;
#endif

/*!
 * \brief Holds general information about the compute resources
 *
 * Singleton class holding meta information about the CPU and GPU resources,
 * number of threads, number of GPUs, allows easy access to BLAS on both
 * CPU and GPU, etc.
 * The main goal of this class is to allow easy use of multiple GPUs handling
 * the corresponding streams and queues, managing memory, and so on.
 */
class compute_resources {
public:
  //! initialize the engine, call once per application
  compute_resources();
private:
};

inline std::optional<compute_resources> compute;

void init_compute() {
  if (not compute)
    compute.emplace();
}

} // namespace asgard
