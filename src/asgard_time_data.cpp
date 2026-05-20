#include "asgard_time_data.hpp"

namespace asgard {

time_data make_time_data(prog_opts const &options)
{
  time_method sm = options.step_method.value_or(time_method::rk2);

  time_data dtime; // initialize below

  double stop = options.stop_time.value_or(-1);
  double dt   = options.dt.value_or(-1);
  int64_t n   = options.num_time_steps.value_or(-1);

  if (sm == time_method::steady) {
    stop  = options.stop_time.value_or(options.default_stop_time.value_or(0));
    return time_data(stop);
  } else {
    rassert(not (stop >= 0 and dt >= 0 and n >= 0),
      "Must provide exactly two of the three time-stepping parameters: -dt, -num-steps, -time");

    // replace options with defaults, when appropriate
    if (n == 0 or stop == 0) { // initial conditions only, no time stepping
      return time_data(sm, time_data::input_dt{0}, 0);
    } else if (n > 0) {
      if (stop < 0 and dt < 0) {
        dt = options.default_dt.value_or(-1);
        if (dt < 0) {
          stop = options.default_stop_time.value_or(-1);
          rassert(stop >= 0, "number of steps provided, but no dt or stop-time");
        }
      }
    } else if (stop >= 0) { // no num-steps, but dt may be provided or have a default
      if (dt < 0) {
        dt = options.default_dt.value_or(-1);
        rassert(dt >= 0, "stop-time provided but no time-step or number of steps");
      }
    } else if (dt >= 0) { // both n and stop are unspecified
      stop = options.default_stop_time.value_or(-1);
      rassert(stop >= 0, "dt provided, but no stop-time or number of steps");
    } else { // nothing provided, look for defaults
      dt   = options.default_dt.value_or(-1);
      stop = options.default_stop_time.value_or(-1);
      rassert(dt >= 0 and stop >= 0, "need at least two time parameters: -dt, -num-steps, -time");
    }

    if (n >= 0 and stop >= 0 and dt < 0)
      return time_data(sm, n, time_data::input_stop_time{stop});
    else if (dt >= 0 and stop >= 0 and n < 0)
      return time_data(sm, time_data::input_dt{dt}, time_data::input_stop_time{stop});
    else if (dt >= 0 and n >= 0 and stop < 0)
      return time_data(sm, time_data::input_dt{dt}, n);
    else
      throw std::runtime_error("how did this happen?");
  }
}

} // namespace asgard
