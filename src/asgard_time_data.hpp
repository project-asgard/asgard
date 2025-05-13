#pragma once
#include "asgard_reconstruct.hpp"
#include "asgard_transformations.hpp"
#include "asgard_moment.hpp"
#include "asgard_solver.hpp"

/*!
 * \internal
 * \file asgard_time_data.hpp
 * \brief Defines the data common for all time-stepping methods
 * \author The ASGarD Team
 * \ingroup asgard_discretization
 *
 * \endinternal
 */

namespace asgard {

/*!
 * \internal
 * \ingroup asgard_discretization
 * \brief Holds initial time, final time, time-step, etc.
 *
 * When constructed, it takes 2 of 3 parameters, stop-time, time-step and number
 * of time steps. Then sets the correct third parameter.
 *
 * When remaining steps hits 0, current_time is equal to final_time,
 * give or take some machine precision.
 *
 * \endinternal
 */
class time_data
{
public:
  //! type-tag for specifying  dt
  struct input_dt {
    //! explicit constructor, temporarily stores dt
    explicit input_dt(double v) : value(v) {}
    //! stored value
    double value;
  };
  //! type-tag for specifying stop-time
  struct input_stop_time {
    //! explicit constructor, temporarily stores the stop-time
    explicit input_stop_time(double v) : value(v) {}
    //! stored value
    double value;
  };
  //! unset time-data, all entries are negative, must be set later
  time_data() = default;
  //! no time-stepping, set the method but the steps are set to zero
  time_data(time_method smethod)
      : smethod_(smethod), stop_time_(0), time_(0), step_(0), num_remain_(0)
  {}
  //! steady state case, sets only the end time and num-steps to 1
  time_data(double endt)
      : smethod_(time_method::steady), stop_time_(endt), time_(0), step_(0), num_remain_(1)
  {}
  //! specify time-step and final time
  time_data(time_method smethod, input_dt dt, input_stop_time stop_time)
      : smethod_(smethod), dt_(dt.value), stop_time_(stop_time.value),
        time_(0), step_(0)
  {
    // assume we are doing at least 1 step and round down end_time / dt
    num_remain_ = static_cast<int64_t>(stop_time_ / dt_);
    if (num_remain_ == 0)
      num_remain_ = 1;
    // readjust dt to minimize rounding error
    if (dt_ * num_remain_ < stop_time_)
      num_remain_ += 1;
    dt_ = stop_time_ / static_cast<double>(num_remain_);
  }
  //! specify number of steps and final time
  time_data(time_method smethod, int64_t num_steps, input_stop_time stop_time)
    : smethod_(smethod), stop_time_(stop_time.value), time_(0), step_(0),
      num_remain_(num_steps)
  {
    dt_ = (num_remain_ == 0) ? 0 : (stop_time_ / static_cast<double>(num_remain_));
  }
  //! specify time-step and number of steps
  time_data(time_method smethod, input_dt dt, int64_t num_steps)
    : smethod_(smethod), dt_(dt.value), time_(0), step_(0),
      num_remain_(num_steps)
  {
    stop_time_ = num_remain_ * dt_;
  }

  //! return the time-advance method
  time_method step_method() const { return smethod_; }

  //! returns the time-step
  double dt() const { return dt_; }
  //! returns the stop-time
  double stop_time() const { return stop_time_; }
  //! returns the current time
  double time() const { return time_; }
  //! returns the current time, non-const ref that can reset the time
  double &time() { return time_; }
  //! returns the current step number
  int64_t step() const { return step_; }
  //! returns the number of remaining time-steps
  int64_t num_remain() const { return num_remain_; }
  //! set final time and zero out the num_remain
  void set_final_time() {
    time_       = stop_time_;
    num_remain_ = 0;
  }

  //! advances the time and updates the current and remaining steps
  void take_step() {
    ++step_;
    --num_remain_;
    time_ += dt_;
  }

  //! prints the stepping data to a stream (human readable format)
  void print_time(std::ostream &os) const {
    os << "  time (t)        " << time_
       << "\n  stop-time (T)   " << stop_time_
       << "\n  num-steps (n)   " << tools::split_style(num_remain_)
       << "\n  time-step (dt)  " << dt_ << '\n';
  }

  //! allows writer to save/load the time data
  friend class h5manager<double>;
  //! allows writer to save/load the time data
  friend class h5manager<float>;

private:
  time_method smethod_ = time_method::rk2;
  // the following entries cannot be negative, negative means "not-set"

  //! current time-step
  double dt_ = -1;
  //! currently set final time
  double stop_time_ = -1;
  //! current time for the simulation
  double time_ = -1;
  //! current number of steps taken
  int64_t step_ = -1;
  //! remaining steps
  int64_t num_remain_ = -1;
};

/*!
 * \ingroup asgard_discretization
 * \brief Allows writing time-data to a stream
 */
inline std::ostream &operator<<(std::ostream &os, time_data const &dtime)
{
  dtime.print_time(os);
  return os;
}

} // namespace asgard
