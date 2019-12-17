#pragma once

#include "pde.hpp"
#include <map>
#include <string>

class options
{
private:
  // FIXME level and degree are unique to dimensions, will
  // need to support inputting level and degree per dimensions
  // in future
  int level  = -1; // resolution. -1 loads default in pde
  int degree = -1; // deg of legendre basis polys. -1 loads default in pde
  int num_time_steps = 10; // number of time loop iterations
  int write_frequency =
      0; // write wavelet space output every this many iterations
  bool use_implicit_stepping = false; // enable implicit(/explicit) stepping
  bool use_full_grid         = false; // enable full(/sparse) grid
  bool do_poisson            = false; // do poisson solve for electric field
  double cfl = 0.01; // the Courant-Friedrichs-Lewy (CFL) condition
  int realspace_output_freq =
      0; // timesteps between realspace output writes to disk

  // default
  std::string selected_pde = "continuity_2";

  // pde to construct/evaluate
  PDE_opts pde_choice;

  // is there a better (testable) way to handle invalid command-line input?
  bool valid = true;

public:
  options(int argc, char **argv);
  int get_level() const;
  void update_level(int const level) { this->level = level; }
  int get_degree() const;
  void update_degree(int const degree) { this->degree = degree; }
  int get_time_steps() const;
  int get_write_frequency() const;
  bool using_implicit() const;
  bool using_full_grid() const;
  double get_cfl() const;
  PDE_opts get_selected_pde() const;
  std::string get_pde_string() const;
  bool do_poisson_solve() const;
  bool is_valid() const;
  int get_realspace_output_freq() const;
  bool write_at_step(int const i) const;
  bool transform_at_step(int const i) const;
};
