#pragma once

#include "tensors.hpp"
#include <limits>
#include <map>
#include <string>
#include <vector>

// implemented solvers for implicit stepping
enum class solve_opts
{
  direct,
  gmres
};

// map those choices to selection strings
using solve_map_t                       = std::map<std::string, solve_opts>;
static solve_map_t const solver_mapping = {
    {"direct", solve_opts::direct},
    {"gmres", solve_opts::gmres},
};

// the choices for supported PDE types
enum class PDE_opts
{
  continuity_1,
  continuity_2,
  continuity_3,
  continuity_6,
  fokkerplanck_1d_4p1a,
  fokkerplanck_1d_4p2,
  fokkerplanck_1d_4p3,
  fokkerplanck_1d_4p4,
  fokkerplanck_1d_4p5,
  fokkerplanck_2d_complete,
  diffusion_1,
  diffusion_2,
  // FIXME will need to add the user supplied PDE choice
};

//
// map those choices to selection strings
//
using pde_map_t                    = std::map<std::string, PDE_opts>;
static pde_map_t const pde_mapping = {
    {"continuity_1", PDE_opts::continuity_1},
    {"continuity_2", PDE_opts::continuity_2},
    {"continuity_3", PDE_opts::continuity_3},
    {"continuity_6", PDE_opts::continuity_6},
    {"fokkerplanck_1d_4p1a", PDE_opts::fokkerplanck_1d_4p1a},
    {"fokkerplanck_1d_4p2", PDE_opts::fokkerplanck_1d_4p2},
    {"fokkerplanck_1d_4p3", PDE_opts::fokkerplanck_1d_4p3},
    {"fokkerplanck_1d_4p4", PDE_opts::fokkerplanck_1d_4p4},
    {"fokkerplanck_1d_4p5", PDE_opts::fokkerplanck_1d_4p5},
    {"fokkerplanck_2d_complete", PDE_opts::fokkerplanck_2d_complete},
    {"diffusion_1", PDE_opts::diffusion_1},
    {"diffusion_2", PDE_opts::diffusion_2},
};

// class to parse command line input
class parser
{
public:
  static auto constexpr NO_USER_VALUE     = -1;
  static auto constexpr NO_USER_VALUE_FP  = std::numeric_limits<double>::min();
  static auto constexpr NO_USER_VALUE_STR = "none";

  static auto constexpr DEFAULT_CFL          = 0.01;
  static auto constexpr DEFAULT_MAX_LEVEL    = 12;
  static auto constexpr DEFAULT_TIME_STEPS   = 10;
  static auto constexpr DEFAULT_WRITE_FREQ   = 0;
  static auto constexpr DEFAULT_USE_IMPLICIT = false;
  static auto constexpr DEFAULT_USE_FG       = false;
  static auto constexpr DEFAULT_DO_POISSON   = false;
  static auto constexpr DEFAULT_PDE_STR      = "continuity_2";
  static auto constexpr DEFAULT_PDE_OPT      = PDE_opts::continuity_2;
  static auto constexpr DEFAULT_SOLVER       = solve_opts::direct;

  // construct from command line
  parser(int argc, char **argv);

  // FIXME todo - will eventually remove "level" argument
  // construct from provided values - to simplify testing
  parser(PDE_opts const pde_choice, int const level, int const degree,
         double const cfl)
      : level(level), degree(degree), cfl(cfl), pde_choice(pde_choice){};

  // construct from provided values - to simplify testing
  parser(PDE_opts const pde_choice, fk::vector<int> starting_levels,
         bool const use_full_grid = DEFAULT_USE_FG,
         int const degree = NO_USER_VALUE, double const cfl = NO_USER_VALUE_FP)
      : use_full_grid(use_full_grid), level(starting_levels(0)),
        starting_levels(starting_levels), degree(degree), cfl(cfl),
        pde_choice(pde_choice){};

  bool using_implicit() const;
  bool using_full_grid() const;
  bool do_poisson_solve() const;

  int get_level() const;
  fk::vector<int> get_starting_levels() const;
  int get_degree() const;
  int get_max_level() const;
  int get_time_steps() const;

  int get_wavelet_output_freq() const;
  int get_realspace_output_freq() const;

  double get_dt() const;
  double get_cfl() const;

  std::string get_pde_string() const;
  std::string get_solver_string() const;

  PDE_opts get_selected_pde() const;
  solve_opts get_selected_solver() const;

  bool is_valid() const;

private:
  bool use_implicit_stepping =
      DEFAULT_USE_IMPLICIT;             // enable implicit(/explicit) stepping
  bool use_full_grid = DEFAULT_USE_FG;  // enable full(/sparse) grid
  bool do_poisson = DEFAULT_DO_POISSON; // do poisson solve for electric field

  // FIXME level and degree are unique to dimensions, will
  // need to support inputting level and degree per dimensions
  // in future

  // FIXME temporary - will remove completely once multiple levels supported
  // throughout code
  int level = NO_USER_VALUE; // resolution. NO_USER_VALUE loads default in pde

  // FIXME this will store the starting levels input by user in dimension order
  fk::vector<int> starting_levels;

  int degree = NO_USER_VALUE; // deg of legendre basis polys. NO_USER_VALUE
                              // loads default in pde
  int max_level =
      DEFAULT_MAX_LEVEL; // max adaptivity level for any given dimension.
  int num_time_steps = DEFAULT_TIME_STEPS; // number of time loop iterations

  int wavelet_output_freq = DEFAULT_WRITE_FREQ; // write wavelet space output
                                                // every this many iterations
  int realspace_output_freq =
      DEFAULT_WRITE_FREQ; // timesteps between realspace output writes to disk

  double cfl = NO_USER_VALUE_FP; // the Courant-Friedrichs-Lewy (CFL) condition
  double dt =
      NO_USER_VALUE_FP; // size of time steps. double::min loads default in pde

  // default
  std::string pde_str = DEFAULT_PDE_STR;
  // pde to construct/evaluate
  PDE_opts pde_choice = DEFAULT_PDE_OPT;

  // default
  std::string solver_str = NO_USER_VALUE_STR;
  // solver to use for implicit timestepping
  solve_opts solver = DEFAULT_SOLVER;

  // is there a better (testable) way to handle invalid command-line input?
  bool valid = true;
};

// simple class to hold non-pde user options

class options
{
public:
  // FIXME will be removed after multi-level PR
  options(parser const &user_vals)
      : starting_level(user_vals.get_level()),
        max_level(user_vals.get_max_level()),
        num_time_steps(user_vals.get_time_steps()),
        wavelet_output_freq(user_vals.get_wavelet_output_freq()),
        realspace_output_freq(user_vals.get_realspace_output_freq()),
        use_implicit_stepping(user_vals.using_implicit()),
        use_full_grid(user_vals.using_full_grid()),
        do_poisson_solve(user_vals.do_poisson_solve()),
        solver(user_vals.get_selected_solver()){};

  bool should_output_wavelet(int const i) const;
  bool should_output_realspace(int const i) const;

  // FIXME temporary, will be replaced with levels vector
  int const starting_level;
  int const max_level;
  int const num_time_steps;
  int const wavelet_output_freq;
  int const realspace_output_freq;

  bool const use_implicit_stepping;
  bool const use_full_grid;
  bool const do_poisson_solve;

  solve_opts const solver;

private:
  // helper for output writing
  bool write_at_step(int const i, int const freq) const;
};
