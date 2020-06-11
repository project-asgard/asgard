#pragma once

#include <limits>
#include <map>
#include <string>

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
  // FIXME the below have not been implemented according to the
  // new specification. david is working on that in the matlab
  vlasov4,  // PDE corresponding to Fig. 4 in FIXME
  vlasov43, // PDE corresponding to Fig. 4.3 in FIXME
  vlasov5,  // PDE corresponding to Fig. 5 in FIXME
  vlasov7,  // PDE corresponding to Fig. 7 in FIXME
  vlasov8,  // PDE corresponding to Fig. 8 in FIXME
  pde_user  // FIXME will need to add the user supplied PDE choice
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
    {"pde_user", PDE_opts::pde_user},
    {"vlasov4", PDE_opts::vlasov4},
    {"vlasov7", PDE_opts::vlasov7},
    {"vlasov8", PDE_opts::vlasov8},
    {"vlasov5", PDE_opts::vlasov5},
    {"vlasov43", PDE_opts::vlasov43}};

class options
{
public:
  static int constexpr NO_USER_VALUE       = -1;
  static double constexpr NO_USER_VALUE_FP = std::numeric_limits<double>::min();
  static double constexpr DEFAULT_CFL      = 0.01;

  // construct from command line
  options(int argc, char **argv);

  // construct from provided values - for testing
  options(PDE_opts const pde_choice, int const level, int const degree,
          double const cfl)
      : level(level), degree(degree), cfl(cfl), pde_choice(pde_choice){};

  int get_level() const;
  void update_level(int const level) { this->level = level; }
  int get_degree() const;
  void update_degree(int const degree) { this->degree = degree; }
  int get_max_level() const;
  double get_dt() const;
  void update_dt(double const dt) { this->dt = dt; }
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
  bool should_output_wavelet(int const i) const;
  bool should_output_realspace(int const i) const;
  solve_opts get_selected_solver() const;

private:
  // FIXME level and degree are unique to dimensions, will
  // need to support inputting level and degree per dimensions
  // in future
  int level  = NO_USER_VALUE; // resolution. NO_USER_VALUE loads default in pde
  int degree = NO_USER_VALUE; // deg of legendre basis polys. NO_USER_VALUE
                              // loads default in pde
  int max_level      = 12;    // max adaptivity level for any given dimension.
  int num_time_steps = 10;    // number of time loop iterations
  int write_frequency =
      0; // write wavelet space output every this many iterations
  bool use_implicit_stepping = false; // enable implicit(/explicit) stepping
  bool use_full_grid         = false; // enable full(/sparse) grid
  bool do_poisson            = false; // do poisson solve for electric field
  double cfl = NO_USER_VALUE_FP; // the Courant-Friedrichs-Lewy (CFL) condition
  double dt =
      NO_USER_VALUE_FP; // size of time steps. double::min loads default in pde

  int realspace_output_freq =
      0; // timesteps between realspace output writes to disk

  // default
  std::string selected_pde = "continuity_2";
  // pde to construct/evaluate
  PDE_opts pde_choice = PDE_opts::continuity_2;

  // default
  std::string selected_solver = "none";
  // solver to use for implicit timestepping
  solve_opts solver = solve_opts::direct;

  // is there a better (testable) way to handle invalid command-line input?
  bool valid = true;

  // helper for output writing
  bool write_at_step(int const i, int const freq) const;
};
