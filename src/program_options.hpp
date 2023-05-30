#pragma once

#include "tensors.hpp"
#include "tools.hpp"

#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace asgard
{
// implemented solvers for implicit stepping
enum class solve_opts
{
  direct,
  gmres,
  scalapack
};

// map those choices to selection strings

using solve_map_t = std::map<std::string_view, solve_opts>;
static solve_map_t const solver_mapping = {
    {"direct", solve_opts::direct},
    {"gmres", solve_opts::gmres},
    {"scalapack", solve_opts::scalapack}};

// the choices for supported PDE types
enum class PDE_opts
{
  advection_1,
  continuity_1,
  continuity_2,
  continuity_3,
  continuity_6,
  fokkerplanck_1d_pitch_E_case1,
  fokkerplanck_1d_pitch_E_case2,
  fokkerplanck_1d_pitch_C,
  fokkerplanck_1d_4p3,
  fokkerplanck_1d_4p4,
  fokkerplanck_1d_4p5,
  fokkerplanck_2d_complete_case1,
  fokkerplanck_2d_complete_case2,
  fokkerplanck_2d_complete_case3,
  fokkerplanck_2d_complete_case4,
  diffusion_1,
  diffusion_2,
  vlasov_lb_full_f,
  vlasov_two_stream,
  collisional_landau
  // FIXME will need to add the user supplied PDE choice
};

enum class PDE_case_opts
{
  case0,
  case1,
  case2,
  case3,
  case4,
  case_count
  // FIXME will need to add the user supplied PDE cases choice
};

/*!
 * \brief Indicates whether we should be using sparse or dense kronmult.
 */
enum class kronmult_mode
{
  //! \brief Using a dense matrix (assumes everything is connected).
  dense,
  //! \brief Using a sparse matrix (checks for connectivity).
  sparse
};

class PDE_descriptor
{
public:
  PDE_descriptor(std::string const info_in, PDE_opts const pde_choice_in)
      : info(info_in), pde_choice(pde_choice_in){};
  std::string const info;
  PDE_opts const pde_choice;
};

//
// map those choices to selection strings
//
using pde_map_t                    = std::map<std::string, PDE_descriptor>;
static pde_map_t const pde_mapping = {
    {"continuity_1",
     PDE_descriptor("1D test case, continuity equation: df/dt + df/dx = 0",
                    PDE_opts::continuity_1)},
    {"continuity_2", PDE_descriptor("2D test case, continuity equation: df/dt "
                                    "+ v_x * df/dx + v_y * df/dy == 0",
                                    PDE_opts::continuity_2)},
    {"continuity_3", PDE_descriptor("3D test case, continuity equation, df/dt "
                                    "+ v.grad(f)==0 where v={1,1,1}",
                                    PDE_opts::continuity_3)},
    {"continuity_6",
     PDE_descriptor("6D test case, continuity equation, df/dt + b.grad_x(f) + "
                    "a.grad_v(f)==0 where b={1,1,3}, a={4,3,2}",
                    PDE_opts::continuity_6)},
    // the following are labelled according to figure number in the runaway
    // electron paper
    {"fokkerplanck_1d_pitch_E_case1",
     PDE_descriptor(
         "1D pitch angle collisional term: df/dt == d/dz ( (1-z^2) df/dz, f0 is"
         " constant.",
         PDE_opts::fokkerplanck_1d_pitch_E_case1)},
    {"fokkerplanck_1d_pitch_E_case2",
     PDE_descriptor(
         "1D pitch angle collisional term: df/dt == d/dz ( (1-z^2) df/dz, f0 is"
         " gaussian.",
         PDE_opts::fokkerplanck_1d_pitch_E_case2)},
    {"fokkerplanck_1d_pitch_C",
     PDE_descriptor(
         "1D pitch angle collisional term: df/dt == d/dz ( (1-z^2) df/dz",
         PDE_opts::fokkerplanck_1d_pitch_C)},
    {"fokkerplanck_1d_4p3",
     PDE_descriptor("Radiation damping term: df/dt == -d/dz ( z(1-z^2)f )",
                    PDE_opts::fokkerplanck_1d_4p3)},

    {"fokkerplanck_1d_4p4",
     PDE_descriptor("Evolution of f's pitch angle dependence with electric "
                    "field acceleration/collision: df/dt == -E d/dz((1-z^2) f) "
                    "+ C d/dz((1-z^2) df/dz)",
                    PDE_opts::fokkerplanck_1d_4p4)},
    {"fokkerplanck_1d_4p5",
     PDE_descriptor(
         "Same as 4p4, but with radiation damping: df/dt == -E d/dz((1-z^2) f) "
         "+ C d/dz((1-z^2) df/dz) - R d/dz(z(1-z^2) f)",
         PDE_opts::fokkerplanck_1d_4p5)},

    {"fokkerplanck_2d_complete_case1",
     PDE_descriptor("Full PDE from the 2D runaway electron paper: d/dt f(p,z) "
                    "== -div(flux_C + flux_E + flux_R), case 1",
                    PDE_opts::fokkerplanck_2d_complete_case1)},
    {"fokkerplanck_2d_complete_case2",
     PDE_descriptor("Full PDE from the 2D runaway electron paper: d/dt f(p,z) "
                    "== -div(flux_C + flux_E + flux_R), case 2",
                    PDE_opts::fokkerplanck_2d_complete_case2)},
    {"fokkerplanck_2d_complete_case3",
     PDE_descriptor("Full PDE from the 2D runaway electron paper: d/dt f(p,z) "
                    "== -div(flux_C + flux_E + flux_R), case 3",
                    PDE_opts::fokkerplanck_2d_complete_case3)},
    {"fokkerplanck_2d_complete_case4",
     PDE_descriptor("Full PDE from the 2D runaway electron paper: d/dt f(p,z) "
                    "== -div(flux_C + flux_E + flux_R), case 4",
                    PDE_opts::fokkerplanck_2d_complete_case4)},
    {"diffusion_1", PDE_descriptor("1D diffusion equation: df/dt = d^2 f/dx^2",
                                   PDE_opts::diffusion_1)},
    {"diffusion_2",
     PDE_descriptor("2D (1x-1y) heat equation. df/dt = d^2 f/dx^2 + d^2 f/dy^2",
                    PDE_opts::diffusion_2)},
    {"advection_1",
     PDE_descriptor(
         "1D test using continuity equation. df/dt == -2*df/dx - 2*sin(x)",
         PDE_opts::advection_1)},
    {"vlasov", PDE_descriptor("Vlasov lb full f. df/dt == -v*grad_x f + div_v( "
                              "(v-u)f + theta*grad_v f)",
                              PDE_opts::vlasov_lb_full_f)},
    {"two_stream",
     PDE_descriptor("Vlasov two-stream. df/dt == -v*grad_x f -E*grad_v f",
                    PDE_opts::vlasov_two_stream)},
    {"landau", PDE_descriptor("Collisional Landau. df/dt == -v*grad_x f "
                              "-E*grad_v f + div_v((v-u)f + theta*grad_v f)",
                              PDE_opts::collisional_landau)}};

// class to parse command line input
class parser
{
public:
  static auto constexpr NO_USER_VALUE     = -1;
  static auto constexpr NO_USER_VALUE_FP  = std::numeric_limits<double>::min();
  static auto constexpr NO_USER_VALUE_STR = "none";

  static auto constexpr DEFAULT_CFL               = 0.01;
  static auto constexpr DEFAULT_ADAPT_THRESH      = 1e-3;
  static auto constexpr DEFAULT_MAX_LEVEL         = 8;
  static auto constexpr DEFAULT_MIXED_GRID_GROUP  = -1;
  static auto constexpr DEFAULT_TIME_STEPS        = 10;
  static auto constexpr DEFAULT_WRITE_FREQ        = 0;
  static auto constexpr DEFAULT_PLOT_FREQ         = 1;
  static auto constexpr DEFAULT_USE_IMPLICIT      = false;
  static auto constexpr DEFAULT_USE_IMEX          = false;
  static auto constexpr DEFAULT_USE_FG            = false;
  static auto constexpr DEFAULT_DO_POISSON        = false;
  static auto constexpr DEFAULT_DO_ADAPT          = false;
  static auto constexpr DEFAULT_PDE_STR           = "continuity_2";
  static auto constexpr DEFAULT_PDE_OPT           = PDE_opts::continuity_2;
  static auto constexpr DEFAULT_SOLVER            = solve_opts::direct;
  static auto constexpr DEFAULT_SOLVER_STR        = std::string_view("direct");
  static auto constexpr DEFAULT_PDE_SELECTED_CASE = PDE_case_opts::case0;
  static auto constexpr DEFAULT_MEMORY_LIMIT_MB   = 10000;
  static auto constexpr DEFAULT_KRONMULT_MODE     = kronmult_mode::dense;
  static auto constexpr DEFAULT_GMRES_TOLERANCE   = NO_USER_VALUE_FP;
  static auto constexpr DEFAULT_GMRES_INNER_ITERATIONS = NO_USER_VALUE;
  static auto constexpr DEFAULT_GMRES_OUTER_ITERATIONS = NO_USER_VALUE;


  // construct from command line
  explicit parser(int argc, char const *const *argv);

  // construct from provided values - to simplify testing
  explicit parser(
      PDE_opts const pde_choice_in, fk::vector<int> const &starting_levels_in,
      int const degree_in = NO_USER_VALUE, double const cfl_in = DEFAULT_CFL,
      bool const use_full_grid_in          = DEFAULT_USE_FG,
      int const max_level_in               = DEFAULT_MAX_LEVEL,
      int const mixed_grid_group_in        = DEFAULT_MIXED_GRID_GROUP,
      int const num_steps                  = DEFAULT_TIME_STEPS,
      bool const use_implicit              = DEFAULT_USE_IMPLICIT,
      bool const do_adapt_levels           = DEFAULT_DO_ADAPT,
      double const adapt_threshold_in      = DEFAULT_ADAPT_THRESH,
      std::string_view const solver_str_in = DEFAULT_SOLVER_STR,
      bool const use_imex                  = DEFAULT_USE_IMEX,
      int const memory_limit_in            = DEFAULT_MEMORY_LIMIT_MB,
      kronmult_mode const kmode_in         = DEFAULT_KRONMULT_MODE,
      double const gmres_tolerance_in      = DEFAULT_GMRES_TOLERANCE,
      int const gmres_inner_iterations_in  = DEFAULT_GMRES_INNER_ITERATIONS,
      int const gmres_outer_iterations_in  = DEFAULT_GMRES_OUTER_ITERATIONS)
      : use_implicit_stepping(use_implicit), use_full_grid(use_full_grid_in),
        do_adapt(do_adapt_levels), starting_levels(starting_levels_in),
        degree(degree_in), max_level(max_level_in),
        mixed_grid_group(mixed_grid_group_in), num_time_steps(num_steps),
        cfl(cfl_in), adapt_threshold(adapt_threshold_in),
        pde_choice(pde_choice_in), solver_str(solver_str_in),
        solver(solver_mapping.at(solver_str_in)), use_imex_stepping(use_imex),
        memory_limit(memory_limit_in), kmode(kmode_in),
        gmres_tolerance(gmres_tolerance_in),
        gmres_inner_iterations(gmres_inner_iterations_in),
        gmres_outer_iterations(gmres_outer_iterations_in){};

  explicit parser(
      std::string const &pde_choice_in, fk::vector<int> starting_levels_in,
      int const degree_in = NO_USER_VALUE, double const cfl_in = DEFAULT_CFL,
      bool const use_full_grid_in          = DEFAULT_USE_FG,
      int const max_level_in               = DEFAULT_MAX_LEVEL,
      int const mixed_grid_group_in        = DEFAULT_MIXED_GRID_GROUP,
      int const num_steps                  = DEFAULT_TIME_STEPS,
      bool const use_implicit              = DEFAULT_USE_IMPLICIT,
      bool const do_adapt_levels           = DEFAULT_DO_ADAPT,
      double const adapt_threshold_in      = DEFAULT_ADAPT_THRESH,
      std::string_view const solver_str_in = DEFAULT_SOLVER_STR,
      bool const use_imex                  = DEFAULT_USE_IMEX,
      int const memory_limit_in            = DEFAULT_MEMORY_LIMIT_MB,
      kronmult_mode const kmode_in         = DEFAULT_KRONMULT_MODE,
      double const gmres_tolerance_in      = DEFAULT_GMRES_TOLERANCE,
      int const gmres_inner_iterations_in  = DEFAULT_GMRES_INNER_ITERATIONS,
      int const gmres_outer_iterations_in  = DEFAULT_GMRES_OUTER_ITERATIONS)
      : parser(pde_mapping.at(pde_choice_in).pde_choice, starting_levels_in,
               degree_in, cfl_in, use_full_grid_in, max_level_in,
               mixed_grid_group_in, num_steps, use_implicit, do_adapt_levels,
               adapt_threshold_in, solver_str_in, use_imex, memory_limit_in,
               kmode_in, gmres_tolerance_in, gmres_inner_iterations_in,
               gmres_outer_iterations_in){};
  /*!
   * \brief Simple utility to modify private members of the parser.
   */
  friend struct parser_mod;

  bool using_implicit() const;
  bool using_imex() const;
  bool using_full_grid() const;
  bool do_poisson_solve() const;
  bool do_adapt_levels() const;

  fk::vector<int> get_starting_levels() const;
  fk::vector<int> get_active_terms() const;

  int get_degree() const;
  int get_max_level() const;
  int get_mixed_grid_group() const;
  int get_time_steps() const;
  int get_memory_limit() const;
  int get_gmres_inner_iterations() const;
  int get_gmres_outer_iterations() const;

  int get_wavelet_output_freq() const;
  int get_realspace_output_freq() const;

  kronmult_mode get_kmode() const;

  double get_dt() const;
  double get_cfl() const;
  double get_adapt_thresh() const;
  double get_gmres_tolerance() const;

  std::string get_pde_string() const;
  std::string get_solver_string() const;

  PDE_opts get_selected_pde() const;
  solve_opts get_selected_solver() const;

  std::string get_ml_session_string() const;
  int get_plot_freq() const;

  bool is_valid() const;

private:
  void print_available_pdes()
  {
    auto const max_name_length = 50;
    std::cerr << "available pdes (select using -p)"
              << "\n\n";
    std::cerr << std::left << std::setw(max_name_length) << "Argument"
              << "Description" << '\n';
    for (auto const &[pde_name, pde_enum_val] : pde_mapping)
    {
      ignore(pde_enum_val);
      expect(pde_name.size() <= max_name_length);
      std::cerr << std::left << std::setw(max_name_length) << pde_name
                << pde_mapping.at(pde_name).info << '\n';
    }
  }

  fk::vector<int> ints_from_string(std::string const &number_string)
  {
    std::stringstream number_stream{number_string};
    std::vector<int> parsed_ints;
    while (!number_stream.eof())
    {
      std::string word;
      number_stream >> word;
      int temp_int;
      if (std::stringstream(word) >> temp_int)
      {
        parsed_ints.push_back(temp_int);
      }
    }
    return fk::vector<int>(parsed_ints);
  }

  bool use_implicit_stepping =
      DEFAULT_USE_IMPLICIT;             // enable implicit(/explicit) stepping
  bool use_full_grid = DEFAULT_USE_FG;  // enable full(/sparse) grid
  bool do_poisson = DEFAULT_DO_POISSON; // do poisson solve for electric field
  bool do_adapt   = DEFAULT_DO_ADAPT;   // adapt number of basis levels

  // if none are provided, default is loaded from pde
  std::string starting_levels_str = NO_USER_VALUE_STR;
  fk::vector<int> starting_levels;
  std::string active_terms_str = NO_USER_VALUE_STR;
  fk::vector<int> active_terms;

  // deg of legendre basis polys. NO_USER_VALUE
  // loads default in pde
  int degree = NO_USER_VALUE;
  // max adaptivity level for any given dimension.
  int max_level = DEFAULT_MAX_LEVEL;
  // number of dimensions to use for the tensor groups in mixed grid
  int mixed_grid_group = DEFAULT_MIXED_GRID_GROUP;
  // number of time loop iterations
  int num_time_steps = DEFAULT_TIME_STEPS;
  // write wavelet space output every this many iterations
  int wavelet_output_freq = DEFAULT_WRITE_FREQ;
  // timesteps between realspace output writes to disk
  int realspace_output_freq = DEFAULT_WRITE_FREQ;
  // the Courant-Friedrichs-Lewy (CFL) condition
  double cfl = NO_USER_VALUE_FP;
  // size of time steps. double::min loads default in pde
  double dt = NO_USER_VALUE_FP;
  // relative adaptivity threshold
  // max(abs(x)) for solution vector x * adapt_thresh
  // is the threshold for refining elements
  double adapt_threshold = DEFAULT_ADAPT_THRESH;

  // default
  std::string pde_str = DEFAULT_PDE_STR;
  // pde to construct/evaluate
  PDE_opts pde_choice = DEFAULT_PDE_OPT;

  // default
  std::string solver_str = NO_USER_VALUE_STR;
  // solver to use for implicit timestepping
  solve_opts solver = DEFAULT_SOLVER;

  // name of matlab session to connect
  std::string matlab_name = NO_USER_VALUE_STR;
  // timesteps between plotting
  int plot_freq = DEFAULT_PLOT_FREQ;

  bool use_imex_stepping = DEFAULT_USE_IMEX;

  int memory_limit = DEFAULT_MEMORY_LIMIT_MB;

  kronmult_mode kmode = DEFAULT_KRONMULT_MODE;

  // gmres solver parameters
  double gmres_tolerance     = DEFAULT_GMRES_TOLERANCE;
  int gmres_inner_iterations = DEFAULT_GMRES_INNER_ITERATIONS;
  int gmres_outer_iterations = DEFAULT_GMRES_OUTER_ITERATIONS;

  // is there a better (testable) way to handle invalid command-line input?
  bool valid = true;
};

struct parser_mod
{
  enum parser_option_entry
  {
    // int values
    degree,
    max_level,
    mixed_grid_group,
    num_time_steps,
    wavelet_output_freq,
    realspace_output_freq,
    plot_freq,
    memory_limit,
    gmres_inner_iterations,
    gmres_outer_iterations,
    // bool values
    use_implicit_stepping,
    use_full_grid,
    do_poisson,
    do_adapt,
    use_imex_stepping,
    // double values
    cfl,
    dt,
    adapt_threshold,
    gmres_tolerance,
    // string
    solver_str
  };
  static void set(parser &p, parser_option_entry entry, int value);
  static void set(parser &p, parser_option_entry entry, bool value);
  static void set(parser &p, parser_option_entry entry, double value);
  static void
  set(parser &p, parser_option_entry entry, std::string const &value);
};

// simple class to hold non-pde user options
class options
{
public:
  options(parser const &user_vals)
      : starting_levels(user_vals.get_starting_levels()),
        adapt_threshold(user_vals.get_adapt_thresh()),
        gmres_tolerance(user_vals.get_gmres_tolerance()),
        max_level(user_vals.get_max_level()),
        mixed_grid_group(user_vals.get_mixed_grid_group()),
        num_time_steps(user_vals.get_time_steps()),
        wavelet_output_freq(user_vals.get_wavelet_output_freq()),
        realspace_output_freq(user_vals.get_realspace_output_freq()),
        plot_freq(user_vals.get_plot_freq()),
        memory_limit(user_vals.get_memory_limit()),
        kmode(user_vals.get_kmode()),
        gmres_inner_iterations(user_vals.get_gmres_inner_iterations()),
        gmres_outer_iterations(user_vals.get_gmres_outer_iterations()),
        use_implicit_stepping(user_vals.using_implicit()),
        use_full_grid(user_vals.using_full_grid()),
        do_poisson_solve(user_vals.do_poisson_solve()),
        do_adapt_levels(user_vals.do_adapt_levels()),
        solver(user_vals.get_selected_solver()),
        use_imex_stepping(user_vals.using_imex()){};

  bool should_output_wavelet(int const i) const;
  bool should_output_realspace(int const i) const;
  bool should_plot(int const i) const;

  fk::vector<int> const starting_levels;

  double const adapt_threshold;
  double const gmres_tolerance;

  int const max_level;
  int const mixed_grid_group;
  int const num_time_steps;
  int const wavelet_output_freq;
  int const realspace_output_freq;
  int const plot_freq;
  int const memory_limit;
  kronmult_mode const kmode;
  int const gmres_inner_iterations;
  int const gmres_outer_iterations;

  bool const use_implicit_stepping;
  bool const use_full_grid;
  bool const do_poisson_solve;
  bool const do_adapt_levels;

  solve_opts const solver;

  bool const use_imex_stepping;

private:
  // helper for output writing
  bool write_at_step(int const i, int const freq) const;
};
} // namespace asgard
