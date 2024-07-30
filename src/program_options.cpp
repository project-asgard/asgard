#include "program_options.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include "clara.hpp"
#pragma GCC diagnostic pop
#include "distribution.hpp"
#ifdef ASGARD_IO_HIGHFIVE
#include "io.hpp"
#endif

namespace asgard
{
parser::parser(int argc, char const *const *argv)
{
  bool show_help        = false;
  bool show_pdes        = false;
  std::string kmode_str = "dense";
  // Parsing...
  auto cli =
      clara::detail::Help(show_help) |
      clara::detail::Opt(show_pdes)["-a"]["--available_pdes"](
          "Print available pdes (for -p argument) and exit") |
      clara::detail::Opt(cfl, "positive float")["-c"]["--cfl"](
          "The Courant-Friedrichs-Lewy (CFL) condition") |
      clara::detail::Opt(dt,
                         "positive float")["-t"]["--dt"]("Size of time steps") |
      clara::detail::Opt(degree, "positive integer")["-d"]["--degree"](
          "Terms in legendre basis polynomials") |
      clara::detail::Opt(use_full_grid)["-f"]["--full_grid"](
          "Use full grid (vs. sparse grid)") |
      clara::detail::Opt(use_linf_nrm)["--l_inf"](
          "Use Linf norm (vs. L2 norm)") |
      clara::detail::Opt(use_implicit_stepping)["-i"]["--implicit"](
          "Use implicit time advance (vs. explicit)") |
      clara::detail::Opt(solver_str,
                         "direct|gmres|bicgstab|scalapack")["-s"]["--solver"](
          "Solver to use for implicit advance") |
      clara::detail::Opt(starting_levels_str,
                         "e.g. for 2d PDE: \"3 2\"")["-l"]["--levels"](
          "Stating hierarchical levels (resolution)") |
      clara::detail::Opt(max_level,
                         "integer >= all starting levels")["-m"]["--max_level"](
          "Maximum hierarchical levels (resolution) for adaptivity") |
      clara::detail::Opt(libinfo)["-v"]["--version"](
          "Show information about this build.") |
      clara::detail::Opt(mixed_grid_group,
                         "integer < num dimensions")["--mixed_grid_group"](
          "Dimension group size for first sparse grid in the mixed "
          "sparse-dense grid") |
      clara::detail::Opt(num_time_steps,
                         "positive integer")["-n"]["--num_steps"](
          "Number of iterations") |
      clara::detail::Opt(pde_str, "e.g. continuity2")["-p"]["--pde"](
          "PDE to solve; use -a option to print list of choices") |
      clara::detail::Opt(do_poisson)["-e"]["--electric_solve"](
          "Do poisson solve for electric field") |
      clara::detail::Opt(wavelet_output_freq,
                         "0-num_time_steps")["-w"]["--wave_freq"](
          "Frequency in steps for writing wavelet space "
          "output") |
      clara::detail::Opt(realspace_output_freq,
                         "0-num_time_steps")["-r"]["--real_freq"](
          "Frequency in steps for writing realspace output") |
      clara::detail::Opt(do_adapt)["--adapt"]("Enable/disable adaptivity") |
      clara::detail::Opt(adapt_threshold, " 0>threshold<1 ")["--thresh"](
          "Relative threshold for adaptivity") |
      clara::detail::Opt(matlab_name, "session name")["--matlab_name"](
          "Name of a shared MATLAB session to connect to") |
      clara::detail::Opt(plot_freq, "0-num_time_steps")["--plot_freq"](
          "Frequency in steps for displaying plots") |
      clara::detail::Opt(active_terms_str, "e.g. \"1 1 1 0 0 0\"")["--terms"](
          "Select specific terms to use (1 = on, 0 = off)") |
      clara::detail::Opt(use_imex_stepping)["-x"]["--imex"](
          "Use IMEX (implicit-explicit) time advance (vs. explicit or "
          "implicit)") |
      clara::detail::Opt(memory_limit, "size > 0")["--memory"](
          "Maximum workspace size in MB that will be resident on an "
          "accelerator") |
      clara::detail::Opt(kmode_str, "dense/sparse")["--kron-mode"](
          "Select dense (default) or sparse mode for the kronmult operations") |
      clara::detail::Opt(gmres_tolerance, "tol > 0")["--tol"](
          "Tolerance used to determine convergence in bicgstab/gmres solvers") |
      clara::detail::Opt(gmres_inner_iterations, "inner_it > 0")["--inner_it"](
          "Number of inner iterations in gmres solver") |
      clara::detail::Opt(gmres_outer_iterations, "outer_it > 0")["--outer_it"](
          "Number of max/outer iterations in bicgstab/gmres solver") |
      clara::detail::Opt(max_adapt_levels_str, "")["--max_adapt_levels"](
          "Maximum hierarchical levels (resolution) for adaptivity") |
      clara::detail::Opt(restart_file, "filename")["--restart"](
          "Load a HDF5 file to restart from");

  auto result = cli.parse(clara::detail::Args(argc, argv));
  if (!result)
  {
    std::cerr << "Error in command line parsing: " << result.errorMessage()
              << '\n';
    valid = false;
  }

  // default to max-level 8 but if max level is not explicitly given
  // and the active levels exceed 8, then bump up the max-level
  bool const missing_max_level = (max_level == DEFAULT_MAX_LEVEL);

  max_level = std::max(max_level, 8);

  for (int i = 1; i < argc; i++)
  {
    this->cli_opts.push_back(std::string(argv[i]));
  }

  if (show_help)
  {
    std::cerr << cli << '\n';
  }
  if (show_pdes)
  {
    print_available_pdes();
  }
  if (show_help || show_pdes)
  {
    exit(0);
  }

  if (do_restart())
  {
#ifdef ASGARD_IO_HIGHFIVE
    if (std::filesystem::exists(get_restart_file()))
    {
      asgard::read_restart_metadata<double>(*this, get_restart_file());
    }
    else
    {
      std::cerr << "Could not open restart file '" << get_restart_file()
                << "'\n";
      valid = false;
    }
#else
    std::cerr << "Must build with ASGARD_IO_HIGHFIVE to use restart option"
              << '\n';
    valid = false;
#endif
  }

  // Validation...
  if (cfl != NO_USER_VALUE_FP)
  {
    if (cfl <= 0.0)
    {
      std::cerr << "CFL must be positive" << '\n';
      valid = false;
    }
    if (dt != NO_USER_VALUE_FP)
    {
      std::cerr << "CFL and explicit dt options are mutually exclusive" << '\n';
      valid = false;
    }
  }
  else
  {
    cfl = DEFAULT_CFL;
  }

  if (degree < 1 && degree != NO_USER_VALUE)
  {
    std::cerr << "Degree must be a natural number" << '\n';
    valid = false;
  }

  if (starting_levels_str != NO_USER_VALUE_STR)
  {
    auto const starting_lev = ints_from_string(starting_levels_str);
    if (starting_lev.empty())
    {
      std::cerr << "Failed to parse starting levels from input argument"
                << '\n';
      valid = false;
    }
    starting_levels.resize(starting_lev.size()) = starting_lev;
    // check that at least one level is greater than 0
    int lev_sum = 0;
    if (missing_max_level)
      for (auto const lev : starting_levels)
        max_level = std::max(max_level, lev);

    for (auto const lev : starting_levels)
    {
      if (lev < 0)
      {
        std::cerr << "Level must be >= 0" << '\n';
        valid = false;
      }
      if (max_level < lev)
      {
        std::cerr
            << "Maximum level must be greater than or equal to starting level"
            << '\n';
        valid = false;
      }
      lev_sum += lev;
    }
    if (lev_sum == 0 && starting_lev.size() > 1)
    {
      std::cerr << "At least one level must be > 0" << '\n';
      valid = false;
    }
  }
  if (memory_limit <= 0)
  {
    std::cerr << "Kronmult max memory size must be a positive integer\n";
    valid = false;
  }
  if (dt != NO_USER_VALUE_FP && dt <= 0.0)
  {
    std::cerr << "Provided dt must be positive" << '\n';
    valid = false;
  }
  if (num_time_steps < 1)
  {
    std::cerr << "Number of timesteps must be a natural number" << '\n';
    valid = false;
  }

  if (auto const choice = pde_mapping.find(pde_str);
      choice != pde_mapping.end())
  {
    pde_choice = choice->second.pde_choice;
  }
  else
  {
    std::cerr << "Invalid pde choice; see options.hpp for valid choices"
              << '\n';
    valid = false;
  }
  if (realspace_output_freq < 0 || wavelet_output_freq < 0 || plot_freq < 0)
  {
    std::cerr << "Write and plot frequencies must be non-negative" << '\n';
    valid = false;
  }

  if (realspace_output_freq > num_time_steps ||
      wavelet_output_freq > num_time_steps)
  {
    std::cerr
        << "Requested a write or plot frequency > number of steps - no output "
           "will be produced"
        << '\n';
    valid = false;
  }

#if not defined(ASGARD_IO_HIGHFIVE) && not defined(ASGARD_USE_MATLAB)
  if (realspace_output_freq > 0 || wavelet_output_freq > 0)
  {
    std::cerr << "Must build with ASGARD_IO_HIGHFIVE to write output or ASGARD_USE_MATLAB to display output" << '\n';
    valid = false;
  }
#endif

  if (use_implicit_stepping && use_imex_stepping)
  {
    std::cerr << "Invalid time stepping choice: only implicit or imex can be "
                 "selected, not both.\n";
    valid = false;
  }

  if (use_imex_stepping && solver_str != "bicgstab")
  {
    // imex defaults to gmres
    solver_str = "gmres";
  }

  if (use_implicit_stepping || use_imex_stepping)
  {
    if (solver_str == "none")
    {
      solver_str = "direct";
    }
#ifndef ASGARD_USE_SCALAPACK
    if (solver_str == "scalapack")
    {
      std::cerr
          << "Invalid solver choice; ASGarD not built with SCALAPACK option "
             "enabled\n";
      valid = false;
    }
#endif
    if (auto const choice = solver_mapping.find(solver_str);
        choice != solver_mapping.end())
    {
      solver = choice->second;
    }
    else
    {
      std::cerr << "Invalid solver choice; see options.hpp for valid choices\n";
      valid = false;
    }
  }
  else // explicit time advance
  {
    if (solver_str != "none")
    {
      std::cerr << "Must set implicit (-i) flag to select a solver\n";
      valid = false;
    }
  }

#ifdef ASGARD_USE_CUDA
  if (use_implicit_stepping && solver_str != "gmres" && solver_str != "bicgstab")
  {
    std::cerr << "GPU acceleration for implicit stepping only supports bicgstab or gmres\n";
    valid = false;
  }
#endif

#ifdef ASGARD_USE_MPI
  if ((use_implicit_stepping || use_imex_stepping) && get_num_ranks() > 1)
  {
    auto const choice = solver_mapping.at(solver_str);
    if (choice != solve_opts::scalapack)
    {
      std::cerr << "Distribution not implemented for implicit stepping\n";
      valid = false;
    }
  }
  if (realspace_output_freq > 0)
  {
    std::cerr << "Distribution does not yet support realspace transform\n";
    valid = false;
  }
#endif

  if (adapt_threshold > 1.0 || adapt_threshold <= 0)
  {
    std::cerr << "input adaptivity threshold between 0 and 1, exclusive"
              << '\n';
    valid = false;
  }
  if (adapt_threshold != DEFAULT_ADAPT_THRESH && !do_adapt)
  {
    std::cerr << "input adaptivity threshold without enabling adaptivity..."
              << '\n';
    valid = false;
  }
  if (use_linf_nrm != DEFAULT_USE_LINF_NRM && !do_adapt)
  {
    std::cerr << "input adaptivity norm without enabling adaptivity..." << '\n';
    valid = false;
  }

#ifndef ASGARD_USE_MATLAB
  if (matlab_name != NO_USER_VALUE_STR)
  {
    std::cerr << "Must be built with ASGARD_USE_MATLAB to use Matlab" << '\n';
    valid = false;
  }
  if (plot_freq != DEFAULT_PLOT_FREQ)
  {
    std::cerr << "Must be built with ASGARD_USE_MATLAB to plot results" << '\n';
    valid = false;
  }
#endif

  if (active_terms_str != NO_USER_VALUE_STR)
  {
    auto const starting_terms = ints_from_string(active_terms_str);
    if (starting_terms.empty())
    {
      std::cerr << "Failed to parse active terms from input argument" << '\n';
      valid = false;
    }
    active_terms.resize(starting_terms.size()) = starting_terms;
    for (auto const term : active_terms)
    {
      if (term != 0 && term != 1)
      {
        std::cerr << "Term must be 0 or 1" << '\n';
        valid = false;
      }
    }
  }

  if (kmode_str == "dense")
  {
    kmode = kronmult_mode::dense;
  }
  else if (kmode_str == "sparse")
  {
    kmode = kronmult_mode::sparse;
  }
  else
  {
    std::cerr << "kron_mode is set to: '" << kmode_str
              << " but it must be either 'dense' or 'sparse'\n";
    valid = false;
  }

  if (solver != solve_opts::gmres && solver != solve_opts::bicgstab && gmres_tolerance != NO_USER_VALUE_FP)
  {
    std::cerr << "gmres tolerance has no effect with solver = " << solver_str
              << '\n';
    valid = false;
  }
  if (gmres_tolerance != NO_USER_VALUE_FP && gmres_tolerance <= 0.0)
  {
    std::cerr << "Provided gmres tolerance must be positive" << '\n';
    valid = false;
  }
  if (solver != solve_opts::gmres && gmres_inner_iterations != NO_USER_VALUE)
  {
    std::cerr << "gmres innter iterations has no effect with solver = "
              << solver_str << '\n';
    valid = false;
  }
  if (gmres_inner_iterations != NO_USER_VALUE && gmres_inner_iterations < 1)
  {
    std::cerr << "Number of gmres inner iterations must be a natural number"
              << '\n';
    valid = false;
  }
  if (solver != solve_opts::gmres && solver != solve_opts::bicgstab && gmres_outer_iterations != NO_USER_VALUE)
  {
    std::cerr << "Number of gmres outer iterations has no effect with solver = "
              << solver_str << '\n';
    valid = false;
  }
  if (gmres_outer_iterations != NO_USER_VALUE && gmres_outer_iterations < 1)
  {
    std::cerr << "Number of gmres outer iterations must be a natural number"
              << '\n';
    valid = false;
  }
  if (max_adapt_levels_str != NO_USER_VALUE_STR)
  {
    max_adapt_levels = ints_from_string(max_adapt_levels_str);
    if (max_adapt_levels.empty())
    {
      std::cerr
          << "Failed to parse maximum adaptivity levels from input argument"
          << '\n';
      valid = false;
    }
    if (!max_adapt_levels.empty() && !do_adapt)
    {
      std::cerr
          << "input maximum adaptivity levels without enabling adaptivity..."
          << '\n';
      valid = false;
    }
    if (missing_max_level)
      for (int i = 0; i < max_adapt_levels.size(); ++i)
        max_level = std::max(max_level, max_adapt_levels[i]);

    for (int i = 0; i < max_adapt_levels.size(); ++i)
    {
      if (max_adapt_levels[i] < 0)
      {
        std::cerr << "Level must be >= 0" << '\n';
        valid = false;
      }
      if (max_adapt_levels[i] < starting_levels[i])
      {
        std::cerr
            << "Maximum level must be greater than or equal to starting level"
            << '\n';
        valid = false;
      }
      if (max_adapt_levels[i] > max_level)
      {
        std::cerr << "Maximum level must be less than or equal to maximum level"
                  << '\n';
        valid = false;
      }
    }
  }
}

bool parser::using_implicit() const { return use_implicit_stepping; }
bool parser::using_imex() const { return use_imex_stepping; }
bool parser::using_full_grid() const { return use_full_grid; }
bool parser::using_linf_nrm() const { return use_linf_nrm; }
bool parser::do_poisson_solve() const { return do_poisson; }
bool parser::do_adapt_levels() const { return do_adapt; }
bool parser::do_restart() const { return restart_file != NO_USER_VALUE_STR; }

fk::vector<int> const &parser::get_starting_levels() const { return starting_levels; }
fk::vector<int> const &parser::get_active_terms() const { return active_terms; }
int parser::get_degree() const { return degree; }
int parser::get_max_level() const { return max_level; }
int parser::get_mixed_grid_group() const { return mixed_grid_group; }
int parser::get_time_steps() const { return num_time_steps; }
int parser::get_memory_limit() const { return memory_limit; }
kronmult_mode parser::get_kmode() const { return kmode; }
int parser::get_wavelet_output_freq() const { return wavelet_output_freq; }
int parser::get_realspace_output_freq() const { return realspace_output_freq; }
int parser::get_gmres_inner_iterations() const
{
  return gmres_inner_iterations;
}
int parser::get_gmres_outer_iterations() const
{
  return gmres_outer_iterations;
}

fk::vector<int> parser::get_max_adapt_levels() const
{
  return max_adapt_levels;
}

double parser::get_cfl() const { return cfl; }
double parser::get_dt() const { return dt; }
double parser::get_adapt_thresh() const { return adapt_threshold; }
double parser::get_gmres_tolerance() const { return gmres_tolerance; }

std::string parser::get_pde_string() const { return pde_str; }
std::string parser::get_solver_string() const { return solver_str; }
std::string parser::get_restart_file() const { return restart_file; }

PDE_opts parser::get_selected_pde() const { return pde_choice; }
solve_opts parser::get_selected_solver() const { return solver; }

std::string parser::get_ml_session_string() const { return matlab_name; }
int parser::get_plot_freq() const { return plot_freq; }

bool parser::is_valid() const { return valid; }

void parser_mod::set(parser &p, parser_option_entry entry, int value)
{
  switch (entry)
  {
  case degree:
    p.degree = value;
    break;
  case max_level:
    p.max_level = value;
    break;
  case mixed_grid_group:
    p.mixed_grid_group = value;
    break;
  case num_time_steps:
    p.num_time_steps = value;
    break;
  case wavelet_output_freq:
    p.wavelet_output_freq = value;
    break;
  case realspace_output_freq:
    p.realspace_output_freq = value;
    break;
  case plot_freq:
    p.plot_freq = value;
    break;
  case memory_limit:
    p.memory_limit = value;
    break;
  case gmres_inner_iterations:
    p.gmres_inner_iterations = value;
    break;
  case gmres_outer_iterations:
    p.gmres_outer_iterations = value;
    break;
  default:
    throw std::runtime_error(
        "Insider parser_mod::set, setting int to non-int entry.");
  }
}
void parser_mod::set(parser &p, parser_option_entry entry, bool value)
{
  switch (entry)
  {
  case use_implicit_stepping:
    p.use_implicit_stepping = value;
    break;
  case use_full_grid:
    p.use_full_grid = value;
    break;
  case use_linf_nrm:
    p.use_linf_nrm = value;
    break;
  case do_poisson:
    p.do_poisson = value;
    break;
  case do_adapt:
    p.do_adapt = value;
    break;
  case use_imex_stepping:
    p.use_imex_stepping = value;
    break;
  default:
    throw std::runtime_error(
        "Insider parser_mod::set, setting bool to non-bool entry.");
  }
}
void parser_mod::set(parser &p, parser_option_entry entry, double value)
{
  switch (entry)
  {
  case cfl:
    p.cfl = value;
    break;
  case dt:
    p.dt = value;
    break;
  case adapt_threshold:
    p.adapt_threshold = value;
    break;
  case gmres_tolerance:
    p.gmres_tolerance = value;
    break;
  default:
    throw std::runtime_error(
        "Insider parser_mod::set, setting double to non-double entry.");
  }
}

void parser_mod::set(parser &p, parser_option_entry entry,
                     const char *value)
{
  parser_mod::set(p, entry, std::string(value));
}

void parser_mod::set(parser &p, parser_option_entry entry,
                     std::string const &value)
{
  switch (entry)
  {
  case solver_str:
    p.solver_str = value;
    p.solver     = solver_mapping.at(value);
    break;
  case pde_str:
    p.pde_str    = value;
    p.pde_choice = pde_mapping.at(value).pde_choice;
    break;
  case starting_levels_str:
    p.starting_levels_str = value;
    break;
  case restart_file:
    p.restart_file = value;
    break;
  default:
    throw std::runtime_error(
        "Insider parser_mod::set, setting std::string to non-string entry.");
  }
}

void parser_mod::set(parser &p, parser_option_entry entry,
                     fk::vector<int> const &value)
{
  switch (entry)
  {
  case max_adapt_level:
    p.max_adapt_levels.resize(value.size()) = value;
    break;
  default:
    throw std::runtime_error("Insider parser_mod::set, setting fk::vector<int> "
                             "to non-vector entry.");
  }
}

bool options::should_output_wavelet(int const i) const
{
  return write_at_step(i, wavelet_output_freq);
}

bool options::should_output_realspace(int const i) const
{
  return write_at_step(i, realspace_output_freq);
}

bool options::should_plot(int const i) const
{
  return write_at_step(i, plot_freq);
}

bool options::write_at_step(int const i, int const freq) const
{
  expect(i >= 0);
  expect(freq >= 0);

  if (freq == 0)
  {
    return false;
  }
  if (freq == 1)
  {
    return true;
  }
  if ((i + 1) % freq == 0)
  {
    return true;
  }
  return false;
}
} // namespace asgard
