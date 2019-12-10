#include "program_options.hpp"
#include "build_info.hpp"
#include "clara.hpp"
#include "distribution.hpp"
#include <iostream>

options::options(int argc, char **argv)
{
  bool show_help = false;

  // Parsing...
  auto cli =
      clara::detail::Help(show_help) |
      clara::detail::Opt(cfl, "cfl")["-c"]["--cfl"](
          "the Courant-Friedrichs-Lewy (CFL) condition") |
      clara::detail::Opt(degree, "degree")["-d"]["--degree"](
          "Terms in legendre basis polynomials") |
      clara::detail::Opt(use_full_grid)["-f"]["--full_grid"](
          "Use full grid (vs. sparse grid)") |
      clara::detail::Opt(use_implicit_stepping)["-i"]["--implicit"](
          "Use implicit time advance (vs. explicit)") |
      clara::detail::Opt(level, "level")["-l"]["--level"](
          "Hierarchical levels (resolution)") |
      clara::detail::Opt(num_time_steps, "time steps")["-n"]["--num_steps"](
          "Number of iterations") |
      clara::detail::Opt(selected_pde, "selected_pde")["-p"]["--pde"](
          "PDE to solve; see pde.hpp for list") |
      clara::detail::Opt(do_poisson)["-s"]["--solve_poisson"](
          "Do poisson solve for electric field") |
      clara::detail::Opt(write_frequency,
                         "write_frequency")["-w"]["--write_freq"](
          "Frequency in steps for writing wavelet space "
          "output") |
      clara::detail::Opt(realspace_output_freq,
                         "realspace_output_freq")["-r"]["--real_freq"](
          "Frequency in steps for writing realspace output");

  auto result = cli.parse(clara::detail::Args(argc, argv));
  if (!result)
  {
    std::cerr << "Error in command line parsing: " << result.errorMessage()
              << '\n';
    valid = false;
  }
  if (show_help)
  {
    std::cerr << cli << '\n';
    exit(0);
  }

  // Validation...
  if (cfl <= 0.0)
  {
    std::cerr << "CFL must be positive" << '\n';
    valid = false;
  }
  if (degree < 1 && degree != -1)
  {
    std::cerr << "Degree must be a natural number" << '\n';
    valid = false;
  }
  if (level < 1 && level != -1)
  {
    std::cerr << "Level must be a natural number" << '\n';
    valid = false;
  }
  if (num_time_steps < 1)
  {
    std::cerr << "Number of timesteps must be a natural number" << '\n';
    valid = false;
  }

  auto const choice = pde_mapping.find(selected_pde);
  if (choice == pde_mapping.end())
  {
    std::cerr << "Invalid pde choice; see pde.hpp for valid choices"
              << '\n';
    valid = false;
  }
  else
  {
    pde_choice = pde_mapping.at(selected_pde);
  }

  if (realspace_output_freq < 0 || write_frequency < 0)
  {
    std::cerr << "Frequencies must be non-negative: " << '\n';
    valid = false;
  }

#ifdef ASGARD_USE_CUDA
  if (use_implicit_stepping)
  {
    std::cerr << "GPU acceleration not implemented for implicit stepping"
              << '\n';
    valid = false;
  }
#endif

#ifdef ASGARD_USE_MPI
  if (use_implicit_stepping && get_num_ranks() > 1)
  {
    std::cerr << "Distribution not implemented for implicit stepping" << '\n';
    valid = false;
  }
  if (realspace_output_freq > 0)
  {
    std::cerr << "Distribution does not yet support realspace transform"
              << '\n';
    valid = false;
  }
#endif
}

int options::get_level() const { return level; }
int options::get_degree() const { return degree; }
int options::get_time_steps() const { return num_time_steps; }
int options::get_write_frequency() const { return write_frequency; }
bool options::using_implicit() const { return use_implicit_stepping; }
bool options::using_full_grid() const { return use_full_grid; }
double options::get_cfl() const { return cfl; }
PDE_opts options::get_selected_pde() const { return pde_choice; }
std::string options::get_pde_string() const { return selected_pde; }
bool options::is_valid() const { return valid; }
bool options::do_poisson_solve() const { return do_poisson; }
int options::get_realspace_output_freq() const { return realspace_output_freq; }

bool options::write_at_step(int const i) const
{
  assert(i >= 0);

  if (write_frequency == 0)
  {
    return false;
  }
  if (i % write_frequency == 0)
  {
    return true;
  }
  return false;
}

bool options::transform_at_step(int const i) const
{
  assert(i >= 0);

  if (realspace_output_freq == 0)
  {
    return false;
  }
  if (i % realspace_output_freq == 0)
  {
    return true;
  }
  return false;
}
