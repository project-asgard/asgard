#pragma once

#include "asgard_vector.hpp"

namespace asgard
{
// implemented solvers for implicit stepping
enum class solve_opts
{
  direct,
  gmres,
  bicgstab,
  scalapack
};

// the choices for supported PDE types
enum class PDE_opts
{
  custom = 0, // user provided pde
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
  relaxation_1x1v,
  relaxation_1x2v,
  relaxation_1x3v,
  riemann_1x2v,
  riemann_1x3v,
  collisional_landau,
  collisional_landau_1x2v,
  collisional_landau_1x3v
};

// Not very informative, maybe rename these of handle in a different wat
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
 *
 * Used by the local kronmult, global kronmult (block or non-block case)
 * will ignore these options.
 */
enum class kronmult_mode
{
  //! \brief Using a dense matrix (assumes everything is connected).
  dense,
  //! \brief Using a sparse matrix (checks for connectivity).
  sparse
};

/*!
 * \brief Type of discretization grid
 */
enum class grid_type
{
  //! Standard spars grid
  sparse,
  //! Dense grid
  dense,
  //! Dense tensor of two sparse grids
  mixed
};

namespace time_advance
{
//! types of time time advance methods, declared here to be used in the program options
enum class method
{
  imp,
  exp, // explicit is reserved keyword
  imex
};
} // namespace time_advance

enum adapt_norm
{
  l2,
  linf
};

namespace solver
{
//! indicates the values is unspecified
int constexpr novalue = -1;
//! indicates the value is unspecified
double constexpr notolerance = -1.0;
} // namespace solver

/*!
 * \internal
 * \brief Internal use only
 *
 * Takes ownership of a vector of strings and creates an associated vector
 * of string_view that can be used until this object is destroyed.
 *
 * Used in conjuction with split_argv to hold the temporary result.
 * \endinternal
 */
struct split_views
{
  split_views(std::vector<std::string> &&own)
      : own_(std::move(own))
  {
    views_.reserve(own_.size() + 1);
    views_.push_back("test");
    for (auto &s : own_)
      views_.emplace_back(s);
  }
  operator std::vector<std::string_view> const &()
  {
    return views_;
  }
  std::vector<std::string> own_;
  std::vector<std::string_view> views_;
};

/*!
 * \internal
 * \brief (testing) splits a single string into multiple strings by spaces
 *
 * The method is intended for testing where it is much easier to write
 * a single string, e.g., "-p continuity_1 -d 3 -l 4" as opposed to multiple
 * lines setting pde_choice, degree and start_levels.
 *
 * However, the parsing of the string has little to no robustness,
 * especially when it comes to passing in lists.
 *
 * The use of this method in production is strongly discouraged.
 * \endinternal
 */
split_views split_argv(std::string_view const &opts);

struct prog_opts
{
  std::string title;
  std::string subtitle;

  std::optional<adapt_norm> anorm;       // norm to use in adapt
  std::optional<bool> set_electric;      // do poisson solve for electric field
  std::optional<double> adapt_threshold; // adapt number of basis levels

  // the starting and max levels can be set per dimenision or with a single
  // value that will be applied to all dimenisons
  // empty vectors mean no user value
  std::vector<int> start_levels;
  std::vector<int> max_levels;

  std::optional<int> degree; // polynomial degree
  // number of dimensions to use for the tensor groups in mixed grid
  std::optional<grid_type> grid;
  std::optional<int> mgrid_group;
  std::optional<int> num_time_steps;

  // output frequency of wavelet or real space data
  std::optional<int> wavelet_output_freq;
  std::optional<int> realspace_output_freq;

  std::optional<time_advance::method> step_method;
  std::optional<double> dt; // time-step

  std::optional<PDE_opts> pde_choice;
  std::optional<solve_opts> solver;

  std::optional<int> memory_limit;        // local-kron only
  std::optional<kronmult_mode> kron_mode; // local-kron only

  // iterative solver section, the solvers can take tolerance and iterations
  // and gmres takes an extra parameter of outer iterations
  std::optional<double> isolver_tolerance;
  std::optional<int> isolver_iterations;
  std::optional<int> isolver_outer_iterations;

  std::string restart_file; // empty means no user value

  bool show_help     = false;
  bool show_version  = false;
  bool show_pde_help = false;

  static void print_help(std::ostream &os = std::cout);
  static void print_version_help(std::ostream &os = std::cout);
  static void print_pde_help(std::ostream &os = std::cout);

  std::vector<std::string_view> externals;

  std::string start_levels_str() const { return vect_to_str(start_levels); }
  std::string max_levels_str() const { return vect_to_str(max_levels); }

  //! create empty options, allows to manually fill the options later
  prog_opts() = default;

  //! process the command line arguments
  prog_opts(int const argc, char const *const *argv,
            bool ignore_unknown = false);

  //! mostly for testing
  prog_opts(std::vector<std::string_view> const &argv,
            bool ignore_unknown = false)
  {
    process_inputs(argv, ignore_unknown);
  }

  void print(std::ostream &os = std::cout) const;

private:
  //! mapping from cli options to variables and actions
  enum class optentry
  {
    show_help,
    version_help,
    pde_help,
    title,
    subtitle,
    grid_mode,
    step_method,
    adapt_norm,
    set_electric,
    adapt_threshold,
    start_levels,
    max_levels,
    degree,
    num_time_steps,
    wavelet_output_freq,
    realspace_output_freq,
    dt,
    pde_choice,
    solver,
    memory_limit,
    kron_mode,
    isol_tolerance,
    isol_iterations,
    isol_outer_iterations,
    restart_file,
  };

  //! not in the constructor so it can be reused when reading from file
  void process_inputs(std::vector<std::string_view> const &argv,
                      bool ignore_unknown);
  //! map pde options string to enum value
  static std::optional<PDE_opts> get_pde_opt(std::string_view const &pde_str);

  //! converts a string of ints into a vector of ints, limited to max_num_dimensions
  static std::vector<int> parse_ints(std::string const &number_string)
  {
    std::stringstream number_stream(number_string);
    std::vector<int> result;
    result.reserve(max_num_dimensions);
    while (!number_stream.eof())
    {
      std::string word;
      number_stream >> word;
      int temp_int = -1;

      // remove any leading or trailing '"'
      size_t pos = word.find_first_of('\"');
      if (pos != std::string::npos)
        word.erase(word.begin() + pos);

      if (std::stringstream(word) >> temp_int)
      {
        if (result.size() == max_num_dimensions)
        { // too many ints, return invalid result
          result.clear();
          return result;
        }
        else
          result.push_back(temp_int);
      }
    }

    return result;
  }

  //! converts vector of ints into a string
  static std::string vect_to_str(std::vector<int> const &ints)
  {
    std::string s = "";
    for (auto i : ints)
      s += ((i < 10) ? "  " : " ") + std::to_string(i);
    return s;
  }
};

/*!
 * \internal
 * \brief Makes a prog_opts object from a single sting, see split_argv
 *
 * \endinternal
 */
inline prog_opts make_opts(std::string const &cli)
{
  return prog_opts(split_argv(cli));
}

} // namespace asgard
