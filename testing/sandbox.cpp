#include "batch.hpp"

#include "build_info.hpp"
#include "coefficients.hpp"
#include "distribution.hpp"
#include "elements.hpp"
#include "tools.hpp"

#ifdef ASGARD_IO_HIGHFIVE
#include "io.hpp"
#endif

#ifdef ASGARD_USE_MPI
#include <mpi.h>
#endif

#include "pde.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include "time_advance.hpp"
#include "transformations.hpp"
#include <numeric>

using namespace asgard;

#ifdef ASGARD_USE_DOUBLE_PREC
using prec = double;
#else
using prec = float;
#endif

static auto constexpr num_steps = 5;

static inline const std::filesystem::path gold_base_dir{ASGARD_GOLD_BASE_DIR};

static auto const time_advance_base_dir = gold_base_dir / "time_advance";

template<typename P>
constexpr P get_tolerance(int ulp)
{
  return std::numeric_limits<P>::epsilon()*ulp;
}

template<typename P, asgard::mem_type mem, asgard::mem_type omem>
void rmse_comparison(asgard::fk::vector<P, mem> const &v0,
                     asgard::fk::vector<P, omem> const &v1, P const tolerance)
{
  auto const diff_norm = asgard::fm::nrm2(v0 - v1);

  auto const abs_compare = [](auto const a, auto const b) {
    return (std::abs(a) < std::abs(b));
  };

  auto const max = std::max(
      static_cast<P>(1.0),
      std::max(std::abs(*std::max_element(v0.begin(), v0.end(), abs_compare)),
               std::abs(*std::max_element(v1.begin(), v1.end(), abs_compare))));
  //Catch::StringMaker<P>::precision = 20;
  //REQUIRE((diff_norm / max) < (tolerance * std::sqrt(v0.size())));
  std::cerr << " diff " << ((diff_norm / max) < (tolerance * std::sqrt(v0.size()))) << "\n";
}

template<typename P>
void time_advance_test(parser const &parse,
                       std::filesystem::path const &filepath,
                       P const tolerance_factor)
{
  auto const num_ranks = get_num_ranks();
  if (num_ranks > 1 && parse.using_implicit() &&
      parse.get_selected_solver() != solve_opts::scalapack)
  {
    // distributed implicit stepping not implemented
    return;
  }

  if (num_ranks == 1 && parse.get_selected_solver() == solve_opts::scalapack)
  {
    // don't bother using scalapack with 1 rank
    return;
  }

  auto pde = make_PDE<P>(parse);
  options const opts(parse);
  elements::table const check(opts, *pde);
  if (check.size() <= num_ranks)
  {
    // don't run tiny problems when MPI testing
    return;
  }
  adapt::distributed_grid adaptive_grid(*pde, opts);
  basis::wavelet_transform<P, resource::host> const transformer(opts, *pde);

  // -- compute dimension mass matrices
  generate_dimension_mass_mat(*pde, transformer);

  // -- set coeffs
  generate_all_coefficients(*pde, transformer);

  // -- generate initial condition vector.
  auto const initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);

  // TODO: look into issue requiring mass mats to be regenerated after init
  // cond. see problem in main.cpp
  generate_dimension_mass_mat(*pde, transformer);

  fk::vector<P> f_val(initial_condition);

  std::cerr << " HERE " << std::endl;
  asgard::kronmult_matrix<P> operator_matrix;
  std::cerr << " HERE " << std::endl;

  // -- time loop
  for (auto i = 0; i < opts.num_time_steps; ++i)
  {
    std::cout.setstate(std::ios_base::failbit);
    auto const time          = i * pde->get_dt();
    auto const update_system = i == 0;
    auto const method = opts.use_implicit_stepping ? time_advance::method::imp
                                                   : time_advance::method::exp;
    auto const sol    = time_advance::adaptive_advance(
        method, *pde, operator_matrix, adaptive_grid, transformer, opts, f_val,
        time, update_system);

    f_val.resize(sol.size()) = sol;
    std::cout.clear();

    auto const file_path =
        filepath.parent_path() /
        (filepath.filename().string() + std::to_string(i) + ".dat");
    auto const gold = fk::vector<P>(read_vector_from_txt_file(file_path));

    // each rank generates partial answer
    auto const dof =
        static_cast<int>(std::pow(parse.get_degree(), pde->num_dims));
    auto const subgrid = adaptive_grid.get_subgrid(get_rank());
    //REQUIRE((subgrid.col_stop + 1) * dof - 1 <= gold.size());
    auto const my_gold = fk::vector<P, mem_type::const_view>(
        gold, subgrid.col_start * dof, (subgrid.col_stop + 1) * dof - 1);
    rmse_comparison(my_gold, f_val, tolerance_factor);
  }
}

static std::string get_level_string(fk::vector<int> const &levels)
{
  return std::accumulate(levels.begin(), levels.end(), std::string(),
                         [](std::string const &accum, int const lev) {
                           return accum + std::to_string(lev) + "_";
                         });
}

// The parser is constructed in one of 5 patterns,
// each is covered by a make method.
// All defaults are assumed automatically, only the adjusted variables are
// modified.
parser make_basic_parser(std::string const &pde_choice,
                         fk::vector<int> starting_levels, int const degree,
                         double const cfl, bool full_grid,
                         int const num_time_steps)
{
  parser parse(pde_choice, starting_levels);
  parser_mod::set(parse, parser_mod::degree, degree);
  parser_mod::set(parse, parser_mod::use_full_grid, full_grid);
  parser_mod::set(parse, parser_mod::num_time_steps, num_time_steps);
  parser_mod::set(parse, parser_mod::cfl, cfl);
  return parse;
}
parser make_basic_parser(std::string const &pde_choice,
                         fk::vector<int> starting_levels, int const degree,
                         double const cfl, bool full_grid,
                         int const num_time_steps, bool use_implicit)
{
  parser parse = make_basic_parser(pde_choice, starting_levels, degree, cfl,
                                   full_grid, num_time_steps);
  parser_mod::set(parse, parser_mod::use_implicit_stepping, use_implicit);
  return parse;
}
parser make_basic_parser(std::string const &pde_choice,
                         fk::vector<int> starting_levels, int const degree,
                         double const cfl, bool full_grid,
                         int const num_time_steps, bool use_implicit,
                         std::string const &solver_str)
{
  parser parse = make_basic_parser(pde_choice, starting_levels, degree, cfl,
                                   full_grid, num_time_steps, use_implicit);
  parser_mod::set(parse, parser_mod::solver_str, solver_str);
  return parse;
}
parser make_basic_parser(std::string const &pde_choice,
                         fk::vector<int> starting_levels, int const degree,
                         double const cfl, bool full_grid,
                         int const num_time_steps, bool use_implicit,
                         bool do_adapt_levels, double adapt_threshold)
{
  parser parse = make_basic_parser(pde_choice, starting_levels, degree, cfl,
                                   full_grid, num_time_steps, use_implicit);
  parser_mod::set(parse, parser_mod::do_adapt, do_adapt_levels);
  parser_mod::set(parse, parser_mod::adapt_threshold, adapt_threshold);
  return parse;
}
parser make_basic_parser(std::string const &pde_choice,
                         fk::vector<int> starting_levels, int const degree,
                         double const cfl, bool full_grid,
                         int const num_time_steps, bool use_implicit,
                         bool do_adapt_levels, double adapt_threshold,
                         std::string const &solver_str)
{
  parser parse = make_basic_parser(pde_choice, starting_levels, degree, cfl,
                                   full_grid, num_time_steps, use_implicit,
                                   do_adapt_levels, adapt_threshold);
  parser_mod::set(parse, parser_mod::solver_str, solver_str);
  return parse;
}

int main(int, char**)
{

  prec const cfl           = 0.01;
  std::string const pde_choice = "diffusion_2";
  int const num_dims           = 2;

  //SECTION("diffusion2, explicit, sparse grid, level 2, degree 2")
  //{
    int const degree = 2;
    int const level  = 2;

    auto constexpr tol_factor = get_tolerance<prec>(100);

    auto const gold_base = time_advance_base_dir / "diffusion2_sg_l2_d2_t";

    auto const full_grid = false;
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  //}


  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior
  return 0;
}
