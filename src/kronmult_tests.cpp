#include "batch.hpp"
#include "coefficients.cpp"
#include "kronmult.hpp"
#include "tests_general.hpp"

using namespace asgard;

struct distribution_test_init
{
  distribution_test_init() { initialize_distribution(); }
  ~distribution_test_init() { finalize_distribution(); }
};

#ifdef ASGARD_USE_MPI
static distribution_test_init const distrib_test_info;
#endif

parser get_parser(PDE_opts const pde_choice,
                  fk::vector<int> const &starting_levels, int const degree,
                  int const memory_limit)
{
  return parser(pde_choice, starting_levels, degree, parser::DEFAULT_CFL,
                parser::DEFAULT_USE_FG, parser::DEFAULT_MAX_LEVEL,
                parser::DEFAULT_TIME_STEPS, parser::DEFAULT_USE_IMPLICIT,
                parser::DEFAULT_DO_ADAPT, parser::DEFAULT_ADAPT_THRESH,
                parser::DEFAULT_SOLVER_STR, parser::DEFAULT_USE_IMEX,
                memory_limit);
}

template<typename P>
void test_kronmult(parser const &parse, P const tol_factor)
{
  auto pde = make_PDE<P>(parse);
  options const opts(parse);
  basis::wavelet_transform<P, resource::host> const transformer(opts, *pde);
  generate_all_coefficients(*pde, transformer);

  // assume uniform degree across dimensions
  auto const degree = pde->get_dimensions()[0].get_degree();

  elements::table const table(opts, *pde);
  element_subgrid const my_subgrid(0, table.size() - 1, 0, table.size() - 1);

  // setup x vector
  unsigned int seed{666};
  std::mt19937 mersenne_engine(seed);
  std::uniform_int_distribution<int> dist(-4, 4);
  auto const gen = [&dist, &mersenne_engine]() {
    return dist(mersenne_engine);
  };
  auto const elem_size  = static_cast<int>(std::pow(degree, pde->num_dims));
  fk::vector<P> const x = [&table, gen, elem_size]() {
    fk::vector<P> output(elem_size * table.size());
    std::generate(output.begin(), output.end(), gen);
    return output;
  }();

  // perform kron product + gemv for gold data
  fk::vector<P> const gold = [&pde, &table, &my_subgrid, x, elem_size]() {
    auto const system_size = elem_size * table.size();
    fk::matrix<P> A(system_size, system_size);
    build_system_matrix(*pde, table, A, my_subgrid);
    return A * x;
  }();

  // perform kronmult using ed's library
  std::cout.setstate(std::ios_base::failbit); // shhh...don't print alloc info
  auto const fx = kronmult::execute(*pde, table, opts, my_subgrid, x);
  std::cout.clear();

  rmse_comparison(gold, fx, tol_factor);
}

TEMPLATE_TEST_CASE("test kronmult", "[kronmult]", float, double)
{
  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("1d")
  {
    auto const pde_choice        = PDE_opts::continuity_1;
    auto const degree            = 2;
    auto const levels            = fk::vector<int>{3};
    auto const workspace_size_MB = 1000;
    parser const test_parse =
        get_parser(pde_choice, levels, degree, workspace_size_MB);
    test_kronmult(test_parse, tol_factor);
  }

  SECTION("2d - uniform level")
  {
    auto const pde_choice        = PDE_opts::continuity_2;
    auto const degree            = 3;
    auto const levels            = fk::vector<int>{2, 2};
    auto const workspace_size_MB = 1000;
    parser const test_parse =
        get_parser(pde_choice, levels, degree, workspace_size_MB);
    test_kronmult(test_parse, tol_factor);
  }
  SECTION("2d - non-uniform level")
  {
    auto const pde_choice        = PDE_opts::continuity_2;
    auto const degree            = 3;
    auto const levels            = fk::vector<int>{3, 2};
    auto const workspace_size_MB = 1000;
    parser const test_parse =
        get_parser(pde_choice, levels, degree, workspace_size_MB);
    test_kronmult(test_parse, tol_factor);
  }

  SECTION("6d - uniform level")
  {
    auto const pde_choice        = PDE_opts::continuity_6;
    auto const degree            = 2;
    auto const levels            = fk::vector<int>{2, 2, 2, 2, 2, 2};
    auto const workspace_size_MB = 1000;
    parser const test_parse =
        get_parser(pde_choice, levels, degree, workspace_size_MB);
    test_kronmult(test_parse, tol_factor);
  }

  SECTION("6d - non-uniform level")
  {
    auto const pde_choice        = PDE_opts::continuity_6;
    auto const degree            = 1;
    auto const levels            = fk::vector<int>{2, 2, 2, 3, 2, 2};
    auto const workspace_size_MB = 1000;
    parser const test_parse =
        get_parser(pde_choice, levels, degree, workspace_size_MB);
    test_kronmult(test_parse, tol_factor);
  }
}

TEMPLATE_TEST_CASE("test kronmult w/ decompose", "[kronmult]", float, double)
{
  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("2d - uniform level")
  {
    auto const pde_choice        = PDE_opts::continuity_2;
    auto const degree            = 2;
    auto const levels            = fk::vector<int>{6, 6};
    auto const workspace_size_MB = 80; // small enough to decompose the problem
    parser const test_parse =
        get_parser(pde_choice, levels, degree, workspace_size_MB);
    test_kronmult(test_parse, tol_factor);
  }

  SECTION("2d - non-uniform level")
  {
    auto const pde_choice        = PDE_opts::continuity_2;
    auto const degree            = 2;
    auto const levels            = fk::vector<int>{6, 5};
    auto const workspace_size_MB = 80; // small enough to decompose the problem
    parser const test_parse =
        get_parser(pde_choice, levels, degree, workspace_size_MB);
    test_kronmult(test_parse, tol_factor);
  }

  SECTION("6d - uniform level")
  {
    auto const pde_choice        = PDE_opts::continuity_6;
    auto const degree            = 2;
    auto const levels            = fk::vector<int>{2, 2, 2, 2, 2, 2};
    auto const workspace_size_MB = 80; // small enough to decompose the problem
    parser const test_parse =
        get_parser(pde_choice, levels, degree, workspace_size_MB);
    test_kronmult(test_parse, tol_factor);
  }
}
