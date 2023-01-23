#include "batch.hpp"
#include "coefficients.cpp"
#include "kronmult.hpp"
#include "solver.hpp"
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
  fk::vector<P> const b = [&table, gen, elem_size]() {
    fk::vector<P> output(elem_size * table.size());
    std::generate(output.begin(), output.end(), gen);
    return output;
  }();

  fk::vector<P> const gold = [&pde, &table, &my_subgrid, &b, elem_size]() {
    auto const system_size = elem_size * table.size();
    fk::matrix<P> A(system_size, system_size);
    fk::vector<P> x(b);
    build_system_matrix(*pde, table, A, my_subgrid);
    // AA = I - dt*A;
    fm::scal(P{-1.0} * pde->get_dt(), A);
    for (int i = 0; i < A.nrows(); ++i)
    {
      A(i, i) += 1.0;
    }
    std::vector<int> ipiv(A.nrows());
    fm::gesv(A, x, ipiv);
    return x;
  }();

  // perform gmres with system matrix A
  fk::vector<P> const gmres = [&pde, &table, &my_subgrid, &gold, &b,
                               elem_size]() {
    auto const system_size = elem_size * table.size();
    fk::matrix<P> A(system_size, system_size);
    fk::vector<P> x(gold);
    int const restart  = A.ncols();
    int const max_iter = A.ncols();
    P const tolerance  = std::is_same_v<float, P> ? 1e-6 : 1e-12;
    build_system_matrix(*pde, table, A, my_subgrid);
    // AA = I - dt*A;
    fm::scal(P{-1.0} * pde->get_dt(), A);
    for (int i = 0; i < A.nrows(); ++i)
    {
      A(i, i) += 1.0;
    }
    solver::simple_gmres(A, x, b, fk::matrix<P>(), restart, max_iter,
                         tolerance);
    return x;
  }();

  // perform gmres with kron product
  fk::vector<P> const gmres_matrix_free = [&pde, &table, &my_subgrid, &gold, &b,
                                           elem_size, &opts]() {
    auto const system_size = elem_size * table.size();
    fk::vector<P> x(gold);
    int const restart  = system_size;
    int const max_iter = system_size;
    P const tolerance  = std::is_same_v<float, P> ? 1e-6 : 1e-12;
    solver::simple_gmres(*pde, table, opts, my_subgrid, x, b, fk::matrix<P>(),
                         restart, max_iter, tolerance);
    return x;
  }();

  rmse_comparison(gold, gmres, tol_factor);
  rmse_comparison(gold, gmres_matrix_free, tol_factor);
}

TEMPLATE_TEST_CASE("simple GMRES", "[solver]", float, double)
{
  fk::matrix<TestType> const A_gold{
      {3.383861628748717e+00, 1.113343240310116e-02, 2.920740795411032e+00},
      {3.210305545769361e+00, 3.412141162288144e+00, 3.934494120167269e+00},
      {1.723479266939425e+00, 1.710451084172946e+00, 4.450671104482062e+00}};

  fk::matrix<TestType> const precond{{3.383861628748717e+00, 0.0, 0.0},
                                     {0.0, 3.412141162288144e+00, 0.0},
                                     {0.0, 0.0, 4.450671104482062e+00}};

  fk::vector<TestType> const b_gold{
      2.084406360034887e-01, 6.444769305362776e-01, 3.687335330031937e-01};

  fk::vector<TestType> const x_gold{
      4.715561567725287e-02, 1.257695999382253e-01, 1.625351700791827e-02};

  fk::vector<TestType> const b_gold_2{
      9.789303188021963e-01, 8.085725142873675e-01, 7.370498473207234e-01};
  fk::vector<TestType> const x_gold_2{
      1.812300946484165e-01, -7.824949213916167e-02, 1.254969087137521e-01};

  SECTION("gmres test case 1")
  {
    fk::vector<TestType> test(x_gold.size());

    std::cout.setstate(std::ios_base::failbit);
    TestType const error = solver::simple_gmres(
        A_gold, test, b_gold, fk::matrix<TestType>(), A_gold.ncols(),
        A_gold.ncols(), std::numeric_limits<TestType>::epsilon());
    std::cout.clear();
    REQUIRE(error < std::numeric_limits<TestType>::epsilon());
    REQUIRE(test == x_gold);
  }

  SECTION("test case 1, point jacobi preconditioned")
  {
    fk::vector<TestType> test(x_gold.size());

    std::cout.setstate(std::ios_base::failbit);
    TestType const error = solver::simple_gmres(
        A_gold, test, b_gold, precond, A_gold.ncols(), A_gold.ncols(),
        std::numeric_limits<TestType>::epsilon());
    std::cout.clear();
    REQUIRE(error < std::numeric_limits<TestType>::epsilon());
    REQUIRE(test == x_gold);
  }

  SECTION("gmres test case 2")
  {
    fk::vector<TestType> test(x_gold_2.size());

    std::cout.setstate(std::ios_base::failbit);
    TestType const error = solver::simple_gmres(
        A_gold, test, b_gold_2, fk::matrix<TestType>(), A_gold.ncols(),
        A_gold.ncols(), std::numeric_limits<TestType>::epsilon());
    std::cout.clear();
    REQUIRE(error < std::numeric_limits<TestType>::epsilon());
    REQUIRE(test == x_gold_2);
  }

  SECTION("test case 2, point jacobi preconditioned")
  {
    fk::vector<TestType> test(x_gold_2.size());
    std::cout.setstate(std::ios_base::failbit);
    TestType const error = solver::simple_gmres(
        A_gold, test, b_gold_2, precond, A_gold.ncols(), A_gold.ncols(),
        std::numeric_limits<TestType>::epsilon());
    std::cout.clear();
    REQUIRE(error < std::numeric_limits<TestType>::epsilon());
    REQUIRE(test == x_gold_2);
  }
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