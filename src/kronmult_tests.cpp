#include "batch.hpp"
#include "chunk.hpp"
#include "coefficients.cpp"
#include "kronmult.hpp"
#include "tests_general.hpp"

struct distribution_test_init
{
  distribution_test_init() { initialize_distribution(); }
  ~distribution_test_init() { finalize_distribution(); }
};

#ifdef ASGARD_USE_MPI
static distribution_test_init const distrib_test_info;
#endif

template<typename P>
void test_kronmult(PDE<P> &pde, int const workspace_size_MB, P const tol_factor)
{
  // assume uniform level and degree across dimensions
  auto const level  = pde.get_dimensions()[0].get_level();
  auto const degree = pde.get_dimensions()[0].get_degree();

  // setup problem
  std::vector<std::string> const args = {"-l", std::to_string(level), "-d",
                                         std::to_string(degree)};

  options const o = make_options(args);
  element_table const table(o, level, pde.num_dims);
  element_subgrid const my_subgrid(0, table.size() - 1, 0, table.size() - 1);
  basis::wavelet_transform<P, resource::host> const transformer(level, degree);
  generate_all_coefficients(pde, transformer);

  // setup x vector
  std::random_device rd;
  std::mt19937 mersenne_engine(rd());
  std::uniform_int_distribution<int> dist(-4, 4);
  auto const gen = [&dist, &mersenne_engine]() {
    return dist(mersenne_engine);
  };
  auto const elem_size  = static_cast<int>(std::pow(degree, pde.num_dims));
  fk::vector<P> const x = [&table, gen, elem_size]() {
    fk::vector<P> x(elem_size * table.size());
    std::generate(x.begin(), x.end(), gen);
    return x;
  }();

  // perform kron product + gemv for gold data
  fk::vector<P> const gold = [&pde, &table, x, elem_size]() {
    auto const system_size = elem_size * table.size();
    fk::matrix<P> A(system_size, system_size);

    element_chunk const chunk = [&table]() {
      element_chunk chunk;
      for (int i = 0; i < table.size(); ++i)
      {
        chunk.insert({i, grid_limits(0, table.size() - 1)});
      }
      return chunk;
    }();

    build_system_matrix(pde, table, chunk, A);
    return A * x;
  }();

  // perform kronmult using ed's library
  std::cout.setstate(std::ios_base::failbit); // shhh...don't print alloc info
  auto const fx =
      kronmult::execute(pde, table, my_subgrid, workspace_size_MB, x);
  std::cout.clear();

  rmse_comparison(gold, fx, tol_factor);
}

TEMPLATE_TEST_CASE("test kronmult", "[kronmult]", float, double)
{
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-15 : 1e-7;

  SECTION("1d")
  {
    auto const degree = 4;
    auto const level  = 3;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    auto const workspace_size_MB = 1000;
    test_kronmult(*pde, workspace_size_MB, tol_factor);
  }

  SECTION("2d")
  {
    auto const degree = 3;
    auto const level  = 2;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    auto const workspace_size_MB = 1000;
    test_kronmult(*pde, workspace_size_MB, tol_factor);
  }

  SECTION("6d")
  {
    auto const degree = 2;
    auto const level  = 2;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
    auto const workspace_size_MB = 1000;
    test_kronmult(*pde, workspace_size_MB, tol_factor);
  }
}

TEMPLATE_TEST_CASE("test kronmult w/ decompose", "[kronmult]", float, double)
{
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-15 : 1e-7;

  SECTION("2d")
  {
    auto const degree = 6;
    auto const level  = 6;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    auto const workspace_size_MB =
        30; // small enough so that the problem is decomposed x2
    test_kronmult(*pde, workspace_size_MB, tol_factor);
  }

  SECTION("6d")
  {
    auto const degree = 2;
    auto const level  = 2;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
    auto const workspace_size_MB =
        100; // small enough so that the problem is decomposed x4
    test_kronmult(*pde, workspace_size_MB, tol_factor);
  }
}
