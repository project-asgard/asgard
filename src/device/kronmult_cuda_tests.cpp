#include "../distribution.hpp"
#include "../element_table.hpp"
#include "kronmult_cuda.hpp"
#include "tests_general.hpp"

template<typename P>
void test_kronmult_staging(int const num_elems, int const num_copies)
{
  std::random_device rd;
  std::mt19937 mersenne_engine(rd());
  std::uniform_real_distribution<P> dist(-4.0, 4.0);
  auto const gen = [&dist, &mersenne_engine]() {
    return dist(mersenne_engine);
  };

  fk::vector<P, mem_type::owner, resource::device> const x = [gen,
                                                              num_elems]() {
    fk::vector<P> x(num_elems);
    std::generate(x.begin(), x.end(), gen);
    return x.clone_onto_device();
  }();
  fk::vector<P> const gold(x.clone_onto_host());

  fk::vector<P, mem_type::owner, resource::device> dest(x.size() * num_copies);

  stage_inputs_kronmult(x.data(), dest.data(), x.size(), num_copies);

  for (auto i = 0; i < num_copies; ++i)
  {
    fk::vector<P, mem_type::const_view, resource::device> const dest_view(
        dest, i * x.size(), (i + 1) * x.size() - 1);
    fk::vector<P> const test(dest_view.clone_onto_host());
    REQUIRE(test == gold);
  }
}

TEMPLATE_TEST_CASE("staging kernel", "[kronmult_cuda]", float, double)
{
  SECTION("1 elems, 1 copies")
  {
    auto const num_elems  = 1;
    auto const num_copies = 1;
    test_kronmult_staging<TestType>(num_elems, num_copies);
  }

  SECTION("1001 elems, 1 copies")
  {
    auto const num_elems  = 1001;
    auto const num_copies = 1;
    test_kronmult_staging<TestType>(num_elems, num_copies);
  }

  SECTION("1 elems, 500 copies")
  {
    auto const num_elems  = 1;
    auto const num_copies = 500;
    test_kronmult_staging<TestType>(num_elems, num_copies);
  }

  SECTION("101010 elems, 555 copies")
  {
    auto const num_elems  = 101010;
    auto const num_copies = 555;
    test_kronmult_staging<TestType>(num_elems, num_copies);
  }
}

// sanity check, ensures lists elements are all assigned
// and in correct range

template<typename P>
void test_kronmult_build(PDE<P> const &pde)
{
  auto const degree     = pde.get_dimensions()[0].get_degree();
  auto const level      = pde.get_dimensions()[0].get_level();
  auto const deg_to_dim = static_cast<int>(std::pow(degree, pde.num_dims));

  std::vector<std::string> const args = {"-l", std::to_string(level), "-d",
                                         std::to_string(degree)};
  options const o                     = make_options(args);
  element_table const table(o, pde.num_dims);
  element_subgrid const my_subgrid(0, table.size() - 1, 0, table.size() - 1);

  P *element_x;
  P *element_work;
  auto const workspace_size = my_subgrid.size() * deg_to_dim * pde.num_terms;
  fk::allocate_device(element_x, workspace_size);
  fk::allocate_device(element_work, workspace_size);

  auto const total_kronmults = my_subgrid.size() * pde.num_terms;
  P **input_ptrs;
  P **work_ptrs;
  P **output_ptrs;
  P **operator_ptrs;
  fk::allocate_device(input_ptrs, total_kronmults);
  fk::allocate_device(work_ptrs, total_kronmults);
  fk::allocate_device(output_ptrs, total_kronmults);
  fk::allocate_device(operator_ptrs, total_kronmults * pde.num_dims);

  fk::vector<P *> const operators = [&pde] {
    fk::vector<P *> builder(pde.num_terms * pde.num_dims);
    for (int i = 0; i < pde.num_terms; ++i)
    {
      for (int j = 0; j < pde.num_dims; ++j)
      {
        builder(i * pde.num_dims + j) = pde.get_coefficients(i, j).data();
      }
    }
    return builder;
  }();

  fk::vector<P *, mem_type::owner, resource::device> const operators_d(
      operators.clone_onto_device());

  auto const lda = pde.get_coefficients(0, 0)
                       .stride(); // leading dimension of coefficient matrices

  fk::vector<P, mem_type::owner, resource::device> fx(my_subgrid.nrows() *
                                                      deg_to_dim);
  prepare_kronmult(table.get_device_table().data(), operators_d.data(), lda,
                   element_x, element_work, fx.data(), operator_ptrs, work_ptrs,
                   input_ptrs, output_ptrs, degree, pde.num_terms, pde.num_dims,
                   my_subgrid.row_start, my_subgrid.row_stop,
                   my_subgrid.col_start, my_subgrid.col_stop);

  P **const input_ptrs_h    = new P *[total_kronmults];
  P **const work_ptrs_h     = new P *[total_kronmults];
  P **const output_ptrs_h   = new P *[total_kronmults];
  P **const operator_ptrs_h = new P *[total_kronmults * pde.num_dims];

  fk::copy_to_host(input_ptrs_h, input_ptrs, total_kronmults);
  fk::copy_to_host(work_ptrs_h, work_ptrs, total_kronmults);
  fk::copy_to_host(output_ptrs_h, output_ptrs, total_kronmults);
  fk::copy_to_host(operator_ptrs_h, operator_ptrs,
                   total_kronmults * pde.num_dims);

  auto const in_range = [](auto const ptr, auto const start, auto const end) {
    return (ptr >= start && ptr <= end);
  };

  for (int64_t row = 0; row < my_subgrid.nrows(); ++row)
  {
    for (int64_t col = 0; col < my_subgrid.ncols(); ++col)
    {
      for (int64_t t = 0; t < pde.num_terms; ++t)
      {
        auto const num_kron =
            my_subgrid.to_local_row(row) * my_subgrid.ncols() * pde.num_terms +
            my_subgrid.to_local_col(col) * pde.num_terms + t;

        REQUIRE(in_range(input_ptrs_h[num_kron], element_x,
                         element_x + workspace_size - 1));
        REQUIRE(in_range(work_ptrs_h[num_kron], element_work,
                         element_work + workspace_size - 1));
        REQUIRE(in_range(output_ptrs_h[num_kron], fx.data(),
                         fx.data() + fx.size() - 1));

        auto const operator_start = num_kron * pde.num_dims;

        for (int d = 0; d < pde.num_dims; ++d)
        {
          auto const &coeff = pde.get_coefficients(t, d);
          REQUIRE(in_range(operator_ptrs_h[operator_start + d], coeff.data(),
                           coeff.data() + coeff.size()));
        }
      }
    }
  }

  fk::delete_device(element_x);
  fk::delete_device(element_work);

  fk::delete_device(input_ptrs);
  fk::delete_device(work_ptrs);
  fk::delete_device(output_ptrs);
  fk::delete_device(operator_ptrs);

  delete[] input_ptrs_h;
  delete[] work_ptrs_h;
  delete[] output_ptrs_h;
  delete[] operator_ptrs_h;
}

TEMPLATE_TEST_CASE("list building kernel", "[kronmult_cuda]", float, double)
{
  SECTION("1d, small")
  {
    auto const level  = 2;
    auto const degree = 1;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    test_kronmult_build(*pde);
  }
  SECTION("1d, large")
  {
    auto const level  = 6;
    auto const degree = 5;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    test_kronmult_build(*pde);
  }
  SECTION("2d, small")
  {
    auto const level  = 3;
    auto const degree = 2;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    test_kronmult_build(*pde);
  }
  SECTION("2d, large")
  {
    auto const level  = 5;
    auto const degree = 7;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    test_kronmult_build(*pde);
  }
  SECTION("3d, small")
  {
    auto const level  = 3;
    auto const degree = 4;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    test_kronmult_build(*pde);
  }
  SECTION("6d, small")
  {
    auto const level  = 2;
    auto const degree = 2;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
    test_kronmult_build(*pde);
  }
}
