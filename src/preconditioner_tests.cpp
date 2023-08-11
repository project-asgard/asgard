#include "coefficients.hpp"
#include "fast_math.hpp"
#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "preconditioner.hpp"
#include "tests_general.hpp"
#include "time_advance.hpp"

using namespace asgard;

TEMPLATE_TEST_CASE("preconditioner interface", "[precond]", test_precs)
{
  SECTION("default constructor")
  {
    preconditioner::preconditioner<TestType> const test;
    REQUIRE(test.empty());
    REQUIRE((test.factored() == false));
  }

  SECTION("constructor from existing dense M matrix")
  {
    fk::matrix<TestType> M = asgard::eye<TestType>(5);
    preconditioner::preconditioner<TestType> const test(std::move(M));
    REQUIRE((test.get_matrix() == asgard::eye<TestType>(5)));
    REQUIRE((test.factored() == false));
  }
}

TEMPLATE_TEST_CASE("block jacobi - relaxation 1x1v", "[precond]", test_precs)
{
  std::string const pde_choice = "relaxation_1x1v";
  fk::vector<int> const levels{0, 4};
  int const degree = 3;

  TestType constexpr tol_factor = get_tolerance<TestType>(100);
  TestType constexpr gmres_tol =
      std::is_same_v<TestType, double> ? 1.0e-10 : 1.0e-6;

  parser parse(pde_choice, levels);
  parser_mod::set(parse, parser_mod::degree, degree);
  parser_mod::set(parse, parser_mod::dt, 5.0e-4);
  parser_mod::set(parse, parser_mod::use_imex_stepping, true);
  parser_mod::set(parse, parser_mod::use_full_grid, true);
  parser_mod::set(parse, parser_mod::gmres_tolerance, gmres_tol);

  auto const pde = make_PDE<TestType>(parse);

  options const opts(parse);
  elements::table const check(opts, *pde);

  adapt::distributed_grid adaptive_grid(*pde, opts);
  basis::wavelet_transform<TestType, resource::host> const transformer(opts,
                                                                       *pde);

  // -- compute dimension mass matrices
  generate_dimension_mass_mat(*pde, transformer);

  // -- set coeffs
  generate_all_coefficients(*pde, transformer);

  // -- generate moments
  for (auto &m : pde->moments)
  {
    m.createFlist(*pde, opts);
    expect(m.get_fList().size() > 0);

    m.createMomentVector(*pde, parse, adaptive_grid.get_table());
    expect(m.get_vector().size() > 0);
  }

  // -- generate initial condition vector.
  auto const initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);

  generate_dimension_mass_mat(*pde, transformer);

  fk::vector<TestType> f_val(initial_condition);
  asgard::matrix_list<TestType> operator_matrices;

  // do one time step to make sure PDE params/moments are updated
  std::cout.setstate(std::ios_base::failbit);
  fk::vector<TestType> const sol = time_advance::adaptive_advance(
      asgard::time_advance::method::imex, *pde, operator_matrices,
      adaptive_grid, transformer, opts, f_val, TestType{0.0}, true);
  std::cout.clear();

  SECTION("construction")
  {
    // Tests the block jacobi construction from PDE coefficients against
    // diagonal blocks of the A matrix
    auto precond = preconditioner::block_jacobi_preconditioner<TestType>();

    // Construct preconditioner matrix from PDE coefficients
    precond.construct(*pde, adaptive_grid.get_table(), sol.size(),
                      pde->get_dt(), imex_flag::imex_implicit);

    size_t const num_blocks =
        static_cast<size_t>(adaptive_grid.get_table().size());
    REQUIRE((precond.precond_blks.size() == num_blocks));
    REQUIRE((precond.blk_pivots.size() == num_blocks));
    REQUIRE((precond.factored() == false));

    int const dof = sol.size();
    auto &mat     = operator_matrices[matrix_entry::imex_implicit];
    // Construct Matrix A by calling kronmult with identity to back-out the A
    // matrix one column at a time
    fk::matrix<TestType> A(dof, dof);
    fk::vector<TestType> kron_x(dof);
    fk::vector<TestType> kron_y(dof);
    for (int col = 0; col < dof; col++)
    {
      // set current row to identity
      kron_x(col) = 1.0;
      if (col > 0)
      {
        // flip prev row value back to 0.
        kron_x(col - 1) = 0.0;
      }

      mat.apply(TestType{1.0}, kron_x.data(), TestType{0.0}, kron_y.data());
      A.update_col(col, kron_y);
    }

    // Calculate (I - dt*A)
    fm::scal(-pde->get_dt(), A);
    for (int col = 0; col < dof; col++)
    {
      A(col, col) += 1.0;
    }

    int const offset = std::pow(degree, pde->num_dims);
    int const n      = num_blocks * offset;
    REQUIRE((A.nrows() == n));

    // Compare preconditioner blocks to the diagonal blocks of A
    for (size_t blk = 0; blk < num_blocks; blk++)
    {
      REQUIRE((precond.precond_blks[blk].nrows() == offset));
      REQUIRE((precond.precond_blks[blk].ncols() == offset));

      int const row = blk * offset;
      rmse_comparison(precond.precond_blks[blk],
                      A.extract_submatrix(row, row, offset, offset),
                      tol_factor);
    }
  }

  SECTION("apply")
  {
    auto precond = preconditioner::block_jacobi_preconditioner<TestType>();

    // Construct preconditioner matrix from PDE coefficients
    precond.construct(*pde, adaptive_grid.get_table(), sol.size(),
                      pde->get_dt(), imex_flag::imex_implicit);

    size_t const num_blocks =
        static_cast<size_t>(adaptive_grid.get_table().size());
    REQUIRE((precond.precond_blks.size() == num_blocks));
    REQUIRE((precond.blk_pivots.size() == num_blocks));
    REQUIRE((precond.factored() == false));

    // Get a dense copy of M before applying
    fk::matrix<TestType> M = precond.get_matrix();

    auto &mat            = operator_matrices[matrix_entry::imex_implicit];
    auto mat_replacement = [&](fk::vector<TestType> const &x_in,
                               fk::vector<TestType> &y, TestType const alpha,
                               TestType const beta) -> void {
      mat.apply(-pde->get_dt() * alpha, x_in.data(), beta, y.data());
      int one = 1, n = y.size();
      lib_dispatch::axpy(n, alpha, x_in.data(), one, y.data(), one);
    };

    // Test the implementation of the apply function
    TestType const alpha = -1.0;
    TestType const beta  = 1.0;

    fk::vector<TestType> b(sol);
    fk::vector<TestType> b_apply(sol);

    mat_replacement(sol, b, alpha, beta);
    std::vector<int> ipiv(b.size());
    fm::gesv(M, b, ipiv);

    mat_replacement(sol, b_apply, alpha, beta);
    precond.apply(b_apply);

    relaxed_fp_comparison(fm::nrm2(b), fm::nrm2(b_apply));

    // preconditioner should be factored now
    REQUIRE((precond.factored() == true));

    b       = sol * 0.1;
    b_apply = sol * 0.1;

    // Apply preconditioner again using the factors
    mat_replacement(sol, b, alpha, beta);
    fm::getrs(M, b, ipiv);

    mat_replacement(sol, b_apply, alpha, beta);
    precond.apply(b_apply);

    relaxed_fp_comparison(fm::nrm2(b), fm::nrm2(b_apply));
  }

  parameter_manager<TestType>::get_instance().reset();
}
