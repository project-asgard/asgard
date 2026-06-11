#include "asgard_test_macros.hpp"

#include "asgard_small_mats.hpp"

using namespace asgard;

template<typename TestType>
void test_transform()
{
  current_test<TestType> name_("fast-transform");
  std::minstd_rand park_miller(42);
  std::uniform_real_distribution<TestType> unif(-1.0, 1.0);

  for (int level = 0; level < 5; level++) {
    for (int degree = 0; degree < 4; degree++)
    {
      hierarchy_manipulator<TestType> hier(degree, 1, {-2,}, {1,}); // dims 1

      int const pdof    = (degree + 1);
      int64_t const num = fm::ipow2(level);

      std::vector<TestType> ref(num * pdof);

      for (auto &x : ref)
        x = unif(park_miller);

      std::vector<TestType> fp;
      std::vector<TestType> work = ref; // note the forward transform is destructive on ref
      hier.transform(level, work, fp); // to hierarchical

      if (level > 0)
        tassert(fm::diff_inf(fp, ref) > 1.E-2); // sanity check, did we transform anything

      std::vector<TestType> inv = fp;
      hier.reconstruct1d(level, inv);

      tassert(fm::diff_inf(ref, inv) < 5.E-6); // inverse transform should get us back
    }
  }
}

template<typename P>
void test_permute()
{
  current_test<P> name_("vector permutation");

  { // zero order
    hierarchy_manipulator<P> hier(0, 1, {0,}, {1,});

    std::array<P, 4> const p = { 1, 0, 0, 1 }; // left bias in point selection

    std::vector<P> x = {1, 2, };
    std::vector<P> ref = {1, 2, };
    std::vector<P> y;
    hier.transform(p.data(), 1, x, y);
    tassert(fm::diff_inf(y, ref) == 0);

    x   = {1, 2, 3, 4};
    ref = {1, 3, 2, 4};
    hier.transform(p.data(), 2, x, y);
    tassert(fm::diff_inf(y, ref) == 0);
  }{ // first order
    hierarchy_manipulator<P> hier(1, 1, {0,}, {1,});

    // permutation for the (1/3, 2/3) points in 1d
    std::array<P, 16> const p = { 0, 0, 1, 0,
                                  1, 0, 0, 0,
                                  0, 1, 0, 0,
                                  0, 0, 0, 1};

    std::vector<P> x   = {1, 2, 3, 4};
    std::vector<P> y;
    std::vector<P> ref = {2, 3, 1, 4};
    hier.transform(p.data(), 1, x, y);
    tassert(fm::diff_inf(y, ref) == 0);

    x   = {1, 2, 3, 4, 5, 6, 7, 8};
    ref = {3, 6, 2, 7, 1, 4, 5, 8};
    hier.transform(p.data(), 2, x, y);
    tassert(fm::diff_inf(y, ref) == 0);
  }
}

// helper method, compare a block n by n to the identity
template<typename P>
P identity_compare(int const n, P const x[]) {
  P err = 0;
  smmat::matrix<P const> A(n, x); // diagonal block

  for (int r = 0; r < n; r++)
    for (int c = 0; c < n; c++)
      err = (r == c) ? std::max(err, std::abs(A(r, c) - 1))
                     : std::max(err, std::abs(A(r, c)));

  return err;
}

template<typename P>
P identity_compare(int const n, connection_patterns const &conns,
                   block_sparse_matrix<P> const &mat)
{
  tassert(n *n == mat.nblock());

  connect_1d::hierarchy const hier = mat;
  auto const &conn = conns[hier];

  P check = 0;
  for (int row = 0; row < conn.num_rows(); row++)
  {
    for (int j = conn.row_begin(row); j < conn.row_diag(row); j++)
      check = std::max(check, fm::nrm_inf(n * n, mat[j]));
    for (int j = conn.row_diag(row) + 1; j < conn.row_diag(row); j++)
      check = std::max(check, fm::nrm_inf(n * n, mat[j]));

    check = std::max(check, identity_compare(n, mat[conn.row_diag(row)]));
  }

  return check;
}


template<typename P>
void test_custom_transform()
{
  P constexpr tol = (is_double<P>) ? 1.E-13 : 1.E-5;

  auto constexpr op_uni = hierarchy_manipulator<P>::operation::custom_unitary;
  auto constexpr op_non = hierarchy_manipulator<P>::operation::custom_non_unitary;

  current_test<P> name_("custom transformations");

  { // zero order
    connection_patterns conn(1);
    hierarchy_manipulator<P> hier(0, 1, {0,}, {1,});

    std::array<P, 4> const p = { 1, 0, 0, 1 }; // left bias in point selection

    block_diag_matrix<P> mat(1, 2);
    for (int i = 0; i < 2; i++) mat[i][0] = 1; // set to identity

    auto res = hier.diag2block(op_uni, p.data(), op_uni, p.data(),
                               mat, 1, conn);
    tassert(res[0][0] == 1);
    tassert(res[1][0] == 0);
    tassert(res[2][0] == 0);
    tassert(res[3][0] == 1);
  }{ // first order
    int constexpr order = 1;
    int const pdof = order + 1;
    for (int level = 0; level < 5; level++)
    {
      connection_patterns conns(level);
      hierarchy_manipulator<P> hier(order, 1, {0,}, {1,});

      std::array<P, 16> const p = { 0, 0, 1, 0,
                                    1, 0, 0, 0,
                                    0, 1, 0, 0,
                                    0, 0, 0, 1};

      block_diag_matrix<P> mat(pdof * pdof, fm::ipow2(level));

      fill_pattern(smmat::make_identity<P>(pdof).data(), mat);

      auto res = hier.diag2block(op_uni, p.data(), op_uni, p.data(),
                                 mat, level, conns);

      tcheckless(level, identity_compare(pdof, conns, res), tol);
    }
  }{ // first order
    int constexpr order = 1;
    int const pdof = order + 1;
    for (int level = 0; level < 10; level++)
    {
      connection_patterns conns(level);
      hierarchy_manipulator<P> hier(order, 1, {0,}, {1,});

      std::array<P, 16> const h = {0, 0,  1,    0,
                                   1, 0, -1.5,  0.5,
                                   0, 1,  0.5, -1.5,
                                   0, 0,  0,    1};

      std::array<P, 16> const h_inv = { 1.5, 1, 0, -0.5,
                                       -0.5, 0, 1,  1.5,
                                        1,   0, 0,  0,
                                        0,   0, 0,  1};

      block_diag_matrix<P> mat(pdof * pdof, fm::ipow2(level));

      fill_pattern(smmat::make_identity<P>(pdof).data(), mat);

      auto res = hier.diag2block(op_non, h_inv.data(), op_non, h.data(),
                                 mat, level, conns);

      // std::cout << " non-unit: " << identity_compare(pdof, conns, res) << "\n";
      tcheckless(level, identity_compare(pdof, conns, res), tol);
    }
  }{ // third order
    int constexpr order = 3;
    int const pdof = order + 1;
    for (int level = 0; level < 10; level++)
    {
      connection_patterns conns(level);
      hierarchy_manipulator<P> hier(order, 1, {0,}, {1,});

      std::array<P, 64> const h = {
                  0,     0,     0,     0,   1.00000,         0,         0,         0,
                1.0,     0,     0,     0,  -2.18750,  -0.31250,  -0.06250,   0.31250,
                  0,     0,     0,     0,         0,   1.00000,         0,         0,
                  0,   1.0,     0,     0,   2.18750,  -0.93750,   0.31250,  -1.31250,
                  0,     0,   1.0,     0,  -1.31250,   0.31250,  -0.93750,   2.18750,
                  0,     0,     0,     0,         0,         0,   1.00000,         0,
                  0,     0,     0,   1.0,   0.31250,  -0.06250,  -0.31250,  -2.18750,
                  0,     0,     0,     0,         0,         0,         0,   1.00000,
          };

      std::array<P, 64> const h_inv = {
                2.1875,   1.0000,   0.3125,        0,        0,   0.0625,        0,  -0.3125,
               -2.1875,        0,   0.9375,   1.0000,        0,  -0.3125,        0,   1.3125,
                1.3125,        0,  -0.3125,        0,   1.0000,   0.9375,        0,  -2.1875,
               -0.3125,        0,   0.0625,        0,        0,   0.3125,   1.0000,   2.1875,
                1.0000,        0,        0,        0,        0,        0,        0,        0,
                     0,        0,   1.0000,        0,        0,        0,        0,        0,
                     0,        0,        0,        0,        0,   1.0000,        0,        0,
                     0,        0,        0,        0,        0,        0,        0,   1.0000,

          };

      block_diag_matrix<P> mat(pdof * pdof, fm::ipow2(level));

      fill_pattern(smmat::make_identity<P>(pdof).data(), mat);

      auto res = hier.diag2block(op_non, h_inv.data(), op_non, h.data(),
                                 mat, level, conns);

      // std::cout << " non-unit: " << identity_compare(pdof, conns, res) << "\n";
      tcheckless(level, identity_compare(pdof, conns, res), tol);
    }
  }
}

template<typename P>
void differentiate_tests()
{
  current_test<P> name_("differentiation");

  auto f1 = [](P x) -> double { return std::sin(x); };
  auto f2 = [](P x) -> double { return std::cos(x); };
  // auto f3 = [](P x) -> double { return std::sin(3 * x); };

  auto df1 = [](P x) -> double { return  std::cos(x); };
  auto df2 = [](P x) -> double { return -std::sin(x); };
  // auto df3 = [](P x) -> double { return 3 * std::cos(3 * x); };

  std::array<kronmult::permutes, 3> perm;
  for (int i = 0; i < 3; i++) perm[i] = kronmult::permutes(std::vector<int>{i, });

  std::array<vector2d<P>, 3> dblock;
  for (size_t i = 0; i < dblock.size(); i++)
    dblock[i] = legendre::poly2diff(static_cast<int>(i));

  auto make_diff = [&](hierarchy_manipulator<P> const &hier, connection_patterns const &conns,
                       int level, P xlength)
        -> block_sparse_matrix<P>
    {
      int const pdof = hier.degree() + 1;
      int const num_cells = fm::ipow2(level);

      vector2d<P> p2d = legendre::poly2diff(hier.degree());
      smmat::scal(pdof * pdof, static_cast<P>(num_cells) / xlength, p2d[0]);

      block_diag_matrix<P> diag(pdof * pdof, num_cells);
      fill_pattern(p2d[0], diag);

      return hier.diag2hierarchical(diag, level, conns);
    };

  kronmult::workspace<P> kwork;

  struct test_entry {
    int degree;
    int level;
    P tol;
  };

  { // differentiate one function, 1d
    int constexpr num_dims = 1;

    std::vector<test_entry> entries;
    entries.reserve(6);
    entries.emplace_back(1, 8, 1.E-3);
    entries.emplace_back(2, 4, 5.E-4);
    entries.emplace_back(2, 5, 1.E-4);
    entries.emplace_back(2, 6, 1.E-5);
    entries.emplace_back(3, 3, 5.E-6);
    entries.emplace_back(3, 5, 1.E-7);

    for (auto const &test : entries) {
      int const degree = test.degree;
      int const level  = test.level;
      sparse_grid grid(make_opts("-l " + std::to_string(level) + " -d " + std::to_string(degree)));
      connection_patterns const conns(level);

      kwork.w1.resize(grid.num_dof());
      kwork.w2.resize(grid.num_dof());

      hierarchy_manipulator<P> hier(degree, num_dims, {0,}, {1,});

      block_sparse_matrix<P> const mat_diff = make_diff(hier, conns, level, P{1});

      separable_func func{{vectorize<P>(f1), }};
      separable_func dfunc{{vectorize<P>(df1), }};

      std::vector<P> proj(grid.num_dof());
      std::vector<P> dproj(grid.num_dof());
      std::vector<P> ref_proj(grid.num_dof());

      hier.project_separable(func, grid, {}, 0, 1, proj.data());
      hier.project_separable(dfunc, grid, {}, 0, 1, ref_proj.data());

      block_cpu(degree + 1, grid, conns, perm[0], mat_diff,
                P{1}, proj.data(), P{0}, dproj.data(), kwork);

      P const err = diff_l2(grid.num_dof(), dproj.data(), ref_proj.data());
      // std::cout << " err = " << err << "    " << test.tol << '\n';
      tassert(err < test.tol);
    }
  }

  { // differentiate one function, 2d
    int constexpr num_dims = 2;

    std::vector<test_entry> entries;
    entries.reserve(6);
    entries.emplace_back(1, 8, 5.E-3);
    entries.emplace_back(2, 4, 1.E-3);
    entries.emplace_back(2, 5, 3.E-4);
    entries.emplace_back(2, 6, 1.E-4);
    entries.emplace_back(3, 4, 5.E-5);
    entries.emplace_back(3, 6, 3.E-7);

    for (auto const &test : entries) {
      int const degree = test.degree;
      int const level  = test.level;
      auto options = make_opts(" -d " + std::to_string(degree));
      options.start_levels = std::vector<int>(num_dims, level);

      sparse_grid grid(options);
      connection_patterns const conns(level);

      kwork.w1.resize(grid.num_dof());
      kwork.w2.resize(grid.num_dof());

      hierarchy_manipulator<P> hier(degree, num_dims, {0, -1}, {2, 2});

      block_sparse_matrix<P> const mat_diff2 = make_diff(hier, conns, level, P{2});
      block_sparse_matrix<P> const mat_diff3 = make_diff(hier, conns, level, P{3});

      separable_func func{{vectorize<P>(f1), vectorize<P>(f2) }};
      separable_func dxfunc{{vectorize<P>(df1), vectorize<P>(f2) }};
      separable_func dyfunc{{vectorize<P>(f1), vectorize<P>(df2) }};

      std::vector<P> proj(grid.num_dof());
      std::vector<P> dxproj(grid.num_dof());
      std::vector<P> dyproj(grid.num_dof());
      std::vector<P> refx(grid.num_dof());
      std::vector<P> refy(grid.num_dof());

      hier.project_separable(func, grid, {}, 0, 1, proj.data());
      hier.project_separable(dxfunc, grid, {}, 0, 1, refx.data());
      hier.project_separable(dyfunc, grid, {}, 0, 1, refy.data());

      block_cpu(degree + 1, grid, conns, perm[0], mat_diff2,
                P{1}, proj.data(), P{0}, dxproj.data(), kwork);

      block_cpu(degree + 1, grid, conns, perm[1], mat_diff3,
                P{1}, proj.data(), P{0}, dyproj.data(), kwork);

      P const errx = diff_l2(grid.num_dof(), dxproj.data(), refx.data());
      // std::cout << " err = " << err << "    " << test.tol << '\n';
      tassert(errx < test.tol);
      P const erry = diff_l2(grid.num_dof(), dxproj.data(), refx.data());
      // std::cout << " err = " << err << "    " << test.tol << '\n';
      tassert(erry < test.tol);
    }
  }

}

template<typename P>
void all_templated_tests()
{
  test_transform<P>();
  test_permute<P>();
  test_custom_transform<P>();
  differentiate_tests<P>();
}

int main(int argc, char **argv)
{
  libasgard_runtime running_(argc, argv);

  all_tests global_("transformation-tests", " hierarchical<->cell-by-cell basis");

  #ifdef ASGARD_ENABLE_DOUBLE
  all_templated_tests<double>();
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  all_templated_tests<float>();
  #endif

  return 0;
}
