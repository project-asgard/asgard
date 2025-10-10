#include "asgard_test_macros.hpp"

#include "asgard_small_mats.hpp"

using namespace asgard;

template<typename TestType>
void test_transform()
{
  current_test<TestType> name_("fast-transform");
  std::minstd_rand park_miller(42);
  std::uniform_real_distribution<TestType> unif(-1.0, 1.0);

  for (int nbatch = 1; nbatch < 5; nbatch++) {
    for (int level = 0; level < 5; level++) {
      for (int degree = 0; degree < 4; degree++)
      {
        hierarchy_manipulator<TestType> hier(degree, 1, {-2,}, {1,}); // dims 1

        int const pdof    = (degree + 1);
        int64_t const num = fm::ipow2(level);

        std::vector<TestType> ref(nbatch * num * pdof);
        std::vector<TestType> hp(nbatch * num * pdof);

        for (auto &x : ref)
          x = unif(park_miller);

        std::vector<TestType> fp(num * pdof);
        for (int b : indexof(nbatch)) // forward project
        {
          TestType *r = ref.data() + b * pdof; // batch begin
          for (int i : indexof(num))
            std::copy_n(r + i * nbatch * pdof, pdof, fp.data() + i * pdof);

          hier.transform(level, fp); // to hierarchical

          TestType *h = hp.data() + b * pdof; // write out in hp
          for (int i : indexof(num))
            std::copy_n(fp.data() + i * pdof, pdof, h + i * nbatch * pdof);
        }

        if (level > 0)
          tassert(fm::diff_inf(ref, hp) > 1.E-2); // sanity check, did we transform anything

        hier.reconstruct1d(nbatch, level, span2d<TestType>(pdof, nbatch * num, hp.data()));

        tassert(fm::diff_inf(ref, hp) < 5.E-6); // inverse transform should get us back
      }
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
void all_templated_tests()
{
  test_transform<P>();
  test_permute<P>();
  test_custom_transform<P>();
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
