#include "asgard_test_macros.hpp"

#include "asgard_coefficients_mats.hpp"

using namespace asgard;

template<typename P>
void test_dense_matrix()
{
  current_test<P> name_("dense matrix");

  tassert(not dense_matrix<P>{});

  dense_matrix<P> mat(4, 5);
  tassert(!!mat);
  tassert(mat.nrows() == 4);
  tassert(mat.ncols() == 5);

  tassert(mat.data() == &mat(0, 0));
  tassert(mat.data() + 1 == &mat(1, 0));

  P constexpr test_val{42};
  mat(1, 2) = test_val;
  tassert(mat(1, 2) == test_val);
  { // get the value through a const overload
    P const val = [](dense_matrix<P> const &m) -> P { return m(1, 2); }(mat);
    tassert(val == test_val);
  }{ // get the value through a const pointer
    P const val = [](dense_matrix<P> const &m) -> P { return *m.data(1, 2); }(mat);
    tassert(val == test_val);
  }{ // get the const pointer
    P const *val = [](dense_matrix<P> const &m) -> P const *{ return m.data(); }(mat);
    tassert(val == mat.data());
    tassert(val + mat.nrows() + 2 == mat.data(2, 1));
  }

  std::stringstream ss;
  mat.print(ss); // do not check the output, this is human-readable only
}

template<typename P>
void test_block_matrix()
{
  current_test<P> name_("block matrix");

  block_matrix<P> mat(4, 3, 5);

  tassert(mat.nblock() == 4);
  tassert(mat.nrows() == 3);
  tassert(mat.ncols() == 5);

  block_matrix<P> const &m2 = mat;
  tassert(mat.data() == m2.data());

  mat.fill(42);
  tassert(*mat.data() == P{42});

  std::stringstream ss;
  mat.print(ss);
  mat.printc(ss);
  mat.printr(ss);
}

template<typename P>
void test_mass_matrix()
{
  current_test<P> name_("mass matrix");

  tassert(mass_matrix<P>{}.empty());

  mass_matrix<P> mass(4, 5);

  tassert(not mass.empty());
  tassert(mass.nblock() == 4);
  tassert(mass.nrows() == 5);

  std::fill_n(mass[1], mass.nblock(), P{42});
  mass_matrix<P> const &m = mass;
  tassert(m[1][2] == P{42});
  tassert(m[0][3] == 0);
  tassert(m.data() == mass.data());
}

template<typename P>
void test_block_diag_matrix()
{
  current_test<P> name_("diag matrix");

  tassert(not block_diag_matrix<P>{});

  block_diag_matrix<P> mat(4, 7);
  tassert(!!mat);

  block_diag_matrix<P> const &m = mat;
  tassert(mat[3] == m[3]);
  tassert(mat.data() == m.data());

  std::fill_n(mat[1], mat.nblock(), P{42});
  mat.resize_and_zero(4, 2);
  for (int64_t r = 0; r < mat.nrows(); r++)
    for (int64_t j = 0; j < mat.nblock(); j++)
      tassert(mat[r][j] == 0);

  auto dense = mat.to_full();
  static_assert(std::is_same_v<decltype(dense), block_matrix<P>>);
  tassert(dense.nblock() == 4);
  tassert(dense.nrows() == 2);
  tassert(dense.ncols() == dense.nrows());
}

template<typename P>
void test_block_tri_matrix()
{
  current_test<P> name_("tri matrix");

  block_tri_matrix<P> mat(9, 3);
  tassert(mat.nblock() == 9);
  tassert(mat.nrows() == 3);

  block_tri_matrix<P> const &m = mat;
  tassert(m[1] == mat[1]);
  tassert(mat.lower(0) == mat.data());
  tassert(mat.diag(0)  == mat[0]);
  tassert(mat.upper(0) == mat.data() + 18);
  tassert(m.data() == mat.data());

  mat.fill(P{5});
  for (int64_t j = 0; j < mat.nblock(); j++)
    tassert(mat[1][j] == 5);

  mat.resize_and_zero(16, 7);
  tassert(m.nblock() == 16);
  for (int64_t j = 0; j < m.nblock(); j++)
    tassert(m[1][j] == 0);

  mat.fill(P{7});
  auto dense = mat.to_full();
  static_assert(std::is_same_v<decltype(dense), block_matrix<P>>);
  tassert(dense.nblock() == 16);
  tassert(dense.nrows() == 7);
  tassert(dense.ncols() == dense.nrows());

  P const *x = dense(3, 3);
  for (int64_t j = 0; j < m.nblock(); j++)
    tassert(x[j] == 7);
  x = dense(3, 4);
  for (int64_t j = 0; j < m.nblock(); j++)
    tassert(x[j] == 7);
  x = dense(0, 6);
  for (int64_t j = 0; j < m.nblock(); j++)
    tassert(x[j] == 7);

  P const *y = dense(3, 5);
  for (int64_t j = 0; j < m.nblock(); j++)
    tassert(y[j] == 0);
  y = dense(3, 6);
  for (int64_t j = 0; j < m.nblock(); j++)
    tassert(y[j] == 0);
}

template<typename P>
void test_block_sparse_matrix()
{
  current_test<P> name_("block sparse matrix");

  tassert(block_sparse_matrix<P>{}.empty());

  block_sparse_matrix<P> mat(4, 10, connect_1d::hierarchy::volume);
  tassert(not mat.empty());
  tassert(mat.nblock() == 4);
  tassert(mat.nnz() == 10);

  mat.fill(7);
  std::vector<P> x;
  mat.copy_out(x);
  tassert(x.size() == 40); // nnz * block-size
  for (auto s : x)
    tassert(s == 7);
}

template<typename P>
void all_templated_tests()
{
  test_dense_matrix<P>();
  test_block_matrix<P>();
  test_mass_matrix<P>();
  test_block_diag_matrix<P>();
  test_block_tri_matrix<P>();
  test_block_sparse_matrix<P>();
}

int main(int argc, char **argv)
{
  libasgard_runtime running_(argc, argv);

  all_tests global_("block-matrix-tests", " using block matrix structures");

  #ifdef ASGARD_ENABLE_DOUBLE
  all_templated_tests<double>();
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  all_templated_tests<float>();
  #endif

  return 0;
}
