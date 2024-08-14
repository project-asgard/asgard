#include "tests_general.hpp"

using namespace asgard;

TEMPLATE_TEST_CASE("fk::sparse interface: constructors, copy/move", "[sparse]",
                   test_precs, int)
{
  // test 4x6 CSR sparse matrix
  int nrows                         = 4;
  int ncols                         = 6;
  int sp_size                       = 8;
  int size                          = nrows * ncols;
  fk::vector<int> const col_indices = {0, 1, 1, 3, 2, 3, 4, 5};
  fk::vector<int> const row_offsets = {0, 2, 4, 7, 8};

  // explicit types for testing converting copy operations
  fk::vector<int> const goldi{10, 20, 30, 40, 50, 60, 70, 80};
  fk::vector<float> const goldf{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0};
  fk::vector<float> const goldd{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0};

  SECTION("default constructor")
  {
    fk::sparse<TestType> const test;
    REQUIRE(test.empty());
  }

  SECTION("constructor")
  {
    fk::vector<TestType> const values{10, 20, 30, 40, 50, 60, 70, 80};
    fk::sparse<TestType> const sp_mat(nrows, ncols, sp_size, row_offsets,
                                      col_indices, values);
    REQUIRE((sp_mat.nrows() == nrows));
    REQUIRE((sp_mat.ncols() == ncols));
    REQUIRE((sp_mat.size() == size));
    REQUIRE((sp_mat.sp_size() == sp_size));
    REQUIRE((sp_mat.get_offsets() == row_offsets));
    REQUIRE((sp_mat.get_columns() == col_indices));
    REQUIRE((sp_mat.get_values() == values));
  }

  SECTION("construct from an fk::matrix")
  {
    // dense matrix
    fk::matrix<TestType> const mat{{10, 20, 0, 0, 0, 0},
                                   {0, 30, 0, 40, 0, 0},
                                   {0, 0, 50, 60, 70, 0},
                                   {0, 0, 0, 0, 0, 80}};
    fk::vector<TestType> const values{10, 20, 30, 40, 50, 60, 70, 80};

    fk::sparse<TestType> const sp_mat(mat);
    REQUIRE((sp_mat.nrows() == nrows));
    REQUIRE((sp_mat.ncols() == ncols));
    REQUIRE((sp_mat.size() == size));
    REQUIRE((sp_mat.sp_size() == sp_size));
    REQUIRE((sp_mat.get_offsets() == row_offsets));
    REQUIRE((sp_mat.get_columns() == col_indices));
    REQUIRE((sp_mat.get_values() == values));

    // transfer back to dense form
    REQUIRE((sp_mat.to_dense() == mat));

#ifdef ASGARD_USE_CUDA
    // create device sparse matrix from host
    fk::sparse<TestType, resource::device> const sp_mat_d(
        sp_mat.clone_onto_device());

    REQUIRE((sp_mat_d.nrows() == nrows));
    REQUIRE((sp_mat_d.ncols() == ncols));
    REQUIRE((sp_mat_d.size() == size));
    REQUIRE((sp_mat_d.sp_size() == sp_size));
    REQUIRE((sp_mat_d.get_offsets().clone_onto_host() == row_offsets));
    REQUIRE((sp_mat_d.get_columns().clone_onto_host() == col_indices));
    REQUIRE((sp_mat_d.get_values().clone_onto_host() == values));

    // clone device sparse matrix back to host
    fk::sparse<TestType, resource::host> const sp_host =
        sp_mat_d.clone_onto_host();
    REQUIRE((sp_host.nrows() == nrows));
    REQUIRE((sp_host.ncols() == ncols));
    REQUIRE((sp_host.size() == size));
    REQUIRE((sp_host.sp_size() == sp_size));
    REQUIRE((sp_host.get_offsets() == row_offsets));
    REQUIRE((sp_host.get_columns() == col_indices));
    REQUIRE((sp_host.get_values() == values));
#endif
  }

  SECTION("construct diagonally from a fk::vector")
  {
    // create a sparse matrix from a vector along diagonal elements
    fk::vector<TestType> const diag{10, 20, 30, 40};

    // dense matrix
    fk::matrix<TestType> const mat{
        {10, 0, 0, 0}, {0, 20, 0, 0}, {0, 0, 30, 0}, {0, 0, 0, 40}};
    fk::vector<int> const indices = {0, 1, 2, 3};
    fk::vector<int> const offsets = {0, 1, 2, 3, 4};

    fk::sparse<TestType> const sp_mat(diag);
    REQUIRE((sp_mat.nrows() == diag.size()));
    REQUIRE((sp_mat.ncols() == diag.size()));
    REQUIRE((sp_mat.sp_size() == diag.size()));
    REQUIRE((sp_mat.get_offsets() == offsets));
    REQUIRE((sp_mat.get_columns() == indices));
    REQUIRE((sp_mat.to_dense() == mat));
  }

  SECTION("copy construction")
  {
    fk::vector<TestType> const values{10, 20, 30, 40, 50, 60, 70, 80};
    fk::sparse<TestType> const sp_mat_gold(nrows, ncols, sp_size, row_offsets,
                                           col_indices, values);

    fk::sparse<TestType> test(sp_mat_gold);
    REQUIRE((test == sp_mat_gold));
  }

  SECTION("copy assignment")
  {
    fk::vector<TestType> const values{10, 20, 30, 40, 50, 60, 70, 80};
    fk::sparse<TestType> sp_mat_gold(nrows, ncols, sp_size, row_offsets,
                                     col_indices, values);

    fk::sparse<TestType> test;
    test = sp_mat_gold;
    REQUIRE((test == sp_mat_gold));
  }

  SECTION("move construction")
  {
    fk::vector<TestType> const values{10, 20, 30, 40, 50, 60, 70, 80};
    fk::sparse<TestType> sp_mat_gold(nrows, ncols, sp_size, row_offsets,
                                     col_indices, values);

    // owners
    fk::sparse<TestType> moved(sp_mat_gold);
    fk::sparse<TestType> test(std::move(moved));
    REQUIRE((moved.data() == nullptr));
    REQUIRE((test == sp_mat_gold));
  }

  SECTION("move assignment")
  {
    // owners
    fk::vector<TestType> const values{10, 20, 30, 40, 50, 60, 70, 80};
    fk::sparse<TestType> sp_mat_gold(nrows, ncols, sp_size, row_offsets,
                                     col_indices, values);

    fk::sparse<TestType> moved(sp_mat_gold);
    TestType *const data = moved.data();
    fk::sparse<TestType> test;
    TestType *const test_data = test.data();
    test                      = std::move(moved);
    REQUIRE((test.data() == data));
    REQUIRE((moved.data() == test_data));
    REQUIRE((test == sp_mat_gold));
  }
} // end fk::sparse constructors, copy/move
