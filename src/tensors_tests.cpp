#include "tests_general.hpp"

using namespace asgard;

// note using widening conversions to floating point type in order to use same
// tests for integer type
// FIXME look for another way to do this

TEMPLATE_TEST_CASE("fk::vector interface: constructors, copy/move", "[tensors]",
                   test_precs, int)
{
  // set up the golden vector
  // orthogonality warnings: all of the tests depend on
  // - the list initializing constructor working correctly
  // - operator== working correctly
  fk::vector<TestType> const gold{2, 3, 4, 5, 6};

  // explicit types for testing converting copy operations
  fk::vector<int> const goldi{2, 3, 4, 5, 6};
  fk::vector<float> const goldf{2.0, 3.0, 4.0, 5.0, 6.0};
  fk::vector<double> const goldd{2.0, 3.0, 4.0, 5.0, 6.0};

  SECTION("default constructor")
  {
    fk::vector<TestType> const test;
    // fk::vector<TestType, mem_type::view> test_v; // disabled
    REQUIRE(test.empty());
  }
  SECTION("give me some size, initialized to zero")
  {
    fk::vector<TestType> const test(5);
    // fk::vector<TestType, mem_type::view> test_v(5); // disabled
    fk::vector<TestType> const zeros{0, 0, 0, 0, 0};
    REQUIRE(test == zeros);
    // REQUIRE(test_v == zeros);
  }
  SECTION("constructor from list initialization")
  {
    fk::vector<TestType> const test{2, 3, 4, 5, 6};
    // fk::vector<TestType, mem_type::view> test_v{2, 3, 4, 5, 6}; // disabled
    REQUIRE(test == gold);
    // REQUIRE(test_v == gold);
  }
  SECTION("construct from a std::vector")
  {
    std::vector<TestType> const v{2, 3, 4, 5, 6};
    fk::vector<TestType> const test(v);
    // fk::vector<TestType, mem_type::view> test_v(v); // disabled
    REQUIRE(test == gold);
    // REQUIRE(test_v == gold);
  }
  SECTION("construct from an fk::matrix")
  {
    fk::matrix<TestType> const mat{{2}, {3}, {4}, {5}, {6}};
    fk::vector<TestType> const test(mat);
    // fk::vector<TestType, mem_type::view> test_v(mat); // disabled
    REQUIRE(test == gold);
    // REQUIRE(test_v == gold);

    fk::vector<TestType> const gold_2 = {1, 2, 3, 4, 5, 6};
    fk::matrix<TestType> const mat_2{{1, 3, 5}, {2, 4, 6}};
    fk::vector<TestType> const test_2(mat_2);
    // fk::vector<TestType, mem_type::view> test_2_v(mat_2); // disabled
    REQUIRE(test_2 == gold_2);

    // check for problems when constructing from an empty matrix
    fk::matrix<TestType> const mat_empty;
    fk::vector<TestType> const test_empty(mat_empty);
    REQUIRE(test_empty.empty());

    // enable on device...
#ifdef ASGARD_USE_CUDA
    fk::matrix<TestType, mem_type::owner, resource::device> const mat_d(
        mat.clone_onto_device());
    fk::vector<TestType, mem_type::owner, resource::device> const vect_d(mat_d);
    REQUIRE(vect_d.clone_onto_host() == gold);
#endif
  }

  SECTION("construct view from owner")
  {
    // default: view of whole vector
    fk::vector<TestType> gold_copy(gold);
    fk::vector<TestType, mem_type::view> test_v(gold_copy);
    REQUIRE(test_v == gold);
    gold_copy(2) = 1000;
    REQUIRE(test_v == gold_copy);
    REQUIRE(gold != gold_copy);
    REQUIRE(test_v != gold);
    test_v(2) = 10;
    REQUIRE(test_v == gold_copy);
    REQUIRE(gold != gold_copy);
    REQUIRE(test_v != gold);
    gold_copy = gold;

    // specify range (start/stop inclusive)
    fk::vector<TestType, mem_type::view> test_v2(gold_copy, 1, 3);
    fk::vector<TestType> const gold_portion = {3, 4, 5};
    fk::vector<TestType, mem_type::view> test_vb(gold_copy);
    fk::vector<TestType, mem_type::view> test_v3(test_vb, 1, 3);
    REQUIRE(test_v2 == gold_portion);
    REQUIRE(test_v3 == gold_portion);

    gold_copy(2) = 1000;
    REQUIRE(test_v2 == gold_copy.extract(1, 3));
    REQUIRE(test_v3 == gold_copy.extract(1, 3));

    REQUIRE(gold_copy != gold);
    REQUIRE(test_v2 != gold_portion);
    REQUIRE(test_v3 != gold_portion);

    test_v2(2) = 10;

    REQUIRE(test_v2 == gold_copy.extract(1, 3));
    REQUIRE(test_v3 == gold_copy.extract(1, 3));
    REQUIRE(gold_copy != gold);
    REQUIRE(test_v2 != gold_portion);
    REQUIRE(test_v3 != gold_portion);

    test_v3(0) = 4;
    REQUIRE(test_v2 == gold_copy.extract(1, 3));
    REQUIRE(test_v3 == gold_copy.extract(1, 3));
    REQUIRE(gold_copy != gold);
    REQUIRE(test_v2 != gold_portion);
    REQUIRE(test_v3 != gold_portion);

    // empty case
    fk::vector<TestType> empty;
    fk::vector<TestType, mem_type::view> empty_v(empty);

    fk::vector<TestType, mem_type::view> empty_v2(empty_v);
    REQUIRE(empty_v == empty);
    REQUIRE(empty_v == empty_v2);
    REQUIRE(empty_v2 == empty);
    REQUIRE(empty_v.data() == nullptr);
    REQUIRE(empty_v.empty());
    REQUIRE(empty_v2.data() == nullptr);
    REQUIRE(empty_v2.empty());
  }

  SECTION("construct const view from owner")
  {
    // default: view of whole vector
    fk::vector<TestType, mem_type::const_view> const test_v(gold);
    REQUIRE(test_v == gold);
    REQUIRE(test_v.data() == gold.data());

    // specify range (start/stop inclusive)
    fk::vector<TestType, mem_type::const_view> const test_v2(gold, 1, 3);
    REQUIRE(test_v2.size() == 3);
    fk::vector<TestType> const gold_portion = {3, 4, 5};
    REQUIRE(test_v2 == gold_portion);
    REQUIRE(test_v2.data() == gold.data() + 1);

    // from another const view
    fk::vector<TestType, mem_type::const_view> const test_v3(test_v2, 1, 2);
    REQUIRE(test_v3.size() == 2);
    fk::vector<TestType> const gold_portion_2 = {4, 5};
    REQUIRE(test_v3 == gold_portion_2);
    REQUIRE(test_v3.data() == gold.data() + 2);

    // empty case
    fk::vector<TestType> const empty;
    fk::vector<TestType, mem_type::const_view> const empty_v(empty);
    fk::vector<TestType, mem_type::const_view> const empty_v2(empty_v);
    REQUIRE(empty_v == empty);
    REQUIRE(empty_v2 == empty);
    REQUIRE(empty_v == empty_v);

    REQUIRE(empty_v.data() == nullptr);
    REQUIRE(empty_v.empty());
    REQUIRE(empty_v2.data() == nullptr);
    REQUIRE(empty_v2.empty());
  }

  SECTION("construct owner from view")
  {
    fk::vector<TestType> gold_copy(gold);
    fk::vector<TestType, mem_type::view> gold_copy_v(gold_copy);
    fk::vector<TestType> test(gold_copy_v);
    REQUIRE(test == gold);
  }
  SECTION("construct owner from const view")
  {
    fk::vector<TestType, mem_type::const_view> const gold_v(gold);
    fk::vector<TestType> test(gold_v);
    REQUIRE(test == gold);
  }

  SECTION("copy assign to owner from view")
  {
    fk::vector<TestType> gold_copy(gold);
    fk::vector<TestType, mem_type::view> gold_v(gold_copy);
    fk::vector<TestType> test(5);
    test = gold_v;
    REQUIRE(test == gold);
  }
  SECTION("copy assign to owner from const view")
  {
    fk::vector<TestType, mem_type::const_view> const gold_v(gold);
    fk::vector<TestType> test(5);
    test = gold_v;
    REQUIRE(test == gold);
  }

  SECTION("copy construction")
  {
    fk::vector<TestType> test(gold);
    REQUIRE(test == gold);

    fk::vector<TestType> base(5);
    fk::vector<TestType, mem_type::view> test_v(base);
    test_v = gold;
    fk::vector<TestType, mem_type::view> test_copy(test_v);
    REQUIRE(test_copy == gold);

    fk::vector<TestType, mem_type::const_view> const gold_view(gold);
    fk::vector<TestType, mem_type::const_view> const gold_view_copy(gold_view);
    REQUIRE(gold_view_copy == gold);
  }

  SECTION("copy assignment")
  {
    fk::vector<TestType> test(5);
    test = gold;
    REQUIRE(test == gold);

    fk::vector<TestType> base(5);
    fk::vector<TestType, mem_type::view> test_v(base);
    fk::vector<TestType, mem_type::const_view> const gold_v(gold);
    test_v = gold_v;
    REQUIRE(test_v == gold);
  }

  SECTION("move construction")
  {
    // owners
    fk::vector<TestType> moved(gold);
    fk::vector<TestType> test(std::move(moved));
    REQUIRE(moved.data() == nullptr);
    REQUIRE(test == gold);

    // views
    fk::vector<TestType> moved_own(gold);
    fk::vector<TestType, mem_type::view> moved_v(moved_own);
    fk::vector<TestType, mem_type::view> test_v(std::move(moved_v));
    REQUIRE(moved_v.data() == nullptr);
    REQUIRE(test_v.data() == moved_own.data());
    REQUIRE(test_v == gold);

    // const views
    fk::vector<TestType, mem_type::const_view> moved_cv(gold);
    fk::vector<TestType, mem_type::const_view> test_cv(std::move(moved_cv));
    REQUIRE(moved_cv.data() == nullptr);
    REQUIRE(test_cv.data() == gold.data());
    REQUIRE(test_cv == gold);
  }

  SECTION("move assignment")
  {
    // owners
    fk::vector<TestType> moved(gold);
    TestType *const data = moved.data();
    fk::vector<TestType> test(5);
    TestType *const test_data = test.data();
    test                      = std::move(moved);
    REQUIRE(test.data() == data);
    REQUIRE(moved.data() == test_data);
    REQUIRE(test == gold);

    // views
    fk::vector<TestType> moved_o(gold);
    fk::vector<TestType, mem_type::view> moved_v(moved_o);
    fk::vector<TestType> test_o(5);
    fk::vector<TestType, mem_type::view> test_v(test_o);
    TestType *const test_data_v = test_v.data();
    test_v                      = std::move(moved_v);
    REQUIRE(test_v.data() == moved_o.data());
    REQUIRE(moved_v.data() == test_data_v);
    REQUIRE(test_v == gold);

    // const views - disabled
  }

  SECTION("copy from std::vector")
  {
    std::vector<TestType> v{2, 3, 4, 5, 6};
    fk::vector<TestType> test(5);
    fk::vector<TestType, mem_type::view> test_v(test);
    test   = v;
    test_v = v;
    REQUIRE(test_v == gold);
    // disabled for const views
  }
  SECTION("copy into std::vector")
  {
    std::vector<TestType> const goldv{2, 3, 4, 5, 6};

    // owners
    std::vector<TestType> const testv(gold.to_std());
    compare_vectors(testv, goldv);

    // views
    fk::vector<TestType> gold_copy(gold);
    fk::vector<TestType, mem_type::view> gold_v(gold_copy);
    std::vector<TestType> const testv_v(gold_v.to_std());
    compare_vectors(testv_v, goldv);

    // const views
    fk::vector<TestType, mem_type::const_view> const gold_cv(gold);
    std::vector<TestType> const testv_cv(gold_cv.to_std());
    compare_vectors(testv_cv, goldv);
  }
} // end fk::vector constructors, copy/move

TEMPLATE_TEST_CASE("fk::vector operators", "[tensors]", test_precs, int)
{
  fk::vector<TestType> const gold{2, 3, 4, 5, 6};

  SECTION("subscript operator (modifying)")
  {
    fk::vector<TestType> test(5);
    fk::vector<TestType> own(5);
    fk::vector<TestType, mem_type::view> test_v(own);
    // clang-format off
    test(0) = 2; test(1) = 3; test(2) = 4; test(3) = 5; test(4) = 6;
    test_v(0) = 2; test_v(1) = 3; test_v(2) = 4; test_v(3) = 5; test_v(4) = 6;
    // clang-format on
    REQUIRE(test == gold);
    REQUIRE(test_v == gold);
    TestType const val   = test(4);
    TestType const val_v = test_v(4);
    REQUIRE(val == 6);
    REQUIRE(val_v == 6);
  }

  SECTION("subscript operator (const)")
  {
    REQUIRE(gold(4) == 6);
    fk::vector<TestType> gold_copy(gold);
    fk::vector<TestType, mem_type::view> gold_v(gold_copy);
    REQUIRE(gold_v(4) == 6);
    fk::vector<TestType, mem_type::const_view> const gold_cv(gold);
    REQUIRE(gold_cv(4) == 6);
  }

  SECTION("array index operator (modifying)")
  {
    fk::vector<TestType> test(5);
    fk::vector<TestType> own(5);
    fk::vector<TestType, mem_type::view> test_v(own);
    // clang-format off
    test[0] = 2; test[1] = 3; test[2] = 4; test[3] = 5; test[4] = 6;
    test_v[0] = 2; test_v[1] = 3; test_v[2] = 4; test_v[3] = 5; test_v[4] = 6;
    // clang-format on
    REQUIRE(test == gold);
    REQUIRE(test_v == gold);
    TestType const val   = test[4];
    TestType const val_v = test_v[4];
    REQUIRE(val == 6);
    REQUIRE(val_v == 6);
  }

  SECTION("array index operator (const)")
  {
    REQUIRE(gold[4] == 6);
    fk::vector<TestType> gold_copy(gold);
    fk::vector<TestType, mem_type::view> gold_v(gold_copy);
    REQUIRE(gold_v[4] == 6);
    fk::vector<TestType, mem_type::const_view> const gold_cv(gold);
    REQUIRE(gold_cv[4] == 6);
  }

  SECTION("comparison operator") // this gets used in every REQUIRE

  SECTION("comparison (negated) operator")
  {
    fk::vector<TestType> test(gold);
    fk::vector<TestType> gold_copy(gold);
    fk::vector<TestType, mem_type::view> test_v(gold_copy);
    fk::vector<TestType> const empty;
    fk::vector<TestType, mem_type::const_view> const empty_cv(empty);

    test(4)   = 333;
    test_v(4) = 333;

    REQUIRE(test != gold);
    REQUIRE(test_v != gold);
    REQUIRE(empty != gold);
    REQUIRE(empty_cv != gold);
  }

  SECTION("less than operator")
  {
    fk::vector<TestType> empty;
    fk::vector<TestType, mem_type::view> const empty_v(empty);
    fk::vector<TestType, mem_type::const_view> const empty_cv(empty);

    fk::vector<TestType> gold_copy = gold;
    fk::vector<TestType, mem_type::view> const gold_copy_v(gold_copy);
    fk::vector<TestType, mem_type::const_view> const gold_copy_cv(gold);

    fk::vector<TestType> gold_prefix{1, 2, 3, 4};
    fk::vector<TestType, mem_type::view> const gold_prefix_v(gold_prefix);
    fk::vector<TestType, mem_type::const_view> const gold_prefix_cv(
        gold_prefix);

    fk::vector<TestType> mismatch{2, 3, 5, 5, 6};
    fk::vector<TestType, mem_type::view> const mismatch_v(mismatch);
    fk::vector<TestType, mem_type::const_view> const mismatch_cv(mismatch);

    // equal vectors return false
    REQUIRE(!(gold_copy < gold));
    REQUIRE(!(gold_copy_v < gold));
    REQUIRE(!(gold_copy_cv < gold));
    // empty range less than non-empty range
    REQUIRE(empty < gold);
    REQUIRE(empty_v < gold);
    REQUIRE(empty_cv < gold);
    // a prefix is less than the complete range
    REQUIRE(gold_prefix < gold);
    REQUIRE(gold_prefix_v < gold);
    REQUIRE(gold_prefix_cv < gold);
    // otherwise compare on first mismatch
    REQUIRE(gold < mismatch);
    REQUIRE(gold < mismatch_v);
    REQUIRE(gold < mismatch_cv);
    // also, empty ranges are equal
    REQUIRE(!(empty < empty));
    REQUIRE(!(empty < empty_v));
    REQUIRE(!(empty < empty_cv));
  }

  SECTION("addition operator")
  {
    fk::vector<TestType> in1{1, 1, 1, 1, 1};
    fk::vector<TestType, mem_type::view> const in1_v(in1);
    fk::vector<TestType, mem_type::const_view> const in1_cv(in1);

    fk::vector<TestType> in2{1, 2, 3, 4, 5};
    fk::vector<TestType, mem_type::view> const in2_v(in2);
    fk::vector<TestType, mem_type::const_view> const in2_cv(in2);

    REQUIRE((in1 + in2) == gold);
    REQUIRE((in1 + in2_v) == gold);
    REQUIRE((in1 + in2_cv) == gold);

    REQUIRE((in1_v + in2) == gold);
    REQUIRE((in1_v + in2_v) == gold);
    REQUIRE((in1_v + in2_cv) == gold);

    REQUIRE((in1_cv + in2) == gold);
    REQUIRE((in1_cv + in2_v) == gold);
    REQUIRE((in1_cv + in2_cv) == gold);
  }

  SECTION("subtraction operator")
  {
    fk::vector<TestType> in1{3, 4, 5, 6, 7};
    fk::vector<TestType, mem_type::view> const in1_v(in1);
    fk::vector<TestType, mem_type::const_view> const in1_cv(in1);

    fk::vector<TestType> in2{1, 1, 1, 1, 1};
    fk::vector<TestType, mem_type::view> const in2_v(in2);
    fk::vector<TestType, mem_type::const_view> const in2_cv(in2);

    REQUIRE((in1 - in2) == gold);
    REQUIRE((in1 - in2_v) == gold);
    REQUIRE((in1 - in2_cv) == gold);

    REQUIRE((in1_v - in2) == gold);
    REQUIRE((in1_v - in2_v) == gold);
    REQUIRE((in1_v - in2_cv) == gold);

    REQUIRE((in1_cv - in2) == gold);
    REQUIRE((in1_cv - in2_v) == gold);
    REQUIRE((in1_cv - in2_cv) == gold);
  }
#ifdef ASGARD_USE_CUDA
  SECTION("vector*vector operator - device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType, mem_type::owner, resource::device> gold_d(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::view, resource::device> const gold_v_d(
          gold_d);
      fk::vector<TestType, mem_type::const_view, resource::device> const
          gold_cv_d(gold_d);

      TestType const inner_prod = 90;

      REQUIRE((gold_d * gold_d) == inner_prod);
      REQUIRE((gold_d * gold_v_d) == inner_prod);
      REQUIRE((gold_d * gold_cv_d) == inner_prod);

      REQUIRE((gold_v_d * gold_d) == inner_prod);
      REQUIRE((gold_v_d * gold_v_d) == inner_prod);
      REQUIRE((gold_v_d * gold_cv_d) == inner_prod);

      REQUIRE((gold_cv_d * gold_d) == inner_prod);
      REQUIRE((gold_cv_d * gold_v_d) == inner_prod);
      REQUIRE((gold_cv_d * gold_cv_d) == inner_prod);
    }
  }
#endif
  SECTION("vector*vector operator")
  {
    fk::vector<TestType> gold_copy(gold);
    fk::vector<TestType, mem_type::view> const gold_v(gold_copy);
    fk::vector<TestType, mem_type::const_view> const gold_cv(gold);

    TestType const inner_prod = 90;

    REQUIRE((gold * gold) == inner_prod);
    REQUIRE((gold * gold_v) == inner_prod);
    REQUIRE((gold * gold_cv) == inner_prod);

    REQUIRE((gold_v * gold) == inner_prod);
    REQUIRE((gold_v * gold_v) == inner_prod);
    REQUIRE((gold_v * gold_cv) == inner_prod);

    REQUIRE((gold_cv * gold) == inner_prod);
    REQUIRE((gold_cv * gold_v) == inner_prod);
    REQUIRE((gold_cv * gold_cv) == inner_prod);
  }

  SECTION("vector*matrix operator")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      // clang-format off
    fk::matrix<TestType> test_mat {
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {16, 26, 36},
    }; // clang-format on

      fk::matrix<TestType, mem_type::view> const test_mat_v(test_mat);
      fk::matrix<TestType, mem_type::const_view> const test_mat_cv(test_mat);

      fk::vector<TestType> test_vect(gold);
      fk::vector<TestType, mem_type::view> const test_vect_v(test_vect);
      fk::vector<TestType, mem_type::const_view> const test_vect_cv(test_vect);
      fk::vector<TestType> const gold_result{290, 490, 690};

      REQUIRE((test_vect * test_mat) == gold_result);
      REQUIRE((test_vect_v * test_mat) == gold_result);
      REQUIRE((test_vect_cv * test_mat) == gold_result);

      REQUIRE((test_vect * test_mat_v) == gold_result);
      REQUIRE((test_vect_v * test_mat_v) == gold_result);
      REQUIRE((test_vect_cv * test_mat_v) == gold_result);

      REQUIRE((test_vect * test_mat_cv) == gold_result);
      REQUIRE((test_vect_v * test_mat_cv) == gold_result);
      REQUIRE((test_vect_cv * test_mat_cv) == gold_result);
    }
  }

  SECTION("vector*scalar operator")
  {
    TestType const scale = TestType{-2};
    fk::vector<TestType> const gold_scaled{-4, -6, -8, -10, -12};
    fk::vector<TestType> gold_copy(gold);
    fk::vector<TestType> const gold_v(gold_copy);
    fk::vector<TestType, mem_type::const_view> const gold_cv(gold);
    REQUIRE((gold * scale) == gold_scaled);
    REQUIRE((gold_v * scale) == gold_scaled);
    REQUIRE((gold_cv * scale) == gold_scaled);
  }

  SECTION("vector (as matrix) kron product")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType> gold_copy(gold);
      fk::vector<TestType, mem_type::view> const gold_v(gold_copy);
      fk::vector<TestType, mem_type::const_view> const gold_cv(gold);

      fk::vector<TestType> identity{1};
      fk::vector<TestType, mem_type::view> const identity_v(identity);
      fk::vector<TestType, mem_type::const_view> const identity_cv(identity);

      REQUIRE(identity.single_column_kron(gold) == gold);
      REQUIRE(identity_v.single_column_kron(gold) == gold);
      REQUIRE(identity_cv.single_column_kron(gold) == gold);

      REQUIRE(identity.single_column_kron(gold_v) == gold);
      REQUIRE(identity_v.single_column_kron(gold_v) == gold);
      REQUIRE(identity_cv.single_column_kron(gold_v) == gold);

      REQUIRE(identity.single_column_kron(gold_cv) == gold);
      REQUIRE(identity_v.single_column_kron(gold_cv) == gold);
      REQUIRE(identity_cv.single_column_kron(gold_cv) == gold);

      fk::vector<TestType> const gold_repeated =
          fk::vector<TestType>(gold).concat(gold);

      fk::vector<TestType> repeat{1, 1};
      fk::vector<TestType, mem_type::view> const repeat_v(repeat);
      fk::vector<TestType, mem_type::const_view> const repeat_cv(repeat);

      REQUIRE(repeat.single_column_kron(gold) == gold_repeated);
      REQUIRE(repeat.single_column_kron(gold_v) == gold_repeated);
      REQUIRE(repeat.single_column_kron(gold_cv) == gold_repeated);

      REQUIRE(repeat_v.single_column_kron(gold) == gold_repeated);
      REQUIRE(repeat_v.single_column_kron(gold_v) == gold_repeated);
      REQUIRE(repeat_v.single_column_kron(gold_cv) == gold_repeated);

      REQUIRE(repeat_cv.single_column_kron(gold) == gold_repeated);
      REQUIRE(repeat_cv.single_column_kron(gold_v) == gold_repeated);
      REQUIRE(repeat_cv.single_column_kron(gold_cv) == gold_repeated);

      fk::vector<TestType> const zeros(gold.size());
      fk::vector<TestType> alternate{1, 0, 2, 0};
      fk::vector<TestType, mem_type::view> const alternate_v(alternate);
      fk::vector<TestType, mem_type::const_view> const alternate_cv(alternate);
      fk::vector<TestType> const ans =
          fk::vector<TestType>(gold).concat(zeros).concat(gold * 2).concat(
              zeros);

      REQUIRE(ans == alternate.single_column_kron(gold));
      REQUIRE(ans == alternate.single_column_kron(gold_v));
      REQUIRE(ans == alternate.single_column_kron(gold_cv));

      REQUIRE(ans == alternate_v.single_column_kron(gold));
      REQUIRE(ans == alternate_v.single_column_kron(gold_v));
      REQUIRE(ans == alternate_v.single_column_kron(gold_cv));

      REQUIRE(ans == alternate_cv.single_column_kron(gold));
      REQUIRE(ans == alternate_cv.single_column_kron(gold_v));
      REQUIRE(ans == alternate_cv.single_column_kron(gold_cv));
    }
  }

  SECTION("vector scale in place")
  {
    TestType const x = 2.0;
    fk::vector<TestType> test(gold);
    fk::vector<TestType> test_own(gold);
    fk::vector<TestType, mem_type::view> test_view(test_own);

    fk::vector<TestType> const ans = {4, 6, 8, 10, 12};

    REQUIRE(test.scale(x) == ans);
    REQUIRE(test_view.scale(x) == ans);
    REQUIRE(test_own == ans);

    test     = gold;
    test_own = gold;

    TestType const x2 = 0.0;
    fk::vector<TestType> const zeros(gold.size());

    REQUIRE(test.scale(x2) == zeros);
    REQUIRE(test_view.scale(x2) == zeros);
    REQUIRE(test_own == zeros);
  }

#ifdef ASGARD_USE_CUDA
  SECTION("vector scale in place")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      TestType const x = 2.0;
      fk::vector<TestType, mem_type::owner, resource::device> test(gold.size());
      fk::copy_vector(test, gold);
      fk::vector<TestType, mem_type::owner, resource::device> test_own(test);
      fk::vector<TestType, mem_type::view, resource::device> test_view(
          test_own);

      fk::vector<TestType> const ans = {4, 6, 8, 10, 12};

      REQUIRE(test.scale(x).clone_onto_host() == ans);
      REQUIRE(test_view.scale(x).clone_onto_host() == ans);
      REQUIRE(test_own.clone_onto_host() == ans);

      fk::copy_vector(test, gold);
      fk::copy_vector(test_own, gold);

      TestType const x2 = 0.0;
      fk::vector<TestType> const zeros(gold.size());

      REQUIRE(test.scale(x2).clone_onto_host() == zeros);
      REQUIRE(test_view.scale(x2).clone_onto_host() == zeros);
      REQUIRE(test_own.clone_onto_host() == zeros);
    }
  }
#endif
} // end fk::vector operators

TEMPLATE_TEST_CASE("fk::vector utilities", "[tensors]", test_precs, int)
{
  fk::vector<TestType> const gold{2, 3, 4, 5, 6};
  fk::vector<TestType> gold_copy(gold);
  fk::vector<TestType, mem_type::view> const gold_v(gold_copy);
  fk::vector<TestType, mem_type::const_view> const gold_cv(gold);

  SECTION("size(): the number of elements")
  {
    REQUIRE(gold.size() == 5);
    REQUIRE(gold_v.size() == 5);
    REQUIRE(gold_cv.size() == 5);
  }
  SECTION("empty(): the number of elements")
  {
    fk::vector<TestType> const gold_empty;
    REQUIRE(gold_empty.empty());
    REQUIRE(!gold.empty());
    REQUIRE(!gold_v.empty());
    REQUIRE(!gold_cv.empty());
  }
  SECTION("data(): const addr to element")
  {
    REQUIRE(*gold.data(4) == 6);
    REQUIRE(*gold_v.data(4) == 6);
    REQUIRE(*gold_cv.data(4) == 6);
  }
  SECTION("print out the values")
  {
    // (effectively) redirect cout
    std::streambuf *old_cout_stream_buf = std::cout.rdbuf();
    std::ostringstream test_str;
    std::ostringstream test_str_v;
    std::ostringstream test_str_cv;

    // generate the output (into test_str)
    std::cout.rdbuf(test_str.rdbuf());
    gold.print("golden vector");
    std::cout.rdbuf(test_str_v.rdbuf());
    gold_v.print("golden vector");
    std::cout.rdbuf(test_str_cv.rdbuf());
    gold_cv.print("golden vector");

    // restore cout destination
    std::cout.rdbuf(old_cout_stream_buf);

    std::string golden_string, golden_string_v, golden_string_cv;
    if constexpr (std::is_floating_point_v<TestType>)
    {
      golden_string = "golden vector(owner)\n  2.0000e+00  3.0000e+00  "
                      "4.0000e+00  5.0000e+00  6.0000e+00\n";
      golden_string_v = "golden vector(view)\n  2.0000e+00  3.0000e+00  "
                        "4.0000e+00  5.0000e+00  6.0000e+00\n";
      golden_string_cv = "golden vector(const view)\n  2.0000e+00  3.0000e+00  "
                         "4.0000e+00  5.0000e+00  6.0000e+00\n";
    }
    else
    {
      golden_string = "golden vector(owner)\n2 3 "
                      "4 5 6 \n";
      golden_string_v = "golden vector(view)\n2 3 "
                        "4 5 6 \n";
      golden_string_cv = "golden vector(const view)\n2 3 "
                         "4 5 6 \n";
    }
    REQUIRE(test_str.str() == golden_string);
    REQUIRE(test_str_v.str() == golden_string_v);
  }
  SECTION("dump to octave")
  {
    gold.dump_to_octave("test_out.dat");
    gold_v.dump_to_octave("test_out_v.dat");
    gold_cv.dump_to_octave("test_out_cv.dat");

    std::ifstream data_stream("test_out.dat");
    std::ifstream data_stream_v("test_out_v.dat");
    std::ifstream data_stream_cv("test_out_cv.dat");

    std::string const test_string((std::istreambuf_iterator<char>(data_stream)),
                                  std::istreambuf_iterator<char>());
    std::string const test_string_v(
        (std::istreambuf_iterator<char>(data_stream_v)),
        std::istreambuf_iterator<char>());
    std::string const test_string_cv(
        (std::istreambuf_iterator<char>(data_stream_cv)),
        std::istreambuf_iterator<char>());

    REQUIRE(std::filesystem::remove("test_out.dat"));
    REQUIRE(std::filesystem::remove("test_out_v.dat"));
    REQUIRE(std::filesystem::remove("test_out_cv.dat"));

    std::string golden_string;
    if constexpr (std::is_floating_point_v<TestType>)
    {
      golden_string =
          "2.000000000000e+00 3.000000000000e+00 4.000000000000e+00 "
          "5.000000000000e+00 6.000000000000e+00 ";
    }
    else
    {
      golden_string = "2 3 4 5 6 ";
    }

    REQUIRE(test_string == golden_string);
    REQUIRE(test_string_v == golden_string);
    REQUIRE(test_string_cv == golden_string);
  }
  SECTION("vector resize")
  {
    fk::vector<TestType> test_reduced{2, 3, 4, 5, 6, 7, 8};
    // fk::vector<TestType, mem_type::view> test_reduced_v(test_reduced);
    // disabled

    fk::vector<TestType> const gold_enlarged{2, 3, 4, 0, 0};

    fk::vector<TestType> test_enlarged{2, 3, 4};
    // fk::vector<TestType, mem_type::view> test_enlarged_v(test_enlarged);

    test_reduced.resize(gold.size());
    test_enlarged.resize(gold.size());
    // test_reduced_v.resize(gold.size());
    // test_enlarged_v.resize(gold.size());

    REQUIRE(test_reduced == gold);
    REQUIRE(test_enlarged == gold_enlarged);

    // REQUIRE(test_reduced_v == gold);
    // REQUIRE(test_enlarged_v == gold_enlarged);
  }
  SECTION("vector resize from 0")
  {
    fk::vector<TestType> const gold_enlarged{0, 0, 0, 0, 0};
    fk::vector<TestType> test_enlarged;

    test_enlarged.resize(gold_enlarged.size());

    REQUIRE(test_enlarged == gold_enlarged);
  }
  SECTION("vector resize to 0")
  {
    fk::vector<TestType> const gold_reduced;
    fk::vector<TestType> test_reduced{1, 2, 3, 4, 5};

    test_reduced.resize(gold_reduced.size());

    REQUIRE(test_reduced == gold_reduced);
  }
  SECTION("vector concatenation")
  {
    fk::vector<TestType> test_left = {2, 3, 4};
    // fk::vector<TestType, mem_type::view> test_left_v(test_left); // disabled
    fk::vector<TestType> const test_right = {5, 6};
    fk::vector<TestType, mem_type::const_view> const test_right_v(test_right);

    REQUIRE(test_left.concat(test_right) == gold);
    test_left.resize(3) = fk::vector<TestType>({2, 3, 4});
    REQUIRE(test_left.concat(test_right_v) == gold);

    // REQUIRE(test_left_v.concat(test_right_v) == gold); // disabled

    fk::vector<TestType> empty;
    // have to make a copy to extract a view from,
    // because you can't concat on an owner w/ outstanding views
    fk::vector<TestType> empty_copy(empty);
    fk::vector<TestType, mem_type::view> empty_v(empty_copy);
    fk::vector<TestType> gold_data_copy(gold);
    fk::vector<TestType, mem_type::view> gold_data_v(gold_data_copy);

    REQUIRE(empty.concat(gold) == gold);
    empty.resize(0);
    REQUIRE(empty.concat(gold_data_v) == gold);
    empty.resize(0);

    // non-const gold copy I can concat with
    fk::vector<TestType> gold_2(gold);
    REQUIRE(gold_2.concat(empty) == gold);
    gold_2.resize(gold.size()) = gold;
    REQUIRE(gold_2.concat(empty_v) == gold);
    gold_2.resize(gold.size()) = gold;
  }

  SECTION("vector set")
  {
    fk::vector<TestType> vector(5);
    fk::vector<TestType> vector_copy(vector);
    fk::vector<TestType, mem_type::view> vector_v(vector_copy);

    fk::vector<TestType> const empty;
    fk::vector<TestType> const begin  = {2, 3};
    fk::vector<TestType> const middle = {3, 4, 5};
    fk::vector<TestType> const end    = {6};
    fk::vector<TestType> const empty_copy(empty);
    fk::vector<TestType> const begin_copy(begin);
    fk::vector<TestType> const middle_copy(middle);
    fk::vector<TestType> const end_copy(end);

    fk::vector<TestType, mem_type::const_view> const empty_v(empty_copy);
    fk::vector<TestType, mem_type::const_view> const begin_v(begin_copy);
    fk::vector<TestType, mem_type::const_view> const middle_v(middle_copy);
    fk::vector<TestType, mem_type::const_view> const end_v(end_copy);

    REQUIRE(vector.set_subvector(0, begin)
                .set_subvector(0, empty)
                .set_subvector(1, middle)
                .set_subvector(4, end) == gold);
    vector = fk::vector<TestType>(5);
    REQUIRE(vector.set_subvector(0, begin_v)
                .set_subvector(0, empty_v)
                .set_subvector(1, middle_v)
                .set_subvector(4, end) == gold);
    vector = fk::vector<TestType>(5);
    REQUIRE(vector_v.set_subvector(0, begin)
                .set_subvector(0, empty)
                .set_subvector(1, middle)
                .set_subvector(4, end) == gold);
    vector_v = vector;
    REQUIRE(vector_v.set_subvector(0, begin_v)
                .set_subvector(0, empty_v)
                .set_subvector(1, middle_v)
                .set_subvector(4, end_v) == gold);
  }

  SECTION("vector extract")
  {
    fk::vector<TestType> const test_begin  = {2, 3, 4};
    fk::vector<TestType> const test_middle = {4, 5};
    fk::vector<TestType> const test_end    = {5, 6};

    REQUIRE(test_begin == gold.extract(0, 2));
    REQUIRE(test_middle == gold.extract(2, 3));
    REQUIRE(test_end == gold.extract(3, 4));

    REQUIRE(test_begin == gold_v.extract(0, 2));
    REQUIRE(test_middle == gold_v.extract(2, 3));
    REQUIRE(test_end == gold_v.extract(3, 4));

    REQUIRE(test_begin == gold_cv.extract(0, 2));
    REQUIRE(test_middle == gold_cv.extract(2, 3));
    REQUIRE(test_end == gold_cv.extract(3, 4));
  }
#ifdef ASGARD_USE_CUDA
  SECTION("vector extract")
  {
    fk::vector<TestType> const test_begin  = {2, 3, 4};
    fk::vector<TestType> const test_middle = {4, 5};
    fk::vector<TestType> const test_end    = {5, 6};

    auto const gold_d = gold.clone_onto_device();
    fk::vector<TestType, mem_type::owner, resource::device> gold_copy_d(gold_d);
    fk::vector<TestType, mem_type::view, resource::device> const gold_v_d(
        gold_copy_d);
    fk::vector<TestType, mem_type::const_view, resource::device> const
        gold_cv_d(gold_d);

    REQUIRE(test_begin == gold_d.extract(0, 2).clone_onto_host());
    REQUIRE(test_middle == gold_d.extract(2, 3).clone_onto_host());
    REQUIRE(test_end == gold_d.extract(3, 4).clone_onto_host());

    REQUIRE(test_begin == gold_v_d.extract(0, 2).clone_onto_host());
    REQUIRE(test_middle == gold_v_d.extract(2, 3).clone_onto_host());
    REQUIRE(test_end == gold_v_d.extract(3, 4).clone_onto_host());

    REQUIRE(test_begin == gold_cv_d.extract(0, 2).clone_onto_host());
    REQUIRE(test_middle == gold_cv_d.extract(2, 3).clone_onto_host());
    REQUIRE(test_end == gold_cv_d.extract(3, 4).clone_onto_host());
  }
#endif
  SECTION("vector transform")
  {
    fk::vector<TestType> test{-1, 1, 2, 3};
    fk::vector<TestType> test_copy(test);
    fk::vector<TestType, mem_type::view> test_v(test_copy);
    fk::vector<TestType> const after{0, 2, 3, 4};
    std::transform(test.begin(), test.end(), test.begin(),
                   std::bind(std::plus<TestType>(), std::placeholders::_1, 1));
    std::transform(test_v.begin(), test_v.end(), test_v.begin(),
                   std::bind(std::plus<TestType>(), std::placeholders::_1, 1));
    REQUIRE(test == after);
    REQUIRE(test_copy == after);
    REQUIRE(test_v == after);
  }

  SECTION("vector maximum element")
  {
    fk::vector<TestType> test{5, 6, 11, 8};
    fk::vector<TestType> test_v(test);
    fk::vector<TestType, mem_type::const_view> const test_cv(test);
    TestType const max = 11;
    REQUIRE(*std::max_element(test.begin(), test.end()) == max);
    REQUIRE(*std::max_element(test_v.begin(), test_v.end()) == max);
    REQUIRE(*std::max_element(test_cv.begin(), test_cv.end()) == max);
  }

  SECTION("vector sum of elements")
  {
    fk::vector<TestType> test{1, 2, 3, 4, 5, 6, 7, 8};
    fk::vector<TestType, mem_type::view> const test_v(test);
    fk::vector<TestType, mem_type::const_view> const test_cv(test);
    TestType const sum = 36;
    REQUIRE(std::accumulate(test.begin(), test.end(), TestType{0}) == sum);
    REQUIRE(std::accumulate(test_v.begin(), test_v.end(), TestType{0}) == sum);
    REQUIRE(std::accumulate(test_cv.begin(), test_cv.end(), TestType{0}) ==
            sum);
  }
} // end fk::vector utilities

TEMPLATE_TEST_CASE("fk::vector device functions", "[tensors]", test_precs, int)
{
  fk::vector<TestType> const gold = {1, 3, 5, 7, 9};

  SECTION("ctors")
  {
#ifdef ASGARD_USE_CUDA
    // default
    {
      fk::vector<TestType, mem_type::owner, resource::device> const vect;
      REQUIRE(vect.empty());
      REQUIRE(vect.data() == nullptr);
    }
    // from init list
    {
      fk::vector<TestType, mem_type::owner, resource::device> const vect(
          {1, 3, 5, 7, 9});
      fk::vector<TestType, mem_type::owner, resource::host> const copy(
          vect.clone_onto_host());
      REQUIRE(copy == gold);
    }
    // from size w/ copy to device
    {
      fk::vector<TestType, mem_type::owner, resource::device> vect(5);
      vect.transfer_from(gold);
      fk::vector<TestType, mem_type::owner, resource::host> const copy(
          vect.clone_onto_host());
      REQUIRE(copy == gold);
    }

    // transfer - new vector - owner device to host
    {
      fk::vector<TestType, mem_type::owner, resource::device> const vect(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::owner, resource::host> const copy(
          vect.clone_onto_host());
      REQUIRE(copy == gold);
    }

    // transfer - new vector - owner host to device
    {
      fk::vector<TestType, mem_type::owner, resource::device> const copy(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::owner, resource::host> const vect_h(
          copy.clone_onto_host());
      REQUIRE(vect_h == gold);
    }

    // transfer - new vector - view device to host
    {
      fk::vector<TestType, mem_type::owner, resource::device> const vect(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::const_view, resource::device> const
          vect_view(vect);
      fk::vector<TestType, mem_type::owner, resource::host> const copy(
          vect_view.clone_onto_host());
      REQUIRE(copy == gold);
    }

    // transfer - new vector - view host to device
    {
      fk::vector<TestType, mem_type::const_view, resource::host> const
          vect_view(gold);
      fk::vector<TestType, mem_type::owner, resource::device> const copy(
          vect_view.clone_onto_device());
      fk::vector<TestType, mem_type::owner, resource::host> const vect_h(
          copy.clone_onto_host());
      REQUIRE(vect_h == gold);
    }
#endif
    SECTION("views from matrix constructor")
    {
      fk::matrix<TestType> base{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};

      fk::matrix<TestType, mem_type::view> view(base, 1, 2, 1, 2);

      {
        // create vector from first column in base
        int const col     = 0;
        int const r_start = 0;
        int const r_end   = 2;
        fk::vector<TestType, mem_type::view> from_owner(base, col, r_start,
                                                        r_end);

        // create vector from partial column in view
        fk::vector<TestType, mem_type::view> from_view(view, 1, 1, 1);

        fk::vector<TestType> const gold_initial{0, 3, 6};
        fk::vector<TestType> const gold_initial_v{8};

        REQUIRE(from_owner == gold_initial);
        REQUIRE(from_view == gold_initial_v);

        from_owner(1) = 8;
        from_view(0)  = 15;

        fk::matrix<TestType> const after_mod{{0, 1, 2}, {8, 4, 5}, {6, 7, 15}};

        fk::matrix<TestType> const after_mod_v{{4, 5}, {7, 15}};

        fk::vector<TestType> const gold_mod{0, 8, 6};
        fk::vector<TestType> const gold_mod_v{15};

        REQUIRE(from_owner == gold_mod);
        REQUIRE(from_view == gold_mod_v);
        REQUIRE(base == after_mod);
        REQUIRE(view == after_mod_v);
      }
    }

    SECTION("const views from vector constructor")
    {
      fk::vector<TestType> const base{0, 1, 2, 3, 4, 5, 6, 7};

      fk::vector<TestType, mem_type::const_view> const view(base, 1, 7);

      {
        // create 2x3 matrix from last six elems in base
        fk::matrix<TestType, mem_type::const_view> const from_owner(base, 2, 3,
                                                                    2);
        // create 2x2 matrix from middle of view
        fk::matrix<TestType, mem_type::const_view> const from_view(view, 2, 2,
                                                                   1);

        // clang-format off
      fk::matrix<TestType> const gold_initial   {{2, 4, 6},
					         {3, 5, 7}};
      fk::matrix<TestType> const gold_initial_v {{2, 4},
					         {3, 5}};
        // clang-format on

        REQUIRE(from_owner == gold_initial);
        REQUIRE(from_view == gold_initial_v);
      }
      fk::matrix<TestType, mem_type::const_view> view_2(base, 1, 7);
      fk::matrix<TestType, mem_type::const_view> const view_m(
          std::move(view_2));
    }
  }
#ifdef ASGARD_USE_CUDA
  SECTION("copy and move")
  {
    // copy owner
    {
      fk::vector<TestType, mem_type::owner, resource::device> const vect(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::owner, resource::device> const copy_d(
          vect);
      fk::vector<TestType, mem_type::owner, resource::host> const copy_h(
          copy_d.clone_onto_host());
      REQUIRE(copy_h == gold);
    }

    // move owner
    {
      fk::vector<TestType, mem_type::owner, resource::device> vect(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::owner, resource::device> const moved_d(
          std::move(vect));
      REQUIRE(vect.data() == nullptr);
      REQUIRE(vect.empty());
      fk::vector<TestType, mem_type::owner, resource::host> const moved_h(
          moved_d.clone_onto_host());
      REQUIRE(moved_h == gold);
    }

    // copy view
    {
      fk::vector<TestType, mem_type::owner, resource::device> vect(
          gold.clone_onto_device());

      fk::vector<TestType, mem_type::view, resource::device> const view_d(vect);
      fk::vector<TestType, mem_type::view, resource::device> const view_copy_d(
          view_d);
      fk::vector<TestType, mem_type::owner, resource::host> const copy_h(
          view_copy_d.clone_onto_host());
      REQUIRE(copy_h == gold);
    }
    // copy const view
    {
      fk::vector<TestType, mem_type::owner, resource::device> const vect(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::const_view, resource::device> const view_d(
          vect);
      fk::vector<TestType, mem_type::const_view, resource::device> const
          view_copy_d(view_d);

      fk::vector<TestType, mem_type::owner, resource::host> const copy_h(
          view_copy_d.clone_onto_host());
      REQUIRE(copy_h == gold);
    }

    // move view
    {
      fk::vector<TestType, mem_type::owner, resource::device> vect(
          gold.clone_onto_device());

      fk::vector<TestType, mem_type::view, resource::device> view_d(vect);

      fk::vector<TestType, mem_type::view, resource::device> view_moved_d(
          std::move(view_d));
      REQUIRE(view_d.data() == nullptr);
      REQUIRE(view_d.empty());
      REQUIRE(view_moved_d.data() == vect.data());

      fk::vector<TestType, mem_type::owner, resource::host> const moved_h(
          view_moved_d.clone_onto_host());
      REQUIRE(moved_h == gold);
    }

    // move immutable view
    {
      fk::vector<TestType, mem_type::owner, resource::device> const vect(
          gold.clone_onto_device());

      fk::vector<TestType, mem_type::const_view, resource::device> view_d(vect);

      fk::vector<TestType, mem_type::const_view, resource::device> view_moved_d(
          std::move(view_d));
      REQUIRE(view_d.data() == nullptr);
      REQUIRE(view_d.empty());
      REQUIRE(view_moved_d.data() == vect.data());

      fk::vector<TestType, mem_type::owner, resource::host> const moved_h(
          view_moved_d.clone_onto_host());
      REQUIRE(moved_h == gold);
    }
  }
  SECTION("transfer copies and assignments")
  {
    // owner device to owner host
    {
      fk::vector<TestType, mem_type::owner, resource::device> const vect(
          gold.clone_onto_device());
      fk::vector<TestType> vect_h(5);
      vect_h.transfer_from(vect);
      REQUIRE(vect_h == gold);
    }

    // owner device to view host
    {
      fk::vector<TestType, mem_type::owner, resource::device> const vect(
          gold.clone_onto_device());
      fk::vector<TestType> vect_h(5);
      fk::vector<TestType, mem_type::view> vect_view(vect_h);
      vect_view.transfer_from(vect);
      REQUIRE(vect_view == gold);
    }

    // owner device to owner device
    {
      fk::vector<TestType, mem_type::owner, resource::device> const vect(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::owner, resource::device> vect_d(5);
      vect_d = vect;
      fk::vector<TestType> const vect_h(vect_d.clone_onto_host());
      REQUIRE(vect_h == gold);
    }

    // owner device to view device
    {
      fk::vector<TestType, mem_type::owner, resource::device> const vect(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::owner, resource::device> vect_d(5);
      fk::vector<TestType, mem_type::view, resource::device> vect_view(vect_d);
      vect_view = vect;
      fk::vector<TestType, mem_type::owner> const vect_h(
          vect_view.clone_onto_host());
      REQUIRE(vect_h == gold);
    }

    // view device to owner host
    {
      fk::vector<TestType, mem_type::owner, resource::device> vect(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::view, resource::device> const vect_view(
          vect);
      fk::vector<TestType, mem_type::owner, resource::host> vect_h(5);
      vect_h.transfer_from(vect_view);
      REQUIRE(vect_h == gold);
    }

    // view device to owner device
    {
      fk::vector<TestType, mem_type::owner, resource::device> vect(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::view, resource::device> const vect_view(
          vect);
      fk::vector<TestType, mem_type::owner, resource::device> vect_d(5);
      vect_d = vect_view;
      fk::vector<TestType, mem_type::owner, resource::host> const vect_h(
          vect_d.clone_onto_host());
      REQUIRE(vect_h == gold);
    }

    // view device to view host
    {
      fk::vector<TestType, mem_type::owner, resource::device> vect(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::view, resource::device> const vect_view(
          vect);
      fk::vector<TestType, mem_type::owner, resource::host> vect_h(5);
      fk::vector<TestType, mem_type::view, resource::host> vect_view_h(vect_h);
      vect_view_h.transfer_from(vect_view);
      REQUIRE(vect_view_h == gold);
    }

    // view device to view device
    {
      fk::vector<TestType, mem_type::owner, resource::device> vect(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::view, resource::device> const vect_view(
          vect);
      fk::vector<TestType, mem_type::owner, resource::device> vect_d(5);
      fk::vector<TestType, mem_type::view, resource::device> vect_view_2(
          vect_d);
      vect_view_2 = vect_view;
      fk::vector<TestType, mem_type::owner, resource::host> const vect_h(
          vect_view_2.clone_onto_host());
      REQUIRE(vect_h == gold);
    }

    // const view device to owner host
    {
      fk::vector<TestType, mem_type::owner, resource::device> const vect(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::const_view, resource::device> const
          vect_view(vect);
      fk::vector<TestType, mem_type::owner, resource::host> vect_h(5);
      vect_h.transfer_from(vect_view);
      REQUIRE(vect_h == gold);
    }

    // const view device to owner device
    {
      fk::vector<TestType, mem_type::owner, resource::device> const vect(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::const_view, resource::device> const
          vect_view(vect);
      fk::vector<TestType, mem_type::owner, resource::device> vect_d(5);
      vect_d = vect_view;
      fk::vector<TestType, mem_type::owner, resource::host> const vect_h(
          vect_d.clone_onto_host());
      REQUIRE(vect_h == gold);
    }

    // const view device to view host
    {
      fk::vector<TestType, mem_type::owner, resource::device> const vect(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::const_view, resource::device> const
          vect_view(vect);
      fk::vector<TestType, mem_type::owner, resource::host> vect_h(5);
      fk::vector<TestType, mem_type::view, resource::host> vect_view_h(vect_h);
      vect_view_h.transfer_from(vect_view);
      REQUIRE(vect_view_h == gold);
    }

    // const view device to view device
    {
      fk::vector<TestType, mem_type::owner, resource::device> const vect(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::const_view, resource::device> const
          vect_view(vect);
      fk::vector<TestType, mem_type::owner, resource::device> vect_d(5);
      fk::vector<TestType, mem_type::view, resource::device> vect_view_2(
          vect_d);
      vect_view_2 = vect_view;
      fk::vector<TestType, mem_type::owner, resource::host> const vect_h(
          vect_view_2.clone_onto_host());
      REQUIRE(vect_h == gold);
    }

    // owner host to owner device
    {
      fk::vector<TestType, mem_type::owner, resource::host> const vect(gold);
      fk::vector<TestType, mem_type::owner, resource::device> vect_d(5);
      vect_d.transfer_from(vect);
      fk::vector<TestType> const vect_h(vect_d.clone_onto_host());
      REQUIRE(vect_h == gold);
    }

    // owner host to view device
    {
      fk::vector<TestType, mem_type::owner, resource::host> const vect(gold);
      fk::vector<TestType, mem_type::owner, resource::device> vect_d(5);
      fk::vector<TestType, mem_type::view, resource::device> vect_view(vect_d);
      vect_view.transfer_from(vect);
      fk::vector<TestType> const vect_h(vect_view.clone_onto_host());
      REQUIRE(vect_h == gold);
    }

    // view host to owner device
    {
      fk::vector<TestType, mem_type::owner, resource::host> vect(gold);
      fk::vector<TestType, mem_type::view, resource::host> const vect_view(
          vect);
      fk::vector<TestType, mem_type::owner, resource::device> vect_d(5);
      vect_d.transfer_from(vect_view);
      fk::vector<TestType> const vect_h(vect_d.clone_onto_host());
      REQUIRE(vect_h == gold);
    }

    // view host to view device
    {
      fk::vector<TestType, mem_type::owner, resource::host> vect(gold);
      fk::vector<TestType, mem_type::view, resource::host> const vect_view(
          vect);
      fk::vector<TestType, mem_type::owner, resource::device> vect_d(5);
      fk::vector<TestType, mem_type::view, resource::device> vect_view_d(
          vect_d);
      vect_view_d.transfer_from(vect_view);
      fk::vector<TestType> const vect_h(vect_view_d.clone_onto_host());
      REQUIRE(vect_h == gold);
    }

    // const view host to owner device
    {
      fk::vector<TestType, mem_type::owner, resource::host> const vect(gold);
      fk::vector<TestType, mem_type::const_view, resource::host> const
          vect_view(vect);
      fk::vector<TestType, mem_type::owner, resource::device> vect_d(5);
      vect_d.transfer_from(vect_view);
      fk::vector<TestType> const vect_h(vect_d.clone_onto_host());
      REQUIRE(vect_h == gold);
    }

    // const view host to view device
    {
      fk::vector<TestType, mem_type::owner, resource::host> const vect(gold);
      fk::vector<TestType, mem_type::const_view, resource::host> const
          vect_view(vect);
      fk::vector<TestType, mem_type::owner, resource::device> vect_d(5);
      fk::vector<TestType, mem_type::view, resource::device> vect_view_d(
          vect_d);
      vect_view_d.transfer_from(vect_view);
      fk::vector<TestType> const vect_h(vect_view_d.clone_onto_host());
      REQUIRE(vect_h == gold);
    }
  }

  SECTION("vector resize")
  {
    fk::vector<TestType, mem_type::owner, resource::device> test_reduced_d{
        1, 3, 5, 7, 9, 11, 13};
    fk::vector<TestType, mem_type::owner> const gold_enlarged{1, 3, 5, 0, 0};
    fk::vector<TestType, mem_type::owner, resource::device> test_enlarged_d{
        1, 3, 5};

    test_reduced_d.resize(gold.size());
    test_enlarged_d.resize(gold.size());

    fk::vector<TestType, mem_type::owner> const test_enlarged(
        test_enlarged_d.clone_onto_host());
    fk::vector<TestType, mem_type::owner> const test_reduced(
        test_reduced_d.clone_onto_host());

    REQUIRE(test_reduced == gold);
    REQUIRE(test_enlarged == gold_enlarged);
  }

  SECTION("views")
  {
    // view semantics on device
    {
      fk::vector<TestType, mem_type::owner, resource::device> vect(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::view, resource::device> vect_view(vect);
      fk::vector<TestType, mem_type::const_view, resource::device> const
          vect_cview(vect);
      REQUIRE(vect_view.data() == vect.data());
      REQUIRE(vect_cview.data() == vect.data());
      fk::vector<TestType, mem_type::view, resource::device> vect_view_2(
          vect_view);
      REQUIRE(vect_view_2.data() == vect.data());
      fk::vector<TestType, mem_type::const_view, resource::device> const
          vect_cview_2(vect_cview);
      REQUIRE(vect_cview_2.data() == vect.data());

      {
        fk::vector<TestType, mem_type::owner, resource::host> const copy(
            vect_view.clone_onto_host());
        fk::vector<TestType, mem_type::owner, resource::host> const ccopy(
            vect_cview.clone_onto_host());
        REQUIRE(copy == gold);
        REQUIRE(ccopy == gold);
        fk::vector<TestType, mem_type::owner, resource::host> const copy_2(
            vect_view_2.clone_onto_host());
        fk::vector<TestType, mem_type::owner, resource::host> const ccopy_2(
            vect_cview_2.clone_onto_host());
        REQUIRE(copy_2 == gold);
        REQUIRE(ccopy_2 == gold);
      }
      fk::vector<TestType, mem_type::owner, resource::host> const gold_2(
          {1, 2, 3, 4, 5});
      vect_view.transfer_from(gold_2);
      {
        fk::vector<TestType, mem_type::owner, resource::host> const copy(
            vect.clone_onto_host());

        fk::vector<TestType, mem_type::owner, resource::host> const copy_2(
            vect_view_2.clone_onto_host());

        fk::vector<TestType, mem_type::owner, resource::host> const copy_3(
            vect_cview.clone_onto_host());

        fk::vector<TestType, mem_type::owner, resource::host> const copy_4(
            vect_cview_2.clone_onto_host());

        REQUIRE(copy == gold_2);
        REQUIRE(copy_2 == gold_2);
        REQUIRE(copy_3 == gold_2);
        REQUIRE(copy_4 == gold_2);
      }

      fk::vector<TestType, mem_type::owner, resource::host> const gold_3(
          {3, 5, 7, 9, 11});
      vect_view_2.transfer_from(gold_3);

      {
        fk::vector<TestType, mem_type::owner, resource::host> const copy(
            vect.clone_onto_host());

        fk::vector<TestType, mem_type::owner, resource::host> const copy_2(
            vect_view.clone_onto_host());

        fk::vector<TestType, mem_type::owner, resource::host> const copy_3(
            vect_cview.clone_onto_host());

        fk::vector<TestType, mem_type::owner, resource::host> const copy_4(
            vect_cview_2.clone_onto_host());

        REQUIRE(copy == gold_3);
        REQUIRE(copy_2 == gold_3);
        REQUIRE(copy_3 == gold_3);
        REQUIRE(copy_4 == gold_3);
      }
    }
  }
#endif
}

TEMPLATE_TEST_CASE("fk::matrix interface: constructors, copy/move", "[tensors]",
                   test_precs, int)
{
  // set up the golden matrix
  // clang-format off
  fk::matrix<TestType> const gold{
    {12, 22, 32},
    {13, 23, 33},
    {14, 24, 34},
    {15, 25, 35},
    {16, 26, 36},
  };
  fk::matrix<int> const goldi{
    {12, 22, 32},
    {13, 23, 33},
    {14, 24, 34},
    {15, 25, 35},
    {16, 26, 36},
  };
  fk::matrix<float> const goldf{
    {12.0, 22.0, 32.0},
    {13.0, 23.0, 33.0},
    {14.0, 24.0, 34.0},
    {15.0, 25.0, 35.0},
    {16.0, 26.0, 36.0},
  };
  fk::matrix<double> const goldd{
    {12.0, 22.0, 32.0},
    {13.0, 23.0, 33.0},
    {14.0, 24.0, 34.0},
    {15.0, 25.0, 35.0},
    {16.0, 26.0, 36.0},
  }; // clang-format on

  fk::matrix<TestType, mem_type::const_view> const gold_cv(gold);
  fk::matrix<TestType> gold_own(gold);
  fk::matrix<TestType, mem_type::view> const gold_v(gold_own);

  SECTION("default constructor")
  {
    fk::matrix<TestType> test;
    // fk::matrix<TestType, mem_type::view> test_v; // disabled
    REQUIRE(test.empty());
  }
  SECTION("give me some size, initialized to zero")
  {
    fk::matrix<TestType> test(5, 3);
    // fk::matrix<TestType, mem_type::view> test_v (5, 3); // disabled
    // clang-format off
    fk::matrix<TestType> const zeros{
      {0, 0, 0},
      {0, 0, 0},
      {0, 0, 0},
      {0, 0, 0},
      {0, 0, 0},
    }; // clang-format on
    REQUIRE(test == zeros);
  }
  SECTION("constructor from list initialization")
  {
    // clang-format off
    fk::matrix<TestType> const test{
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {16, 26, 36},
    }; // clang-format on

    // clang-format off
    /* fk::matrix<TestType, mem_type::view> const test_v { // disabled
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {16, 26, 36},
    };*/ // clang-format on

    REQUIRE(test == gold);
  }
  SECTION("copy construction")
  {
    fk::matrix<TestType> const test(gold);
    fk::matrix<TestType> const test_v(gold_v);
    fk::matrix<TestType, mem_type::const_view> const test_cv(gold_cv);
    REQUIRE(test == gold);
    REQUIRE(test_v == gold);
    REQUIRE(test_cv == gold);
  }
  SECTION("copy assignment")
  {
    fk::matrix<TestType> test(5, 3);
    test = gold;
    REQUIRE(test == gold);
    fk::matrix<TestType> own(5, 3);
    fk::matrix<TestType> test_v(own);
    test_v = gold_v;
    REQUIRE(test_v == gold);
    // disabled for const views
  }
  SECTION("converting copy construction")
  {
    fk::matrix<int> const testi(gold);
    fk::matrix<int> const testi_v(gold_v);
    fk::matrix<int> const testi_cv(gold_cv);
    REQUIRE(testi == goldi);
    REQUIRE(testi_v == goldi);
    REQUIRE(testi_cv == goldi);

    fk::matrix<float> const testf(gold);
    fk::matrix<float> const testf_v(gold_v);
    fk::matrix<float> const testf_cv(gold_cv);
    REQUIRE(testf == goldf);
    REQUIRE(testf_v == goldf);
    REQUIRE(testf_cv == goldf);

    fk::matrix<double> const testd(gold);
    fk::matrix<double> const testd_v(gold_v);
    fk::matrix<double> const testd_cv(gold_cv);
    REQUIRE(testd == goldd);
    REQUIRE(testd_v == goldd);
    REQUIRE(testd_cv == goldd);
  }
  SECTION("converting copy assignment")
  {
    fk::matrix<int> testi(5, 3);
    fk::matrix<int> testi_own(5, 3);
    fk::matrix<int, mem_type::view> testi_v(testi_own);

    testi = gold;
    REQUIRE(testi == goldi);
    std::fill(testi.begin(), testi.end(), 0);
    testi = gold_v;
    REQUIRE(testi == goldi);
    std::fill(testi.begin(), testi.end(), 0);
    testi = gold_cv;
    REQUIRE(testi == goldi);

    testi_v = gold;
    REQUIRE(testi_v == goldi);
    std::fill(testi_own.begin(), testi_own.end(), 0);
    testi_v = gold_v;
    REQUIRE(testi_v == goldi);
    std::fill(testi_own.begin(), testi_own.end(), 0);
    testi_v = gold_cv;
    REQUIRE(testi_v == goldi);

    fk::matrix<float> testf(5, 3);
    fk::matrix<float> testf_own(5, 3);
    fk::matrix<float, mem_type::view> testf_v(testf_own);

    testf = gold;
    REQUIRE(testf == goldf);
    std::fill(testf.begin(), testf.end(), 0);
    testf = gold_v;
    REQUIRE(testf == goldf);
    std::fill(testf.begin(), testf.end(), 0);
    testf = gold_cv;
    REQUIRE(testf == goldf);

    testf_v = gold;
    REQUIRE(testf_v == goldf);
    std::fill(testf_own.begin(), testf_own.end(), 0);
    testf_v = gold_v;
    REQUIRE(testf_v == goldf);
    std::fill(testf_own.begin(), testf_own.end(), 0);
    testf_v = gold_cv;
    REQUIRE(testf_v == goldf);

    fk::matrix<double> testd(5, 3);
    fk::matrix<double> testd_own(5, 3);
    fk::matrix<double, mem_type::view> testd_v(testd_own);

    testd = gold;
    REQUIRE(testd == goldd);
    std::fill(testd.begin(), testd.end(), 0);
    testd = gold_v;
    REQUIRE(testd == goldd);
    std::fill(testd.begin(), testd.end(), 0);
    testd = gold_cv;
    REQUIRE(testd == goldd);

    testd_v = gold;
    REQUIRE(testd_v == goldd);
    std::fill(testd_own.begin(), testd_own.end(), 0);
    testd_v = gold_v;
    REQUIRE(testd_v == goldd);
    std::fill(testd_own.begin(), testd_own.end(), 0);
    testd_v = gold_cv;
    REQUIRE(testd_v == goldd);
  }
  SECTION("move construction")
  {
    // clang-format off
    fk::matrix<TestType> moved{
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {16, 26, 36},
    }; // clang-format on

    // clang-format off
    fk::matrix<TestType> moved_own{
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {16, 26, 36},
    }; // clang-format on

    fk::matrix<TestType, mem_type::view> moved_v(moved_own);
    fk::matrix<TestType, mem_type::const_view> moved_cv(moved_own);

    TestType *const test_data = moved.data();
    fk::matrix<TestType> const test(std::move(moved));
    REQUIRE(test == gold);
    REQUIRE(test.data() == test_data);
    REQUIRE(moved.data() == nullptr);

    fk::matrix<TestType, mem_type::view> const test_v(std::move(moved_v));
    REQUIRE(test_v == gold);
    REQUIRE(test_v.data() == moved_own.data());
    REQUIRE(moved_v.data() == nullptr);

    fk::matrix<TestType, mem_type::const_view> const test_cv(
        std::move(moved_cv));
    REQUIRE(test_cv == gold);
    REQUIRE(test_cv.data() == moved_own.data());
    REQUIRE(moved_cv.data() == nullptr);
  }
  SECTION("move assignment")
  {
    // clang-format off
    fk::matrix<TestType> moved{
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {16, 26, 36},
    }; // clang-format on

    // clang-format off
    fk::matrix<TestType> moved_own{
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {16, 26, 36},
    }; // clang-format on
    fk::matrix<TestType> moved_v(moved_own);

    fk::matrix<TestType> test(5, 3);
    fk::matrix<TestType> own(5, 3);
    fk::matrix<TestType> test_v(own);

    test   = std::move(moved);
    test_v = std::move(moved_v);
    REQUIRE(test == gold);
    REQUIRE(test_v == gold);
    // disabled for const views
  }
  SECTION("copy from fk::vector")
  {
    // clang-format off
    fk::vector<TestType> const vfk
      {12, 22, 32,
       13, 23, 33,
       14, 24, 34,
       15, 25, 35,
       16, 26, 36};
    // clang-format on

    fk::matrix<TestType> testfk(5, 3);
    fk::matrix<TestType> own(5, 3);
    fk::matrix<TestType, mem_type::view> test_v(own);

    testfk = vfk;
    test_v = vfk;
    REQUIRE(testfk == gold);
    REQUIRE(test_v == gold);
  }

  SECTION("views constructor")
  {
    // default one
    fk::matrix<TestType> base(gold);
    fk::matrix<TestType, mem_type::view> view(base);
    REQUIRE(base == view);

    fk::matrix<TestType, mem_type::view> view_2(view);
    REQUIRE(base == view_2);
    REQUIRE(view == view_2);

    // ranged
    fk::matrix<TestType, mem_type::view> view_3(base, 0, 2, 1, 2);
    fk::matrix<TestType> const gold_partial_3 =
        gold.extract_submatrix(0, 1, 3, 2);
    REQUIRE(view_3 == gold_partial_3);

    fk::matrix<TestType, mem_type::view> const view_4(view_3, 1, 1, 0, 1);

    fk::matrix<TestType> const gold_partial_4 =
        gold.extract_submatrix(1, 1, 1, 2);
    REQUIRE(view_4 == gold_partial_4);
  }

  SECTION("const views constructor")
  {
    // default one
    fk::matrix<TestType> base(gold);
    fk::matrix<TestType, mem_type::const_view> const view(base);
    REQUIRE(base == view);

    fk::matrix<TestType, mem_type::const_view> const view_v(view);
    REQUIRE(base == view_v);
    REQUIRE(view == view_v);

    fk::matrix<TestType, mem_type::view> const view_b(base);
    fk::matrix<TestType, mem_type::const_view> const view_view(view_b);

    REQUIRE(view_b == view_view);
    REQUIRE(base == view_view);

    // ranged
    fk::matrix<TestType, mem_type::const_view> const view_2(gold, 0, 2, 1, 2);
    fk::matrix<TestType> const gold_partial_2 =
        gold.extract_submatrix(0, 1, 3, 2);
    REQUIRE(view_2 == gold_partial_2);

    fk::matrix<TestType, mem_type::const_view> const view_3(view_2, 1, 1, 0, 1);
    fk::matrix<TestType> const gold_partial_3 =
        gold.extract_submatrix(1, 1, 1, 2);
    REQUIRE(view_3 == gold_partial_3);

    fk::matrix<TestType, mem_type::view> const view_4(base);
    fk::matrix<TestType, mem_type::const_view> const view_5(view_4, 0, 1, 0, 1);
    fk::matrix<TestType> const gold_partial_4 =
        gold.extract_submatrix(0, 0, 2, 2);
    REQUIRE(view_5 == gold_partial_4);
  }

  SECTION("views from vector constructor")
  {
    fk::vector<TestType> base{0, 1, 2, 3, 4, 5, 6, 7};

    fk::vector<TestType, mem_type::view> view(base, 1, 7);

    {
      // create 2x3 matrix from last six elems in base
      fk::matrix<TestType, mem_type::view> from_owner(base, 2, 3, 2);
      // create 2x2 matrix from middle of view
      fk::matrix<TestType, mem_type::view> from_view(view, 2, 2, 1);

      // clang-format off
      fk::matrix<TestType> const gold_initial   {{2, 4, 6},
					         {3, 5, 7}};
      fk::matrix<TestType> const gold_initial_v {{2, 4},
					         {3, 5}};
      // clang-format on

      REQUIRE(from_owner == gold_initial);
      REQUIRE(from_view == gold_initial_v);

      from_owner(1, 1) = 8;
      from_view(0, 1)  = 8;

      // clang-format off
      fk::vector<TestType> const after_mod = {0, 1, 2, 3, 8, 8, 6, 7};
      fk::vector<TestType> const after_mod_v = {1, 2, 3, 8, 8, 6, 7};
      fk::matrix<TestType> const gold_mod   {{2, 8, 6},
					     {3, 8, 7}};
      fk::matrix<TestType> const gold_mod_v {{2, 8},
					     {3, 8}};
      // clang-format on

      REQUIRE(from_owner == gold_mod);
      REQUIRE(from_view == gold_mod_v);
      REQUIRE(base == after_mod);
      REQUIRE(view == after_mod_v);
    }
    fk::matrix<TestType, mem_type::view> view_2(base, 1, 7);
    fk::matrix<TestType, mem_type::view> const view_m(std::move(view_2));
  }

  SECTION("const views from vector constructor")
  {
    fk::vector<TestType> const base{0, 1, 2, 3, 4, 5, 6, 7};

    fk::vector<TestType, mem_type::const_view> const view(base, 1, 7);

    {
      // create 2x3 matrix from last six elems in base
      fk::matrix<TestType, mem_type::const_view> const from_owner(base, 2, 3,
                                                                  2);

      // create 2x2 matrix from middle of view
      fk::matrix<TestType, mem_type::const_view> const from_view(view, 2, 2, 1);

      // clang-format off
      fk::matrix<TestType> const gold_initial   {{2, 4, 6},
					         {3, 5, 7}};
      fk::matrix<TestType> const gold_initial_v {{2, 4},
					         {3, 5}};
      // clang-format on

      REQUIRE(from_owner == gold_initial);
      REQUIRE(from_view == gold_initial_v);
    }
    fk::matrix<TestType, mem_type::const_view> view_2(base, 1, 7);
    fk::matrix<TestType, mem_type::const_view> const view_m(std::move(view_2));
  }

} // end fk::matrix constructors, copy/move

TEMPLATE_TEST_CASE("fk::matrix operators", "[tensors]", test_precs, int)
{
  // set up the golden matrix
  // clang-format off
  fk::matrix<TestType> const gold{
    {12, 22, 32},
    {13, 23, 33},
    {14, 24, 34},
    {15, 25, 35},
    {16, 26, 36},
  };

  fk::matrix<TestType> gold_own{
    {12, 22, 32},
    {13, 23, 33},
    {14, 24, 34},
    {15, 25, 35},
    {16, 26, 36},
  }; // clang-format on
  fk::matrix<TestType, mem_type::view> const gold_v(gold_own);
  fk::matrix<TestType, mem_type::const_view> const gold_cv(gold);

  SECTION("subscript operator (modifying)")
  {
    fk::matrix<TestType> test(5, 3);
    fk::matrix<TestType> own(5, 3);
    fk::matrix<TestType, mem_type::view> test_v(own);
    fk::matrix<TestType> own_p(5, 3);
    // extract subview of own_p
    fk::matrix<TestType, mem_type::view> test_v_p(own_p, 1, 3, 0, 2);
    // extract subview of gold for comparison
    fk::matrix<TestType> gold_copy(gold);
    fk::matrix<TestType, mem_type::view> gold_v_p(gold_copy, 1, 3, 0, 2);

    // clang-format off
    test(0,0) = 12;  test(0,1) = 22;  test(0,2) = 32;
    test(1,0) = 13;  test(1,1) = 23;  test(1,2) = 33;
    test(2,0) = 14;  test(2,1) = 24;  test(2,2) = 34;
    test(3,0) = 15;  test(3,1) = 25;  test(3,2) = 35;
    test(4,0) = 16;  test(4,1) = 26;  test(4,2) = 36;

    test_v(0,0) = 12;  test_v(0,1) = 22;  test_v(0,2) = 32;
    test_v(1,0) = 13;  test_v(1,1) = 23;  test_v(1,2) = 33;
    test_v(2,0) = 14;  test_v(2,1) = 24;  test_v(2,2) = 34;
    test_v(3,0) = 15;  test_v(3,1) = 25;  test_v(3,2) = 35;
    test_v(4,0) = 16;  test_v(4,1) = 26;  test_v(4,2) = 36;

    test_v_p(0,0) = 13;  test_v_p(0,1) = 23;  test_v_p(0,2) = 33;
    test_v_p(1,0) = 14;  test_v_p(1,1) = 24;  test_v_p(1,2) = 34;
    test_v_p(2,0) = 15;  test_v_p(2,1) = 25;  test_v_p(2,2) = 35;

    // clang-format on
    // clang-format on
    REQUIRE(test == gold);
    REQUIRE(test_v == gold);
    REQUIRE(test_v_p == gold_v_p);

    TestType val     = test(4, 2);
    TestType val_v   = test_v(4, 2);
    TestType val_v_p = test_v_p(2, 2);
    REQUIRE(val == 36);
    REQUIRE(val_v == 36);
    REQUIRE(val_v_p == 35);
  }
  SECTION("subscript operator (const)")
  {
    TestType const gold_value = 36;
    TestType const test       = gold(4, 2);
    TestType const test_v     = gold_v(4, 2);
    TestType const test_cv    = gold_cv(4, 2);
    fk::matrix<TestType, mem_type::const_view> const gold_v_p(gold, 4, 4, 1, 2);
    TestType const test_v_p = gold_v_p(0, 1);
    REQUIRE(test == gold_value);
    REQUIRE(test_v == gold_value);
    REQUIRE(test_cv == gold_value);
    REQUIRE(test_v_p == gold_value);
  }
  SECTION("comparison operator") // this gets used in every REQUIRE
  SECTION("comparison (negated) operator")
  {
    fk::matrix<TestType> test(gold);

    fk::matrix<TestType> own(gold);
    fk::matrix<TestType, mem_type::view> test_v(own);

    test(4, 2)   = 333;
    test_v(4, 2) = 333;

    REQUIRE(test != gold);
    REQUIRE(test_v != gold);
    REQUIRE(gold_cv != test);
    REQUIRE(gold_cv != test_v);
  }

  SECTION("less than operator")
  {
    fk::matrix<TestType> empty;

    fk::matrix<TestType, mem_type::view> const empty_v(empty);
    fk::matrix<TestType, mem_type::const_view> const empty_cv(empty);

    fk::matrix<TestType> gold_copy = gold;
    fk::matrix<TestType, mem_type::view> const gold_copy_v(gold_copy);
    fk::matrix<TestType, mem_type::const_view> const gold_copy_cv(gold_copy);

    fk::matrix<TestType> gold_prefix{{12, 13, 14}};
    fk::matrix<TestType, mem_type::view> const gold_prefix_v(gold_prefix);
    fk::matrix<TestType, mem_type::const_view> const gold_prefix_cv(
        gold_prefix);

    fk::matrix<TestType> mismatch{{12, 13, 15}};
    fk::matrix<TestType, mem_type::view> const mismatch_v(mismatch);
    fk::matrix<TestType, mem_type::const_view> const mismatch_cv(mismatch);

    // equal vectors return false
    REQUIRE(!(gold_copy < gold));
    REQUIRE(!(gold_copy < gold_v));
    REQUIRE(!(gold_copy < gold_cv));
    REQUIRE(!(gold_copy_v < gold));
    REQUIRE(!(gold_copy_v < gold_v));
    REQUIRE(!(gold_copy_v < gold_cv));
    REQUIRE(!(gold_copy_cv < gold));
    REQUIRE(!(gold_copy_cv < gold_v));
    REQUIRE(!(gold_copy_cv < gold_cv));

    // empty range less than non-empty range
    REQUIRE(empty < gold);
    REQUIRE(empty < gold_v);
    REQUIRE(empty < gold_cv);
    REQUIRE(empty_v < gold);
    REQUIRE(empty_v < gold_v);
    REQUIRE(empty_v < gold_cv);
    REQUIRE(empty_cv < gold);
    REQUIRE(empty_cv < gold_v);
    REQUIRE(empty_cv < gold_cv);

    // a prefix is less than the complete range
    REQUIRE(gold_prefix < gold);
    REQUIRE(gold_prefix < gold_v);
    REQUIRE(gold_prefix < gold_cv);
    REQUIRE(gold_prefix_v < gold);
    REQUIRE(gold_prefix_v < gold_v);
    REQUIRE(gold_prefix_v < gold_cv);
    REQUIRE(gold_prefix_cv < gold);
    REQUIRE(gold_prefix_cv < gold_v);
    REQUIRE(gold_prefix_cv < gold_cv);

    // otherwise compare on first mismatch
    REQUIRE(gold < mismatch);
    REQUIRE(gold < mismatch_v);
    REQUIRE(gold < mismatch_cv);
    REQUIRE(gold_v < mismatch);
    REQUIRE(gold_v < mismatch_v);
    REQUIRE(gold_v < mismatch_cv);
    REQUIRE(gold_cv < mismatch);
    REQUIRE(gold_cv < mismatch_v);
    REQUIRE(gold_cv < mismatch_cv);

    // also, empty ranges are equal
    REQUIRE(!(empty < empty));
    REQUIRE(!(empty < empty_v));
    REQUIRE(!(empty < empty_cv));
    REQUIRE(!(empty_v < empty));
    REQUIRE(!(empty_v < empty_v));
    REQUIRE(!(empty_v < empty_cv));
    REQUIRE(!(empty_cv < empty));
    REQUIRE(!(empty_cv < empty_v));
    REQUIRE(!(empty_cv < empty_cv));
  }

  SECTION("matrix+matrix addition")
  {
    // clang-format off
    fk::matrix<TestType> in1 {
      {11, 20, 0},
      {12, 21, 0},
      {13, 22, 0},
      {14, 23, 0},
      {15, 24, 0},
    };
    fk::matrix<TestType> in2 {
      {1, 2, 32},
      {1, 2, 33},
      {1, 2, 34},
      {1, 2, 35},
      {1, 2, 36},
    }; // clang-format on

    fk::matrix<TestType, mem_type::view> const in1_v(in1);
    fk::matrix<TestType, mem_type::view> const in2_v(in2);
    fk::matrix<TestType, mem_type::const_view> const in1_cv(in1);
    fk::matrix<TestType, mem_type::const_view> const in2_cv(in2);

    fk::matrix<TestType, mem_type::const_view> const in1_v_p(in1, 1, 2, 0, 2);
    fk::matrix<TestType, mem_type::const_view> const in2_v_p(in2, 1, 2, 0, 2);
    fk::matrix<TestType, mem_type::const_view> const gold_v_p(gold, 1, 2, 0, 2);

    REQUIRE((in1 + in2) == gold);
    REQUIRE((in1 + in2_v) == gold);
    REQUIRE((in1 + in2_cv) == gold);

    REQUIRE((in1_v + in2) == gold);
    REQUIRE((in1_v + in2_v) == gold);
    REQUIRE((in1_v + in2_cv) == gold);

    REQUIRE((in1_cv + in2) == gold);
    REQUIRE((in1_cv + in2_v) == gold);
    REQUIRE((in1_cv + in2_cv) == gold);

    REQUIRE((in1_v_p + in2_v_p) == gold_v_p);
  }
  SECTION("matrix-matrix subtraction")
  {
    // clang-format off
    fk::matrix<TestType> in1 {
      {13, 22, 34},
      {14, 23, 35},
      {15, 24, 36},
      {16, 25, 37},
      {17, 26, 38},
    };
    fk::matrix<TestType> in2 {
      {1, 0, 2},
      {1, 0, 2},
      {1, 0, 2},
      {1, 0, 2},
      {1, 0, 2}
    }; // clang-format on

    fk::matrix<TestType, mem_type::view> const in1_v(in1);
    fk::matrix<TestType, mem_type::view> const in2_v(in2);
    fk::matrix<TestType, mem_type::const_view> const in1_cv(in1);
    fk::matrix<TestType, mem_type::const_view> const in2_cv(in2);

    fk::matrix<TestType, mem_type::const_view> const in1_v_p(in1, 0, 3, 1, 2);
    fk::matrix<TestType, mem_type::const_view> const in2_v_p(in2, 0, 3, 1, 2);
    fk::matrix<TestType, mem_type::const_view> const gold_v_p(gold, 0, 3, 1, 2);

    REQUIRE((in1 - in2) == gold);
    REQUIRE((in1 - in2_v) == gold);
    REQUIRE((in1 - in2_cv) == gold);

    REQUIRE((in1_v - in2) == gold);
    REQUIRE((in1_v - in2_v) == gold);
    REQUIRE((in1_v - in2_cv) == gold);

    REQUIRE((in1_cv - in2) == gold);
    REQUIRE((in1_cv - in2_v) == gold);
    REQUIRE((in1_cv - in2_cv) == gold);

    REQUIRE((in1_v_p - in2_v_p) == gold_v_p);
  }

  SECTION("matrix*scalar multiplication")
  {
    // clang-format off
    fk::matrix<TestType> in {
      {3, 4, 5},
      {6, 7, 8},
      {9, 10, 11},
      {12, 13, 14},
      {15, 16, 17},
    };
    fk::matrix<TestType> const in_scaled {
      {12, 16, 20},
      {24, 28, 32},
      {36, 40, 44},
      {48, 52, 56},
      {60, 64, 68},
    }; // clang-format on

    fk::matrix<TestType, mem_type::view> const in_v(in);
    fk::matrix<TestType, mem_type::const_view> const in_cv(in);
    fk::matrix<TestType, mem_type::view> const in_v_p(in, 4, 4, 0, 2);
    fk::matrix<TestType, mem_type::const_view> const in_scaled_v_p(in_scaled, 4,
                                                                   4, 0, 2);
    REQUIRE(in * 4 == in_scaled);
    REQUIRE(in_v * 4 == in_scaled);
    REQUIRE(in_cv * 4 == in_scaled);
    REQUIRE(in_v_p * 4 == in_scaled_v_p);
  }
  SECTION("matrix*vector multiplication")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      // clang-format off
    fk::matrix<TestType> testm{
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {16, 26, 36},
    }; // clang-format on
      fk::matrix<TestType, mem_type::view> const testm_v(testm);
      fk::matrix<TestType, mem_type::const_view> const testm_cv(testm);
      fk::matrix<TestType, mem_type::const_view> const testm_v_p(testm, 1, 2, 0,
                                                                 2);

      fk::vector<TestType> testv{2, 3, 4};
      fk::vector<TestType, mem_type::view> const testv_v(testv);
      fk::vector<TestType, mem_type::const_view> const testv_cv(testv);

      fk::vector<TestType> const gold_data{218, 227, 236, 245, 254};
      REQUIRE((testm * testv) == gold_data);
      REQUIRE((testm * testv_v) == gold_data);
      REQUIRE((testm * testv_cv) == gold_data);

      REQUIRE((testm_v * testv) == gold_data);
      REQUIRE((testm_v * testv_v) == gold_data);
      REQUIRE((testm_v * testv_cv) == gold_data);

      REQUIRE((testm_cv * testv) == gold_data);
      REQUIRE((testm_cv * testv_v) == gold_data);
      REQUIRE((testm_cv * testv_cv) == gold_data);

      fk::vector<TestType> const gold_p = gold_data.extract(1, 2);
      REQUIRE((testm_v_p * testv_v) == gold_p);
      REQUIRE((testm_v_p * testv) == gold_p);
    }
  }
  SECTION("matrix*matrix multiplication")
  {
    // I'm not factoring the golden matrix, so here's a new answer (calculated
    // from octave)
    // clang-format off
    fk::matrix<TestType> const ans {
      {360, 610, 860},
      {710, 1210, 1710},
    };
    fk::matrix<TestType> const ans_p {
      {347, 497},
      {692, 992}
    };
    fk::matrix<TestType> in1 {
      {3, 4, 5, 6, 7},
      {8, 9, 10, 11, 12},
    };

    fk::matrix<TestType> in2 {
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {16, 26, 36},
    };
    // clang-format on

    fk::matrix<TestType, mem_type::view> const in1_v(in1);
    fk::matrix<TestType, mem_type::view> const in2_v(in2);

    fk::matrix<TestType, mem_type::const_view> const in1_cv(in1);
    fk::matrix<TestType, mem_type::const_view> const in2_cv(in2);

    fk::matrix<TestType, mem_type::const_view> const in1_v_p(in1, 0, 1, 1, 3);
    fk::matrix<TestType, mem_type::const_view> const in2_v_p(in2, 0, 2, 1, 2);

    REQUIRE((in1 * in2) == ans);
    REQUIRE((in1 * in2_v) == ans);
    REQUIRE((in1 * in2_cv) == ans);

    REQUIRE((in1_v * in2) == ans);
    REQUIRE((in1_v * in2_v) == ans);
    REQUIRE((in1_v * in2_cv) == ans);

    REQUIRE((in1_cv * in2) == ans);
    REQUIRE((in1_cv * in2_v) == ans);
    REQUIRE((in1_cv * in2_cv) == ans);

    REQUIRE((in1_v_p * in2_v_p) == ans_p);
  }
  SECTION("matrix kron product")
  {
    // clang-format off

    fk::matrix<TestType> A {{1,2,3}};

    fk::matrix<TestType> B {{2,3},
	                    {4,5},
	                    {6,7},
	                    {8,9}};

    fk::matrix<TestType> const ans {{2,3,4,6,6,9},
	                            {4,5,8,10,12,15},
				    {6,7,12,14,18,21},
				    {8,9,16,18,24,27}};
    // clang-format on

    fk::matrix<TestType, mem_type::view> const A_v(A);
    fk::matrix<TestType, mem_type::view> const B_v(B);

    fk::matrix<TestType, mem_type::const_view> const A_cv(A);
    fk::matrix<TestType, mem_type::const_view> const B_cv(B);

    REQUIRE(A.kron(B) == ans);
    REQUIRE(A.kron(B_v) == ans);
    REQUIRE(A.kron(B_cv) == ans);

    REQUIRE(A_v.kron(B) == ans);
    REQUIRE(A_v.kron(B_v) == ans);
    REQUIRE(A_v.kron(B_cv) == ans);

    REQUIRE(A_cv.kron(B) == ans);
    REQUIRE(A_cv.kron(B_v) == ans);
    REQUIRE(A_cv.kron(B_cv) == ans);

    // add some larger matrices to test partial views...

    // clang-format off
    fk::matrix<TestType> const A_own{{0, 1, 2, 3, 4},
				     {5, 6, 7, 8, 9},
				     {10, 11, 12, 13}};
    fk::matrix<TestType, mem_type::const_view> const A_v_p(A_own, 0, 1, 1, 3);
    fk::matrix<TestType> const B_own{{14, 15, 16, 17, 18},
	    			     {19, 20, 21, 22, 23},
				     {24, 25, 26, 27, 28}};
    // clang-format on
    fk::matrix<TestType, mem_type::const_view> const B_v_p(B_own, 1, 2, 2, 4);
    fk::matrix<TestType> const ans_p = {
        {21, 22, 23, 42, 44, 46, 63, 66, 69},
        {26, 27, 28, 52, 54, 56, 78, 81, 84},
        {126, 132, 138, 147, 154, 161, 168, 176, 184},
        {156, 162, 168, 182, 189, 196, 208, 216, 224}};
    REQUIRE(A_v_p.kron(B_v_p) == ans_p);
    fk::matrix<TestType, mem_type::const_view> const ans_a_p(ans_p, 0, 1, 0, 8);
    REQUIRE(A.kron(B_v_p) == ans_a_p);

    // clang-format off
    fk::matrix<TestType> const ans_b_p = {
	    {2,4,6,3,6,9},
	    {12,14,16,18,21,24},
            {4,8,12,5,10,15},
	    {24,28,32,30,35,40},
	    {6,12,18,7,14,21},
	    {36,42,48,42,49,56},
	    {8,16,24,9,18,27},
	    {48,56,64,54,63,72}};
    // clang-format on

    REQUIRE(B.kron(A_v_p) == ans_b_p);
  }
  SECTION("matrix inverse")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      // (square slices of) our golden matrix is singular, so here's a
      // well conditioned one
      fk::matrix<TestType> const test{{0.767135868133925, -0.641484652834663},
                                      {0.641484652834663, 0.767135868133926}};

      fk::matrix<TestType> test_own{
          {1.0, 0.767135868133925, -0.641484652834663, 1.0},
          {1.0, 0.641484652834663, 0.767135868133926, 1.0},
          {1.0, 1.0, 1.0}};
      fk::matrix<TestType, mem_type::view> test_v_p(test_own, 0, 1, 1, 2);

      fk::matrix<TestType> test_copy(test);

      fk::matrix<TestType> own(test);
      fk::matrix<TestType, mem_type::view> test_v(own);

      test_copy.invert();
      test_v.invert();
      test_v_p.invert();

      // A * inv(A) == I
      REQUIRE((test * test_copy) == eye<TestType>(2));
      REQUIRE((test * test_v) == eye<TestType>(2));
      REQUIRE((test * test_v_p) == eye<TestType>(2));

      // we haven't implemented a matrix inversion routine for integral types;
      // that function is disabled for now in the class if instantiated for
      // non-floating point type; code won't compile if this routine is called
      // on integer matrix
    }
    else
    {
      REQUIRE(true);
    }
  }
  SECTION("matrix determinant")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      // clang-format off
    fk::matrix<TestType> in {
      {12.130, 14.150, 1.00},
      {13.140, 13.150, 1.00},
      {14.150, 12.130, 1.00},
    };
    fk::matrix<TestType> const in_own {
	{1.0, 1.0000, 1.0000, 1.00},
	{1.0, 12.130, 14.150, 1.00},
        {1.0, 13.140, 13.150, 1.00},
        {1.0, 14.150, 12.130, 1.00},
      }; // clang-format on
      fk::matrix<TestType, mem_type::const_view> const in_v_p(in_own, 1, 3, 1,
                                                              3);
      fk::matrix<TestType, mem_type::view> const in_v(in);
      fk::matrix<TestType, mem_type::const_view> const in_cv(in);

      TestType const gold_value = -0.020200;
      TestType const tol_factor = 1e2;
      relaxed_fp_comparison(in.determinant(), gold_value, tol_factor);
      relaxed_fp_comparison(in_v.determinant(), gold_value, tol_factor);
      relaxed_fp_comparison(in_cv.determinant(), gold_value, tol_factor);
      relaxed_fp_comparison(in_v_p.determinant(), gold_value, tol_factor);

      // we haven't implemented a determinant routine for integral types; as
      // with inversion, code won't compile if this routine is invoked on a
      // matrix of integers
    }
    else
    {
      REQUIRE(true);
    }
  }
  SECTION("nrows(): the number of rows")
  {
    fk::matrix<TestType, mem_type::const_view> const gold_v_p(gold, 0, 3, 0, 2);
    REQUIRE(gold.nrows() == 5);
    REQUIRE(gold_v.nrows() == 5);
    REQUIRE(gold_cv.nrows() == 5);
    REQUIRE(gold_v_p.nrows() == 4);
  }
  SECTION("ncols(): the number of columns")
  {
    fk::matrix<TestType, mem_type::const_view> const gold_v_p(gold, 0, 4, 0, 1);
    REQUIRE(gold.ncols() == 3);
    REQUIRE(gold_v.ncols() == 3);
    REQUIRE(gold_cv.ncols() == 3);
    REQUIRE(gold_v_p.ncols() == 2);
  }
  SECTION("size(): the number of elements")
  {
    fk::matrix<TestType, mem_type::const_view> const gold_v_p(gold, 1, 3, 1, 2);
    REQUIRE(gold.size() == 15);
    REQUIRE(gold_v.size() == 15);
    REQUIRE(gold_cv.size() == 15);
    REQUIRE(gold_v_p.size() == 6);
  }
  SECTION("empty(): the number of elements")
  {
    fk::matrix<TestType> const gold_empty;
    REQUIRE(gold_empty.empty());

    REQUIRE(!gold.empty());
    REQUIRE(!gold_v.empty());
    REQUIRE(!gold_cv.empty());
    fk::matrix<TestType, mem_type::const_view> const gold_v_p(gold, 1, 3, 1, 2);
    REQUIRE(!gold_v_p.empty());
  }
  SECTION("data(): const get address to an element")
  {
    fk::matrix<TestType, mem_type::const_view> const gold_v_p(gold, 3, 4, 1, 2);
    TestType const gold_value = 36;
    REQUIRE(*gold.data(4, 2) == gold_value);
    REQUIRE(*gold_v.data(4, 2) == gold_value);
    REQUIRE(*gold_cv.data(4, 2) == gold_value);
    REQUIRE(*gold_v_p.data(1, 1) == gold_value);
  }
} // end fk::matrix operators

TEMPLATE_TEST_CASE("fk::matrix utilities", "[tensors]", test_precs, int)
{
  // set up the golden matrix
  // clang-format off
  fk::matrix<TestType> const gold {
    {12, 22, 32},
    {13, 23, 33},
    {14, 24, 34},
    {15, 25, 35},
    {16, 26, 36},
  }; // clang-format on

  fk::matrix<TestType> gold_mutable(gold);
  fk::matrix<TestType, mem_type::view> const gold_v(gold_mutable);
  fk::matrix<TestType, mem_type::const_view> const gold_cv(gold);

  SECTION("matrix update_col(fk::vector)")
  {
    // clang-format off
    fk::matrix<TestType> test {
      {12, 22, 0},
      {13, 23, 0},
      {14, 24, 0},
      {15, 25, 0},
      {16, 26, 52},
    }; // clang-format on
    fk::matrix<TestType> const orig(test);

    // clang-format off
    fk::matrix<TestType> test_p {
      {13, 23, 0},
      {14, 24, 0},
      {15, 25, 0}
    }; // clang-format on
    fk::matrix<TestType> const orig_p(test_p);

    fk::matrix<TestType, mem_type::const_view> const gold_v_p(gold, 1, 3, 0, 2);

    fk::matrix<TestType> own(test);
    fk::matrix<TestType, mem_type::view> test_v(own);

    fk::matrix<TestType> own_p(test);
    fk::matrix<TestType, mem_type::view> test_v_p(own_p, 1, 3, 0, 2);

    fk::vector<TestType> testv{32, 33, 34, 35, 36};
    fk::vector<TestType, mem_type::view> const testv_v(testv);
    fk::vector<TestType, mem_type::const_view> const testv_cv(testv);

    fk::vector<TestType> const testv_p = testv.extract(1, 3);
    fk::vector<TestType, mem_type::const_view> const testv_v_p(testv_p);

    REQUIRE(test.update_col(2, testv) == gold);
    test = orig;
    REQUIRE(test.update_col(2, testv_v) == gold);
    test = orig;
    REQUIRE(test.update_col(2, testv_cv) == gold);

    REQUIRE(test_v.update_col(2, testv) == gold);
    test_v = orig;
    REQUIRE(test_v.update_col(2, testv_v) == gold);
    test_v = orig;
    REQUIRE(test_v.update_col(2, testv_cv) == gold);

    REQUIRE(test_p.update_col(2, testv_p) == gold_v_p);
    test_p = orig_p;
    REQUIRE(test_p.update_col(2, testv_v_p) == gold_v_p);
  }
#ifdef ASGARD_USE_CUDA
  SECTION("matrix update_col(fk::vector)")
  {
    // clang-format off
    fk::matrix<TestType,mem_type::owner,resource::device> test {
      {12, 22, 0},
      {13, 23, 0},
      {14, 24, 0},
      {15, 25, 0},
      {16, 26, 52},
    }; // clang-format on
    fk::matrix<TestType, mem_type::owner, resource::device> const orig(test);

    // clang-format off
    fk::matrix<TestType, mem_type::owner,resource::device> test_p {
      {13, 23, 0},
      {14, 24, 0},
      {15, 25, 0}
    }; // clang-format on
    fk::matrix<TestType, mem_type::owner, resource::device> const orig_p(
        test_p);

    fk::matrix<TestType, mem_type::const_view> const gold_v_p(gold, 1, 3, 0, 2);

    fk::matrix<TestType, mem_type::owner, resource::device> own(test);
    fk::matrix<TestType, mem_type::view, resource::device> test_v(own);

    fk::matrix<TestType, mem_type::owner, resource::device> own_p(test);
    fk::matrix<TestType, mem_type::view, resource::device> test_v_p(own_p, 1, 3,
                                                                    0, 2);

    fk::vector<TestType, mem_type::owner, resource::device> testv{32, 33, 34,
                                                                  35, 36};
    fk::vector<TestType, mem_type::view, resource::device> const testv_v(testv);
    fk::vector<TestType, mem_type::const_view, resource::device> const testv_cv(
        testv);

    fk::vector<TestType, mem_type::owner, resource::device> const testv_p =
        testv.extract(1, 3);
    fk::vector<TestType, mem_type::const_view, resource::device> const
        testv_v_p(testv_p);

    REQUIRE(test.update_col(2, testv).clone_onto_host() == gold);
    test = orig;
    REQUIRE(test.update_col(2, testv_v).clone_onto_host() == gold);
    test = orig;
    REQUIRE(test.update_col(2, testv_cv).clone_onto_host() == gold);

    REQUIRE(test_v.update_col(2, testv).clone_onto_host() == gold);
    test_v = orig;
    REQUIRE(test_v.update_col(2, testv_v).clone_onto_host() == gold);
    test_v = orig;
    REQUIRE(test_v.update_col(2, testv_cv).clone_onto_host() == gold);

    REQUIRE(test_p.update_col(2, testv_p).clone_onto_host() == gold_v_p);
    test_p = orig_p;
    REQUIRE(test_p.update_col(2, testv_v_p).clone_onto_host() == gold_v_p);
  }
#endif
  SECTION("matrix update_col(std::vector)")
  {
    // clang-format off
    fk::matrix<TestType> test {
      {12, 22, 0},
      {13, 23, 0},
      {14, 24, 0},
      {15, 25, 0},
      {16, 26, 52},
    }; // clang-format on
    fk::matrix<TestType> own(test);
    fk::matrix<TestType, mem_type::view> test_v(own);

    fk::matrix<TestType> own_p(test);
    fk::matrix<TestType, mem_type::view> test_v_p(own_p, 2, 4, 1, 2);
    fk::matrix<TestType, mem_type::const_view> const gold_v_p(gold, 2, 4, 1, 2);

    std::vector<TestType> const testv{32, 33, 34, 35, 36};
    std::vector<TestType> const testv_p{34, 35, 36};

    REQUIRE(test.update_col(2, testv) == gold);
    REQUIRE(test_v.update_col(2, testv) == gold);
    REQUIRE(test_v_p.update_col(1, testv_p) == gold_v_p);
  }

  SECTION("matrix update_row(fk::vector)")
  {
    // clang-format off
    fk::matrix<TestType> test {
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {0, 0, 35},
    }; // clang-format on
    fk::matrix<TestType> const orig(test);

    fk::matrix<TestType> own_p(test);
    fk::matrix<TestType, mem_type::view> test_v_p(own_p, 3, 4, 0, 2);
    fk::matrix<TestType> const orig_p(test_v_p);
    fk::matrix<TestType, mem_type::const_view> const gold_v_p(gold, 3, 4, 0, 2);

    fk::matrix<TestType> own(test);
    fk::matrix<TestType, mem_type::view> test_v(own);

    fk::vector<TestType> testv{16, 26, 36};
    fk::vector<TestType, mem_type::view> const testv_v(testv);
    fk::vector<TestType, mem_type::const_view> const testv_cv(testv);

    REQUIRE(test.update_row(4, testv) == gold);
    test = orig;
    REQUIRE(test.update_row(4, testv_v) == gold);
    test = orig;
    REQUIRE(test.update_row(4, testv_cv) == gold);

    REQUIRE(test_v.update_row(4, testv) == gold);
    test_v = orig;
    REQUIRE(test_v.update_row(4, testv_v) == gold);
    test_v = orig;
    REQUIRE(test_v.update_row(4, testv_cv) == gold);

    REQUIRE(test_v_p.update_row(1, testv) == gold_v_p);
    test_v_p = orig_p;
    REQUIRE(test_v_p.update_row(1, testv_v) == gold_v_p);
  }
  SECTION("matrix update_row(std::vector)")
  {
    // clang-format off
    fk::matrix<TestType> test {
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {0, 0, 35},
    }; // clang-format on

    fk::matrix<TestType> own_p(test);
    fk::matrix<TestType, mem_type::view> test_v_p(own_p, 3, 4, 0, 2);
    fk::matrix<TestType, mem_type::const_view> const gold_v_p(gold, 3, 4, 0, 2);

    fk::matrix<TestType> own(test);
    fk::matrix<TestType, mem_type::view> test_v(own);

    std::vector<TestType> const testv{16, 26, 36};

    REQUIRE(test.update_row(4, testv) == gold);
    REQUIRE(test_v.update_row(4, testv) == gold);
    REQUIRE(test_v_p.update_row(1, testv) == gold_v_p);
  }

  SECTION("matrix clear and resize")
  {
    fk::matrix<TestType> gold_copy = gold;
    gold_copy.clear_and_resize(2, 1);
    // clang-format off
    fk::matrix<TestType> const test {
      {0},
      {0},
      };
    // clang-format on
    REQUIRE(gold_copy == test);
  }

  SECTION("matrix set submatrix(row, col, submatrix)")
  {
    // clang-format off
    fk::matrix<TestType> test {
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      { 0,  0, 35},
    };

    fk::matrix<TestType> const orig(test);
    fk::matrix<TestType> own(test);
    fk::matrix<TestType, mem_type::view> test_v(own);

    fk::matrix<TestType> own_p(test);
    fk::matrix<TestType, mem_type::view> test_v_p(own_p, 0, 3, 1, 2);
    fk::matrix<TestType> const orig_p(test_v_p);

    fk::matrix<TestType> sub {
      {-13, -23},
      {-14, -24},
      {-15, -25},
    };
    fk::matrix<TestType, mem_type::view> const sub_v(sub);
    fk::matrix<TestType, mem_type::const_view> const sub_cv(sub);

    fk::matrix<TestType> const after_set {
      {12,  22,  32},
      {13, -13, -23},
      {14, -14, -24},
      {15, -15, -25},
      {0,    0,  35},
    }; // clang-format on
    fk::matrix<TestType, mem_type::const_view> const after_set_v_p(after_set, 0,
                                                                   3, 1, 2);

    REQUIRE(test.set_submatrix(1, 1, sub) == after_set);
    test = orig;
    REQUIRE(test.set_submatrix(1, 1, sub_v) == after_set);
    test = orig;
    REQUIRE(test.set_submatrix(1, 1, sub_cv) == after_set);
    test = orig;

    REQUIRE(test_v.set_submatrix(1, 1, sub) == after_set);
    test_v = orig;
    REQUIRE(test_v.set_submatrix(1, 1, sub_v) == after_set);
    test_v = orig;
    REQUIRE(test_v.set_submatrix(1, 1, sub_cv) == after_set);
    test_v = orig;

    REQUIRE(test_v_p.set_submatrix(1, 0, sub) == after_set_v_p);
    test_v_p = orig_p;
    REQUIRE(test_v_p.set_submatrix(1, 0, sub_v) == after_set_v_p);
    test_v_p = orig_p;
    REQUIRE(test_v_p.set_submatrix(1, 0, sub_cv) == after_set_v_p);
    test_v_p = orig_p;

    // now, test setting a partial view
    fk::matrix<TestType, mem_type::const_view> const sub_v_p(sub, 0, 1, 0, 1);

    // clang-format off
    fk::matrix<TestType> const after_set_p {
      {12,  22,  32},
      {13, -13, -23},
      {14, -14, -24},
      {15,  25,  35},
      {0,    0,  35},
    }; // clang-format on
    fk::matrix<TestType, mem_type::const_view> const after_set_p_v(after_set_p,
                                                                   0, 3, 1, 2);

    REQUIRE(test.set_submatrix(1, 1, sub_v_p) == after_set_p);
    REQUIRE(test_v.set_submatrix(1, 1, sub_v_p) == after_set_p);
    REQUIRE(test_v_p.set_submatrix(1, 0, sub_v_p) == after_set_p_v);
  }

  SECTION("matrix extract submatrix(row, col, nrows, ncols")
  {
    // clang-format off
    fk::matrix<TestType> const test {
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {0, 0, 35},
    };

    fk::matrix<TestType> own(test);
    fk::matrix<TestType, mem_type::view> const test_v(own);
    fk::matrix<TestType, mem_type::const_view> const test_cv(own);

    fk::matrix<TestType> const own_p(test);
    fk::matrix<TestType, mem_type::const_view> const test_v_p(own_p, 1, 4, 0, 1);

    fk::matrix<TestType> const sub {
      {13, 23},
      {14, 24},
      {15, 25},
    }; // clang-format on

    REQUIRE(test.extract_submatrix(1, 0, 3, 2) == sub);
    REQUIRE(test_v.extract_submatrix(1, 0, 3, 2) == sub);
    REQUIRE(test_cv.extract_submatrix(1, 0, 3, 2) == sub);
    REQUIRE(test_v_p.extract_submatrix(0, 0, 3, 2) == sub);
  }

  SECTION("print out the values")
  {
    // (effectively) redirect cout
    std::streambuf *old_cout_stream_buf = std::cout.rdbuf();
    std::ostringstream test_str, test_str_v, test_str_cv, test_str_vp;

    fk::matrix<TestType, mem_type::const_view> const gold_v_p(gold, 1, 3, 0, 2);

    // generate the output (into test_strs)
    std::cout.rdbuf(test_str.rdbuf());
    gold.print("golden matrix");
    std::cout.rdbuf(test_str_v.rdbuf());
    gold_v.print("golden matrix");
    std::cout.rdbuf(test_str_cv.rdbuf());
    gold_cv.print("golden matrix");
    std::cout.rdbuf(test_str_vp.rdbuf());
    gold_v_p.print("golden matrix");

    // restore cout destination
    std::cout.rdbuf(old_cout_stream_buf);

    std::string golden_string, golden_string_v, golden_string_cv,
        golden_string_vp;
    if constexpr (std::is_floating_point_v<TestType>)
    {
      golden_string =
          "golden matrix(owner)\n  1.2000e+01  "
          "2.2000e+01  3.2000e+01\n  "
          "1.3000e+01  "
          "2.3000e+01  3.3000e+01\n  1.4000e+01  2.4000e+01  3.4000e+01\n  "
          "1.5000e+01  2.5000e+01  3.5000e+01\n  1.6000e+01  2.6000e+01  "
          "3.6000e+01\n";

      golden_string_v =
          "golden matrix(view, stride == 5)\n  1.2000e+01  2.2000e+01  "
          "3.2000e+01\n  "
          "1.3000e+01  "
          "2.3000e+01  3.3000e+01\n  1.4000e+01  2.4000e+01  3.4000e+01\n  "
          "1.5000e+01  2.5000e+01  3.5000e+01\n  1.6000e+01  2.6000e+01  "
          "3.6000e+01\n";

      golden_string_cv =
          "golden matrix(const view, stride == 5)\n  1.2000e+01  2.2000e+01  "
          "3.2000e+01\n  "
          "1.3000e+01  "
          "2.3000e+01  3.3000e+01\n  1.4000e+01  2.4000e+01  3.4000e+01\n  "
          "1.5000e+01  2.5000e+01  3.5000e+01\n  1.6000e+01  2.6000e+01  "
          "3.6000e+01\n";

      golden_string_vp =
          "golden matrix(const view, stride == 5)\n  1.3000e+01  "
          "2.3000e+01  3.3000e+01\n  "
          "1.4000e+01  2.4000e+01  3.4000e+01\n  "
          "1.5000e+01  2.5000e+01  3.5000e+01\n";
    }
    else
    {
      golden_string = "golden matrix(owner)\n12 22 32 "
                      "\n13 23 33 \n14 24 34 \n15 25 "
                      "35 \n16 26 36 \n";

      golden_string_v = "golden matrix(view, stride == 5)\n12 22 32 \n13 23 33 "
                        "\n14 24 34 \n15 25 "
                        "35 \n16 26 36 \n";
      golden_string_cv =
          "golden matrix(const view, stride == 5)\n12 22 32 \n13 23 33 "
          "\n14 24 34 \n15 25 "
          "35 \n16 26 36 \n";
      golden_string_vp =
          "golden matrix(const view, stride == 5)\n13 23 33 \n14 24 34 \n15 25 "
          "35 \n";
    }

    REQUIRE(test_str.str() == golden_string);
    REQUIRE(test_str_v.str() == golden_string_v);
    REQUIRE(test_str_cv.str() == golden_string_cv);
    REQUIRE(test_str_vp.str() == golden_string_vp);
  }

  SECTION("dump to octave")
  {
    gold.dump_to_octave("test_out.dat");
    gold_v.dump_to_octave("test_out_v.dat");
    gold_cv.dump_to_octave("test_out_cv.dat");

    fk::matrix<TestType, mem_type::const_view> const gold_v_p(gold, 1, 3, 0, 1);
    gold_v_p.dump_to_octave("test_out_v_p.dat");

    std::ifstream data_stream("test_out.dat");
    std::ifstream data_stream_v("test_out_v.dat");
    std::ifstream data_stream_cv("test_out_cv.dat");
    std::ifstream data_stream_v_p("test_out_v_p.dat");

    std::string test_string((std::istreambuf_iterator<char>(data_stream)),
                            std::istreambuf_iterator<char>());
    std::string test_string_v((std::istreambuf_iterator<char>(data_stream_v)),
                              std::istreambuf_iterator<char>());

    std::string test_string_cv((std::istreambuf_iterator<char>(data_stream_cv)),
                               std::istreambuf_iterator<char>());
    std::string test_string_v_p(
        (std::istreambuf_iterator<char>(data_stream_v_p)),
        std::istreambuf_iterator<char>());

    REQUIRE(std::filesystem::remove("test_out.dat"));
    REQUIRE(std::filesystem::remove("test_out_v.dat"));
    REQUIRE(std::filesystem::remove("test_out_cv.dat"));
    REQUIRE(std::filesystem::remove("test_out_v_p.dat"));

    std::string golden_string, golden_string_p;

    if constexpr (std::is_floating_point_v<TestType>)
    {
      golden_string =
          "1.200000000000e+01 2.200000000000e+01 3.200000000000e+01 \n"
          "1.300000000000e+01 2.300000000000e+01 3.300000000000e+01 \n"
          "1.400000000000e+01 2.400000000000e+01 3.400000000000e+01 \n"
          "1.500000000000e+01 2.500000000000e+01 3.500000000000e+01 \n"
          "1.600000000000e+01 2.600000000000e+01 3.600000000000e+01 \n";

      golden_string_p = "1.300000000000e+01 2.300000000000e+01 \n"
                        "1.400000000000e+01 2.400000000000e+01 \n"
                        "1.500000000000e+01 2.500000000000e+01 \n";
    }
    else
    {
      golden_string = "12 22 32 \n"
                      "13 23 33 \n"
                      "14 24 34 \n"
                      "15 25 35 \n"
                      "16 26 36 \n";

      golden_string_p = "13 23 \n"
                        "14 24 \n"
                        "15 25 \n";
    }

    REQUIRE(test_string == golden_string);
    REQUIRE(test_string_v == golden_string);
    REQUIRE(test_string_cv == golden_string);
    REQUIRE(test_string_v_p == golden_string_p);
  }

  SECTION("matrix transform")
  {
    // clang-format off
    fk::matrix<TestType> test {
     {0, 1, 2, 3},
     {4, 5, 6, 7},
    };
    fk::matrix<TestType> own(test);
    fk::matrix<TestType, mem_type::view> test_v(own);

    fk::matrix<TestType> own_p {
     {0, 1, 2, 3},
     {4, 5, 6, 7},
     {8, 9, 0, 1},
    };
    fk::matrix<TestType, mem_type::view> test_v_p(own_p, 0, 1, 0, 3);

    fk::matrix<TestType> const after {
     {1, 2, 3, 4},
     {5, 6, 7, 8},
    };

    fk::matrix<TestType> const after_p {
     {1, 2, 3, 4},
     {5, 6, 7, 8},
     {8, 9, 0, 1},
    };
    // clang-format on

    std::transform(test.begin(), test.end(), test.begin(),
                   std::bind(std::plus<TestType>(), std::placeholders::_1, 1));
    std::transform(test_v.begin(), test_v.end(), test_v.begin(),
                   std::bind(std::plus<TestType>(), std::placeholders::_1, 1));
    REQUIRE(test == after);
    REQUIRE(test_v == after);

    std::transform(test_v_p.begin(), test_v_p.end(), test_v_p.begin(),
                   std::bind(std::plus<TestType>(), std::placeholders::_1, 1));
    REQUIRE(test_v_p == after);
    // make sure we didn't modify the underlying matrix outside the view
    REQUIRE(own_p == after_p);
  }

  SECTION("matrix maximum element")
  {
    // clang-format off
    fk::matrix<TestType> test {
     {1, 2, 3, 4},
     {5, 6, 11, 8},
    }; // clang-format on
    fk::matrix<TestType, mem_type::view> const test_v(test);
    fk::matrix<TestType, mem_type::const_view> const test_cv(test);
    fk::matrix<TestType, mem_type::const_view> const test_v_p(test, 0, 1, 1, 2);

    TestType const max = 11;
    REQUIRE(*std::max_element(test.begin(), test.end()) == max);
    REQUIRE(*std::max_element(test_v.begin(), test_v.end()) == max);
    REQUIRE(*std::max_element(test_cv.begin(), test_cv.end()) == max);
    REQUIRE(*std::max_element(test_v_p.begin(), test_v_p.end()) == max);
  }

  SECTION("matrix sum of elements")
  {
    // clang-format off
    fk::matrix<TestType> test {
     {1, 2, 3, 4},
     {5, 6, 7, 8},
    }; // clang-format on

    fk::matrix<TestType, mem_type::view> const test_v(test);
    fk::matrix<TestType, mem_type::const_view> const test_cv(test);
    fk::matrix<TestType, mem_type::const_view> const test_v_p(test, 0, 0, 1, 3);

    TestType const sum   = 36;
    TestType const sum_p = 9;
    REQUIRE(std::accumulate(test.begin(), test.end(), 0) == sum);
    REQUIRE(std::accumulate(test_v.begin(), test_v.end(), 0) == sum);
    REQUIRE(std::accumulate(test_cv.begin(), test_cv.end(), 0) == sum);
    REQUIRE(std::accumulate(test_v_p.begin(), test_v_p.end(), 0) == sum_p);
  }
} // end fk::matrix utilities
#ifdef ASGARD_USE_CUDA
TEMPLATE_TEST_CASE("fk::matrix device transfer functions", "[tensors]",
                   test_precs, int)
{
  // clang-format off
  fk::matrix<TestType> const gold = {{ 1,  3,  5,  7,  9},
  				     {11, 13, 15, 17, 19}};
  // clang-format on

  SECTION("ctors")
  {
    SECTION("default")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> const mat;
      REQUIRE(mat.empty());
      REQUIRE(mat.data() == nullptr);
    }

    SECTION("from init list")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> const mat(
          {{1, 3, 5, 7, 9}, {11, 13, 15, 17, 19}});
      fk::matrix<TestType, mem_type::owner, resource::host> const copy(
          mat.clone_onto_host());
      REQUIRE(copy == gold);
    }
    SECTION("from size w/ copy to device")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> mat(2, 5);
      mat.transfer_from(gold);
      fk::matrix<TestType, mem_type::owner, resource::host> const copy(
          mat.clone_onto_host());
      REQUIRE(copy == gold);
    }

    SECTION("transfer - new matrix - owner device to host")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> const mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::owner, resource::host> const copy(
          mat.clone_onto_host());
      REQUIRE(copy == gold);
    }

    SECTION("transfer - new matrix - owner host to device")
    {
      fk::matrix<TestType, mem_type::owner, resource::host> const mat(gold);
      fk::matrix<TestType, mem_type::owner, resource::device> const copy(
          mat.clone_onto_device());
      fk::matrix<TestType, mem_type::owner, resource::host> const mat_h(
          copy.clone_onto_host());
      REQUIRE(mat_h == gold);
    }

    SECTION("transfer - new matrix - view device to host")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::view, resource::device> const mat_view(
          mat);
      fk::matrix<TestType, mem_type::owner, resource::host> const copy(
          mat_view.clone_onto_host());
      REQUIRE(copy == gold);
    }

    SECTION("transfer - new matrix - const view device to host")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> const mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::const_view, resource::device> const
          mat_view(mat);
      fk::matrix<TestType, mem_type::owner, resource::host> const copy(
          mat_view.clone_onto_host());
      REQUIRE(copy == gold);
    }

    SECTION("transfer - new matrix - view host to device")
    {
      fk::matrix<TestType, mem_type::owner, resource::host> mat(gold);
      fk::matrix<TestType, mem_type::view, resource::host> const mat_view(mat);
      fk::matrix<TestType, mem_type::owner, resource::device> const copy(
          mat_view.clone_onto_device());
      fk::matrix<TestType, mem_type::owner, resource::host> const mat_h(
          copy.clone_onto_host());
      REQUIRE(mat_h == gold);
    }

    SECTION("transfer - new matrix - const view host to device")
    {
      fk::matrix<TestType, mem_type::owner, resource::host> const mat(gold);
      fk::matrix<TestType, mem_type::const_view, resource::host> const mat_view(
          mat);
      fk::matrix<TestType, mem_type::owner, resource::device> const copy(
          mat_view.clone_onto_device());
      fk::matrix<TestType, mem_type::owner, resource::host> const mat_h(
          copy.clone_onto_host());
      REQUIRE(mat_h == gold);
    }
  }

  SECTION("copy and move")
  {
    SECTION("copy owner")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> const mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::owner, resource::device> const copy_d(mat);
      fk::matrix<TestType, mem_type::owner, resource::host> const copy_h(
          copy_d.clone_onto_host());
      REQUIRE(copy_h == gold);
    }

    SECTION("move owner")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::owner, resource::device> const moved_d(
          std::move(mat));
      REQUIRE(mat.data() == nullptr);
      REQUIRE(mat.empty());
      fk::matrix<TestType, mem_type::owner, resource::host> const moved_h(
          moved_d.clone_onto_host());
      REQUIRE(moved_h == gold);
    }

    SECTION("copy view")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::view, resource::device> const view_d(mat);

      fk::matrix<TestType, mem_type::view, resource::device> const view_copy_d(
          view_d);

      fk::matrix<TestType, mem_type::owner, resource::host> const copy_h(
          view_copy_d.clone_onto_host());
      REQUIRE(copy_h == gold);
    }

    SECTION("copy const view")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> const mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::const_view, resource::device> const view_d(
          mat);

      fk::matrix<TestType, mem_type::const_view, resource::device> const
          view_copy_d(view_d);

      fk::matrix<TestType, mem_type::owner, resource::host> const copy_h(
          view_copy_d.clone_onto_host());
      REQUIRE(copy_h == gold);
    }

    SECTION("move view")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> mat(
          gold.clone_onto_device());

      fk::matrix<TestType, mem_type::view, resource::device> view_d(mat);

      fk::matrix<TestType, mem_type::view, resource::device> view_moved_d(
          std::move(view_d));
      REQUIRE(view_d.data() == nullptr);
      REQUIRE(view_d.empty());
      REQUIRE(view_moved_d.data() == mat.data());

      fk::matrix<TestType, mem_type::owner, resource::host> const moved_h(
          view_moved_d.clone_onto_host());
      REQUIRE(moved_h == gold);
    }

    SECTION("move const view")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> mat(
          gold.clone_onto_device());

      fk::matrix<TestType, mem_type::const_view, resource::device> view_d(mat);

      fk::matrix<TestType, mem_type::const_view, resource::device> view_moved_d(
          std::move(view_d));
      REQUIRE(view_d.data() == nullptr);
      REQUIRE(view_d.empty());
      REQUIRE(view_moved_d.data() == mat.data());

      fk::matrix<TestType, mem_type::owner, resource::host> const moved_h(
          view_moved_d.clone_onto_host());
      REQUIRE(moved_h == gold);
    }
  }

  SECTION("transfers and copies")
  {
    SECTION("owner device to owner host")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> const mat(
          gold.clone_onto_device());
      fk::matrix<TestType> mat_h(2, 5);
      mat_h.transfer_from(mat);
      REQUIRE(mat_h == gold);
    }

    SECTION("owner device to view host")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> const mat(
          gold.clone_onto_device());
      fk::matrix<TestType> mat_h(2, 5);
      fk::matrix<TestType, mem_type::view> mat_view(mat_h);
      mat_view.transfer_from(mat);
      REQUIRE(mat_view == gold);
    }

    SECTION("owner device to owner device")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> const mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::owner, resource::device> mat_d(2, 5);
      mat_d = mat;
      fk::matrix<TestType> const mat_h(mat_d.clone_onto_host());
      REQUIRE(mat_h == gold);
    }

    SECTION("owner device to view device")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> const mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::owner, resource::device> mat_d(2, 5);
      fk::matrix<TestType, mem_type::view, resource::device> mat_view(mat_d);
      mat_view = mat;
      fk::matrix<TestType, mem_type::owner> const mat_h(
          mat_view.clone_onto_host());
      REQUIRE(mat_h == gold);
    }

    SECTION("view device to owner host")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::view, resource::device> const mat_view(
          mat);
      fk::matrix<TestType, mem_type::owner, resource::host> mat_h(2, 5);
      mat_h.transfer_from(mat_view);
      REQUIRE(mat_h == gold);
    }

    SECTION("view device to owner device")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::view, resource::device> const mat_view(
          mat);
      fk::matrix<TestType, mem_type::owner, resource::device> mat_d(2, 5);
      mat_d = mat_view;
      fk::matrix<TestType, mem_type::owner, resource::host> const mat_h(
          mat_d.clone_onto_host());
      REQUIRE(mat_h == gold);
    }

    SECTION("view device to view host")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::view, resource::device> const mat_view(
          mat);
      fk::matrix<TestType, mem_type::owner, resource::host> mat_h(2, 5);
      fk::matrix<TestType, mem_type::view, resource::host> mat_view_h(mat_h);
      mat_view_h.transfer_from(mat_view);
      REQUIRE(mat_view_h == gold);
    }

    SECTION("view device to view device")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::view, resource::device> const mat_view(
          mat);
      fk::matrix<TestType, mem_type::owner, resource::device> mat_d(2, 5);
      fk::matrix<TestType, mem_type::view, resource::device> mat_view_2(mat_d);
      mat_view_2 = mat_view;
      fk::matrix<TestType, mem_type::owner, resource::host> const mat_h(
          mat_view_2.clone_onto_host());
      REQUIRE(mat_h == gold);
    }

    SECTION("const view device to owner host")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> const mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::const_view, resource::device> const
          mat_view(mat);
      fk::matrix<TestType, mem_type::owner, resource::host> mat_h(2, 5);
      mat_h.transfer_from(mat_view);
      REQUIRE(mat_h == gold);
    }

    SECTION("const view device to owner device")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> const mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::const_view, resource::device> const
          mat_view(mat);
      fk::matrix<TestType, mem_type::owner, resource::device> mat_d(2, 5);
      mat_d = mat_view;
      fk::matrix<TestType, mem_type::owner, resource::host> const mat_h(
          mat_d.clone_onto_host());
      REQUIRE(mat_h == gold);
    }

    SECTION("const view device to view host")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> const mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::const_view, resource::device> const
          mat_view(mat);
      fk::matrix<TestType, mem_type::owner, resource::host> mat_h(2, 5);
      fk::matrix<TestType, mem_type::view, resource::host> mat_view_h(mat_h);
      mat_view_h.transfer_from(mat_view);
      REQUIRE(mat_view_h == gold);
    }

    SECTION("const view device to view device")
    {
      fk::matrix<TestType, mem_type::owner, resource::device> const mat(
          gold.clone_onto_device());
      fk::matrix<TestType, mem_type::const_view, resource::device> const
          mat_view(mat);
      fk::matrix<TestType, mem_type::owner, resource::device> mat_d(2, 5);
      fk::matrix<TestType, mem_type::view, resource::device> mat_view_2(mat_d);
      mat_view_2 = mat_view;
      fk::matrix<TestType, mem_type::owner, resource::host> const mat_h(
          mat_view_2.clone_onto_host());
      REQUIRE(mat_h == gold);
    }

    SECTION("owner host to owner device")
    {
      fk::matrix<TestType, mem_type::owner, resource::host> const mat(gold);
      fk::matrix<TestType, mem_type::owner, resource::device> mat_d(2, 5);
      mat_d.transfer_from(mat);
      fk::matrix<TestType> const mat_h(mat_d.clone_onto_host());
      REQUIRE(mat_h == gold);
    }

    SECTION("owner host to view device")
    {
      fk::matrix<TestType, mem_type::owner, resource::host> const mat(gold);
      fk::matrix<TestType, mem_type::owner, resource::device> mat_d(2, 5);
      fk::matrix<TestType, mem_type::view, resource::device> mat_view(mat_d);
      mat_view.transfer_from(mat);
      fk::matrix<TestType> const mat_h(mat_view.clone_onto_host());
      REQUIRE(mat_h == gold);
    }

    SECTION("view host to owner device")
    {
      fk::matrix<TestType, mem_type::owner, resource::host> mat(gold);
      fk::matrix<TestType, mem_type::view, resource::host> const mat_view(mat);
      fk::matrix<TestType, mem_type::owner, resource::device> mat_d(2, 5);
      mat_d.transfer_from(mat_view);
      fk::matrix<TestType> const mat_h(mat_d.clone_onto_host());
      REQUIRE(mat_h == gold);
    }

    SECTION("view host to view device")
    {
      fk::matrix<TestType, mem_type::owner, resource::host> mat(gold);
      fk::matrix<TestType, mem_type::view, resource::host> const mat_view(mat);
      fk::matrix<TestType, mem_type::owner, resource::device> mat_d(2, 5);
      fk::matrix<TestType, mem_type::view, resource::device> mat_view_d(mat_d);
      mat_view_d.transfer_from(mat_view);
      fk::matrix<TestType> const mat_h(mat_view_d.clone_onto_host());
      REQUIRE(mat_h == gold);
    }

    SECTION("const view host to owner device")
    {
      fk::matrix<TestType, mem_type::owner, resource::host> const mat(gold);
      fk::matrix<TestType, mem_type::const_view, resource::host> const mat_view(
          mat);
      fk::matrix<TestType, mem_type::owner, resource::device> mat_d(2, 5);
      mat_d.transfer_from(mat_view);
      fk::matrix<TestType> const mat_h(mat_d.clone_onto_host());
      REQUIRE(mat_h == gold);
    }

    SECTION("const view host to view device")
    {
      fk::matrix<TestType, mem_type::owner, resource::host> const mat(gold);
      fk::matrix<TestType, mem_type::const_view, resource::host> const mat_view(
          mat);
      fk::matrix<TestType, mem_type::owner, resource::device> mat_d(2, 5);
      fk::matrix<TestType, mem_type::view, resource::device> mat_view_d(mat_d);
      mat_view_d.transfer_from(mat_view);
      fk::matrix<TestType> const mat_h(mat_view_d.clone_onto_host());
      REQUIRE(mat_h == gold);
    }
  }
  SECTION("clear and resize")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> gold_copy(
        gold.clone_onto_device());
    gold_copy.clear_and_resize(2, 1);
    // clang-format off
    fk::matrix<TestType> const test {
      {0},
      {0},
      };
    // clang-format on
    fk::matrix<TestType> const gold_copy_h(gold_copy.clone_onto_host());
    REQUIRE(gold_copy_h == test);
  }

  SECTION("views - semantics on device")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> mat(
        gold.clone_onto_device());

    fk::matrix<TestType, mem_type::const_view, resource::device> const
        mat_cview(mat);
    fk::matrix<TestType, mem_type::view, resource::device> mat_view(mat);

    fk::matrix<TestType, mem_type::view, resource::device> mat_view_v(mat_view);
    fk::matrix<TestType, mem_type::const_view, resource::device> const
        mat_view_cv(mat_view);
    fk::matrix<TestType, mem_type::const_view, resource::device> const
        mat_view_cv_2(mat_cview);

    REQUIRE(mat_view.data() == mat.data());
    REQUIRE(mat_cview.data() == mat.data());
    REQUIRE(mat_view_v.data() == mat.data());
    REQUIRE(mat_view_cv.data() == mat.data());
    REQUIRE(mat_view_cv_2.data() == mat.data());

    {
      fk::matrix<TestType, mem_type::owner, resource::host> const copy(
          mat_view.clone_onto_host());
      fk::matrix<TestType, mem_type::owner, resource::host> const ccopy(
          mat_cview.clone_onto_host());

      fk::matrix<TestType, mem_type::owner, resource::host> const copy_2(
          mat_view_v.clone_onto_host());
      fk::matrix<TestType, mem_type::owner, resource::host> const ccopy_2(
          mat_view_cv.clone_onto_host());
      fk::matrix<TestType, mem_type::owner, resource::host> const ccopy_3(
          mat_view_cv_2.clone_onto_host());

      REQUIRE(copy == gold);
      REQUIRE(ccopy == gold);

      REQUIRE(copy_2 == gold);
      REQUIRE(ccopy_2 == gold);
      REQUIRE(ccopy_3 == gold);
    }
    fk::matrix<TestType, mem_type::owner, resource::host> const gold_2(
        {{1, 2, 3, 4, 5}, {11, 12, 13, 14, 15}});
    mat_view.transfer_from(gold_2);
    {
      fk::matrix<TestType, mem_type::owner, resource::host> const copy(
          mat.clone_onto_host());

      fk::matrix<TestType, mem_type::owner, resource::host> const copy_2(
          mat_view_cv.clone_onto_host());

      fk::matrix<TestType, mem_type::owner, resource::host> const copy_3(
          mat_view_v.clone_onto_host());

      fk::matrix<TestType, mem_type::owner, resource::host> const copy_4(
          mat_view_cv_2.clone_onto_host());

      REQUIRE(copy == gold_2);
      REQUIRE(copy_2 == gold_2);
      REQUIRE(copy_3 == gold_2);
      REQUIRE(copy_4 == gold_2);
    }

    fk::matrix<TestType, mem_type::owner, resource::host> const gold_3(
        {{1, 2, 3, 4, 5}, {15, 16, 17, 18, 19}});
    mat_view_v.transfer_from(gold_2);
    {
      fk::matrix<TestType, mem_type::owner, resource::host> const copy(
          mat.clone_onto_host());

      fk::matrix<TestType, mem_type::owner, resource::host> const copy_2(
          mat_view_cv.clone_onto_host());

      fk::matrix<TestType, mem_type::owner, resource::host> const copy_3(
          mat_view_v.clone_onto_host());

      fk::matrix<TestType, mem_type::owner, resource::host> const copy_4(
          mat_view_cv_2.clone_onto_host());

      REQUIRE(copy == gold_2);
      REQUIRE(copy_2 == gold_2);
      REQUIRE(copy_3 == gold_2);
      REQUIRE(copy_4 == gold_2);
    }
  }

  SECTION("views - copying - stride != num_rows - host to device")
  {
    fk::matrix<TestType, mem_type::const_view, resource::host> const gold_view(
        gold, 0, 0, 0, 4);
    fk::matrix<TestType, mem_type::owner, resource::host> const mat(gold);
    fk::matrix<TestType, mem_type::const_view, resource::host> const mat_view(
        mat, 0, 0, 0, 4);
    fk::matrix<TestType, mem_type::owner, resource::device> const mat_d(
        mat_view.clone_onto_device());
    fk::matrix<TestType, mem_type::owner, resource::host> const mat_h(
        mat_d.clone_onto_host());
    REQUIRE(mat_h == gold_view);
  }

  SECTION("views - copying - stride != num_rows - device to device")
  {
    fk::matrix<TestType, mem_type::const_view, resource::host> const gold_view(
        gold, 0, 0, 0, 4);
    fk::matrix<TestType, mem_type::owner, resource::host> const mat(gold);
    fk::matrix<TestType, mem_type::owner, resource::device> const mat_d(
        mat.clone_onto_device());
    fk::matrix<TestType, mem_type::const_view, resource::device> const mat_view(
        mat_d, 0, 0, 0, 4);
    fk::matrix<TestType, mem_type::owner, resource::device> const mat_d2(
        mat_view);
    fk::matrix<TestType, mem_type::owner, resource::host> const mat_h(
        mat_d2.clone_onto_host());
    REQUIRE(mat_h == gold_view);
  }

  SECTION("views - copying - stride != num_rows - device to host")
  {
    fk::matrix<TestType, mem_type::const_view, resource::host> const gold_view(
        gold, 0, 0, 0, 4);
    fk::matrix<TestType, mem_type::owner, resource::host> const mat(gold);
    fk::matrix<TestType, mem_type::owner, resource::device> const mat_d(
        mat.clone_onto_device());
    fk::matrix<TestType, mem_type::const_view, resource::device> const mat_view(
        mat_d, 0, 0, 0, 4);
    fk::matrix<TestType, mem_type::owner, resource::host> const mat_h(
        mat_view.clone_onto_host());
    REQUIRE(mat_h == gold_view);
  }
}
#endif
TEMPLATE_TEST_CASE("fk::matrix transpose", "[tensors]", test_precs)
{
  fk::matrix<TestType> const m_0 = {{0, 4, 8, 12, 16, 20, 24, 28},
                                    {1, 5, 9, 13, 17, 21, 25, 29},
                                    {2, 6, 10, 14, 18, 22, 26, 30},
                                    {3, 7, 11, 15, 19, 23, 27, 31}};

  fk::matrix<TestType> const m_1 = {
      {0, 1, 2, 3},     {4, 5, 6, 7},     {8, 9, 10, 11},   {12, 13, 14, 15},
      {16, 17, 18, 19}, {20, 21, 22, 23}, {24, 25, 26, 27}, {28, 29, 30, 31}};

  fk::matrix<TestType> const m_2 = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};

  fk::matrix<TestType> const m_3 = {{0}, {1}, {2}, {3}, {4},
                                    {5}, {6}, {7}, {8}, {9}};

  fk::matrix<TestType> const m_4 = {{0, 1, 2}, {3, 4, 5}};

  fk::matrix<TestType> const m_5 = {{0, 3}, {1, 4}, {2, 5}};

  fk::matrix<TestType> const m_6 = {
      {0, 4, 8, 12}, {1, 5, 9, 13}, {2, 6, 10, 14}, {3, 7, 11, 15}};

  fk::matrix<TestType> const m_7 = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};

  auto correct_transpose =
      [](fk::matrix<TestType> const &m,
         fk::matrix<TestType> const &m_transposed) -> bool {
    fk::matrix<TestType> m_copy(m);
    m_copy.transpose();
    return m_copy == m_transposed;
  };

  SECTION("in-place transpose")
  {
    REQUIRE(correct_transpose(m_1, m_0));
    REQUIRE(correct_transpose(m_0, m_1));
    REQUIRE(correct_transpose(m_4, m_5));
    REQUIRE(correct_transpose(m_5, m_4));
    REQUIRE(correct_transpose(m_2, m_3));
    REQUIRE(correct_transpose(m_3, m_2));
    REQUIRE(correct_transpose(m_6, m_7));
    REQUIRE(correct_transpose(m_7, m_6));
  }
}

TEMPLATE_TEST_CASE("Copy matrix functions for various resources", "[resources]",
                   test_precs, int)
{
  fk::matrix<TestType> const a{{1, 2}, {3, 4}};

  SECTION("check host -> host")
  {
    fk::matrix<TestType> b(a.nrows(), a.ncols());
    copy_matrix(b, a);
    REQUIRE(b == a);
  }

  SECTION("check host -> host view")
  {
    fk::matrix<TestType> b(a.nrows(), a.ncols());
    fk::matrix<TestType, mem_type::view> b_v(b);
    copy_matrix(b_v, a);
    REQUIRE(b_v == a);
    REQUIRE(b == a);
  }

  SECTION("check host view -> host")
  {
    fk::matrix<TestType, mem_type::const_view> a_v(a);
    fk::matrix<TestType> b(a.nrows(), a.ncols());

    copy_matrix(b, a_v);
    REQUIRE(b == a_v);
    REQUIRE(b == a);
  }

#ifdef ASGARD_USE_CUDA
  fk::matrix<TestType, mem_type::owner, resource::device> a_d(
      a.clone_onto_device());
  SECTION("check host -> device")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> b(a.nrows(),
                                                              a.ncols());
    copy_matrix(b, a);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check host -> device view")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> b(a.nrows(),
                                                              a.ncols());
    fk::matrix<TestType, mem_type::view, resource::device> b_v(b);
    copy_matrix(b_v, a);
    REQUIRE(b_v.clone_onto_host() == a);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check host view -> device")
  {
    fk::matrix<TestType, mem_type::const_view> a_v(a);
    fk::matrix<TestType, mem_type::owner, resource::device> b_d(a.nrows(),
                                                                a.ncols());

    copy_matrix(b_d, a_v);
    REQUIRE(b_d.clone_onto_host() == a_v);
    REQUIRE(b_d.clone_onto_host() == a);
  }

  SECTION("check device -> host")
  {
    fk::matrix<TestType> b(a.nrows(), a.ncols());
    copy_matrix(b, a_d);
    REQUIRE(b == a);
  }

  SECTION("check device -> host view")
  {
    fk::matrix<TestType, mem_type::owner> b(a.nrows(), a.ncols());
    fk::matrix<TestType, mem_type::view> b_v(b);
    copy_matrix(b_v, a_d);
    REQUIRE(b_v == a);
    REQUIRE(b == a);
  }

  SECTION("check device view -> host")
  {
    fk::matrix<TestType, mem_type::const_view> a_v(a);
    fk::matrix<TestType, mem_type::owner, resource::device> b(a.nrows(),
                                                              a.ncols());

    copy_matrix(b, a_v);
    REQUIRE(b.clone_onto_host() == a_v);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check device -> device")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> b(a.nrows(),
                                                              a.ncols());
    copy_matrix(b, a_d);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check device -> device view")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> b(a.nrows(),
                                                              a.ncols());
    fk::matrix<TestType, mem_type::view, resource::device> b_v(b);
    copy_matrix(b_v, a_d);
    REQUIRE(b_v.clone_onto_host() == a);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check device view -> device")
  {
    fk::matrix<TestType, mem_type::const_view, resource::device> a_v(a_d);
    fk::matrix<TestType, mem_type::owner, resource::device> b(a.nrows(),
                                                              a.ncols());
    copy_matrix(b, a_v);
    REQUIRE(b.clone_onto_host() == a_v.clone_onto_host());
    REQUIRE(b.clone_onto_host() == a);
  }
#endif
}

TEMPLATE_TEST_CASE("Copy vector functions for various resources", "[resources]",
                   test_precs, int)
{
  fk::vector<TestType> const a{1, 2, 3, 4};

  SECTION("check host -> host")
  {
    fk::vector<TestType> b(a.size());
    copy_vector(b, a);
    REQUIRE(b == a);
  }

  SECTION("check host -> host view")
  {
    fk::vector<TestType> b(a.size());
    fk::vector<TestType, mem_type::view> b_v(b);
    copy_vector(b_v, a);
    REQUIRE(b_v == a);
    REQUIRE(b == a);
  }

  SECTION("check host view -> host")
  {
    fk::vector<TestType, mem_type::const_view> a_v(a);
    fk::vector<TestType> b(a.size());
    copy_vector(b, a_v);
    REQUIRE(b == a_v);
    REQUIRE(b == a);
  }

#ifdef ASGARD_USE_CUDA
  fk::vector<TestType, mem_type::owner, resource::device> a_d(
      a.clone_onto_device());
  SECTION("check host -> device")
  {
    fk::vector<TestType, mem_type::owner, resource::device> b(a.size());
    copy_vector(b, a);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check host -> device view")
  {
    fk::vector<TestType, mem_type::owner, resource::device> b(a.size());
    fk::vector<TestType, mem_type::view, resource::device> b_v(b);
    copy_vector(b_v, a);
    REQUIRE(b_v.clone_onto_host() == a);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check host view -> device")
  {
    fk::vector<TestType, mem_type::const_view> a_v(a);
    fk::vector<TestType, mem_type::owner, resource::device> b_d(a.size());
    copy_vector(b_d, a_v);
    REQUIRE(b_d.clone_onto_host() == a_v);
    REQUIRE(b_d.clone_onto_host() == a);
  }

  SECTION("check device -> host")
  {
    fk::vector<TestType> b(a.size());
    copy_vector(b, a_d);
    REQUIRE(b == a);
  }

  SECTION("check device -> host view")
  {
    fk::vector<TestType, mem_type::owner> b(a.size());
    fk::vector<TestType, mem_type::view> b_v(b);
    copy_vector(b_v, a_d);
    REQUIRE(b_v == a);
    REQUIRE(b == a);
  }

  SECTION("check device view -> host")
  {
    fk::vector<TestType, mem_type::const_view> a_v(a);
    fk::vector<TestType, mem_type::owner, resource::device> b(a.size());

    copy_vector(b, a_v);
    REQUIRE(b.clone_onto_host() == a_v);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check device -> device")
  {
    fk::vector<TestType, mem_type::owner, resource::device> b(a.size());
    copy_vector(b, a_d);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check device -> device view")
  {
    fk::vector<TestType, mem_type::owner, resource::device> b(a.size());
    fk::vector<TestType, mem_type::view, resource::device> b_v(b);
    copy_vector(b_v, a_d);
    REQUIRE(b_v.clone_onto_host() == a);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check device view -> device")
  {
    fk::vector<TestType, mem_type::const_view, resource::device> a_v(a_d);
    fk::vector<TestType, mem_type::owner, resource::device> b(a.size());
    copy_vector(b, a_v);
    REQUIRE(b.clone_onto_host() == a_v.clone_onto_host());
    REQUIRE(b.clone_onto_host() == a);
  }
#endif
}

#ifdef ASGARD_USE_CUDA
template<typename vec_type_1, typename vec_type_2>
bool data_match(vec_type_1 const &v1, vec_type_2 const &v2)
{
  if (static_cast<int64_t>(v1.size()) != static_cast<int64_t>(v2.size()))
    return false;
  static_assert(std::is_same_v<typename vec_type_1::value_type, typename vec_type_2::value_type>);
  std::vector<typename vec_type_1::value_type> x = v1;
  std::vector<typename vec_type_2::value_type> y = v2;
  if (x.size() != y.size()) // something happened during copy
    return false;
  for (size_t i = 0; i < x.size(); i++)
    if (x[i] != y[i]) // this checks data copies, so it's OK for floating point numbers
      return false;
  return true;
}

TEMPLATE_TEST_CASE("gpu::vector", "[gpu::vector]", test_precs, int)
{
  SECTION("allocate_device, delete_device")
  {
    gpu::vector<TestType> gpu0; // make empty
    REQUIRE(gpu0.size() == 0);
    REQUIRE(gpu0.data() == nullptr);
    REQUIRE(gpu0.empty());

    gpu0.resize(10); // resize
    REQUIRE(gpu0.size() == 10);
    REQUIRE(gpu0.data() != nullptr);
    REQUIRE(not gpu0.empty());

    gpu0 = gpu::vector<TestType>(); // move-assign
    REQUIRE(gpu0.size() == 0);
    REQUIRE(gpu0.data() == nullptr);

    std::vector<TestType> cpu1 = {1, 2, 3, 4};
    gpu::vector<TestType> gpu1(cpu1); // copy construct (std::vector)
    REQUIRE(data_match(cpu1, gpu1));

    gpu::vector<TestType> gpu2(std::vector<TestType>{1, 2}); // move construct
    REQUIRE(data_match(std::vector<TestType>{1, 2}, gpu2));

    std::vector<TestType> cpu2;
    cpu2 = gpu0 = gpu2 = cpu1; // copy assignments
    REQUIRE(data_match(cpu1, gpu2));
    REQUIRE(data_match(gpu2, gpu0));
    REQUIRE(data_match(gpu0, cpu2));

    gpu0 = std::vector<TestType>{1, 2, 3, 4, 5, 6}; // move assign (std::vector)
    REQUIRE(data_match(std::vector<TestType>{1, 2, 3, 4, 5, 6}, gpu0));

    gpu1 = std::move(gpu0); // move assign
    REQUIRE(gpu0.size() == 0);
    REQUIRE(gpu0.data() == nullptr);
    REQUIRE(data_match(std::vector<TestType>{1, 2, 3, 4, 5, 6}, gpu1));

    gpu1.clear();
    REQUIRE(gpu1.size() == 0);
    REQUIRE(gpu1.data() == nullptr);
    REQUIRE(gpu1.empty());

    cpu1 = {1, 2, 3, 4, 5, 6, 7, 8};
    gpu0 = cpu1;
    gpu::vector<TestType> gpu3(std::move(gpu0)); // move construct
    REQUIRE(gpu0.empty());
    REQUIRE(data_match(cpu1, gpu3));
  }
}
#endif
