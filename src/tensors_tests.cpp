
#include "tensors.hpp"
#include "tests_general.hpp"
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>

#include "matlab_utilities.hpp"

// note using widening conversions to floating point type in order to use same
// tests for integer type
// FIXME look for another way to do this

TEMPLATE_TEST_CASE("fk::vector interface: constructors, copy/move", "[tensors]",
                   float, double, int)
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
    REQUIRE(test.size() == 0);
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
    REQUIRE(test_v2 == gold_portion);
    gold_copy(2) = 1000;
    REQUIRE(test_v2 == gold_copy.extract(1, 3));
    REQUIRE(gold_copy != gold);
    REQUIRE(test_v2 != gold_portion);
    test_v2(2) = 10;
    REQUIRE(test_v2 == gold_copy.extract(1, 3));
    REQUIRE(gold_copy != gold);
    REQUIRE(test_v2 != gold_portion);

    // empty case
    fk::vector<TestType> const empty;
    fk::vector<TestType, mem_type::view> const empty_v(empty);
    REQUIRE(empty_v == empty);
    REQUIRE(empty_v.data() == nullptr);
    REQUIRE(empty_v.size() == 0);
  }
  SECTION("construct owner from view")
  {
    fk::vector<TestType, mem_type::view> gold_v(gold);
    fk::vector<TestType> test(gold_v);
    REQUIRE(test == gold);
  }
  SECTION("copy assign to owner from view")
  {
    fk::vector<TestType, mem_type::view> gold_v(gold);
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
  }
  SECTION("copy assignment")
  {
    fk::vector<TestType> test(5);
    test = gold;
    REQUIRE(test == gold);

    fk::vector<TestType> base(5);
    fk::vector<TestType, mem_type::view> test_v(base);
    fk::vector<TestType, mem_type::view> gold_v(gold);
    test_v = gold_v;
    REQUIRE(test_v == gold);
  }
  SECTION("converting construction (from owners)")
  {
    fk::vector<int> testi(gold);
    // fk::vector<int, mem_type::view> testi_v(gold); // disabled
    REQUIRE(testi == goldi);
    // REQUIRE(testi_v == goldi);
    fk::vector<float> testf(gold);
    // fk::vector<float, mem_type::view> testf_v(gold);
    REQUIRE(testf == goldf);
    // REQUIRE(testf_v == goldf);
    fk::vector<double> testd(gold);
    // fk::vector<double, mem_type::view> testd_v(gold);
    REQUIRE(testd == goldd);
    // REQUIRE(testd_v == goldd);
  }
  SECTION("converting construction (from views)")
  {
    fk::vector<TestType, mem_type::view> const gold_v(gold);
    fk::vector<int> testi(gold_v);
    // fk::vector<int, mem_type::view> testi_v(gold_v); // disabled
    REQUIRE(testi == goldi);
    // REQUIRE(testi_v == goldi);
    fk::vector<float> testf(gold_v);
    // fk::vector<float, mem_type::view> testf_v(gold_v);
    REQUIRE(testf == goldf);
    // REQUIRE(testf_v == goldf);
    fk::vector<double> testd(gold_v);
    // fk::vector<double, mem_type::view> testd_v(gold_v);
    REQUIRE(testd == goldd);
    // REQUIRE(testd_v == goldd);
  }
  SECTION("converting assignment (from owners)")
  {
    fk::vector<int> testi(5);
    fk::vector<int, mem_type::view> testi_v(testi);
    testi   = gold;
    testi_v = gold;
    REQUIRE(testi == goldi);
    REQUIRE(testi_v == goldi);
    fk::vector<float> testf(5);
    fk::vector<float, mem_type::view> testf_v(testf);
    testf   = gold;
    testf_v = gold;
    REQUIRE(testf == goldf);
    REQUIRE(testf_v == goldf);
    fk::vector<double> testd(5);
    fk::vector<double, mem_type::view> testd_v(testd);
    testd   = gold;
    testd_v = gold;
    REQUIRE(testd == goldd);
    REQUIRE(testd_v == goldd);
  }
  SECTION("converting assignment (from views)")
  {
    fk::vector<TestType, mem_type::view> const gold_v(gold);
    fk::vector<int> testi(5);
    fk::vector<int, mem_type::view> testi_v(testi);
    testi   = gold_v;
    testi_v = gold_v;
    REQUIRE(testi == goldi);
    REQUIRE(testi_v == goldi);
    fk::vector<float> testf(5);
    fk::vector<float, mem_type::view> testf_v(testf);
    testf   = gold_v;
    testf_v = gold_v;
    REQUIRE(testf == goldf);
    REQUIRE(testf_v == goldf);
    fk::vector<double> testd(5);
    fk::vector<double, mem_type::view> testd_v(testd);
    testd   = gold_v;
    testd_v = gold_v;
    REQUIRE(testd == goldd);
    REQUIRE(testd_v == goldd);
  }
  SECTION("move construction")
  {
    fk::vector<TestType> moved{2, 3, 4, 5, 6};
    fk::vector<TestType> test(std::move(moved));
    REQUIRE(test == gold);

    fk::vector<TestType> moved_o{2, 3, 4, 5, 6};
    fk::vector<TestType, mem_type::view> moved_v(moved_o);
    fk::vector<TestType, mem_type::view> test_v(std::move(moved_v));
    REQUIRE(test_v == gold);
  }
  SECTION("move assignment")
  {
    fk::vector<TestType> moved{2, 3, 4, 5, 6};
    fk::vector<TestType> test(5);
    test = std::move(moved);
    REQUIRE(test == gold);

    fk::vector<TestType> moved_o{2, 3, 4, 5, 6};
    fk::vector<TestType, mem_type::view> moved_v(moved_o);
    fk::vector<TestType> test_o(5);
    fk::vector<TestType, mem_type::view> test_v(test_o);
    test_v = std::move(moved_v);
    REQUIRE(test_v == gold);
  }
  SECTION("copy from std::vector")
  {
    std::vector<TestType> v{2, 3, 4, 5, 6};
    fk::vector<TestType> test(5);
    fk::vector<TestType, mem_type::view> test_v(test);
    test   = v;
    test_v = v;
    REQUIRE(test_v == gold);
  }
  SECTION("copy into std::vector")
  {
    std::vector<TestType> goldv{2, 3, 4, 5, 6};

    std::vector<TestType> testv;
    testv = gold.to_std();
    compare_vectors(testv, goldv);

    fk::vector<TestType, mem_type::view> gold_v(gold);
    std::vector<TestType> testv_v;
    testv_v = gold_v.to_std();
    compare_vectors(testv_v, goldv);
  }
} // end fk::vector constructors, copy/move

TEMPLATE_TEST_CASE("fk::vector operators", "[tensors]", double, float, int)
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
    fk::vector<TestType, mem_type::view> const gold_v(gold);
    REQUIRE(gold_v(4) == 6);
  }
  SECTION("comparison operator") // this gets used in every REQUIRE
  SECTION("comparison (negated) operator")
  {
    fk::vector<TestType> test(gold);
    fk::vector<TestType> gold_copy(gold);
    fk::vector<TestType, mem_type::view> test_v(gold_copy);
    fk::vector<TestType> const empty;
    fk::vector<TestType, mem_type::view> const empty_v(empty);

    test(4)   = 333;
    test_v(4) = 333;

    REQUIRE(test != gold);
    REQUIRE(test_v != gold);
    REQUIRE(empty != gold);
    REQUIRE(empty_v != gold);
  }
  SECTION("less than operator")
  {
    fk::vector<TestType> const empty;
    fk::vector<TestType, mem_type::view> const empty_v(empty);
    ;
    fk::vector<TestType> const gold_copy = gold;
    fk::vector<TestType, mem_type::view> const gold_copy_v(gold);

    fk::vector<TestType> const gold_prefix{1, 2, 3, 4};
    fk::vector<TestType, mem_type::view> const gold_prefix_v(gold_prefix);
    fk::vector<TestType> const mismatch{2, 3, 5, 5, 6};
    fk::vector<TestType, mem_type::view> const mismatch_v(mismatch);

    // equal vectors return false
    REQUIRE(!(gold_copy < gold));
    REQUIRE(!(gold_copy_v < gold));
    // empty range less than non-empty range
    REQUIRE(empty < gold);
    REQUIRE(empty_v < gold);
    // a prefix is less than the complete range
    REQUIRE(gold_prefix < gold);
    REQUIRE(gold_prefix_v < gold);
    // otherwise compare on first mismatch
    REQUIRE(gold < mismatch);
    REQUIRE(gold < mismatch_v);
    // also, empty ranges are equal
    REQUIRE(!(empty < empty));
    REQUIRE(!(empty < empty_v));
  }

  SECTION("addition operator")
  {
    fk::vector<TestType> const in1{1, 1, 1, 1, 1};
    fk::vector<TestType, mem_type::view> const in1_v(in1);
    fk::vector<TestType> const in2{1, 2, 3, 4, 5};
    fk::vector<TestType, mem_type::view> const in2_v(in2);
    REQUIRE((in1 + in2) == gold);
    REQUIRE((in1_v + in2_v) == gold);
    REQUIRE((in1 + in2_v) == gold);
    REQUIRE((in1_v + in2) == gold);
  }
  SECTION("subtraction operator")
  {
    fk::vector<TestType> const in1{3, 4, 5, 6, 7};
    fk::vector<TestType, mem_type::view> const in1_v(in1);
    fk::vector<TestType> const in2{1, 1, 1, 1, 1};
    fk::vector<TestType, mem_type::view> const in2_v(in2);
    REQUIRE((in1 - in2) == gold);
    REQUIRE((in1_v - in2_v) == gold);
    REQUIRE((in1 - in2_v) == gold);
    REQUIRE((in1_v - in2) == gold);
  }
  SECTION("vector*vector operator")
  {
    fk::vector<TestType, mem_type::view> const gold_v(gold);
    REQUIRE((gold * gold) == 90);
    REQUIRE((gold_v * gold_v) == 90);
    REQUIRE((gold_v * gold) == 90);
    REQUIRE((gold * gold_v) == 90);
  }
  SECTION("vector*matrix operator")
  {
    // clang-format off
    fk::matrix<TestType> const test_mat{
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {16, 26, 36},
    }; // clang-format on
    fk::matrix<TestType, mem_type::view> const test_mat_v(test_mat);
    fk::vector<TestType> const test_vect{2, 3, 4, 5, 6};
    fk::vector<TestType, mem_type::view> const test_vect_v(test_vect);
    fk::vector<TestType> const gold{290, 490, 690};

    REQUIRE((test_vect * test_mat_v) == gold);
    REQUIRE((test_vect_v * test_mat_v) == gold);
    REQUIRE((test_vect * test_mat) == gold);
    REQUIRE((test_vect_v * test_mat) == gold);
  }
  SECTION("vector*scalar operator")
  {
    TestType const scale = static_cast<TestType>(-2);
    fk::vector<TestType> const gold_scaled{-4, -6, -8, -10, -12};
    fk::vector<TestType, mem_type::view> const gold_v(gold);
    REQUIRE((gold * scale) == gold_scaled);
    REQUIRE((gold_v * scale) == gold_scaled);
  }
  SECTION("vector (as matrix) kron product")
  {
    fk::vector<TestType, mem_type::view> const gold_v(gold);

    fk::vector<TestType> const identity{1};
    fk::vector<TestType, mem_type::view> const identity_v(identity);
    REQUIRE(identity.single_column_kron(gold) == gold);
    REQUIRE(identity_v.single_column_kron(gold) == gold);
    REQUIRE(identity.single_column_kron(gold_v) == gold);
    REQUIRE(identity_v.single_column_kron(gold_v) == gold);

    fk::vector<TestType> const repeat{1, 1};
    fk::vector<TestType, mem_type::view> const repeat_v(repeat);
    fk::vector<TestType> const gold_repeated =
        fk::vector<TestType>(gold).concat(gold);
    REQUIRE(repeat.single_column_kron(gold) == gold_repeated);
    REQUIRE(repeat.single_column_kron(gold_v) == gold_repeated);
    REQUIRE(repeat_v.single_column_kron(gold) == gold_repeated);
    REQUIRE(repeat_v.single_column_kron(gold_v) == gold_repeated);

    fk::vector<TestType> const zeros(gold.size());
    fk::vector<TestType> const alternate{1, 0, 2, 0};
    fk::vector<TestType, mem_type::view> const alternate_v(alternate);
    fk::vector<TestType> const ans =
        fk::vector<TestType>(gold).concat(zeros).concat(gold * 2).concat(zeros);
    REQUIRE(ans == alternate.single_column_kron(gold));
    REQUIRE(ans == alternate_v.single_column_kron(gold_v));
    REQUIRE(ans == alternate_v.single_column_kron(gold));
    REQUIRE(ans == alternate.single_column_kron(gold_v));
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
} // end fk::vector operators

TEMPLATE_TEST_CASE("fk::vector utilities", "[tensors]", double, float, int)
{
  fk::vector<TestType> const gold{2, 3, 4, 5, 6};
  fk::vector<TestType, mem_type::view> const gold_v(
      gold); // increases ref_count
  SECTION("size(): the number of elements")
  {
    REQUIRE(gold.size() == 5);
    REQUIRE(gold_v.size() == 5);
  }
  SECTION("data(): const addr to element")
  {
    REQUIRE(*gold.data(4) == 6);
    REQUIRE(*gold_v.data(4) == 6);
  }
  SECTION("print out the values")
  {
    // (effectively) redirect cout
    std::streambuf *old_cout_stream_buf = std::cout.rdbuf();
    std::ostringstream test_str;
    std::ostringstream test_str_v;

    // generate the output (into test_str)
    std::cout.rdbuf(test_str.rdbuf());
    gold.print("golden vector");
    std::cout.rdbuf(test_str_v.rdbuf());
    gold_v.print("golden vector");

    // restore cout destination
    std::cout.rdbuf(old_cout_stream_buf);
    std::string golden_string, golden_string_v;
    if constexpr (std::is_floating_point<TestType>::value)
    {
      golden_string =
          "golden vector(owner, ref_count = 2)\n  2.0000e+00  3.0000e+00  "
          "4.0000e+00  5.0000e+00  6.0000e+00\n";
      golden_string_v = "golden vector(view)\n  2.0000e+00  3.0000e+00  "
                        "4.0000e+00  5.0000e+00  6.0000e+00\n";
    }
    else
    {
      golden_string = "golden vector(owner, ref_count = 2)\n2 3 "
                      "4 5 6 \n";

      golden_string_v = "golden vector(view)\n2 3 "
                        "4 5 6 \n";
    }
    REQUIRE(test_str.str() == golden_string);
    REQUIRE(test_str_v.str() == golden_string_v);
  }
  SECTION("dump to octave")
  {
    gold.dump_to_octave("test_out.dat");
    gold_v.dump_to_octave("test_out_v.dat");
    std::ifstream data_stream("test_out.dat");
    std::ifstream data_stream_v("test_out_v.dat");
    std::string const test_string((std::istreambuf_iterator<char>(data_stream)),
                                  std::istreambuf_iterator<char>());
    std::string const test_string_v(
        (std::istreambuf_iterator<char>(data_stream_v)),
        std::istreambuf_iterator<char>());
    std::remove("test_out.dat");
    std::remove("test_out_v.dat");

    std::string golden_string;
    if constexpr (std::is_floating_point<TestType>::value)
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

  SECTION("vector concatenation")
  {
    fk::vector<TestType> test_left = {2, 3, 4};
    // fk::vector<TestType, mem_type::view> test_left_v(test_left); // disabled
    fk::vector<TestType> const test_right = {5, 6};
    fk::vector<TestType, mem_type::view> const test_right_v(test_right);

    REQUIRE(test_left.concat(test_right) == gold);
    test_left.resize(3) = fk::vector<TestType>({2, 3, 4});
    REQUIRE(test_left.concat(test_right_v) == gold);

    // REQUIRE(test_left_v.concat(test_right_v) == gold); // disabled

    fk::vector<TestType> empty;
    // have to make a copy to extract a view from,
    // because you can't concat on an owner w/ outstanding views
    fk::vector<TestType> empty_copy(empty);
    fk::vector<TestType, mem_type::view> empty_v(empty_copy);
    fk::vector<TestType> gold_copy(gold);
    fk::vector<TestType, mem_type::view> gold_v(gold_copy);

    REQUIRE(empty.concat(gold) == gold);
    empty.resize(0);
    REQUIRE(empty.concat(gold_v) == gold);
    empty.resize(0);

    // REQUIRE(empty_v.concat(gold) == gold);
    // empty_v.resize(0);
    // REQUIRE(empty_v.concat(gold_v) == gold);
    // empty_v.resize(0);

    // non-const gold copy I can concat with
    fk::vector<TestType> gold_2(gold);
    REQUIRE(gold_2.concat(empty) == gold);
    gold_2.resize(gold.size()) = gold;
    REQUIRE(gold_2.concat(empty_v) == gold);
    gold_2.resize(gold.size()) = gold;
    // REQUIRE(gold_copy_v.concat(empty) == gold);
    // gold_copy_v = gold_copy;
    // REQUIRE(gold_copy_v.concat(empty_v) == gold);
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

    fk::vector<TestType, mem_type::view> const empty_v(empty_copy);
    fk::vector<TestType, mem_type::view> const begin_v(begin_copy);
    fk::vector<TestType, mem_type::view> const middle_v(middle_copy);
    fk::vector<TestType, mem_type::view> const end_v(end_copy);

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
  }
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
    fk::vector<TestType> const test{5, 6, 11, 8};
    fk::vector<TestType, mem_type::view> const test_v(test);
    TestType const max = 11;
    REQUIRE(*std::max_element(test.begin(), test.end()) == max);
    REQUIRE(*std::max_element(test_v.begin(), test_v.end()) == max);
  }

  SECTION("vector sum of elements")
  {
    fk::vector<TestType> const test{1, 2, 3, 4, 5, 6, 7, 8};
    fk::vector<TestType, mem_type::view> const test_v(test);
    TestType const sum = 36;
    REQUIRE(std::accumulate(test.begin(), test.end(), 0.0) == sum);
    REQUIRE(std::accumulate(test_v.begin(), test_v.end(), 0.0) == sum);
  }

  SECTION("vector ref counting")
  {
    // on construction, vectors have 0 views
    fk::vector<TestType> const test;
    assert(test.get_num_views() == 0);
    fk::vector<TestType> const test_init({1});
    assert(test_init.get_num_views() == 0);
    fk::vector<TestType> const test_sz(1);
    assert(test_sz.get_num_views() == 0);
    fk::vector<TestType> const test_conv(fk::vector<int>(1));
    assert(test_conv.get_num_views() == 0);
    fk::vector<TestType> const test_copy(test);
    assert(test_copy.get_num_views() == 0);

    // creating views increments view count
    fk::vector<TestType, mem_type::view> const test_view(test);
    assert(test.get_num_views() == 1);
    fk::vector<TestType, mem_type::view> const test_view_2(test);
    assert(test.get_num_views() == 2);

    // copies have a fresh view count
    fk::vector<TestType> const test_cp(test);
    assert(test_cp.get_num_views() == 0);
    assert(test.get_num_views() == 2);

    // test that view count gets decremented when views go out of scope
    {
      fk::vector<TestType, mem_type::view> test_view_3(test);
      assert(test.get_num_views() == 3);
    }
    assert(test.get_num_views() == 2);
  }
} // end fk::vector utilities

TEMPLATE_TEST_CASE("fk::matrix interface: constructors, copy/move", "[tensors]",
                   double, float, int)
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

  fk::matrix<TestType> const gold_own(gold);
  fk::matrix<TestType, mem_type::view> const gold_v(gold_own);

  SECTION("default constructor")
  {
    fk::matrix<TestType> test;
    // fk::matrix<TestType, mem_type::view> test_v; // disabled
    REQUIRE(test.size() == 0);
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
    fk::matrix<TestType> test(gold);
    fk::matrix<TestType> test_v(gold_v);
    REQUIRE(test == gold);
    REQUIRE(test_v == gold);
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
  }
  SECTION("converting copy construction")
  {
    fk::matrix<int> testi(gold);
    fk::matrix<int> testi_fv(gold_v);
    REQUIRE(testi == goldi);
    REQUIRE(testi_fv == goldi);

    fk::matrix<float> testf(gold);
    fk::matrix<float> testf_fv(gold_v);
    REQUIRE(testf == goldf);
    REQUIRE(testf_fv == goldf);

    fk::matrix<double> testd(gold);
    fk::matrix<double> testd_fv(gold_v);
    REQUIRE(testd == goldd);
    REQUIRE(testd_fv == goldd);
  }
  SECTION("converting copy assignment")
  {
    fk::matrix<int> testi(5, 3);
    fk::matrix<int> testi_own(5, 3);
    fk::matrix<int, mem_type::view> testi_v(testi_own);

    testi = gold;
    REQUIRE(testi == goldi);
    testi = fk::matrix<int>(5, 3);
    testi = gold_v;
    REQUIRE(testi == goldi);

    testi_v = gold;
    REQUIRE(testi_v == goldi);

    for (int i = 0; i < testi_own.nrows(); ++i)
    {
      for (int j = 0; j < testi_own.ncols(); ++j)
      {
        testi_own(i, j) = 0;
      }
    }

    testi_v = gold_v;
    REQUIRE(testi_v == goldi);

    fk::matrix<float> testf(5, 3);
    fk::matrix<float> testf_own(5, 3);
    fk::matrix<float, mem_type::view> testf_v(testf_own);
    testf = gold;
    REQUIRE(testf == goldf);
    testf = fk::matrix<float>(5, 3);
    testf = gold_v;
    REQUIRE(testf == goldf);

    testf_v = gold;
    REQUIRE(testf_v == goldf);

    for (int i = 0; i < testf_own.nrows(); ++i)
    {
      for (int j = 0; j < testf_own.ncols(); ++j)
      {
        testf_own(i, j) = 0;
      }
    }

    testf_v = gold_v;
    REQUIRE(testf_v == goldf);

    fk::matrix<double> testd(5, 3);
    fk::matrix<double> testd_own(5, 3);
    fk::matrix<double, mem_type::view> testd_v(testd_own);

    testd = gold;
    REQUIRE(testd == goldd);
    testd = fk::matrix<double>(5, 3);

    testd = gold_v;
    REQUIRE(testd == goldd);

    testd_v = gold;
    REQUIRE(testd_v == goldd);

    for (int i = 0; i < testd_own.nrows(); ++i)
    {
      for (int j = 0; j < testd_own.ncols(); ++j)
      {
        testd_own(i, j) = 0;
      }
    }
    testd_v = gold_v;
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

    // FIXME what is this??
    fk::matrix<TestType> test(
        [](fk::matrix<TestType> in) -> fk::matrix<TestType> {
          return in;
        }(moved));
    // fk::matrix test(std::move(moved));

    fk::matrix<TestType, mem_type::view> test_v(std::move(moved_v));

    REQUIRE(test == gold);
    REQUIRE(test_v == gold);
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
  }
  SECTION("copy from fk::vector")
  {
    // clang-format off
    std::vector<TestType> vstd
      {12, 22, 32,
       13, 23, 33,
       14, 24, 34,
       15, 25, 35,
       16, 26, 36};
    fk::vector<TestType> vfk
      {12, 22, 32,
       13, 23, 33,
       14, 24, 34,
       15, 25, 35,
       16, 26, 36};
    // clang-format on

    // FIXME what is under test here? looks like also
    // testing conversion to fk vect?

    fk::matrix<TestType> teststd(5, 3);
    teststd = fk::vector<TestType>{vstd};

    fk::matrix<TestType> own(5, 3);
    fk::matrix<TestType, mem_type::view> test_v(own);
    test_v = fk::vector<TestType>{vstd};

    REQUIRE(teststd == gold);
    REQUIRE(test_v == gold);

    fk::matrix<TestType> testfk(5, 3);
    own = testfk;

    testfk = vfk;
    test_v = vfk;
    REQUIRE(testfk == gold);
    REQUIRE(test_v == gold);
  }

  SECTION("views constructor")
  {
    // default one
    fk::matrix<TestType> const base(gold);
    fk::matrix<TestType, mem_type::view> const view(base);
    REQUIRE(base == view);

    // ranged
    fk::matrix<TestType, mem_type::view> const view_2(gold, 0, 2, 1, 2);
    fk::matrix<TestType> const gold_partial_2 =
        gold.extract_submatrix(0, 1, 3, 2);
    REQUIRE(view_2 == gold_partial_2);

    fk::matrix<TestType, mem_type::view> const view_3(gold, 1, 1, 0, 2);
    fk::matrix<TestType> const gold_partial_3 =
        gold.extract_submatrix(1, 0, 1, 3);
    REQUIRE(view_3 == gold_partial_3);
  }

  SECTION("views from vector constructor")
  {
    fk::vector<TestType> base{0, 1, 2, 3, 4, 5, 6, 7};

    REQUIRE(base.get_num_views() == 0);
    fk::vector<TestType, mem_type::view> view(base, 1, 7);
    REQUIRE(base.get_num_views() == 1);
    {
      // create 2x3 matrix from last six elems in base
      fk::matrix<TestType, mem_type::view> from_owner(base, 2, 3, 2);
      REQUIRE(base.get_num_views() == 2);
      // create 2x2 matrix from middle of view
      fk::matrix<TestType, mem_type::view> from_view(view, 2, 2, 1);
      REQUIRE(base.get_num_views() == 3);

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
    REQUIRE(base.get_num_views() == 1);
  }

} // end fk::matrix constructors, copy/move

TEMPLATE_TEST_CASE("fk::matrix operators", "[tensors]", double, float, int)
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

  fk::matrix<TestType> const gold_own{
    {12, 22, 32},
    {13, 23, 33},
    {14, 24, 34},
    {15, 25, 35},
    {16, 26, 36},
  }; // clang-format on
  fk::matrix<TestType, mem_type::view> const gold_v(gold_own);

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
    TestType const test   = gold(4, 2);
    TestType const test_v = gold_v(4, 2);
    fk::matrix<TestType, mem_type::view> const gold_v_p(gold, 4, 4, 1, 2);
    TestType const test_v_p = gold_v_p(0, 1);
    REQUIRE(test == 36);
    REQUIRE(test_v == 36);
    REQUIRE(test_v_p == 36);
  }
  SECTION("comparison operator") // this gets used in every REQUIRE
  SECTION("comparison (negated) operator")
  {
    fk::matrix<TestType> test(gold);

    fk::matrix<TestType> own(gold);
    fk::matrix<TestType> test_v(own);

    test(4, 2)   = 333;
    test_v(4, 2) = 333;

    REQUIRE(test != gold);
    REQUIRE(test_v != gold);
  }

  SECTION("less than operator")
  {
    fk::matrix<TestType> const empty;
    fk::matrix<TestType, mem_type::view> const empty_v(empty);

    fk::matrix<TestType> const gold_copy = gold;
    fk::matrix<TestType, mem_type::view> const gold_copy_v(gold_copy);

    fk::matrix<TestType> const gold_prefix{{12, 13, 14}};
    fk::matrix<TestType, mem_type::view> const gold_prefix_v(gold_prefix);

    fk::matrix<TestType> const mismatch{{12, 13, 15}};
    fk::matrix<TestType, mem_type::view> const mismatch_v(mismatch);

    // equal vectors return false
    REQUIRE(!(gold_copy < gold));
    REQUIRE(!(gold_copy < gold_v));
    REQUIRE(!(gold_copy_v < gold));
    REQUIRE(!(gold_copy_v < gold_v));

    // empty range less than non-empty range
    REQUIRE(empty < gold);
    REQUIRE(empty < gold_v);
    REQUIRE(empty_v < gold);
    REQUIRE(empty_v < gold_v);

    // a prefix is less than the complete range
    REQUIRE(gold_prefix < gold);
    REQUIRE(gold_prefix < gold_v);
    REQUIRE(gold_prefix_v < gold);
    REQUIRE(gold_prefix_v < gold_v);

    // otherwise compare on first mismatch
    REQUIRE(gold < mismatch);
    REQUIRE(gold < mismatch_v);
    REQUIRE(gold_v < mismatch);
    REQUIRE(gold_v < mismatch_v);

    // also, empty ranges are equal
    REQUIRE(!(empty < empty));
    REQUIRE(!(empty < empty_v));
    REQUIRE(!(empty_v < empty));
    REQUIRE(!(empty_v < empty_v));
  }

  SECTION("matrix+matrix addition")
  {
    // clang-format off
    fk::matrix<TestType> const in1 {
      {11, 20, 0},
      {12, 21, 0},
      {13, 22, 0},
      {14, 23, 0},
      {15, 24, 0},
    };
    fk::matrix<TestType> const in2 {
      {1, 2, 32},
      {1, 2, 33},
      {1, 2, 34},
      {1, 2, 35},
      {1, 2, 36},
    }; // clang-format on

    fk::matrix<TestType, mem_type::view> const in1_v(in1);
    fk::matrix<TestType, mem_type::view> const in2_v(in2);

    fk::matrix<TestType, mem_type::view> const in1_v_p(in1, 1, 2, 0, 2);
    fk::matrix<TestType, mem_type::view> const in2_v_p(in2, 1, 2, 0, 2);
    fk::matrix<TestType, mem_type::view> const gold_v_p(gold, 1, 2, 0, 2);

    REQUIRE((in1 + in2) == gold);
    REQUIRE((in1_v + in2) == gold);
    REQUIRE((in1 + in2_v) == gold);
    REQUIRE((in1_v + in2_v) == gold);
    REQUIRE((in1_v_p + in2_v_p) == gold_v_p);
  }
  SECTION("matrix-matrix subtraction")
  {
    // clang-format off
    fk::matrix<TestType> const in1 {
      {13, 22, 34},
      {14, 23, 35},
      {15, 24, 36},
      {16, 25, 37},
      {17, 26, 38},
    };
    fk::matrix<TestType> const in2 {
      {1, 0, 2},
      {1, 0, 2},
      {1, 0, 2},
      {1, 0, 2},
      {1, 0, 2}
    }; // clang-format on

    fk::matrix<TestType, mem_type::view> const in1_v(in1);
    fk::matrix<TestType, mem_type::view> const in2_v(in2);

    fk::matrix<TestType, mem_type::view> const in1_v_p(in1, 0, 3, 1, 2);
    fk::matrix<TestType, mem_type::view> const in2_v_p(in2, 0, 3, 1, 2);
    fk::matrix<TestType, mem_type::view> const gold_v_p(gold, 0, 3, 1, 2);

    REQUIRE((in1 - in2) == gold);
    REQUIRE((in1_v - in2) == gold);
    REQUIRE((in1 - in2_v) == gold);
    REQUIRE((in1_v - in2_v) == gold);
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
    fk::matrix<TestType> in_scaled {
      {12, 16, 20},
      {24, 28, 32},
      {36, 40, 44},
      {48, 52, 56},
      {60, 64, 68},
    }; // clang-format on

    fk::matrix<TestType> own(in);
    fk::matrix<TestType, mem_type::view> in_v(own);
    fk::matrix<TestType> own_p(in);
    fk::matrix<TestType, mem_type::view> in_v_p(own_p, 4, 4, 0, 2);
    fk::matrix<TestType, mem_type::view> const in_scaled_v_p(in_scaled, 4, 4, 0,
                                                             2);
    REQUIRE(in * 4 == in_scaled);
    REQUIRE(in_v * 4 == in_scaled);
    REQUIRE(in_v_p * 4 == in_scaled_v_p);
  }
  SECTION("matrix*vector multiplication")
  {
    // clang-format off
    fk::matrix<TestType> const testm{
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {16, 26, 36},
    }; // clang-format on
    fk::matrix<TestType, mem_type::view> const testm_v(testm);
    fk::matrix<TestType, mem_type::view> const testm_v_p(testm, 1, 2, 0, 2);

    fk::vector<TestType> const testv{2, 3, 4};
    fk::vector<TestType> const testv_v(testv);

    fk::vector<TestType> const gold{218, 227, 236, 245, 254};
    REQUIRE((testm * testv) == gold);
    REQUIRE((testm * testv_v) == gold);
    REQUIRE((testm_v * testv) == gold);
    REQUIRE((testm_v * testv_v) == gold);

    fk::vector<TestType> const gold_p = gold.extract(1, 2);
    REQUIRE((testm_v_p * testv_v) == gold_p);
    REQUIRE((testm_v_p * testv) == gold_p);
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
    fk::matrix<TestType> const in1 {
      {3, 4, 5, 6, 7},
      {8, 9, 10, 11, 12},
    };

    fk::matrix<TestType> const in2 {
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {16, 26, 36},
    };
    // clang-format on
    fk::matrix<TestType, mem_type::view> const in1_v(in1);
    fk::matrix<TestType, mem_type::view> const in2_v(in2);

    fk::matrix<TestType, mem_type::view> const in1_v_p(in1, 0, 1, 1, 3);
    fk::matrix<TestType, mem_type::view> const in2_v_p(in2, 0, 2, 1, 2);

    REQUIRE((in1 * in2) == ans);
    REQUIRE((in1 * in2_v) == ans);
    REQUIRE((in1_v * in2) == ans);
    REQUIRE((in1_v * in2_v) == ans);

    REQUIRE((in1_v_p * in2_v_p) == ans_p);
  }
  SECTION("matrix kron product")
  {
    // clang-format off

    fk::matrix<TestType> const A {{1,2,3}};

    fk::matrix<TestType> const B {{2,3},
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

    REQUIRE(A.kron(B) == ans);
    REQUIRE(A.kron(B_v) == ans);
    REQUIRE(A_v.kron(B) == ans);
    REQUIRE(A_v.kron(B_v) == ans);

    // add some larger matrices to test partial views...

    // clang-format off
    fk::matrix<TestType> const A_own{{0, 1, 2, 3, 4},
				     {5, 6, 7, 8, 9},
				     {10, 11, 12, 13}};
    fk::matrix<TestType, mem_type::view> const A_v_p(A_own, 0, 1, 1, 3);
    fk::matrix<TestType> const B_own{{14, 15, 16, 17, 18},
	    			     {19, 20, 21, 22, 23},
				     {24, 25, 26, 27, 28}};
    // clang-format on
    fk::matrix<TestType, mem_type::view> const B_v_p(B_own, 1, 2, 2, 4);
    fk::matrix<TestType> const ans_p = {
        {21, 22, 23, 42, 44, 46, 63, 66, 69},
        {26, 27, 28, 52, 54, 56, 78, 81, 84},
        {126, 132, 138, 147, 154, 161, 168, 176, 184},
        {156, 162, 168, 182, 189, 196, 208, 216, 224}};
    REQUIRE(A_v_p.kron(B_v_p) == ans_p);
    fk::matrix<TestType, mem_type::view> const ans_a_p(ans_p, 0, 1, 0, 8);
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
    if constexpr (std::is_floating_point<TestType>::value)
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
    if constexpr (std::is_floating_point<TestType>::value)
    {
      // clang-format off
    fk::matrix<TestType> const in {
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
      fk::matrix<TestType, mem_type::view> const in_v_p(in_own, 1, 3, 1, 3);
      fk::matrix<TestType, mem_type::view> const in_v(in);
      REQUIRE(in.determinant() == Approx(-0.020200));
      REQUIRE(in_v.determinant() == Approx(-0.020200));
      REQUIRE(in_v_p.determinant() == Approx(-0.020200));

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
    fk::matrix<TestType, mem_type::view> const gold_v_p(gold, 0, 3, 0, 2);
    REQUIRE(gold.nrows() == 5);
    REQUIRE(gold_v.nrows() == 5);
    REQUIRE(gold_v_p.nrows() == 4);
  }
  SECTION("ncols(): the number of columns")
  {
    fk::matrix<TestType, mem_type::view> const gold_v_p(gold, 0, 4, 0, 1);
    REQUIRE(gold.ncols() == 3);
    REQUIRE(gold_v.ncols() == 3);
    REQUIRE(gold_v_p.ncols() == 2);
  }
  SECTION("size(): the number of elements")
  {
    fk::matrix<TestType, mem_type::view> const gold_v_p(gold, 1, 3, 1, 2);
    REQUIRE(gold.size() == 15);
    REQUIRE(gold_v.size() == 15);
    REQUIRE(gold_v_p.size() == 6);
  }
  SECTION("data(): const get address to an element")
  {
    fk::matrix<TestType, mem_type::view> const gold_v_p(gold, 3, 4, 1, 2);
    REQUIRE(*gold.data(4, 2) == 36);
    REQUIRE(*gold_v.data(4, 2) == 36);
    REQUIRE(*gold_v_p.data(1, 1) == 36);
  }
} // end fk::matrix operators

TEMPLATE_TEST_CASE("fk::matrix utilities", "[tensors]", double, float, int)
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

  fk::matrix<TestType, mem_type::view> const gold_v(gold);

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

    fk::matrix<TestType, mem_type::view> const gold_v_p(gold, 1, 3, 0, 2);

    fk::matrix<TestType> own(test);
    fk::matrix<TestType, mem_type::view> test_v(own);

    fk::matrix<TestType> own_p(test);
    fk::matrix<TestType, mem_type::view> test_v_p(own_p, 1, 3, 0, 2);

    fk::vector<TestType> const testv{32, 33, 34, 35, 36};
    fk::vector<TestType, mem_type::view> const testv_v(testv);

    fk::vector<TestType> const testv_p = testv.extract(1, 3);
    fk::vector<TestType, mem_type::view> const testv_v_p(testv_p);

    REQUIRE(test.update_col(2, testv) == gold);
    test = orig;
    REQUIRE(test.update_col(2, testv_v) == gold);

    REQUIRE(test_v.update_col(2, testv) == gold);
    test_v = orig;
    REQUIRE(test_v.update_col(2, testv_v) == gold);

    REQUIRE(test_p.update_col(2, testv_p) == gold_v_p);
    test_p = orig_p;
    REQUIRE(test_p.update_col(2, testv_v_p) == gold_v_p);
  }
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
    fk::matrix<TestType, mem_type::view> const gold_v_p(gold, 2, 4, 1, 2);

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
    fk::matrix<TestType, mem_type::view> const gold_v_p(gold, 3, 4, 0, 2);

    fk::matrix<TestType> own(test);
    fk::matrix<TestType, mem_type::view> test_v(own);

    fk::vector<TestType> const testv{16, 26, 36};
    fk::vector<TestType, mem_type::view> const testv_v(testv);

    REQUIRE(test.update_row(4, testv) == gold);
    test = orig;
    REQUIRE(test.update_row(4, testv_v) == gold);

    REQUIRE(test_v.update_row(4, testv) == gold);
    test_v = orig;
    REQUIRE(test_v.update_row(4, testv_v) == gold);

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
    fk::matrix<TestType, mem_type::view> const gold_v_p(gold, 3, 4, 0, 2);

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

    fk::matrix<TestType> const sub {
      {-13, -23},
      {-14, -24},
      {-15, -25},
    };
    fk::matrix<TestType, mem_type::view> const sub_v(sub);


    fk::matrix<TestType> const after_set {
      {12,  22,  32},
      {13, -13, -23},
      {14, -14, -24},
      {15, -15, -25},
      {0,    0,  35},
    }; // clang-format on
    fk::matrix<TestType, mem_type::view> const after_set_v_p(after_set, 0, 3, 1,
                                                             2);

    REQUIRE(test.set_submatrix(1, 1, sub) == after_set);
    test = orig;
    REQUIRE(test.set_submatrix(1, 1, sub_v) == after_set);
    test = orig;
    REQUIRE(test_v.set_submatrix(1, 1, sub) == after_set);
    test_v = orig;
    REQUIRE(test_v.set_submatrix(1, 1, sub_v) == after_set);
    test_v = orig;
    REQUIRE(test_v_p.set_submatrix(1, 0, sub) == after_set_v_p);
    test_v_p = orig_p;
    REQUIRE(test_v_p.set_submatrix(1, 0, sub_v) == after_set_v_p);
    test_v_p = orig_p;

    // now, test setting a partial view
    fk::matrix<TestType, mem_type::view> const sub_v_p(sub, 0, 1, 0, 1);

    // clang-format off
    fk::matrix<TestType> const after_set_p {
      {12,  22,  32},
      {13, -13, -23},
      {14, -14, -24},
      {15,  25,  35},
      {0,    0,  35},
    }; // clang-format on
    fk::matrix<TestType, mem_type::view> const after_set_p_v(after_set_p, 0, 3,
                                                             1, 2);

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

    fk::matrix<TestType> const own(test);
    fk::matrix<TestType, mem_type::view> const test_v(own);

    fk::matrix<TestType> const own_p(test);
    fk::matrix<TestType, mem_type::view> const test_v_p(own_p, 1, 4, 0, 1);


    fk::matrix<TestType> const sub {
      {13, 23},
      {14, 24},
      {15, 25},
    }; // clang-format on

    REQUIRE(test.extract_submatrix(1, 0, 3, 2) == sub);
    REQUIRE(test_v.extract_submatrix(1, 0, 3, 2) == sub);
    REQUIRE(test_v_p.extract_submatrix(0, 0, 3, 2) == sub);
  }

  SECTION("print out the values")
  {
    // (effectively) redirect cout
    std::streambuf *old_cout_stream_buf = std::cout.rdbuf();
    std::ostringstream test_str;
    std::cout.rdbuf(test_str.rdbuf());
    // generate the output (into test_str)
    gold.print("golden matrix");
    // restore cout destination
    std::cout.rdbuf(old_cout_stream_buf);

    std::string golden_string;
    if constexpr (std::is_floating_point<TestType>::value)
    {
      golden_string =
          "golden matrix(owner, outstanding views == 1)\n  1.2000e+01  "
          "2.2000e+01  3.2000e+01\n  "
          "1.3000e+01  "
          "2.3000e+01  3.3000e+01\n  1.4000e+01  2.4000e+01  3.4000e+01\n  "
          "1.5000e+01  2.5000e+01  3.5000e+01\n  1.6000e+01  2.6000e+01  "
          "3.6000e+01\n";
    }
    else
    {
      golden_string = "golden matrix(owner, outstanding views == 1)\n12 22 32 "
                      "\n13 23 33 \n14 24 34 \n15 25 "
                      "35 \n16 26 36 \n";
    }
    REQUIRE(test_str.str() == golden_string);

    old_cout_stream_buf = std::cout.rdbuf();
    std::ostringstream test_str_v;
    std::cout.rdbuf(test_str_v.rdbuf());
    gold_v.print("golden matrix");
    std::cout.rdbuf(old_cout_stream_buf);

    if constexpr (std::is_floating_point<TestType>::value)
    {
      golden_string =
          "golden matrix(view, stride == 5)\n  1.2000e+01  2.2000e+01  "
          "3.2000e+01\n  "
          "1.3000e+01  "
          "2.3000e+01  3.3000e+01\n  1.4000e+01  2.4000e+01  3.4000e+01\n  "
          "1.5000e+01  2.5000e+01  3.5000e+01\n  1.6000e+01  2.6000e+01  "
          "3.6000e+01\n";
    }
    else
    {
      golden_string = "golden matrix(view, stride == 5)\n12 22 32 \n13 23 33 "
                      "\n14 24 34 \n15 25 "
                      "35 \n16 26 36 \n";
    }

    REQUIRE(test_str_v.str() == golden_string);

    fk::matrix<TestType, mem_type::view> const gold_v_p(gold, 1, 3, 0, 2);
    old_cout_stream_buf = std::cout.rdbuf();
    std::ostringstream test_str_v_p;
    std::cout.rdbuf(test_str_v_p.rdbuf());
    gold_v_p.print("golden matrix");
    std::cout.rdbuf(old_cout_stream_buf);

    if constexpr (std::is_floating_point<TestType>::value)
    {
      golden_string = "golden matrix(view, stride == 5)\n  1.3000e+01  "
                      "2.3000e+01  3.3000e+01\n  "
                      "1.4000e+01  2.4000e+01  3.4000e+01\n  "
                      "1.5000e+01  2.5000e+01  3.5000e+01\n";
    }
    else
    {
      golden_string =
          "golden matrix(view, stride == 5)\n13 23 33 \n14 24 34 \n15 25 "
          "35 \n";
    }

    REQUIRE(test_str_v_p.str() == golden_string);
  }

  SECTION("dump to octave")
  {
    gold.dump_to_octave("test_out.dat");
    gold_v.dump_to_octave("test_out_v.dat");
    fk::matrix<TestType, mem_type::view> const gold_v_p(gold, 1, 3, 0, 1);
    gold_v_p.dump_to_octave("test_out_v_p.dat");

    std::ifstream data_stream("test_out.dat");
    std::ifstream data_stream_v("test_out_v.dat");
    std::ifstream data_stream_v_p("test_out_v_p.dat");

    std::string test_string((std::istreambuf_iterator<char>(data_stream)),
                            std::istreambuf_iterator<char>());
    std::string test_string_v((std::istreambuf_iterator<char>(data_stream_v)),
                              std::istreambuf_iterator<char>());
    std::string test_string_v_p(
        (std::istreambuf_iterator<char>(data_stream_v_p)),
        std::istreambuf_iterator<char>());

    std::remove("test_out.dat");
    std::remove("test_out_v.dat");
    std::remove("test_out_v_p.dat");

    std::string golden_string;

    if constexpr (std::is_floating_point<TestType>::value)
    {
      golden_string =
          "1.200000000000e+01 2.200000000000e+01 3.200000000000e+01 \n"
          "1.300000000000e+01 2.300000000000e+01 3.300000000000e+01 \n"
          "1.400000000000e+01 2.400000000000e+01 3.400000000000e+01 \n"
          "1.500000000000e+01 2.500000000000e+01 3.500000000000e+01 \n"
          "1.600000000000e+01 2.600000000000e+01 3.600000000000e+01 \n";
    }
    else
    {
      golden_string = "12 22 32 \n"
                      "13 23 33 \n"
                      "14 24 34 \n"
                      "15 25 35 \n"
                      "16 26 36 \n";
    }

    REQUIRE(test_string == golden_string);
    REQUIRE(test_string_v == golden_string);

    if constexpr (std::is_floating_point<TestType>::value)
    {
      golden_string = "1.300000000000e+01 2.300000000000e+01 \n"
                      "1.400000000000e+01 2.400000000000e+01 \n"
                      "1.500000000000e+01 2.500000000000e+01 \n";
    }
    else
    {
      golden_string = "13 23 \n"
                      "14 24 \n"
                      "15 25 \n";
    }

    REQUIRE(test_string_v_p == golden_string);
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
    fk::matrix<TestType> const test {
     {1, 2, 3, 4},
     {5, 6, 11, 8},
    }; // clang-format on
    fk::matrix<TestType, mem_type::view> const test_v(test);
    fk::matrix<TestType, mem_type::view> const test_v_p(test, 0, 1, 1, 2);

    TestType const max = 11;
    REQUIRE(*std::max_element(test.begin(), test.end()) == max);
    REQUIRE(*std::max_element(test_v.begin(), test_v.end()) == max);
    REQUIRE(*std::max_element(test_v_p.begin(), test_v_p.end()) == max);
  }

  SECTION("matrix sum of elements")
  {
    // clang-format off
    fk::matrix<TestType> const test {
     {1, 2, 3, 4},
     {5, 6, 7, 8},
    }; // clang-format on
    fk::matrix<TestType, mem_type::view> const test_v(test);
    fk::matrix<TestType, mem_type::view> const test_v_p(test, 0, 0, 1, 3);

    TestType const sum   = 36;
    TestType const sum_p = 9;
    REQUIRE(std::accumulate(test.begin(), test.end(), 0) == sum);
    REQUIRE(std::accumulate(test_v.begin(), test_v.end(), 0) == sum);
    REQUIRE(std::accumulate(test_v_p.begin(), test_v_p.end(), 0) == sum_p);
  }

  SECTION("matrix ref counting")
  {
    // on construction, matrices have 0 views
    fk::matrix<TestType> const test;
    assert(test.get_num_views() == 0);
    fk::matrix<TestType> const test_init({{1}});
    assert(test_init.get_num_views() == 0);
    fk::matrix<TestType> const test_sz(1, 1);
    assert(test_sz.get_num_views() == 0);
    fk::matrix<TestType> const test_conv(fk::matrix<int>(1, 1));
    assert(test_conv.get_num_views() == 0);
    fk::matrix<TestType> const test_copy(test);
    assert(test_copy.get_num_views() == 0);

    // creating views increments view count
    fk::matrix<TestType, mem_type::view> const test_view(test);
    assert(test.get_num_views() == 1);
    fk::matrix<TestType, mem_type::view> const test_view_2(test);
    assert(test.get_num_views() == 2);

    // copies have a fresh view count
    fk::matrix<TestType> const test_cp(test);
    assert(test_cp.get_num_views() == 0);
    assert(test.get_num_views() == 2);

    // test that view count gets decremented when views go out of scope
    {
      fk::matrix<TestType, mem_type::view> test_view_3(test);
      assert(test.get_num_views() == 3);
    }
    assert(test.get_num_views() == 2);
  }

} // end fk::matrix utilities

TEMPLATE_TEST_CASE("wrapped free BLAS", "[tensors]", float, double, int)
{
  fk::vector<TestType> const gold = {2, 3, 4, 5, 6};
  SECTION("vector scale and accumulate (axpy)")
  {
    TestType const scale = 2.0;

    fk::vector<TestType> test(gold);
    fk::vector<TestType> test_own(gold);
    fk::vector<TestType, mem_type::view> test_view(test_own);

    fk::vector<TestType> rhs{7, 8, 9, 10, 11};
    fk::vector<TestType> rhs_own(rhs);
    fk::vector<TestType, mem_type::view> rhs_view(rhs_own);

    fk::vector<TestType> const ans = {16, 19, 22, 25, 28};

    REQUIRE(axpy(scale, rhs, test) == ans);
    test = gold;
    REQUIRE(axpy(scale, rhs_view, test) == ans);

    REQUIRE(axpy(scale, rhs, test_view) == ans);
    REQUIRE(test_own == ans);
    test_view = gold;
    REQUIRE(axpy(scale, rhs_view, test_view) == ans);
    REQUIRE(test_own == ans);
  }

  SECTION("vector copy (copy)")
  {
    fk::vector<TestType> test(gold.size());
    fk::vector<TestType> test_own(gold.size());
    fk::vector<TestType, mem_type::view> test_view(test_own);

    fk::vector<TestType, mem_type::view> const gold_view(gold);

    REQUIRE(copy(gold, test) == gold);
    test.scale(0);
    REQUIRE(copy(gold_view, test) == gold);

    REQUIRE(copy(gold, test_view) == gold);
    REQUIRE(test_own == gold);
    test_own.scale(0);
    REQUIRE(copy(gold_view, test_view) == gold);
    REQUIRE(test_own == gold);
  }

  SECTION("vector scale (scal)")
  {
    TestType const x = 2.0;
    fk::vector<TestType> test(gold);
    fk::vector<TestType> test_own(gold);
    fk::vector<TestType, mem_type::view> test_view(test_own);

    fk::vector<TestType> const ans = {4, 6, 8, 10, 12};

    REQUIRE(scal(x, test) == ans);
    REQUIRE(scal(x, test_view) == ans);
    REQUIRE(test_own == ans);

    test     = gold;
    test_own = gold;

    TestType const x2 = 0.0;
    fk::vector<TestType> const zeros(gold.size());

    REQUIRE(scal(x2, test) == zeros);
    REQUIRE(scal(x2, test_view) == zeros);
    REQUIRE(test_own == zeros);
  }
}
