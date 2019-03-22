
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
    fk::vector<TestType> test;
    // fk::vector<TestType, mem_type::view> test_v; // disabled
    REQUIRE(test.size() == 0);
  }
  SECTION("give me some size, initialized to zero")
  {
    fk::vector<TestType> test(5);
    // fk::vector<TestType, mem_type::view> test_v(5); //disabled
    fk::vector<TestType> zeros{0, 0, 0, 0, 0};
    REQUIRE(test == zeros);
  }
  SECTION("constructor from list initialization")
  {
    fk::vector<TestType> test{2, 3, 4, 5, 6};
    // fk::vector<TestType, mem_type::view> test_v{2, 3, 4, 5, 6}; // disabled
    REQUIRE(test == gold);
  }
  SECTION("construct from a std::vector")
  {
    std::vector<TestType> v{2, 3, 4, 5, 6};
    fk::vector<TestType> test(v);
    // fk::vector<TestType, mem_type::view> test_v(v); // disabled
    REQUIRE(test == gold);
  }
  SECTION("construct from an fk::matrix")
  {
    fk::matrix<TestType> mat{{2}, {3}, {4}, {5}, {6}};
    fk::vector<TestType> test(mat);
    // fk::vector<TestType, mem_type::view> test_v(mat); // disabled
    REQUIRE(test == gold);

    fk::vector<TestType> gold_2 = {1, 2, 3, 4, 5, 6};
    fk::matrix<TestType> mat_2{{1, 3, 5}, {2, 4, 6}};
    fk::vector<TestType> test_2(mat_2);
    // fk::vector<TestType, mem_type::view> test_2_v(mat_2); // disabled
    REQUIRE(test_2 == gold_2);
  }
  SECTION("view construction from owner with size")
  {
    fk::vector<TestType> gold = {2, 3, 4, 5, 6};
    fk::vector<TestType, mem_type::view> test0(gold, 0, gold.size() - 1);
    REQUIRE(test0 == gold);
    fk::vector<TestType> gold2 = {3, 4, 5};
    fk::vector<TestType, mem_type::view> test1(gold, 1, 3);
    REQUIRE(test1 == gold2);
  }
  SECTION("copy construction")
  {
    fk::vector<TestType> test(gold);
    // fk::vector<TestType, mem_type::view> test_v(gold); // ill-defined
    REQUIRE(test == gold);
  }
  SECTION("copy assignment")
  {
    fk::vector<TestType> test(5);
    fk::vector<TestType, mem_type::view> test_v(test);
    test = gold;
    // test_v = gold; // ill-defined; covered below by converting ctor
    REQUIRE(test == gold);
  }
  SECTION("converting construction (from owners)")
  {
    fk::vector<int> testi(gold);
    fk::vector<int, mem_type::view> testi_v(gold);
    REQUIRE(testi == goldi);
    REQUIRE(testi_v == goldi);
    fk::vector<float> testf(gold);
    fk::vector<float, mem_type::view> testf_v(gold);
    REQUIRE(testf == goldf);
    REQUIRE(testf_v == goldf);
    fk::vector<double> testd(gold);
    fk::vector<double, mem_type::view> testd_v(gold);
    REQUIRE(testd == goldd);
    REQUIRE(testd_v == goldd);
  }
  SECTION("converting construction (from views)")
  {
    fk::vector<TestType, mem_type::view> const gold_v(gold);
    fk::vector<int> testi(gold_v);
    fk::vector<int, mem_type::view> testi_v(gold_v);
    REQUIRE(testi == goldi);
    REQUIRE(testi_v == goldi);
    fk::vector<float> testf(gold_v);
    fk::vector<float, mem_type::view> testf_v(gold_v);
    REQUIRE(testf == goldf);
    REQUIRE(testf_v == goldf);
    fk::vector<double> testd(gold_v);
    fk::vector<double, mem_type::view> testd_v(gold_v);
    REQUIRE(testd == goldd);
    REQUIRE(testd_v == goldd);
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
    fk::vector<TestType, mem_type::view> moved_v(moved);
    fk::vector<TestType> test(std::move(moved));
    REQUIRE(test == gold);
    // FIXME we need to pay attention here; is this what we want?
    fk::vector<TestType, mem_type::view> test_v(std::move(moved_v));
    REQUIRE(test_v == gold);
  }
  SECTION("move assignment")
  {
    fk::vector<TestType> moved{2, 3, 4, 5, 6};
    fk::vector<TestType, mem_type::view> moved_v(moved);
    fk::vector<TestType> test(5);
    test = std::move(moved);
    REQUIRE(test == gold);
    // FIXME we need to pay attention here; is this what we want?
    fk::vector<TestType, mem_type::view> test_v(moved_v);
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
    fk::vector<TestType> gold_v{2, 3, 4, 5, 6};
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
    fk::vector<TestType, mem_type::view> test_v(test);
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
    fk::vector<TestType, mem_type::view> test_v(gold);
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
    fk::vector<TestType> const gold_copy                   = gold;
    fk::vector<TestType, mem_type::view> const gold_copy_v = gold;

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
    fk::vector<TestType> const test_vect{2, 3, 4, 5, 6};
    fk::vector<TestType, mem_type::view> const test_vect_v(test_vect);
    fk::vector<TestType> const gold{290, 490, 690};

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
} // end fk::vector operators

TEMPLATE_TEST_CASE("fk::vector utilities", "[tensors]", double, float, int)
{
  fk::vector<TestType> const gold{2, 3, 4, 5, 6};
  fk::vector<TestType, mem_type::view> const gold_v(gold);
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
          "golden vector(owner, ref_count = 0)\n  2.0000e+00  3.0000e+00  "
          "4.0000e+00  5.0000e+00  6.0000e+00\n";
      golden_string_v = "golden vector(view)\n  2.0000e+00  3.0000e+00  "
                        "4.0000e+00  5.0000e+00  6.0000e+00\n";
    }
    else
    {
      golden_string = "golden vector(owner, ref_count = 0)\n2 3 "
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
    fk::vector<TestType, mem_type::view> test_reduced_v(test_reduced);

    fk::vector<TestType> const gold_enlarged{2, 3, 4, 0, 0};

    fk::vector<TestType> test_enlarged{2, 3, 4};
    fk::vector<TestType, mem_type::view> test_enlarged_v(test_enlarged);

    test_reduced.resize(gold.size());
    test_enlarged.resize(gold.size());
    test_reduced_v.resize(gold.size());
    test_enlarged_v.resize(gold.size());

    REQUIRE(test_reduced == gold);
    REQUIRE(test_enlarged == gold_enlarged);

    REQUIRE(test_reduced_v == gold);
    REQUIRE(test_enlarged_v == gold_enlarged);
  }

  SECTION("vector concatenation")
  {
    fk::vector<TestType> test_left = {2, 3, 4};
    fk::vector<TestType, mem_type::view> test_left_v(test_left);
    fk::vector<TestType> const test_right = {5, 6};
    fk::vector<TestType, mem_type::view> const test_right_v(test_right);

    REQUIRE(test_left.concat(test_right) == gold);
    REQUIRE(test_left_v.concat(test_right) == gold);

    test_left.resize(3)   = fk::vector<TestType>({2, 3, 4});
    test_left_v.resize(3) = test_left;

    REQUIRE(test_left.concat(test_right_v) == gold);
    REQUIRE(test_left_v.concat(test_right_v) == gold);

    fk::vector<TestType> empty;
    fk::vector<TestType, mem_type::view> empty_v(empty);
    fk::vector<TestType> gold_copy = gold;
    fk::vector<TestType, mem_type::view> gold_copy_v(gold_copy);

    REQUIRE(empty.concat(gold) == gold);
    empty.resize(0);
    REQUIRE(empty.concat(gold_v) == gold);
    empty.resize(0);
    REQUIRE(empty_v.concat(gold) == gold);
    empty_v.resize(0);
    REQUIRE(empty_v.concat(gold_v) == gold);
    empty_v.resize(0);

    REQUIRE(gold_copy.concat(empty) == gold);
    gold_copy = gold;
    REQUIRE(gold_copy.concat(empty_v) == gold);
    gold_copy = gold;
    REQUIRE(gold_copy_v.concat(empty) == gold);
    gold_copy_v = gold_copy;
    REQUIRE(gold_copy_v.concat(empty_v) == gold);
  }
  SECTION("vector set")
  {
    fk::vector<TestType> vector(5);
    fk::vector<TestType, mem_type::view> vector_v(vector);

    fk::vector<TestType> const empty;
    fk::vector<TestType> const begin  = {2, 3};
    fk::vector<TestType> const middle = {3, 4, 5};
    fk::vector<TestType> const end    = {6};

    fk::vector<TestType, mem_type::view> const empty_v(empty);
    fk::vector<TestType, mem_type::view> const begin_v(begin);
    fk::vector<TestType, mem_type::view> const middle_v(middle);
    fk::vector<TestType, mem_type::view> const end_v(end);

    REQUIRE(vector.set(0, begin).set(0, empty).set(1, middle).set(4, end) ==
            gold);
    vector = fk::vector<TestType>(5);
    REQUIRE(
        vector.set(0, begin_v).set(0, empty_v).set(1, middle_v).set(4, end) ==
        gold);
    vector = fk::vector<TestType>(5);
    REQUIRE(vector_v.set(0, begin).set(0, empty).set(1, middle).set(4, end) ==
            gold);
    vector_v = vector;
    REQUIRE(vector_v.set(0, begin_v)
                .set(0, empty_v)
                .set(1, middle_v)
                .set(4, end_v) == gold);
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
    fk::vector<TestType, mem_type::view> test_v(test);
    fk::vector<TestType> const after{0, 2, 3, 4};
    std::transform(test.begin(), test.end(), test.begin(),
                   std::bind1st(std::plus<TestType>(), 1));
    std::transform(test_v.begin(), test_v.end(), test_v.begin(),
                   std::bind1st(std::plus<TestType>(), 1));
    REQUIRE(test == after);
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

  SECTION("default constructor")
  {
    fk::matrix<TestType> test;
    REQUIRE(test.size() == 0);
  }
  SECTION("give me some size, initialized to zero")
  {
    fk::matrix<TestType> test(5, 3);
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
    REQUIRE(test == gold);
  }
  SECTION("copy construction")
  {
    fk::matrix<TestType> test(gold);
    REQUIRE(test == gold);
  }
  SECTION("copy assignment")
  {
    fk::matrix<TestType> test(5, 3);
    test = gold;
    REQUIRE(test == gold);
  }
  SECTION("converting copy construction")
  {
    fk::matrix<int> testi(gold);
    REQUIRE(testi == goldi);
    fk::matrix<float> testf(gold);
    REQUIRE(testf == goldf);
    fk::matrix<double> testd(gold);
    REQUIRE(testd == goldd);
  }
  SECTION("converting copy assignment")
  {
    fk::matrix<int> testi(5, 3);
    testi = gold;
    REQUIRE(testi == goldi);
    fk::matrix<float> testf(5, 3);
    testf = gold;
    REQUIRE(testf == goldf);
    fk::matrix<double> testd(5, 3);
    testd = gold;
    REQUIRE(testd == goldd);
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
    fk::matrix<TestType> test(
        [](fk::matrix<TestType> in) -> fk::matrix<TestType> {
          return in;
        }(moved));
    // fk::matrix test(std::move(moved));
    REQUIRE(test == gold);
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
    fk::matrix<TestType> test(5, 3);
    test = std::move(moved);
    REQUIRE(test == gold);
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
    fk::matrix<TestType> teststd(5, 3);
    teststd = fk::vector<TestType>{vstd};
    REQUIRE(teststd == gold);
    fk::matrix<TestType> testfk(5, 3);
    testfk = vfk;
    REQUIRE(testfk == gold);
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
  }; // clang-format on

  SECTION("subscript operator (modifying)")
  {
    fk::matrix<TestType> test(5, 3);
    // clang-format off
    test(0,0) = 12;  test(0,1) = 22;  test(0,2) = 32;
    test(1,0) = 13;  test(1,1) = 23;  test(1,2) = 33;
    test(2,0) = 14;  test(2,1) = 24;  test(2,2) = 34;
    test(3,0) = 15;  test(3,1) = 25;  test(3,2) = 35;
    test(4,0) = 16;  test(4,1) = 26;  test(4,2) = 36;
    // clang-format on
    REQUIRE(test == gold);
    TestType val = test(4, 2);
    REQUIRE(val == 36);
  }
  SECTION("subscript operator (const)")
  {
    TestType test = gold(4, 2);
    REQUIRE(test == 36);
  }
  SECTION("comparison operator") // this gets used in every REQUIRE
  SECTION("comparison (negated) operator")
  {
    fk::matrix<TestType> test(gold);
    test(4, 2) = 333;
    REQUIRE(test != gold);
  }

  SECTION("less than operator")
  {
    fk::matrix<TestType> const empty;
    fk::matrix<TestType> const gold_copy = gold;
    fk::matrix<TestType> const gold_prefix{{12, 13, 14}};
    fk::matrix<TestType> const mismatch{{12, 13, 15}};
    // equal vectors return false
    REQUIRE(!(gold_copy < gold));
    // empty range less than non-empty range
    REQUIRE(empty < gold);
    // a prefix is less than the complete range
    REQUIRE(gold_prefix < gold);
    // otherwise compare on first mismatch
    REQUIRE(gold < mismatch);
    // also, empty ranges are equal
    REQUIRE(!(empty < empty));
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
    REQUIRE((in1 + in2) == gold);
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
    REQUIRE((in1 - in2) == gold);
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
    REQUIRE(in * 4 == in_scaled);
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
    fk::vector<TestType> const testv{2, 3, 4};
    fk::vector<TestType> const gold{218, 227, 236, 245, 254};
    REQUIRE((testm * testv) == gold);
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
    fk::matrix<TestType> test = in1 * in2;
    REQUIRE(test == ans);
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
    REQUIRE(A.kron(B) == ans);
  }
  SECTION("matrix inverse")
  {
    if constexpr (std::is_floating_point<TestType>::value)
    {
      // (square slices of) our golden matrix is singular, so here's a
      // well conditioned one
      fk::matrix<TestType> test{{0.767135868133925, -0.641484652834663},
                                {0.641484652834663, 0.767135868133926}};
      fk::matrix<TestType> test_copy(test);
      test_copy.invert();

      // A * inv(A) == I
      REQUIRE(test * test_copy == eye<TestType>(2));

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
    fk::matrix<double> in {
      {12.130, 14.150, 1.00},
      {13.140, 13.150, 1.00},
      {14.150, 12.130, 1.00},
    }; // clang-format on
      REQUIRE(in.determinant() == Approx(-0.020200));
      // we haven't implemented a determinant routine for integral types; as
      // with inversion, code won't compile if this routine is invoked on a
      // matrix of integers
    }
    else
    {
      REQUIRE(true);
    }
  }
  SECTION("nrows(): the number of rows") { REQUIRE(gold.nrows() == 5); }
  SECTION("ncols(): the number of columns") { REQUIRE(gold.ncols() == 3); }
  SECTION("size(): the number of elements") { REQUIRE(gold.size() == 15); }
  SECTION("data(): const get address to an element")
  {
    REQUIRE(*gold.data(4, 2) == 36);
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
    fk::vector<TestType> testv{32, 33, 34, 35, 36};
    REQUIRE(test.update_col(2, testv) == gold);
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
    std::vector<TestType> testv{32, 33, 34, 35, 36};
    REQUIRE(test.update_col(2, testv) == gold);
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
    fk::vector<TestType> testv{16, 26, 36};
    REQUIRE(test.update_row(4, testv) == gold);
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
    std::vector<TestType> testv{16, 26, 36};
    REQUIRE(test.update_row(4, testv) == gold);
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
      {0, 0, 35},
    }; 
    fk::matrix<TestType> sub {
      {-13, -23},
      {-14, -24},
      {-15, -25},
    };
    fk::matrix<TestType> after_set {
      {12, 22, 32},
      {13, -13, -23},
      {14, -14, -24},
      {15, -15, -25},
      {0, 0, 35},
    }; // clang-format on

    REQUIRE(test.set_submatrix(1, 1, sub) == after_set);
  }

  SECTION("matrix extract submatrix(row, col, nrows, ncols")
  {
    // clang-format off
    fk::matrix<TestType> test {
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {0, 0, 35},
    }; 
    fk::matrix<TestType> sub {
      {13, 23},
      {14, 24},
      {15, 25},
    }; // clang-format on

    REQUIRE(test.extract_submatrix(1, 0, 3, 2) == sub);
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
          "golden matrix\n  1.2000e+01  2.2000e+01  3.2000e+01\n  1.3000e+01  "
          "2.3000e+01  3.3000e+01\n  1.4000e+01  2.4000e+01  3.4000e+01\n  "
          "1.5000e+01  2.5000e+01  3.5000e+01\n  1.6000e+01  2.6000e+01  "
          "3.6000e+01\n";
    }
    else
    {
      golden_string = "golden matrix\n12 22 32 \n13 23 33 \n14 24 34 \n15 25 "
                      "35 \n16 26 36 \n";
    }
    REQUIRE(test_str.str() == golden_string);
  }
  SECTION("dump to octave")
  {
    gold.dump_to_octave("test_out.dat");
    std::ifstream data_stream("test_out.dat");
    std::string test_string((std::istreambuf_iterator<char>(data_stream)),
                            std::istreambuf_iterator<char>());
    // std::remove("test_out.dat");
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
  }

  SECTION("matrix transform")
  {
    // clang-format off
  fk::matrix<TestType> test {
   {0, 1, 2, 3},
   {4, 5, 6, 7},
  };
  fk::matrix<TestType> after {
   {1, 2, 3, 4},
   {5, 6, 7, 8},
  }; // clang-format on 
  std::transform(test.begin(), test.end(), test.begin(), std::bind1st(std::plus<TestType>(), 1));
  REQUIRE(test == after);
  }

  SECTION("matrix maximum element") {
  // clang-format off
  fk::matrix<TestType> test {
   {1, 2, 3, 4},
   {5, 6, 11, 8},
  }; // clang-format on
    TestType max = 11;

    REQUIRE(*std::max_element(test.begin(), test.end()) == max);
  }

  SECTION("matrix sum of elements")
  {
    // clang-format off
  fk::matrix<TestType> test {
   {1, 2, 3, 4},
   {5, 6, 7, 8},
  }; // clang-format on
    TestType max = 36;

    REQUIRE(std::accumulate(test.begin(), test.end(), 0) == max);
  }

} // end fk::matrix utilities
