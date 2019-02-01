
#include "tensors.hpp"
#include "tests_general.hpp"
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>

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
    REQUIRE(test.size() == 0);
  }
  SECTION("give me some size, initialized to zero")
  {
    fk::vector<TestType> test(5);
    fk::vector<TestType> zeros{0, 0, 0, 0, 0};
    REQUIRE(test == zeros);
  }
  SECTION("constructor from list initialization")
  {
    fk::vector<TestType> test{2, 3, 4, 5, 6};
    REQUIRE(test == gold);
  }
  SECTION("construct from a std::vector")
  {
    std::vector<TestType> v{2, 3, 4, 5, 6};
    fk::vector<TestType> test(v);
    REQUIRE(test == gold);
  }
  SECTION("construct from an fk::matrix")
  {
    fk::matrix<TestType> mat{{2}, {3}, {4}, {5}, {6}};
    fk::vector<TestType> test(mat);
    REQUIRE(test == gold);

    fk::vector<TestType> gold_2 = {1, 2, 3, 4, 5, 6};
    fk::matrix<TestType> mat_2{{1, 3, 5}, {2, 4, 6}};
    fk::vector<TestType> test_2(mat_2);
    REQUIRE(test_2 == gold_2);
  }
  SECTION("copy construction")
  {
    fk::vector<TestType> test(gold);
    REQUIRE(test == gold);
  }
  SECTION("copy assignment")
  {
    fk::vector<TestType> test(5);
    test = gold;
    REQUIRE(test == gold);
  }
  SECTION("converting copy construction")
  {
    fk::vector<int> testi(gold);
    REQUIRE(testi == goldi);
    fk::vector<float> testf(gold);
    REQUIRE(testf == goldf);
    fk::vector<double> testd(gold);
    REQUIRE(testd == goldd);
  }
  SECTION("converting copy assignment")
  {
    fk::vector<int> testi(5);
    testi = gold;
    REQUIRE(testi == goldi);
    fk::vector<float> testf(5);
    testf = gold;
    REQUIRE(testf == goldf);
    fk::vector<double> testd(5);
    testd = gold;
    REQUIRE(testd == goldd);
  }
  SECTION("move construction")
  {
    fk::vector<TestType> moved{2, 3, 4, 5, 6};
    fk::vector<TestType> test(std::move(moved));
    REQUIRE(test == gold);
  }
  SECTION("move assignment")
  {
    fk::vector<TestType> moved{2, 3, 4, 5, 6};
    fk::vector<TestType> test(5);
    test = std::move(moved);
    REQUIRE(test == gold);
  }
  SECTION("copy from std::vector")
  {
    std::vector<TestType> v{2, 3, 4, 5, 6};
    fk::vector<TestType> test(5);
    test = v;
    REQUIRE(test == gold);
  }
  SECTION("copy into std::vector")
  {
    std::vector<TestType> goldv{2, 3, 4, 5, 6};
    std::vector<TestType> testv;
    testv = gold.to_std();
    compare_vectors(testv, goldv);
  }
} // end fk::vector constructors, copy/move

TEMPLATE_TEST_CASE("fk::vector operators", "[tensors]", double, float, int)
{
  fk::vector<TestType> const gold{2, 3, 4, 5, 6};
  SECTION("subscript operator (modifying)")
  {
    fk::vector<TestType> test(5);
    // clang-format off
    test(0) = 2; test(1) = 3; test(2) = 4; test(3) = 5; test(4) = 6;
    // clang-format on
    REQUIRE(test == gold);
    TestType val = test(4);
    REQUIRE(val == 6);
  }
  SECTION("subscript operator (const)") { REQUIRE(gold(4) == 6); }
  SECTION("comparison operator") // this gets used in every REQUIRE
  SECTION("comparison (negated) operator")
  {
    fk::vector<TestType> test(gold);
    test(4) = 333;
    REQUIRE(test != gold);
  }
  SECTION("less than operator")
  {
    fk::vector<TestType> const empty;
    fk::vector<TestType> const gold_copy = gold;
    fk::vector<TestType> const gold_prefix{1, 2, 3, 4};
    fk::vector<TestType> const mismatch{2, 3, 5, 5, 6};
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

  SECTION("addition operator")
  {
    fk::vector<TestType> const in1{1, 1, 1, 1, 1};
    fk::vector<TestType> const in2{1, 2, 3, 4, 5};
    REQUIRE((in1 + in2) == gold);
  }
  SECTION("subtraction operator")
  {
    fk::vector<TestType> const in1{3, 4, 5, 6, 7};
    fk::vector<TestType> const in2{1, 1, 1, 1, 1};
    REQUIRE((in1 - in2) == gold);
  }
  SECTION("vector*vector operator") { REQUIRE((gold * gold) == 90); }
  SECTION("vector*matrix operator")
  {
    // clang-format off
    fk::matrix<TestType> const testm{
      {12, 22, 32},
      {13, 23, 33},
      {14, 24, 34},
      {15, 25, 35},
      {16, 26, 36},
    }; // clang-format on
    fk::vector<TestType> const testv{2, 3, 4, 5, 6};
    fk::vector<TestType> const gold{290, 490, 690};
    REQUIRE((testv * testm) == gold);
  }
  SECTION("vector*scalar operator")
  {
    TestType scale = static_cast<TestType>(-2);
    fk::vector<TestType> const gold_scaled{-4, -6, -8, -10, -12};
    REQUIRE((gold * scale) == gold_scaled);
  }
  SECTION("vector kron product")
  {
    fk::vector<TestType> identity{1};
    REQUIRE(identity.kron(gold) == gold);

    fk::vector<TestType> repeat{1, 1};
    fk::vector<TestType> gold_copy = gold;
    REQUIRE(repeat.kron(gold) == gold_copy.concat(gold));

    fk::vector<TestType> zeros(gold.size());
    fk::vector<TestType> alternate{1, 0, 2, 0};
    fk::vector<TestType> ans;
    ans.concat(gold).concat(zeros).concat(gold * 2).concat(zeros);
    REQUIRE(ans == alternate.kron(gold));
  }
} // end fk::vector operators

TEMPLATE_TEST_CASE("fk::vector utilities", "[tensors]", double, float, int)
{
  fk::vector<TestType> const gold{2, 3, 4, 5, 6};
  SECTION("size(): the number of elements") { REQUIRE(gold.size() == 5); }
  SECTION("data(): const addr to element") { REQUIRE(*gold.data(4) == 6); }
  SECTION("print out the values")
  {
    // (effectively) redirect cout
    std::streambuf *old_cout_stream_buf = std::cout.rdbuf();
    std::ostringstream test_str;
    std::cout.rdbuf(test_str.rdbuf());
    // generate the output (into test_str)
    gold.print("golden vector");
    // restore cout destination
    std::cout.rdbuf(old_cout_stream_buf);
    std::string golden_string;
    if constexpr (std::is_floating_point<TestType>::value)
    {
      golden_string = "golden vector\n  2.0000e+00  3.0000e+00  "
                      "4.0000e+00  5.0000e+00  6.0000e+00\n";
    }
    else
    {
      golden_string = "golden vector\n2 3 "
                      "4 5 6 \n";
    }
    REQUIRE(test_str.str() == golden_string);
  }
  SECTION("dump to octave")
  {
    gold.dump_to_octave("test_out.dat");
    std::ifstream data_stream("test_out.dat");
    std::string test_string((std::istreambuf_iterator<char>(data_stream)),
                            std::istreambuf_iterator<char>());
    std::remove("test_out.dat");
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
  }
  SECTION("vector resize")
  {
    fk::vector<TestType> test_reduced{2, 3, 4, 5, 6, 7, 8};
    fk::vector<TestType> const gold_enlarged{2, 3, 4, 0, 0};
    fk::vector<TestType> test_enlarged{2, 3, 4};
    test_reduced.resize(gold.size());
    test_enlarged.resize(gold.size());
    REQUIRE(test_reduced == gold);
    REQUIRE(test_enlarged == gold_enlarged);
  }

  SECTION("vector concatenation")
  {
    fk::vector<TestType> test_left        = {2, 3, 4};
    fk::vector<TestType> const test_right = {5, 6};
    fk::vector<TestType> empty;
    fk::vector<TestType> gold_copy = gold;

    REQUIRE(test_left.concat(test_right) == gold);
    REQUIRE(empty.concat(gold) == gold);
    empty.resize(0);
    REQUIRE(gold_copy.concat(empty) == gold);
  }
  SECTION("vector set")
  {
    fk::vector<TestType> vector(5);

    fk::vector<TestType> const empty;
    fk::vector<TestType> const begin  = {2, 3};
    fk::vector<TestType> const middle = {3, 4, 5};
    fk::vector<TestType> const end    = {6};

    REQUIRE(vector.set(0, begin).set(0, empty).set(1, middle).set(4, end) ==
            gold);
  }
  SECTION("vector extract")
  {
    fk::vector<TestType> const test_begin  = {2, 3, 4};
    fk::vector<TestType> const test_middle = {4, 5};
    fk::vector<TestType> const test_end    = {5, 6};

    REQUIRE(test_begin == gold.extract(0, 2));
    REQUIRE(test_middle == gold.extract(2, 3));
    REQUIRE(test_end == gold.extract(3, 4));
  }
  SECTION("vector transform")
  {
    fk::vector<TestType> test{-1, 1, 2, 3};
    fk::vector<TestType> const after{0, 2, 3, 4};
    std::transform(test.begin(), test.end(), test.begin(),
                   std::bind1st(std::plus<TestType>(), 1));
    REQUIRE(test == after);
  }

  SECTION("vector maximum element")
  {
    fk::vector<TestType> const test{5, 6, 11, 8};
    TestType max = 11;
    REQUIRE(*std::max_element(test.begin(), test.end()) == max);
  }

  SECTION("vector sum of elements")
  {
    fk::vector<TestType> const test{1, 2, 3, 4, 5, 6, 7, 8};
    TestType max = 36;
    REQUIRE(std::accumulate(test.begin(), test.end(), 0.0) == max);
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
      // (square slices of) our golden matrix is singular, so here's another
      // clang-format off
    fk::matrix<double> const ans_double {
       {-50.4950495049486321, 99.9999999999963762, -49.504950495047737},
       {-49.9999999999981171, 99.9999999999963478, -49.9999999999982379},
       {1321.00495049500046, -2627.9999999999045, 1307.99504950490405},
    };
    fk::matrix<float> const ans_single {
       {-50.4952354431152344, 100.000373840332031, -49.5051422119140625},
       {-50.0001907348632812, 100.000389099121094,  -50.0002021789550781},
       {1321.0098876953125, -2628.010009765625, 1308.0001220703125},
    };
    fk::matrix<TestType> test {
      {12.130, 14.150, 1.00},
      {13.140, 13.150, 1.00},
      {14.150, 12.130, 1.00},
    }; // clang-format on

      test.invert();
      if constexpr (std::is_same<TestType, double>::value)
      {
        REQUIRE(test == ans_double);
      }
      else if constexpr (std::is_same<TestType, float>::value)
      {
        REQUIRE(test == ans_single);
      }
      else
      {
        FAIL("Tests only configured for float and double precisions");
      }

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
