
#include "tensors.hpp"
#include "tests_general.hpp"
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>

TEST_CASE("fk::vector interface: constructors, copy/move", "[tensors]")
{
  // set up the golden vector
  // orthogonality warnings: all of the tests depend on
  // - the list initializing constructor working correctly
  // - operator== working correctly
  fk::vector<double> const gold{2.3, 3.4, 4.5, 5.6, 6.7};

  SECTION("default constructor")
  {
    fk::vector<double> test;
    REQUIRE(test.size() == 0);
  }
  SECTION("give me some size, initialized to zero")
  {
    fk::vector<double> test(5);
    fk::vector<double> zeros{0.0, 0.0, 0.0, 0.0, 0.0};
    REQUIRE(test == zeros);
  }
  SECTION("constructor from list initialization")
  {
    fk::vector<double> test{2.3, 3.4, 4.5, 5.6, 6.7};
    REQUIRE(test == gold);
  }
  SECTION("construct from a std::vector")
  {
    std::vector v{2.3, 3.4, 4.5, 5.6, 6.7};
    fk::vector<double> test(v);
    REQUIRE(test == gold);
  }
  SECTION("copy construction")
  {
    fk::vector<double> test(gold);
    REQUIRE(test == gold);
  }
  SECTION("copy assignment")
  {
    fk::vector<double> test(5);
    test = gold;
    REQUIRE(test == gold);
  }
  SECTION("move construction")
  {
    fk::vector<double> moved{2.3, 3.4, 4.5, 5.6, 6.7};
    fk::vector<double> test(std::move(moved));
    REQUIRE(test == gold);
  }
  SECTION("move assignment")
  {
    fk::vector<double> moved{2.3, 3.4, 4.5, 5.6, 6.7};
    fk::vector<double> test(5);
    test = std::move(moved);
    REQUIRE(test == gold);
  }
  SECTION("copy from std::vector")
  {
    std::vector v{2.3, 3.4, 4.5, 5.6, 6.7};
    fk::vector<double> test(5);
    test = v;
    REQUIRE(test == gold);
  }
  SECTION("copy into std::vector")
  {
    std::vector goldv{2.3, 3.4, 4.5, 5.6, 6.7};
    std::vector<double> testv;
    testv = gold.to_std();
    compareVectors(testv, goldv);
  }
} // end fk::vector constructors, copy/move

TEST_CASE("fk::vector operators", "[tensors]")
{
  fk::vector<double> const gold{2.3, 3.4, 4.5, 5.6, 6.7};
  SECTION("subscript operator (modifying)")
  {
    fk::vector<double> test(5);
    // clang-format off
    test(0) = 2.3; test(1) = 3.4; test(2) = 4.5; test(3) = 5.6; test(4) = 6.7;
    // clang-format on
    REQUIRE(test == gold);
    double val = test(4);
    REQUIRE(val == 6.7);
  }
  SECTION("subscript operator (const)") { REQUIRE(gold(4) == 6.7); }
  SECTION("comparison operator") // this gets used in every REQUIRE
  SECTION("comparison (negated) operator")
  {
    fk::vector<double> test(gold);
    test(4) = 333.33;
    REQUIRE(test != gold);
  }
  SECTION("addition operator")
  {
    fk::vector<double> const in1{1.2, 1.3, 1.4, 1.5, 1.6};
    fk::vector<double> const in2{1.1, 2.1, 3.1, 4.1, 5.1};
    REQUIRE((in1 + in2) == gold);
  }
  SECTION("subtraction operator")
  {
    fk::vector<double> const in1{3.6, 4.8, 5.9, 6.9, 7.9};
    fk::vector<double> const in2{1.3, 1.4, 1.4, 1.3, 1.2};
    REQUIRE((in1 - in2) == gold);
  }
  SECTION("vector*vector operator") { REQUIRE((gold * gold) == 113.35); }
  SECTION("vector*matrix operator")
  {
    // clang-format off
    fk::matrix<double> const testm{
      {12.13, 22.23, 32.33},
      {13.14, 23.24, 33.34},
      {14.15, 24.25, 34.35},
      {15.16, 25.26, 35.36},
      {16.17, 26.27, 36.37},
    }; // clang-format on
    fk::vector<double> const testv{2.3, 3.4, 4.5, 5.6, 6.7};
    fk::vector<double> const gold{329.485, 556.735, 783.985};
    REQUIRE((testv * testm) == gold);
  }
} // end fk::vector operators

TEST_CASE("fk::vector utilities", "[tensors]")
{
  fk::vector<double> const gold{2.3, 3.4, 4.5, 5.6, 6.7};
  SECTION("size(): the number of elements") { REQUIRE(gold.size() == 5); }
  SECTION("data(): const addr to element") { REQUIRE(*gold.data(4) == 6.7); }
  SECTION("print out the values")
  {
    // (effectively) redirect cout
    std::streambuf *oldCoutStreamBuf = std::cout.rdbuf();
    std::ostringstream test_str;
    std::cout.rdbuf(test_str.rdbuf());
    // generate the output (into test_str)
    gold.print("golden vector");
    // restore cout destination
    std::cout.rdbuf(oldCoutStreamBuf);
    std::string golden_string("golden vector\n  2.3000e+00  3.4000e+00  "
                              "4.5000e+00  5.6000e+00  6.7000e+00\n");
    REQUIRE(test_str.str() == golden_string);
  }
  SECTION("dump to octave")
  {
    gold.dump_to_octave("test_out.dat");
    std::ifstream data_stream("test_out.dat");
    std::string test_string((std::istreambuf_iterator<char>(data_stream)),
                            std::istreambuf_iterator<char>());
    std::remove("test_out.dat");
    std::string golden_string(
        "2.300000000000e+00 3.400000000000e+00 4.500000000000e+00 "
        "5.600000000000e+00 6.700000000000e+00 ");
    REQUIRE(test_string == golden_string);
  }
  SECTION("vector resize")
  {
    fk::vector<double> test_reduced{2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9};
    fk::vector<double> const gold_enlarged{2.3, 3.4, 4.5, 0.0, 0.0};
    fk::vector<double> test_enlarged{2.3, 3.4, 4.5};
    test_reduced.resize(gold.size());
    test_enlarged.resize(gold.size());
    REQUIRE(test_reduced == gold);
    REQUIRE(test_enlarged == gold_enlarged);
  }

  SECTION("vector transform")
  {
    fk::vector<double> test{-1.0, 1.0, 2.0, 3.0};
    fk::vector<double> after{0.0, 2.0, 3.0, 4.0};
    std::transform(test.begin(), test.end(), test.begin(),
                   std::bind1st(std::plus<double>(), 1.0));
    REQUIRE(test == after);
  }

  SECTION("vector maximum element")
  {
    fk::vector<double> test{5.0, 6.0, 11.0, 8.0};
    double max = 11.0;
    REQUIRE(*std::max_element(test.begin(), test.end()) == max);
  }

  SECTION("vector sum of elements")
  {
    fk::vector<double> test{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double max = 36.0;
    REQUIRE(std::accumulate(test.begin(), test.end(), 0.0) == max);
  }
} // end fk::vector utilities

TEST_CASE("fk::matrix interface: constructors, copy/move", "[tensors]")
{
  // set up the golden vector
  // clang-format off
  fk::matrix<double> const gold{
    {12.13, 22.23, 32.33},
    {13.14, 23.24, 33.34},
    {14.15, 24.25, 34.35},
    {15.16, 25.26, 35.36},
    {16.17, 26.27, 36.37},
  }; // clang-format on

  SECTION("default constructor")
  {
    fk::matrix<double> test;
    REQUIRE(test.size() == 0);
  }
  SECTION("give me some size, initialized to zero")
  {
    fk::matrix<double> test(5, 3);
    // clang-format off
    fk::matrix<double> const zeros{
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0},
    }; // clang-format on
    REQUIRE(test == zeros);
  }
  SECTION("constructor from list initialization")
  {
    // clang-format off
    fk::matrix<double> const test{
      {12.13, 22.23, 32.33},
      {13.14, 23.24, 33.34},
      {14.15, 24.25, 34.35},
      {15.16, 25.26, 35.36},
      {16.17, 26.27, 36.37},
    }; // clang-format on
    REQUIRE(test == gold);
  }
  SECTION("copy construction")
  {
    fk::matrix<double> test(gold);
    REQUIRE(test == gold);
  }
  SECTION("copy assignment")
  {
    fk::matrix<double> test(5, 3);
    test = gold;
    REQUIRE(test == gold);
  }
  SECTION("move construction")
  {
    // clang-format off
    fk::matrix<double> moved{
      {12.13, 22.23, 32.33},
      {13.14, 23.24, 33.34},
      {14.15, 24.25, 34.35},
      {15.16, 25.26, 35.36},
      {16.17, 26.27, 36.37},
    }; // clang-format on
    fk::matrix<double> test(
        [](fk::matrix<double> in) -> fk::matrix<double> { return in; }(moved));
    // fk::matrix test(std::move(moved));
    REQUIRE(test == gold);
  }
  SECTION("move assignment")
  {
    // clang-format off
    fk::matrix<double> moved{
      {12.13, 22.23, 32.33},
      {13.14, 23.24, 33.34},
      {14.15, 24.25, 34.35},
      {15.16, 25.26, 35.36},
      {16.17, 26.27, 36.37},
    }; // clang-format on
    fk::matrix<double> test(5, 3);
    test = std::move(moved);
    REQUIRE(test == gold);
  }
  SECTION("copy from std::vector")
  {
    // clang-format off
    std::vector v
      {12.13, 22.23, 32.33,
       13.14, 23.24, 33.34,
       14.15, 24.25, 34.35,
       15.16, 25.26, 35.36,
       16.17, 26.27, 36.37};
    // clang-format on
    fk::matrix<double> test(5, 3);
    test = v;
    REQUIRE(test == gold);
  }
} // end fk::matrix constructors, copy/move

TEST_CASE("fk::matrix operators", "[tensors]")
{
  // set up the golden vector
  // clang-format off
  fk::matrix<double> const gold {
    {12.13, 22.23, 32.33},
    {13.14, 23.24, 33.34},
    {14.15, 24.25, 34.35},
    {15.16, 25.26, 35.36},
    {16.17, 26.27, 36.37},
  }; // clang-format on

  SECTION("subscript operator (modifying)")
  {
    fk::matrix<double> test(5, 3);
    // clang-format off
    test(0,0) = 12.13;  test(0,1) = 22.23;  test(0,2) = 32.33;
    test(1,0) = 13.14;  test(1,1) = 23.24;  test(1,2) = 33.34;
    test(2,0) = 14.15;  test(2,1) = 24.25;  test(2,2) = 34.35;
    test(3,0) = 15.16;  test(3,1) = 25.26;  test(3,2) = 35.36;
    test(4,0) = 16.17;  test(4,1) = 26.27;  test(4,2) = 36.37;
    // clang-format on
    REQUIRE(test == gold);
    double val = test(4, 2);
    REQUIRE(val == 36.37);
  }
  SECTION("subscript operator (const)")
  {
    double test = gold(4, 2);
    REQUIRE(test == 36.37);
  }
  SECTION("comparison operator") // this gets used in every REQUIRE
  SECTION("comparison (negated) operator")
  {
    fk::matrix<double> test(gold);
    test(4, 2) = 333.33;
    REQUIRE(test != gold);
  }
  SECTION("matrix+matrix addition")
  {
    // clang-format off
    fk::matrix<double> const in1 {
      {11.10, 20.21, 0.0},
      {12.10, 21.22, 0.0},
      {13.10, 22.23, 0.0},
      {14.10, 23.24, 0.0},
      {15.10, 24.25, 0.0},
    };
    fk::matrix<double> const in2 {
      {1.03, 2.02, 32.33},
      {1.04, 2.02, 33.34},
      {1.05, 2.02, 34.35},
      {1.06, 2.02, 35.36},
      {1.07, 2.02, 36.37},
    }; // clang-format on
    REQUIRE((in1 + in2) == gold);
  }
  SECTION("matrix-matrix subtraction")
  {
    // clang-format off
    fk::matrix<double> const in1 {
      {13.23, 22.23, 32.34},
      {14.24, 23.24, 33.35},
      {15.25, 24.25, 34.36},
      {16.26, 25.26, 35.37},
      {17.27, 26.27, 36.38},
    };
    fk::matrix<double> const in2 {
      {1.1, 0.0, 0.01},
      {1.1, 0.0, 0.01},
      {1.1, 0.0, 0.01},
      {1.1, 0.0, 0.01},
      {1.1, 0.0, 0.01}
    }; // clang-format on
    REQUIRE((in1 - in2) == gold);
  }
  SECTION("matrix*integer multiplication")
  {
    // clang-format off
    fk::matrix<double> in {
      {3.03250, 5.55750, 8.08250},
      {3.28500, 5.81000, 8.33500},
      {3.53750, 6.06250, 8.58750},
      {3.79000, 6.31500, 8.84000},
      {4.04250, 6.56750, 9.09250},
    }; // clang-format on
    REQUIRE((in * 4) == gold);
  }
  SECTION("matrix*matrix multiplication")
  {
    // I'm not factoring the golden matrix, so here's a new answer (calculated
    // from octave)
    // clang-format off
    fk::matrix<double> const ans {
      {252.8283750,  431.4721250,  610.1158750},
      {431.4721250,  737.6283750, 1043.7846250},
    };
    fk::matrix<double> const in1 {
      {3.03250, 3.2850, 3.53750, 3.7900, 4.04250},
      {5.55750, 5.8100, 6.06250, 6.3150, 6.56750},
    };
    fk::matrix<double> const in2 {
      {12.13, 22.23, 32.33},
      {13.14, 23.24, 33.34},
      {14.15, 24.25, 34.35},
      {15.16, 25.26, 35.36},
      {16.17, 26.27, 36.37},
    };
    // clang-format on
    fk::matrix<double> test = in1 * in2;
    REQUIRE(test == ans);
  }
  SECTION("matrix inverse")
  {
    // (square slices of) our golden matrix is singular, so here's another
    // clang-format off
    fk::matrix<double> const ans {
      {-50.4950495049486, 100.00, -49.5049504950477},
      {-50.00, 100.00, -50.00},
      {1321.0049504950016, -2628.00, 1307.9950495049027},
    };
    fk::matrix<double> test {
      {12.130, 14.150, 1.00},
      {13.140, 13.150, 1.00},
      {14.150, 12.130, 1.00},
    }; // clang-format on
    fk::matrix<double> test2 = test.invert();
    REQUIRE(test == ans);
    REQUIRE(test2 == ans);
  }
  SECTION("matrix determinant")
  {
    // clang-format off
    fk::matrix<double> in {
      {12.130, 14.150, 1.00},
      {13.140, 13.150, 1.00},
      {14.150, 12.130, 1.00},
    }; // clang-format on
    REQUIRE(in.determinant() == Approx(-0.020200));
  }
  SECTION("nrows(): the number of rows") { REQUIRE(gold.nrows() == 5); }
  SECTION("ncols(): the number of columns") { REQUIRE(gold.ncols() == 3); }
  SECTION("size(): the number of elements") { REQUIRE(gold.size() == 15); }
  SECTION("data(): const get address to an element")
  {
    REQUIRE(*gold.data(4, 2) == 36.37);
  }
} // end fk::matrix operators

TEST_CASE("fk::matrix utilities", "[tensors]")
{
  // set up the golden vector
  // clang-format off
  fk::matrix<double> const gold {
    {12.13, 22.23, 32.33},
    {13.14, 23.24, 33.34},
    {14.15, 24.25, 34.35},
    {15.16, 25.26, 35.36},
    {16.17, 26.27, 36.37},
  }; // clang-format on

  SECTION("matrix update_col(fk::vector)")
  {
    // clang-format off
    fk::matrix<double> test {
      {12.13, 22.23, 00.00},
      {13.14, 23.24, 00.00},
      {14.15, 24.25, 00.00},
      {15.16, 25.26, 00.00},
      {16.17, 26.27, 52.51},
    }; // clang-format on
    fk::vector<double> testv{32.33, 33.34, 34.35, 35.36, 36.37};
    REQUIRE(test.update_col(2, testv) == gold);
  }
  SECTION("matrix update_col(std::vector)")
  {
    // clang-format off
    fk::matrix<double> test {
      {12.13, 22.23, 00.00},
      {13.14, 23.24, 00.00},
      {14.15, 24.25, 00.00},
      {15.16, 25.26, 00.00},
      {16.17, 26.27, 52.51},
    }; // clang-format on
    std::vector testv{32.33, 33.34, 34.35, 35.36, 36.37};
    REQUIRE(test.update_col(2, testv) == gold);
  }

  SECTION("matrix update_row(fk::vector)")
  {
    // clang-format off
    fk::matrix<double> test {
      {12.13, 22.23, 32.33},
      {13.14, 23.24, 33.34},
      {14.15, 24.25, 34.35},
      {15.16, 25.26, 35.36},
      {00.00, 00.00, 35.36},
    }; // clang-format on
    fk::vector<double> testv{16.17, 26.27, 36.37};
    REQUIRE(test.update_row(4, testv) == gold);
  }
  SECTION("matrix update_row(std::vector)")
  {
    // clang-format off
    fk::matrix<double> test {
      {12.13, 22.23, 32.33},
      {13.14, 23.24, 33.34},
      {14.15, 24.25, 34.35},
      {15.16, 25.26, 35.36},
      {00.00, 00.00, 35.36},
    }; // clang-format on
    std::vector testv{16.17, 26.27, 36.37};
    REQUIRE(test.update_row(4, testv) == gold);
  }
  SECTION("matrix set submatrix(row, col, submatrix)")
  {
    // clang-format off
    fk::matrix<double> test {
      {12.13, 22.23, 32.33},
      {13.14, 23.24, 33.34},
      {14.15, 24.25, 34.35},
      {15.16, 25.26, 35.36},
      {00.00, 00.00, 35.36},
    }; 
    fk::matrix<double> sub {
      {-13.14, -23.24},
      {-14.15, -24.25},
      {-15.16, -25.26},
    };
    fk::matrix<double> after_set {
      {12.13, 22.23, 32.33},
      {13.14, -13.14, -23.24},
      {14.15, -14.15, -24.25},
      {15.16, -15.16, -25.26},
      {00.00, 00.00, 35.36},
    }; // clang-format on

    REQUIRE(test.set_submatrix(1, 1, sub) == after_set);
  }

  SECTION("matrix extract submatrix(row, col, nrows, ncols")
  {
    // clang-format off
    fk::matrix<double> test {
      {12.13, 22.23, 32.33},
      {13.14, 23.24, 33.34},
      {14.15, 24.25, 34.35},
      {15.16, 25.26, 35.36},
      {00.00, 00.00, 35.36},
    }; 
    fk::matrix<double> sub {
      {13.14, 23.24},
      {14.15, 24.25},
      {15.16, 25.26},
    }; // clang-format on

    REQUIRE(test.extract_submatrix(1, 0, 3, 2) == sub);
  }

  SECTION("print out the values")
  {
    // (effectively) redirect cout
    std::streambuf *oldCoutStreamBuf = std::cout.rdbuf();
    std::ostringstream test_str;
    std::cout.rdbuf(test_str.rdbuf());
    // generate the output (into test_str)
    gold.print("golden matrix");
    // restore cout destination
    std::cout.rdbuf(oldCoutStreamBuf);
    std::string golden_string(
        "golden matrix\n  1.2130e+01  2.2230e+01  3.2330e+01\n  1.3140e+01  "
        "2.3240e+01  3.3340e+01\n  1.4150e+01  2.4250e+01  3.4350e+01\n  "
        "1.5160e+01  2.5260e+01  3.5360e+01\n  1.6170e+01  2.6270e+01  "
        "3.6370e+01\n");
    REQUIRE(test_str.str() == golden_string);
  }
  SECTION("dump to octave")
  {
    gold.dump_to_octave("test_out.dat");
    std::ifstream data_stream("test_out.dat");
    std::string test_string((std::istreambuf_iterator<char>(data_stream)),
                            std::istreambuf_iterator<char>());
    std::remove("test_out.dat");
    std::string golden_string(
        "1.213000000000e+01 2.223000000000e+01 3.233000000000e+01 \n"
        "1.314000000000e+01 2.324000000000e+01 3.334000000000e+01 \n"
        "1.415000000000e+01 2.425000000000e+01 3.435000000000e+01 \n"
        "1.516000000000e+01 2.526000000000e+01 3.536000000000e+01 \n"
        "1.617000000000e+01 2.627000000000e+01 3.637000000000e+01 \n");
    REQUIRE(test_string == golden_string);
  }

  SECTION("matrix transform")
  {
    // clang-format off
  fk::matrix<double> test {
   {0.0, 1.0, 2.0, 3.0},
   {4.0, 5.0, 6.0, 7.0},
  };
  fk::matrix<double> after {
   {1.0, 2.0, 3.0, 4.0},
   {5.0, 6.0, 7.0, 8.0},
  }; // clang-format on 
  std::transform(test.begin(), test.end(), test.begin(), std::bind1st(std::plus<double>(), 1.0));
  REQUIRE(test == after);
  }

  SECTION("matrix maximum element") {
  // clang-format off
  fk::matrix<double> test {
   {1.0, 2.0, 3.0, 4.0},
   {5.0, 6.0, 11.0, 8.0},
  }; // clang-format on
    double max = 11.0;

    REQUIRE(*std::max_element(test.begin(), test.end()) == max);
  }

  SECTION("matrix sum of elements")
  {
    // clang-format off
  fk::matrix<double> test {
   {1.0, 2.0, 3.0, 4.0},
   {5.0, 6.0, 7.0, 8.0},
  }; // clang-format on
    double max = 36.0;

    REQUIRE(std::accumulate(test.begin(), test.end(), 0.0) == max);
  }

} // end fk::matrix utilities
