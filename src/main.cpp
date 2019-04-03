#include "coefficients.hpp"
#include "connectivity.hpp"
#include "element_table.hpp"
#include "pde.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include "transformations.hpp"

int main(int argc, char **argv)
{
  options opts(argc, argv);
  fk::vector<double> A = {1, 2, 3};
  A.print("A");
  fk::vector<double, mem_type::owner> B;
  B.print("B");
  // fk::vector<double, mem_type::view> C; // sfinae'd away as expected
  // C.print("C");

  fk::vector<double> D(2);
  D.print("D");
  fk::vector<double, mem_type::owner> E(2);
  E.print("E");

  // views are enabled, but behave exactly as owners (except print out that they
  // are a view)
  // fk::vector<double, mem_type::view> F(2); // sfinae'd away as expected
  // F.print("F");

  // testing ref counting
  //
  {
    A.print("A before view");
    fk::vector<double, mem_type::view> G(A, 0, A.size() - 1);
    A.print("A after view");
    fk::vector<double> H = {1, 2, 3, 4};
    fk::vector<double> J(4);
    fk::vector<double, mem_type::view> I(H);
    H.print("H");
    I.print("I");
    // J = std::move(H);
    J.print("J");
  }
  A.print("A next scope");
  B.print("B again");

  return 0;
}
