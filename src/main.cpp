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
  fk::vector<double> A;
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

  return 0;
}
