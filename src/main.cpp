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

  // doesn't work because we haven't explicitly instantiated all of the
  // (non-templated) methods using
  // 'extern template class fk::vector<double, mem_type::view>'
  // This, however doesn't work either because anything that takes a vector as
  // an argument will get that argument as an owner, and so a different type,
  // and so can't access private members. This needs to be cleaned up.
  //
  // fk::vector<double, mem_type::view> F(2);
  // F.print("F");
  return 0;
}
