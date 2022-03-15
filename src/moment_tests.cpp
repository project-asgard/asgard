#include "moment.hpp"
#include "tests_general.hpp"

TEMPLATE_TEST_CASE("Multiwavelet", "[transformations]", double, float)
{
  std::vector<vector_func<TestType>> md_func;	
  SECTION("Constructor")
  {	  
    moment<TestType> mymoment(md_func);
  }
}
