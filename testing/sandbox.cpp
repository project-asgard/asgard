#include "asgard.hpp"

using namespace asgard;

using prec = asgard::default_precision;

std::string pad_string(double x)
{
  std::ostringstream os;
  os.precision(3);
  os << x;

  std::string res = os.str();
  
  std::string::size_type dot = res.find(".");
  if (dot < res.size()) {
    std::string::size_type rem = 4 + dot - res.size();

    while(--rem)
      res += '0';    
  } else {
    res += ".00";
  }

  std::string pre = "";
  std::string::size_type rem = 15 - res.size();
  while (--rem)
    pre += ' ';

  return pre + res;
}

int main(int argc, char **argv)
{
  ignore(argc);
  ignore(argv);
  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior

  std::cout << pad_string(3);  
  std::cout << pad_string(3.1) << "\n";
  std::cout << pad_string(3.13);
  std::cout << pad_string(3.138296) << "\n";
 
 
  return 0;
}
