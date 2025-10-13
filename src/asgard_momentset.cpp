
#include "asgard_momentset.hpp"

namespace asgard
{

std::ostream& operator<<(std::ostream& os, moment const &m) {
  if (m.num_dims() == 0) os << "()";
  else if (m.num_dims() == 1) os << "(" << m.pows[0] << ")";
  else {
    os << "(" << m.pows[0];
    for (int i = 1; i < m.num_dims(); i++)
      os << ", " << m.pows[i];
    os << ")";
  }
  return os;
}


}
