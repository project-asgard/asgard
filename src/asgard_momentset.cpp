
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

bool moments_list::have_all_dimension(int const dims) const {
  for (auto const &m : moms_)
    if (m.num_dims() != dims)
      return false;
  return true;
}

std::vector<moment_id>
moments_list::find_as_subset_of(moments_list const &superset) const {
  std::vector<moment_id> result;
  result.reserve(moms_.size());
  for (auto const &m : moms_)
    result.push_back(superset.get_id(m));
  return result;
}

moment moments_list::max_moment() const {
  moment result(-1, -1, -1);
  for (auto const &m : moms_)
    for (int i = 0; i < max_mom_dims; i++)
      result.pows[i] = std::max(result.pows[i], m.pows[i]);
  return result;
}

int moments_list::max_moment(int dim) const {
  int result = -1;
  for (auto const &m : moms_)
    result = std::max(result, m.pows[dim]);
  return result;
}

}
