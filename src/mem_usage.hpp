#include "pde.hpp"

// This object just allows us to less verbosely total up mem usage
// When displaying each partial contribution to the total memory
// consumption, we can just `std::cout << (mem += mem_used) << std::endl`
class mem_tracker
{
public:
  mem_tracker() {}

  double operator+=(double mem_used)
  {
    total_ += mem_used;
    last_ = mem_used;
    return mem_used;
  }

  double total_mem_usage() { return total_; }

private:
  double last_  = 0;
  double total_ = 0;
};
