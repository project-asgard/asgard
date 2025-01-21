#pragma once

#include "asgard.hpp"

std::string asgard_test_name;   // the name of the currently running test
bool asgard_test_pass  = true;  // helps in reporting whether the last test passed
bool asgard_all_tests  = true;  // reports total result of all tests


// test assert macro
#define tassert(_result_)      \
  if (!(_result_)){            \
    asgard_test_pass = false;  \
    asgard_all_tests = false;  \
    throw std::runtime_error("  test " + asgard_test_name \
                             + " in file: " + __FILE__    \
                             + " line: " + std::to_string(__LINE__) );  \
  }

#define tcheckless(_iinx_, _terr_, _ttol_)      \
  if ((_terr_) >= (_ttol_)){            \
    asgard_test_pass = false;  \
    asgard_all_tests = false;  \
    std::cerr << "at iteration = " << _iinx_ \
              << "  error = " << (_terr_) << "  tol = " << (_ttol_) << '\n'; \
    throw std::runtime_error("  test " + asgard_test_name \
                             + " in file: " + __FILE__    \
                             + " line: " + std::to_string(__LINE__) );  \
  }

namespace asgard {

struct all_tests {
  all_tests(std::string cname, std::string longer = "") : name(std::move(cname)) {
    std::cout << "\n ------------------------------------------------------------------------------ \n";
    std::cout << "    " << name << longer << "\n";
    std::cout << " ------------------------------------------------------------------------------ \n\n";
  }
  ~all_tests(){
    std::cout << "\n ------------------------------------------------------------------------------ \n";
    std::cout << "    " << name << " " << ((asgard_all_tests) ? "pass" : "FAIL") << "\n";
    std::cout << " ------------------------------------------------------------------------------ \n\n";
  }
  std::string name;
};

template<typename P>
struct current_test{
  current_test(std::string const &name) {
    asgard_test_name = name;
    asgard_test_pass = true;
  }
  current_test(std::string const &name, int num_dims, int level, bool adapt = false) {
    asgard_test_name = name + " " + std::to_string(num_dims) + "D  level " + std::to_string(level);
    asgard_test_name += (adapt) ? "  adapt" : "  no-adapt";
    asgard_test_pass = true;
  }
  current_test(std::string const &name, int num_dims, std::string const &extra = std::string()) {
    asgard_test_name = std::to_string(num_dims) + "D  '" + name + "'";
    if (not extra.empty())
      asgard_test_name += " (" + extra + ")";
    asgard_test_pass = true;
  }
  ~current_test(){
    std::string s ="";
    if constexpr (std::is_same_v<P, double>)
      s += "    (double) ";
    else
      s += "    (float)  ";

    s += asgard_test_name;

    if (s.size() < 60)
      std::cout << s << std::setw(70 - s.size()) << ((asgard_test_pass) ? "pass" : "FAIL") << '\n';
    else
      std::cout << s << "  " << ((asgard_test_pass) ? "pass" : "FAIL") << '\n';
  };
};

//! makes a grid over the domain of n points in each direction
template<typename P, typename P1>
vector2d<P> make_grid(pde_domain<P1> const &domain, int const n)
{
  int const num_dims = domain.num_dims();
  int const num_pnts = fm::ipow(n, num_dims);

  vector2d<P> pnts(num_dims, num_pnts);

  std::array<P, max_num_dimensions> dx;
  for (int d : iindexof(num_dims))
    dx[d] = domain.length(d) / (n + 1);

#pragma omp parallel for
  for (int i = 0; i < num_pnts; i++)
  {
    int t = i;
    for (int d = num_dims - 1; d >= 0; d--) {
      int const g = t / n;
      pnts[i][d] = domain.xleft(d) + (1 + t - g * n) * dx[d];
      t = g;
    }
  }

  return pnts;
}

}
