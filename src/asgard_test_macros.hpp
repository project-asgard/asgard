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
    throw std::runtime_error("test: " + asgard_test_name \
                             + "\n        in file: " + __FILE__    \
                             + "\n           line: " + std::to_string(__LINE__) );  \
  }

#define tcheckless_loud(_iinx_, _terr_, _ttol_)      \
  std::cerr << "at iteration = " << (_iinx_) \
            << "  error = " << (_terr_) << "  tol = " << (_ttol_) << '\n';

#define tcheckless(_iinx_, _terr_, _ttol_)      \
  if (std::isnan(_terr_) or (_terr_) >= (_ttol_)){            \
    asgard_test_pass = false;  \
    asgard_all_tests = false;  \
    tcheckless_loud(_iinx_, _terr_, _ttol_) \
    throw std::runtime_error("test: " + asgard_test_name \
                             + "\n        in file: " + __FILE__    \
                             + "\n           line: " + std::to_string(__LINE__) );  \
  }

#define terror_message(_code_, _message_) \
  try { \
    (_code_); \
  } catch (std::runtime_error &err) { \
    if (std::string_view(err.what()).find((_message_)) != 0) { \
      throw std::runtime_error(" expected error: '" + std::string((_message_)) + "' but found '" + err.what() + "'"); \
    } \
  } \

namespace asgard {

struct all_tests {
  all_tests(std::string cname, std::string longer = "") : name(std::move(cname)) {
    if (mpi::is_world_rank(0)) {
      std::cout << "\n ------------------------------------------------------------------------------ \n";
      std::cout << "    " << name << longer << "\n";
      std::cout << " ------------------------------------------------------------------------------ \n\n";
    }
  }
  ~all_tests(){
    if (mpi::is_world_rank(0)) {
      std::cout << "\n ------------------------------------------------------------------------------ \n";
      std::cout << "    " << name << " " << ((asgard_all_tests) ? "pass" : "FAIL") << "\n";
      std::cout << " ------------------------------------------------------------------------------ \n\n";
    }
  }
  std::string name;
};

template<typename P>
std::string prepend_type(std::string const &name) {
  if constexpr (std::is_same_v<P, double>) {
    return "(double) " + name;
  } else if constexpr (std::is_same_v<P, float>) {
    return "(float) " + name;
  } else if constexpr (std::is_same_v<P, int>) {
    return "(int) " + name;
  } else {
    return name;
  }
}

template<typename P = void>
struct current_test{
  current_test(std::string const &name) {
    asgard_test_name = prepend_type<P>(name);
    asgard_test_pass = true;
  }
  current_test(std::string const &name, int num_dims, int level, bool adapt = false) {
    asgard_test_name = prepend_type<P>(name) + " " + std::to_string(num_dims) + "D  level " + std::to_string(level);
    asgard_test_name += (adapt) ? "  adapt" : "  no-adapt";
    asgard_test_pass = true;
  }
  current_test(std::string const &name, int num_dims, std::string const &extra = std::string()) {
    asgard_test_name = prepend_type<P>(std::to_string(num_dims) + "D  '" + name + "'");
    if (not extra.empty())
      asgard_test_name += " (" + extra + ")";
    asgard_test_pass = true;
  }
  ~current_test(){
    if (mpi::is_world_rank(0)) {
      std::string s = "    " + asgard_test_name;

      if (s.size() < 60)
        std::cout << s << std::setw(70 - s.size()) << ((asgard_test_pass) ? "pass" : "FAIL") << '\n';
      else
        std::cout << s << "  " << ((asgard_test_pass) ? "pass" : "FAIL") << '\n';
    }
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
