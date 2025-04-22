#include "asgard_test_macros.hpp"

using namespace asgard;

template<typename P>
void interp_nodes() {

  P constexpr tol = (std::is_same_v<P, double>) ? 1.E-12 : 1.E-5;

  {
    current_test<P> name_("linear nodes");

    int const max_level = 1;
    interpolation_manager1d<P, 1> interp;
    tassert(not interp); // default constructor
    interp = interpolation_manager1d<P, 1>(max_level);
    tassert(!!interp);

    tassert(interp.nodes().num_strips() == 2);
    tassert(interp.nodes().stride() == 2);

    P const *r = interp.nodes()[0];
    std::vector<P> ref = {1.0/3.0, 2.0/3.0, 1.0/6.0, 5.0/6.0};
    for (auto i : indexof(ref))
      tassert(std::abs(r[i] - ref[i]) < tol);

    interp = interpolation_manager1d<P, 1>(max_level + 1);
    tassert(!!interp);

    tassert(interp.nodes().num_strips() == 4);
    tassert(interp.nodes().stride() == 2);

    r = interp.nodes()[0];
    ref = {1.0/3.0, 2.0/3.0, 1.0/6.0, 5.0/6.0, 1.0/12.0, 5.0/12.0, 7.0/12.0, 11.0/12.0};
    for (auto i : indexof(ref))
      tassert(std::abs(r[i] - ref[i]) < tol);
  }
  {
    current_test<P> name_("quadratic nodes");

    int const max_level = 1;
    interpolation_manager1d<P, 2> interp(max_level);

    tassert(interp.nodes().num_strips() == 2);
    tassert(interp.nodes().stride() == 3);

    P const *r = interp.nodes()[0];
    std::vector<P> ref = {0.0, 1.0/3.0, 2.0/3.0, 1.0/6.0, 0.5, 5.0/6.0};
    for (auto i : indexof(ref))
      tassert(std::abs(r[i] - ref[i]) < tol);

    interp = interpolation_manager1d<P, 2>(max_level + 1);
    tassert(!!interp);

    tassert(interp.nodes().num_strips() == 4);
    tassert(interp.nodes().stride() == 3);

    r = interp.nodes()[0];
    ref = {0.0, 1.0/3.0, 2.0/3.0, 1.0/6.0, 0.5, 5.0/6.0,
           1.0/12.0, 0.25, 5.0/12.0, 7.0/12.0, 0.75, 11.0/12.0};
    for (auto i : indexof(ref))
      tassert(std::abs(r[i] - ref[i]) < tol);
  }
  {
    current_test<P> name_("cubic nodes");

    int const max_level = 2;
    interpolation_manager1d<P, 3> interp(max_level);

    tassert(interp.nodes().num_strips() == 4);
    tassert(interp.nodes().stride() == 4);

    P const *r = interp.nodes()[0];
    std::vector<P> ref = {1.0/5.0, 2.0/5.0, 3.0/5.0, 4.0/5.0,
                          1.0/10.0, 3.0/10.0, 7.0/10.0, 9.0/10.0,
                          1.0/20.0, 3.0/20.0, 7.0/20.0, 9.0/20.0,
                          11.0/20.0, 13.0/20.0, 17.0/20.0, 19.0/20.0};
    for (auto i : indexof(ref))
      tassert(std::abs(r[i] - ref[i]) < tol);
  }

}

// connect_1d conn(max_level, connect_1d::hierarchy::volume);

template<typename P>
void do_all_tests() {
  interp_nodes<P>();
}

int main(int, char**) {

  all_tests global_("interpolation framework", " handles non-separable operators");

  #ifdef ASGARD_ENABLE_DOUBLE
  do_all_tests<double>();
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  do_all_tests<float>();
  #endif

  return 0;
}
