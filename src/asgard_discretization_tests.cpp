#include "asgard_test_macros.hpp"

#include "asgard_testpdes.hpp"

using namespace asgard;

template<typename P>
void test_domain()
{
  {
    current_test<P> name_("domain settings");

    pde_domain<P> dom1(3);
    tassert(dom1.num_dims() == 3 and dom1.num_pos() == 0 and dom1.num_vel() == 0);
    tassert(dom1.xleft(0) == 0 and dom1.xright(0) == 1 and dom1.length(0) == 1);

    pde_domain<P> dom2(position_dims{1}, velocity_dims{2});
    tassert(dom2.num_dims() == 3 and dom2.num_pos() == 1 and dom2.num_vel() == 2);
    tassert(dom2.xleft(1) == 0 and dom2.xright(1) == 1 and dom2.length(1) == 1);
  }
  {
    current_test<P> name_("domain error-checking");

    terror_message(pde_domain<P>(0),
                   "pde_domain created with zero or negative dimensions");
    terror_message(pde_domain<P>(max_num_dimensions + 1),
                   "pde_domain created with too many dimensions");
    terror_message(pde_domain<P>(position_dims{-1}, velocity_dims{1}),
                   "pde_domain created with negative position dimensions");
    terror_message(pde_domain<P>(position_dims{1}, velocity_dims{-2}),
                   "pde_domain created with negative velocity dimensions");
  }
}


template<typename pde_type, typename P>
discretization_manager<P> disc_testpde(int num_dims, prog_opts const &opts) {
  return discretization_manager<P>(make_testpde<pde_type, P>(num_dims, opts),
                                   verbosity_level::quiet);
}

template<typename P>
void compile_tests()
{
  {
    current_test<P> name_("discretization manager compile tests");
    discretization_manager<P> disc_null;
    tassert(disc_null.num_dims() == 0);
    tassert(not disc_null.has_moments());
    tassert(not disc_null.high_verbosity());
    tassert(not disc_null.low_verbosity());
    tassert(disc_null.stop_verbosity());
    static_assert(std::is_copy_constructible_v<discretization_manager<P>>);
    static_assert(std::is_move_constructible_v<discretization_manager<P>>);
    static_assert(std::is_copy_assignable_v<discretization_manager<P>>);
    static_assert(std::is_move_assignable_v<discretization_manager<P>>);
    static_assert(std::is_same_v<typename discretization_manager<P>::precision_type, P>);
  }
}

template<typename P>
void init_tests()
{
  {
    current_test<P> name_("zero steps time");

    auto disc1 = disc_testpde<pde_contcos, P>(1, make_opts("-l 1 -d 0 -n 0"));
    tassert(disc1.time() == 0);
    tassert(disc1.degree() == 0);
    tassert(disc1.num_dims() == 1);
    tassert(disc1.remaining_steps() == 0);
    tassert(disc1.current_state().size() == 2);

    auto disc2 = disc_testpde<pde_contcos, P>(2, make_opts("-l 1 -d 1 -t 0"));
    tassert(disc2.num_dims() == 2);
    tassert(disc2.remaining_steps() == 0);
    tassert(disc2.current_state().size() == 12);
  }
  {
    current_test<P> name_("final time");

    auto disc1 = disc_testpde<pde_contcos, P>(1, make_opts("-l 1 -d 0 -t 9 -dt 1"));
    tassert(disc1.time() == 0);
    tassert(disc1.remaining_steps() == 9);

    auto disc2 = disc_testpde<pde_contcos, P>(3, make_opts("-l 0 -d 1 -n 7 -dt 1"));
    tassert(disc2.num_dims() == 3);
    tassert(disc2.remaining_steps() == 7);
    tassert(disc2.stop_time() == 7);
    tassert(disc2.degree() == 1);
  }
}

template<typename P>
void do_all_tests() {
  test_domain<P>();
  compile_tests<P>();
  init_tests<P>();
}

int main(int argc, char **argv)
{
  libasgard_runtime running_(argc, argv);

  all_tests global_("discretization-manager", " discretization details");

  #ifdef ASGARD_ENABLE_DOUBLE
  do_all_tests<double>();
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  do_all_tests<float>();
  #endif

  return 0;
}
