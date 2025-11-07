#include "asgard.hpp"

#include "asgard_test_pdes.hpp"
#include "asgard_test_macros.hpp"

using namespace asgard;

using P = asgard::default_precision;

template<typename P>
void restart_nonlinear() {
  current_test<P> name_("nonlinear restart");

  using pde = pde_burgers;

  auto options = make_opts("-l 6 -m 8 -d 2 -a 1.E-5 -n 0 -of _asg_testfile.h5");
  discretization_manager<P> init_disc(make_testpde<pde, P>(2, options));

  double const default_dt = init_disc.options().default_dt.value();

  init_disc.advance_time();
  tassert(init_disc.time() == 0);
  init_disc.save_final_snapshot();

  { // read from a file with already initialized grid
    auto ropts = make_opts("-restart _asg_testfile.h5 -t 0.25");
    discretization_manager<P> rdisc(make_testpde<pde, P>(2, ropts));

    // careful here, the assumption is that -t 0.25 divides evenly into dt
    std::cout << rdisc.dt() << "   " << default_dt << '\n';
    tassert(rdisc.dt() == default_dt);
  }

}

int main(int argc, char **argv)
{
  ignore(argc);
  ignore(argv);
  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior

  restart_nonlinear<double>();

  return 0;
}
