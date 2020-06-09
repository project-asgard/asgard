#include "build_info.hpp"
#include "distribution.hpp"

#include "program_options.hpp"

#include "tests_general.hpp"
#include <iostream>
#include <string>

struct distribution_test_init
{
  distribution_test_init() { initialize_distribution(); }
  ~distribution_test_init() { finalize_distribution(); }
};

#ifdef ASGARD_USE_MPI
static distribution_test_init const distrib_test_info;
#endif

TEST_CASE("options constructor/getters", "[options]")
{
  SECTION("create from valid string")
  {
    // the golden values
    std::string pde_choice = "vlasov43";
    PDE_opts pde           = PDE_opts::vlasov43;
    int const level        = 3;
    int const degree       = 4;
    int const write        = 1;
    int const realspace    = 1;
    double const cfl       = 2.0;

    // set up test inputs directly from golden values
    options o =
        make_options({"-p", pde_choice, "-l", std::to_string(level), "-d",
                      std::to_string(degree), "-w", std::to_string(write), "-r",
                      std::to_string(realspace), "-f", "-i", "-e", "-c",
                      std::to_string(cfl)});

    REQUIRE(o.get_degree() == degree);
    REQUIRE(o.get_level() == level);
    REQUIRE(o.get_write_frequency() == write);
    REQUIRE(o.get_realspace_output_freq() == realspace);
    REQUIRE(o.using_implicit());
    REQUIRE(o.using_full_grid());
    REQUIRE(o.do_poisson_solve());
    REQUIRE(o.get_cfl() == cfl);
    REQUIRE(o.get_selected_pde() == pde);
  }

  SECTION("run w/ defaults")
  {
    int const def_level          = -1;
    int const def_degree         = -1;
    int const def_max_level      = 12;
    int const def_num_steps      = 10;
    int const def_write_freq     = 0;
    int const def_realspace_freq = 0;
    bool const def_implicit      = false;
    bool const def_full_grid     = false;
    bool const def_poisson       = false;
    double const def_cfl         = 0.01;
    PDE_opts const def_pde       = PDE_opts::continuity_2;

    options o = make_options({});

    REQUIRE(o.get_degree() == def_degree);
    REQUIRE(o.get_level() == def_level);
    REQUIRE(o.get_max_level() == def_max_level);
    REQUIRE(o.get_time_steps() == def_num_steps);
    REQUIRE(o.get_write_frequency() == def_write_freq);
    REQUIRE(o.get_realspace_output_freq() == def_realspace_freq);
    REQUIRE(o.using_implicit() == def_implicit);
    REQUIRE(o.using_full_grid() == def_full_grid);
    REQUIRE(o.do_poisson_solve() == def_poisson);
    REQUIRE(o.get_cfl() == def_cfl);
    REQUIRE(o.get_selected_pde() == def_pde);
    REQUIRE(o.is_valid());
  }

  SECTION("out of range pde")
  {
    std::cerr.setstate(std::ios_base::failbit);
    options o = make_options({"asgard", "-p", "2 1337 4 u gg"});
    std::cerr.clear();
    REQUIRE(!o.is_valid());
  }

  SECTION("out of range solver")
  {
    std::cerr.setstate(std::ios_base::failbit);
    options o = make_options({"asgard", "-s", "2 1337 4 u gg"});
    std::cerr.clear();
    REQUIRE(!o.is_valid());
  }

  SECTION("negative level")
  {
    std::cerr.setstate(std::ios_base::failbit);
    options o = make_options({"asgard", "-l=-2"});
    std::cerr.clear();
    REQUIRE(!o.is_valid());
  }

  SECTION("negative degree")
  {
    std::cerr.setstate(std::ios_base::failbit);
    options o = make_options({"asgard", "-d=-2"});
    std::cerr.clear();
    REQUIRE(!o.is_valid());
  }

  SECTION("max level < starting level")
  {
    std::cerr.setstate(std::ios_base::failbit);
    options o = make_options({"asgard", "-l=3", "-m=2"});
    std::cerr.clear();
    REQUIRE(!o.is_valid());
  }

  SECTION("negative cfl")
  {
    std::cerr.setstate(std::ios_base::failbit);
    options o = make_options({"asgard", "-c=-2.0"});
    std::cerr.clear();
    REQUIRE(!o.is_valid());
  }
}
