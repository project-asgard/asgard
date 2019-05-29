#include "program_options.hpp"

#include "tests_general.hpp"
#include <iostream>
#include <string>

TEST_CASE("options constructor/getters", "[options]")
{
  SECTION("create from valid string")
  {
    // the golden values
    std::string pde_choice = "vlasov43";
    PDE_opts pde           = PDE_opts::vlasov43;
    int level              = 3;
    int degree             = 4;
    int write              = 1;
    int vis                = 1;
    double cfl             = 2.0;

    // set up test inputs directly from golden values
    options o = make_options({"-p", pde_choice, "-l", std::to_string(level),
                              "-d", std::to_string(degree), "-w",
                              std::to_string(write), "-z", std::to_string(vis),
                              "-f", "-i", "-s", "-c", std::to_string(cfl)});

    REQUIRE(o.get_degree() == degree);
    REQUIRE(o.get_level() == level);
    REQUIRE(o.get_write_frequency() == write);
    REQUIRE(o.using_implicit());
    REQUIRE(o.using_full_grid());
    REQUIRE(o.do_poisson_solve());
    REQUIRE(o.get_cfl() == cfl);
    REQUIRE(o.get_selected_pde() == pde);
    REQUIRE(o.is_valid());
  }

  SECTION("run w/ defaults")
  {
    int def_level      = -1;
    int def_degree     = -1;
    int def_num_steps  = 10;
    int def_write_freq = 0;
    int def_vis_freq   = 0;
    bool def_implicit  = false;
    bool def_full_grid = false;
    bool def_poisson   = false;
    double def_cfl     = 0.1;
    PDE_opts def_pde   = PDE_opts::continuity_2;

    options o = make_options({});

    REQUIRE(o.get_degree() == def_degree);
    REQUIRE(o.get_level() == def_level);
    REQUIRE(o.get_time_steps() == def_num_steps);
    REQUIRE(o.get_write_frequency() == def_write_freq);
    REQUIRE(o.get_visualization_frequency() == def_vis_freq);
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

  SECTION("negative cfl")
  {
    std::cerr.setstate(std::ios_base::failbit);
    options o = make_options({"asgard", "-c=-2.0"});
    std::cerr.clear();
    REQUIRE(!o.is_valid());
  }
}
