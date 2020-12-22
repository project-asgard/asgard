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

TEST_CASE("parser constructor/getters", "[program_options]")
{
  SECTION("create from valid string")
  {
    // the golden values
    std::string const pde_choice   = "continuity_3";
    PDE_opts const pde             = PDE_opts::continuity_3;
    std::string const input_levels = "3 4 5";
    int const degree               = 4;
    int const write                = 1;
    int const realspace            = 1;
    double const cfl               = 2.0;
    double const thresh            = 0.1;

    // set up test inputs directly from golden values
    std::cerr.setstate(std::ios_base::failbit);
    parser const p = make_parser(
        {"-p", pde_choice, "-l", input_levels, "-d", std::to_string(degree),
         "-w", std::to_string(write), "-r", std::to_string(realspace), "-f",
         "-i", "-e", "-c", std::to_string(cfl), "--adapt", "--thresh",
         std::to_string(thresh)});
    std::cerr.clear();

    REQUIRE(p.get_degree() == degree);
    REQUIRE(p.get_starting_levels() == fk::vector<int>{3, 4, 5});
    REQUIRE(p.get_wavelet_output_freq() == write);
    REQUIRE(p.get_realspace_output_freq() == realspace);
    REQUIRE(p.using_implicit());
    REQUIRE(p.using_full_grid());
    REQUIRE(p.do_poisson_solve());
    REQUIRE(p.do_adapt_levels());
    REQUIRE(p.get_adapt_thresh() == thresh);
    REQUIRE(p.get_cfl() == cfl);
    REQUIRE(p.get_selected_pde() == pde);
  }

  SECTION("run w/ defaults")
  {
    auto const def_levels         = fk::vector<int>();
    auto const def_degree         = parser::NO_USER_VALUE;
    auto const def_max_level      = parser::DEFAULT_MAX_LEVEL;
    auto const def_num_steps      = parser::DEFAULT_TIME_STEPS;
    auto const def_write_freq     = parser::DEFAULT_WRITE_FREQ;
    auto const def_realspace_freq = parser::DEFAULT_WRITE_FREQ;
    auto const def_implicit       = parser::DEFAULT_USE_IMPLICIT;
    auto const def_full_grid      = parser::DEFAULT_USE_FG;
    auto const def_poisson        = parser::DEFAULT_DO_POISSON;

    auto const def_cfl = parser::DEFAULT_CFL;
    auto const def_dt  = parser::NO_USER_VALUE_FP;
    auto const def_pde = parser::DEFAULT_PDE_OPT;

    auto const def_pde_str   = parser::DEFAULT_PDE_STR;
    auto const def_solve_str = parser::NO_USER_VALUE_STR;

    auto const def_solver = parser::DEFAULT_SOLVER;

    auto const def_adapt     = parser::DEFAULT_DO_ADAPT;
    auto const def_threshold = parser::DEFAULT_ADAPT_THRESH;

    auto const p = make_parser({});

    REQUIRE(p.get_degree() == def_degree);
    REQUIRE(p.get_starting_levels() == def_levels);
    REQUIRE(p.get_max_level() == def_max_level);
    REQUIRE(p.get_time_steps() == def_num_steps);
    REQUIRE(p.get_wavelet_output_freq() == def_write_freq);
    REQUIRE(p.get_realspace_output_freq() == def_realspace_freq);
    REQUIRE(p.using_implicit() == def_implicit);
    REQUIRE(p.using_full_grid() == def_full_grid);
    REQUIRE(p.do_poisson_solve() == def_poisson);
    REQUIRE(p.get_cfl() == def_cfl);
    REQUIRE(p.get_selected_pde() == def_pde);
    REQUIRE(p.get_selected_solver() == def_solver);
    REQUIRE(p.get_pde_string() == def_pde_str);
    REQUIRE(p.get_solver_string() == def_solve_str);
    REQUIRE(p.get_dt() == def_dt);
    REQUIRE(p.get_adapt_thresh() == def_threshold);
    REQUIRE(p.do_adapt_levels() == def_adapt);
    REQUIRE(p.is_valid());
  }

  SECTION("out of range pde")
  {
    std::cerr.setstate(std::ios_base::failbit);
    parser const p = make_parser({"asgard", "-p", "2 1337 4 u gg"});
    std::cerr.clear();
    REQUIRE(!p.is_valid());
  }

  SECTION("out of range solver")
  {
    std::cerr.setstate(std::ios_base::failbit);
    parser const p = make_parser({"asgard", "-s", "2 1337 4 u gg"});
    std::cerr.clear();
    REQUIRE(!p.is_valid());
  }
  SECTION("negative level")
  {
    std::cerr.setstate(std::ios_base::failbit);
    parser const p = make_parser({"asgard", "-l=-2"});
    std::cerr.clear();
    REQUIRE(!p.is_valid());
  }
  SECTION("out of range level, 2nd entry")
  {
    std::cerr.setstate(std::ios_base::failbit);
    parser const p = make_parser({"asgard", "-l=\"2, 0\""});
    std::cerr.clear();
    REQUIRE(!p.is_valid());
  }
  SECTION("negative degree")
  {
    std::cerr.setstate(std::ios_base::failbit);
    parser const p = make_parser({"asgard", "-d=-2"});
    std::cerr.clear();
    REQUIRE(!p.is_valid());
  }
  SECTION("max level < starting level")
  {
    std::cerr.setstate(std::ios_base::failbit);
    parser const p = make_parser({"asgard", "-l=3", "-m=2"});
    std::cerr.clear();
    REQUIRE(!p.is_valid());
  }
  SECTION("negative cfl")
  {
    std::cerr.setstate(std::ios_base::failbit);
    parser const p = make_parser({"asgard", "-c=-2.0"});
    std::cerr.clear();
    REQUIRE(!p.is_valid());
  }
  SECTION("non positive dt")
  {
    std::cerr.setstate(std::ios_base::failbit);
    parser const p = make_parser({"asgard", "-t=-0.0"});
    std::cerr.clear();
    REQUIRE(!p.is_valid());
  }
  SECTION("providing both dt and cfl")
  {
    std::cerr.setstate(std::ios_base::failbit);
    parser const p = make_parser({"asgard", "-t=-1.0", "-c=0.5"});
    std::cerr.clear();
    REQUIRE(!p.is_valid());
  }
  SECTION("providing adapt threshold but disabled adapt")
  {
    std::cerr.setstate(std::ios_base::failbit);
    parser const p = make_parser({"asgard", "--thresh=-0.2"});
    std::cerr.clear();
    REQUIRE(!p.is_valid());
  }
  SECTION("out of range threshold")
  {
    std::cerr.setstate(std::ios_base::failbit);
    parser const p = make_parser({"asgard", "--adapt", "--thresh=-2.0"});
    std::cerr.clear();
    REQUIRE(!p.is_valid());
  }
  SECTION("out of range threshold - neg")
  {
    std::cerr.setstate(std::ios_base::failbit);
    parser const p = make_parser({"asgard", "--adapt", "--thresh=1.1"});
    std::cerr.clear();
    REQUIRE(!p.is_valid());
  }
}
