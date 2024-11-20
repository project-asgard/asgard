#include "tests_general.hpp"

using namespace asgard;

struct distribution_test_init
{
  distribution_test_init() { initialize_distribution(); }
  ~distribution_test_init() { finalize_distribution(); }
};

#ifdef ASGARD_USE_MPI
static distribution_test_init const distrib_test_info;
#endif

TEST_CASE("new program options", "[single options]")
{
  SECTION("no opts")
  {
    prog_opts prog(vecstrview({""}));
    REQUIRE_FALSE((prog.show_help and prog.show_pde_help));
  }
  SECTION("no help")
  {
    prog_opts prog(vecstrview({"", "--help"}));
    REQUIRE(prog.show_help);
    REQUIRE(prog_opts(vecstrview({"", "-?"})).show_help);
    REQUIRE(prog_opts(vecstrview({"", "-h"})).show_help);
    REQUIRE(prog_opts(vecstrview({"", "-help"})).show_help);
    REQUIRE(prog_opts(vecstrview({"", "help"})).show_help);
    prog_opts prog2(vecstrview({"", "-p?"}));
    REQUIRE(prog2.show_pde_help);
    REQUIRE(prog_opts(vecstrview({"", "-pde?"})).show_pde_help);
  }
  SECTION("-step-method")
  {
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-step-method"})),
                        "-step-method must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-s", "dummy"})),
                        "invalid value for -s, see exe -help");
    prog_opts prog(vecstrview({"", "-s", "expl"}));
    REQUIRE(prog.step_method);
    REQUIRE(*prog.step_method == time_advance::method::exp);
    prog = prog_opts(vecstrview({"", "-s", "impl"}));
    REQUIRE(prog.step_method);
    REQUIRE(*prog.step_method == time_advance::method::imp);
    REQUIRE(prog_opts(vecstrview({"", "-s", "imex"})).step_method.value()
            == time_advance::method::imex);
  }
  SECTION("-adapt-norm")
  {
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-adapt-norm"})),
                        "-adapt-norm must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-an", "dummy"})),
                        "invalid value for -an, see exe -help");
    prog_opts prog(vecstrview({"", "-an", "l2"}));
    REQUIRE(!!prog.anorm);
    REQUIRE(*prog.anorm == adapt_norm::l2);
    prog_opts prog2(vecstrview({"", "-an", "linf"}));
    REQUIRE(!!prog2.anorm);
    REQUIRE(*prog2.anorm == adapt_norm::linf);
  }
  SECTION("-grid")
  {
    REQUIRE(prog_opts(vecstrview({"exe", "-grid", "sparse"})).grid);
    REQUIRE(prog_opts(vecstrview({"exe", "-grid", "sparse"})).grid.value() == grid_type::sparse);
    REQUIRE(prog_opts(vecstrview({"exe", "-g", "dense"})).grid.value() == grid_type::dense);
    REQUIRE(prog_opts(vecstrview({"exe", "-g", "mix", "1"})).grid.value() == grid_type::mixed);
    REQUIRE(prog_opts(vecstrview({"exe", "-g", "mixed", "2"})).grid.value() == grid_type::mixed);
    prog_opts opts(vecstrview({"exe", "-g", "mixed", "2"}));
    REQUIRE(opts.mgrid_group);
    REQUIRE(opts.mgrid_group.value() == 2);
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-g"})),
                        "-g must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-g", "dummy"})),
                        "invalid value for -g, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-g", "mix"})),
                        "missing mixed grid number, see exe -help");
    prog_opts opts3(vecstrview({"exe", "-g", "sparse"}));
    REQUIRE_FALSE(opts3.mgrid_group);
  }
  SECTION("-electric-solve")
  {
    REQUIRE_FALSE(!!prog_opts(vecstrview({""})).set_electric);
    prog_opts prog(vecstrview({"", "-electric-solve"}));
    REQUIRE(prog.set_electric);
    REQUIRE(prog.set_electric.value());
    prog_opts prog2(vecstrview({"", "-es"}));
    REQUIRE(prog2.set_electric);
    REQUIRE(prog2.set_electric.value());
  }
  SECTION("-start-levels")
  {
    prog_opts prog(vecstrview({"", "-start-levels", "3 4"}));
    REQUIRE_FALSE(prog.start_levels.empty());
    REQUIRE(prog.start_levels.size() == 2);
    auto const &arr = prog.start_levels;
    REQUIRE(arr[0] == 3);
    REQUIRE(arr[1] == 4);
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-l"})),
                        "-l must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-l", "\"\""})),
                        "invalid value for -l, see exe -help");
    // the test checks if guard agains overflor over max_num_dimensions
    // must be updated if we ever go above 6 dimensions
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-l", "1 1 1 1 1 1 1"})),
                        "invalid value for -l, see exe -help");
  }
  SECTION("-max-levels")
  {
    prog_opts prog(vecstrview({"", "-max-levels", "9 8 3"}));
    REQUIRE_FALSE(prog.max_levels.empty());
    REQUIRE(prog.max_levels.size() == 3);
    auto const &arr = prog.max_levels;
    REQUIRE(arr[0] == 9);
    REQUIRE(arr[1] == 8);
    REQUIRE(arr[2] == 3);
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-m"})),
                        "-m must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-m", "\"\""})),
                        "invalid value for -m, see exe -help");
  }
  SECTION("-degree")
  {
    prog_opts prog(vecstrview({"", "-degree", "2"}));
    REQUIRE(prog.degree);
    REQUIRE(prog.degree.value() == 2);
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-d"})),
                        "-d must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-d", "dummy"})),
                        "invalid value for -d, see exe -help");
    // checks for out-of-range overflow
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-d", "8100100100"})),
                        "invalid value for -d, see exe -help");
  }
  SECTION("-num_steps")
  {
    prog_opts prog(vecstrview({"", "-num-steps", "2"}));
    REQUIRE(prog.num_time_steps);
    REQUIRE(prog.num_time_steps.value() == 2);
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-m"})),
                        "-m must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-num-steps", "dummy"})),
                        "invalid value for -num-steps, see exe -help");
  }
  SECTION("-wave_freq")
  {
    prog_opts prog(vecstrview({"", "-wave-freq", "10"}));
    REQUIRE(prog.wavelet_output_freq);
    REQUIRE(prog.wavelet_output_freq.value() == 10);
    REQUIRE(prog_opts(vecstrview({"exe", "-w", "4"})).wavelet_output_freq);
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-w"})),
                        "-w must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-wave-freq", "dummy"})),
                        "invalid value for -wave-freq, see exe -help");
  }
  SECTION("-dt")
  {
    prog_opts prog(vecstrview({"", "-dt", "0.5"}));
    REQUIRE(prog.dt);
    REQUIRE(prog.dt.value() == 0.5);
    REQUIRE(prog_opts(vecstrview({"exe", "-dt", "0.1"})).dt);
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-dt"})),
                        "-dt must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-dt", "dummy"})),
                        "invalid value for -dt, see exe -help");
  }
  SECTION("-adapt")
  {
    prog_opts prog(vecstrview({"", "-adapt", "0.5"}));
    REQUIRE(prog.adapt_threshold);
    REQUIRE(prog.adapt_threshold.value() == 0.5);
    REQUIRE(prog_opts(vecstrview({"exe", "-adapt", "0.1"})).adapt_threshold);
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-a"})),
                        "-a must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-adapt", "dummy"})),
                        "invalid value for -adapt, see exe -help");
  }
  SECTION("-solver")
  {
    prog_opts prog(vecstrview({"", "-solver", "direct"}));
    REQUIRE(prog.solver);
    REQUIRE(prog.solver.value() == solve_opts::direct);
    REQUIRE(prog_opts(vecstrview({"exe", "-sv", "gmres"})).solver.value() == solve_opts::gmres);
    REQUIRE(prog_opts(vecstrview({"exe", "-solver", "bicgstab"})).solver.value() == solve_opts::bicgstab);
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-solver", "dummy"})),
                        "invalid value for -solver, see exe -help");
  }
  SECTION("-memory")
  {
    prog_opts prog(vecstrview({"", "-memory", "1024"}));
    REQUIRE(prog.memory_limit);
    REQUIRE(prog.memory_limit.value() == 1024);
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-memory"})),
                        "-memory must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-memory", "dummy"})),
                        "invalid value for -memory, see exe -help");
  }
  SECTION("-kron-mode")
  {
    prog_opts prog(vecstrview({"", "-kron-mode", "dense"}));
    REQUIRE(prog.kron_mode);
    REQUIRE(prog.kron_mode.value() == kronmult_mode::dense);
    REQUIRE(prog_opts(vecstrview({"", "-kron-mode", "sparse"})).kron_mode.value()
            == kronmult_mode::sparse);
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-kron-mode"})),
                        "-kron-mode must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-kron-mode", "dummy"})),
                        "invalid value for -kron-mode, see exe -help");
  }
  SECTION("-isolve_tol")
  {
    prog_opts prog(vecstrview({"", "-isolve-tol", "0.25"}));
    REQUIRE(prog.isolver_tolerance);
    REQUIRE(prog.isolver_tolerance.value() == 0.25);
    REQUIRE(prog_opts(vecstrview({"exe", "-ist", "0.1"})).isolver_tolerance);
    REQUIRE(prog_opts(vecstrview({"exe", "-isolve-tol", "0.01"})).isolver_tolerance.value() < 0.02);
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-isolve-tol"})),
                        "-isolve-tol must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-ist", "dummy"})),
                        "invalid value for -ist, see exe -help");
  }
  SECTION("-isolve_iter")
  {
    prog_opts prog(vecstrview({"", "-isolve-iter", "100"}));
    REQUIRE(prog.isolver_iterations);
    REQUIRE(prog.isolver_iterations.value() == 100);
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-isolve-iter"})),
                        "-isolve-iter must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-isi", "dummy"})),
                        "invalid value for -isi, see exe -help");
  }
  SECTION("-isolve_outer")
  {
    prog_opts prog(vecstrview({"", "-isolve-outer", "200"}));
    REQUIRE(prog.isolver_outer_iterations);
    REQUIRE(prog.isolver_outer_iterations.value() == 200);
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-isolve-outer"})),
                        "-isolve-outer must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-iso", "dummy"})),
                        "invalid value for -iso, see exe -help");
  }
  SECTION("-pde")
  {
    prog_opts prog(vecstrview({"", "-pde", "continuity_6"}));
    REQUIRE(prog.pde_choice);
    REQUIRE(prog.pde_choice.value() == PDE_opts::continuity_6);
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-pde"})),
                        "-pde must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-p", "dummy"})),
                        "invalid pde 'dummy', see 'exe -pde?' for full list");
  }
  SECTION("-title")
  {
    prog_opts prog(vecstrview({"", "-title", "mypde"}));
    REQUIRE_FALSE(prog.title.empty());
    REQUIRE(prog.title == "mypde");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-title"})),
                        "-title must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-title", ""})),
                        "invalid value for -title, see exe -help");
    prog_opts prog2(vecstrview({"", "-title", "continuity_6"}));
    REQUIRE_FALSE(prog2.title.empty());
    REQUIRE(prog2.title == "continuity_6");
  }
  SECTION("-subtitle")
  {
    prog_opts prog(vecstrview({"", "-subtitle", "mypde-variant"}));
    REQUIRE_FALSE(prog.subtitle.empty());
    REQUIRE(prog.subtitle == "mypde-variant");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-subtitle"})),
                        "-subtitle must be followed by a value, see exe -help");
    REQUIRE(prog_opts(vecstrview({"exe", "-subtitle", "dummy", "-subtitle", ""})).subtitle.empty());
  }
  SECTION("-outfile")
  {
    prog_opts prog(vecstrview({"", "-outfile", "some-file"}));
    REQUIRE_FALSE(prog.outfile.empty());
    REQUIRE(prog.outfile == "some-file");
    REQUIRE_THROWS_WITH(prog_opts(vecstrview({"exe", "-of"})),
                        "-of must be followed by a value, see exe -help");
    REQUIRE(prog_opts(vecstrview({"exe", "-outfile", "dummy", "-of", ""})).subtitle.empty());
  }
}

TEST_CASE("input file processing", "[file i/o]")
{
  SECTION("test_input1.txt")
  {
    prog_opts prog(vecstrview({"", "-l", "3", "-if", "test_input1.txt"}));
    REQUIRE_FALSE(prog.start_levels.empty());
    REQUIRE(prog.start_levels[0] == 5);
    REQUIRE(prog.adapt_threshold);
    REQUIRE(prog.adapt_threshold.value() == 0.125);
    REQUIRE(prog.grid);
    REQUIRE(prog.grid.value() == grid_type::dense);
    REQUIRE(prog.step_method);
    REQUIRE(prog.step_method.value() == time_advance::method::exp);

    REQUIRE_FALSE(prog.file_value<int>("missing"));
    auto bbool = prog.file_value<bool>("bb1");
    static_assert(std::is_same_v<decltype(bbool), std::optional<bool>>);
    REQUIRE(bbool);
    REQUIRE(bbool.value());
    auto iint = prog.file_value<int>("some_int");
    static_assert(std::is_same_v<decltype(iint), std::optional<int>>);
    REQUIRE(iint);
    REQUIRE(iint.value() == 8);

    prog_opts prog2(vecstrview({"", "-infile", "test_input1.txt", "-l", "3"}));
    REQUIRE_FALSE(prog2.start_levels.empty());
    REQUIRE(prog2.start_levels[0] == 3);
  }

  SECTION("test_input2.txt")
  {
    prog_opts prog(vecstrview({"", "-if", "test_input2.txt"}));
    REQUIRE(prog.start_levels.size() == 4);
    REQUIRE(prog.start_levels[0] ==  9);
    REQUIRE(prog.start_levels[1] == 11);
    REQUIRE(prog.start_levels[2] ==  1);
    REQUIRE(prog.start_levels[3] == 88);
    REQUIRE(prog.title == "some long title");
    REQUIRE(prog.subtitle == "short title");

    auto dbl = prog.file_value<double>("v_thermal");
    static_assert(std::is_same_v<decltype(dbl), std::optional<double>>);
    REQUIRE(dbl);
    REQUIRE(dbl.value() == 0.5);

    auto flt = prog.file_value<float>("half percent");
    static_assert(std::is_same_v<decltype(flt), std::optional<float>>);
    REQUIRE(flt);
    REQUIRE(std::abs(flt.value() - 5.E-3) < 1.E-6);

    REQUIRE_FALSE(prog.file_value<float>("misspelled"));

    auto name = prog.file_value<std::string>("extra name");
    static_assert(std::is_same_v<decltype(name), std::optional<std::string>>);
    REQUIRE(!!name);
    REQUIRE(name.value() == "some-name test");

    auto nu = prog.file_value<int>("var nu");
    static_assert(std::is_same_v<decltype(nu), std::optional<int>>);
    REQUIRE(!!nu);
    REQUIRE(nu.value() == 314);
  }

  SECTION("test_input1.txt -- direct ")
  {
    prog_opts prog("test_input1.txt");
    REQUIRE(prog.start_levels.size() == 1);
    REQUIRE(prog.start_levels[0] == 5);
    REQUIRE(prog.adapt_threshold);
    REQUIRE(prog.adapt_threshold.value() == 0.125);

    auto iint = prog.file_value<int>("some_int");
    static_assert(std::is_same_v<decltype(iint), std::optional<int>>);
    REQUIRE(iint);
    REQUIRE(iint.value() == 8);
  }
}
