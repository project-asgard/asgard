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
    prog_opts prog({""});
    REQUIRE_FALSE((prog.show_help and prog.show_pde_help));
  }
  SECTION("no help")
  {
    prog_opts prog({"", "--help"});
    REQUIRE(prog.show_help);
    REQUIRE(prog_opts({"", "-?"}).show_help);
    REQUIRE(prog_opts({"", "-h"}).show_help);
    REQUIRE(prog_opts({"", "-help"}).show_help);
    REQUIRE(prog_opts({"", "help"}).show_help);
    prog_opts prog2({"", "-p?"});
    REQUIRE(prog2.show_pde_help);
    REQUIRE(prog_opts({"", "-pde?"}).show_pde_help);
  }
  SECTION("-step-method")
  {
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-step-method"}), "-step-method must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-s", "dummy"}), "invalid value for -s, see exe -help");
    prog_opts prog({"", "-s", "expl"});
    REQUIRE(prog.step_method);
    REQUIRE(*prog.step_method == time_advance::method::exp);
    prog = prog_opts({"", "-s", "impl"});
    REQUIRE(prog.step_method);
    REQUIRE(*prog.step_method == time_advance::method::imp);
    REQUIRE(prog_opts({"", "-s", "imex"}).step_method.value() == time_advance::method::imex);
  }
  SECTION("-adapt-norm")
  {
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-adapt-norm"}),
                        "-adapt-norm must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-an", "dummy"}), "invalid value for -an, see exe -help");
    prog_opts prog({"", "-an", "l2"});
    REQUIRE(!!prog.anorm);
    REQUIRE(*prog.anorm == adapt_norm::l2);
    prog_opts prog2({"", "-an", "linf"});
    REQUIRE(!!prog2.anorm);
    REQUIRE(*prog2.anorm == adapt_norm::linf);
  }
  SECTION("-grid")
  {
    REQUIRE(prog_opts({"exe", "-grid", "sparse"}).grid);
    REQUIRE(prog_opts({"exe", "-grid", "sparse"}).grid.value() == grid_type::sparse);
    REQUIRE(prog_opts({"exe", "-g", "dense"}).grid.value() == grid_type::dense);
    REQUIRE(prog_opts({"exe", "-g", "mix", "1"}).grid.value() == grid_type::mixed);
    REQUIRE(prog_opts({"exe", "-g", "mixed", "2"}).grid.value() == grid_type::mixed);
    prog_opts opts({"exe", "-g", "mixed", "2"});
    REQUIRE(opts.mgrid_group);
    REQUIRE(opts.mgrid_group.value() == 2);
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-g"}), "-g must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-g", "dummy"}), "invalid value for -g, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-g", "mix"}), "mix must be followed by a value, see exe -help");
    prog_opts opts3({"exe", "-g", "sparse"});
    REQUIRE_FALSE(opts3.mgrid_group);
  }
  SECTION("-electric-solve")
  {
    REQUIRE_FALSE(!!prog_opts({""}).set_electric);
    prog_opts prog({"", "-electric-solve"});
    REQUIRE(prog.set_electric);
    REQUIRE(prog.set_electric.value());
    prog_opts prog2({"", "-es"});
    REQUIRE(prog2.set_electric);
    REQUIRE(prog2.set_electric.value());
  }
  SECTION("-start-levels")
  {
    prog_opts prog({"", "-start-levels", "3 4"});
    REQUIRE_FALSE(prog.start_levels.empty());
    REQUIRE(prog.start_levels.size() == 2);
    auto const &arr = prog.start_levels;
    REQUIRE(arr[0] == 3);
    REQUIRE(arr[1] == 4);
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-l"}), "-l must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-l", "\"\""}), "invalid value for -l, see exe -help");
    // the test checks if guard agains overflor over max_num_dimensions
    // must be updated if we ever go above 6 dimensions
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-l", "1 1 1 1 1 1 1"}), "invalid value for -l, see exe -help");
  }
  SECTION("-max-levels")
  {
    prog_opts prog({"", "-max-levels", "9 8 3"});
    REQUIRE_FALSE(prog.max_levels.empty());
    REQUIRE(prog.max_levels.size() == 3);
    auto const &arr = prog.max_levels;
    REQUIRE(arr[0] == 9);
    REQUIRE(arr[1] == 8);
    REQUIRE(arr[2] == 3);
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-m"}), "-m must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-m", "\"\""}), "invalid value for -m, see exe -help");
  }
  SECTION("-degree")
  {
    prog_opts prog({"", "-degree", "2"});
    REQUIRE(prog.degree);
    REQUIRE(prog.degree.value() == 2);
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-d"}), "-d must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-d", "dummy"}), "invalid value for -d, see exe -help");
    // checks for out-of-range overflow
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-d", "8100100100"}), "invalid value for -d, see exe -help");
  }
  SECTION("-num_steps")
  {
    prog_opts prog({"", "-num-steps", "2"});
    REQUIRE(prog.num_time_steps);
    REQUIRE(prog.num_time_steps.value() == 2);
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-m"}), "-m must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-num-steps", "dummy"}), "invalid value for -num-steps, see exe -help");
  }
  SECTION("-wave_freq")
  {
    prog_opts prog({"", "-wave-freq", "10"});
    REQUIRE(prog.wavelet_output_freq);
    REQUIRE(prog.wavelet_output_freq.value() == 10);
    REQUIRE(prog_opts({"exe", "-w", "4"}).wavelet_output_freq);
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-w"}), "-w must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-wave-freq", "dummy"}), "invalid value for -wave-freq, see exe -help");
  }
  SECTION("-real_freq")
  {
    prog_opts prog({"", "-real-freq", "12"});
    REQUIRE(prog.realspace_output_freq);
    REQUIRE(prog.realspace_output_freq.value() == 12);
    REQUIRE(prog_opts({"exe", "-r", "9"}).realspace_output_freq);
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-r"}), "-r must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-real-freq", "dummy"}), "invalid value for -real-freq, see exe -help");
  }
  SECTION("-dt")
  {
    prog_opts prog({"", "-dt", "0.5"});
    REQUIRE(prog.dt);
    REQUIRE(prog.dt.value() == 0.5);
    REQUIRE(prog_opts({"exe", "-dt", "0.1"}).dt);
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-dt"}), "-dt must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-dt", "dummy"}), "invalid value for -dt, see exe -help");
  }
  SECTION("-adapt")
  {
    prog_opts prog({"", "-adapt", "0.5"});
    REQUIRE(prog.adapt_threshold);
    REQUIRE(prog.adapt_threshold.value() == 0.5);
    REQUIRE(prog_opts({"exe", "-adapt", "0.1"}).adapt_threshold);
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-a"}), "-a must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-adapt", "dummy"}), "invalid value for -adapt, see exe -help");
  }
  SECTION("-solver")
  {
    prog_opts prog({"", "-solver", "direct"});
    REQUIRE(prog.solver);
    REQUIRE(prog.solver.value() == solve_opts::direct);
    REQUIRE(prog_opts({"exe", "-sv", "gmres"}).solver.value() == solve_opts::gmres);
    REQUIRE(prog_opts({"exe", "-solver", "bicgstab"}).solver.value() == solve_opts::bicgstab);
    REQUIRE(prog_opts({"exe", "-sv", "scalapack"}).solver.value() == solve_opts::scalapack);
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-solver", "dummy"}), "invalid value for -solver, see exe -help");
  }
  SECTION("-memory")
  {
    prog_opts prog({"", "-memory", "1024"});
    REQUIRE(prog.memory_limit);
    REQUIRE(prog.memory_limit.value() == 1024);
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-memory"}), "-memory must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-memory", "dummy"}), "invalid value for -memory, see exe -help");
  }
  SECTION("-kron-mode")
  {
    prog_opts prog({"", "-kron-mode", "dense"});
    REQUIRE(prog.kron_mode);
    REQUIRE(prog.kron_mode.value() == kronmult_mode::dense);
    REQUIRE(prog_opts({"", "-kron-mode", "sparse"}).kron_mode.value() == kronmult_mode::sparse);
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-kron-mode"}),
                        "-kron-mode must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-kron-mode", "dummy"}),
                        "invalid value for -kron-mode, see exe -help");
  }
  SECTION("-isolve_tol")
  {
    prog_opts prog({"", "-isolve-tol", "0.25"});
    REQUIRE(prog.isolver_tolerance);
    REQUIRE(prog.isolver_tolerance.value() == 0.25);
    REQUIRE(prog_opts({"exe", "-ist", "0.1"}).isolver_tolerance);
    REQUIRE(prog_opts({"exe", "-isolve-tol", "0.01"}).isolver_tolerance.value() < 0.02);
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-isolve-tol"}),
                        "-isolve-tol must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-ist", "dummy"}),
                        "invalid value for -ist, see exe -help");
  }
  SECTION("-isolve_iter")
  {
    prog_opts prog({"", "-isolve-iter", "100"});
    REQUIRE(prog.isolver_iterations);
    REQUIRE(prog.isolver_iterations.value() == 100);
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-isolve-iter"}),
                        "-isolve-iter must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-isi", "dummy"}),
                        "invalid value for -isi, see exe -help");
  }
  SECTION("-isolve_outer")
  {
    prog_opts prog({"", "-isolve-outer", "200"});
    REQUIRE(prog.isolver_outer_iterations);
    REQUIRE(prog.isolver_outer_iterations.value() == 200);
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-isolve-outer"}),
                        "-isolve-outer must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-iso", "dummy"}),
                        "invalid value for -iso, see exe -help");
  }
  SECTION("-pde")
  {
    prog_opts prog({"", "-pde", "continuity_6"});
    REQUIRE(prog.pde_choice);
    REQUIRE(prog.pde_choice.value() == PDE_opts::continuity_6);
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-pde"}),
                        "-pde must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-p", "dummy"}),
                        "invalid pde 'dummy', see 'exe -pde?' for full list");
  }
  SECTION("-title")
  {
    prog_opts prog({"", "-title", "mypde"});
    REQUIRE_FALSE(prog.title.empty());
    REQUIRE(prog.title == "mypde");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-title"}),
                        "-title must be followed by a value, see exe -help");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-title", ""}),
                        "invalid value for -title, see exe -help");
    prog_opts prog2({"", "-title", "continuity_6"});
    REQUIRE_FALSE(prog2.title.empty());
    REQUIRE(prog2.title == "continuity_6");
  }
  SECTION("-subtitle")
  {
    prog_opts prog({"", "-subtitle", "mypde-variant"});
    REQUIRE_FALSE(prog.subtitle.empty());
    REQUIRE(prog.subtitle == "mypde-variant");
    REQUIRE_THROWS_WITH(prog_opts({"exe", "-subtitle"}),
                        "-subtitle must be followed by a value, see exe -help");
    REQUIRE(prog_opts({"exe", "-subtitle", "dummy", "-subtitle", ""}).subtitle.empty());
  }
}
