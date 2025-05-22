#include "asgard_test_macros.hpp"

using namespace asgard;

void new_prog_opts() {
  {
    current_test name_("no opts");
    prog_opts prog(vecstrview({""}));
    tassert(not prog.show_help);
  }{
    current_test name_("no help");
    prog_opts prog(vecstrview({"", "--help"}));
    tassert(prog.show_help);
    tassert(prog_opts(vecstrview({"", "-?"})).show_help);
    tassert(prog_opts(vecstrview({"", "-h"})).show_help);
    tassert(prog_opts(vecstrview({"", "-help"})).show_help);
    tassert(prog_opts(vecstrview({"", "help"})).show_help);
  }{
    current_test name_("-step-method");
    terror_message(prog_opts(vecstrview({"exe", "-step-method"})),
                   "-step-method must be followed by a value, see exe -help");
    terror_message(prog_opts(vecstrview({"exe", "-s", "dummy"})),
                   "invalid value for -s, see exe -help");

    tassert(prog_opts(vecstrview({"exe", "-s", "steady"})).step_method);
    tassert(prog_opts(vecstrview({"exe", "-s", "steady"})).step_method.value() == time_method::steady);
    tassert(prog_opts(vecstrview({"exe", "-s", "fe"})).step_method);
    tassert(prog_opts(vecstrview({"exe", "-s", "fe"})).step_method.value() == time_method::forward_euler);
    tassert(prog_opts(vecstrview({"exe", "-s", "forward-euler"})).step_method);
    tassert(prog_opts(vecstrview({"exe", "-s", "forward-euler"})).step_method.value() == time_method::forward_euler);
    tassert(prog_opts(vecstrview({"exe", "-s", "rk2"})).step_method);
    tassert(prog_opts(vecstrview({"exe", "-s", "rk2"})).step_method.value() == time_method::rk2);
    tassert(prog_opts(vecstrview({"exe", "-s", "rk3"})).step_method);
    tassert(prog_opts(vecstrview({"exe", "-s", "rk3"})).step_method.value() == time_method::rk3);
    tassert(prog_opts(vecstrview({"exe", "-s", "rk4"})).step_method);
    tassert(prog_opts(vecstrview({"exe", "-s", "rk4"})).step_method.value() == time_method::rk4);
    tassert(prog_opts(vecstrview({"exe", "-s", "cn"})).step_method);
    tassert(prog_opts(vecstrview({"exe", "-s", "cn"})).step_method.value() == time_method::cn);
    tassert(prog_opts(vecstrview({"exe", "-s", "crank-nicolson"})).step_method.value() == time_method::cn);
    tassert(prog_opts(vecstrview({"exe", "-s", "be"})).step_method);
    tassert(prog_opts(vecstrview({"exe", "-s", "be"})).step_method.value() == time_method::back_euler);
    tassert(prog_opts(vecstrview({"exe", "-s", "backward-euler"})).step_method.value() == time_method::back_euler);
    tassert(prog_opts(vecstrview({"exe", "-s", "imex1"})).step_method);
    tassert(prog_opts(vecstrview({"exe", "-s", "imex1"})).step_method.value() == time_method::imex1);
    tassert(prog_opts(vecstrview({"exe", "-s", "imex2"})).step_method);
    tassert(prog_opts(vecstrview({"exe", "-s", "imex2"})).step_method.value() == time_method::imex2);
  }{
    current_test name_("-grid");
    tassert(prog_opts(vecstrview({"exe", "-grid", "sparse"})).grid);
    tassert(prog_opts(vecstrview({"exe", "-grid", "sparse"})).grid.value() == grid_type::sparse);
    tassert(prog_opts(vecstrview({"exe", "-g", "dense"})).grid.value() == grid_type::dense);
    tassert(prog_opts(vecstrview({"exe", "-g", "mix", "1"})).grid.value() == grid_type::mixed);
    tassert(prog_opts(vecstrview({"exe", "-g", "mixed", "2"})).grid.value() == grid_type::mixed);
    prog_opts opts(vecstrview({"exe", "-g", "mixed", "2"}));
    tassert(opts.mgrid_group);
    tassert(opts.mgrid_group.value() == 2);
    terror_message(prog_opts(vecstrview({"exe", "-g"})),
                   "-g must be followed by a value, see exe -help");
    terror_message(prog_opts(vecstrview({"exe", "-g", "dummy"})),
                   "invalid value for -g, see exe -help");
    terror_message(prog_opts(vecstrview({"exe", "-g", "mix"})),
                   "missing mixed grid number, see exe -help");
    prog_opts opts3(vecstrview({"exe", "-g", "sparse"}));
    tassert(not opts3.mgrid_group);
  }{
    current_test name_("-start-levels");
    prog_opts prog(vecstrview({"", "-start-levels", "3 4"}));
    tassert(not prog.start_levels.empty());
    tassert(prog.start_levels.size() == 2);
    auto const &arr = prog.start_levels;
    tassert(arr[0] == 3);
    tassert(arr[1] == 4);
    terror_message(prog_opts(vecstrview({"exe", "-l"})),
                   "-l must be followed by a value, see exe -help");
    terror_message(prog_opts(vecstrview({"exe", "-l", "\"\""})),
                   "invalid value for -l, see exe -help");
    // the test checks if guard agains overflor over max_num_dimensions
    // must be updated if we ever go above 6 dimensions
    terror_message(prog_opts(vecstrview({"exe", "-l", "1 1 1 1 1 1 1"})),
                   "invalid value for -l, see exe -help");
  }{
    current_test name_("-max-levels");
    prog_opts prog(vecstrview({"", "-max-levels", "9 8 3"}));
    tassert(not prog.max_levels.empty());
    tassert(prog.max_levels.size() == 3);
    auto const &arr = prog.max_levels;
    tassert(arr[0] == 9);
    tassert(arr[1] == 8);
    tassert(arr[2] == 3);
    terror_message(prog_opts(vecstrview({"exe", "-m"})),
                   "-m must be followed by a value, see exe -help");
    terror_message(prog_opts(vecstrview({"exe", "-m", "\"\""})),
                   "invalid value for -m, see exe -help");
  }{

    prog_opts prog(vecstrview({"", "-degree", "2"}));
    tassert(prog.degree);
    tassert(prog.degree.value() == 2);
    terror_message(prog_opts(vecstrview({"exe", "-d"})),
                   "-d must be followed by a value, see exe -help");
    terror_message(prog_opts(vecstrview({"exe", "-d", "dummy"})),
                   "invalid value for -d, see exe -help");
    // checks for out-of-range overflow
    terror_message(prog_opts(vecstrview({"exe", "-d", "8100100100"})),
                   "invalid value for -d, see exe -help");
  }{
    current_test name_("-num_steps");
    prog_opts prog(vecstrview({"", "-num-steps", "2"}));
    tassert(prog.num_time_steps);
    tassert(prog.num_time_steps.value() == 2);
    terror_message(prog_opts(vecstrview({"exe", "-m"})),
                   "-m must be followed by a value, see exe -help");
    terror_message(prog_opts(vecstrview({"exe", "-num-steps", "dummy"})),
                   "invalid value for -num-steps, see exe -help");
  }{
    current_test name_("-dt");
    prog_opts prog(vecstrview({"", "-dt", "0.5"}));
    tassert(prog.dt);
    tassert(prog.dt.value() == 0.5);
    tassert(prog_opts(vecstrview({"exe", "-dt", "0.1"})).dt);
    terror_message(prog_opts(vecstrview({"exe", "-dt"})),
                   "-dt must be followed by a value, see exe -help");
    terror_message(prog_opts(vecstrview({"exe", "-dt", "dummy"})),
                   "invalid value for -dt, see exe -help");
  }{
    current_test name_("-time");
    prog_opts prog(vecstrview({"", "-time", "2.5"}));
    tassert(prog.stop_time);
    tassert(prog.stop_time.value() == 2.5);
    tassert(prog_opts(vecstrview({"exe", "-t", "0.1"})).stop_time);
    terror_message(prog_opts(vecstrview({"exe", "-time"})),
                   "-time must be followed by a value, see exe -help");
    terror_message(prog_opts(vecstrview({"exe", "-t", "dummy"})),
                   "invalid value for -t, see exe -help");
  }{
    current_test name_("-adapt");
    prog_opts prog(vecstrview({"", "-adapt", "0.5"}));
    tassert(prog.adapt_threshold);
    tassert(prog.adapt_threshold.value() == 0.5);
    tassert(prog_opts(vecstrview({"exe", "-adapt", "0.1"})).adapt_threshold);
    tassert(not prog_opts(vecstrview({"exe", "-adapt", "0.1", "-noadapt"})).adapt_threshold);
    terror_message(prog_opts(vecstrview({"exe", "-a"})),
                   "-a must be followed by a value, see exe -help");
    terror_message(prog_opts(vecstrview({"exe", "-adapt", "dummy"})),
                   "invalid value for -adapt, see exe -help");
  }{
    current_test name_("-solver");
    prog_opts prog(vecstrview({"", "-solver", "direct"}));
    tassert(prog.solver);
    tassert(prog.solver.value() == solver_method::direct);
    tassert(prog_opts(vecstrview({"exe", "-sv", "gmres"})).solver.value() == solver_method::gmres);
    tassert(prog_opts(vecstrview({"exe", "-solver", "bicgstab"})).solver.value() == solver_method::bicgstab);
    terror_message(prog_opts(vecstrview({"exe", "-solver", "dummy"})),
                   "invalid value for -solver, see exe -help");

    prog = prog_opts(vecstrview({"", "-dt", "0.1"}));
    tassert(not prog.solver);
    prog.force_solver(solver_method::gmres);
    tassert(prog.solver);
    tassert(*prog.solver == solver_method::gmres);
  }{
    current_test name_("-isolve_tol");
    prog_opts prog(vecstrview({"", "-isolve-tol", "0.25"}));
    tassert(prog.isolver_tolerance);
    tassert(prog.isolver_tolerance.value() == 0.25);
    tassert(prog_opts(vecstrview({"exe", "-ist", "0.1"})).isolver_tolerance);
    tassert(prog_opts(vecstrview({"exe", "-isolve-tol", "0.01"})).isolver_tolerance.value() < 0.02);
    terror_message(prog_opts(vecstrview({"exe", "-isolve-tol"})),
                   "-isolve-tol must be followed by a value, see exe -help");
    terror_message(prog_opts(vecstrview({"exe", "-ist", "dummy"})),
                   "invalid value for -ist, see exe -help");
  }{
    current_test name_("-isolve_iter");
    prog_opts prog(vecstrview({"", "-isolve-iter", "100"}));
    tassert(prog.isolver_iterations);
    tassert(prog.isolver_iterations.value() == 100);
    terror_message(prog_opts(vecstrview({"exe", "-isolve-iter"})),
                   "-isolve-iter must be followed by a value, see exe -help");
    terror_message(prog_opts(vecstrview({"exe", "-isi", "dummy"})),
                   "invalid value for -isi, see exe -help");
  }{
    current_test name_("-isolve_inner");
    prog_opts prog(vecstrview({"", "-isolve-inner", "200"}));
    tassert(prog.isolver_inner_iterations);
    tassert(prog.isolver_inner_iterations.value() == 200);
    terror_message(prog_opts(vecstrview({"exe", "-isolve-inner"})),
                   "-isolve-inner must be followed by a value, see exe -help");
    terror_message(prog_opts(vecstrview({"exe", "-isn", "dummy"})),
                   "invalid value for -isn, see exe -help");
  }{
    current_test name_("-title");
    prog_opts prog(vecstrview({"", "-title", "mypde"}));
    tassert(not prog.title.empty());
    tassert(prog.title == "mypde");
    terror_message(prog_opts(vecstrview({"exe", "-title"})),
                   "-title must be followed by a value, see exe -help");
    terror_message(prog_opts(vecstrview({"exe", "-title", ""})),
                   "invalid value for -title, see exe -help");
    prog.set_default_title("otherpde");
    tassert(prog.title == "mypde");

    prog_opts prog2(vecstrview({"", "-title", "continuity-6"}));
    tassert(not prog2.title.empty());
    tassert(prog2.title == "continuity-6");

    prog_opts prog3(vecstrview({"exe",}));
    tassert(prog3.title.empty());
    prog3.set_default_title("something");
    tassert(prog3.title == "something");
  }{
    current_test name_("-subtitle");
    prog_opts prog(vecstrview({"", "-subtitle", "mypde-variant"}));
    tassert(not prog.subtitle.empty());
    tassert(prog.subtitle == "mypde-variant");
    terror_message(prog_opts(vecstrview({"exe", "-subtitle"})),
                   "-subtitle must be followed by a value, see exe -help");
    tassert(prog_opts(vecstrview({"exe", "-subtitle", "dummy", "-subtitle", ""})).subtitle.empty());
  }{
    current_test name_("-verbosity");
    prog_opts prog(vecstrview({"", "-verbosity", "0"}));
    tassert(prog.verbosity);
    tassert(prog.verbosity.value() == verbosity_level::quiet);
    terror_message(prog_opts(vecstrview({"exe", "-verbosity"})),
                   "-verbosity must be followed by a value, see exe -help");
    terror_message(prog_opts(vecstrview({"exe", "-verbosity", "wrong"})),
                   "invalid value for -verbosity, see exe -help");
    tassert(prog_opts(vecstrview({"exe", "-verbosity", "1"})).verbosity.value() == verbosity_level::low);
    tassert(prog_opts(vecstrview({"exe", "-vv", "high"})).verbosity.value() == verbosity_level::high);
  }{
    current_test name_("-outfile");
    prog_opts prog(vecstrview({"", "-outfile", "some-file"}));
    tassert(not prog.outfile.empty());
    tassert(prog.outfile == "some-file");
    terror_message(prog_opts(vecstrview({"exe", "-of"})),
                        "-of must be followed by a value, see exe -help");
    tassert(prog_opts(vecstrview({"exe", "-outfile", "dummy", "-of", ""})).subtitle.empty());
  }{
    current_test name_("-view");
    prog_opts prog(vecstrview({"", "-view", "some-view"}));
    tassert(not prog.default_plotter_view.empty());
    tassert(prog.default_plotter_view == "some-view");
    terror_message(prog_opts(vecstrview({"exe", "-view"})),
                   "-view must be followed by a value, see exe -help");
  }{
    current_test name_("file_required");
    terror_message(prog_opts(vecstrview({"exe", "-ist", "0.1"})).file_required<int>("none"),
                   "missing an input file with required entry 'none'");
  }{
    current_test name_("extra_cli_value_group");
    prog_opts options(vecstrview({"exe", "-nu", "42", "-test"}));
    auto nu = options.extra_cli_value_group<double>({"-nu",});
    tassert((std::is_same_v<decltype(nu), std::optional<double>>));
    tassert(nu.value_or(0) == 42.0);
    tassert(options.has_cli_entry("-test"));

    options.throw_if_argv_not_in({"-test", }, {"-nu",}); // should not throw
    terror_message(options.throw_if_argv_not_in({"-test", }, {}),
                   "unknown command line argument(s) encountered\n-nu\n42");
  }{
    current_test name_("make_opts");
    prog_opts const null_opts;
    tassert(null_opts.start_levels.empty());
    tassert(not null_opts.solver);
    prog_opts const parsed = make_opts("-l 5 -sv gmres");
    tassert(not parsed.start_levels.empty());
    tassert(parsed.start_levels.size() == 1 and parsed.start_levels[0] == 5);
    tassert(parsed.solver.value_or(solver_method::direct) == solver_method::gmres);
  }
}

void input_files() {
  {
    current_test name_("test_input1.txt");
    prog_opts prog(vecstrview({"", "-l", "3", "-if", "test_input1.txt"}));
    tassert(not prog.start_levels.empty());
    tassert(prog.start_levels[0] == 5);
    tassert(prog.adapt_threshold);
    tassert(prog.adapt_threshold.value() == 0.125);
    tassert(prog.grid);
    tassert(prog.grid.value() == grid_type::dense);
    tassert(prog.step_method);
    tassert(prog.step_method.value() == time_method::rk4);

    tassert(not prog.file_value<int>("missing"));
    auto bbool = prog.file_value<bool>("bb1");
    static_assert(std::is_same_v<decltype(bbool), std::optional<bool>>);
    tassert(bbool);
    tassert(bbool.value());
    auto iint = prog.file_value<int>("some_int");
    static_assert(std::is_same_v<decltype(iint), std::optional<int>>);
    tassert(iint);
    tassert(iint.value() == 8);

    tassert(prog.file_required<bool>("bb1"));
    tassert(prog.file_required<int>("some_int") == 8);

    terror_message(prog.file_required<std::string>("none"),
                   "file 'test_input1.txt' is missing required entry 'none'");

    prog_opts prog2(vecstrview({"", "-infile", "test_input1.txt", "-l", "3"}));
    tassert(not prog2.start_levels.empty());
    tassert(prog2.start_levels[0] == 3);
  }
  {
    current_test name_("test_input2.txt");
    prog_opts prog(vecstrview({"", "-if", "test_input2.txt"}));
    tassert(prog.start_levels.size() == 4);
    tassert(prog.start_levels[0] ==  9);
    tassert(prog.start_levels[1] == 11);
    tassert(prog.start_levels[2] ==  1);
    tassert(prog.start_levels[3] == 88);
    tassert(prog.title == "some long title");
    tassert(prog.subtitle == "short title");

    auto dbl = prog.file_value<double>("v_thermal");
    static_assert(std::is_same_v<decltype(dbl), std::optional<double>>);
    tassert(dbl);
    tassert(dbl.value() == 0.5);

    auto flt = prog.file_value<float>("half percent");
    static_assert(std::is_same_v<decltype(flt), std::optional<float>>);
    tassert(flt);
    tassert(std::abs(flt.value() - 5.E-3) < 1.E-6);

    tassert(not prog.file_value<float>("misspelled"));

    auto name = prog.file_value<std::string>("extra name");
    static_assert(std::is_same_v<decltype(name), std::optional<std::string>>);
    tassert(!!name);
    tassert(name.value() == "some-name test");

    auto nu = prog.file_value<int>("var nu");
    static_assert(std::is_same_v<decltype(nu), std::optional<int>>);
    tassert(!!nu);
    tassert(nu.value() == 314);
  }
  {
    current_test name_("test_input1.txt -- direct read");
    prog_opts prog("test_input1.txt");
    tassert(prog.start_levels.size() == 1);
    tassert(prog.start_levels[0] == 5);
    tassert(prog.adapt_threshold);
    tassert(prog.adapt_threshold.value() == 0.125);

    auto iint = prog.file_value<int>("some_int");
    static_assert(std::is_same_v<decltype(iint), std::optional<int>>);
    tassert(iint);
    tassert(iint.value() == 8);
  }
}

int main(int argc, char **argv) {

  libasgard_runtime running_(argc, argv);

  all_tests global_("asgard-prog-opts", " command-line args and input-files");

  new_prog_opts();
  input_files();

  return 0;
}
