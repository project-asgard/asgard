#include "asgard_program_options.hpp"

namespace asgard
{
split_views split_argv(std::string_view const &opts)
{
  std::stringstream inopts{std::string(opts)};
  std::vector<std::string> splits;
  while (!inopts.eof())
  {
    splits.emplace_back();
    inopts >> splits.back();
  }
  return split_views(std::move(splits));
}

prog_opts::prog_opts(int const argc, char const *const *argv,
                     bool ignore_unknown)
{
  std::vector<std::string_view> view_argv;
  view_argv.reserve(argc);
  for (auto i : indexof(argc))
    view_argv.emplace_back(argv[i]);

  process_inputs(
      view_argv, (ignore_unknown) ? handle_mode::ignore_unknown : handle_mode::warn_on_unknown);
}

void prog_opts::print_help(std::ostream &os)
{
// keep the padding to 100 characters                                                      100 -> //
// ---------------------------------------------------------------------------------------------- //
  os << R"help(

Options          Short   Value      Description
-help/--help     -h/-?   -          Show help information (this text).
--version        -v      -          Show version, git info and build options.
-pde?            -p?     -          Show list of builtin PDEs.
-pde             -p      string     accepts: PDEs shown by -p?
                                    Indicates the PDE to use from the list builtin PDEs,
                                    defaults to custom PDE and should not be used by
                                    custom projects.
-title             -     string     Human redable string focused on organizing i/o files,
                                    will be saved, reloaded and printed to the screen.
                                    If omitted, the string will assume the name of the PDE.
-subtitle          -     string     An addition to the title, optional use.
-infile          -if     filename   Read options and values from a provided file.

<<< discretization of the domain options >>>
-grid            -g      string     accepts: sparse/dense/full/mixed/mix
                                    Sparse grid is the standard approach for error balance.
                                    Dense grid (also full) will uniformly fill the domain
                                    but at a very high cost in 3d and above.
                                    The mixed grid (also mix) takes a dense tensor of two sparse
                                    grids, must provide an additional 'int' indicating the number
                                    of dimensions to use in the first sparse grid.
-degree          -d      int        accepts: non-negative integer
                                    Polynomial degree for the basis, constant (0), linear (1),
                                    quadratic (2) or so on.
-start-levels    -l      int/list   accepts: one int or one int per dimension
                                    The starting level for the simulation, can be one int
                                    indicating uniform initial level or a list of ints
                                    indicating the level per-dimension.
                                    If missing, the default level will be used as defined
                                    in the PDE class specification.
-max-levels      -m      int/list   accepts: one int or one int per dimension
                                    Maximum level for the refinement process,
                                    if missing, the starting levels will be used as the max.

-adapt           -a      double     Enable grid adaptivity and set the tolerance threshold.
-adapt-norm      -an     string     accepts: linf/l2
                                    The norm to use for the refinement criteria.

<<< time stepping options >>>
-step-method     -s      string     accepts: expl/impl/imex
                                    indicates explicit (rk3), explicit (backward-Euler) or
                                    imex (implicit-explicit) time-stepping scheme.
-num-steps       -n      int        Positive integer indicating the number of time steps to take.
-dt                      double     Fixed time step to use (must be positive).

-noexact         -ne     -          If a pde has known exact solution, it must be for testing
                                    of the code, therefore for each time step the computed solution
                                    will be compared to the exact one. This option will disable
                                    the comparison and speed the simulations of the test.

<<< i/o options >>>
-wave-freq       -w      int        Interval (in time steps) for outputting the hierarchical
                                    wavelet data, compatible with Python plotting.
-restart                 filename   Wavelet output file to restart the simulation.
-outfile         -of     filename   File to write the last step of the simulation.

<<< solvers and linear algebra options >>>
-kron-mode       -       string     accepts: dense/sparse
                                    Applies to the local Kronmult algorithms, the sparse approach
                                    results in flop savings but, also high additional RAM usage
                                    due to the extra indexing.
-solver          -sv     string     accepts: direct/gmres/bicgstab (implicit/imex methods only)
                                    Direct: use LAPACK, expensive but stable.
                                    GMRES: general but sensitive to restart selection.
                                    bicgstab: cheaper alternative to GMRES
-isolve-tol      -ist    double     Iterative solver tolerance, applies to GMRES and BICG.
-isolve-iter     -isi    double     Iterative solver maximum number of iterations,
                                    for GMRES this is the number of inner iterations.
-isolve-outer    -iso    double     (GMRES only) The maximum number of outer GMRES iterations.

Coming soon:
-time            -t      double     Final time for integration.

Leaving soon:
-memory                  int        Memory limit for the GPU, applied to the earlier versions
                                    of Kronmult, where data was kept in CPU RAM and moved
                                    on-the-fly in an out-of-core algorithm. The data-transfer
                                    cost makes the approach impractical.

)help";
}

void prog_opts::print_pde_help(std::ostream &os)
{
// keep the padding to 100 characters                                                      100 -> //
// ---------------------------------------------------------------------------------------------- //
  os << R"help(

Option          Description
custom          (default) user provided pde, can be omitted for the custom projects
continuity_1    1D test case, continuity equation: df/dt + df/dx = 0
continuity_2    2D test case, continuity equation: df/dt + df/dx + df/dy = 0"
continuity_3    3D test case, continuity equation, df/dt + v.grad(f) = 0 where v={1,1,1}
continuity_6    6D test case, continuity equation, df/dt + v.grad(f) = 0 where v={1,1,3,4,3,2}"
diffusion_1     1D diffusion equation: df/dt = d^2 f/dx^2
diffusion_2     2D (1x-1y) heat equation. df/dt = d^2 f/dx^2 + d^2 f/dy^2
advection_1     1D test using continuity equation. df/dt = -2*df/dx - 2*sin(x)
vlasov          Vlasov lb full f. df/dt = -v*grad_x f + div_v((v-u)f + theta*grad_v f)
two_stream      Vlasov two-stream. df/dt = -v*grad_x f -E*grad_v f

fokkerplanck_1d_pitch_E_case1    1D pitch angle collisional term:
                                 df/dt = d/dz ( (1-z^2) df/dz, f0 is constant.

fokkerplanck_1d_pitch_E_case2    1D pitch angle collisional term:
                                 df/dt = d/dz ( (1-z^2) df/dz, f0 is gaussian.

fokkerplanck_1d_pitch_C        1D pitch angle collisional term: df/dt = d/dz ( (1-z^2) df/dz
fokkerplanck_1d_4p3            Radiation damping term: df/dt = -d/dz ( z(1-z^2)f )
fokkerplanck_1d_4p4            Evolution of f's pitch angle dependence with electric
                               field acceleration/collision:
                               df/dt = -E d/dz((1-z^2) f) + C d/dz((1-z^2) df/dz)
fokkerplanck_1d_4p5            Same as 4p4, but with radiation damping:
                               df/dt = -E d/dz((1-z^2) f) + C d/dz((1-z^2) df/dz)
                                       -R d/dz(z(1-z^2) f)

fokkerplanck_2d_complete_case1    Full PDE from the 2D runaway electron paper:
                                  d/dt f(p,z) = -div(flux_C + flux_E + flux_R), case 1
fokkerplanck_2d_complete_case2    Full PDE from the 2D runaway electron paper:
                                  d/dt f(p,z) = -div(flux_C + flux_E + flux_R), case 2
fokkerplanck_2d_complete_case3    Full PDE from the 2D runaway electron paper:
                                  d/dt f(p,z) = -div(flux_C + flux_E + flux_R), case 3
fokkerplanck_2d_complete_case4    Full PDE from the 2D runaway electron paper:
                                  d/dt f(p,z) = -div(flux_C + flux_E + flux_R), case 4

relaxation_1x1v    Relaxation 1x1v. df/dt = div_{v} v f + d_{v} -u f  + d_{v}(th q), q = d_{v} f
relaxation_1x2v    Relaxation 1x2v.
                   df/dt = div_{v1} v_1 f + d_{v1} -u_1 f + div_{v2} v_2 f
                          + d_{v2} -u_2 f + d_{v1}(th q),
                   q = d_{v1} f + d_{v2}(th q), q = d_{v2} f

relaxation_1x3v    Relaxation 1x3v.
                   df/dt = div_{v1} v_1 f + d_{v1} -u_1 f + div_{v2} v_2 f
                          + d_{v2} -u_2 f + d_{v1}(th q),
                   q = d_{v1} f + d_{v2}(th q), q = d_{v2} f

riemann_1x2v    Riemann 1x2v
riemann_1x3v    Riemann 1x3v

landau         Collisional Landau.
               df/dt = -v*grad_x f -E*grad_v f + div_v((v-u)f + theta*grad_v f)
landau_1x2v    Collisional Landau 1x2v.
               df/dt = -v*grad_x f -E*grad_v f + div_v((v-u)f + theta*grad_v f)
landau_1x3v    Collisional Landau 1x3v.
               df/dt == -v*grad_x f -E*grad_v f + div_v((v-u)f + theta*grad_v f)

)help";
}

void prog_opts::process_inputs(std::vector<std::string_view> const &argv,
                               handle_mode mode)
{
  std::map<std::string_view, optentry> commands = {
      {"help", optentry::show_help}, {"-help", optentry::show_help}, {"--help", optentry::show_help},
      {"-h", optentry::show_help}, {"-?", optentry::show_help},
      {"--version", optentry::version_help}, {"-version", optentry::version_help},
      {"version", optentry::version_help}, {"-v", optentry::version_help},
      {"-pde?", optentry::pde_help}, {"-p?", optentry::pde_help},
      {"-pde", optentry::pde_choice}, {"-p", optentry::pde_choice},
      {"-infile", optentry::input_file}, {"-if", optentry::input_file},
      {"-noexact", optentry::ignore_exact}, {"-ne", optentry::ignore_exact},
      {"-title", optentry::title},
      {"-subtitle", optentry::subtitle},
      {"-grid", optentry::grid_mode}, {"-g", optentry::grid_mode},
      {"-step-method", optentry::step_method}, {"-s", optentry::step_method},
      {"-adapt-norm", optentry::adapt_norm}, {"-an", optentry::adapt_norm},
      {"-electric-solve", optentry::set_electric}, {"-es", optentry::set_electric},
      {"-adapt", optentry::adapt_threshold},  {"-a", optentry::adapt_threshold},
      {"-start-levels", optentry::start_levels}, {"-l", optentry::start_levels},
      {"-max-levels", optentry::max_levels}, {"-m", optentry::max_levels},
      {"-degree", optentry::degree}, {"-d", optentry::degree},
      {"-num-steps", optentry::num_time_steps}, {"-n", optentry::num_time_steps},
      {"-wave-freq", optentry::wavelet_output_freq}, {"-w", optentry::wavelet_output_freq},
      {"-outfile", optentry::output_file}, {"-of", optentry::output_file},
      {"-dt", optentry::dt},
      {"-available-pdes", optentry::pde_help},
      {"-solver", optentry::solver}, {"-sv", optentry::solver},
      {"-memory", optentry::memory_limit},
      {"-kron-mode", optentry::kron_mode},
      {"-isolve-tol", optentry::isol_tolerance}, {"-ist", optentry::isol_tolerance},
      {"-isolve-iter", optentry::isol_iterations}, {"-isi", optentry::isol_iterations},
      {"-isolve-outer", optentry::isol_outer_iterations},
      {"-iso", optentry::isol_outer_iterations},
      {"-restart", optentry::restart_file},
  };

  auto iarg = argv.cbegin();

  auto report_no_value = [&]()
      -> std::string {
    return std::string(*iarg) + " must be followed by a value, see "
           + std::string(argv.front()) + " -help";
  };
  auto report_wrong_value = [&]()
      -> std::string {
    return std::string("invalid value for ") + std::string(*(iarg - 1))
           + ", see " + std::string(argv.front()) + " -help";
  };
  auto report_wrong_pde = [&]()
      -> std::string {
    return std::string("invalid pde '") + std::string(*iarg) + "', see '"
           + std::string(argv.front()) + " -pde?' for full list";
  };

  auto move_process_next = [&]()
      -> std::optional<std::string_view>
  {
    if (iarg + 1 == argv.end())
      return {};
    return *++iarg;
  };

  // on entry into the loop, iarg is incremented ignoring argv[0]
  // argv[0] is the name of the executable
  while (++iarg != argv.end())
  {
    auto imap = commands.find(*iarg);
    if (imap == commands.end())
    { // entry not found
      if (mode == handle_mode::warn_on_unknown)
        std::cerr << "  unrecognized option: " << *iarg << "\n";
      if (mode == handle_mode::save_unknown)
        filedata.emplace_back(*iarg);
      else
        externals.emplace_back(*iarg);
      continue;
    }

    switch (imap->second)
    {
    case optentry::show_help:
      show_help = true;
      break;
    case optentry::version_help:
      show_version = true;
      break;
    case optentry::pde_help:
      show_pde_help = true;
      break;
    case optentry::ignore_exact:
      ignore_exact = true;
      break;
    case optentry::input_file: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      if (not infile.empty())
        throw std::runtime_error("cannot read from two input files");
      infile = *selected;
      process_file(argv.front());
    }
    break;
    case optentry::grid_mode: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      if (*selected == "sparse")
        grid = grid_type::sparse;
      else if (*selected == "dense" or *selected == "full")
        grid = grid_type::dense;
      else if (*selected == "mixed" or *selected == "mix")
      {
        auto s2 = move_process_next();
        if (not s2)
          throw std::runtime_error(
              "missing mixed grid number, see " + std::string(argv.front())
               + " -help");
        try {
          grid        = grid_type::mixed;
          mgrid_group = std::stoi(s2->data());
        } catch(std::invalid_argument &) {
          throw std::runtime_error(report_wrong_value());
        } catch(std::out_of_range &) {
          throw std::runtime_error(report_wrong_value());
        }
      }
      else if (selected->size() > 6 and (*selected).find("mixed") != std::string::npos)
      {
        auto pos = (*selected).rfind("mixed") + 5; // 5 == length of "mixed"
        try {
          grid        = grid_type::mixed;
          mgrid_group = std::stoi(std::string((*selected).substr(pos)));
        } catch(std::invalid_argument &) {
          throw std::runtime_error(report_wrong_value());
        } catch(std::out_of_range &) {
          throw std::runtime_error(report_wrong_value());
        }
      }
      else
        throw std::runtime_error(report_wrong_value());
    }
    break;
    case optentry::step_method: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      if (*selected == "expl")
        step_method = time_advance::method::exp;
      else if (*selected == "impl")
        step_method = time_advance::method::imp;
      else if (*selected == "imex")
        step_method = time_advance::method::imex;
      else {
        throw std::runtime_error(report_wrong_value());
      }
    }
    break;
    case optentry::adapt_norm: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      if (*selected == "l2")
        anorm = adapt_norm::l2;
      else if (*selected == "linf")
        anorm = adapt_norm::linf;
      else {
        throw std::runtime_error(report_wrong_value());
      }
    }
    break;
    case optentry::set_electric:
    set_electric = true;
    break;
    case optentry::start_levels: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      start_levels = parse_ints(selected->data());
      if (start_levels.empty())
        throw std::runtime_error(report_wrong_value());
    }
    break;
    case optentry::max_levels: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      max_levels = parse_ints(selected->data());
      if (max_levels.empty())
        throw std::runtime_error(report_wrong_value());
    }
    break;
    case optentry::degree: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      try {
        degree = std::stoi(selected->data());
      } catch(std::invalid_argument &) {
        throw std::runtime_error(report_wrong_value());
      } catch(std::out_of_range &) {
        throw std::runtime_error(report_wrong_value());
      }
    }
    break;
    case optentry::num_time_steps: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      try {
        num_time_steps = std::stoi(selected->data());
      } catch(std::invalid_argument &) {
        throw std::runtime_error(report_wrong_value());
      } catch(std::out_of_range &) {
        throw std::runtime_error(report_wrong_value());
      }
    }
    break;
    case optentry::wavelet_output_freq: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      try {
        wavelet_output_freq = std::stoi(selected->data());
      } catch(std::invalid_argument &) {
        throw std::runtime_error(report_wrong_value());
      } catch(std::out_of_range &) {
        throw std::runtime_error(report_wrong_value());
      }
    }
    break;
    case optentry::output_file: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      outfile = *selected;
    }
    break;
    case optentry::dt: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      try {
        dt = std::stod(selected->data());
      } catch(std::invalid_argument &) {
        throw std::runtime_error(report_wrong_value());
      } catch(std::out_of_range &) {
        throw std::runtime_error(report_wrong_value());
      }
    }
    break;
    case optentry::adapt_threshold: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      try {
        adapt_threshold = std::stod(selected->data());
      } catch(std::invalid_argument &) {
        throw std::runtime_error(report_wrong_value());
      } catch(std::out_of_range &) {
        throw std::runtime_error(report_wrong_value());
      }
    }
    break;
    case optentry::solver: {
      // with only a handful of solvers we don't need to use a map here
      // if we go to 20+ solvers we may change that
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      if (*selected == "direct")
        solver = solve_opts::direct;
      else if (*selected == "gmres")
        solver = solve_opts::gmres;
      else if (*selected == "bicgstab")
        solver = solve_opts::bicgstab;
      else
        throw std::runtime_error(report_wrong_value());
    }
    break;
    case optentry::memory_limit: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      try {
        memory_limit = std::stoi(selected->data());
      } catch(std::invalid_argument &) {
        throw std::runtime_error(report_wrong_value());
      } catch(std::out_of_range &) {
        throw std::runtime_error(report_wrong_value());
      }
    }
    break;
    case optentry::kron_mode: {
      // this may go away soon, does not apply to global-kron
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      if (*selected == "sparse")
        kron_mode = kronmult_mode::sparse;
      else if (*selected == "dense")
        kron_mode = kronmult_mode::dense;
      else
        throw std::runtime_error(report_wrong_value());
    }
    break;
    case optentry::isol_tolerance: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      try {
        isolver_tolerance = std::stod(selected->data());
      } catch(std::invalid_argument &) {
        throw std::runtime_error(report_wrong_value());
      } catch(std::out_of_range &) {
        throw std::runtime_error(report_wrong_value());
      }
    }
    break;
    case optentry::isol_iterations: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      try {
        isolver_iterations = std::stoi(selected->data());
      } catch(std::invalid_argument &) {
        throw std::runtime_error(report_wrong_value());
      } catch(std::out_of_range &) {
        throw std::runtime_error(report_wrong_value());
      }
    }
    break;
    case optentry::isol_outer_iterations: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      try {
        isolver_outer_iterations = std::stoi(selected->data());
      } catch(std::invalid_argument &) {
        throw std::runtime_error(report_wrong_value());
      } catch(std::out_of_range &) {
        throw std::runtime_error(report_wrong_value());
      }
    }
    break;
    case optentry::restart_file: {
      // this may go away soon, does not apply to global-kron
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      restart_file = *selected;
    }
    break;
    case optentry::pde_choice: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      pde_choice = get_pde_opt(*selected);
      if (not pde_choice)
        throw std::runtime_error(report_wrong_pde());
      if (title.empty())
        title = *selected;
    }
    break;
    case optentry::title: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      title = *selected;
      if (title.empty())
        throw std::runtime_error(report_wrong_value());
    }
    break;
    case optentry::subtitle: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      subtitle = *selected;
    }
    break;
    };
  }
}

std::optional<PDE_opts> prog_opts::get_pde_opt(std::string_view const &pde_str)
{
  std::map<std::string_view, PDE_opts> pdes = {
      {"custom", PDE_opts::custom},
      {"continuity_1", PDE_opts::continuity_1},
      {"continuity_2", PDE_opts::continuity_2},
      {"continuity_3", PDE_opts::continuity_3},
      {"continuity_6", PDE_opts::continuity_6},
      {"fokkerplanck_1d_pitch_E_case1", PDE_opts::fokkerplanck_1d_pitch_E_case1},
      {"fokkerplanck_1d_pitch_E_case2", PDE_opts::fokkerplanck_1d_pitch_E_case2},
      {"fokkerplanck_1d_pitch_C", PDE_opts::fokkerplanck_1d_pitch_C},
      {"fokkerplanck_1d_4p3", PDE_opts::fokkerplanck_1d_4p3},
      {"fokkerplanck_1d_4p4", PDE_opts::fokkerplanck_1d_4p4},
      {"fokkerplanck_1d_4p5", PDE_opts::fokkerplanck_1d_4p5},
      {"fokkerplanck_2d_complete_case1", PDE_opts::fokkerplanck_2d_complete_case1},
      {"fokkerplanck_2d_complete_case2", PDE_opts::fokkerplanck_2d_complete_case2},
      {"fokkerplanck_2d_complete_case3", PDE_opts::fokkerplanck_2d_complete_case3},
      {"fokkerplanck_2d_complete_case4", PDE_opts::fokkerplanck_2d_complete_case4},
      {"diffusion_1", PDE_opts::diffusion_1},
      {"diffusion_2", PDE_opts::diffusion_2},
      {"advection_1", PDE_opts::advection_1},
      {"vlasov", PDE_opts::vlasov_lb_full_f},
      {"two_stream", PDE_opts::vlasov_two_stream},
      {"relaxation_1x1v", PDE_opts::relaxation_1x1v},
      {"relaxation_1x2v", PDE_opts::relaxation_1x2v},
      {"relaxation_1x3v", PDE_opts::relaxation_1x3v},
      {"riemann_1x2v", PDE_opts::riemann_1x2v},
      {"riemann_1x3v", PDE_opts::riemann_1x3v},
      {"landau", PDE_opts::collisional_landau},
      {"landau_1x2v", PDE_opts::collisional_landau_1x2v},
      {"landau_1x3v", PDE_opts::collisional_landau_1x3v},
  };

  auto imap = pdes.find(pde_str);

  return (imap != pdes.end()) ? imap->second : std::optional<PDE_opts>();
}

void prog_opts::process_file(std::string_view const &exec_name)
{
  // 1. read the lines from the file
  // 2. split each line into 2 strings
  // 3. feed the result into process_inputs()
  //    (this will extract all standard known inputs)
  // 4. the remaining inputs will be stored in filedata
  rassert(std::filesystem::exists(infile),
          "cannot find the input file " + std::string(infile));

  std::ifstream ifs(infile, std::ifstream::in);
  rassert(ifs, "cannot open the input file " + std::string(infile));

  auto strip_line = [](std::string const &s)
      -> std::string
  {
    if (s.empty())
      return s;
    std::string::size_type be = 0;
    while (std::isspace(static_cast<unsigned char>(s[be])))
      be++;
    std::string::size_type en = s.size() - 1;
    std::string::size_type sharp = s.find('#');
    if (sharp < s.size())
      en = sharp - 1;
    while (en > be and std::isspace(static_cast<unsigned char>(s[en])))
      en--;
    return s.substr(be, en - be + 1);
  };

  std::vector<std::string> line_pairs;
  std::string line;
  int line_num = 0;
  while (getline(ifs, line))
  {
    line_num++;
    line = strip_line(line);
    if (line[0] == '#') // ignore lines starting with '#' (comments)
      continue;
    if (line.empty())
      continue;
    auto pos = line.find(':');
    rassert(pos < line.size(),
            "invalid file format, lines must be '<option> : <value>\nline: "
              + std::to_string(line_num) + " is missing ':'");

    std::string op = strip_line(line.substr(0, pos));
    std::string va = strip_line(line.substr(pos + 1, line.size() - pos));

    line_pairs.emplace_back(op);
    line_pairs.emplace_back(va);
  }

  std::vector<std::string_view> views;
  views.reserve(line_pairs.size() + 1);
  views.emplace_back(exec_name);
  for (auto &s : line_pairs)
    views.emplace_back(s);

  process_inputs(views, handle_mode::save_unknown);
}

void prog_opts::print_options(std::ostream &os) const
{
  os << "ASGarD problem configuration:\n";
  os << "  title: " << title << '\n';
  if (not subtitle.empty())
    os << "  sub:   " << subtitle << '\n';

  os << "discretization:\n";
  if (degree)
    switch (degree.value())
    {
    case 0:
      os << "  degree: constant (0) \n";
      break;
    case 1:
      os << "  degree: linear (1) \n";
      break;
    case 2:
      os << "  degree: quadratic (2) \n";
      break;
    case 3:
      os << "  degree: cubic (3) \n";
      break;
    default:
      os << "  degree: " << degree.value() << '\n';
    };

  if (grid)
    switch (grid.value())
    {
    case grid_type::dense:
      os << "  gird mode: dense/full grid\n";
      break;
    case grid_type::mixed:
      os << "  gird mode: mixed (tensor of two sparse grids)\n";
      if (mgrid_group)
        os << "  group size: " << mgrid_group.value() << '\n';
      else
        os << "  -- warning: missing mixed group size\n";
      break;
    default:
      os << "  gird mode: sparse grid\n";
      break;
    };

  if (not start_levels.empty())
    os << "  start levels: " << start_levels_str() << '\n';

  if (not max_levels.empty())
    os << "    max levels: " << max_levels_str() << '\n';

  if (adapt_threshold)
  {
    os << "  adaptive tolerance: " << adapt_threshold.value() << '\n';
    if (anorm and anorm.value() == adapt_norm::l2)
      os << "  adaptive norm: l2\n";
    else
      os << "  adaptive norm: l-infty\n";
  }
  else
    os << "  non-adaptive grid\n";

  os << "time stepping:\n";
  if (step_method)
    switch (step_method.value())
    {
    case time_advance::method::imex:
      os << "  method: IMEX\n";
      break;
    case time_advance::method::imp:
      os << "  method: Backward Euler\n";
      break;
    default:
      os << "  method: RK3\n";
      break;
    };

  if (dt)
    os << "  time-step (dt): " << dt.value() << '\n';

  if (num_time_steps)
    os << "  number of time-steps: " << num_time_steps.value() << '\n';
  else
    os << "  -- warning: missing number of time steps\n";

  if (restart_file.empty() and not wavelet_output_freq)
    os << "input-output (i/o): none\n";
  else
    os << "input-output (i/o):\n";
  if (not restart_file.empty())
    os << "  restarting from: " << restart_file << '\n';
  if (wavelet_output_freq)
    os << "  write freq: " << wavelet_output_freq.value() << '\n';
}

void prog_opts::print_version_help(std::ostream &os)
{
  os << "\nASGarD v" << ASGARD_VERSION << "  git-hash: " << GIT_COMMIT_HASH << "\n";
  os << "git-branch (" << GIT_BRANCH << ")\n";
#ifdef ASGARD_USE_MPI
  os << "Kronmult method          Local\n";
#else
  os << "Kronmult method          Block-Global\n";
#endif
#ifdef ASGARD_USE_CUDA
  os << "GPU Acceleration         CUDA\n";
#else
  os << "GPU Acceleration         Disabled\n";
#endif
#ifdef ASGARD_USE_OPENMP
  os << "OpenMP multithreading    Enablded\n";
#else
  os << "OpenMP multithreading    Disabled\n";
#endif
#ifdef ASGARD_USE_MPI
  os << "MPI distributed grid     Enabled\n";
#else
  os << "MPI distributed grid     Disabled\n";
#endif
#ifdef ASGARD_USE_HIGHFIVE
  os << "HDF5 - HighFive I/O      Enabled\n";
#else
  os << "HDF5 - HighFive I/O      Disabled\n";
#endif
#ifdef ASGARD_ENABLE_DOUBLE
#ifdef ASGARD_ENABLE_FLOAT
  os << "Available precisions     double/float\n";
#else
  os << "Available precision      double\n";
#endif
#else
  os << "Available precision      float\n";
#endif
  os << '\n';
}

} // namespace asgard
