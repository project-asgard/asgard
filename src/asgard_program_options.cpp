#include "asgard_program_options.hpp"

namespace asgard
{

bool is_explicit(time_method method) {
  switch (method) {
    case time_method::forward_euler:
    case time_method::rk2:
    case time_method::rk3:
    case time_method::rk4:
      return true;
    default:
      return false;
  }
}
bool is_implicit(time_method method) {
  switch (method) {
    case time_method::back_euler:
    case time_method::cn:
      return true;
    default:
      return false;
  }
}
bool is_imex(time_method method) {
  switch (method) {
    case time_method::imex1:
    case time_method::imex2:
      return true;
    default:
      return false;
  }
}

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

prog_opts::prog_opts(int const argc, char const *const *argv)
{
  std::vector<std::string_view> view_argv;
  view_argv.reserve(argc);
  for (auto i : indexof(argc))
    view_argv.emplace_back(argv[i]);

  process_inputs(view_argv, handle_mode::from_cli);
}

void prog_opts::print_help(std::ostream &os)
{
// keep the padding to 100 characters                                                      100 -> //
// ---------------------------------------------------------------------------------------------- //
  os << R"help(

Options          Short   Value      Description
-help/--help     -h/-?   -          Show help information (this text).
--version        -v      -          Show version, git info and build options.

-title             -     string     Human readable string focused on organizing i/o files,
                                    will be saved, reloaded and printed to the screen.
                                    If omitted, the string will assume the name of the PDE.
-subtitle          -     string     An addition to the title, optional use.
-infile          -if     filename   Read options and values from a provided file.
-view              -     string     example: "* : * : 1.57" or "* : 2 : *"
                                    passed into the default view of the plotter indicating the plane
                                    to plot, the view is a string with ":" separated entries
                                    holding up to two "*" entreis indicating the dimensions that
                                    will vary and numbers for the other dimensions
-verbosity       -vv     int/string accepts: 0/1/2 or quiet/low/high
                                    Adjusts the amount and frequency of cout logging.

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

-adapt           -a      double     Enable grid adaptivity and absolute tolerance threshold.
-adapt-abs       -aa     double     Identical to -a, alias for consistency with -ar
-adapt-rel       -ar     double     Enable grid adaptivity and relative tolerance threshold.

-noadapt         -noa    -          Ignore any previously set adapt options, can be used
                                    to override adaptivity set in an input file or restart file.

<<< time stepping options >>>
-step-method     -s      string     accepts:
                                      steady
                                      forward-euler/fe/rk1/rk2/rk3/rk4
                                      backwar-euler/be/crank-nicolson/cn
                                      imex1/imex2
                                    (fe, be and cn are shorthand acronyms for the longer names)
                                    (rk1 is the same as forward-euler)
                                    steady computes the steady state, not a time-stepping method
-time            -t      double     accepts: positive number (zero for no stepping)
                                    Final time for integration (v2 pdes only)
-num-steps       -n      int        Positive integer indicating the number of time steps to take.
-dt                      double     Fixed time step to use (must be positive).

<<< i/o options >>>
-restart                 filename   Wavelet output file to restart the simulation.
-outfile         -of     filename   File to write the last step of the simulation.

<<< solvers and linear algebra options >>>
-solver          -sv     string     accepts: direct/gmres/bicgstab (implicit/imex methods only)
                                    Direct: use LAPACK, expensive but stable.
                                    GMRES: general but sensitive to restart selection.
                                    bicgstab: cheaper (per-iteration) alternative to GMRES
-precon          -pc     string     accepts: none/jacobi (iterative solvers only)
                                    specifies the preconditioner for the iterative method
                                    none - is not advisable as it takes too long
                                    jacobi - preconditioner that applies basic rescaling
-isolve-tol      -ist    double     Iterative solver tolerance, applies to GMRES and BICG.
-isolve-iter     -isi    int        Iterative solver maximum number of iterations,
                                    for GMRES this is the number of outer iterations.
-isolve-inner    -isn    int        (GMRES only) The maximum number of inner GMRES iterations,
                                    this is ignored by BiCGSTAB.

)help";
}

void prog_opts::process_inputs(std::vector<std::string_view> const &argv, handle_mode mode)
{
  std::map<std::string_view, optentry> commands = {
      {"help", optentry::show_help}, {"-help", optentry::show_help}, {"--help", optentry::show_help},
      {"-h", optentry::show_help}, {"-?", optentry::show_help},
      {"--version", optentry::version_help}, {"-version", optentry::version_help},
      {"version", optentry::version_help}, {"-v", optentry::version_help},
      {"-infile", optentry::input_file}, {"-if", optentry::input_file},
      {"-view", optentry::view},
      {"-noexact", optentry::ignore_exact}, {"-ne", optentry::ignore_exact},
      {"-title", optentry::title},
      {"-subtitle", optentry::subtitle},
      {"-verbosity", optentry::set_verbosity}, {"-vv", optentry::set_verbosity},
      {"-grid", optentry::grid_mode}, {"-g", optentry::grid_mode},
      {"-step-method", optentry::step_method}, {"-s", optentry::step_method},
      {"-adapt", optentry::adapt_threshold},  {"-a", optentry::adapt_threshold},
      {"-adapt-abs", optentry::adapt_threshold},  {"-aa", optentry::adapt_threshold},
      {"-adapt-rel", optentry::adapt_relative},  {"-ar", optentry::adapt_relative},
      {"-noadapt", optentry::no_adapt},  {"-noa", optentry::no_adapt},
      {"-start-levels", optentry::start_levels}, {"-l", optentry::start_levels},
      {"-max-levels", optentry::max_levels}, {"-m", optentry::max_levels},
      {"-degree", optentry::degree}, {"-d", optentry::degree},
      {"-num-steps", optentry::num_time_steps}, {"-n", optentry::num_time_steps},
      {"-outfile", optentry::output_file}, {"-of", optentry::output_file},
      {"-dt", optentry::dt},
      {"-time", optentry::stop_time}, {"-t", optentry::stop_time},
      {"-solver", optentry::solver}, {"-sv", optentry::solver},
      {"-precon", optentry::precond}, {"-pc", optentry::precond},
      {"-isolve-tol", optentry::isol_tolerance}, {"-ist", optentry::isol_tolerance},
      {"-isolve-iter", optentry::isol_iterations}, {"-isi", optentry::isol_iterations},
      {"-isolve-inner", optentry::isol_inner_iterations},
      {"-isn", optentry::isol_inner_iterations},
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
    {
      if (mode == handle_mode::from_cli)
        externals.emplace_back(*iarg);
      else
        filedata.emplace_back(*iarg);
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
    case optentry::view: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      default_plotter_view = *selected;
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
      // allow longer options here, short for the command line
      // long for more expressive files
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());

      std::map<std::string_view, time_method> vals = {
        {"steady", time_method::steady},
        {"forward-euler", time_method::forward_euler}, {"fe", time_method::forward_euler}, {"rk1", time_method::forward_euler},
        {"rk2", time_method::rk2},
        {"rk3", time_method::rk3},
        {"rk4", time_method::rk4},
        {"backward-euler", time_method::back_euler}, {"be", time_method::back_euler},
        {"crank-nicolson", time_method::cn}, {"cn", time_method::cn},
        {"imex1", time_method::imex1},
        {"imex2", time_method::imex2},
      };

      auto it = vals.find(*selected);
      if (it == vals.end())
        throw std::runtime_error(report_wrong_value());
      step_method = it->second;
    }
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
    case optentry::output_file: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      outfile = *selected;
    }
    break;
    case optentry::stop_time: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      try {
        stop_time = std::stod(selected->data());
      } catch(std::invalid_argument &) {
        throw std::runtime_error(report_wrong_value());
      } catch(std::out_of_range &) {
        throw std::runtime_error(report_wrong_value());
      }
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
    case optentry::adapt_relative: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      try {
        adapt_ralative = std::stod(selected->data());
      } catch(std::invalid_argument &) {
        throw std::runtime_error(report_wrong_value());
      } catch(std::out_of_range &) {
        throw std::runtime_error(report_wrong_value());
      }
    }
    break;
    case optentry::no_adapt:
      // sufficient to override a deck file adapt options
      adapt_threshold.reset();
      adapt_ralative.reset();
      // needed to cancel adaptivity from a restart file
      set_no_adapt = true;
    break;
    case optentry::solver: {
      // with only a handful of solvers we don't need to use a map here
      // if we go to 20+ solvers we may change that
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      if (*selected == "direct")
        solver = solver_method::direct;
      else if (*selected == "gmres")
        solver = solver_method::gmres;
      else if (*selected == "bicgstab")
        solver = solver_method::bicgstab;
      else
        throw std::runtime_error(report_wrong_value());
    }
    break;
    case optentry::precond: {
      // if we get more preconditioners we may switch to a map
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      if (*selected == "none")
        precon = precon_method::none;
      else if (*selected == "jacobi")
        precon = precon_method::jacobi;
      else if (*selected == "adi")
        precon = precon_method::adi;
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
    case optentry::isol_inner_iterations: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      try {
        isolver_inner_iterations = std::stoi(selected->data());
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
    case optentry::set_verbosity: {
      auto selected = move_process_next();
      if (not selected)
        throw std::runtime_error(report_no_value());
      if (*selected == "0" or *selected == "quiet")
        verbosity = verbosity_level::quiet;
      else if (*selected == "1" or *selected == "low")
        verbosity = verbosity_level::low;
      else if (*selected == "2" or *selected == "high")
        verbosity = verbosity_level::high;
      else
        throw std::runtime_error(report_wrong_value());
    }
    break;
    };
  }
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
    if (sharp == 0)
      return s;
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

  process_inputs(views, handle_mode::from_file);
}

void prog_opts::print_options(std::ostream &os) const
{
  os << "ASGarD problem configuration:\n";
  os << "  title: " << title << '\n';
  if (not subtitle.empty())
    os << "         " << subtitle << '\n';

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

  if (adapt_threshold or adapt_ralative)
  {
    os << "  adaptive tolerance:";
    if (adapt_ralative)
      os << " (relative)" << adapt_ralative.value();
    if (adapt_threshold)
      os << " (absolute)" << adapt_threshold.value();
    os << '\n';
  }
  else
    os << "  non-adaptive grid\n";

  os << "time stepping:\n";
  if (step_method)
    os << "  " << get_name(step_method.value()) << "\n";

  if (stop_time)
    os << "  stop-time: " << stop_time.value() << '\n';
  else if (default_stop_time)
    os << "  stop-time (default): " << default_stop_time.value() << '\n';

  if (dt)
    os << "  time-step (dt): " << dt.value() << '\n';
  else if (default_dt)
    os << "  time-step (default dt): " << default_dt.value() << '\n';

  if (num_time_steps)
    os << "  number of time-steps: " << num_time_steps.value() << '\n';

  if (not restart_file.empty())
    os << "  restarting from: " << restart_file << '\n';
}

void prog_opts::print_version_help(std::ostream &os)
{
  os << "\nASGarD v" << ASGARD_VERSION << "  git-hash: " << ASGARD_GIT_COMMIT_HASH << "\n";
  os << "git-branch (" << ASGARD_GIT_BRANCH << ")\n";

#ifdef ASGARD_USE_OPENMP
  os << "OpenMP multithreading    Enablded\n";
#else
  os << "OpenMP multithreading    Disabled\n";
#endif
#ifdef ASGARD_USE_GPU
  #ifdef ASGARD_USE_CUDA
    os << "GPU Acceleration         CUDA\n";
  #else
    os << "GPU Acceleration         ROCm\n";
  #endif
#else
  os << "GPU Acceleration         Disabled\n";
#endif
#ifdef ASGARD_USE_MPI
  os << "MPI distributed terms    Enabled\n";
#else
  os << "MPI distributed terms    Disabled\n";
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

std::string prog_opts::get_name(time_method t)
{
  std::map<time_method, std::string> names = {
    {time_method::steady, "Steady state solver"},
    {time_method::forward_euler, "Forward-Euler 1-step (explicit)"},
    {time_method::rk2, "Runge-Kutta 2-step (explicit)"},
    {time_method::rk3, "Runge-Kutta 3-step (explicit)"},
    {time_method::rk4, "Runge-Kutta 4-step (explicit)"},
    {time_method::back_euler, "Backward-Euler 1-step (implicit)"},
    {time_method::cn, "Crank-Nicolson 1-step (implicit)"},
    {time_method::imex1, "Implicit-Explicit 1-step (imex)"},
    {time_method::imex2, "Implicit-Explicit 2-step (imex)"},
  };

  return names.find(t)->second;
}

} // namespace asgard
