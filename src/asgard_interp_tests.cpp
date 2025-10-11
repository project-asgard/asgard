#include "asgard_test_macros.hpp"

using namespace asgard;

template<typename P>
void interp_wav2nodal() {
  P constexpr tol = (std::is_same_v<P, double>) ? 1.E-12 : 1.E-5;

  pde_domain<P> domain(2); // work in 2d
  separable_func<P> ic({1, 1});
  ic.set(0, vectorize_t<P>([](P x)->P { return std::sin(x); }));
  ic.set(1, vectorize_t<P>([](P x)->P { return std::exp(x); }));

  auto vec2d = [](vector2d<P> const &vec) -> vector2d<double> {
    vector2d<double> result(vec.stride(), vec.num_strips());
    for (int64_t i = 0; i < vec.stride() * vec.num_strips(); i++)
      result[0][i] = static_cast<double>(vec[0][i]);
    return result;
  };

  std::map<int, std::string> mode = {{0, "constant"}, {1, "linear"},
                                     {2, "quadratic"}, {3, "cubic"}};

  domain = pde_domain<P>(2);
  for (int degree = 0; degree <= 0; degree++)
  {
    current_test<P> name_("wav2nodal l = 5, " + mode[degree]);

    int const max_level = 5;

    connection_patterns conn(max_level);
    hierarchy_manipulator<P> hier(degree, domain);

    interpolation_manager<P> interp(domain, hier, conn);

    prog_opts options = make_opts("-l " + std::to_string(max_level) + " -n 0");
    options.degree = degree;
    pde_scheme<P> pde(options, domain);
    pde.add_initial(ic);

    discretization_manager<P> disc(pde, verbosity_level::quiet);

    // check the loaded nodes
    sparse_grid const &grid = disc.get_grid();

    vector2d<P> nodes = interp.nodes(grid);
    tassert(nodes.stride() == 2);
    tassert(nodes.num_strips() == 112 * (degree + 1) * (degree + 1));

    if (degree == 0)
      for (int i = 0; i < nodes.num_strips(); i++) {
        nodes[i][0] += 1.E-7;
        nodes[i][1] += 1.E-7;
      }

    std::vector<P> rref(nodes.num_strips());
    for (int i = 0; i < nodes.num_strips(); i++)
      rref[i] = ic.eval(nodes[i], 0);

    // using the reconstructor to compute reference data
    reconstruct_solution rec = disc.get_snapshot();
    vector2d<double> dnodes = vec2d(nodes);
    std::vector<double> ref(nodes.num_strips());
    rec.reconstruct(dnodes[0], nodes.num_strips(), ref.data());

    std::vector<P> vals(ref.size());
    interp.wav2nodal(grid, disc.get_conn(), disc.current_state().data(),
                     vals.data(), disc.get_terms().kwork);

    // std::cout << "  err = " << fm::diff_inf(vals, ref) << '\n';
    tassert(vals.size() == ref.size());
    for (auto i : indexof(ref))
      tcheckless(i, std::abs(vals[i] - ref[i]), tol);
  }

  P tols[4] = {0.0, 1.E-3, 5.E-6, 5.E-9};
  if constexpr (is_float<P>) tols[3] = 5.E-6; // hitting the max in float
  for (int degree = 1; degree <= 3; degree++)
  {
    current_test<P> name_("wav2nodal l = 5, " + mode[degree]);

    int const max_level = 5;

    connection_patterns conn(max_level);
    hierarchy_manipulator<P> hier(degree, domain);

    interpolation_manager<P> interp(domain, hier, conn);

    prog_opts options = make_opts("-l " + std::to_string(max_level) + " -n 0");
    options.degree = degree;
    pde_scheme<P> pde(options, domain);
    pde.add_initial(ic);

    discretization_manager<P> disc(pde, verbosity_level::quiet);

    // check the loaded nodes
    sparse_grid const &grid = disc.get_grid();

    vector2d<P> const &nodes = interp.nodes(grid);
    tassert(nodes.stride() == 2);
    tassert(nodes.num_strips() == 112 * (degree + 1) * (degree + 1));

    std::vector<P> ref(nodes.num_strips());
    for (int i = 0; i < nodes.num_strips(); i++)
      ref[i] = ic.eval(nodes[i], 0);

    std::vector<P> vals(ref.size());
    interp.wav2nodal(grid, disc.get_conn(), disc.current_state().data(),
                     vals.data(), disc.get_terms().kwork);

    // std::cout << "  err = " << fm::diff_inf(vals, ref) << '\n';
    tassert(vals.size() == ref.size());
    tcheckless(degree, fm::diff_inf(vals, ref), tols[degree]);
  }
}

template<typename P>
void interp_identity(P tol, int degree, int max_level)
{
  pde_domain<P> domain(2); // work in 2d
  separable_func<P> ic;
  ic.set(0, vectorize_t<P>([](P x)->P { return std::sin(x); }));
  ic.set(1, vectorize_t<P>([](P x)->P { return std::exp(x); }));

  std::map<int, std::string> mode = {{0, "constant"}, {1, "linear"},
                                     {2, "quadratic"}, {3, "cubic"}};

  current_test<P> name_("interp l = " + std::to_string(max_level) + ", " + mode[degree]);

  connection_patterns conn(max_level);
  hierarchy_manipulator<P> hier(degree, domain);

  interpolation_manager<P> interp(domain, hier, conn);

  prog_opts options = make_opts("-n 0");
  options.degree = degree;
  options.start_levels = {max_level, };
  pde_scheme<P> pde(options, domain);
  pde.add_initial(ic);

  discretization_manager<P> disc(pde, verbosity_level::quiet);

  // check the loaded nodes
  sparse_grid const &grid = disc.get_grid();

  vector2d<P> const &nodes = interp.nodes(grid);
  tassert(nodes.stride() == 2);

  std::vector<P> vals(nodes.num_strips());
  for (int64_t i = 0; i < nodes.num_strips(); i++)
    vals[i] = ic.eval(nodes[i], 0);

  std::vector<P> wav(disc.current_state().size());
  std::vector<P> t1(wav.size());
  interp.nodal2wav(grid, disc.get_conn(), P{1}, vals.data(), P{0}, wav.data(),
                   disc.get_terms().kwork, t1);

  // std::cout << " degree = " << degree << " level = " << max_level
  //           << "  err = " << fm::diff_inf(wav, disc.current_state()) << "\n";
  tcheckless(degree, fm::diff_inf(wav, disc.current_state()), tol);
}

template<typename P>
void interp_identity_domain(P tol, int degree, int max_level)
{
  pde_domain<P> domain({{-1, 1}, {0, 3}}); // work in 2d
  separable_func<P> ic;
  ic.set(0, vectorize_t<P>([](P x)->P { return std::sin(x); }));
  ic.set(1, vectorize_t<P>([](P x)->P { return std::exp(x); }));

  std::map<int, std::string> mode = {{0, "constant"}, {1, "linear"},
                                     {2, "quadratic"}, {3, "cubic"}};

  current_test<P> name_("interp l = " + std::to_string(max_level) + ", " + mode[degree] + " (domain)");

  connection_patterns conn(max_level);
  hierarchy_manipulator<P> hier(degree, domain);

  interpolation_manager<P> interp(domain, hier, conn);

  prog_opts options = make_opts("-dt 0 -n 0");
  options.degree = degree;
  options.start_levels = {max_level, };
  pde_scheme<P> pde(options, domain);
  pde.add_initial(ic);

  discretization_manager<P> disc(pde, verbosity_level::quiet);

  // check the loaded nodes
  sparse_grid const &grid = disc.get_grid();

  vector2d<P> const &nodes = interp.nodes(grid);
  tassert(nodes.stride() == 2);

  std::vector<P> vals(nodes.num_strips());
  for (int64_t i = 0; i < nodes.num_strips(); i++)
    vals[i] = ic.eval(nodes[i], 0);

  std::vector<P> wav(disc.current_state().size());
  std::vector<P> t1(wav.size());
  interp.nodal2wav(grid, disc.get_conn(), P{1}, vals.data(), P{0}, wav.data(),
                   disc.get_terms().kwork, t1);

  // std::cout << " degree = " << degree << " level = " << max_level
  //           << "  err = " << fm::diff_inf(wav, disc.current_state()) << "\n";
  tcheckless(degree, fm::diff_inf(wav, disc.current_state()), tol);
}

template<typename P>
void interp_identity()
{
  if constexpr (std::is_same_v<P, double>) {
    interp_identity<double>(1.E-1, 0, 5);
    interp_identity<double>(1.E-3, 1, 1);
    interp_identity<double>(1.E-5, 1, 6);
    interp_identity<double>(1.E-3, 2, 1);
    interp_identity<double>(1.E-7, 2, 5);
    interp_identity<double>(5.E-5, 3, 1);
    interp_identity<double>(1.E-8, 3, 4);

    interp_identity_domain<double>(1.E-3, 1, 6);
    interp_identity_domain<double>(5.E-5, 2, 5);
    interp_identity_domain<double>(5.E-6, 3, 4);
  } else {
    interp_identity<float>(1.E-1, 0, 5);
    interp_identity<float>(1.E-3, 1, 1);
    interp_identity<float>(2.E-5, 1, 5);
    interp_identity<float>(2.E-4, 2, 2);
    interp_identity<float>(2.E-6, 2, 4);
    interp_identity<float>(1.E-5, 3, 2);

    interp_identity_domain<float>(1.E-2, 1, 4);
    interp_identity_domain<float>(1.E-4, 2, 5);
  }
}

template<typename P>
void do_all_tests() {
  interp_wav2nodal<P>();
  interp_identity<P>();
}

int main(int argc, char **argv) {

  libasgard_runtime running_(argc, argv);

  all_tests global_("interpolation framework", " handles non-separable operators");

  #ifdef ASGARD_ENABLE_DOUBLE
  do_all_tests<double>();
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  do_all_tests<float>();
  #endif

  return 0;
}
