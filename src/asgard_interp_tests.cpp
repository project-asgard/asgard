#include "asgard_test_macros.hpp"

using namespace asgard;

template<typename P>
void interp_nodes() {

  P constexpr tol = (std::is_same_v<P, double>) ? 1.E-12 : 1.E-5;

  {
    current_test<P> name_("nodes constant");

    int const max_level = 2;
    interpolation_manager1d<P, 0> interp;
    tassert(not interp); // default constructor
    interp = interpolation_manager1d<P, 0>(max_level);
    tassert(!!interp);

    tassert(interp.nodes().num_strips() == 4);
    tassert(interp.nodes().stride() == 1);

    P const *r = interp.nodes()[0];
    std::vector<P> ref = {0, 0.5, 0.25, 0.75};
    for (auto i : indexof(ref))
      tcheckless(i, std::abs(r[i] - ref[i]), tol);
  }
  {
    current_test<P> name_("nodes linear");

    int const max_level = 1;
    interpolation_manager1d<P, 1> interp;
    tassert(not interp); // default constructor
    interp = interpolation_manager1d<P, 1>(max_level);
    tassert(!!interp);

    tassert(interp.nodes().num_strips() == 2);
    tassert(interp.nodes().stride() == 2);

    P const *r = interp.nodes()[0];
    std::vector<P> ref = {1.0/3.0, 2.0/3.0, 1.0/6.0, 5.0/6.0};
    for (auto i : indexof(ref))
      tassert(std::abs(r[i] - ref[i]) < tol);

    interp = interpolation_manager1d<P, 1>(max_level + 1);
    tassert(!!interp);

    tassert(interp.nodes().num_strips() == 4);
    tassert(interp.nodes().stride() == 2);

    r = interp.nodes()[0];
    ref = {1.0/3.0, 2.0/3.0, 1.0/6.0, 5.0/6.0, 1.0/12.0, 5.0/12.0, 7.0/12.0, 11.0/12.0};
    for (auto i : indexof(ref))
      tassert(std::abs(r[i] - ref[i]) < tol);
  }
  {
    current_test<P> name_("nodes quadratic");

    int const max_level = 1;
    interpolation_manager1d<P, 2> interp(max_level);

    tassert(interp.nodes().num_strips() == 2);
    tassert(interp.nodes().stride() == 3);

    P const *r = interp.nodes()[0];
    std::vector<P> ref = {0.2, 0.4, 0.8, 0.1, 0.6, 0.9};
    for (auto i : indexof(ref))
      tassert(std::abs(r[i] - ref[i]) < tol);

    interp = interpolation_manager1d<P, 2>(max_level + 1);
    tassert(!!interp);

    tassert(interp.nodes().num_strips() == 4);
    tassert(interp.nodes().stride() == 3);

    r = interp.nodes()[0];
    ref = {0.2, 0.4, 0.8, 0.1, 0.6, 0.9,
           0.05, 0.3, 0.45, 0.55, 0.8, 0.95};
    for (auto i : indexof(ref))
      tassert(std::abs(r[i] - ref[i]) < tol);
  }
  {
    current_test<P> name_("nodes cubic");

    int const max_level = 2;
    interpolation_manager1d<P, 3> interp(max_level);

    tassert(interp.nodes().num_strips() == 4);
    tassert(interp.nodes().stride() == 4);

    P const *r = interp.nodes()[0];
    std::vector<P> ref = {1.0/5.0, 2.0/5.0, 3.0/5.0, 4.0/5.0,
                          1.0/10.0, 3.0/10.0, 7.0/10.0, 9.0/10.0,
                          1.0/20.0, 3.0/20.0, 7.0/20.0, 9.0/20.0,
                          11.0/20.0, 13.0/20.0, 17.0/20.0, 19.0/20.0};
    for (auto i : indexof(ref))
      tassert(std::abs(r[i] - ref[i]) < tol);
  }
}

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

  {
    int constexpr degree = 1;
    current_test<P> name_("wav2nodal l = 1, linear");

    int const max_level = 1;

    connection_patterns conn(max_level);

    interpolation_manager<P> interp(domain, conn, degree);

    prog_opts options = make_opts("-l 1 -dt 0 -n 0");
    options.degree = degree;
    pde_scheme<P> pde(options, domain);
    pde.add_initial(ic);

    discretization_manager<P> disc(pde, verbosity_level::quiet);

    // check the loaded nodes
    sparse_grid const &grid = disc.get_grid();

    vector2d<P> const &nodes = interp.nodes(grid);
    tassert(nodes.stride() == 2);
    tassert(nodes.num_strips() == 12);

    // check the generated nodes
    std::vector<P> const expected_nodes = {
      1.0/3.0, 1.0/3.0, 1.0/3.0, 2.0/3.0, 2.0/3.0, 1.0/3.0, 2.0/3.0, 2.0/3.0,
      1.0/3.0, 1.0/6.0, 1.0/3.0, 5.0/6.0, 2.0/3.0, 1.0/6.0, 2.0/3.0, 5.0/6.0,
      1.0/6.0, 1.0/3.0, 1.0/6.0, 2.0/3.0, 5.0/6.0, 1.0/3.0, 5.0/6.0, 2.0/3.0,
    };
    for (size_t i = 0; i < expected_nodes.size(); i++)
      tcheckless(i, std::abs(nodes[i/2][i%2] - expected_nodes[i]), tol);

    // using the reconstructor to compute reference data
    vector2d<double> dnodes = vec2d(nodes);
    reconstruct_solution rec = disc.get_snapshot();
    std::vector<double> ref(nodes.num_strips());
    rec.reconstruct(dnodes[0], nodes.num_strips(), ref.data());

    std::vector<P> vals;
    interp.wav2nodal(grid, disc.get_conn(), disc.current_state(), vals, disc.get_terms().kwork);

    tassert(vals.size() == ref.size());
    for (auto i : indexof(ref))
      tcheckless(i, std::abs(vals[i] - ref[i]), tol);
  }
  {
    domain = pde_domain<P>({{-1, 1}, {0, 3}});
    int constexpr degree = 1;
    current_test<P> name_("wav2nodal l = 1, linear (domain)");

    int const max_level = 1;

    connection_patterns conn(max_level);

    interpolation_manager<P> interp(domain, conn, degree);

    prog_opts options = make_opts("-l 1 -dt 0 -n 0");
    options.degree = degree;
    options.start_levels = {max_level, };
    pde_scheme<P> pde(options, domain);
    pde.add_initial(ic);

    discretization_manager<P> disc(pde, verbosity_level::quiet);

    // check the loaded nodes
    sparse_grid const &grid = disc.get_grid();

    vector2d<P> const &nodes = interp.nodes(grid);
    tassert(nodes.stride() == 2);
    tassert(nodes.num_strips() == 12);

    // check the generated nodes
    std::vector<P> const expected_nodes = {
      1.0/3.0, 1.0/3.0, 1.0/3.0, 2.0/3.0, 2.0/3.0, 1.0/3.0, 2.0/3.0, 2.0/3.0,
      1.0/3.0, 1.0/6.0, 1.0/3.0, 5.0/6.0, 2.0/3.0, 1.0/6.0, 2.0/3.0, 5.0/6.0,
      1.0/6.0, 1.0/3.0, 1.0/6.0, 2.0/3.0, 5.0/6.0, 1.0/3.0, 5.0/6.0, 2.0/3.0,
    };
    for (size_t i = 0; i < expected_nodes.size(); i++) {
      if (i % 2 == 0) {
        tcheckless(i, std::abs(nodes[i/2][i%2] + 1 - 2 * expected_nodes[i]), tol);
      } else {
        tcheckless(i, std::abs(nodes[i/2][i%2] - 3 * expected_nodes[i]), tol);
      }
    }

    // using the reconstructor to compute reference data
    vector2d<double> dnodes = vec2d(nodes);
    reconstruct_solution rec = disc.get_snapshot();
    std::vector<double> ref(nodes.num_strips());
    rec.reconstruct(dnodes[0], nodes.num_strips(), ref.data());

    std::vector<P> vals;
    interp.wav2nodal(grid, disc.get_conn(), disc.current_state(), vals, disc.get_terms().kwork);

    tassert(vals.size() == ref.size());
    for (auto i : indexof(ref))
      tcheckless(i, std::abs(vals[i] - ref[i]), tol);
  }

  std::map<int, std::string> mode = {{0, "constant"}, {1, "linear"}, {2, "quadratic"}, {3, "cubic"}};

  domain = pde_domain<P>(2);
  for (int degree = 0; degree <= 3; degree++)
  {
    current_test<P> name_("wav2nodal l = 5, " + mode[degree]);

    int const max_level = 5;

    connection_patterns conn(max_level);

    interpolation_manager<P> interp(domain, conn, degree);

    prog_opts options = make_opts("-l 5 -dt 0 -n 0");
    options.degree = degree;
    pde_scheme<P> pde(options, domain);
    pde.add_initial(ic);

    discretization_manager<P> disc(pde, verbosity_level::quiet);

    // check the loaded nodes
    sparse_grid const &grid = disc.get_grid();

    vector2d<P> const &nodes = interp.nodes(grid);
    tassert(nodes.stride() == 2);
    tassert(nodes.num_strips() == 112 * (degree + 1) * (degree + 1));

    // using the reconstructor to compute reference data
    reconstruct_solution rec = disc.get_snapshot();
    vector2d<double> dnodes = vec2d(nodes);
    std::vector<double> ref(nodes.num_strips());
    rec.reconstruct(dnodes[0], nodes.num_strips(), ref.data());

    std::vector<P> vals;
    interp.wav2nodal(grid, disc.get_conn(), disc.current_state(), vals, disc.get_terms().kwork);

    tassert(vals.size() == ref.size());
    for (auto i : indexof(ref))
      tcheckless(i, std::abs(vals[i] - ref[i]), tol);
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

  interpolation_manager<P> interp(domain, conn, degree);

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

  interp.nodal2hier(grid, disc.get_conn(), vals, disc.get_terms().kwork);

  std::vector<P> wav;
  interp.hier2wav(grid, disc.get_conn(), vals, wav, disc.get_terms().kwork);

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

  interpolation_manager<P> interp(domain, conn, degree);

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

  interp.nodal2hier(grid, disc.get_conn(), vals, disc.get_terms().kwork);

  std::vector<P> wav;
  interp.hier2wav(grid, disc.get_conn(), vals, wav, disc.get_terms().kwork);

  // std::cout << " degree = " << degree << " level = " << max_level
  //           << "  err = " << fm::diff_inf(wav, disc.current_state()) << "\n";
  tcheckless(degree, fm::diff_inf(wav, disc.current_state()), tol);
}

template<typename P>
void interp_identity()
{
  if constexpr (std::is_same_v<P, double>) {
    interp_identity<double>(1.E-1, 0, 6);
    interp_identity<double>(1.E-5, 1, 6);
    interp_identity<double>(5.E-5, 2, 8);
    interp_identity<double>(5.E-9, 3, 6);

    interp_identity_domain<double>(1.E-3, 1, 6);
    interp_identity_domain<double>(5.E-3, 2, 8);
    interp_identity_domain<double>(1.E-7, 3, 6);
  } else {
    interp_identity<float>(1.E-5, 1, 6);
    interp_identity<float>(5.E-5, 2, 8);
    interp_identity<float>(1.E-5, 3, 6);

    interp_identity_domain<float>(1.E-3, 1, 6);
    interp_identity_domain<float>(3.E-4, 3, 6);
  }
}

template<typename P>
void do_all_tests() {
  interp_nodes<P>();
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
