#include "asgard_test_macros.hpp"
#include "asgard_small_mats.hpp"

using namespace asgard;

template<typename P>
void interp_wav2nodal() {
  P constexpr tol = (std::is_same_v<P, double>) ? 1.E-12 : 1.E-5;

  pde_domain<P> domain(2); // work in 2d
  separable_func<P> ic({1, 1});
  ic.set(dimension_id{0}, vectorize_t<P>([](P x)->P { return std::sin(x); }));
  ic.set(dimension_id{1}, vectorize_t<P>([](P x)->P { return std::exp(x); }));
  ic.set_time_non_separable();

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
    interp.wav2nodal(grid, disc.current_state().data(), vals.data(), disc.get_terms().kwork);

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
    interp.wav2nodal(grid, disc.current_state().data(), vals.data(), disc.get_terms().kwork);

    // std::cout << "  err = " << fm::diff_inf(vals, ref) << '\n';
    tassert(vals.size() == ref.size());
    tcheckless(degree, fm::diff_inf(vals, ref), tols[degree]);
  }
}

template<typename P>
void interp_identity(P tol, int degree, int max_level)
{
  pde_domain<P> domain(2); // work in 2d
  separable_func<P> ic = separable_func<P>::const_one(number_of_dimensions{2});
  ic.set(dimension_id{0}, vectorize_t<P>([](P x)->P { return std::sin(x); }));
  ic.set(dimension_id{1}, vectorize_t<P>([](P x)->P { return std::exp(x); }));

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
  auto ic = separable_func<P>::const_one(number_of_dimensions{2});
  static_assert(std::is_same_v<decltype(ic), separable_func<P>>,
                "incorrect return type for separable_func<P>::const_one");
  ic.set(asgard::dimension_id{0}, vectorize_t<P>([](P x)->P { return std::sin(x); }));
  ic.set(asgard::dimension_id{1}, vectorize_t<P>([](P x)->P { return std::exp(x); }));

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
P maxwellian1d(P n, P u, P theta, P v)
{
  P const d = v - u;
  return n * std::exp(-P{0.5} * d * d / theta) / std::sqrt(2 * PI * theta);
}

template<typename P>
void hybrid_maxwellian_collision()
{
  current_test<P> name_("hybrid Maxwellian collision preserves moments");

  int constexpr degree = 2;
  int constexpr x_level = 2;
  int constexpr v_level = 3;
  P const nu = P{2};
  P const dt = P{0.2} / nu;

  pde_domain<P> domain(position_dims{1}, velocity_dims{1},
                       {domain_range{0, 1}, domain_range{-10, 10}});

  prog_opts options;
  options.degree = degree;
  options.start_levels = {x_level, v_level};
  options.dt = dt;
  options.num_time_steps = 1;
  options.step_method = time_method::back_euler;
  options.solver = solver_method::scaled_identity;

  pde_scheme<P> pde(options, domain);
  auto const basis_mats_double = legendre::generate_multi_wavelets(degree);
  std::array<std::vector<P>, 4> basis_mats;
  for (int i = 0; i < 4; i++)
    basis_mats[i].assign(basis_mats_double[i].begin(),
                         basis_mats_double[i].end());

  moment_id const im0 = pde.register_moment(moment{0});
  moment_id const im1 = pde.register_moment(moment{1});
  moment_id const im2 = pde.register_moment(moment{2});
  std::vector<moment_id> const mids = {im0, im1, im2};

  auto non_equilibrium = [](P, vector2d<P> const &nodes, std::vector<P> &vals)
    -> void {
    assert(vals.size() == static_cast<size_t>(nodes.num_strips()));

    for (int64_t i = 0; i < nodes.num_strips(); i++) {
      P const x = nodes[i][0];
      P const v = nodes[i][1];

      P const n = P{1} + P{0.2} * std::sin(P{0.3} * x);
      vals[i] = P{0.6} * maxwellian1d(P{1}, P{-0.6}, P{0.5}, v)
              + P{0.4} * maxwellian1d(P{1}, P{0.7},  P{1.0}, v);
      vals[i] *= n;
    }
  };

  md_mom_and_idx_func<P> hybrid_maxwellian =
      [=](P, vector2d<P> const &nodes, momentset<P> const &moments,
          std::vector<int> const &indexes, std::vector<P> &vals) -> void {
    std::vector<P> const &m0 = moments[im0];
    std::vector<P> const &m1 = moments[im1];
    std::vector<P> const &m2 = moments[im2];

    assert(vals.size() == static_cast<size_t>(nodes.num_strips()));
    assert(m0.size() == vals.size());
    assert(m1.size() == vals.size());
    assert(m2.size() == vals.size());
    assert(indexes.size() % 2 == 0);

    int constexpr pdof = degree + 1;
    int constexpr block_size = pdof * pdof;
    int64_t const num_cells = static_cast<int64_t>(indexes.size() / 2);

    auto level_position = [](int idx) -> std::array<int64_t, 2> {
      int64_t level = 0;
      while ((int64_t{1} << level) <= idx)
        level++;
      int64_t const pos = (level == 0) ? 0 : idx - (int64_t{1} << (level - 1));
      return {level, pos};
    };

    auto wavelet_maxwell =
        [&](int64_t const v_level, int64_t const v_pos, P const u, P const theta,
            std::vector<P> &mulin, std::vector<P> &mulout) -> void {
      P const a = domain.xleft(1);
      P const b = domain.xright(1);
      P const dv = (v_level == 0) ? b - a : (b - a) / (int64_t{1} << (v_level - 1));
      P const loca = a + v_pos * dv;

      if (v_level == 0) {
        P const mid = P{0.5} * (b + a);

        P const I0 = 1;
        P const I1 = u;
        P const I2 = theta + u * u;

        P const jv  = std::sqrt(P{2} / dv);
        P const jv2 = jv * jv;
        mulout[0] = std::sqrt(P{0.5}) * jv * I0;
        mulout[1] = std::sqrt(P{1.5}) * jv * jv2 * (-I0 * mid + I1);
        mulout[2] = std::sqrt(P{0.625}) * jv
                  * ((P{3} * jv2 * jv2 * mid * mid - P{1}) * I0
                     - P{6} * jv2 * jv2 * mid * I1
                     + P{3} * jv2 * jv2 * I2);
      } else {
        for (int side = 0; side < 2; side++) {
          P const left  = loca + side * dv / P{2};
          P const right = loca + (side + 1) * dv / P{2};
          P const mid = P{0.5} * (left + right);

          P const z_left  = (left - u) / std::sqrt(theta);
          P const z_right = (right - u) / std::sqrt(theta);

          P const T0_left  = P{0.5} * std::erf(z_left / std::sqrt(P{2}));
          P const T0_right = P{0.5} * std::erf(z_right / std::sqrt(P{2}));
          P const T1_left  = std::exp(-P{0.5} * z_left * z_left) / std::sqrt(P{2} * PI);
          P const T1_right = std::exp(-P{0.5} * z_right * z_right) / std::sqrt(P{2} * PI);
          P const C0 = T0_right - T0_left;
          P const C1 = -(T1_right - T1_left);
          P const C2 = -(z_right * T1_right - z_left * T1_left) + C0;

          P const I0 = C0;
          P const I1 = u * C0 + std::sqrt(theta) * C1;
          P const I2 = u * u * C0 + P{2} * u * std::sqrt(theta) * C1
                     + theta * C2;

          P const jv  = std::sqrt(P{4} / dv);
          P const jv2 = jv * jv;
          P const jv4 = jv2 * jv2;

          mulin[0] = std::sqrt(P{0.5}) * jv * I0;
          mulin[1] = std::sqrt(P{1.5}) * jv * jv * jv * (-I0 * mid + I1);
          mulin[2] = std::sqrt(P{0.625}) * jv
                   * ((P{3} * jv4 * mid * mid - P{1}) * I0
                      - P{6} * jv4 * mid * I1 + P{3} * jv4 * I2);

          if (side == 0)
            smmat::gemv(pdof, pdof, basis_mats[2].data(), mulin.data(),
                        mulout.data());
          else
            smmat::gemv1(pdof, pdof, basis_mats[3].data(), mulin.data(),
                         mulout.data());
        }
      }
    };

    #pragma omp parallel
    {
      std::vector<P> mulin(pdof);
      std::vector<P> mulout(pdof);

      #pragma omp for
      for (int64_t cell = 0; cell < num_cells; cell++) {
        int const velocity_index = indexes[2 * cell + 1];
        auto const [velocity_level, velocity_pos] = level_position(velocity_index);

        for (int ix = 0; ix < pdof; ix++) {
          int64_t const base = cell * block_size + ix * pdof;

          P const n = m0[base];
          P const u = m1[base] / n;
          P const theta = std::max(m2[base] / n - u * u, P{1.E-8});

          wavelet_maxwell(velocity_level, velocity_pos, u, theta, mulin, mulout);

          for (int iv = 0; iv < pdof; iv++)
            vals[base + iv] = nu * n * mulout[iv];
        }
      }
    }
  };

  std::vector<term_1d<P>> nu_identity;
  nu_identity.emplace_back(term_volume<P>{nu});
  nu_identity.emplace_back(term_identity{});
  pde += term_md<P>(std::move(nu_identity));
  pde += source<P>(hybrid_maxwellian, mids);
  pde.set_initial(non_equilibrium);

  discretization_manager<P> disc(pde, verbosity_level::quiet);

  std::array<std::vector<P>, 3> before = {
      disc.get_moment(im0),
      disc.get_moment(im1),
      disc.get_moment(im2),
  };

  tassert(disc.advance_time(1));

  std::array<std::vector<P>, 3> after = {
      disc.get_moment(im0),
      disc.get_moment(im1),
      disc.get_moment(im2),
  };

  P const tol = (is_double<P>) ? P{1.E-6} : P{1.E-4};
  for (int i = 0; i < 3; i++) {
    tassert(before[i].size() == after[i].size());
    tcheckless(i, fm::diff_inf(before[i], after[i]), tol);
  }
}

template<typename P>
void do_all_tests() {
  interp_wav2nodal<P>();
  interp_identity<P>();
  hybrid_maxwellian_collision<P>();
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
