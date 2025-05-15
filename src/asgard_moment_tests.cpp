#include "asgard_test_macros.hpp"

using P = asgard::default_precision;

using namespace asgard;

double test_moments(std::vector<P> const &drange, int level, int degree, int num_mom,
                    std::vector<std::function<P(P)>> const base,
                    std::vector<std::function<P(P)>> const moments)
{
  expect(drange.size() % 2 == 0);
  expect(drange.size() / 2 == base.size());
  expect(not base.empty());

  std::vector<domain_range> ranges;
  for (size_t i = 0; i < drange.size(); i += 2)
    ranges.emplace_back(drange[i], drange[i + 1]);

  prog_opts options;
  options.default_degree = degree;
  options.start_levels = {level, };
  options.default_dt        = 0.01;
  options.default_stop_time = 1;

  pde_domain domain(position_dims{1},
                    velocity_dims{static_cast<int>(base.size()) - 1},
                    ranges);

  // make the reference PDE
  pde_scheme<P> pde(options, domain);

  separable_func<P> vbase(std::vector<P>(base.size(), 1));
  for (int d : iindexof(base))
    vbase.set(d, vectorize_t<P>(base[d]));

  pde.add_initial(vbase);

  discretization_manager<P> disc(pde, verbosity_level::quiet);

  int const num_moms = static_cast<int>(moments.size());

  moments1d<P> moms(num_mom, degree, disc.max_level(), domain);

  std::vector<std::unique_ptr<discretization_manager<P>>> dmoms;

  for (int m = 0; m < num_moms; m++)
  {
    pde_scheme<P> pde2(options, pde_domain({ranges[0], }));

    pde2.add_initial(separable_func<P>({vectorize_t<P>(moments[m]), }));

    dmoms.emplace_back(std::make_unique<discretization_manager<P>>
                       (std::move(pde2), verbosity_level::quiet));
  }

  std::vector<P> raw_moments;
  moms.project_moments(disc.get_grid(), disc.current_state(), raw_moments);

  // the raw_moments are stored interlaces, e.g., cell0-mom0, cell0-mom1, cell1-mom0 ...
  // splitting into separate vectors, for easier comparison against the reference states
  int num_comp = 1 + (pde.num_dims() - 1) * (num_mom - 1);
  std::vector<std::vector<P>> vmoms(num_comp, std::vector<P>(raw_moments.size() / num_comp));
  {
    std::vector<decltype(vmoms.front().begin())> imoms(num_comp);
    for (int m : iindexof(num_comp))
      imoms[m] = vmoms[m].begin();

    auto im = raw_moments.begin();
    while (imoms.front() != vmoms[0].end()) {
      for (int i : iindexof(num_comp))
      {
        imoms[i] = std::copy_n(im, degree + 1, imoms[i]);
        std::advance(im, degree + 1);
      }
    }
  }

  P err = 0;
  for (int m = 0; m < num_comp; m++)
  {
    std::vector<P> const &ref = dmoms[m]->current_state();

    err = std::max(err, fm::diff_inf(vmoms[m], ref));

    // also include comparison with the solution of a single moment
    std::vector<P> single_mom;
    moms.project_moment(m, disc.get_grid(), disc.current_state(), single_mom);
    err = std::max(err, fm::diff_inf(single_mom, ref));
  }

  return err;
}

void test_compute_moments()
{
  double constexpr tol = (std::is_same_v<P, double>) ? 5.E-14 : 5.E-6;

  {
    current_test<P> name_("compute moments", 2);
    std::vector<std::function<P(P)>> base(2), moms(3);

    base[0] = [](P x) -> P { return std::sin(x); };
    base[1] = [](P) -> P { return 1.0; };

    moms[0] = [](P x) -> P { return 3.0 * std::sin(x); };
    moms[1] = [](P x) -> P { return -1.5 * std::sin(x); };
    moms[2] = [](P x) -> P { return 3.0 * std::sin(x); };

    for (int d = 0; d < 4; d++) {
      for (int l = 1; l < 7; l++) {
        double err = test_moments({-2, 1, -2, 1}, l, d, 3, base, moms);
        tassert(err < tol);
      }
    }

    base[0] = [](P x) -> P { return std::sin(x); };
    base[1] = [](P v) -> P { return std::cos(v); };

    moms[0] = [](P x) -> P { return 1.75076841163357 * std::sin(x); };
    moms[1] = [](P x) -> P { return -2.067472642818473e-02 * std::sin(x); };
    moms[2] = [](P x) -> P { return 0.393141134391177 * std::sin(x); };

    for (int d = 0; d < 4; d++) {
      std::vector<std::function<P(P)>> rmoms;
      for (int m = 0; m < std::min(d+1, 3); m++)
        rmoms.push_back(moms[m]);
      for (int l = 1; l < 7; l++) {
        double err = test_moments({-2, 1, -2, 1}, l, d, std::min(d+1, 3), base, rmoms);
        tassert(err < 5 * tol);
      }
    }
  }
  {
    current_test<P> name_("compute moments", 3);
    std::vector<std::function<P(P)>> base(3), moms(5);

    base[0] = [](P x) -> P { return std::sin(x); };
    base[1] = [](P v) -> P { return std::cos(v); };
    base[2] = [](P v) -> P { return std::exp(v); };

    moms[0] = [](P x) -> P { return 7.021176657759206 * 1.75076841163357 * std::sin(x); };
    moms[1] = [](P x) -> P { return 7.021176657759206 * -2.067472642818473e-02 * std::sin(x); };
    moms[2] = [](P x) -> P { return 8.124814981273536 * 1.75076841163357 * std::sin(x); };
    moms[3] = [](P x) -> P { return 7.021176657759206 * 0.393141134391177 * std::sin(x); };
    moms[4] = [](P x) -> P { return 12.93871499200409 * 1.75076841163357 * std::sin(x); };

    for (int d = 0; d < 4; d++) {
      std::vector<std::function<P(P)>> rmoms;
      int const npow = std::min(d+1, 3);
      int const nm   = 1 + 2 * (npow - 1);
      for (int m = 0; m < nm; m++)
        rmoms.push_back(moms[m]);
      for (int l = 1; l < 7; l++) {
        double err = test_moments({-2, 1, -2, 1, -1, 2}, l, d, npow, base, rmoms);
        tassert(err < 10 * tol);
      }
    }
  }
  {
    current_test<P> name_("compute moments", 4);
    std::vector<std::function<P(P)>> base(4), moms(7);

    base[0] = [](P x) -> P { return std::sin(x); };
    base[1] = [](P v) -> P { return std::cos(v); };
    base[2] = [](P v) -> P { return std::exp(v); };
    base[3] = [](P v) -> P { return std::sin(v); };

    moms[0] = [](P x) -> P {
      return -4.347843211251236e-02 * 7.021176657759206 * 1.75076841163357 * std::sin(x);
    };
    moms[1] = [](P x) -> P {
      return -4.347843211251236e-02 * 7.021176657759206 * -2.067472642818473e-02 * std::sin(x);
    };
    moms[2] = [](P x) -> P {
      return -4.347843211251236e-02 * 8.124814981273536 * 1.75076841163357 * std::sin(x);
    };
    moms[3] = [](P x) -> P {
      return 6.162820236651310e-02 * 7.021176657759206 * 1.75076841163357 * std::sin(x);
    };
    moms[4] = [](P x) -> P {
      return -4.347843211251236e-02 * 7.021176657759206 * 0.393141134391177 * std::sin(x);
    };
    moms[5] = [](P x) -> P {
      return -4.347843211251236e-02 * 12.93871499200409 * 1.75076841163357 * std::sin(x);
    };
    moms[6] = [](P x) -> P {
      return -8.908119100126307e-03 * 7.021176657759206 * 1.75076841163357 * std::sin(x);
    };

    for (int d = 0; d < 4; d++) {
      std::vector<std::function<P(P)>> rmoms;
      int const npow = std::min(d+1, 3);
      int const nm   = 1 + 3 * (npow - 1);
      for (int m = 0; m < nm; m++)
        rmoms.push_back(moms[m]);
      for (int l = 1; l < 7; l++) {
        double err = test_moments({-2, 1, -2, 1, -1, 2, -0.5, 0.4}, l, d, npow, base, rmoms);
        tassert(err < tol);
      }
    }
  }
}

int main(int argc, char **argv)
{
  libasgard_runtime running_(argc, argv);

  all_tests global_("computing moments", " field integrals in velocity domain");

  test_compute_moments();

  return 0;
}
