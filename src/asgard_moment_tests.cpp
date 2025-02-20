#include "tests_general.hpp"

using P = asgard::default_precision;

using namespace asgard;

class somepde : public PDE<P> {
public:
  somepde(std::vector<P> const &drange, int level, int degree,
          std::vector<std::function<P(P)>> const funcs)
    : funcs_(std::move(funcs))
  {
    expect(drange.size() % 2 == 0);
    expect(drange.size() / 2 == funcs_.size());
    expect(not funcs.empty());
    expect(static_cast<int>(funcs_.size()) <= max_num_dimensions);

    int const ndims   = static_cast<int>(funcs_.size());
    this->interp_nox_ = [](P, std::vector<P> const &, std::vector<P> &) -> void {};

    std::vector<dimension<P>> dims;
    dims.reserve(ndims);

    for (int i = 0; i < ndims; i++)
    {
      dims.push_back(
        dimension<P>(drange[2*i], drange[2*i + 1], level, degree,
          [ff = funcs_[i]](fk::vector<P> const &x, P const) -> fk::vector<P> {
            fk::vector<P> fx(x.size());
            for (auto k : indexof(x))
              fx[k] = ff(x[k]);
            return fx;
          },
          nullptr, std::string("x_") + std::to_string(i))
      );
    }

    prog_opts opts;

    this->initialize(opts, ndims, 0, dims,
                     term_set<P>{}, std::vector<source<P>>{},
                     std::vector<md_func_type<P>>{},
                     get_dt_, false, false);
  }

  std::vector<std::function<P(P)>> funcs_;

  static P get_dt_(dimension<P> const &) { return 0.0; }
};

double test_moments(std::vector<P> const &drange, int level, int degree, int num_mom,
                    std::vector<std::function<P(P)>> const base,
                    std::vector<std::function<P(P)>> const moments)
{
  int const num_moms = static_cast<int>(moments.size());

  discretization_manager<P> disc( std::make_unique<somepde>(drange, level, degree, base) );

  std::vector<std::unique_ptr<discretization_manager<P>>> dmoms;

  for (int m = 0; m < num_moms; m++)
  {
    dmoms.emplace_back(
      std::make_unique<discretization_manager<P>>(
        std::make_unique<somepde>(
          std::vector<P>{drange[0], drange[1]}, level, degree,
                         std::vector<std::function<P(P)>>{moments[m], })));
  }

  auto const &pde   = disc.get_pde();
  auto const &dims  = pde.get_dimensions();
  auto const &grid  = disc.get_grid();
  auto const &table = grid.get_table();

  int const level0  = dims[0].get_level();

  moments1d<P> moms(num_mom, degree, pde.max_level(), dims);

  std::vector<P> raw_moments;
  moms.project_moments(level0, disc.current_state(), table, raw_moments);

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
    // reorder the nodes for the reference solution, to match the order of the 1d moments
    vector2d<int> cells1d = dmoms[m]->get_grid().get_table().get_cells();
    dimension_sort dsort(cells1d);
    auto const &state1d = dmoms[m]->current_state();

    std::vector<P> ref(state1d.size());
    {
      int const dim = 0;
      int64_t size = cells1d.num_strips();
      span2d<P const> sstate(degree + 1, size, state1d.data());
      span2d<P> sref(degree + 1, size, ref.data());
      for (auto i : indexof(size))
        std::copy_n(sstate[dsort.map(dim, i)], degree + 1, sref[i]);
    }

    err = std::max(err, fm::diff_inf(vmoms[m], ref));

    // also include comparison with the solution of a single moment
    std::vector<P> single_mom;
    moms.project_moment(m, level0, disc.current_state(), table, single_mom);
    err = std::max(err, fm::diff_inf(single_mom, ref));
  }

  return err;
}


TEST_CASE("compute moments", "[moments]")
{
  double tol = (std::is_same_v<P, double>) ? 5.E-14 : 5.E-6;

  SECTION("2D")
  {
    std::vector<std::function<P(P)>> base(2), moms(3);

    base[0] = [](P x) -> P { return std::sin(x); };
    base[1] = [](P) -> P { return 1.0; };

    moms[0] = [](P x) -> P { return 3.0 * std::sin(x); };
    moms[1] = [](P x) -> P { return -1.5 * std::sin(x); };
    moms[2] = [](P x) -> P { return 3.0 * std::sin(x); };

    for (int d = 0; d < 4; d++) {
      for (int l = 1; l < 7; l++) {
        double err = test_moments({-2, 1, -2, 1}, l, d, 3, base, moms);
        REQUIRE(err < tol);
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
        REQUIRE(err < 5 * tol);
      }
    }
  }

  SECTION("3D")
  {
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
        REQUIRE(err < 10 * tol);
      }
    }
  }

  SECTION("4D")
  {
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
        REQUIRE(err < tol);
      }
    }
  }
}

struct distribution_test_init
{
  distribution_test_init() { initialize_distribution(); }
  ~distribution_test_init() { finalize_distribution(); }
};

#ifdef ASGARD_USE_MPI
static distribution_test_init const distrib_test_info;
#endif
