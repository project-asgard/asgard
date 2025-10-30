#include "asgard_test_macros.hpp"

// using P = asgard::default_precision;

using namespace asgard;

template<typename P>
struct test_function
{
  std::string name;
  domain_range range;
  std::function<P(P)> base; // base signature
  std::function<void(std::vector<P> const &, P, std::vector<P> &)> func;
  std::array<P, 4> moms; // moments
};

std::vector<test_function<float>> ffuncs;
std::vector<test_function<double>> dfuncs;

template<typename P>
std::vector<test_function<P>> const &get_functions()
{
  std::vector<test_function<P>> &funcs = []() -> std::vector<test_function<P>> & {
      if constexpr (is_double<P>)
        return dfuncs;
      else
        return ffuncs;
    }();

  if (not funcs.empty()) return funcs;
  // initialize the functions

  funcs.reserve(10); // maybe an overkill
  // exp(x) over (-1, 1)
  funcs.push_back(test_function<P>{"exp(x)", {-1, 1}, [](P x)->P{ return std::exp(x); }, nullptr,
                  {2.350402387287603, 0.735758882342885, 0.878884622601834, 0.449507401824987}});
  // sin(x) over (0, 1)
  funcs.push_back(test_function<P>{"sin(x)", {0, 1}, [](P x)->P{ return std::sin(x); }, nullptr,
                  {0.459697694131860, 0.301168678939757, 0.223244275483933, 0.177098574917009}});
  // cos(x) over (1, 2)
  funcs.push_back(test_function<P>{"cos(x)", {1, 2}, [](P x)->P{ return std::cos(x); }, nullptr,
                  {6.782644201778519e-02, 2.067472642818477e-02, -8.512611946558914e-02, -0.305808884941680}});
  // sin(x) over (-1, 1)
  funcs.push_back(test_function<P>{"sin(x)", {-1, 1}, [](P x)->P{ return std::sin(x); }, nullptr,
                  {0, 0.602337357879514, 0, 0.354197149834018}});
  // exp(x) over (0.5, 1.5)
  funcs.push_back(test_function<P>{"exp(x)", {0.5, 1.5}, [](P x)->P{ return std::exp(x); }, nullptr,
                  {2.832967799637936, 3.065205170519097, 3.541209749547421, 4.295981204911191}});

  for (auto &f : funcs)
    f.func = vectorize_t<P>(f.base);

  return funcs;
}

template<typename P>
pde_scheme<P> make_pde(pde_domain<P> const &domain, int level, int degree,
                       std::vector<std::function<void(std::vector<P> const &, P, std::vector<P> &)>> const &funcs)
{
  expect(static_cast<int>(funcs.size()) == domain.num_dims());

  prog_opts options;
  options.default_degree = degree;
  options.default_start_levels = {level, };

  options.num_time_steps = 0;

  pde_scheme<P> pde(options, domain);

  pde.add_initial(separable_func<P>(funcs));

  return pde;
}

struct test_props {
  int degree = 0;
  int level = 0;
  std::vector<double> tols;
};

template<typename P>
void test_case(std::string info, int num_pos, std::vector<int> ifuncs,
               std::vector<moment> const &moms,
               std::vector<test_props> const &props)
{
  expect(num_pos + 1 <= static_cast<int>(ifuncs.size()));
  for (auto const &p : props) {
    expect(p.tols.size() == moms.size());
  }

  int const num_vel = static_cast<int>(ifuncs.size() - num_pos);

  current_test<P> name_("compute moments " + std::to_string(num_pos) + "x"
                                           + std::to_string(num_vel) + "v  ("
                                           + info + ")");
  tassert(num_vel > 0);

  auto const &funcs = get_functions<P>();

  std::vector<domain_range> ranges; ranges.reserve(ifuncs.size());
  for (auto i : ifuncs) ranges.push_back(funcs[i].range);
  pde_domain<P> domain(position_dims{num_pos}, velocity_dims{num_vel});
  domain.set(ranges);

  moments_list mlist;
  for (auto const &m : moms) mlist.add_moment(m);

  moment_manager<P> manager;
  tassert(not manager);

  for (auto const &p : props)
  {
    int const degree = p.degree;
    int const level  = p.level;

    std::vector<svector_func1d<P>> f1d; f1d.reserve(ifuncs.size());
    for (auto i : ifuncs) f1d.push_back(funcs[i].func);
    auto pde = make_pde<P>(domain, level, degree, f1d);
    f1d.resize(num_pos);
    auto pos_pde = make_pde<P>(domain.position_domain(), level, degree, f1d);

    std::vector<moment_id> mid; mid.reserve(moms.size());
    for (auto const &m : moms)
      mid.push_back( pde.register_moment(m) );

    discretization_manager<P> disc(pde, verbosity_level::quiet);
    discretization_manager<P> pos_disc(pos_pde, verbosity_level::quiet);

    std::vector<P> const &ref = pos_disc.current_state();

    for (int i : iindexof(moms)) {
      P scale = 1;
      for (int d = num_pos; d < domain.num_dims(); d++)
        scale *= funcs[ifuncs[d]].moms[ moms[i].pows[d - num_pos] ];

      std::vector<P> vals = disc.get_moment(mid[i]);
      tassert(ref.size() == vals.size());

      P err = 0;
      for (size_t j = 0; j < ref.size(); j++)
        err = std::max(err, std::abs(scale * ref[j] - vals[j]));

      // std::cout << "  err = " << err << "  degree = " << degree
      //           << "  level = " << level << "  mom = " << i << " :: "
      //           << moms[i] << '\n';
      tcheckless(i, err, p.tols[i]);
    }
  }
}

template<typename P>
void do_all_tests() {
  constexpr P tol = (is_double<P>) ? 1.E-14 : 1.E-5;

  test_case<P>("case 1", 1, {0, 1}, {moment(0), moment(1), moment(2)},
               {test_props{0, 0, {tol, 2.E-1, 2.E-1}},
                test_props{0, 7, {tol, 5.E-3, 5.E-3}},
                test_props{1, 0, {tol, tol, 5.E-3}},
                test_props{1, 1, {tol, tol, 5.E-3}},
                test_props{1, 2, {tol, tol, 1.E-4}},
                test_props{2, 0, {tol, tol, tol}},
                test_props{2, 1, {tol, tol, tol}},
                test_props{2, 5, {tol, tol, tol}},
                test_props{3, 0, {tol, tol, tol}},
                test_props{3, 1, {tol, tol, tol}},
                test_props{3, 3, {tol, tol, tol}},
                });

  test_case<P>("case 2", 1, {1, 2}, {moment(0), moment(1), moment(2)},
               {test_props{0, 0, {tol, 2.E-1, 2.E-1}},
                test_props{0, 7, {tol, 5.E-3, 5.E-3}},
                test_props{1, 0, {tol, tol, 5.E-3}},
                test_props{1, 1, {tol, tol, 5.E-3}},
                test_props{1, 2, {tol, tol, 1.E-4}},
                test_props{2, 0, {tol, tol, tol}},
                test_props{2, 1, {tol, tol, tol}},
                test_props{2, 5, {tol, tol, tol}},
                test_props{3, 0, {tol, tol, tol}},
                test_props{3, 1, {tol, tol, tol}},
                test_props{3, 3, {tol, tol, tol}},
                });
  test_case<P>("case 3", 1, {2, 1}, {moment(0), moment(2)},
               {test_props{0, 0, {tol, 2.E-1}},
                test_props{0, 7, {tol, 5.E-3}},
                test_props{1, 0, {tol, 5.E-3}},
                test_props{1, 1, {tol, 5.E-3}},
                test_props{1, 2, {tol, 1.E-4}},
                test_props{2, 0, {tol, tol}},
                test_props{2, 1, {tol, tol}},
                test_props{2, 5, {tol, tol}},
                test_props{3, 0, {tol, tol}},
                test_props{3, 1, {tol, tol}},
                test_props{3, 3, {tol, tol}},
                });

  test_case<P>("case 1", 1, {0, 1, 2}, {moment(0, 0), moment(0, 1), moment(2, 0), moment(2, 1)},
               {test_props{0, 0, {tol, 1.E-1, 1.E-2, 5.E-2}},
                test_props{0, 7, {tol, 5.E-4, 5.E-5, 1.E-4}},
                test_props{1, 0, {tol, tol, 5.E-3, 1.E-4}},
                test_props{1, 1, {tol, tol, 5.E-5, 1.E-5}},
                test_props{1, 4, {tol, tol, 5.E-7, 1.E-7}},
                test_props{2, 0, {tol, tol, tol, tol}},
                test_props{2, 1, {tol, tol, tol, tol}},
                test_props{2, 5, {tol, tol, tol, tol}},
                test_props{3, 0, {tol, tol, tol, tol}},
                test_props{3, 1, {tol, tol, tol, tol}},
                test_props{3, 3, {tol, tol, tol, tol}},
                });
  test_case<P>("case 2", 1, {2, 1, 0}, {moment(0, 0), moment(0, 1), moment(2, 0), moment(2, 1)},
               {test_props{0, 7, {tol, 5.E-4, 1.E-4, 5.E-3}},
                test_props{1, 0, {tol, tol, 5.E-3, 5.E-3}},
                test_props{1, 4, {tol, tol, 5.E-7, 1.E-7}},
                test_props{2, 0, {tol, tol, tol, tol}},
                test_props{3, 0, {tol, tol, tol, tol}},
                });
  test_case<P>("case 3", 1, {1, 2, 0}, {moment(0, 1), moment(3, 0), moment(3, 1)},
               {test_props{1, 4, {tol, 5.E-7, 5.E-7}},
                test_props{2, 0, {tol, 1.E-4, 1.E-4}},
                test_props{3, 0, {tol, tol, tol}},
                });

  test_case<P>("case 1", 1, {0, 1, 2, 4}, {moment(0, 0, 0), moment(0, 1, 1), moment(3, 0, 0)},
               {test_props{0, 7, {tol, 5.E-4, 1.E-4,}},
                test_props{1, 0, {tol, tol, 1.E-3}},
                test_props{1, 4, {tol, tol, 3.E-6}},
                test_props{2, 0, {tol, tol, 3.E-5}},
                test_props{3, 0, {tol, tol, tol}},
                });

  test_case<P>("case 2", 1, {4, 1, 2, 2}, {moment(0, 0, 0), moment(0, 0, 3)},
               {test_props{0, 7, {tol, 1.E-1}},
                test_props{1, 0, {tol, 1.E-1}},
                test_props{1, 6, {tol, 3.E-6}},
                test_props{2, 0, {tol, 3.E-5}},
                test_props{3, 0, {tol, tol}},
                });

  test_case<P>("all", 2, {0, 1, 2}, {moment(0), moment(1), moment(2)},
               {test_props{0, 7, {tol, 5.E-4, 1.E-3}},
                test_props{1, 0, {tol, 1.E-8, 5.E-4}},
                test_props{1, 6, {tol, 3.E-8, 5.E-8}},
                test_props{2, 0, {tol, tol, tol}},
                test_props{2, 4, {tol, tol, tol}},
                test_props{3, 0, {tol, tol, tol}},
                });

  test_case<P>("all", 2, {1, 2, 3, 4}, {moment(2, 0), moment(0, 0), moment(0, 1)},
               {test_props{1, 0, {tol, 1.E-8, 5.E-4}},
                test_props{1, 6, {tol, 1.E-8, 1.E-8}},
                test_props{2, 0, {tol, tol, tol}},
                test_props{2, 4, {tol, tol, tol}},
                test_props{3, 0, {tol, tol, tol}},
                });

  test_case<P>("all", 3, {1, 2, 0, 0, 2, 1}, {moment(0, 0, 0), moment(2, 0, 0)},
               {test_props{1, 0, {tol, 1.E-3}},
                test_props{1, 6, {tol, 5.E-8}},
                test_props{2, 0, {tol, tol}},
                test_props{2, 4, {tol, tol}},
                test_props{3, 0, {tol, tol}},
                });
}

int main(int argc, char **argv)
{
  libasgard_runtime running_(argc, argv);

  all_tests global_("computing moments", " field integrals in velocity domain");

  #ifdef ASGARD_ENABLE_DOUBLE
  do_all_tests<double>();
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  do_all_tests<float>();
  #endif

  return 0;
}
