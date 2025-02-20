#include "tests_general.hpp"

static auto const pde_eps_multiplier = 1e2;

static auto const pde_base_dir = gold_base_dir / "pde";

using namespace asgard;

TEMPLATE_TEST_CASE("pde book-keeping", "[pde]", test_precs)
{
  SECTION("pde_domain")
  {
    REQUIRE(pde_domain<TestType>(1).num_dims() == 1);
    REQUIRE(pde_domain<TestType>(2).num_dims() == 2);
    REQUIRE(pde_domain<TestType>(2).num_pos() == 0);
    REQUIRE(pde_domain<TestType>(3).num_vel() == 0);
    REQUIRE(pde_domain<TestType>(1).length(0) == TestType{1});
    REQUIRE(pde_domain<TestType>(1).name(0) == std::string("x1"));
    REQUIRE(pde_domain<TestType>(4).name(3) == std::string("x4"));

    REQUIRE_THROWS_WITH(pde_domain<TestType>(-3),
                        "pde_domain created with zero or negative dimensions");
    REQUIRE_THROWS_WITH(pde_domain<TestType>(max_num_dimensions + 1),
                        "pde_domain created with too many dimensions, max is 6D");

    REQUIRE(pde_domain<TestType>({{0, 2}, {-2, 1}}).length(0) == TestType{2});
    REQUIRE(pde_domain<TestType>({{0, 2}, {-2, 1}}).xleft(1) == TestType{-2});
    REQUIRE(pde_domain<TestType>({{0, 2}, {-2, 1}}).length(1) == TestType{3});
    REQUIRE(pde_domain<TestType>({{0, 2}, {-2, 1}}).xright(0) == TestType{2});

    REQUIRE_THROWS_WITH(pde_domain<TestType>({{0, 1}, {6, -6}}),
                        "domain_range specified with negative length");

    pde_domain<TestType> dom(3);
    REQUIRE_THROWS_WITH(dom.set({{0, 1}, {-6, 6}}),
                        "provided number of domain_range entries does not match the number of dimensions");
    dom.set({{0, 1}, {-6, 6}, {-4, 4}});
    REQUIRE(dom.length(2) == TestType{8});

    REQUIRE_THROWS_WITH(dom.set_names({"d1", "d2"}),
                        "provided number of names does not match the number of dimensions");
    dom.set_names({"d1", "d2", "d3"});
    REQUIRE(dom.name(1) == std::string("d2"));
  }

  auto rhs = [](std::vector<TestType> const &, std::vector<TestType> &) -> void {};
  auto mhs = [](std::vector<TestType> const &x, std::vector<TestType> &fx)
    -> void {
      for (auto i : indexof(x))
        fx[i] = 2 * x[i];
    };

  SECTION("term_1d - identity") {
    term_1d<TestType> ptI1;
    REQUIRE(ptI1.is_identity());
    term_1d<TestType> ptI2 = term_identity{};
    REQUIRE(ptI2.is_identity());
  }

  SECTION("term_1d - mass") {
    term_1d<TestType> ptM = term_mass<TestType>{3.5};
    REQUIRE_FALSE(ptM.is_identity());
    REQUIRE(ptM.rhs_const() == 3.5);
    REQUIRE(term_1d<TestType>(term_mass<TestType>{rhs}).rhs()); // loaded a function
  }

  SECTION("term_1d - div") {
    term_1d<TestType> ptD = term_div<TestType>{mhs, flux_type::upwind, boundary_type::dirichlet};
    REQUIRE_FALSE(ptD.is_identity());
    REQUIRE(ptD.is_div());
    REQUIRE(ptD.optype() == operation_type::div);
    REQUIRE(ptD.flux() == flux_type::upwind);
    std::vector<TestType> x = {1, 2, 3}, fx(3);
    ptD.rhs(x, fx);
    REQUIRE(fm::diff_inf(fx, std::vector<TestType>{2, 4, 6}) == 0);
  }

  SECTION("term_1d - grad") {
    term_1d<TestType> ptG = term_grad<TestType>{mhs, flux_type::downwind, boundary_type::free};
    REQUIRE_FALSE(ptG.is_identity());
    REQUIRE(ptG.is_grad());
    REQUIRE(ptG.optype() == operation_type::grad);
    REQUIRE(ptG.flux() == flux_type::upwind); // grad swaps the fluxes
    std::vector<TestType> x = {-1, 5, 2}, fx(3);
    ptG.rhs()(x, fx);
    REQUIRE(fm::diff_inf(fx, std::vector<TestType>{-2, 10, 4}) == 0);
  }

  SECTION("term_1d - chain 1 term") {
    term_1d<TestType> ptD = term_div<TestType>{1, flux_type::upwind, boundary_type::periodic};
    term_1d<TestType> chain({ptD, });
    REQUIRE_FALSE(chain.is_identity());
    REQUIRE_FALSE(chain.is_chain());
    REQUIRE(chain.is_div());
    REQUIRE(chain.num_chain() == 0);
  }

  SECTION("term_1d - 2 terms") {
    term_1d<TestType> ptI;
    REQUIRE(term_1d<TestType>({ptI, ptI}).is_identity());
    REQUIRE_FALSE(term_1d<TestType>({ptI, ptI}).is_chain());
    REQUIRE(term_1d<TestType>({ptI, ptI}).num_chain() == 0);

    term_1d<TestType> ptD = term_div<TestType>{mhs, flux_type::upwind, boundary_type::dirichlet};
    term_1d<TestType> ptG = term_div<TestType>{mhs, flux_type::downwind, boundary_type::dirichlet};

    REQUIRE_FALSE(term_1d<TestType>({ptI, ptD}).is_chain());
    REQUIRE(term_1d<TestType>({ptI, ptD}).is_div());
    REQUIRE_FALSE(term_1d<TestType>({ptD, ptI}).is_chain());
    REQUIRE(term_1d<TestType>({ptD, ptI}).is_div());

    REQUIRE(term_1d<TestType>({ptD, ptG}).is_chain());
    REQUIRE(term_1d<TestType>({ptD, ptG}).num_chain() == 2);

    // REQUIRE_THROWS_WITH(term_1d<TestType>({ptD, ptD}),
    //                     "incompatible flux combination used in a term_1d chain, must split into a term_md chain");
  }

  SECTION("term_1d - extra") {
    term_1d<TestType> ptI;
    term_1d<TestType> ptM = term_mass<TestType>(3);
    term_1d<TestType> ptD = term_div<TestType>{mhs, flux_type::upwind, boundary_type::dirichlet};
    term_1d<TestType> ptG = term_grad<TestType>{mhs, flux_type::downwind, boundary_type::dirichlet};
    term_1d<TestType> ptGc = term_grad<TestType>{3.5, flux_type::central, boundary_type::dirichlet};

    REQUIRE(term_1d<TestType>({ptI, ptD, ptM}).num_chain() == 2);
    REQUIRE(term_1d<TestType>({ptG, ptI, ptM, ptD, ptM}).num_chain() == 4);
    REQUIRE(term_1d<TestType>({ptGc, ptM}).num_chain() == 2);

    // REQUIRE_THROWS_WITH(term_1d<TestType>({ptGc, ptD}),
    //                     "incompatible flux combination used in a term_1d chain, must split into a term_md chain");

    term_1d<TestType> chain({ptI, ptG, ptM, ptD, ptM});
    REQUIRE(chain[0].optype() == operation_type::grad);
    REQUIRE(chain.chain()[1].optype() == operation_type::mass);
    REQUIRE(chain[2].optype() == operation_type::div);
    REQUIRE(chain[3].optype() == operation_type::mass);
  }

  SECTION("term_md") {
    term_1d<TestType> ptI = term_identity{};
    term_1d<TestType> ptM = term_mass<TestType>{3.5};

    REQUIRE(term_md<TestType>({ptM, ptI}).num_dims() == 2);
    REQUIRE(term_md<TestType>({ptM, ptI}).term_mode() == term_md<TestType>::mode::separable);

    REQUIRE_THROWS_WITH(term_md<TestType>({ptI, ptI}),
                        "cannot create term_md with all terms being identities");

    term_md<TestType> t1({ptM, ptI});
    REQUIRE(term_md<TestType>({t1, t1}).term_mode() == term_md<TestType>::mode::chain);
    REQUIRE(term_md<TestType>({t1, t1}).num_dims() == 2);
    REQUIRE(term_md<TestType>({t1, t1, t1}).num_chain() == 3);

    term_md<TestType> t2({ptI, ptI, ptM});
    REQUIRE(term_md<TestType>({t2, t2}).num_dims() == 3);
    REQUIRE_THROWS_WITH(term_md<TestType>({t1, t2}),
                        "inconsistent dimension of terms in the chain");

    std::vector<term_1d<TestType>> ptc = {ptI, ptI, ptI};
    for (int i = 0; i < 3; i++)
    {
      ptc[i] = ptM;
      term_md<TestType> tm(ptc);
      REQUIRE(tm.num_dims() == 3);
      REQUIRE(tm.term_mode() == term_md<TestType>::mode::separable);
      ptc[i] = ptI;
    }
  }
}

TEMPLATE_TEST_CASE("pde v2", "[pde]", test_precs)
{
  SECTION("constructors")
  {
    PDEv2<TestType> empty_pde;
    REQUIRE_FALSE(empty_pde);
    prog_opts opts;
    opts.degree = 4;
    opts.start_levels = {3,};
    pde_domain<TestType> domain({{1, 3}, {-1, 6}});
    PDEv2<TestType> pde(opts, std::move(domain));
    REQUIRE(!!pde);
    REQUIRE(pde.domain().length(1) == TestType{7});
    REQUIRE(!!pde.options().degree);
    REQUIRE(pde.options().degree.value() == 4);
  }
  SECTION("constructors")
  {
    prog_opts opts = make_opts("-l 3 -d 1");
    pde_domain<TestType> domain({{1, 3}, {-1, 6}});
    PDEv2<TestType> pde(opts, std::move(domain));
    REQUIRE(pde.mass().dim(0).is_identity());
    REQUIRE(pde.mass().dim(1).is_identity());
    REQUIRE(pde.mass().is_identity());
    // REQUIRE_THROWS_WITH(pde.set_mass(term_md<TestType>{}),
    //                     "the mass term must be separable");
    pde.set_mass({term_mass{2}, term_mass{3}});
    REQUIRE_FALSE(pde.mass().dim(0).is_identity());
    REQUIRE(pde.mass().dim(0).rhs_const() == 2);
    REQUIRE_FALSE(pde.mass().dim(1).is_identity());
    REQUIRE(pde.mass().dim(1).rhs_const() == 3);
  }
}

TEST_CASE("helper wrappers", "[pde]")
{
  SECTION("compile wrappers")
  {
    sfixed_func1d<double> dfx = vectorize<double>([](double x)->double { return std::sin(x); });
    sfixed_func1d<float> ffx = vectorize<float>([](float x)->float { return std::sin(x); });

    svector_func1d<double> dfxt = vectorize_t<double>([](double x)->double { return std::sin(x); });
    svector_func1d<float> ffxt = vectorize_t<float>([](float x)->float { return std::sin(x); });

    svector_func1d<double> dfxtt = vectorize_t<double>([](double x, double t)->double { return t * std::sin(x); });
    svector_func1d<float> ffxtt = vectorize_t<float>([](float x, double t)->float { return t * std::sin(x); });
  }
}

template<typename P>
void test_initial_condition(PDE<P> const &pde, std::filesystem::path base_dir,
                            fk::vector<P> const &x)
{
  auto const filename = base_dir.filename().string();
  for (auto i = 0; i < pde.num_dims(); ++i)
  {
    auto const gold = read_vector_from_txt_file<P>(base_dir.replace_filename(
        filename + "initial_dim" + std::to_string(i) + ".dat"));
    auto const fx   = pde.get_dimensions()[i].initial_condition[0](x, 0);

    auto constexpr tol_factor = get_tolerance<P>(10);

    rmse_comparison(fx, gold, tol_factor);
  }
}

template<typename P>
void test_exact_solution(PDE<P> const &pde, std::filesystem::path base_dir,
                         fk::vector<P> const &x, P const time)
{
  if (not pde.has_analytic_soln())
  {
    return;
  }

  auto constexpr tol_factor = get_tolerance<P>(10);
  auto const filename       = base_dir.filename().string();
  for (auto i = 0; i < pde.num_dims(); ++i)
  {
    auto const gold = read_vector_from_txt_file<P>(base_dir.replace_filename(
        filename + "exact_dim" + std::to_string(i) + ".dat"));
    auto const fx   = pde.exact_vector_funcs()[0][i](x, time);
    rmse_comparison(fx, gold, tol_factor);
  }

  P const gold = read_scalar_from_txt_file(
      base_dir.replace_filename(filename + "exact_time.dat"));
  P const fx = pde.exact_time(time);
  relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
}

template<typename P>
void test_source_vectors(PDE<P> const &pde, std::filesystem::path base_dir,
                         fk::vector<P> const &x, P const time)
{
  auto constexpr tol_factor = get_tolerance<P>(10);
  auto const filename       = base_dir.filename().string();

  for (auto i = 0; i < pde.num_sources(); ++i)
  {
    auto const source_string = filename + "source" + std::to_string(i) + "_";
    auto const &source_funcs = pde.sources()[i].source_funcs();
    for (auto j = 0; j < pde.num_dims(); ++j)
    {
      auto const full_path = base_dir.replace_filename(
          source_string + "dim" + std::to_string(j) + ".dat");
      auto const gold = read_vector_from_txt_file<P>(full_path);
      auto const fx   = source_funcs[j](x, time);
      rmse_comparison(fx, gold, tol_factor);
    }
    P const gold = read_scalar_from_txt_file(
        base_dir.replace_filename(source_string + "time.dat"));
    auto const fx = pde.sources()[i].time_func()(time);
    relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
  }
}

TEMPLATE_TEST_CASE("testing fokkerplanck2_complete_case4 implementations",
                   "[pde]", test_precs)
{
  auto const pde = make_PDE<TestType>("-p fokkerplanck_2d_complete_case4 -l 5 -d 4");
  auto const base_dir          = pde_base_dir / "fokkerplanck2_complete_";
  fk::vector<TestType> const x = {0.1, 0.2, 0.3, 0.4, 0.5};
  TestType const time          = 5;

  SECTION("fp2 complete initial condition functions")
  {
    test_initial_condition<TestType>(*pde, base_dir, x);
  }

  SECTION("fp2 complete exact solution functions")
  {
    test_exact_solution<TestType>(*pde, base_dir, x, time);
  }

  SECTION("fp2 complete source functions")
  {
    test_source_vectors(*pde, base_dir, x, time);
  }

  SECTION("fp2 complete dt")
  {
    // TestType const gold = read_scalar_from_txt_file(base_dir + "dt.dat");
    // TestType const dt = pde->get_dt() / parser::DEFAULT_CFL;
    // REQUIRE(dt == gold); // not testing this for now
    // different domain mins between matlab/C++ will produce different dts
  }

  SECTION("fp2 complete pterm funcs")
  {
    auto filename   = base_dir.filename().string();
    auto const gold = read_matrix_from_txt_file<TestType>(
        pde_base_dir / (filename + "gfuncs.dat"));
    auto const gold_dvs = read_matrix_from_txt_file<TestType>(
        pde_base_dir / (filename + "dvfuncs.dat"));

    int row = 0;
    for (auto i = 0; i < pde->num_dims(); ++i)
    {
      for (auto j = 0; j < pde->num_terms(); ++j)
      {
        auto const &term_1D       = pde->get_terms()[j][i];
        auto const &partial_terms = term_1D.get_partial_terms();
        for (auto k = 0; k < static_cast<int>(partial_terms.size()); ++k)
        {
          fk::vector<TestType> transformed(x);
          auto const &g_func = partial_terms[k].g_func();
          if (g_func)
          {
            std::transform(x.begin(), x.end(), transformed.begin(),
                           [g_func, time](TestType const x_elem) -> TestType {
                             return g_func(x_elem, time);
                           });
          }
          else
          {
            std::fill(transformed.begin(), transformed.end(), TestType{1.0});
          }
          fk::vector<TestType> gold_pterm(
              gold.extract_submatrix(row, 0, 1, x.size()));
          auto constexpr tol_factor = get_tolerance<TestType>(100);
          rmse_comparison(transformed, gold_pterm, tol_factor);

          fk::vector<TestType> dv(x);
          auto const &dv_func = partial_terms[k].dv_func();
          if (dv_func)
          {
            std::transform(x.begin(), x.end(), dv.begin(),
                           [dv_func, time](TestType const x_elem) -> TestType {
                             return dv_func(x_elem, time);
                           });
          }
          else
          {
            std::fill(dv.begin(), dv.end(), TestType{1.0});
          }

          fk::vector<TestType> gold_dvfunc(
              gold_dvs.extract_submatrix(row, 0, 1, x.size()));
          rmse_comparison(dv, gold_dvfunc, tol_factor);

          row++;
        }
      }
    }
  }
}

TEMPLATE_TEST_CASE("testing vlasov full f implementations", "[pde]", test_precs)
{
  prog_opts opts;
  opts.pde_choice     = PDE_opts::vlasov_lb_full_f;
  opts.degree         = 2;
  opts.start_levels   = {4, 3};
  opts.grid           = grid_type::dense;
  opts.num_time_steps = 1;

  auto const pde = make_PDE<TestType>(opts);
  //auto const pde               = make_PDE<TestType>(parse);
  auto const base_dir          = pde_base_dir / "vlasov_lb_full_f_";
  fk::vector<TestType> const x = {0.1, 0.2, 0.3, 0.4, 0.5};

  SECTION("vlasov full f initial condition functions")
  {
    test_initial_condition<TestType>(*pde, base_dir, x);
  }
}
