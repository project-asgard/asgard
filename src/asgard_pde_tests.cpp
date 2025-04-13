#include "tests_general.hpp"

#include "asgard_test_macros.hpp"

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

    terror_message(pde_domain<TestType>(-3),
                   "pde_domain created with zero or negative dimensions");
    terror_message(pde_domain<TestType>(max_num_dimensions + 1),
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
    term_1d<TestType> ptM = term_volume<TestType>{3.5};
    REQUIRE_FALSE(ptM.is_identity());
    REQUIRE(ptM.rhs_const() == 3.5);
    REQUIRE(term_1d<TestType>(term_volume<TestType>{rhs}).rhs()); // loaded a function
  }

  SECTION("term_1d - div") {
    term_1d<TestType> ptD = term_div<TestType>{mhs, flux_type::upwind, boundary_type::bothsides};
    REQUIRE_FALSE(ptD.is_identity());
    REQUIRE(ptD.is_div());
    REQUIRE(ptD.optype() == operation_type::div);
    REQUIRE(ptD.flux() == flux_type::upwind);
    std::vector<TestType> x = {1, 2, 3}, fx(3);
    ptD.rhs(x, fx);
    REQUIRE(fm::diff_inf(fx, std::vector<TestType>{2, 4, 6}) == 0);
  }

  SECTION("term_1d - grad") {
    term_1d<TestType> ptG = term_grad<TestType>{mhs, flux_type::downwind, boundary_type::none};
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

    term_1d<TestType> ptD = term_div<TestType>{mhs, flux_type::upwind, boundary_type::bothsides};
    term_1d<TestType> ptG = term_div<TestType>{mhs, flux_type::downwind, boundary_type::bothsides};

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
    term_1d<TestType> ptM = term_volume<TestType>(3);
    term_1d<TestType> ptD = term_div<TestType>{mhs, flux_type::upwind, boundary_type::bothsides};
    term_1d<TestType> ptG = term_grad<TestType>{mhs, flux_type::downwind, boundary_type::bothsides};
    term_1d<TestType> ptGc = term_grad<TestType>{3.5, flux_type::central, boundary_type::bothsides};

    REQUIRE(term_1d<TestType>({ptI, ptD, ptM}).num_chain() == 2);
    REQUIRE(term_1d<TestType>({ptG, ptI, ptM, ptD, ptM}).num_chain() == 4);
    REQUIRE(term_1d<TestType>({ptGc, ptM}).num_chain() == 2);

    // REQUIRE_THROWS_WITH(term_1d<TestType>({ptGc, ptD}),
    //                     "incompatible flux combination used in a term_1d chain, must split into a term_md chain");

    term_1d<TestType> chain({ptI, ptG, ptM, ptD, ptM});
    REQUIRE(chain[0].optype() == operation_type::grad);
    REQUIRE(chain.chain()[1].optype() == operation_type::volume);
    REQUIRE(chain[2].optype() == operation_type::div);
    REQUIRE(chain[3].optype() == operation_type::volume);
  }

  SECTION("term_md") {
    term_1d<TestType> ptI = term_identity{};
    term_1d<TestType> ptM = term_volume<TestType>{3.5};

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
    terror_message(pde.set_mass(mass_md<TestType>{2}),
                   "the mass term must be separable");
    pde.set_mass({term_volume{2}, term_volume{3}});
    REQUIRE_FALSE(pde.mass().dim(0).is_identity());
    REQUIRE(pde.mass().dim(0).rhs_const() == 2);
    REQUIRE_FALSE(pde.mass().dim(1).is_identity());
    REQUIRE(pde.mass().dim(1).rhs_const() == 3);
  }
  SECTION("imex")
  {
    prog_opts opts = make_opts("-l 2 -d 1 -s imex2");
    REQUIRE(opts.step_method);
    REQUIRE(opts.step_method.value() == time_method::imex2);
    PDEv2<TestType> pde(opts, pde_domain<TestType>(2));
    pde.set(imex_implicit_group{2}, imex_explicit_group{5});
    REQUIRE(pde.imex_im().gid == 2);
    REQUIRE(pde.imex_ex().gid == 5);
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
