#include "asgard_test_macros.hpp"

using namespace asgard;

template<typename TestType>
void test_bookkeeping() {
  {
    current_test<TestType> name_("pde_domain");
    tassert(pde_domain<TestType>(1).num_dims() == 1);
    tassert(pde_domain<TestType>(2).num_dims() == 2);
    tassert(pde_domain<TestType>(2).num_pos() == 0);
    tassert(pde_domain<TestType>(3).num_vel() == 0);
    tassert(pde_domain<TestType>(1).length(0) == TestType{1});
    tassert(pde_domain<TestType>(1).name(0) == std::string("x1"));
    tassert(pde_domain<TestType>(4).name(3) == std::string("x4"));

    terror_message(pde_domain<TestType>(-3),
                   "pde_domain created with zero or negative dimensions");
    terror_message(pde_domain<TestType>(max_num_dimensions + 1),
                   "pde_domain created with too many dimensions, max is 6D");

    tassert(pde_domain<TestType>({{0, 2}, {-2, 1}}).length(0) == TestType{2});
    tassert(pde_domain<TestType>({{0, 2}, {-2, 1}}).xleft(1) == TestType{-2});
    tassert(pde_domain<TestType>({{0, 2}, {-2, 1}}).length(1) == TestType{3});
    tassert(pde_domain<TestType>({{0, 2}, {-2, 1}}).xright(0) == TestType{2});

    terror_message(pde_domain<TestType>({{0, 1}, {6, -6}}),
                   "domain_range specified with negative length");

    pde_domain<TestType> dom(3);
    terror_message(dom.set({{0, 1}, {-6, 6}}),
                   "provided number of domain_range entries does not match the number of dimensions");
    dom.set({{0, 1}, {-6, 6}, {-4, 4}});
    tassert(dom.length(2) == TestType{8});

    terror_message(dom.set_names({"d1", "d2"}),
                   "provided number of names does not match the number of dimensions");
    dom.set_names({"d1", "d2", "d3"});
    tassert(dom.name(1) == std::string("d2"));
  }

  auto rhs = [](std::vector<TestType> const &, std::vector<TestType> &) -> void {};
  auto mhs = [](std::vector<TestType> const &x, std::vector<TestType> &fx)
    -> void {
      for (auto i : indexof(x))
        fx[i] = 2 * x[i];
    };

  {
    current_test<TestType> name_("term identity");
    term_1d<TestType> ptI1;
    tassert(ptI1.is_identity());
    term_1d<TestType> ptI2 = term_identity{};
    tassert(ptI2.is_identity());
  }
  {
    current_test<TestType> name_("term volume");
    term_1d<TestType> ptM = term_volume<TestType>{3.5};
    tassert(not ptM.is_identity());
    tassert(ptM.rhs_const() == 3.5);
    tassert(term_1d<TestType>(term_volume<TestType>{rhs}).rhs()); // loaded a function
  }
  {
    current_test<TestType> name_("term div");
    term_1d<TestType> ptD = term_div<TestType>{mhs, flux_type::upwind, boundary_type::bothsides};
    tassert(not ptD.is_identity());
    tassert(ptD.is_div());
    tassert(ptD.optype() == operation_type::div);
    tassert(ptD.flux() == flux_type::upwind);
    std::vector<TestType> x = {1, 2, 3}, fx(3);
    ptD.rhs(x, fx);
    tassert(fm::diff_inf(fx, std::vector<TestType>{2, 4, 6}) == 0);
  }
  {
    current_test<TestType> name_("term grad");
    term_1d<TestType> ptG = term_grad<TestType>{mhs, flux_type::downwind, boundary_type::none};
    tassert(not ptG.is_identity());
    tassert(ptG.is_grad());
    tassert(ptG.optype() == operation_type::grad);
    tassert(ptG.flux() == flux_type::upwind); // grad swaps the fluxes
    std::vector<TestType> x = {-1, 5, 2}, fx(3);
    ptG.rhs()(x, fx);
    tassert(fm::diff_inf(fx, std::vector<TestType>{-2, 10, 4}) == 0);
  }
  {
    current_test<TestType> name_("term chain 1d");
    term_1d<TestType> ptD = term_div<TestType>{1, flux_type::upwind, boundary_type::periodic};
    term_1d<TestType> chain({ptD, });
    tassert(not chain.is_identity());
    tassert(not chain.is_chain());
    tassert(chain.is_div());
    tassert(chain.num_chain() == 0);
  }
  {
    current_test<TestType> name_("term 2 instances");
    term_1d<TestType> ptI;
    tassert(term_1d<TestType>({ptI, ptI}).is_identity());
    tassert(not term_1d<TestType>({ptI, ptI}).is_chain());
    tassert(term_1d<TestType>({ptI, ptI}).num_chain() == 0);

    term_1d<TestType> ptD = term_div<TestType>{mhs, flux_type::upwind, boundary_type::bothsides};
    term_1d<TestType> ptG = term_div<TestType>{mhs, flux_type::downwind, boundary_type::bothsides};

    tassert(not term_1d<TestType>({ptI, ptD}).is_chain());
    tassert(term_1d<TestType>({ptI, ptD}).is_div());
    tassert(not term_1d<TestType>({ptD, ptI}).is_chain());
    tassert(term_1d<TestType>({ptD, ptI}).is_div());

    tassert(term_1d<TestType>({ptD, ptG}).is_chain());
    tassert(term_1d<TestType>({ptD, ptG}).num_chain() == 2);

    terror_message(term_1d<TestType>({ptD, ptD, ptD}),
                   "cannot chain more than two non-central fluxes");

    term_1d<TestType> ptC = term_div<TestType>{mhs, flux_type::central};
    terror_message(term_1d<TestType>({ptC, ptC}),
                   "cannot chain two central fluxes together");
    terror_message(term_1d<TestType>({ptC, ptD}),
                   "cannot chain a central flux with a side flux");
  }
  {
    current_test<TestType> name_("term 1d - extras");
    term_1d<TestType> ptI;
    term_1d<TestType> ptM = term_volume<TestType>(3);
    term_1d<TestType> ptD = term_div<TestType>{mhs, flux_type::upwind, boundary_type::bothsides};
    term_1d<TestType> ptG = term_grad<TestType>{mhs, flux_type::downwind, boundary_type::bothsides};
    term_1d<TestType> ptGc = term_grad<TestType>{3.5, flux_type::central, boundary_type::bothsides};

    tassert(term_1d<TestType>({ptI, ptD, ptM}).num_chain() == 2);
    tassert(term_1d<TestType>({ptG, ptI, ptM, ptD, ptM}).num_chain() == 4);
    tassert(term_1d<TestType>({ptGc, ptM}).num_chain() == 2);

    term_1d<TestType> chain({ptI, ptG, ptM, ptD, ptM});
    tassert(chain[0].optype() == operation_type::grad);
    tassert(chain.chain()[1].optype() == operation_type::volume);
    tassert(chain[2].optype() == operation_type::div);
    tassert(chain[3].optype() == operation_type::volume);
  }
  {
    current_test<TestType> name_("term_md");
    term_1d<TestType> ptI = term_identity{};
    term_1d<TestType> ptM = term_volume<TestType>{3.5};

    tassert(term_md<TestType>({ptM, ptI}).num_dims() == 2);
    tassert(term_md<TestType>({ptM, ptI}).term_mode() == term_md<TestType>::mode::separable);

    terror_message(term_md<TestType>({ptI, ptI}),
                   "cannot create term_md with all terms being identities");

    term_md<TestType> t1({ptM, ptI});
    tassert(term_md<TestType>({t1, t1}).term_mode() == term_md<TestType>::mode::chain);
    tassert(term_md<TestType>({t1, t1}).num_dims() == 2);
    tassert(term_md<TestType>({t1, t1, t1}).num_chain() == 3);

    term_md<TestType> t2({ptI, ptI, ptM});
    tassert(term_md<TestType>({t2, t2}).num_dims() == 3);
    terror_message(term_md<TestType>({t1, t2}),
                   "inconsistent dimension of terms in the chain");

    std::vector<term_1d<TestType>> ptc = {ptI, ptI, ptI};
    for (int i = 0; i < 3; i++)
    {
      ptc[i] = ptM;
      term_md<TestType> tm(ptc);
      tassert(tm.num_dims() == 3);
      tassert(tm.term_mode() == term_md<TestType>::mode::separable);
      ptc[i] = ptI;
    }
  }
}

template<typename TestType>
void test_pde_class() {
  {
    current_test<TestType> name_("pde empty");
    pde_scheme<TestType> empty_pde;
    tassert(not empty_pde);
    prog_opts opts;
    opts.degree = 4;
    opts.start_levels = {3,};
    pde_domain<TestType> domain({{1, 3}, {-1, 6}});
    pde_scheme<TestType> pde(opts, std::move(domain));
    tassert(!!pde);
    tassert(pde.domain().length(1) == TestType{7});
    tassert(!!pde.options().degree);
    tassert(pde.options().degree.value() == 4);
  }
  {
    current_test<TestType> name_("pde constructors");
    prog_opts opts = make_opts("-l 3 -d 1");
    pde_domain<TestType> domain({{1, 3}, {-1, 6}});
    pde_scheme<TestType> pde(opts, std::move(domain));
    tassert(pde.mass().dim(0).is_identity());
    tassert(pde.mass().dim(1).is_identity());
    tassert(pde.mass().is_identity());
    terror_message(pde.set_mass(mass_md<TestType>{2}),
                   "the mass term must be separable");
    pde.set_mass({term_volume{2}, term_volume{3}});
    tassert(not pde.mass().dim(0).is_identity());
    tassert(pde.mass().dim(0).rhs_const() == 2);
    tassert(not pde.mass().dim(1).is_identity());
    tassert(pde.mass().dim(1).rhs_const() == 3);
  }
  {
    current_test<TestType> name_("pde imex methods");
    prog_opts opts = make_opts("-l 2 -d 1 -s imex2");
    tassert(opts.step_method);
    tassert(opts.step_method.value() == time_method::imex2);
    pde_scheme<TestType> pde(opts, pde_domain<TestType>(2));
    pde.set(imex_implicit_group{2}, imex_explicit_group{5});
    tassert(pde.imex_im().gid == 2);
    tassert(pde.imex_ex().gid == 5);
  }
}

template<typename P>
void test_discretization_manager() {
  {
    current_test<P> name_("discretization test");
    static_assert(std::is_copy_constructible_v<pde_scheme<P>>);
    static_assert(std::is_move_constructible_v<pde_scheme<P>>);
    static_assert(std::is_copy_assignable_v<pde_scheme<P>>);
    static_assert(std::is_move_assignable_v<pde_scheme<P>>);
    static_assert(std::is_copy_constructible_v<term_1d<P>>);
    static_assert(std::is_move_constructible_v<term_1d<P>>);
    static_assert(std::is_copy_assignable_v<term_1d<P>>);
    static_assert(std::is_move_assignable_v<term_1d<P>>);
    static_assert(std::is_copy_constructible_v<term_md<P>>);
    static_assert(std::is_move_constructible_v<term_md<P>>);
    static_assert(std::is_copy_assignable_v<term_md<P>>);
    static_assert(std::is_move_assignable_v<term_md<P>>);
  }
}

template<typename P>
void pde_tests() {
  test_bookkeeping<P>();
  test_pde_class<P>();
  test_discretization_manager<P>();
}

void pde_functions() {
  sfixed_func1d<double> dfx = vectorize<double>([](double x)->double { return std::sin(x); });
  sfixed_func1d<float> ffx = vectorize<float>([](float x)->float { return std::sin(x); });

  svector_func1d<double> dfxt = vectorize_t<double>([](double x)->double { return std::sin(x); });
  svector_func1d<float> ffxt = vectorize_t<float>([](float x)->float { return std::sin(x); });

  svector_func1d<double> dfxtt = vectorize_t<double>([](double x, double t)->double { return t * std::sin(x); });
  svector_func1d<float> ffxtt = vectorize_t<float>([](float x, double t)->float { return t * std::sin(x); });
}

int main(int argc, char **argv) {

  libasgard_runtime running_(argc, argv);

  all_tests global_("asgard-pde-tests", " setting up the pde");

  #ifdef ASGARD_ENABLE_DOUBLE
  pde_tests<double>();
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  pde_tests<float>();
  #endif

  pde_functions();

  return 0;
}
