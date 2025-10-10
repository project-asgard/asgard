#include "asgard.hpp"

#include "asgard_test_macros.hpp"

using namespace asgard;

using prec = asgard::default_precision;

template<typename P>
void interp_identity_v2(P tol, int degree, int max_level)
{
    // max_level += 10;
    // max_level += 1;
    // max_level = 3;

  pde_domain<P> domain(2); // work in 2d
  separable_func<P> ic;
  ic.set(0, vectorize_t<P>([](P x)->P { return std::sin(x); }));
  ic.set(1, vectorize_t<P>([](P x)->P { return std::exp(x); }));

  std::map<int, std::string> mode = {{0, "constant"}, {1, "linear"},
                                     {2, "quadratic"}, {3, "cubic"}};

  current_test<P> name_("interp l = " + std::to_string(max_level) + ", " + mode[degree]);

  connection_patterns conn(max_level);
  hierarchy_manipulator<P> hier(degree, domain);

  quadmd_manager<P> quad(domain, hier, conn);

  // interpolation_manager<P> interp(domain, conn, degree);

  // prog_opts options = make_opts("-n 0 -grid dense");
  prog_opts options = make_opts("-n 0");
  options.degree = degree;
  options.start_levels = {max_level, };
  pde_scheme<P> pde(options, domain);
  pde.add_initial(ic);

  discretization_manager<P> disc(pde, verbosity_level::quiet);

  // check the loaded nodes
  sparse_grid const &grid = disc.get_grid();

  vector2d<P> const &nodes = quad.nodes(grid);
  tassert(nodes.stride() == 2);

  std::vector<P> vals(nodes.num_strips());
  for (int64_t i = 0; i < nodes.num_strips(); i++)
    vals[i] = ic.eval(nodes[i], 0);

  std::vector<P> wav(disc.current_state().size());
  std::vector<P> t1(wav.size());
  quad.nodal2wav(grid, disc.get_conn(), P{1}, vals.data(), P{0}, wav.data(),
                 disc.get_terms().kwork, t1);

  // std::vector<P> old(wav.size());
  // interp.nodal2hier(grid, disc.get_conn(), vals.data(), disc.get_terms().kwork);
  // interp.hier2wav(grid, conn, P{1}, vals.data(), P{0}, old.data(), disc.get_terms().kwork);

  // std::cout << std::scientific;
  // std::cout.precision(18);
  // for (size_t i = 0; i < wav.size(); i++)
  //   std::cout << wav[i] << "    " << disc.current_state()[i] << "\n";
  //   //std::cout << old[i] << "    " << wav[i] << "    " << disc.current_state()[i] << "\n";
  // std::cout << "   diff = " << fm::diff_inf(wav, disc.current_state()) << "\n";

  // std::cout.precision(8);
  // std::cout << "   ----wav2nodal_------------ \n";
  // quad.wav2nodal_.to_full(conn).print();
  // std::cout << "   ------wav2nodal1d--------- \n";
  // interp.wav2nodal1d().to_full(conn).print();
  //
  // std::cout << "   ---nodal2hier_----------- \n";
  // quad.nodal2hier_.to_full(conn).print();
  // std::cout << "   ----nodal2hier1d--------- \n";
  // interp.nodal2hier1d().to_full(conn).print();
  //
  // std::cout << "   ----hier2wav_------------- \n";
  // quad.hier2wav_.to_full(conn).print();
  // std::cout << "   ----hier2wav1d------------ \n";
  // interp.hier2wav1d().to_full(conn).print();


  std::cout << " degree = " << degree << " level = " << max_level
            << "  err = " << fm::diff_inf(wav, disc.current_state()) << "\n";
  tcheckless(degree, fm::diff_inf(wav, disc.current_state()), tol);
}

int main(int argc, char **argv)
{
  ignore(argc);
  ignore(argv);
  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior

  interp_identity_v2<prec>(1.E-3, 1, 1);


  return 0;
}
