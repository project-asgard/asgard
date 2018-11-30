#pragma once

#include <map>
#include <string>

// vlasov4 = 0 (Non-linear vlasov-poisson : bump on tail instability, Ex4.1 in
// PDF)
//
// vlasov7 = 1  (Linear vlasov : time dependent E)
//
// vlasov8 = 2  (Linear vlasov : time & space dependent E - see mathematica
// file)
//
// vlasov5 = 3 // (Non-linear vlasov-poisson : two-stream instability 1, Ex4.2
// in PDF)
//
// vlasov43 = 4 (Non-linear vlasov-poisson : two-stream instability 2, Ex4.3 in
// PDF)

enum class PDE_opts { pde_user, vlasov4, vlasov7, vlasov8, vlasov5, vlasov43 };
typedef std::map<std::string, PDE_opts> pde_map_t;

class Options {

private:
  int level = 2;                      // resolution
  int degree = 3;                     // degree of legendre basis polynomials
  int num_time_steps = 10;            // number of time loop iterations
  int write_frequency = 0;            // write output every this many iterations
  int visualization_frequency = 0;    // visualize output every this many iters
  bool use_implicit_stepping = false; // enable implicit(/explicit) stepping
  bool use_full_grid = false;         // enable full(/sparse) grid
  bool do_poisson = false;            // do poisson solve for electric field
  double cfl = 0.1; // the Courant-Friedrichs-Lewy (CFL) condition

  pde_map_t pde_mapping = {
      {"pde_user", PDE_opts::pde_user}, {"vlasov4", PDE_opts::vlasov4},
      {"vlasov7", PDE_opts::vlasov7},   {"vlasov8", PDE_opts::vlasov8},
      {"vlasov5", PDE_opts::vlasov5},   {"vlasov43", PDE_opts::vlasov43}};
  std::string selected_pde = "vlasov4";

  // pde to construct/evaluate
  PDE_opts pde_choice;

  // is there a better (testable) way to handle invalid command-line input?
  bool valid = true;

public:
  Options(int argc, char **argv);
  int get_level() const;
  int get_degree() const;
  int get_time_steps() const;
  int get_write_frequency() const;
  int get_visualization_frequency() const;
  bool using_implicit() const;
  bool using_full_grid() const;
  double get_cfl() const;
  PDE_opts get_selected_pde() const;
  bool do_poisson_solve() const;
  bool is_valid() const;
};
