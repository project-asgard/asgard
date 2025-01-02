#include "asgard_term_manager.hpp"

#include "asgard_coefficients_mats.hpp" // also brings in small-mats module

namespace asgard
{

template<typename P>
void term_entry<P>::set_perms()
{
  expect(not tmd.is_chain());
  if (tmd.is_interpolatory()) // no permutation to set
    return;

  int const num_dims = tmd.num_dims();
  std::vector<int> active_dirs;
  active_dirs.reserve(num_dims);
  int flux_dir = -1;
  for (int d : iindexof(num_dims))
  {
    auto const &t1d = tmd.dim(d);
    if (not t1d.is_identity()) {
      active_dirs.push_back(d);
      if (t1d.has_flux()) {
        flux_dir = d;
        if (active_dirs.size() > 1)
          std::swap(active_dirs.front(), active_dirs.back());
      }
    }
  }

  perm = kronmult::permutes(active_dirs, flux_dir);
}

template<typename P>
term_manager<P>::term_manager(PDEv2<P> &pde)
  : num_dims(pde.num_dims()), max_level(pde.max_level()), legendre(pde.degree())
{
  if (num_dims == 0)
    return;

  std::vector<term_md<P>> &pde_terms = pde.terms_;
  int num_terms = [&]() -> int {
    // get the effective number of terms, chained or not
    int n = 0;
    for (auto const &t : pde_terms)
      n += (t.is_chain()) ? t.num_chain() : 1;
    return n;
  }();

  terms.resize(num_terms);

  auto ir = terms.begin();
  for (int i : iindexof(pde_terms.size()))
  {
    if (pde_terms[i].is_chain()) {
      int const num_chain = pde_terms[i].num_chain();
      auto ic = ir + num_chain - 1;
      for (int j : iindexof(pde_terms[i].num_chain())) {
        ic->tmd = std::move(pde_terms[i].chain_[num_chain - j - 1]);
        ic->set_perms();
        --ic;
      }
      ir->num_chain = num_chain;
      ir += num_chain;
    } else {
      ir->tmd = std::move(pde_terms[i]);
      ir->set_perms();
      ir += 1;
    }
  }

  for (int d : iindexof(num_dims)) {
    xleft[d]  = pde.domain().xleft(d);
    xright[d] = pde.domain().xright(d);
  }
}

template<typename P>
void term_manager<P>::rebuld_term(
    int const tid, sparse_grid const &grid, connection_patterns const &conn,
    hierarchy_manipulator<P> const &hier)
{
  expect(legendre.pdof == hier.degree() + 1);
  expect(not terms[tid].tmd.is_chain());
  expect(not terms[tid].tmd.is_interpolatory());

  for (int d : iindexof(num_dims)) {
    auto const &t1d = terms[tid].tmd.dim(d);
    // do not build identity 1d terms
    if (t1d.is_identity())
      continue;

    int level = grid.current_level(d); // required level

    // terms that don't change should be build only once
    if (t1d.change() == changes_with::none) {
      if (terms[tid].coeffs[d].empty())
        level = max_level; // build up to the max
      else
        continue; // already build, we can skip
    }
    // terms that change only on level should be build only on level change
    if (t1d.change() == changes_with::level and terms[tid].level[d] == level)
      continue;

    static block_tri_matrix<P> raw_tri;
    static block_diag_matrix<P> raw_diag;

    bool is_diag = t1d.is_mass();
    if (t1d.is_chain()) {
      rebuld_chain(d, t1d, level, is_diag, raw_diag, raw_tri);
    } else {
      build_raw_mat(d, t1d, level, raw_diag, raw_tri);
    }
    if (is_diag)
      terms[tid].coeffs[d] = hier.diag2hierarchical(raw_diag, level, conn);
    else
      terms[tid].coeffs[d] = hier.tri2hierarchical(raw_tri, level, conn);
  } // move to next dimension d
}

template<typename P>
void term_manager<P>::build_raw_mat(
    int d, term_1d<P> const &t1d, int level, block_diag_matrix<P> &raw_diag,
    block_tri_matrix<P> &raw_tri)
{
  expect(not t1d.is_chain());
  static block_diag_matrix<P> raw_mass;
  switch (t1d.optype())
  {
    case operation_type::mass:
      if (t1d.rhs()) {
        gen_diag_cmat<P, operation_type::mass, rhs_type::is_func>
          (legendre, xleft[d], xright[d], level, t1d.rhs(), 0, raw_diag);
      } else {
        gen_diag_cmat<P, operation_type::mass, rhs_type::is_const>
          (legendre, xleft[d], xright[d], level, nullptr, t1d.rhs_const(), raw_diag);
      }
      break;
    case operation_type::div:
      if (t1d.rhs()) {
        gen_tri_cmat<P, operation_type::div, rhs_type::is_func>
          (legendre, xleft[d], xright[d], level, t1d.rhs(), 0, t1d.flux(), t1d.boundary(), raw_tri);
      } else {
        gen_tri_cmat<P, operation_type::div, rhs_type::is_const>
          (legendre, xleft[d], xright[d], level, nullptr, t1d.rhs_const(), t1d.flux(), t1d.boundary(), raw_tri);
      }
      break;
    case operation_type::grad:
      if (t1d.rhs()) {
        gen_tri_cmat<P, operation_type::grad, rhs_type::is_func>
          (legendre, xleft[d], xright[d], level, t1d.rhs(), 0, t1d.flux(), t1d.boundary(), raw_tri);
      } else {
        gen_tri_cmat<P, operation_type::grad, rhs_type::is_const>
          (legendre, xleft[d], xright[d], level, nullptr, t1d.rhs_const(), t1d.flux(), t1d.boundary(), raw_tri);
      }
      break;
    case operation_type::penalty:
      if (t1d.rhs()) {
        gen_tri_cmat<P, operation_type::penalty, rhs_type::is_func>
          (legendre, xleft[d], xright[d], level, t1d.rhs(), 0, t1d.flux(), t1d.boundary(), raw_tri);
      } else {
        gen_tri_cmat<P, operation_type::penalty, rhs_type::is_const>
          (legendre, xleft[d], xright[d], level, nullptr, t1d.rhs_const(), t1d.flux(), t1d.boundary(), raw_tri);
      }
      break;
    default:
      // must be a unreachable
      break;
  }
  if (t1d.lhs()) { // we have a lhs mass
    gen_diag_cmat<P, operation_type::mass, rhs_type::is_func>
      (legendre, xleft[d], xright[d], level, t1d.lhs(), 0, raw_mass);
    if (t1d.is_mass())
      raw_mass.apply_inverse(legendre.pdof, raw_diag);
    else
      raw_mass.apply_inverse(legendre.pdof, raw_tri);
  }
}

template<typename P>
void term_manager<P>::rebuld_chain(
    int const d, term_1d<P> const &t1d, int const level, bool &is_diag,
    block_diag_matrix<P> &raw_diag, block_tri_matrix<P> &raw_tri)
{
  expect(t1d.is_chain());
  int const num_chain = t1d.num_chain();
  expect(num_chain > 1);

  if (num_chain >= 2 and t1.empty())
    t1.resize(1);
  if (num_chain >= 3 and t2.empty())
    t2.resize(1);

  is_diag = true;
  for (int i : iindexof(num_chain)) {
    if (not t1d[i].is_mass()) {
      is_diag = false;
      break;
    }
  }

  // workspace matrices
  static block_tri_matrix<P> raw_tri0, raw_tri1;
  static block_diag_matrix<P> raw_diag0, raw_diag1;

  if (is_diag) { // a bunch of diag matrices, easy case
    // raw_tri will not be referenced, it's just passed in
    // using raw_diag to make the intermediate matrices, until the last one
    // the last product has to be written to raw_diag
    block_diag_matrix<P> *diag0 = &raw_diag0;
    block_diag_matrix<P> *diag1 = &raw_diag1;
    build_raw_mat(d, t1d[num_chain - 1], level, *diag0, raw_tri);
    for (int i = num_chain - 2; i > 0; i--) {
      build_raw_mat(d, t1d[i], level, raw_diag, raw_tri);
      diag1->check_resize(raw_diag);
      gemm_block_diag(legendre.pdof, raw_diag, *diag0, *diag1);
      std::swap(diag0, diag1);
    }
    build_raw_mat(d, t1d[0], level, *diag1, raw_tri);
    raw_diag.check_resize(*diag1);
    gemm_block_diag(legendre.pdof, *diag1, *diag0, raw_diag);
    return;
  }

  // the final is always a tri-diagonal matrix
  // but we have to keep track of upper/lower and diagonal
  block_diag_matrix<P> *diag0 = &raw_diag0;
  block_diag_matrix<P> *diag1 = &raw_diag1;

  block_tri_matrix<P> *tri0 = &raw_tri0;
  block_tri_matrix<P> *tri1 = &raw_tri1;

  enum class fill {
    diag, upper, lower, tri
  };
  auto get_fill = [](term_1d<P> const &t)
      -> fill {
      if (t.is_mass()) return fill::diag;
      if (t.is_penalty() or t.flux() == flux_type::central)
        return fill::tri;
      return (t.flux() == flux_type::upwind) ? fill::upper : fill::lower;
    };

  fill current = get_fill(t1d[num_chain - 1]);
  build_raw_mat(d, t1d[num_chain - 1], level, *diag0, *tri0);

  for (int i = num_chain - 2; i > 0; i--)
  {
    build_raw_mat(d, t1d[i], level, raw_diag, raw_tri);
    // the result is in either raw_diag or raw_tri and must be multiplied and put
    // into either diag1 or tri1, then those should swap with diag0 and tri0
    switch (get_fill(t1d[i])) // computed fill
    {
      case fill::diag: // computed a diagonal fill
        if (current == fill::diag){ // diag-to-diag
          diag1->check_resize(raw_diag);
          gemm_block_diag(legendre.pdof, raw_diag, *diag0, *diag1);
          std::swap(diag0, diag1);
        } else { // the form of the tri-matrix does not matter
          tri1->check_resize(raw_diag);
          gemm_diag_tri(legendre.pdof, raw_diag, *tri0, *tri1);
          std::swap(tri0, tri1);
        }
        break;
      case fill::upper: // computed upper fill
        if (current == fill::diag) {
          tri1->check_resize(raw_tri);
          gemm_tri_diag(legendre.pdof, raw_tri, *diag0, *tri1);
          std::swap(tri0, tri1);
          current = fill::upper;
        } else {
          // must be fill::lower, cannot be another upper or tri
          tri1->check_resize(raw_tri);
          gemm_block_tri_ul(legendre.pdof, raw_tri, *tri0, *tri1);
          std::swap(tri0, tri1);
          current = fill::tri;
        }
        break;
      case fill::lower: // computed upper fill
        if (current == fill::diag ) {
          tri1->check_resize(raw_tri);
          gemm_tri_diag(legendre.pdof, raw_tri, *diag0, *tri1);
          std::swap(tri0, tri1);
          current = fill::lower;
        } else {
          // must be fill::upper, cannot be another lower or tri
          tri1->check_resize(raw_tri);
          gemm_block_tri_lu(legendre.pdof, raw_tri, *tri0, *tri1);
          std::swap(tri0, tri1);
          current = fill::tri;
        }
        break;
      default: // computed tri matrix, the current must be diagonal
        tri1->check_resize(raw_tri);
        gemm_tri_diag(legendre.pdof, raw_tri, *diag0, *tri1);
        std::swap(tri0, tri1);
        current = fill::tri;
        break;
    }
  }

  // last term, compute in diag1/tri1 and multiply into raw_tri
  build_raw_mat(d, t1d[0], level, *diag1, *tri1);

  switch (get_fill(t1d[0])) // computed fill
  {
    case fill::diag: // computed a diagonal fill
      // the rest must be a tri-diagonal matrix already
      raw_tri.check_resize(*tri0);
      gemm_diag_tri(legendre.pdof, *diag1, *tri0, raw_tri);
      break;
    case fill::upper: // computed upper fill
      if (current == fill::diag) {
        raw_tri.check_resize(*tri1);
        gemm_tri_diag(legendre.pdof, *tri1, *diag0, raw_tri);
      } else {
        // must be fill::lower, cannot be another upper or tri
        raw_tri.check_resize(*tri1);
        gemm_block_tri_ul(legendre.pdof, *tri1, *tri0, raw_tri);
      }
      break;
    case fill::lower: // computed upper fill
      if (current == fill::diag ) {
        raw_tri.check_resize(*tri1);
        gemm_tri_diag(legendre.pdof, *tri1, *diag0, raw_tri);
      } else {
        // must be fill::upper, cannot be another lower or tri
        raw_tri.check_resize(*tri1);
        gemm_block_tri_lu(legendre.pdof, *tri1, *tri0, raw_tri);
      }
      break;
    default: // computed tri matrix, the current must be diagonal
      raw_tri.check_resize(*tri1);
      gemm_tri_diag(legendre.pdof, *tri1, *diag0, raw_tri);
      break;
  }
}


template<typename P>
void term_manager<P>::apply_all(
    sparse_grid const &grid, connection_patterns const &conns,
    P alpha, std::vector<P> const &x, P beta, std::vector<P> &y) const
{
  expect(x.size() == y.size());
  expect(x.size() == kwork.w1.size());

  P b = beta; // on first iteration, overwrite y

  auto it = terms.begin();
  while (it < terms.end())
  {
    if (it->num_chain == 1) {
      kron_term(grid, conns, *it, alpha, x, b, y);
      ++it;
    } else {
      // dealing with a chain
      int const num_chain = it->num_chain;

      kron_term(grid, conns, *(it + num_chain - 1), 1, x, 0, t1);
      for (int i = num_chain - 2; i > 0; --i) {
        kron_term(grid, conns, *(it + i), 1, t1, 0, t2);
        std::swap(t1, t2);
      }
      kron_term(grid, conns, *it, alpha, t1, b, y);

      it += it->num_chain;
    }

    b = 1; // next iteration appends on y
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template struct term_entry<double>;
template struct term_manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template struct term_entry<float>;
template struct term_manager<float>;
#endif

}
