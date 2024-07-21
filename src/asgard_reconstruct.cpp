#include "asgard_reconstruct.hpp"

#include "asgard_wavelet_basis.hpp"

namespace asgard
{

template<typename precision>
reconstruct_solution::reconstruct_solution(
    int dims, int64_t num_cells, int const asg_cells[],
    int degree, precision const solution[])
    : pterms_(degree), block_size_(fm::ipow(pterms_, dims)), domain_scale(1.0)
{
  vector2d<int> raw_cells(dims, num_cells);
  for (int64_t i = 0; i < num_cells; i++)
    asg2tsg_convert(dims, asg_cells + 2 * dims * i, raw_cells[i]);

  std::fill(inv_slope.begin(), inv_slope.end(), 1.0);
  std::fill(shift.begin(), shift.end(), 0.0);

  // compute a map that gives the sorted order of the indexes
  std::vector<int64_t> imap(num_cells);
  std::iota(imap.begin(), imap.end(), 0);

  std::sort(imap.begin(), imap.end(),
            [&](int64_t a, int64_t b) -> bool {
              for (int d = 0; d < dims; d++)
              {
                if (raw_cells[a][d] < raw_cells[b][d])
                  return true;
                if (raw_cells[a][d] > raw_cells[b][d])
                  return false;
              }
              return false; // equal should be false, as in < operator
            });

  // copy the solution and indexes in the correct order
  coeff_.resize(num_cells * block_size_);
  std::vector<int> sorted_cells(dims * num_cells);

  auto is = sorted_cells.begin();
  auto ic = coeff_.begin();
  for (int64_t i = 0; i < num_cells; i++)
  {
    is = std::copy_n(raw_cells[imap[i]], dims, is);
    ic = std::copy_n(solution + block_size_ * imap[i], block_size_, ic);
  }

  cells_ = indexset(dims, std::move(sorted_cells));

  // analyze the graph and prepare cache data
  build_tree();
}

void reconstruct_solution::set_domain_bounds(double const amin[], double const amax[])
{
  domain_scale = 1.0;
  for (int d = 0; d < cells_.num_dimensions(); d++)
  {
    shift[d]    = amin[d];
    double size = amax[d] - amin[d];

    inv_slope[d]  = 1.0 / size;
    domain_scale *= size;
  }
  domain_scale = 1.0 / std::sqrt(domain_scale);
}

void reconstruct_solution::reconstruct(double const x[], int num_x, double y[]) const
{
  int const num_dimensions = cells_.num_dimensions();
  span2d<double const> x2d(num_dimensions, num_x, x);
#pragma omp parallel
{
  std::array<double, max_num_dimensions> xn;
#pragma omp for
  for (int i = 0; i < num_x; i++)
  {
    for (int d = 0; d < num_dimensions; d++)
      xn[d] = inv_slope[d] * (x2d[i][d] - shift[d]);
    y[i] = domain_scale * walk_tree(xn.data());
  }
}
}

vector2d<int> reconstruct_solution::compute_dag_down() const
{
  int constexpr max_1d_kids = 2; // change for a different hierarchy

  int num_dimensions = cells_.num_dimensions();
  int64_t num_cells  = cells_.num_indexes();

  vector2d<int> kids(num_dimensions * max_1d_kids, num_cells);

#pragma omp parallel
{
  std::array<int, max_num_dimensions> kid;

#pragma omp for
  for (int64_t i = 0; i < num_cells; i++)
  {
    std::copy_n(cells_[i], num_dimensions, kid.data());
    int *family = kids[i];

    for (int j = 0; j < num_dimensions; j++)
    {
      int const current = kid[j];
      if (current > 0)
      {
        kid[j]    = 1;
        *family++ = cells_.find(kid.data());
        *family++ = -1;
      }
      else
      {
        kid[j]    = 2 * current;
        *family++ = cells_.find(kid.data());
        ++kid[j];
        *family++ = cells_.find(kid.data());
      }
      kid[j] = current;
    } // for j - num_dimensions
  } // #pragma omp for
} // #pragma omp parallel
  return kids;
}

std::vector<int> reconstruct_solution::compute_levels() const
{
  int num_dimensions = cells_.num_dimensions();
  int64_t num_cells  = cells_.num_indexes();

  auto level1d = [](int p) -> int {
    return fm::intlog2(p) + ((p > 0) ? 1 : 0);
  };

  std::vector<int> level(num_cells);
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < num_cells; i++)
  {
    int const *p = cells_[i];
    level[i] = level1d(p[0]);
    for (int j = 1; j < num_dimensions; j++)
      level[i] += level1d(p[j]);
  }
  return level;
}

void reconstruct_solution::build_tree()
{
  vector2d<int> kids     = compute_dag_down();
  std::vector<int> level = compute_levels();

  int max_kids      = static_cast<int>(kids.stride());
  int64_t num_cells = cells_.num_indexes();
  vector2d<int> tree(max_kids, std::vector<int>(max_kids * num_cells, -1));

  std::vector<bool> free(num_cells, true);

  // monkey business here:
  // monkeys are cute animals that are very good at moving around the branches
  //   of trees or meshed branches of jungles that can form complex graphs
  // monkey_count indicates the offset of the current branch that the monkey
  //   is holding and will process recursively
  // monkey_tail always points the branch that the monkey came from

  // at level 31 we overflow the 32-bit int, no need for more
  std::array<int, 31> monkey_count;
  std::array<int, 31> monkey_tail;

  roots.clear();
  int next_root = 0; // index at 0 is always the zero-index and the first root

  while (next_root != -1)
  {
    roots.push_back(next_root);
    free[next_root] = false;

    monkey_tail[0] = next_root;
    monkey_count[0] = 0;
    size_t current = 0;

    while (monkey_count[0] < max_kids){
      if (monkey_count[current] < max_kids){
        int kid = kids[monkey_tail[current]][monkey_count[current]];
        if (kid == -1 or not free[kid]){
            monkey_count[current]++; // no kid, keep counting
        }else{
            tree[monkey_tail[current]][monkey_count[current]] = kid;
            monkey_count[++current] = 0;
            monkey_tail[current] = kid;
            free[kid] = false;
        }
      }else{
        monkey_count[--current]++; // done with all kids here
      }
    }

    next_root = -1;
    int next_level = 32;
    for (int64_t i = 0; i < num_cells; i++){
      if (free[i] and level[i] < next_level){
        next_root  = i;
        next_level = level[i];
      }
    }
  }

  pntr.resize(num_cells + 1, 0);
  for (int64_t i = 0; i < num_cells; i++)
    pntr[i + 1] = pntr[i] + std::count_if(tree[i], tree[i] + max_kids, [](int k)->bool{ return (k > -1); });

  indx.resize(std::max(pntr[num_cells], 1));
  std::copy_if(tree[0], tree[0] + tree.total_size(), indx.begin(), [](int t)->bool{ return (t > -1); });
}

std::optional<double>
reconstruct_solution::basis_value(int const p[], double const x[],
                                  double const c[]) const
{
  // first translat the x values to normalized values for the given cell
  std::array<double, max_num_dimensions> xn;
  std::array<double, max_num_dimensions> w;
  int const &num_dimensions = cells_.num_dimensions();
  for (int d = 0; d < num_dimensions; d++)
  {
    int p2l2;
    fm::intlog2_pow2pows2(p[d], p2l2, w[d]);
    xn[d] = (p[d] > 1) ? (p2l2 * (x[d] + 1.0) - p[d]) : x[d];

    // if the normalized point is out of bounds, return no-value
    if (xn[d] < 0.0 or xn[d] > 1.0)
      return std::optional<double>();
  }

  // loop over all the basis inside the cell
  double sum = 0.0;
  for (int i = 0; i < block_size_; i++)
  {
    int t = i;
    double val = 1.0;
    for (int d = num_dimensions - 1; d >= 0; d--)
    {
      val *= w[d] * linear_basis<double>::pbasis(p[d], t % pterms_, xn[d]);
      t /= pterms_;
    }

    sum += val * c[i];
  }

  return std::optional<double>(sum);
}

double reconstruct_solution::walk_tree(double const x[]) const
{
  std::array<int, 31> monkey_count;
  std::array<int, 31> monkey_tail;

  std::optional<double> basis;
  double result = 0;

  for(const auto &r : roots){
    basis = basis_value(cells_[r], x, coeff_.data() + r * block_size_);

    if (basis)
    {
      result += basis.value();

      int current = 0;
      monkey_tail[0] = r;
      monkey_count[0] = pntr[r];

      while (monkey_count[0] < pntr[monkey_tail[0] + 1])
      {
        if (monkey_count[current] < pntr[monkey_tail[current]+1]){
          int const p = indx[monkey_count[current]];
          basis = basis_value(cells_[p], x, coeff_.data() + p * block_size_);

          if (basis)
          {
            result += basis.value();

            monkey_tail[++current] = p;
            monkey_count[current] = pntr[p];
          }else{
            monkey_count[current]++;
          }
        }else{
          monkey_count[--current]++;
        }
      }
    }
  }

  return result;
}

} // namespace asgard

extern "C"
{

void *asgard_make_dreconstruct_solution(
    int dims, int64_t num_cells, int const asg_cells[],
    int degree, double const solution[])
{
  return reinterpret_cast<void *>(new asgard::reconstruct_solution(
      dims, num_cells, asg_cells, degree, solution));
}
void *asgard_make_freconstruct_solution(
    int dims, int64_t num_cells, int const asg_cells[],
    int degree, float const solution[])
{
  return reinterpret_cast<void *>(new asgard::reconstruct_solution(
      dims, num_cells, asg_cells, degree, solution));
}

void asgard_pydelete_reconstruct_solution(void *pntr)
{
  delete reinterpret_cast<asgard::reconstruct_solution *>(pntr);
}
void asgard_delete_reconstruct_solution(void **pntr)
{
  asgard_pydelete_reconstruct_solution(*pntr);
  *pntr = nullptr;
}

void asgard_reconstruct_solution_setbounds(void *pntr, double const amin[],
                                           double const amax[])
{
  reinterpret_cast<asgard::reconstruct_solution *>(pntr)->set_domain_bounds(amin, amax);
}

void asgard_reconstruct_solution(void *pntr, double const x[], int num_x, double y[])
{
  reinterpret_cast<asgard::reconstruct_solution *>(pntr)
      ->reconstruct(x, num_x, y);
}

} // extern "C"
