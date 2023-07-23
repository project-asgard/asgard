#pragma once
#include "build_info.hpp"

#include "sparse.hpp"
#include "tensors.hpp"

namespace asgard::preconditioner
{

template<typename P>
class preconditioner
{
public:
  preconditioner() {}
  ~preconditioner() {}

  // constructs from an already existing matrix
  preconditioner(fk::matrix<P> const &&M)
  {
    this->precond = std::move(M);
    this->pivots.resize(M.nrows());
    this->is_factored = false;
  }

  virtual void construct(int const n)
  {
    this->precond.clear_and_resize(n, n);
    pivots.resize(n);
  }
  virtual void
  construct(PDE<P> const &pde, elements::table const &table, int const n)
  {
    ignore(pde);
    ignore(table);
    this->construct(n);
  }

  virtual void apply(fk::vector<P> &B)
  {
    if (!is_factored)
    {
      fm::gesv(precond, B, pivots);
      is_factored = true;
    }
    else
    {
      fm::getrs(precond, B, pivots);
    }
  }
  // virtual void apply(fk::vector<P> &B, std::vector<int> &pivots) {}

  bool empty() const { return this->precond.empty(); }

protected:
  bool is_factored = false;
  fk::matrix<P> precond;
  std::vector<int> pivots;
};

template<typename P>
class eye_preconditioner : public preconditioner<P>
{
public:
  eye_preconditioner() {}

  virtual void construct(int const n) override
  {
    std::cout << "constructing eye preconditioner\n";
    fk::sparse<P> sp_y(speye<P>(n));
    this->precond.clear_and_resize(n, n) = std::move(sp_y.to_dense());
    this->pivots.resize(n);
    std::cout << " end eye construct\n";
  }

  virtual void construct(PDE<P> const &pde, elements::table const &table,
                         int const n) override
  {
    this->construct(n);
  }

  // virtual void apply(fk::vector<P> &B) override {}
  // virtual void apply(fk::vector<P> &B, std::vector<int> &pivots) override {}
};

template<typename P>
class block_jacobi_preconditioner : public preconditioner<P>
{
public:
  block_jacobi_preconditioner() {}
  ~block_jacobi_preconditioner() {}

  virtual void construct(PDE<P> const &pde, elements::table const &table,
                         int const n) override
  {
    // calculates a block jacobi preconditioner into and updates the precond
    // matrix
    // expect(this->precond.nrows() == n);
    // expect(this->precond.ncols() == n);
    this->precond.clear_and_resize(n, n);
    this->pivots.resize(n);

    int const num_dims = pde.num_dims;
    int const degree   = pde.get_dimensions()[0].get_degree();

    std::cout << "PRECOND SIZE n = " << n << "\n";

#pragma omp parallel for
    for (int element = 0; element < table.size(); element++)
    {
      fk::vector<int> const &coords = table.get_coords(element);

      // get 1D operator indices for each dimension
      int indices[num_dims];
      for (int i = 0; i < num_dims; ++i)
      {
        indices[i] =
            elements::get_1d_index(coords[i], coords[i + num_dims]) * degree;
      }

      // the index where this block is placed in the preconditioner matrix
      int const matrix_offset = element * std::pow(degree, num_dims);

      fk::matrix<P> kron0(1, 1);
      kron0(0, 0) = 1.0;

      for (int term = 0; term < pde.num_terms; term++)
      {
        // Vector containing coefficient blocks for each dimension
        std::vector<fk::matrix<P, mem_type::const_view>> blocks;
        for (int dim = 0; dim < num_dims; dim++)
        {
          int const start_index = indices[dim];
          int const end_index   = indices[dim] + degree - 1;

          blocks.push_back(fk::matrix<P, mem_type::const_view>(
              pde.get_coefficients(term, 0), start_index, end_index,
              start_index, end_index));
        }

        // Vector containing kron products of each block. The final kron product
        // is stored at the last element.
        std::vector<fk::matrix<P>> krons;
        krons.push_back(kron0);
        for (int dim = 0; dim < num_dims; dim++)
        {
          krons.push_back(std::move(krons[dim].kron(blocks[dim])));
        }

        expect(krons.back().nrows() == std::pow(degree, num_dims));
        expect(krons.back().ncols() == std::pow(degree, num_dims));

        // sum the kron product into the preconditioner matrix
        this->precond.set_submatrix(
            matrix_offset, matrix_offset,
            fk::matrix<P, mem_type::view>(
                this->precond, matrix_offset,
                matrix_offset + std::pow(degree, num_dims) - 1, matrix_offset,
                matrix_offset + std::pow(degree, num_dims) - 1) +
                krons.back());
      }
    }
  }
};

} // namespace asgard::preconditioner