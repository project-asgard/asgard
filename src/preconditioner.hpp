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
    this->is_factored = false;
    pivots.resize(n);
  }

  virtual void construct(PDE<P> const &pde, elements::table const &table,
                         int const n, P const dt,
                         imex_flag const imex = imex_flag::unspecified)
  {
    ignore(pde);
    ignore(table);
    ignore(dt);
    ignore(imex);
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

  virtual bool empty() const { return this->precond.empty(); }

  virtual fk::matrix<P> get_matrix() const { return this->precond; }

  bool factored() const { return is_factored; }

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
    fk::sparse<P> sp_y(speye<P>(n));
    this->precond.clear_and_resize(n, n) = std::move(sp_y.to_dense());
    this->pivots.resize(n);
    this->is_factored = false;
  }

  virtual void construct(PDE<P> const &pde, elements::table const &table,
                         int const n, P const dt,
                         imex_flag const imex = imex_flag::unspecified) override
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
                         int const n, P const dt,
                         imex_flag const imex = imex_flag::unspecified) override
  {
    // calculates a block jacobi preconditioner into and updates the precond
    // matrix
    this->num_blocks   = table.size();
    this->degree       = pde.get_dimensions()[0].get_degree();
    this->num_dims     = pde.num_dims;
    this->precond_blks = std::vector<fk::matrix<P>>(this->num_blocks);
    this->blk_pivots   = std::vector<std::vector<int>>(this->num_blocks);
    this->is_factored  = false;

    int const block_size = std::pow(degree, num_dims);
    expect(n == num_blocks * block_size);

#pragma omp parallel for
    for (int element = 0; element < table.size(); element++)
    {
      fk::vector<int> const &coords = table.get_coords(element);

      precond_blks[element].clear_and_resize(block_size, block_size);

      // get 1D operator indices for each dimension
      std::vector<int> indices(num_dims);
      for (int i = 0; i < num_dims; ++i)
      {
        indices[i] =
            elements::get_1d_index(coords[i], coords[i + num_dims]) * degree;
      }

      // the index where this block is placed in the preconditioner matrix
      // int const matrix_offset = element * std::pow(degree, num_dims);

      for (int term = 0; term < pde.num_terms; term++)
      {
        // Vector containing coefficient blocks for each dimension
        std::vector<fk::matrix<P, mem_type::const_view>> blocks;
        for (int dim = 0; dim < num_dims; dim++)
        {
          int const start_index = indices[dim];
          int const end_index   = indices[dim] + degree - 1;

          blocks.push_back(fk::matrix<P, mem_type::const_view>(
              pde.get_coefficients(term, dim), start_index, end_index,
              start_index, end_index));
        }

        // Vector containing kron products of each block. The final kron product
        // is stored at the last element.
        std::vector<fk::matrix<P>> krons;

        fk::matrix<P> kron0(1, 1);
        // if using imex, include only terms that match the flag
        if (imex == imex_flag::unspecified ||
            pde.get_terms()[term][0].flag == imex)
        {
          kron0(0, 0) = 1.0;
        }
        else
        {
          kron0(0, 0) = 0.0;
        }

        krons.push_back(kron0);
        for (int dim = 0; dim < num_dims; dim++)
        {
          krons.push_back(std::move(krons[dim].kron(blocks[dim])));
        }

        expect(krons.back().nrows() == block_size);
        expect(krons.back().ncols() == block_size);

        // sum the kron product into the preconditioner matrix
        precond_blks[element] = precond_blks[element] + krons[num_dims];
      }

      // Compute M = I - dt*M
      precond_blks[element] =
          eye<P>(block_size) - fm::scal(dt, precond_blks[element]);
    }
  }

  virtual void apply(fk::vector<P> &B) override
  {
    auto id = asgard::tools::timer.start("precond apply");
    if (!this->is_factored)
    {
      int const piv_size = std::pow(this->degree, this->num_dims);
      for (int i = 0; i < this->num_blocks; i++)
      {
        blk_pivots[i] = std::vector<int>(piv_size);
      }
    }

    for (int block = 0; block < this->num_blocks; block++)
    {
      apply_block(block, B);
    }

    if (!this->is_factored)
    {
      this->is_factored = true;
    }
    asgard::tools::timer.stop(id);
  }

  virtual bool empty() const override { return this->precond_blks.empty(); }

  void apply_block(int const block_index, fk::vector<P> &B)
  {
    int const block_size = std::pow(this->degree, this->num_dims);
    int const offset     = block_index * block_size;

    // extract the given block from the preconditioner matrix
    auto B_block =
        fk::vector<P, mem_type::view>(B, offset, offset + block_size - 1);

    if (!this->is_factored)
    {
      fm::gesv(precond_blks[block_index], B_block,
               this->blk_pivots[block_index]);
    }
    else
    {
      fm::getrs(precond_blks[block_index], B_block,
                this->blk_pivots[block_index]);
    }
  }

  virtual fk::matrix<P> get_matrix() const override
  {
    int const offset = std::pow(degree, num_dims);
    int const n      = num_blocks * offset;

    fk::matrix<P> dense_precond(n, n);
    for (int blk = 0; blk < num_blocks; blk++)
    {
      expect(precond_blks[blk].nrows() == offset);
      expect(precond_blks[blk].ncols() == offset);

      int const row = blk * offset;

      dense_precond.set_submatrix(row, row, precond_blks[blk]);
    }

    return dense_precond;
  }

  int num_blocks;
  int degree;
  int num_dims;

  std::vector<fk::matrix<P>> precond_blks;
  std::vector<std::vector<int>> blk_pivots;
};

} // namespace asgard::preconditioner
