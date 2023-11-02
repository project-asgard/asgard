#pragma once
#include "build_info.hpp"

#include "asgard_matrix.hpp"
#include "asgard_vector.hpp"
#include "elements.hpp"
#include "pde.hpp"
#include "sparse.hpp"

namespace asgard::preconditioner
{
template<typename P, resource resrc = resource::host>
class preconditioner
{
public:
  preconditioner() {}
  ~preconditioner() {}

  // constructs from an already existing matrix
  preconditioner(fk::matrix<P, mem_type::owner, resrc> const &&M) : precond{M}
  {
    pivots.resize(M.nrows());

    if (!precond.empty())
    {
      // factorize the preconditioner initially
      fm::getrf(precond, pivots);
    }
  }

  virtual void construct(int const n)
  {
    precond.clear_and_resize(n, n);
    pivots = fk::vector<int, mem_type::owner, resrc>(n);

    // factorize the preconditioner initially
    fm::getrf(precond, pivots);
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

  virtual void apply(fk::vector<P, mem_type::owner, resrc> &B) const
  {
    fm::getrs(precond, B, pivots);
  }

#ifdef ASGARD_USE_CUDA
  virtual void
  apply_batched(fk::vector<P, mem_type::owner, resource::device> &B)
  {
    ignore(B);
  }
#endif

  virtual bool empty() const { return this->precond.empty(); }

  virtual fk::matrix<P, mem_type::owner, resource::host> get_matrix() const
  {
    if constexpr (resrc == resource::host)
    {
      return this->precond;
    }
    else if constexpr (resrc == resource::device)
    {
      return this->precond.clone_onto_host();
    }
  }

protected:
  fk::matrix<P, mem_type::owner, resrc> precond;
  fk::vector<int, mem_type::owner, resrc> pivots;
};

template<typename P, resource resrc = resource::host>
class block_jacobi_preconditioner : public preconditioner<P, resrc>
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

    int const block_size = std::pow(degree, num_dims);
    expect(n == num_blocks * block_size);

    if constexpr (resrc == resource::device)
    {
#ifdef ASGARD_USE_CUDA
      this->dev_precond_blks =
          std::vector<fk::matrix<P, mem_type::owner, resource::device>>(
              this->num_blocks);
#endif
    }
    else if constexpr (resrc == resource::host)
    {
      this->blk_pivots   = std::vector<fk::vector<int>>(this->num_blocks);
      int const piv_size = std::pow(this->degree, this->num_dims);
      for (int i = 0; i < this->num_blocks; i++)
      {
        blk_pivots[i] = fk::vector<int>(piv_size);
      }
    }

    fk::matrix<P> const I = eye<P>(block_size);

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
      precond_blks[element] = I - fm::scal(dt, precond_blks[element]);
    }

    if constexpr (resrc == resource::device)
    {
#ifdef ASGARD_USE_CUDA
      // copy blocks over to device
      int const piv_size = std::pow(this->degree, this->num_dims);
      for (int b = 0; b < this->num_blocks; b++)
      {
        this->dev_precond_blks[b] = this->precond_blks[b].clone_onto_device();
      }

      // pivot indices are stored as a 1D vector for all blocks
      this->dev_pivots = fk::vector<int, mem_type::owner, resource::device>(
          piv_size * num_blocks);
#endif
    }

    // factorize all blocks initially
    for (int block = 0; block < this->num_blocks; block++)
    {
      if constexpr (resrc == resource::host)
      {
        fm::getrf(precond_blks[block], this->blk_pivots[block]);
      }
      else if constexpr (resrc == resource::device)
      {
#ifdef ASGARD_USE_CUDA
        // TODO: this should be batched
        lib_dispatch::getrf<resource::device>(
            dev_precond_blks[block].nrows(), dev_precond_blks[block].ncols(),
            dev_precond_blks[block].data(), dev_precond_blks[block].stride(),
            dev_pivots.data() + (block * block_size));
#endif
      }
    }
  };

  virtual void apply(fk::vector<P, mem_type::owner, resrc> &B) const override
  {
    auto id = asgard::tools::timer.start("precond apply");
    for (int block = 0; block < this->num_blocks; block++)
    {
      apply_block(block, B);
    }
    asgard::tools::timer.stop(id);
  }

  virtual bool empty() const override { return this->precond_blks.empty(); }

  void apply_block(int const block_index,
                   fk::vector<P, mem_type::owner, resrc> &B) const
  {
    int const block_size = std::pow(this->degree, this->num_dims);
    int const offset     = block_index * block_size;

    // applies a single block of the preconditioner
    if constexpr (resrc == resource::host)
    {
      // extract the given block from the preconditioner matrix
      auto B_block =
          fk::vector<P, mem_type::view>(B, offset, offset + block_size - 1);

      fm::getrs(precond_blks[block_index], B_block,
                this->blk_pivots[block_index]);
    }
    else if constexpr (resrc == resource::device)
    {
      // TODO: this can be removed, the batched version is used instead
#ifdef ASGARD_USE_CUDA
      // extract the given block from the preconditioner matrix
      auto B_block = fk::vector<P, mem_type::view, resource::device>(
          B, offset, offset + block_size - 1);

      fm::getrs(dev_precond_blks[block_index], B_block,
                this->dev_blk_pivots[block_index]);
#endif
    }
  }

#ifdef ASGARD_USE_CUDA
  virtual void
  apply_batched(fk::vector<P, mem_type::owner, resource::device> &B) override
  {
    auto id = asgard::tools::timer.start("precond apply batched");

    int const block_size = std::pow(degree, num_dims);

    std::vector<fk::vector<P, mem_type::view, resource::device>> B_blocks(
        num_blocks);

    std::vector<P *> B_block_ptrs(num_blocks);
    std::vector<P *> precond_blk_ptrs(num_blocks);

    for (int block = 0; block < num_blocks; block++)
    {
      int const offset = block * block_size;
      // extract the given block from the preconditioner matrix
      B_blocks[block] = fk::vector<P, mem_type::view, resource::device>(
          B, offset, offset + block_size - 1);

      B_block_ptrs[block]     = B_blocks[block].data();
      precond_blk_ptrs[block] = dev_precond_blks[block].data();
    }

    lib_dispatch::batched_getrs('N', block_size, 1, precond_blk_ptrs.data(),
                                dev_precond_blks[0].stride(), dev_pivots.data(),
                                B_block_ptrs.data(), B_blocks[0].size(),
                                num_blocks);

    asgard::tools::timer.stop(id);
  }
#endif

  virtual fk::matrix<P, mem_type::owner, resource::host>
  get_matrix() const override
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
  std::vector<fk::vector<int>> blk_pivots;

#ifdef ASGARD_USE_CUDA
  std::vector<fk::matrix<P, mem_type::owner, resource::device>>
      dev_precond_blks;
  std::vector<fk::vector<int, mem_type::owner, resource::device>>
      dev_blk_pivots;

  fk::vector<int, mem_type::owner, resource::device> dev_pivots;
#endif
};

} // namespace asgard::preconditioner
