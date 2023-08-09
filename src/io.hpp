#pragma once

#include "build_info.hpp"

#include "asgard_matrix.hpp"
#include "asgard_vector.hpp"
#include "pde.hpp"
#include "program_options.hpp"
#include "solver.hpp"
#include "tools.hpp"
#include "transformations.hpp"

// workaround for missing include issue with highfive
// clang-format off
#include <numeric>
#include <filesystem>
#include <highfive/H5Easy.hpp>
#include <highfive/H5DataType.hpp>
#include <highfive/H5DataSpace.hpp>
// clang-format on

namespace asgard
{
template<typename P>
HighFive::CompoundType create_timing_stats()
{
  return {{"avg", HighFive::create_datatype<double>()},
          {"min", HighFive::create_datatype<double>()},
          {"max", HighFive::create_datatype<double>()},
          {"med", HighFive::create_datatype<double>()},
          {"gflops", HighFive::create_datatype<double>()},
          {"ncalls", HighFive::create_datatype<size_t>()}};
}
} // namespace asgard

HIGHFIVE_REGISTER_TYPE(asgard::tools::timing_stats,
                       asgard::create_timing_stats<double>)

namespace asgard
{
template<typename P>
HighFive::DataSet
initialize_output_file(fk::vector<P> const &vec,
                       std::string const output_dataset_name = "asgard")
{
  std::string const output_file_name = output_dataset_name + ".h5";
  unsigned int vec_size              = (unsigned int)vec.size();

  // Open file object
  HighFive::File file(output_file_name, HighFive::File::ReadWrite |
                                            HighFive::File::Create |
                                            HighFive::File::Truncate);

  // Create dataspace
  HighFive::DataSpace dataspace = HighFive::DataSpace(
      {1, vec_size}, {HighFive::DataSpace::UNLIMITED, vec_size});

  // Use chunking
  HighFive::DataSetCreateProps props;
  props.add(HighFive::Chunking(std::vector<hsize_t>{2, 2}));

  // Create dataset
  HighFive::DataSet dataset = file.createDataSet(
      output_dataset_name, dataspace, HighFive::AtomicType<P>(), props);

  // Write initial contion to t=0 slice of output file
  dataset.select({0, 0}, {1, vec_size}).write(vec.to_std());

  return dataset;
}

template<typename P>
void update_output_file(HighFive::DataSet &dataset, fk::vector<P> const &vec,
                        std::string const output_dataset_name = "asgard")
{
  std::string const output_file_name = output_dataset_name + ".h5";
  unsigned int vec_size              = (unsigned int)vec.size();

  // Get the size of the existing dataset
  auto dataset_size = dataset.getDimensions();
  // Resize in the time dimension by 1
  dataset.resize({dataset_size[0] + 1, dataset_size[1]});
  // Write the latest vec into the new row
  dataset.select({dataset_size[0], 0}, {1, vec_size}).write(vec.to_std());
}

template<typename P>
void generate_initial_moments(
    PDE<P> &pde, options const &program_opts,
    adapt::distributed_grid<P> const &adaptive_grid,
    asgard::basis::wavelet_transform<P, resource::host> const &transformer,
    fk::vector<P> const &initial_condition)
{
  // create 1D version of PDE and element table for wavelet->realspace
  // mappings
  PDE pde_1d = PDE(pde, PDE<P>::extract_dim0);
  adapt::distributed_grid adaptive_grid_1d(pde_1d, program_opts);

  // Create workspace for wavelet transform
  auto const dense_size = dense_space_size(pde_1d);
  fk::vector<P, mem_type::owner, resource::host> workspace(dense_size * 2);
  std::array<fk::vector<P, mem_type::view, resource::host>, 2> tmp_workspace = {
      fk::vector<P, mem_type::view, resource::host>(workspace, 0,
                                                    dense_size - 1),
      fk::vector<P, mem_type::view, resource::host>(workspace, dense_size,
                                                    dense_size * 2 - 1)};

#ifdef ASGARD_USE_CUDA
  fk::vector<P, mem_type::owner, resource::device> initial_condition_d =
      initial_condition.clone_onto_device();
#endif
  for (size_t i = 0; i < pde.moments.size(); ++i)
  {
    pde.moments[i].createMomentReducedMatrix(pde, adaptive_grid.get_table());
#ifdef ASGARD_USE_CUDA
    fk::vector<P, mem_type::owner, resource::device> moment_vec(dense_size);

    fm::sparse_gemv(pde.moments[i].get_moment_matrix_dev(), initial_condition_d,
                    moment_vec);
#else
    fk::vector<P, mem_type::owner, resource::host> moment_vec(dense_size);

    fm::sparse_gemv(pde.moments[i].get_moment_matrix_dev(), initial_condition,
                    moment_vec);
#endif
    pde.moments[i].create_realspace_moment(pde_1d, moment_vec,
                                           adaptive_grid_1d.get_table(),
                                           transformer, tmp_workspace);
  }
}

template<typename P>
void write_output(PDE<P> const &pde, parser const &cli_input,
                  fk::vector<P> const &vec, P const time, int const file_index,
                  int const dof, elements::table const &hash_table,
                  std::string const output_dataset_name = "asgard")
{
  tools::timer.start("write_output");
  std::string const output_file_name =
      output_dataset_name + "_" + std::to_string(file_index) + ".h5";

  // TODO: Rewrite this entirely!
  HighFive::File file(output_file_name, HighFive::File::ReadWrite |
                                            HighFive::File::Create |
                                            HighFive::File::Truncate);

  H5Easy::DumpOptions opts;
  opts.setChunkSize(std::vector<hsize_t>{2});

  // TODO: needs to be checked further based on problem sizes
  HighFive::DataSetCreateProps plist;
  plist.add(HighFive::Chunking(hsize_t{64}));
  plist.add(HighFive::Deflate(9));

  auto const dims = pde.get_dimensions();
  H5Easy::dump(file, "pde", cli_input.get_pde_string());
  H5Easy::dump(file, "degree", dims[0].get_degree());
  H5Easy::dump(file, "dt", pde.get_dt());
  H5Easy::dump(file, "time", time);
  H5Easy::dump(file, "ndims", pde.num_dims);
  H5Easy::dump(file, "max_level", pde.max_level);
  H5Easy::dump(file, "dof", dof);
  H5Easy::dump(file, "cli", cli_input.cli_opts);
  for (size_t dim = 0; dim < dims.size(); ++dim)
  {
    auto const nodes =
        gen_realspace_nodes(dims[dim].get_degree(), dims[dim].get_level(),
                            dims[dim].domain_min, dims[dim].domain_max);
    file.createDataSet<P>(
            "nodes" + std::to_string(dim),
            HighFive::DataSpace({static_cast<size_t>(nodes.size())}))
        .write_raw(nodes.data());
    H5Easy::dump(file, "dim" + std::to_string(dim) + "_level",
                 dims[dim].get_level());
    H5Easy::dump(file, "dim" + std::to_string(dim) + "_min",
                 dims[dim].domain_min);
    H5Easy::dump(file, "dim" + std::to_string(dim) + "_max",
                 dims[dim].domain_max);
  }

  auto &elements = hash_table.get_active_table();
  file.createDataSet<int>(
          "elements",
          HighFive::DataSpace({static_cast<size_t>(elements.size())}), plist)
      .write_raw(elements.data());

  file.createDataSet<P>(
          "soln", HighFive::DataSpace({static_cast<size_t>(vec.size())}), plist)
      .write_raw(vec.data());

  // save E field
  H5Easy::dump(file, "Efield", pde.E_field.to_std(), opts);
  H5Easy::dump(file, "Esource", pde.E_source.to_std(), opts);
  H5Easy::dump(file, "phi", pde.phi.to_std(), opts);

  if (pde.moments.size() > 0)
  {
    // save realspace moments
    H5Easy::dump(file, "nmoments", pde.moments.size());
    for (size_t i = 0; i < pde.moments.size(); ++i)
    {
      file.createDataSet<P>("moment" + std::to_string(i),
                            HighFive::DataSpace({static_cast<size_t>(
                                pde.moments[i].get_realspace_moment().size())}))
          .write_raw(pde.moments[i].get_realspace_moment().data());
    }
  }

  // save gmres error and iteration counts
  for (size_t i = 0; i < pde.gmres_outputs.size(); ++i)
  {
    H5Easy::dump(file, "gmres" + std::to_string(i) + "_err",
                 pde.gmres_outputs[i].error, opts);
    H5Easy::dump(file, "gmres" + std::to_string(i) + "_num_outer",
                 pde.gmres_outputs[i].outer_iter, opts);
    H5Easy::dump(file, "gmres" + std::to_string(i) + "_num_inner",
                 pde.gmres_outputs[i].inner_iter, opts);
  }

  H5Easy::dump(file, "do_adapt", cli_input.do_adapt_levels());
  H5Easy::dump(file, "using_fullgrid", cli_input.using_full_grid());
  H5Easy::dump(file, "starting_levels",
               cli_input.get_starting_levels().to_std());
  if (cli_input.get_active_terms().size() > 0)
  {
    // save list of terms this was run with if --terms option used
    H5Easy::dump(file, "active_terms", cli_input.get_active_terms().to_std());
  }
  if (cli_input.do_adapt_levels())
  {
    H5Easy::dump(file, "adapt_thresh", cli_input.get_adapt_thresh());

    // save the max adaptivity levels for each dimension
    H5Easy::dump(file, "max_adapt_levels",
                 cli_input.get_max_adapt_levels().to_std());

    // if using adaptivity, save some stats about DOF coarsening/refining and
    // GMRES stats for each adapt step
    H5Easy::dump(file, "adapt_initial_dof", pde.adapt_info.initial_dof);
    H5Easy::dump(file, "adapt_coarsen_dof", pde.adapt_info.coarsen_dof);
    H5Easy::dump(file, "adapt_num_refines", pde.adapt_info.refine_dofs.size());
    H5Easy::dump(file, "adapt_refine_dofs", pde.adapt_info.refine_dofs);

    // Transform GMRES stats for each adaptive step into arrays to reduce number
    // of H5 datasets and make it easier to process later.
    // TODO: this needs to be refactored into its own dataset within the H5
    // file.
    size_t num_gmres_calls = pde.gmres_outputs.size();
    size_t num_adapt_steps = pde.adapt_info.gmres_stats.size();
    std::vector<std::vector<P>> step_errors(num_gmres_calls);
    std::vector<std::vector<int>> step_num_inner(num_gmres_calls);
    std::vector<std::vector<int>> step_num_outer(num_gmres_calls);

    for (size_t gmres = 0; gmres < num_gmres_calls; gmres++)
    {
      step_errors[gmres].resize(num_adapt_steps);
      step_num_inner[gmres].resize(num_adapt_steps);
      step_num_outer[gmres].resize(num_adapt_steps);
      // Combine stats for all steps into a single array
      for (size_t step = 0; step < num_adapt_steps; step++)
      {
        step_errors[gmres][step] =
            pde.adapt_info.gmres_stats[step][gmres].error;
        step_num_inner[gmres][step] =
            pde.adapt_info.gmres_stats[step][gmres].inner_iter;
        step_num_outer[gmres][step] =
            pde.adapt_info.gmres_stats[step][gmres].outer_iter;
      }

      std::string const prefix = "adapt_gmres" + std::to_string(gmres);
      H5Easy::dump(file, prefix + "_err", step_errors[gmres]);
      H5Easy::dump(file, prefix + "_num_inner", step_num_inner[gmres]);
      H5Easy::dump(file, prefix + "_num_outer", step_num_outer[gmres]);
    }
  }

  P gmres_tol = cli_input.get_gmres_tolerance();
  if (gmres_tol == parser::NO_USER_VALUE_FP)
  {
    gmres_tol = std::is_same_v<float, P> ? 1e-6 : 1e-12;
  }
  H5Easy::dump(file, "gmres_tolerance", gmres_tol);

  int gmres_restart = cli_input.get_gmres_inner_iterations();
  if (gmres_restart == parser::NO_USER_VALUE)
  {
    // calculate default based on size of solution vector
    gmres_restart = solver::default_gmres_restarts<P>(vec.size());
  }
  H5Easy::dump(file, "gmres_restart", gmres_restart);

  int gmres_max_iter = cli_input.get_gmres_outer_iterations();
  if (gmres_max_iter == parser::NO_USER_VALUE)
  {
    // default value is to use size of solution vector
    gmres_max_iter = vec.size();
  }
  H5Easy::dump(file, "gmres_max_iter", gmres_max_iter);

  // save some basic build info
  H5Easy::dump(file, "GIT_BRANCH", std::string(GIT_BRANCH));
  H5Easy::dump(file, "GIT_COMMIT_HASH", std::string(GIT_COMMIT_HASH));
  H5Easy::dump(file, "GIT_COMMIT_SUMMARY", std::string(GIT_COMMIT_SUMMARY));
  H5Easy::dump(file, "BUILD_TIME", std::string(BUILD_TIME));
#if defined(ASGARD_USE_CUDA)
  bool constexpr using_gpu = true;
#else
  bool constexpr using_gpu = false;
#endif
  H5Easy::dump(file, "USING_GPU", using_gpu);

  // save performance timers to the /timings/ group
  auto timing_stat_type = create_timing_stats<double>();
  timing_stat_type.commit(file, "timing_stats");

  std::map<std::string, tools::timing_stats> timings;
  tools::timer.get_timing_stats(timings);
  auto timing_group = file.createGroup("timings");
  for (auto [id, times] : timings)
  {
    timing_group
        .createDataSet(
            id,
            HighFive::DataSpace(
                HighFive::DataSpace::DataspaceType::dataspace_scalar),
            timing_stat_type)
        .write(times);
  }

  file.flush();
  tools::timer.stop("write_output");
}

template<typename P>
void write_gmres_temp(PDE<P> const &pde, parser const &cli_input,
                      kronmult_matrix<P> const &mat, fk::vector<P> &x,
                      fk::vector<P> const &b,
                      preconditioner::preconditioner<P> *precond, P const dt,
                      P const time, int const file_index, int const dof,
                      elements::table const &hash_table,
                      std::string const output_dataset_name = "gmres_data")
{
  tools::timer.start("write_output");
  std::string const output_file_name =
      output_dataset_name + "_" + std::to_string(file_index) + ".h5";

  std::cout << " starting to write GMRES data.." << std::endl;

  // Construct Matrix A by calling kronmult with identity to back-out the A
  // matrix one column at a time
  fk::matrix<P> A(dof, dof);
  fk::vector<P> kron_x(dof);
  fk::vector<P> kron_y(dof);
  for (int col = 0; col < dof; col++)
  {
    // set current row to identity
    kron_x(col) = 1.0;
    if (col > 0)
    {
      // flip prev row value back to 0.
      kron_x(col - 1) = 0.0;
    }

    mat.apply(P{1.0}, kron_x.data(), P{0.0}, kron_y.data());
    A.update_col(col, kron_y);
  }

  // Calculate (I - dt*A)
  fm::scal(-dt, A);
  for (int col = 0; col < dof; col++)
  {
    A(col, col) += 1.0;
  }

  // fk::matrix<P> precond_M = precond->get_matrix();

  std::cout << " WRITING OUTPUT FILE '" << output_file_name << "'" << std::endl;

  // TODO: Rewrite this entirely!
  HighFive::File file(output_file_name, HighFive::File::ReadWrite |
                                            HighFive::File::Create |
                                            HighFive::File::Truncate);

  H5Easy::DumpOptions opts;
  opts.setChunkSize(std::vector<hsize_t>{2});

  HighFive::DataSetCreateProps plist;
  plist.add(HighFive::Chunking(hsize_t{64}));
  plist.add(HighFive::Deflate(9));

  HighFive::DataSetCreateProps plist_2d;
  plist_2d.add(HighFive::Chunking({hsize_t{8}, hsize_t{8}}));
  plist_2d.add(HighFive::Deflate(9));

  H5Easy::dump(file, "pde", cli_input.get_pde_string());
  H5Easy::dump(file, "degree", cli_input.get_degree());
  H5Easy::dump(file, "dt", cli_input.get_dt());
  H5Easy::dump(file, "time", time);
  H5Easy::dump(file, "dof", dof);
  H5Easy::dump(file, "max_level", pde.max_level);

  // initial guess (x) of GMRES
  file.createDataSet<P>(
          "x", HighFive::DataSpace({static_cast<size_t>(x.size())}), plist)
      .write_raw(x.data());
  // RHS (b) of GMRES
  file.createDataSet<P>(
          "b", HighFive::DataSpace({static_cast<size_t>(b.size())}), plist)
      .write_raw(b.data());

  // (I - dt*A)
  file.createDataSet<P>("A",
                        HighFive::DataSpace({static_cast<size_t>(dof),
                                             static_cast<size_t>(dof)}),
                        plist_2d)
      .write_raw(A.data());

  // Get preconditioner M matrix
  // If block jacobi, M matrix is stored as a vector of (degree^ndims,
  // degree^ndims) matrix blocks
  auto precond_jacobi =
      dynamic_cast<preconditioner::block_jacobi_preconditioner<P> *>(precond);
  if (precond_jacobi)
  {
    // get preconditioner matrix M
    std::vector<fk::matrix<P>> &blocks = precond_jacobi->precond_blks;
    size_t nblocks                     = blocks.size();
    int block_size                     = blocks[0].nrows();

    H5Easy::dump(file, "M_nblocks", nblocks);
    H5Easy::dump(file, "M_block_size", block_size);
    H5Easy::dump(file, "M_factored", precond_jacobi->factored());

    // TODO: this is a hackish way to implement all of this.. can be done better
    auto row_dset = file.createDataSet<int>(
        "M_rows",
        HighFive::DataSpace(
            {static_cast<size_t>(nblocks * block_size * block_size)}),
        plist);
    auto col_dset = file.createDataSet<int>(
        "M_cols",
        HighFive::DataSpace(
            {static_cast<size_t>(nblocks * block_size * block_size)}),
        plist);
    auto val_dset = file.createDataSet<P>(
        "M_vals",
        HighFive::DataSpace(
            {static_cast<size_t>(nblocks * block_size * block_size)}),
        plist);

    fk::vector<int> rows(block_size * block_size);
    fk::vector<int> cols(block_size * block_size);

    // col major to match fk::matrix layout
    for (int col = 0; col < block_size; col++)
    {
      for (int row = 0; row < block_size; row++)
      {
        rows[col * block_size + row] = row;
        cols[col * block_size + row] = col;
      }
    }

    // for each block, write the values based on the (row,col) tuples. Add
    // offset to block index vectors for next block
    for (size_t blk = 0; blk < nblocks; blk++)
    {
      auto &block = blocks[blk];

      size_t block_offset = blk * block_size;

      row_dset
          .select({block_offset},
                  {static_cast<size_t>(block_size * block_size)})
          .write_raw(rows.data());
      col_dset
          .select({block_offset},
                  {static_cast<size_t>(block_size * block_size)})
          .write_raw(cols.data());
      val_dset
          .select({block_offset},
                  {static_cast<size_t>(block_size * block_size)})
          .write_raw(block.data());

      // shift col, row indices by block offset
      // TODO: no element wise operator defined?
      // rows = rows + block_size;
      // cols = cols + block_size;
      for (int col = 0; col < block_size; col++)
      {
        for (int row = 0; row < block_size; row++)
        {
          rows[col * block_size + row] = row + block_offset;
          cols[col * block_size + row] = col + block_offset;
        }
      }
    }
  }
  // else
  //{
  //  others store as dense M.
  fk::matrix<P> precond_M = precond->get_matrix();
  file.createDataSet<P>("M",
                        HighFive::DataSpace({static_cast<size_t>(dof),
                                             static_cast<size_t>(dof)}),
                        plist_2d)
      .write_raw(precond_M.data());
  //}

  auto &elements = hash_table.get_active_table();
  file.createDataSet<int>(
          "elements",
          HighFive::DataSpace({static_cast<size_t>(elements.size())}), plist)
      .write_raw(elements.data());

  // save the term coefficient matrices
  auto coeff_group = file.createGroup("coeffs");
  for (int term = 0; term < pde.num_terms; term++)
  {
    auto term_group = coeff_group.createGroup("term" + std::to_string(term));

    for (int dim = 0; dim < pde.num_dims; dim++)
    {
      term_group
          .createDataSet<P>(
              "dim" + std::to_string(dim),
              HighFive::DataSpace(
                  {static_cast<size_t>(pde.get_coefficients(term, dim).nrows()),
                   static_cast<size_t>(
                       pde.get_coefficients(term, dim).ncols())}),
              plist_2d)
          .write_raw(pde.get_coefficients(term, dim).data());
    }
  }

  HighFive::DataSetCreateProps plist_pterm;
  plist_pterm.add(HighFive::Chunking({hsize_t{2}, hsize_t{2}}));
  plist_pterm.add(HighFive::Deflate(9));

  // save the partial term coefficient matrices
  auto pterm_coeff_group = file.createGroup("pterm_coeffs");
  auto term_set          = pde.get_terms();
  for (int term = 0; term < pde.num_terms; term++)
  {
    auto term_group =
        pterm_coeff_group.createGroup("term" + std::to_string(term));

    for (int dim = 0; dim < pde.num_dims; dim++)
    {
      int const level = pde.get_dimensions()[dim].get_level();

      auto dim_group = term_group.createGroup("dim" + std::to_string(dim));

      auto &pterms = term_set[term][dim].get_partial_terms();
      for (size_t pterm = 0; pterm < pterms.size(); pterm++)
      {
        dim_group
            .createDataSet<P>(
                "pterm" + std::to_string(pterm),
                HighFive::DataSpace(
                    {static_cast<size_t>(
                         pterms[pterm].get_coefficients(level).nrows()),
                     static_cast<size_t>(
                         pterms[pterm].get_coefficients(level).ncols())}),
                plist_pterm)
            .write_raw(pterms[pterm].get_coefficients(level).data());
      }
    }
  }

  file.flush();
  tools::timer.stop("write_output");

  std::cout << " DONE FILE WRITE" << std::endl;
}

template<typename P>
void read_restart_metadata(parser &user_vals, std::string const &restart_file)
{
  std::cout << "--- Reading metadata from restart file '" << restart_file
            << "' ---\n";

  if (!std::filesystem::exists(restart_file))
  {
    throw std::runtime_error("Could not open restart file: " + restart_file);
  }

  HighFive::File file(restart_file, HighFive::File::ReadOnly);

  std::string const pde_string =
      H5Easy::load<std::string>(file, std::string("pde"));
  int const degree = H5Easy::load<int>(file, std::string("degree"));
  P const dt       = H5Easy::load<P>(file, std::string("dt"));
  P const time     = H5Easy::load<P>(file, std::string("time"));

  int const ndims = H5Easy::load<int>(file, std::string("ndims"));
  std::string levels;
  for (int dim = 0; dim < ndims; ++dim)
  {
    levels += std::to_string(H5Easy::load<int>(
        file, std::string("dim" + std::to_string(dim) + "_level")));
    levels += " ";
  }
  int const max_level = H5Easy::load<int>(file, std::string("max_level"));
  int const dof       = H5Easy::load<int>(file, std::string("dof"));
  bool const use_fg   = H5Easy::load<bool>(file, std::string("using_fullgrid"));

  // if the user requested a FG but the restart is a SG, let the user know the
  // FG option is ignored
  if (user_vals.using_full_grid() && !use_fg)
  {
    std::cerr << "WARN: Requested FG but restart file contains a SG. The FG "
                 "option will be ignored."
              << std::endl;
    // ensure FG is disabled in CLI since we always use the grid from the
    // restart file
    parser_mod::set(user_vals, parser_mod::use_full_grid, false);
  }

  // TODO: this will be used for validation in the future
  ignore(dof);

  parser_mod::set(user_vals, parser_mod::pde_str, pde_string);
  parser_mod::set(user_vals, parser_mod::degree, degree);
  parser_mod::set(user_vals, parser_mod::dt, dt);
  parser_mod::set(user_vals, parser_mod::starting_levels_str, levels);
  parser_mod::set(user_vals, parser_mod::max_level, max_level);

  // check if the restart file was run with adaptivity
  bool const restart_used_adapt =
      H5Easy::load<bool>(file, std::string("do_adapt"));

  // restore the max adaptivity levels if set in the file
  std::string max_adapt_str;
  if (restart_used_adapt)
  {
    std::vector<int> max_adapt_levels =
        H5Easy::load<std::vector<int>>(file, std::string("max_adapt_levels"));
    assert(max_adapt_levels.size() == static_cast<size_t>(ndims));

    parser_mod::set(user_vals, parser_mod::max_adapt_level,
                    fk::vector<int>(max_adapt_levels));

    for (size_t lev = 0; lev < max_adapt_levels.size(); lev++)
    {
      max_adapt_str += std::to_string(max_adapt_levels[lev]);
      if (lev < max_adapt_levels.size() - 1)
        max_adapt_str += " ";
    }
  }

  std::cout << "  - PDE: " << pde_string << ", ndims = " << ndims
            << ", degree = " << degree << '\n';
  std::cout << "  - time = " << time << ", dt = " << dt << '\n';
  std::cout << "  - file used adaptivity = "
            << (restart_used_adapt ? "true" : "false") << '\n';
  if (restart_used_adapt)
  {
    std::cout << "    - max_level = " << max_level
              << ", max_adapt_levels = " << max_adapt_str << '\n';
  }
  std::cout << "---------------" << '\n';
}

template<typename P>
struct restart_data
{
  fk::vector<P> solution;
  P const time;
  int step_index;
  std::vector<int64_t> active_table;
  int max_level;
};

template<typename P>
restart_data<P> read_output(PDE<P> &pde, elements::table const &hash_table,
                            std::string const &restart_file)
{
  tools::timer.start("read_output");

  std::cout << "--- Loading from restart file '" << restart_file << "' ---\n";

  if (!std::filesystem::exists(restart_file))
  {
    throw std::runtime_error("Could not open restart file: " + restart_file);
  }

  HighFive::File file(restart_file, HighFive::File::ReadOnly);

  int const max_level = H5Easy::load<int>(file, std::string("max_level"));
  P const dt          = H5Easy::load<P>(file, std::string("dt"));
  P const time        = H5Easy::load<P>(file, std::string("time"));

  std::vector<int64_t> active_table =
      H5Easy::load<std::vector<int64_t>>(file, std::string("elements"));

  fk::vector<P> solution =
      fk::vector<P>(H5Easy::load<std::vector<P>>(file, std::string("soln")));

  // load E field
  pde.E_field = std::move(
      fk::vector<P>(H5Easy::load<std::vector<P>>(file, std::string("Efield"))));

  for (int dim = 0; dim < pde.num_dims; ++dim)
  {
    int level = H5Easy::load<int>(
        file, std::string("dim" + std::to_string(dim) + "_level"));
    pde.get_dimensions()[dim].set_level(level);
    pde.update_dimension(dim, level);
    pde.rechain_dimension(dim);
  }

  // load realspace moments
  int const num_moments = H5Easy::load<int>(file, std::string("nmoments"));
  expect(static_cast<int>(pde.moments.size()) == num_moments);
  for (int i = 0; i < num_moments; ++i)
  {
    pde.moments[i].createMomentReducedMatrix(pde, hash_table);
    pde.moments[i].set_realspace_moment(
        fk::vector<P>(H5Easy::load<std::vector<P>>(
            file, std::string("moment" + std::to_string(i)))));
  }

  int step_index = static_cast<int>(time / dt);

  std::cout << " Setting time step index as = " << step_index << "\n";

  tools::timer.stop("read_output");

  return restart_data<P>{solution, time, step_index, active_table, max_level};
}

} // namespace asgard
