#include "matlab_plot.hpp"

#include "asgard_dimension.hpp"
#include "quadrature.hpp"
#include <cmath>
#include <type_traits>
namespace asgard::ml
{
void matlab_plot::connect_async()
{
  if (!is_open())
  {
    matlab::engine::FutureResult<std::unique_ptr<matlab::engine::MATLABEngine>>
        ml_connection;
    ml_connection = matlab::engine::connectMATLABAsync();

    this->matlab_inst_ = ml_connection.get();
  }
}

void matlab_plot::connect(std::string const &name)
{
  // Connects to a shared MATLAB session with given name, or first available
  // if no name is given. Creates a session if none exist.
  if (!is_open())
  {
    if (name.empty() || name.compare("none") == 0)
    {
      // add message since this could take awhile if a new MATLAB session
      // needs to be started
      std::cout << "  connecting with MATLAB...\n";
      auto const &sessions = matlab::engine::findMATLAB();
      if (sessions.size() == 0)
      {
        // if there were no shared sessions, set flag to wait for any plots at
        // the end
        started_ = true;
      }
      this->matlab_inst_ = matlab::engine::connectMATLAB();
    }
    else
    {
      bool get_name            = false;
      std::string session_name = name;
      std::cout << "  connecting with MATLAB session '" << session_name
                << "'\n";
      // Check if the named session exists
      if (!find_session(name, false, session_name))
      {
        std::cerr << "  Specified MATLAB session '" << name
                  << "' does not exist" << '\n';
        // Try to find next session instead of exiting
        get_name = true;
      }
      // If not found, try to connect to first running session
      if (get_name)
      {
        bool found = find_session(name, true, session_name);
        if (found)
        {
          std::cout << "  Found next running MATLAB session '" << session_name
                    << "'\n";
        }
        else
        {
          std::cerr << "  Could not find another running MATLAB session.. "
                       "creating a new one\n";
          // Make a new shared session in this case
          this->matlab_inst_ = matlab::engine::connectMATLAB();
          started_           = true;
          return;
        }
      }
      // convert to utf16 for matlab
      ml_string const name_conv =
          matlab::engine::convertUTF8StringToUTF16String(session_name);
      this->matlab_inst_ = matlab::engine::connectMATLAB(name_conv);
    }
  }
}

void matlab_plot::start(std::vector<ml_string> const &args)
{
  if (!is_open())
  {
    this->matlab_inst_ = matlab::engine::startMATLAB(args);
  }
}

void matlab_plot::close()
{
  if (!is_open())
  {
    matlab::engine::terminateEngineClient();
  }
}

// Simple wrapper for the subplot command (doesn't handle pos or ax
// parameters, assumes position is scalar for now)
void matlab_plot::subplot(int const &rows, int const &cols, int const &pos)
{
  expect(is_open());

  // Temporary workaround.. calling subplot using feval is giving a type error
  // in subplot.m
  std::stringstream stmt;
  stmt << "subplot(" << rows << "," << cols << "," << pos << ")";
  matlab::engine::String ml_stmt =
      matlab::engine::convertUTF8StringToUTF16String(stmt.str());
  matlab_inst_->eval(ml_stmt);
}

// Simple wrapper for the matlab plot command
void matlab_plot::plot()
{
  expect(is_open());

  matlab::data::TypedArray<matlab::data::Object> plt =
      matlab_inst_->feval(u"plot", m_args_);
  m_args_.clear();
  plots_.push_back(
      std::unique_ptr<matlab::data::TypedArray<matlab::data::Object>>(&plt));
}

void matlab_plot::call(std::string const func)
{
  expect(is_open());

  matlab_inst_->feval(func.c_str(), 0, m_args_);
  m_args_.clear();
}

std::vector<matlab::data::Array>
matlab_plot::call(std::string const func, int const n_returns)
{
  expect(is_open());

  std::vector<matlab::data::Array> plt =
      matlab_inst_->feval(func.c_str(), n_returns, m_args_);
  m_args_.clear();
  return plt;
}

void matlab_plot::call_async(std::string const func)
{
  expect(is_open());

  matlab_inst_->fevalAsync(func.c_str(), 0, m_args_);
  m_args_.clear();
}

std::string matlab_plot::eval(ml_string const &stmt, std::string &err)
{
  expect(is_open());

  auto output    = std::make_shared<ml_stringbuf>();
  auto err_utf16 = std::make_shared<ml_stringbuf>();

  matlab_inst_->eval(stmt, output, err_utf16);

  std::string const output_utf8 =
      matlab::engine::convertUTF16StringToUTF8String(output->str());
  err = matlab::engine::convertUTF16StringToUTF8String(err_utf16->str());

  return output_utf8;
}

void matlab_plot::set_var(std::string const &name,
                          matlab::data::Array const &var,
                          ml_wksp_type const type)
{
  expect(is_open());
  matlab_inst_->setVariable(name, var, type);
}

// pushes a vector to a matlab workspace with the given name
template<typename T>
void matlab_plot::push(std::string const &name, fk::vector<T> const &data,
                       ml_wksp_type const type)
{
  // create a matlab array object
  auto ml_array = create_array({static_cast<size_t>(data.size()), 1}, data);

  set_var(name, ml_array, type);
}

// pushes a matrix to a matlab workspace with the given name
template<typename T>
void matlab_plot::push(std::string const &name, fk::matrix<T> const &data,
                       ml_wksp_type const type)
{
  // create a matlab array object
  auto const &v = fk::vector<T>(data);
  auto ml_array = create_array(
      {static_cast<size_t>(data.ncols()), static_cast<size_t>(data.nrows())},
      v);

  set_var(name, ml_array, type);
}

template<typename T>
fk::vector<T> matlab_plot::generate_nodes(int const degree, int const level,
                                          T const min, T const max) const
{
  // Trimmed version of matrix_plot_D.m to get only the nodes
  int const n        = pow(2, level);
  int const mat_dims = degree * n;
  T const h          = (max - min) / n;

  // TODO: fully implement the output_grid options from matlab (this is just
  // the 'else' case)
  auto const lgwt  = legendre_weights(degree, -1.0, 1.0, true);
  auto const roots = lgwt[0];

  unsigned int const dof = roots.size();

  fk::vector<T> nodes(mat_dims);

  for (int i = 0; i < n; i++)
  {
    auto p_val = legendre(roots, degree, legendre_normalization::lin);

    p_val[0] = p_val[0] * sqrt(1.0 / h);

    std::vector<T> xi(dof);
    for (std::size_t j = 0; j < dof; j++)
    {
      xi[j] = (0.5 * (roots(j) + 1.0) + i) * h + min;
    }

    std::vector<int> Iu(degree);
    for (int j = 0, je = degree - 1; j < je; j++)
    {
      Iu[j] = dof * i + j + 1;
    }
    Iu[degree - 1] = dof * (i + 1);

    for (std::size_t j = 0; j < dof; j++)
    {
      nodes(Iu[j] - 1) = xi[j];
    }
  }

  return nodes;
}

template<typename P>
fk::vector<P> matlab_plot::gen_elem_coords(PDE<P> const &pde,
                                           elements::table const &table) const
{
  return gen_elem_coords(pde.get_dimensions(), table);
}

template<typename P>
fk::vector<P>
matlab_plot::gen_elem_coords(std::vector<dimension<P>> const &dims,
                             elements::table const &table) const
{
  int const ndims = dims.size();

  fk::vector<P> center_coords(ndims * table.size());

  // Iterate over dimensions first since matlab needs col-major order
  for (int d = 0; d < ndims; d++)
  {
    P const domain_range = dims[d].domain_max - dims[d].domain_min;
    for (int i = 0; i < table.size(); i++)
    {
      fk::vector<int> const &coords = table.get_coords(i);

      int const lev = coords(d);
      int const pos = coords(ndims + d);

      expect(lev >= 0);
      expect(pos >= 0);

      P x0;
      if (lev > 1)
      {
        P const s = pow(2, lev - 1) - 1.0;
        P const h = 1.0 / (pow(2, lev - 1));
        P const w = 1.0 - h;
        P const o = 0.5 * h;

        x0 = pos / s * w + o;
      }
      else
      {
        x0 = pos + 0.5;
      }

      P const x = x0 * domain_range + dims[d].domain_min;

      center_coords(d * table.size() + i) = x;
    }
  }
  return center_coords;
}

template<typename P>
fk::vector<P>
matlab_plot::gen_elem_coords(std::vector<dimension_description<P>> const &dims,
                             elements::table const &table) const
{
  int const ndims = dims.size();

  fk::vector<P> center_coords(ndims * table.size());

  // Iterate over dimensions first since matlab needs col-major order
  for (int d = 0; d < ndims; d++)
  {
    P const domain_range = dims[d].d_max - dims[d].d_min;
    for (int i = 0; i < table.size(); i++)
    {
      fk::vector<int> const &coords = table.get_coords(i);

      int const lev = coords(d);
      int const pos = coords(ndims + d);

      expect(lev >= 0);
      expect(pos >= 0);

      P x0;
      if (lev > 1)
      {
        P const s = pow(2, lev - 1) - 1.0;
        P const h = 1.0 / (pow(2, lev - 1));
        P const w = 1.0 - h;
        P const o = 0.5 * h;

        x0 = pos / s * w + o;
      }
      else
      {
        x0 = pos + 0.5;
      }

      P const x = x0 * domain_range + dims[d].d_min;

      center_coords(d * table.size() + i) = x;
    }
  }
  return center_coords;
}

template<typename P>
void matlab_plot::init_plotting(PDE<P> const &pde, elements::table const &table)
{
  init_plotting(pde.get_dimensions(), table);
}

template<typename P>
void matlab_plot::init_plotting(std::vector<dimension<P>> const &dims,
                                elements::table const &table)
{
  // Generates cell array of nodes and element coordinates needed for plotting
  sol_sizes_ = get_soln_sizes(dims);

  nodes_.clear();
  
  int d = 0;
  for (auto const &dim : dims)
  {
    nodes_.push_back(create_array(generate_nodes(dim)));
    this->set_var(std::string("nodes_dim" + std::to_string(d)), nodes_.back());
    d += 1;
  }

  auto const &elem_coords = gen_elem_coords(dims, table);
  elem_coords_ = create_array({static_cast<size_t>(table.size()), dims.size()},
                              elem_coords);

  this->set_var("elem_coords", elem_coords_);
}

template<typename P>
void matlab_plot::init_plotting(
    std::vector<dimension_description<P>> const &dims,
    elements::table const &table)
{
  // Generates cell array of nodes and element coordinates needed for plotting
  sol_sizes_ = get_soln_sizes(dims);

  nodes_.clear();

  int d = 0;
  for (auto const &dim : dims)
  {
    nodes_.push_back(create_array(generate_nodes(dim)));
    this->set_var(std::string("nodes_dim" + std::to_string(d)), nodes_.back());
    d += 1;
  }

  auto const &elem_coords = gen_elem_coords(dims, table);
  elem_coords_ = create_array({static_cast<size_t>(table.size()), dims.size()},
                              elem_coords);

  this->set_var("elem_coords", elem_coords_);
}

template<typename P>
void matlab_plot::plot_fval(PDE<P> const &pde, elements::table const &table,
                            fk::vector<P> const &f_val,
                            fk::vector<P> const &analytic_soln)
{
  plot_fval(pde.get_dimensions(), table, f_val, analytic_soln);
}

template<typename P>
void matlab_plot::plot_fval(std::vector<dimension<P>> const &dims,
                            elements::table const &table,
                            fk::vector<P> const &f_val,
                            fk::vector<P> const &analytic_soln)
{
  expect(sol_sizes_.size() > 0);

  size_t const ndims = static_cast<size_t>(dims.size());

  if (table.size() * ndims != elem_coords_.getNumberOfElements())
  {
    // Regenerate the element coordinates and nodes if the grid was adapted
    init_plotting(dims, table);
  }

  m_args_.clear();
  add_param(ndims);
  add_param({1, static_cast<size_t>(analytic_soln.size())}, analytic_soln);
  add_param({1, ndims}, nodes_);
  add_param(sol_sizes_, f_val, analytic_soln);
  push_param(elem_coords_);

  call("plot_fval");
}

template<typename P>
void matlab_plot::plot_fval(std::vector<dimension_description<P>> const &dims,
                            elements::table const &table,
                            fk::vector<P> const &f_val,
                            fk::vector<P> const &analytic_soln)
{
  expect(sol_sizes_.size() > 0);

  size_t const ndims = static_cast<size_t>(dims.size());

  if (table.size() * ndims != elem_coords_.getNumberOfElements())
  {
    // Regenerate the element coordinates and nodes if the grid was adapted
    init_plotting(dims, table);
  }

  m_args_.clear();
  add_param(ndims);
  add_param({1, static_cast<size_t>(analytic_soln.size())}, analytic_soln);
  add_param({1, ndims}, nodes_);
  add_param(sol_sizes_, f_val, analytic_soln);
  push_param(elem_coords_);

  call("plot_fval");
}

// copies data about a PDE object to a matlab workspace
template<typename P>
void matlab_plot::copy_pde(PDE<P> const &pde, std::string const name)
{
  expect(is_open());

  std::vector<std::string> pde_names = {"solvePoisson",
                                        "hasAnalyticSoln",
                                        "coefficients",
                                        "dims",
                                        "terms",
                                        "sources",
                                        "dt"};

  matlab::data::StructArray ml_pde = factory_.createStructArray({1}, pde_names);

  ml_pde[0]["solvePoisson"] = factory_.createScalar<bool>(pde.do_poisson_solve);
  ml_pde[0]["hasAnalyticSoln"] =
      factory_.createScalar<bool>(pde.has_analytic_soln);

  // create the dimensions cell
  matlab::data::CellArray ml_dims =
      factory_.createCellArray({1, static_cast<size_t>(pde.num_dims)});
  for (auto i = 0; i < pde.num_dims; ++i)
  {
    ml_dims[0][i] = make_dimension(pde.get_dimensions()[i]);
  }
  ml_pde[0]["dims"] = ml_dims;

  // create the terms cell
  matlab::data::CellArray ml_terms = factory_.createCellArray(
      {static_cast<size_t>(pde.num_dims), static_cast<size_t>(pde.num_terms)});
  for (auto i = 0; i < pde.num_dims; ++i)
  {
    for (auto j = 0; j < pde.num_terms; ++j)
    {
      ml_terms[i][j] = make_term(pde.get_terms()[j][i], pde.max_level);
    }
  }
  ml_pde[0]["terms"] = std::move(ml_terms);

  ml_pde[0]["dt"] = factory_.createScalar<P>(pde.get_dt());

  // push the pde struct to the matlab workspace
  matlab_inst_->setVariable(name, ml_pde);
}

template<typename P>
matlab::data::StructArray
matlab_plot::make_term(term<P> const &term, int const max_lev)
{
  std::vector<std::string> names    = {"time_dependent", "name", "coefficients",
                                    "pterms"};
  matlab::data::StructArray ml_term = factory_.createStructArray({1}, names);

  ml_term[0]["time_dependent"] =
      factory_.createScalar<bool>(term.time_dependent);
  ml_term[0]["name"]         = factory_.createCharArray(term.name);
  auto const &coeffs         = term.get_coefficients().clone_onto_host();
  ml_term[0]["coefficients"] = matrix_to_array(coeffs);

  auto const &pterms = term.get_partial_terms();
  matlab::data::CellArray ml_pterms =
      factory_.createCellArray({1, static_cast<size_t>(pterms.size())});
  for (auto i = 0; i < static_cast<int>(pterms.size()); ++i)
  {
    ml_pterms[0][i] = make_partial_term(pterms[i], max_lev);
  }
  ml_term[0]["pterms"] = ml_pterms;

  return ml_term;
}

template<typename P>
matlab::data::StructArray
matlab_plot::make_partial_term(partial_term<P> const &pterm, int const max_lev)
{
  std::vector<std::string> names = {"LF",   "BCL",  "BCR",         "IBCL",
                                    "IBCR", "type", "LHS_mass_mat"};
  for (int i = 1; i <= max_lev; ++i)
  {
    names.push_back("mat" + std::to_string(i));
  }
  matlab::data::StructArray ml_pterm = factory_.createStructArray({1}, names);

  auto const get_bc_str = [](boundary_condition const &type) -> std::string {
    switch (type)
    {
    case boundary_condition::periodic:
      return "P";
    case boundary_condition::dirichlet:
      return "D";
    case boundary_condition::neumann:
      return "N";
    default:
      return "";
    }
  };

  auto const get_coeff_str = [](coefficient_type const &type) -> std::string {
    std::vector<std::string> types = {"grad", "mass", "div", "diff"};
    return types[static_cast<int>(type)];
  };

  ml_pterm[0]["LF"]   = factory_.createScalar<P>(pterm.get_flux_scale());
  ml_pterm[0]["BCL"]  = factory_.createCharArray(get_bc_str(pterm.left));
  ml_pterm[0]["BCR"]  = factory_.createCharArray(get_bc_str(pterm.right));
  ml_pterm[0]["IBCL"] = factory_.createCharArray(get_bc_str(pterm.ileft));
  ml_pterm[0]["IBCR"] = factory_.createCharArray(get_bc_str(pterm.iright));
  ml_pterm[0]["type"] =
      factory_.createCharArray(get_coeff_str(pterm.coeff_type));

  // get the pterm coefficients that are stored for each level
  for (int i = 1; i <= max_lev; ++i)
  {
    ml_pterm[0]["mat" + std::to_string(i)] =
        matrix_to_array(pterm.get_coefficients(i));
  }
  ml_pterm[0]["LHS_mass_mat"] = matrix_to_array(pterm.get_lhs_mass());

  return ml_pterm;
}

template<typename P>
matlab::data::StructArray matlab_plot::make_dimension(dimension<P> const &dim)
{
  std::vector<std::string> names = {
      "name",    "min", "max", "lev", "init_cond_fn", "volume_jacobian_dV",
      "mass_mat"};
  matlab::data::StructArray ml_dim = factory_.createStructArray({1}, names);

  ml_dim[0]["name"] = factory_.createCharArray(dim.name);
  ml_dim[0]["min"]  = factory_.createScalar<P>(dim.domain_min);
  ml_dim[0]["max"]  = factory_.createScalar<P>(dim.domain_max);
  ml_dim[0]["lev"]  = factory_.createScalar<int>(dim.get_level());
  // TODO: find a better way to represent these functions?
  ml_dim[0]["init_cond_fn"] =
      factory_.createCharArray(dim.initial_condition[0].target_type().name());
  ml_dim[0]["volume_jacobian_dV"] =
      factory_.createCharArray(dim.volume_jacobian_dV.target_type().name());

  ml_dim[0]["mass_mat"] = matrix_to_array(dim.get_mass_matrix());

  return ml_dim;
}

template<typename P>
fk::vector<P> matlab_plot::col_slice(fk::vector<P> const &vec, int const n,
                                     int const col) const
{
  fk::vector<P> slice(n);
  for (int i = 0; i < n; i++)
  {
    slice(i) = vec(i * n + col);
  }
  return slice;
}

bool matlab_plot::find_session(std::string const &name, bool const find_name,
                               std::string &session_name) const
{
  // Check if there is a running matlab session with name
  std::vector<ml_string> const &sessions = matlab::engine::findMATLAB();
  if (sessions.size() == 0 && !find_name)
  {
    std::cerr
        << "  Found no running matlab sessions! Ensure the session has been "
           "shared by running `matlab.engine.shareEngine` within Matlab and "
           "check that asgard was linked with the correct version of Matlab."
        << '\n';
    return false;
  }

  for (auto const &session : sessions)
  {
    // Convert from UTF16 to UTF8
    std::string const conv =
        matlab::engine::convertUTF16StringToUTF8String(session);
    if (find_name)
    {
      // Return first found session name
      session_name = conv;
      return true;
    }
    if (name.compare(conv) == 0)
    {
      return true;
    }
  }
  return false;
}

template<typename P>
inline std::vector<size_t> matlab_plot::get_soln_sizes(
    std::vector<dimension_description<P>> const &dims) const
{
  // Returns a vector of the solution size for each dimension
  std::vector<size_t> sizes(dims.size());
  for (size_t i = 0; i < dims.size(); i++)
  {
    sizes[i] = dims[i].degree * std::pow(2, dims[i].level);
  }
  return sizes;
}

template<typename P>
inline std::vector<size_t>
matlab_plot::get_soln_sizes(std::vector<dimension<P>> const &dims) const
{
  // Returns a vector of the solution size for each dimension
  std::vector<size_t> sizes(dims.size());
  for (size_t i = 0; i < dims.size(); i++)
  {
    sizes[i] = dims[i].get_degree() * std::pow(2, dims[i].get_level());
  }
  return sizes;
}

template<typename P>
inline std::vector<size_t> matlab_plot::get_soln_sizes(PDE<P> const &pde) const
{
  return get_soln_sizes(pde.get_dimensions());
}

template<typename P>
inline int matlab_plot::get_soln_size(PDE<P> const &pde, int const dim) const
{
  // Gets the solution size for a given dimension (see dense_space_size() in
  // transformations)
  return pde.get_dimensions()[dim].get_degree() *
         std::pow(2, pde.get_dimensions()[pde].get_level());
}

/* explicit instantiations */
template void matlab_plot::push(std::string const &name,
                                fk::vector<float> const &data,
                                ml_wksp_type const type);

template void matlab_plot::push(std::string const &name,
                                fk::vector<double> const &data,
                                ml_wksp_type const type);

template void matlab_plot::push(std::string const &name,
                                fk::matrix<float> const &data,
                                ml_wksp_type const type);

template void matlab_plot::push(std::string const &name,
                                fk::matrix<double> const &data,
                                ml_wksp_type const type);

template fk::vector<double>
matlab_plot::generate_nodes(int const degree, int const level, double const min,
                            double const max) const;

template fk::vector<float>
matlab_plot::generate_nodes(int const degree, int const level, float const min,
                            float const max) const;

template fk::vector<double>
matlab_plot::gen_elem_coords(PDE<double> const &pde,
                             elements::table const &table) const;

template fk::vector<float>
matlab_plot::gen_elem_coords(PDE<float> const &pde,
                             elements::table const &table) const;

template fk::vector<double>
matlab_plot::gen_elem_coords(std::vector<dimension<double>> const &pde,
                             elements::table const &table) const;

template fk::vector<float>
matlab_plot::gen_elem_coords(std::vector<dimension<float>> const &pde,
                             elements::table const &table) const;

template fk::vector<double> matlab_plot::gen_elem_coords(
    std::vector<dimension_description<double>> const &pde,
    elements::table const &table) const;

template fk::vector<float> matlab_plot::gen_elem_coords(
    std::vector<dimension_description<float>> const &pde,
    elements::table const &table) const;

template void matlab_plot::init_plotting(PDE<double> const &pde,
                                         elements::table const &table);

template void
matlab_plot::init_plotting(PDE<float> const &pde, elements::table const &table);

template void
matlab_plot::init_plotting(std::vector<dimension<float>> const &pde,
                           elements::table const &table);

template void
matlab_plot::init_plotting(std::vector<dimension<double>> const &pde,
                           elements::table const &table);

template void
matlab_plot::init_plotting(std::vector<dimension_description<float>> const &pde,
                           elements::table const &table);

template void matlab_plot::init_plotting(
    std::vector<dimension_description<double>> const &pde,
    elements::table const &table);

template void matlab_plot::plot_fval(PDE<double> const &pde,
                                     elements::table const &table,
                                     fk::vector<double> const &f_val,
                                     fk::vector<double> const &analytic_soln);

template void matlab_plot::plot_fval(PDE<float> const &pde,
                                     elements::table const &table,
                                     fk::vector<float> const &f_val,
                                     fk::vector<float> const &analytic_soln);

template void matlab_plot::plot_fval(std::vector<dimension<double>> const &dims,
                                     elements::table const &table,
                                     fk::vector<double> const &f_val,
                                     fk::vector<double> const &analytic_soln);

template void matlab_plot::plot_fval(std::vector<dimension<float>> const &pde,
                                     elements::table const &table,
                                     fk::vector<float> const &f_val,
                                     fk::vector<float> const &analytic_soln);

template void
matlab_plot::plot_fval(std::vector<dimension_description<double>> const &dims,
                       elements::table const &table,
                       fk::vector<double> const &f_val,
                       fk::vector<double> const &analytic_soln);

template void
matlab_plot::plot_fval(std::vector<dimension_description<float>> const &pde,
                       elements::table const &table,
                       fk::vector<float> const &f_val,
                       fk::vector<float> const &analytic_soln);

template void
matlab_plot::copy_pde(PDE<float> const &pde, std::string const name);
template void
matlab_plot::copy_pde(PDE<double> const &pde, std::string const name);

template matlab::data::StructArray
matlab_plot::make_term(term<float> const &term, int const max_lev);
template matlab::data::StructArray
matlab_plot::make_term(term<double> const &term, int const max_lev);

template matlab::data::StructArray
matlab_plot::make_partial_term(partial_term<float> const &pterm,
                               int const max_lev);
template matlab::data::StructArray
matlab_plot::make_partial_term(partial_term<double> const &pterm,
                               int const max_lev);

template matlab::data::StructArray
matlab_plot::make_dimension(dimension<float> const &dim);
template matlab::data::StructArray
matlab_plot::make_dimension(dimension<double> const &dim);

template fk::vector<double>
matlab_plot::col_slice(fk::vector<double> const &vec, int const n,
                       int const col) const;

template fk::vector<float> matlab_plot::col_slice(fk::vector<float> const &vec,
                                                  int const n,
                                                  int const col) const;

} // namespace asgard::ml
