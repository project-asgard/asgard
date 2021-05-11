#include "matlab_plot.hpp"

#include "quadrature.hpp"
#include <cmath>
#include <type_traits>
namespace ml
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
  int const ndims = pde.num_dims;

  fk::vector<P> center_coords(ndims * table.size());

  // Iterate over dimensions first since matlab needs col-major order
  for (int d = 0; d < ndims; d++)
  {
    P const max = pde.get_dimensions()[d].domain_max;
    P const min = pde.get_dimensions()[d].domain_min;
    P const rng = max - min;

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

      P const x = x0 * rng + min;

      center_coords(d * table.size() + i) = x;
    }
  }

  return center_coords;
}

template<typename P>
void matlab_plot::init_plotting(PDE<P> const &pde, elements::table const &table)
{
  // Generates cell array of nodes and element coordinates needed for plotting
  sol_sizes_ = get_soln_sizes(pde);

  nodes_          = std::vector<matlab::data::Array>(pde.num_dims);
  auto const dims = pde.get_dimensions();

  for (int i = 0; i < pde.num_dims; i++)
  {
    auto const &dim       = dims[i];
    auto const &node_list = generate_nodes(dim.get_degree(), dim.get_level(),
                                           dim.domain_min, dim.domain_max);
    nodes_[i] =
        create_array({1, static_cast<size_t>(node_list.size())}, node_list);
  }

  auto const &elem_coords = gen_elem_coords(pde, table);
  elem_coords_            = create_array(
      {static_cast<size_t>(table.size()), static_cast<size_t>(pde.num_dims)},
      elem_coords);
}

template<typename P>
void matlab_plot::plot_fval(PDE<P> const &pde, elements::table const &table,
                            fk::vector<P> const &f_val,
                            fk::vector<P> const &analytic_soln)
{
  expect(sol_sizes_.size() > 0);

  size_t const ndims = static_cast<size_t>(pde.num_dims);

  if (table.size() * ndims != elem_coords_.getNumberOfElements())
  {
    // Regenerate the element coordinates and nodes if the grid was adapted
    init_plotting(pde, table);
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
inline std::vector<size_t> matlab_plot::get_soln_sizes(PDE<P> const &pde) const
{
  // Returns a vector of the solution size for each dimension
  std::vector<size_t> sizes(pde.num_dims);
  for (int i = 0; i < pde.num_dims; i++)
  {
    sizes[i] = pde.get_dimensions()[i].get_degree() *
               std::pow(2, pde.get_dimensions()[i].get_level());
  }
  return sizes;
}

template<typename P>
inline int matlab_plot::get_soln_size(PDE<P> const &pde, int const dim) const
{
  // Gets the solution size for a given dimension (see real_solution_size() in
  // transformations)
  return pde.get_dimensions()[dim].get_degree() *
         std::pow(2, pde.get_dimensions()[pde].get_level());
}

/* explicit instantiations */
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

template void matlab_plot::init_plotting(PDE<double> const &pde,
                                         elements::table const &table);

template void
matlab_plot::init_plotting(PDE<float> const &pde, elements::table const &table);

template void matlab_plot::plot_fval(PDE<double> const &pde,
                                     elements::table const &table,
                                     fk::vector<double> const &f_val,
                                     fk::vector<double> const &analytic_soln);

template void matlab_plot::plot_fval(PDE<float> const &pde,
                                     elements::table const &table,
                                     fk::vector<float> const &f_val,
                                     fk::vector<float> const &analytic_soln);

template fk::vector<double>
matlab_plot::col_slice(fk::vector<double> const &vec, int const n,
                       int const col) const;

template fk::vector<float> matlab_plot::col_slice(fk::vector<float> const &vec,
                                                  int const n,
                                                  int const col) const;

} // namespace ml
