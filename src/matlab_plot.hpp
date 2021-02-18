#pragma once

#include "elements.hpp"
#include "pde.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include "tools.hpp"
#include <cmath>
#include <type_traits>
#include <vector>

#include <MatlabDataArray.hpp>
#include <MatlabEngine.hpp>

// matlab::data::String is defined as std::basic_string<char16_t>
using ml_string = matlab::data::String;
// used to hold results of utf16 strings returned from matlab
using ml_stringbuf = std::basic_stringbuf<char16_t>;

using ml_wksp_type = matlab::engine::WorkspaceType;

// This class provides an interface for communicating with Matlab
// for sharing data and calling functions. While this class is mainly
// intended for plotting data in Matlab, this can be used to evaluate
// different Matlab functions and share data with a Matlab session.

// connect, connect_async, or start should be called first to connect
// to a Matlab session. connect will try to find running shared sessions
// before creating a new session.

// for plotting results in Matlab, the init_plotting function should be called.
// it is a helper function to compute element coordinates and node locations
// to pass to the plot_fval.m matlab script

// to call a matlab function or evaluate an expression, if any arguments are
// needed, then they can be created with the add_param function. Function
// arguments are held internally in an array. call, call_async, and eval
// will pop off any arguments. If arguments added by add_param are no
// longer needed, the entire array can be cleared via reset_params().

class matlab_plot
{
public:
  matlab_plot() {}

  ~matlab_plot()
  {
    if (is_open())
    {
      this->close();
    }
  }

  void connect_async()
  {
    if (!is_open())
    {
      matlab::engine::FutureResult<
          std::unique_ptr<matlab::engine::MATLABEngine>>
          ml_connection;
      ml_connection = matlab::engine::connectMATLABAsync();

      this->matlab_inst_ = ml_connection.get();
    }
  }

  void connect(std::string const &name = std::string())
  {
    // Connects to a shared MATLAB session with given name, or first available
    // if no name is given. Creates a session if none exist.
    if (!is_open())
    {
      if (name.empty() || name.compare("none") == 0)
      {
        this->matlab_inst_ = matlab::engine::connectMATLAB();
      }
      else
      {
        // Check if the named session exists
        if (!find_session(name))
        {
          std::cerr << "Specified MATLAB session '" << name
                    << "' does not exist" << '\n';
          exit(EXIT_FAILURE);
        }
        // convert to utf16 for matlab
        ml_string const name_conv =
            matlab::engine::convertUTF8StringToUTF16String(name);
        this->matlab_inst_ = matlab::engine::connectMATLAB(name_conv);
      }
    }
  }

  void start(std::vector<ml_string> const &args = std::vector<ml_string>())
  {
    if (!is_open())
    {
      this->matlab_inst_ = matlab::engine::startMATLAB(args);
    }
  }

  bool is_open() { return matlab_inst_.get() != nullptr; }

  void close()
  {
    if (!is_open())
    {
      matlab::engine::terminateEngineClient();
    }
  }

  template<typename T>
  void add_param(T const &t)
  {
    m_args_.push_back(factory_.createScalar(t));
  }

  template<typename T>
  void add_param(matlab::data::ArrayDimensions const dims, T const &t)
  {
    expect(!std::is_scalar<T>::value);
    m_args_.push_back(factory_.createArray(dims, t.begin(), t.end()));
  }

  void push_param(matlab::data::Array const &arg) { m_args_.push_back(arg); }

  void reset_params() { m_args_.clear(); }

  template<typename T>
  matlab::data::Array
  create_array(matlab::data::ArrayDimensions const dims, T const &t)
  {
    return factory_.createArray(dims, t.begin(), t.end());
  }

  // Simple wrapper for the subplot command (doesn't handle pos or ax
  // parameters, assumes position is scalar for now)
  void subplot(int const &rows, int const &cols, int const &pos)
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
  void plot()
  {
    expect(is_open());

    matlab::data::TypedArray<matlab::data::Object> plt =
        matlab_inst_->feval(u"plot", m_args_);
    m_args_.clear();
    plots_.push_back(
        std::unique_ptr<matlab::data::TypedArray<matlab::data::Object>>(&plt));
  }

  void call(std::string const func)
  {
    expect(is_open());

    matlab_inst_->feval(func.c_str(), 0, m_args_);
    m_args_.clear();
  }

  std::vector<matlab::data::Array>
  call(std::string const func, int const n_returns)
  {
    expect(is_open());

    std::vector<matlab::data::Array> plt =
        matlab_inst_->feval(func.c_str(), n_returns, m_args_);
    m_args_.clear();
    return plt;
  }

  void call_async(std::string const func)
  {
    expect(is_open());

    matlab_inst_->fevalAsync(func.c_str(), 0, m_args_);
    m_args_.clear();
  }

  std::string eval(ml_string const &stmt, std::string &err)
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

  void set_var(std::string const &name, matlab::data::Array const &var,
               ml_wksp_type const type = ml_wksp_type::BASE)
  {
    expect(is_open());
    matlab_inst_->setVariable(name, var, type);
  }

  template<typename T>
  fk::vector<T> generate_nodes(int const degree, int const level, T const min,
                               T const max) const
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

      std::vector<T> xi;
      for (const T &root : roots)
      {
        T p_map = (0.5 * (root + 1.0) + i) * h + min;
        xi.push_back(p_map);
      }

      std::vector<int> Iu;
      for (int j = 0; j < degree - 1; j++)
      {
        Iu.push_back(dof * i + j + 1);
      }
      Iu.push_back(dof * (i + 1));

      for (unsigned int j = 0; j < xi.size(); j++)
      {
        nodes(Iu[j] - 1) = xi[j];
      }
    }

    return nodes;
  }

  template<typename P>
  fk::vector<P>
  gen_elem_coords(PDE<P> const &pde, elements::table const &table) const
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
  void init_plotting(PDE<P> const &pde, elements::table const &table)
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
  void plot_fval(PDE<P> const &pde, fk::vector<P> const &f_val,
                 fk::vector<P> const &analytic_soln)
  {
    expect(sol_sizes_.size() > 0);

    size_t const ndims = static_cast<size_t>(pde.num_dims);

    m_args_.clear();
    add_param(ndims);
    add_param({1, static_cast<size_t>(analytic_soln.size())}, analytic_soln);
    add_param({1, ndims}, nodes_);
    add_param(sol_sizes_, f_val);
    add_param(sol_sizes_, analytic_soln);
    push_param(elem_coords_);

    call_async("plot_fval");
  }

private:
  template<typename P>
  fk::vector<P>
  col_slice(fk::vector<P> const &vec, int const n, int const col) const
  {
    fk::vector<P> slice(n);
    for (int i = 0; i < n; i++)
    {
      slice(i) = vec(i * n + col);
    }
    return slice;
  }

  bool find_session(std::string const &name) const
  {
    // Check if there is a running matlab session with name
    std::vector<ml_string> const &sessions = matlab::engine::findMATLAB();
    if (sessions.size() == 0)
    {
      return false;
    }

    for (auto const &session : sessions)
    {
      // Convert from UTF16 to UTF8
      std::string const conv =
          matlab::engine::convertUTF16StringToUTF8String(session);
      if (name.compare(conv) == 0)
      {
        return true;
      }
    }
    return false;
  }

  template<typename P>
  inline std::vector<size_t> get_soln_sizes(PDE<P> const &pde) const
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
  inline int get_soln_size(PDE<P> const &pde, int const dim) const
  {
    // Gets the solution size for a given dimension (see real_solution_size() in
    // transformations)
    return pde.get_dimensions()[dim].get_degree() *
           std::pow(2, pde.get_dimensions()[pde].get_level());
  }

  // Pointer to the matlab instance
  std::unique_ptr<matlab::engine::MATLABEngine> matlab_inst_ = nullptr;

  // Matlab DataAPI factory
  matlab::data::ArrayFactory factory_;

  // Holds a pointer to each plot object
  std::vector<std::unique_ptr<matlab::data::TypedArray<matlab::data::Object>>>
      plots_;

  // Holds the current list of arguments for the next matlab call
  std::vector<matlab::data::Array> m_args_;

  // Coordinates of nodes to use for plotting
  std::vector<matlab::data::Array> nodes_;

  matlab::data::Array elem_coords_;

  std::vector<size_t> sol_sizes_;
};
