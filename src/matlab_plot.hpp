#pragma once

#include "elements.hpp"
#include "pde.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
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

class matlab_plot
{
public:
  matlab_plot() {}

  ~matlab_plot()
  {
    if (!matlab_inst_.get())
    {
      // TODO: should this wait until all plots are closed before shutting down
      // the matlab session? closing the session will close all plot windows
      // immediately, so they won't stay open afterwards
      if (plots_.size() > 0) {}
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

  void connect(const std::string &name = std::string())
  {
    // Connects to a shared MATLAB session with given name, or first available
    // if no name is given. Creates a session if none exist.
    if (!is_open())
    {
      if (name.empty() || name.compare("none") != 0)
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
        ml_string name_conv =
            matlab::engine::convertUTF8StringToUTF16String(name);
        this->matlab_inst_ = matlab::engine::connectMATLAB(name_conv);
      }
    }
  }

  void start(const std::vector<ml_string> &args = std::vector<ml_string>())
  {
    if (!is_open())
    {
      this->matlab_inst_ = matlab::engine::startMATLAB(args);
    }
  }

  bool is_open() { return matlab_inst_.get() != nullptr; }

  void close()
  {
    // TODO: process doesn't seem to return properly when this is called
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
  void add_param(const matlab::data::ArrayDimensions dims, T const &t)
  {
    assert(!std::is_scalar<T>::value);
    m_args_.push_back(factory_.createArray(dims, t.begin(), t.end()));
  }

  void push_param(const matlab::data::Array &arg) { m_args_.push_back(arg); }

  void reset_params() { m_args_.clear(); }

  template<typename T>
  matlab::data::Array
  create_array(matlab::data::ArrayDimensions dims, T const &t)
  {
    return factory_.createArray(dims, t.begin(), t.end());
  }

  // Simple wrapper for the subplot command (doesn't handle pos or ax
  // parameters, assumes position is scalar for now)
  void subplot(int const &rows, int const &cols, int const &pos)
  {
    /*
    m_args_.clear();
    add_param(rows);
    add_param(cols);
    add_param(pos);

    matlab::data::TypedArray<matlab::data::Object> plt =
    matlab_inst_.get()->feval(u"subplot", m_args_); m_args_.clear();
    plots_.push_back(std::unique_ptr<matlab::data::TypedArray<matlab::data::Object>>(&plt));
    */

    // Temporary workaround.. calling subplot using feval is giving a type error
    // in subplot.m
    std::stringstream stmt;
    stmt << "subplot(" << rows << "," << cols << "," << pos << ")";
    matlab::engine::String ml_stmt =
        matlab::engine::convertUTF8StringToUTF16String(stmt.str());
    matlab_inst_.get()->eval(ml_stmt);
  }

  // Simple wrapper for the matlab plot command
  void plot()
  {
    matlab::data::TypedArray<matlab::data::Object> plt =
        matlab_inst_.get()->feval(u"plot", m_args_);
    m_args_.clear();
    plots_.push_back(
        std::unique_ptr<matlab::data::TypedArray<matlab::data::Object>>(&plt));
  }

  void call(std::string func)
  {
    matlab_inst_.get()->feval(func.c_str(), 0, m_args_);
    m_args_.clear();
  }

  std::vector<matlab::data::Array> call(std::string func, int n_returns)
  {
    std::vector<matlab::data::Array> plt =
        matlab_inst_.get()->feval(func.c_str(), n_returns, m_args_);
    m_args_.clear();
    return plt;
  }

  std::string eval(const ml_string &stmt, std::string &err)
  {
    assert(is_open());

    auto output    = std::make_shared<ml_stringbuf>();
    auto err_utf16 = std::make_shared<ml_stringbuf>();

    matlab_inst_.get()->eval(stmt, output, err_utf16);

    std::string output_utf8 =
        matlab::engine::convertUTF16StringToUTF8String(output->str());
    err = matlab::engine::convertUTF16StringToUTF8String(err_utf16->str());

    return output_utf8;
  }

  void set_var(const std::string &name, const matlab::data::Array &var,
               ml_wksp_type type = ml_wksp_type::BASE)
  {
    matlab_inst_->setVariable(name, var, type);
  }

  template<typename T>
  fk::vector<T> generate_nodes(int degree, int level, T min, T max)
  {
    // Trimmed version of matrix_plot_D.m to get only the nodes
    int n        = pow(2, level);
    int mat_dims = degree * n;
    T h          = (max - min) / n;

    // TODO: fully implement the output_grid options from matlab (this is just
    // the 'else' case)
    auto lgwt  = legendre_weights(degree, -1.0, 1.0, true);
    auto roots = lgwt[0];

    uint dof = roots.size();

    // Using the same variable names as in matrix_plot_D.m for easier comparing
    // TODO: make these names better..
    fk::matrix<T> Meval = fk::matrix<T>(mat_dims, mat_dims);

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

      std::vector<int> Iv;
      std::vector<int> Iu;
      for (int j = 0; j < degree - 1; j++)
      {
        Iv.push_back(degree * i + j + 1);
        Iu.push_back(dof * i + j + 1);
      }
      Iv.push_back(degree * (i + 1));
      Iu.push_back(dof * (i + 1));

      Meval.set_submatrix(Iu[0] - 1, Iv[0] - 1, p_val[0]);

      for (uint j = 0; j < xi.size(); j++)
      {
        nodes(Iu[j] - 1) = xi[j];
      }
    }

    return nodes;
  }

  template<typename P>
  std::vector<P>
  gen_elem_coords(PDE<P> const &pde, elements::table const &table)
  {
    int ndims = pde.num_dims;

    // TODO: this could be replaced by fk::matrix<P>, but it needs a forward
    // iterator defined
    std::vector<P> center_coords;

    // Iterate over dimensions first since matlab needs col-major order
    for (int d = 0; d < ndims; d++)
    {
      P max = pde.get_dimensions()[d].domain_max;
      P min = pde.get_dimensions()[d].domain_min;
      P rng = max - min;

      for (int i = 0; i < table.size(); i++)
      {
        fk::vector<int> coords = table.get_coords(i);

        int lev = coords(d);
        int pos = coords(ndims + d);

        assert(lev >= 0);
        assert(pos >= 0);

        P x0;
        if (lev > 1)
        {
          P s = pow(2, lev - 1) - 1.0;
          P h = 1.0 / (pow(2, lev - 1));
          P w = 1.0 - h;
          P o = 0.5 * h;

          x0 = pos / s * w + o;
        }
        else
        {
          x0 = pos + 0.5;
        }

        P x = x0 * rng + min;

        center_coords.push_back(x);
      }
    }

    return center_coords;
  }

  template<typename P>
  void init_plotting(PDE<P> const &pde, elements::table const &table)
  {
    // Generates cell array of nodes and element coordinates needed for plotting
    sol_sizes_ = get_soln_sizes(pde);

    nodes_    = std::vector<matlab::data::Array>(pde.num_dims);
    auto dims = pde.get_dimensions();
    // for (auto const &dim : dims)
    for (int i = 0; i < pde.num_dims; i++)
    {
      auto const &dim = dims[i];
      auto node_list  = generate_nodes(dim.get_degree(), dim.get_level(),
                                      dim.domain_min, dim.domain_max);
      nodes_[i] =
          create_array({1, static_cast<size_t>(node_list.size())}, node_list);
    }

    auto elem_coords = gen_elem_coords(pde, table);
    elem_coords_     = create_array(
        {static_cast<size_t>(table.size()), static_cast<size_t>(pde.num_dims)},
        elem_coords);
  }

  template<typename P>
  inline std::vector<size_t> get_soln_sizes(PDE<P> const &pde)
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
  inline int get_soln_size(PDE<P> const &pde, int dim)
  {
    // Gets the solution size for a given dimension (see real_solution_size() in
    // transformations)
    return pde.get_dimensions()[dim].get_degree() *
           std::pow(2, pde.get_dimensions()[pde].get_level());
  }

  template<typename P>
  void plot_fval_ml(PDE<P> const &pde, fk::vector<P> const &f_val,
                    fk::vector<P> const &analytic_soln)
  {
    assert(sol_sizes_.size() > 0);

    size_t ndims = static_cast<size_t>(pde.num_dims);

    m_args_.clear();
    add_param(ndims);
    add_param({1, static_cast<size_t>(analytic_soln.size())}, analytic_soln);
    add_param({1, ndims}, nodes_);
    add_param(sol_sizes_, f_val);
    add_param(sol_sizes_, analytic_soln);
    push_param(elem_coords_);

    call("plot_fval_cpp");
  }

  template<typename P>
  void plot_fval(PDE<P> const &pde, fk::vector<P> const &f_val,
                 fk::vector<P> const &analytic_soln, bool overplot = true)
  {
    // C++ version of the plot_fval matlab function

    assert(sol_sizes_.size() > 0);
    assert(nodes_.size() == pde.num_dims);

    // Don't plot analytical if the PDE doesn't have one
    if (overplot && !pde.has_analytic_soln)
    {
      overplot = false;
    }

    size_t ndims = static_cast<size_t>(pde.num_dims);

    m_args_.clear();

    if (ndims == 1)
    {
      push_param(nodes_[0]);
      add_param(sol_sizes_, f_val);
      add_param("-o");
      call("plot");

      if (overplot)
      {
        // call("hold on");
        push_param(nodes_[0]);
        add_param(sol_sizes_, analytic_soln);
        add_param("-");
        call("plot");
        // call("hold off");
      }
    }

    else if (ndims == 2)
    {
      subplot(2, 2, 1);

      int sy = std::max(1, static_cast<int>(sol_sizes_[1]) / 2 - 1);
      if (sol_sizes_[1] > 2)
      {
        sy++;
      }

      // Column slice
      auto yslice = col_slice(f_val, sol_sizes_[1], sy);
      yslice.print();

      push_param(nodes_[0]);
      add_param({1, sol_sizes_[1]}, yslice);
      add_param("-o");
      call("plot");
      add_param("1D slice (vertical)");
      call("title");

      if (overplot)
      {
        yslice = col_slice(analytic_soln, sol_sizes_[1], sy);
        // call("hold on");
        matlab_inst_.get()->eval(u"hold on");
        push_param(nodes_[0]);
        add_param({1, sol_sizes_[1]}, yslice);
        add_param("-");
        call("plot");
        matlab_inst_.get()->eval(u"hold off");
        // call("hold off");
      }

      int sx = std::max(1, static_cast<int>(sol_sizes_[0]) / 2 - 1);
      if (sol_sizes_[0] > 2)
      {
        sx++;
      }

      // Row slice
      auto xslice = f_val.extract(sx * sol_sizes_[1],
                                  sx * sol_sizes_[1] + sol_sizes_[0] - 1);
      xslice.print();

      subplot(2, 2, 2);
      push_param(nodes_[1]);
      add_param({1, sol_sizes_[0]}, xslice);
      add_param("-o");
      call("plot");
      add_param("1D slice (horizontal)");
      call("title");

      if (overplot)
      {
        xslice = analytic_soln.extract(sx * sol_sizes_[1],
                                       sx * sol_sizes_[1] + sol_sizes_[0] - 1);
        matlab_inst_.get()->eval(u"hold on");
        push_param(nodes_[1]);
        add_param({1, sol_sizes_[0]}, xslice);
        add_param("-");
        call("plot");
        matlab_inst_.get()->eval(u"hold off");
      }

      subplot(2, 2, 3);
      auto f_noise = f_val * 1.0001;
      push_param(nodes_[0]);
      push_param(nodes_[1]);
      add_param(sol_sizes_, f_noise);
      add_param("LineColor");
      add_param("None");
      call("contourf");
      add_param("numeric 2D solution");
      call("title");
    }

    else if (ndims == 3)
    {}
  }

private:
  template<typename P>
  fk::vector<P> col_slice(fk::vector<P> const &vec, const int n, const int col)
  {
    fk::vector<P> slice(n);
    for (int i = 0; i < n; i++)
    {
      slice(i) = vec(i * n + col);
    }
    return slice;
  }

  bool find_session(const std::string &name)
  {
    // Check if there is a running matlab session with name
    std::vector<ml_string> sessions = matlab::engine::findMATLAB();
    if (sessions.size() == 0)
    {
      return false;
    }

    for (const auto &session : sessions)
    {
      // Convert from UTF16 to UTF8
      std::string conv =
          matlab::engine::convertUTF16StringToUTF8String(session);
      printf("Comparing ML session '%s' to given '%s'\n", conv.c_str(),
             name.c_str());
      if (name.compare(conv) == 0)
      {
        return true;
      }
    }
    return false;
  }

  // Pointer to the matlab instance
  std::unique_ptr<matlab::engine::MATLABEngine> matlab_inst_ = nullptr;

  // Matlab DataAPI factory
  matlab::data::ArrayFactory factory_;

  // Whether this is connected to a shared matlab session
  bool is_shared_ = false;

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
