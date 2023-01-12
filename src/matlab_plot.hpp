#pragma once

#include "asgard_dimension.hpp"
#include "elements.hpp"
#include "pde.hpp"
#include "tensors.hpp"
#include "tools.hpp"
#include <MatlabDataArray.hpp>
#include <MatlabEngine.hpp>
#include <type_traits>
#include <vector>

namespace asgard::ml
{
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
  matlab_plot() = default;

  ~matlab_plot()
  {
    if (is_open())
    {
      if (started_)
      {
        // Call matlab to wait for any open figure handles before closing
        call("wait_for_plots");
      }
      this->close();
    }
  }

  void connect_async();

  void connect(std::string const &name = std::string());

  void start(std::vector<ml_string> const &args = std::vector<ml_string>());

  bool is_open() { return matlab_inst_.get() != nullptr; }

  void close();

  template<typename T, typename... Args>
  void add_param(T const &t, Args &&...args)
  {
    m_args_.push_back(factory_.createScalar(t));
    add_param(args...);
  }

  template<typename T>
  void add_param(T const &t)
  {
    m_args_.push_back(factory_.createScalar(t));
  }

  template<typename T, typename... Args>
  void add_param(matlab::data::ArrayDimensions const dims, T const &t,
                 Args &&...args)
  {
    static_assert(!std::is_scalar<T>::value);
    m_args_.push_back(factory_.createArray(dims, t.begin(), t.end()));
    add_param(dims, args...);
  }

  template<typename T>
  void add_param(matlab::data::ArrayDimensions const dims, T const &t)
  {
    static_assert(!std::is_scalar<T>::value);
    m_args_.push_back(factory_.createArray(dims, t.begin(), t.end()));
  }

  void push_param(matlab::data::Array const &arg) { m_args_.push_back(arg); }

  void reset_params() { m_args_.clear(); }

  template<typename T>
  matlab::data::Array create_array(T const &t)
  {
    return create_array({size_t{1}, static_cast<size_t>(t.size())}, t);
  }

  template<typename T>
  matlab::data::Array
  create_array(matlab::data::ArrayDimensions const dims, T const &t)
  {
    return factory_.createArray(dims, t.begin(), t.end());
  }

  template<typename T>
  matlab::data::Array matrix_to_array(fk::matrix<T> const &mat)
  {
    // temporary workaround: convert to vector so matlab can use its iterator
    auto const &vec = fk::vector<T>(mat);
    return create_array(
        {static_cast<size_t>(mat.ncols()), static_cast<size_t>(mat.nrows())},
        vec);
  }

  // Simple wrapper for the subplot command (doesn't handle pos or ax
  // parameters, assumes position is scalar for now)
  void subplot(int const &rows, int const &cols, int const &pos);

  // Simple wrapper for the matlab plot command
  void plot();

  void call(std::string const func);

  std::vector<matlab::data::Array>
  call(std::string const func, int const n_returns);

  void call_async(std::string const func);

  std::string eval(ml_string const &stmt, std::string &err);

  void set_var(std::string const &name, matlab::data::Array const &var,
               ml_wksp_type const type = ml_wksp_type::BASE);

  template<typename T>
  void push(std::string const &name, fk::matrix<T> const &data,
            ml_wksp_type const type = ml_wksp_type::BASE);

  template<typename T>
  fk::vector<T> generate_nodes(dimension<T> const &dim)
  {
    return generate_nodes(dim.get_degree(), dim.get_level(), dim.domain_min,
                          dim.domain_max);
  }

  template<typename T>
  fk::vector<T> generate_nodes(dimension_description<T> const &dim)
  {
    return generate_nodes(dim.degree, dim.level, dim.d_min, dim.d_max);
  }

  template<typename T>
  fk::vector<T> generate_nodes(int const degree, int const level, T const min,
                               T const max) const;

  template<typename P>
  fk::vector<P>
  gen_elem_coords(PDE<P> const &pde, elements::table const &table) const;

  template<typename P>
  fk::vector<P> gen_elem_coords(std::vector<dimension<P>> const &dims,
                                elements::table const &table) const;

  template<typename P>
  fk::vector<P>
  gen_elem_coords(std::vector<dimension_description<P>> const &dims,
                  elements::table const &table) const;

  template<typename P>
  void init_plotting(PDE<P> const &pde, elements::table const &table);

  template<typename P>
  void init_plotting(std::vector<dimension<P>> const &dims,
                     elements::table const &table);

  template<typename P>
  void init_plotting(std::vector<dimension_description<P>> const &dims,
                     elements::table const &table);

  template<typename P>
  void
  plot_fval(PDE<P> const &pde, elements::table const &table,
            fk::vector<P> const &f_val, fk::vector<P> const &analytic_soln);

  template<typename P>
  void
  plot_fval(std::vector<dimension<P>> const &pde, elements::table const &table,
            fk::vector<P> const &f_val, fk::vector<P> const &analytic_soln);

  template<typename P>
  void plot_fval(std::vector<dimension_description<P>> const &pde,
                 elements::table const &table, fk::vector<P> const &f_val,
                 fk::vector<P> const &analytic_soln);

  template<typename P>
  void copy_pde(PDE<P> const &pde, std::string const name = std::string("pde"));

  template<typename P>
  matlab::data::StructArray make_term(term<P> const &term, int const max_lev);

  template<typename P>
  matlab::data::StructArray
  make_partial_term(partial_term<P> const &pterm, int const max_lev);

  template<typename P>
  matlab::data::StructArray make_dimension(dimension<P> const &dim);

private:
  template<typename P>
  fk::vector<P>
  col_slice(fk::vector<P> const &vec, int const n, int const col) const;

  bool find_session(std::string const &name, bool const find_name,
                    std::string &session_name) const;

  template<typename P>
  inline std::vector<size_t> get_soln_sizes(PDE<P> const &pde) const;

  template<typename P>
  inline std::vector<size_t>
  get_soln_sizes(std::vector<dimension<P>> const &dims) const;

  template<typename P>
  inline std::vector<size_t>
  get_soln_sizes(std::vector<dimension_description<P>> const &dims) const;

  template<typename P>
  inline int get_soln_size(PDE<P> const &pde, int const dim) const;

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

  // Whether MATLAB was started by asgard - this will wait on any plots
  bool started_ = false;
};

} // namespace asgard::ml
