#include "io.hpp"

#if defined(ASGARD_IO_MATLAB_DIR)

template<typename P>
matlab_engine<P>::matlab_engine()
    : time_step_index(0), matlab_solution_name("asgard_time_series_solution"),
      matlab_tmp_var_name("vector")
{
  std::vector<std::u16string> names = matlab::engine::findMATLAB();

  assert(names.size() > 0);

  std::u16string session_name = names[0];

  try
  {
    engine = matlab::engine::connectMATLAB(session_name);
  }

  catch (const std::exception &e)
  {
    std::cout << e.what() << std::endl;
  }

  return;
}

template<typename P>
matlab_engine<P>::~matlab_engine()
{
  std::string cleanup_command = "clear ";
  cleanup_command += matlab_tmp_var_name += ";";
  engine->eval(matlab::engine::convertUTF8StringToUTF16String(cleanup_command));
  return;
}

template<typename P>
void matlab_engine<P>::send_to_matlab(fk::vector<P> const &v)
{
  assert(v.size() > 0);

  std::string append_vector_command = matlab_solution_name;
  append_vector_command += "{";
  append_vector_command += std::to_string(time_step_index + 1) += "} = ";
  append_vector_command += matlab_tmp_var_name + ";";

  matlab::data::TypedArray<P> array =
      array_factory.createArray({1, v.size()}, v.data(), v.data() + v.size());

  engine->setVariable(matlab_tmp_var_name, array,
                      matlab::engine::WorkspaceType::GLOBAL);
  engine->eval(
      matlab::engine::convertUTF8StringToUTF16String(append_vector_command));

  time_step_index++;

  return;
}

/* Explicit instantiations of matlab_engine class */
template class matlab_engine<double>;
template class matlab_engine<float>;

#endif
