
#include "asgard_test_macros.hpp"

#include "asgard_testpdes.hpp"

// reintroduce private headers
#include <highfive/H5Easy.hpp>
#include <highfive/H5DataType.hpp>
#include <highfive/H5DataSpace.hpp>

using namespace asgard;

template<typename P>
HighFive::DataSet
initialize_output_file(std::vector<P> const &vec,
                       std::string const output_dataset_name = "asgard")
{
  std::string const output_file_name = output_dataset_name + ".h5";

  size_t vec_size = vec.size();

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
  dataset.select({0, 0}, {1, vec_size}).write(vec);

  return dataset;
}

template<typename P>
void update_output_file(HighFive::DataSet &dataset, std::vector<P> const &vec,
                        std::string const output_dataset_name = "asgard")
{
  std::string const output_file_name = output_dataset_name + ".h5";
  unsigned int vec_size              = (unsigned int)vec.size();

  // Get the size of the existing dataset
  auto dataset_size = dataset.getDimensions();
  // Resize in the time dimension by 1
  dataset.resize({dataset_size[0] + 1, dataset_size[1]});
  // Write the latest vec into the new row
  dataset.select({dataset_size[0], 0}, {1, vec_size}).write(vec);
}

template<typename TestType>
void check_api_hdf5() {
  current_test<TestType> name_("HighFive <-> HDF5");

  // check if the API writes the correct file
  std::string const output_dataset_name("_asgard_test");

  // the golden values
  std::vector<TestType> const gold_vec1{1, 2, 3, 4, 5};
  std::vector<TestType> const gold_vec2{5, 6, 7, 8, 9};

  // setup output file and write initial condition
  auto output_dataset = initialize_output_file(gold_vec1);

  // write output to file
  update_output_file(output_dataset, gold_vec2);

  // now read back what we wrote out
  std::vector<std::vector<TestType>> read_data;
  output_dataset.read(read_data);

  auto const dataset_size = output_dataset.getDimensions();
  auto const vec1         = read_data[0];
  auto const vec2         = read_data[1];
  tassert(vec1.size() == gold_vec1.size());
  tassert(vec2.size() == gold_vec2.size());

  for (int i = 0; i < static_cast<int>(vec1.size()); i++)
  {
    tassert(vec1[i] == gold_vec1[i]);
    tassert(vec2[i] == gold_vec2[i]);
  }
}

std::string const filename = "_asgard_save_test.h5";

template<typename TestType>
void simple_restart() {
  current_test<TestType> name_("simple restart logic");

  TestType constexpr tol = (std::is_same_v<TestType, double>) ? 1.E-14 : 1.E-5;

  int const num_dims = 4;
  std::string const title    = "basic io test";
  std::string const subtitle = "test 1";

  prog_opts options = make_opts("-d 3 -l 3 -m 4 -dt 0.5 -time 1.0");
  options.title    = title;
  options.subtitle = subtitle;
  pde_domain<TestType> domain(num_dims);
  discretization_manager<TestType> ref(pde_scheme<TestType>(options, domain));

  ref.save_snapshot(filename);

  prog_opts opts2 = make_opts("-restart " + filename);
  discretization_manager<TestType> disc(pde_scheme<TestType>(opts2, domain));

  tassert(ref.num_dims() == num_dims);
  tassert(disc.num_dims() == num_dims);

  tassert(disc.num_dims() == num_dims);
  tassert(disc.num_dims() == num_dims);

  tassert(disc.options().title == title);
  tassert(disc.options().subtitle == subtitle);

  tassert(disc.degree() == 3);

  tassert(std::abs(ref.dt() - disc.dt()) < tol);
  tassert(std::abs(ref.time() - disc.time()) < tol);
  tassert(std::abs(ref.stop_time() - disc.stop_time()) < tol);
  tassert(ref.remaining_steps() == disc.remaining_steps());

  tassert(ref.get_grid().num_indexes() == disc.get_grid().num_indexes());
  tassert(ref.get_grid().num_dims() == disc.get_grid().num_dims());
  tassert(ref.get_grid().generation() == disc.get_grid().generation());
  {
    int const *g1 = ref.get_grid()[0];
    int const *g2 = disc.get_grid()[0];
    int64_t const num = ref.get_grid().num_indexes() * ref.get_grid().num_dims();
    int max_index_error = 0;
    for (int64_t i = 0; i < num; i++)
      max_index_error = std::max(max_index_error, std::abs(g1[i] - g2[i]));
    tassert(max_index_error == 0);

    auto const &grid = disc.get_grid();
    for (int d : iindexof(num_dims)) {
      tassert(grid.current_level(d) == 3);
      tassert(grid.max_index(d) == 16);
    }
  }
}

template<typename TestType>
void reset_time_params() {
  current_test<TestType> name_("modify time-params in restart");

  int const num_dims = 1;
  std::string const title    = "restart changes the time parameters";
  std::string const subtitle = "test 2";

  prog_opts options = make_opts("-d 3 -l 3 -m 4 -dt 0.5 -time 3.0 -a 0.0625");
  options.title    = title;
  options.subtitle = subtitle;
  pde_domain<TestType> domain(num_dims);
  discretization_manager<TestType> ref(pde_scheme<TestType>(options, domain));
  ref.set_time(TestType{2});
  tassert(ref.time() == 2);
  tassert(ref.stop_time() == 3);
  tassert(ref.remaining_steps() == 6);

  ref.save_snapshot(filename);

  prog_opts opts2 = make_opts("-restart " + filename + " -time 4");
  discretization_manager<TestType> d1(pde_scheme<TestType>(opts2, domain));
  tassert(d1.time() == 2);
  tassert(d1.stop_time() == 4);
  tassert(d1.options().adapt_threshold);
  tassert(d1.options().adapt_threshold.value() == 0.0625);

  opts2 = make_opts("-restart " + filename + " -dt 0.25 -a 0.125");
  discretization_manager<TestType> d2(pde_scheme<TestType>(opts2, domain));
  tassert(d2.dt() == TestType{0.25});
  // stop time minus current time is 1, with dt = 0.25 we have 4 steps
  tassert(d2.remaining_steps() == 4);
  tassert(d2.options().adapt_threshold.value() == 0.125);

  opts2 = make_opts("-restart " + filename + " -n 8 -noa");
  discretization_manager<TestType> d3(pde_scheme<TestType>(opts2, domain));
  tassert(d3.remaining_steps() == 8);
  // stop time minus current time is 1, with 8 streps, we have dt = 0.25
  tassert(d3.dt() == TestType{0.125});
  tassert(not d3.options().adapt_threshold);
}

template<typename TestType>
void restart_errors() {
  current_test<TestType> name_("error handling during resrat");

  int const num_dims = 2;
  std::string const title    = "restart errors";
  std::string const subtitle = "test 3";

  prog_opts options = make_opts("-d 3 -l 3 -m 4 -dt 0.5 -time 3.0");
  options.title    = title;
  options.subtitle = subtitle;
  pde_domain<TestType> domain(num_dims);
  discretization_manager<TestType> ref(pde_scheme<TestType>(options, domain));
  ref.set_time(TestType{2});

  ref.save_snapshot(filename);

  // try to restart from a missing file
  prog_opts opts2 = make_opts("-restart wrong_file");
  terror_message(discretization_manager<TestType>(pde_scheme<TestType>(opts2, domain)),
                 "Cannot find file: 'wrong_file'");

  // the file is correct, but the dimensions are wrong
  opts2 = make_opts("-restart " + filename);
  terror_message(discretization_manager<TestType>(
                 pde_scheme<TestType>(opts2, pde_domain<TestType>(num_dims + 1))),
                 "Mismatch in the number of dimensions, pde is set for '3' "
                 "but the file contains data for '2'. "
                 "The restart file must match the dimensions.");

  // dimension is correct but there are too many time parameters
  opts2 = make_opts("-restart " + filename + " -dt 0.5 -time 1.0 -n 20");
  terror_message(discretization_manager<TestType>(pde_scheme<TestType>(opts2, domain)),
                 "cannot simultaneously specify -dt, -num-steps, and -time");

  // setting end time before the current time
  opts2 = make_opts("-restart " + filename + " -dt 0.5 -time 1.0");
  terror_message(discretization_manager<TestType>(pde_scheme<TestType>(opts2, domain)),
                 "cannot reset the final time to an instance before the current time");
}

template<typename P>
void restart_longer() {
  current_test<P> name_("longer restart stability");

  using pde = pde_contcos;

  // 1. make a pde and set 4 time-steps, advance in time and save the state
  //    - check initial and final error, and file-existing
  // 2. restart and set 4 additional time-steps, advance in time
  //    - make sure restarted matches the saved and new end-time is set
  // 3. compare against a one-shot integration using 4 steps

  auto options = make_opts("-l 5 -d 2 -dt 0.01 -n 4 -of _asg_testfile.h5");
  discretization_manager<P> disc(make_testpde<pde, P>(2, options));

  tassert((get_qoi_indicator<pde, P>(disc) < 1.E-2));
  disc.advance_time();
  tassert((get_qoi_indicator<pde, P>(disc) < 1.E-2));

  disc.save_final_snapshot();
  tassert(std::filesystem::exists("_asg_testfile.h5"));

  auto ropts = make_opts("-n 4 -dt 0.01 -restart _asg_testfile.h5");
  discretization_manager<P> rdisc(make_testpde<pde, P>(2, ropts));

  tassert(std::abs(get_qoi_indicator<pde, P>(disc) - get_qoi_indicator<pde, P>(rdisc)) < 1.E-10);

  tassert(std::abs(rdisc.stop_time() - 0.08) < 2.E-9); // updated the stop time
  rdisc.advance_time();

  tassert(std::abs(rdisc.time() - 0.08) < 1.E-8);

  options = make_opts("-l 5 -d 2 -dt 0.01 -n 8");
  discretization_manager<P> reff(make_testpde<pde, P>(2, options));
  reff.advance_time();

  tassert(std::abs(reff.time() - 0.08) < 1.E-8);

  tassert(std::abs(get_qoi_indicator<pde, P>(reff) - get_qoi_indicator<pde, P>(rdisc)) < 1.E-10);
}

template<typename P>
void restart_adapt() {
  current_test<P> name_("adaptive restart");

  using pde = pde_contcos;

  // similar to above but uses adaptivity

  auto options = make_opts("-l 8 -d 2 -dt 0.01 -n 8 -a 1.E-2 -of _asg_testfile.h5");
  discretization_manager<P> disc(make_testpde<pde, P>(2, options));

  tassert((get_qoi_indicator<pde, P>(disc) < 1.E-2));
  disc.advance_time(4);
  tassert((get_qoi_indicator<pde, P>(disc) < 1.E-2));

  size_t const aux_size = disc.current_state().size();
  std::vector<P> vnum(aux_size, P{11});
  vnum[1] = 42;
  vnum[2] = 3;
  disc.add_aux_field({"aux-field", std::move(vnum)}); // add some AUX data

  disc.save_final_snapshot();
  tassert(std::filesystem::exists("_asg_testfile.h5"));

  auto ropts = make_opts("-restart _asg_testfile.h5");
  discretization_manager<P> rdisc(make_testpde<pde, P>(2, ropts));

  tassert(rdisc.get_grid().num_indexes() == disc.get_grid().num_indexes());
  tassert(std::abs(get_qoi_indicator<pde, P>(disc) - get_qoi_indicator<pde, P>(rdisc)) < 1.E-10);

  tassert(rdisc.get_aux_fields().size() == 1);
  tassert(rdisc.get_aux_fields().front().name == "aux-field");
  tassert(rdisc.get_aux_fields().front().data.size() == aux_size);
  tassert(rdisc.get_aux_fields().front().grid.size() ==
          static_cast<size_t>(2 * disc.get_grid().num_indexes()));
  tassert(rdisc.get_aux_fields().front().data[1] == 42);
  tassert(rdisc.get_aux_fields().front().data[2] == 3);

  rdisc.clear_aux_fields();
  tassert(rdisc.get_aux_fields().empty());

  tassert(std::abs(rdisc.stop_time() - 0.08) < 2.E-9); // updated the stop time
  rdisc.advance_time();

  tassert(std::abs(rdisc.time() - 0.08) < 1.E-8);

  options = make_opts("-l 8 -d 2 -dt 0.01 -n 8 -a 1.E-2");
  discretization_manager<P> reff(make_testpde<pde, P>(2, options));
  reff.advance_time();

  tassert(rdisc.get_grid().num_indexes() == reff.get_grid().num_indexes());

  tassert(std::abs(reff.time() - 0.08) < 1.E-8);

  tassert(std::abs(get_qoi_indicator<pde, P>(reff) - get_qoi_indicator<pde, P>(rdisc)) < 1.E-10);
}

template<typename P>
void restart_moments() {
  current_test<P> name_("adaptive restart");

  using pde = pde_twostream;

  // similar to above but uses adaptivity

  auto options = make_opts("-l 7 -d 2 -dt 1.953125E-3 -n 8 -a 1.E-6 -of _asg_testfile.h5");
  discretization_manager<P> disc(make_testpde<pde, P>(2, options));
  double const ienergy = get_qoi_indicator<pde, P>(disc);
  disc.advance_time(4);
  double tol = (std::is_same_v<P, double>) ? 1.E-8 : 1.E-5;
  tassert(std::abs(ienergy - get_qoi_indicator<pde, P>(disc)) < tol);

  disc.save_final_snapshot();
  tassert(std::filesystem::exists("_asg_testfile.h5"));

  auto ropts = make_opts("-restart _asg_testfile.h5");
  discretization_manager<P> rdisc(make_testpde<pde, P>(2, ropts));

  tassert(rdisc.get_grid().num_indexes() == disc.get_grid().num_indexes());
  tassert(std::abs(get_qoi_indicator<pde, P>(disc) - get_qoi_indicator<pde, P>(rdisc)) < 1.E-10);

  tassert(std::abs(rdisc.stop_time() - 1.5625E-2) < 1.E-10); // updated the stop time
  rdisc.advance_time();

  tassert(std::abs(rdisc.time() - 1.5625E-2) < 1.E-10);

  disc.advance_time();
  double constexpr tol2 = (std::is_same_v<P, double>) ? 1.E-14 : 5.E-6;
  tassert(std::abs(get_qoi_indicator<pde, P>(rdisc) - get_qoi_indicator<pde, P>(disc)) < tol2);
}

template<typename P>
void all_templated_tests() {
  check_api_hdf5<P>();

  simple_restart<P>();
  reset_time_params<P>();
  restart_errors<P>();
  restart_longer<P>();
  restart_adapt<P>();
  restart_moments<P>();
}

int main(int argc, char **argv) {

  libasgard_runtime running_(argc, argv);

  all_tests global_("asgard-io-tests", " consistency between HDF5/HighFive/ASGarD");

  #ifdef ASGARD_ENABLE_DOUBLE
  all_templated_tests<double>();
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  all_templated_tests<float>();
  #endif


  return 0;
}
