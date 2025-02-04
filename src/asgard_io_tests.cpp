#include "tests_general.hpp"

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

TEMPLATE_TEST_CASE("highfive interface to HDF5", "[io]", test_precs, int)
{
  SECTION("check it writes the correct file")
  {
    std::string const output_file_name("asgard_test.h5");
    std::string const output_dataset_name("asgard_test");

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
    REQUIRE(vec1.size() == gold_vec1.size());
    REQUIRE(vec2.size() == gold_vec2.size());

    for (int i = 0; i < static_cast<int>(vec1.size()); i++)
    {
      REQUIRE(vec1[i] == gold_vec1[i]);
      REQUIRE(vec2[i] == gold_vec2[i]);
    }
  }
}

TEMPLATE_TEST_CASE("save/restart logic", "[io]", test_precs)
{
  TestType constexpr tol = (std::is_same_v<TestType, double>) ? 1.E-14 : 1.E-5;

  std::string const filename = (std::is_same_v<TestType, double>)
      ? "_asgard_dsave_test1.h5" : "_asgard_fsave_test1.h5";

  SECTION("simple restart")
  {
    int const num_dims = 4;
    std::string const title    = "basic io test";
    std::string const subtitle = "test 1";

    prog_opts options = make_opts("-d 3 -l 3 -m 4 -dt 0.5 -time 1.0");
    options.title    = title;
    options.subtitle = subtitle;
    pde_domain<TestType> domain(num_dims);
    discretization_manager<TestType> ref(PDEv2<TestType>(options, domain));

    ref.save_snapshot2(filename);

    prog_opts opts2 = make_opts("-restart " + filename);
    discretization_manager<TestType> disc(PDEv2<TestType>(opts2, domain));

    REQUIRE(ref.get_pde2().num_dims() == num_dims);
    REQUIRE(disc.get_pde2().num_dims() == num_dims);

    REQUIRE(disc.get_pde2().num_dims() == num_dims);
    REQUIRE(disc.get_pde2().num_dims() == num_dims);

    REQUIRE(disc.get_pde2().options().title == title);
    REQUIRE(disc.get_pde2().options().subtitle == subtitle);

    REQUIRE(disc.degree() == 3);

    REQUIRE(std::abs(ref.time_props().dt() - disc.time_props().dt()) < tol);
    REQUIRE(std::abs(ref.time_props().time() - disc.time_props().time()) < tol);
    REQUIRE(std::abs(ref.time_props().stop_time() - disc.time_props().stop_time()) < tol);
    REQUIRE(std::abs(ref.time_props().num_remain() - disc.time_props().num_remain()) < tol);

    REQUIRE(ref.get_sgrid().num_indexes() == disc.get_sgrid().num_indexes());
    REQUIRE(ref.get_sgrid().num_dims() == disc.get_sgrid().num_dims());
    REQUIRE(ref.get_sgrid().generation() == disc.get_sgrid().generation());
    {
      int const *g1 = ref.get_sgrid()[0];
      int const *g2 = disc.get_sgrid()[0];
      int64_t const num = ref.get_sgrid().num_indexes() * ref.get_sgrid().num_dims();
      int max_index_error = 0;
      for (int64_t i = 0; i < num; i++)
        max_index_error = std::max(max_index_error, std::abs(g1[i] - g2[i]));
      REQUIRE(max_index_error == 0);

      auto const &grid = disc.get_sgrid();
      for (int d : iindexof(num_dims)) {
        REQUIRE(grid.current_level(d) == 3);
        REQUIRE(grid.max_index(d) == 16);
      }
    }
  }

  SECTION("reset time parameters")
  {
    int const num_dims = 1;
    std::string const title    = "restart changes the time parameters";
    std::string const subtitle = "test 2";

    prog_opts options = make_opts("-d 3 -l 3 -m 4 -dt 0.5 -time 3.0 -a 0.0625");
    options.title    = title;
    options.subtitle = subtitle;
    pde_domain<TestType> domain(num_dims);
    discretization_manager<TestType> ref(PDEv2<TestType>(options, domain));
    ref.set_time(TestType{2});
    REQUIRE(ref.time_props().time() == 2);
    REQUIRE(ref.time_props().stop_time() == 3);
    REQUIRE(ref.time_props().num_remain() == 6);

    ref.save_snapshot2(filename);

    prog_opts opts2 = make_opts("-restart " + filename + " -time 4");
    discretization_manager<TestType> d1(PDEv2<TestType>(opts2, domain));
    REQUIRE(d1.time_params().time() == 2);
    REQUIRE(d1.time_params().stop_time() == 4);
    REQUIRE(d1.get_pde2().options().adapt_threshold);
    REQUIRE(d1.get_pde2().options().adapt_threshold.value() == 0.0625);

    opts2 = make_opts("-restart " + filename + " -dt 0.25 -a 0.125");
    discretization_manager<TestType> d2(PDEv2<TestType>(opts2, domain));
    REQUIRE(d2.time_params().dt() == TestType{0.25});
    // stop time minus current time is 1, with dt = 0.25 we have 4 steps
    REQUIRE(d2.time_params().num_remain() == 4);
    REQUIRE(d2.get_pde2().options().adapt_threshold.value() == 0.125);

    opts2 = make_opts("-restart " + filename + " -n 8 -noa");
    discretization_manager<TestType> d3(PDEv2<TestType>(opts2, domain));
    REQUIRE(d3.time_params().num_remain() == 8);
    // stop time minus current time is 1, with 8 streps, we have dt = 0.25
    REQUIRE(d3.time_params().dt() == TestType{0.125});
    REQUIRE_FALSE(d3.get_pde2().options().adapt_threshold);
  }

  SECTION("error handling during resrat")
  {
    int const num_dims = 2;
    std::string const title    = "restart errors";
    std::string const subtitle = "test 3";

    prog_opts options = make_opts("-d 3 -l 3 -m 4 -dt 0.5 -time 3.0");
    options.title    = title;
    options.subtitle = subtitle;
    pde_domain<TestType> domain(num_dims);
    discretization_manager<TestType> ref(PDEv2<TestType>(options, domain));
    ref.set_time(TestType{2});

    ref.save_snapshot2(filename);

    // try to restart from a missing file
    prog_opts opts2 = make_opts("-restart wrong_file");
    REQUIRE_THROWS_WITH(discretization_manager<TestType>(PDEv2<TestType>(opts2, domain)),
                        "Cannot find file: 'wrong_file'");

    // the file is correct, but the dimensions are wrong
    opts2 = make_opts("-restart " + filename);
    REQUIRE_THROWS_WITH(discretization_manager<TestType>(
                           PDEv2<TestType>(opts2, pde_domain<TestType>(num_dims + 1))),
                        "Mismatch in the number of dimensions, pde is set for '3' "
                        "but the file contains data for '2'. "
                        "The restart file must match the dimensions.");

    // dimension is correct but there are too many time parameters
    opts2 = make_opts("-restart " + filename + " -dt 0.5 -time 1.0 -n 20");
    REQUIRE_THROWS_WITH(discretization_manager<TestType>(PDEv2<TestType>(opts2, domain)),
                        "cannot simultaneously specify -dt, -num-steps, and -time");

    // setting end time before the current time
    opts2 = make_opts("-restart " + filename + " -dt 0.5 -time 1.0");
    REQUIRE_THROWS_WITH(discretization_manager<TestType>(PDEv2<TestType>(opts2, domain)),
                        "cannot reset the final time to an instance before the current time");
  }
}

TEMPLATE_TEST_CASE("save/restart logic (longer)", "[io]", test_precs)
{
  using P = TestType;
  SECTION("basic restart")
  {
    using pde = pde_contcos;

    // 1. make a pde and set 4 time-steps, advance in time and save the state
    //    - check initial and final error, and file-existing
    // 2. restart and set 4 additional time-steps, advance in time
    //    - make sure restarted matches the saved and new end-time is set
    // 3. compare against a one-shot integration using 4 steps

    auto options = make_opts("-l 5 -d 2 -dt 0.01 -n 4 -of _asg_testfile.h5");
    discretization_manager<P> disc(make_testpde<pde, P>(2, options));
    REQUIRE(get_qoi_indicator<pde, P>(disc) < 1.E-2);
    disc.advance_time();
    REQUIRE(get_qoi_indicator<pde, P>(disc) < 1.E-2);

    disc.save_final_snapshot();
    REQUIRE(std::filesystem::exists("_asg_testfile.h5"));

    auto ropts = make_opts("-n 4 -dt 0.01 -restart _asg_testfile.h5");
    discretization_manager<P> rdisc(make_testpde<pde, P>(2, ropts));

    REQUIRE(std::abs(get_qoi_indicator<pde, P>(disc) - get_qoi_indicator<pde, P>(rdisc)) < 1.E-10);

    REQUIRE(std::abs(rdisc.time_params().stop_time() - 0.08) < 2.E-9); // updated the stop time
    rdisc.advance_time();

    REQUIRE(std::abs(rdisc.time_params().time() - 0.08) < 1.E-8);

    options = make_opts("-l 5 -d 2 -dt 0.01 -n 8");
    discretization_manager<P> reff(make_testpde<pde, P>(2, options));
    reff.advance_time();

    REQUIRE(std::abs(reff.time_params().time() - 0.08) < 1.E-8);

    REQUIRE(std::abs(get_qoi_indicator<pde, P>(reff) - get_qoi_indicator<pde, P>(rdisc)) < 1.E-10);
  }

  SECTION("adaptive restart")
  {
    using pde = pde_contcos;

    // similar to above but uses adaptivity

    auto options = make_opts("-l 8 -d 2 -dt 0.01 -n 8 -a 1.E-2 -of _asg_testfile.h5");
    discretization_manager<P> disc(make_testpde<pde, P>(2, options));
    REQUIRE(get_qoi_indicator<pde, P>(disc) < 1.E-2);
    disc.advance_time(4);
    REQUIRE(get_qoi_indicator<pde, P>(disc) < 1.E-2);

    disc.save_final_snapshot();
    REQUIRE(std::filesystem::exists("_asg_testfile.h5"));

    auto ropts = make_opts("-restart _asg_testfile.h5");
    discretization_manager<P> rdisc(make_testpde<pde, P>(2, ropts));

    REQUIRE(rdisc.get_sgrid().num_indexes() == disc.get_sgrid().num_indexes());
    REQUIRE(std::abs(get_qoi_indicator<pde, P>(disc) - get_qoi_indicator<pde, P>(rdisc)) < 1.E-10);

    REQUIRE(std::abs(rdisc.time_params().stop_time() - 0.08) < 2.E-9); // updated the stop time
    rdisc.advance_time();

    REQUIRE(std::abs(rdisc.time_params().time() - 0.08) < 1.E-8);

    options = make_opts("-l 8 -d 2 -dt 0.01 -n 8 -a 1.E-2");
    discretization_manager<P> reff(make_testpde<pde, P>(2, options));
    reff.advance_time();

    REQUIRE(rdisc.get_sgrid().num_indexes() == reff.get_sgrid().num_indexes());

    REQUIRE(std::abs(reff.time_params().time() - 0.08) < 1.E-8);

    REQUIRE(std::abs(get_qoi_indicator<pde, P>(reff) - get_qoi_indicator<pde, P>(rdisc)) < 1.E-10);
  }

  SECTION("moments restart")
  {
    using pde = pde_twostream;

    // similar to above but uses adaptivity

    auto options = make_opts("-l 7 -d 2 -dt 1.953125E-3 -n 8 -a 1.E-6 -of _asg_testfile.h5");
    discretization_manager<P> disc(make_testpde<pde, P>(2, options));
    double const ienergy = get_qoi_indicator<pde, P>(disc);
    disc.advance_time(4);
    double tol = (std::is_same_v<P, double>) ? 1.E-8 : 1.E-5;
    REQUIRE(std::abs(ienergy - get_qoi_indicator<pde, P>(disc)) < tol);

    disc.save_final_snapshot();
    REQUIRE(std::filesystem::exists("_asg_testfile.h5"));

    auto ropts = make_opts("-restart _asg_testfile.h5");
    discretization_manager<P> rdisc(make_testpde<pde, P>(2, ropts));

    REQUIRE(rdisc.get_sgrid().num_indexes() == disc.get_sgrid().num_indexes());
    REQUIRE(std::abs(get_qoi_indicator<pde, P>(disc) - get_qoi_indicator<pde, P>(rdisc)) < 1.E-10);

    REQUIRE(std::abs(rdisc.time_params().stop_time() - 1.5625E-2) < 1.E-10); // updated the stop time
    rdisc.advance_time();

    REQUIRE(std::abs(rdisc.time_params().time() - 1.5625E-2) < 1.E-10);

    disc.advance_time();
    REQUIRE(std::abs(get_qoi_indicator<pde, P>(rdisc) - get_qoi_indicator<pde, P>(disc)) < 1.E-8);
  }
}
