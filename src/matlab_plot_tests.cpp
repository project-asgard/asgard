#include "matlab_plot.hpp"

#include "tests_general.hpp"
#include <string_view>

static auto const matlab_plot_base_dir = gold_base_dir / "matlab_plot";

const static std::string ml_plot_tag("[matlab_plot]");

using namespace asgard;

TEST_CASE("create matlab session", ml_plot_tag)
{
  ml::matlab_plot &ml_plot = ml::matlab_plot::get_instance();

  SECTION("using sync connect") { REQUIRE_NOTHROW(ml_plot.connect()); }

  SECTION("starting new with no cmd line opts")
  {
    REQUIRE_NOTHROW(ml_plot.start());
  }

  REQUIRE(ml_plot.is_open());

  ml_plot.close();
}

static ml::matlab_plot *ml_plot = &ml::matlab_plot::get_instance();

ml::matlab_plot *get_session(ml::matlab_plot *instance)
{
  if (!instance)
  {
    instance = &ml::matlab_plot::get_instance();
  }
  if (!instance->is_open())
  {
    instance->start();
  }
  return instance;
}

TEST_CASE("creating scalar params", ml_plot_tag)
{
  ml_plot = get_session(ml_plot);

  SECTION("string")
  {
    std::string test("test string");
    REQUIRE_NOTHROW(ml_plot->add_param(test));
  }

  SECTION("integer")
  {
    int test = 25;
    REQUIRE_NOTHROW(ml_plot->add_param(test));
  }

  ml_plot->reset_params();
}

TEMPLATE_TEST_CASE("creating vector params", ml_plot_tag, float, double)
{
  ml_plot = get_session(ml_plot);

  SECTION("fk::vector")
  {
    fk::vector<TestType> testvec(10);
    std::iota(testvec.begin(), testvec.end(), 1.0);

    REQUIRE_NOTHROW(
        ml_plot->add_param({1, static_cast<size_t>(testvec.size())}, testvec));
  }

  ml_plot->reset_params();
}

TEST_CASE("generate plotting nodes", ml_plot_tag)
{
  std::vector<double> const min{-1.0, -2.0, -3.0};
  std::vector<double> const max{1.0, 2.0, 3.0};

  ml_plot = get_session(ml_plot);

  SECTION("deg = 2, lev = 2")
  {
    std::string const gold_file = "nodes_continuity2_d2_l2_";

    for (int dim = 0; dim < 2; dim++)
    {
      fk::vector<double> gold = read_matrix_from_txt_file<double>(
          matlab_plot_base_dir / (gold_file + std::to_string(dim) + ".dat"));
      fk::vector<double> nodes =
          ml_plot->generate_nodes(2, 2, min[dim], max[dim]);

      REQUIRE(nodes == gold);
    }
  }

  SECTION("deg = 3, lev = 3")
  {
    std::string const gold_file = "nodes_continuity3_d3_l3_";

    for (int dim = 0; dim < 3; dim++)
    {
      fk::vector<double> gold = read_matrix_from_txt_file<double>(
          matlab_plot_base_dir / (gold_file + std::to_string(dim) + ".dat"));
      fk::vector<double> nodes =
          ml_plot->generate_nodes(3, 3, min[dim], max[dim]);

      REQUIRE(nodes == gold);
    }
  }
}

void test_element_coords(PDE_opts const pde_name, int const level,
                         int const degree, std::string const gold_file,
                         bool const fullgrid)
{
  auto const pde                = make_PDE<double>(pde_name, level, degree);
  std::vector<std::string> opts = {"-l", std::to_string(level), "-d",
                                   std::to_string(degree + 1)};
  if (fullgrid)
  {
    opts.push_back("-f");
  }

  elements::table const table(make_options(opts), *pde);

  fk::vector<double> gold           = read_matrix_from_txt_file<double>(gold_file);
  fk::vector<double> element_coords = ml_plot->gen_elem_coords(*pde, table);

  REQUIRE(element_coords == gold);
}

TEST_CASE("generate element coords for plotting", ml_plot_tag)
{
  ml_plot = get_session(ml_plot);

  SECTION("continuity2d, SG")
  {
    int const level      = 2;
    int const degree     = 2;
    auto const gold_file = matlab_plot_base_dir / "elements_2d_l2_d3_SG.dat";

    test_element_coords(PDE_opts::continuity_2, level, degree, gold_file,
                        false);
  }

  SECTION("continuity2d, FG")
  {
    int const level      = 2;
    int const degree     = 2;
    auto const gold_file = matlab_plot_base_dir / "elements_2d_l2_d3_FG.dat";

    test_element_coords(PDE_opts::continuity_2, level, degree, gold_file, true);
  }

  SECTION("continuity3d, SG")
  {
    int const level      = 2;
    int const degree     = 2;
    auto const gold_file = matlab_plot_base_dir / "elements_3d_l2_d3_SG.dat";

    test_element_coords(PDE_opts::continuity_3, level, degree, gold_file,
                        false);
  }

  SECTION("continuity3d, FG")
  {
    int const level      = 2;
    int const degree     = 2;
    auto const gold_file = matlab_plot_base_dir / "elements_3d_l2_d3_FG.dat";

    test_element_coords(PDE_opts::continuity_3, level, degree, gold_file, true);
  }
}

// This might be a problem if test ordering is not guaranteed..
TEST_CASE("close session")
{
  ml_plot = get_session(ml_plot);

  REQUIRE(ml_plot->is_open());

  REQUIRE_NOTHROW(ml_plot->close());
}
