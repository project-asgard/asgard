#include "matlab_plot.hpp"

#include "tests_general.hpp"
#include <string_view>

const static std::string ml_plot_tag("[matlab_plot]");

TEST_CASE("create matlab session", ml_plot_tag)
{
  matlab_plot ml;

  SECTION("using sync connect") { REQUIRE_NOTHROW(ml.connect()); }

  SECTION("starting new with no cmd line opts") { REQUIRE_NOTHROW(ml.start()); }

  REQUIRE(ml.is_open());

  ml.close();
}

static matlab_plot *ml = nullptr;

matlab_plot *get_session(matlab_plot *instance)
{
  if (!instance)
  {
    instance = new matlab_plot();
  }
  if (!instance->is_open())
  {
    instance->start();
  }
  return instance;
}

TEST_CASE("creating scalar params", ml_plot_tag)
{
  ml = get_session(ml);

  SECTION("string")
  {
    std::string test("test string");
    REQUIRE_NOTHROW(ml->add_param(test));
  }

  SECTION("integer")
  {
    int test = 25;
    REQUIRE_NOTHROW(ml->add_param(test));
  }

  ml->reset_params();
}

TEMPLATE_TEST_CASE("creating vector params", ml_plot_tag, float, double)
{
  SECTION("fk::vector") {}

  SECTION("fk::matrix") {}
}

TEST_CASE("generate plotting nodes", ml_plot_tag)
{
  std::string const gold_base = "../testing/generated-inputs/matlab_plot/";
  std::vector<double> const min{-1.0, -2.0, -3.0};
  std::vector<double> const max{1.0, 2.0, 3.0};

  ml = get_session(ml);

  SECTION("deg = 2, lev = 2")
  {
    std::string const gold_file = gold_base + "nodes_continuity2_d2_l2_";

    for (int dim = 0; dim < 2; dim++)
    {
      fk::vector<double> gold =
          read_matrix_from_txt_file(gold_file + std::to_string(dim) + ".dat");
      fk::vector<double> nodes = ml->generate_nodes(2, 2, min[dim], max[dim]);

      REQUIRE(nodes == gold);
    }
  }

  SECTION("deg = 3, lev = 3")
  {
    std::string const gold_file = gold_base + "nodes_continuity3_d3_l3_";

    for (int dim = 0; dim < 3; dim++)
    {
      fk::vector<double> gold =
          read_matrix_from_txt_file(gold_file + std::to_string(dim) + ".dat");
      fk::vector<double> nodes = ml->generate_nodes(3, 3, min[dim], max[dim]);

      REQUIRE(nodes == gold);
    }
  }
}

TEST_CASE("generate element coords for plotting", ml_plot_tag)
{
  std::string const gold_base = "../testing/generated-inputs/matlab_plot/";

  ml = get_session(ml);

  SECTION("continuity2d, SG")
  {
    int const level             = 2;
    int const degree            = 3;
    std::string const gold_file = gold_base + "elements_2d_l2_d3_SG.dat";

    auto const pde = make_PDE<double>(PDE_opts::continuity_2, level, degree);
    elements::table const table(make_options({"-l", std::to_string(level), "-d",
                                              std::to_string(degree)}),
                                *pde);

    fk::vector<double> gold           = read_matrix_from_txt_file(gold_file);
    fk::vector<double> element_coords = ml->gen_elem_coords(*pde, table);

    REQUIRE(element_coords == gold);
  }

  SECTION("continuity2d, FG")
  {
    int const level             = 2;
    int const degree            = 3;
    std::string const gold_file = gold_base + "elements_2d_l2_d3_FG.dat";

    auto const pde = make_PDE<double>(PDE_opts::continuity_2, level, degree);
    elements::table const table(make_options({"-l", std::to_string(level), "-d",
                                              std::to_string(degree), "-f"}),
                                *pde);

    fk::vector<double> gold           = read_matrix_from_txt_file(gold_file);
    fk::vector<double> element_coords = ml->gen_elem_coords(*pde, table);

    REQUIRE(element_coords == gold);
  }

  SECTION("continuity3d, SG")
  {
    int const level             = 2;
    int const degree            = 3;
    std::string const gold_file = gold_base + "elements_3d_l2_d3_SG.dat";

    auto const pde = make_PDE<double>(PDE_opts::continuity_2, level, degree);
    elements::table const table(make_options({"-l", std::to_string(level), "-d",
                                              std::to_string(degree)}),
                                *pde);

    fk::vector<double> gold           = read_matrix_from_txt_file(gold_file);
    fk::vector<double> element_coords = ml->gen_elem_coords(*pde, table);

    REQUIRE(element_coords == gold);
  }

  SECTION("continuity3d, FG")
  {
    int const level             = 2;
    int const degree            = 3;
    std::string const gold_file = gold_base + "elements_3d_l2_d3_FG.dat";

    auto const pde = make_PDE<double>(PDE_opts::continuity_2, level, degree);
    elements::table const table(make_options({"-l", std::to_string(level), "-d",
                                              std::to_string(degree), "-f"}),
                                *pde);

    fk::vector<double> gold           = read_matrix_from_txt_file(gold_file);
    fk::vector<double> element_coords = ml->gen_elem_coords(*pde, table);

    REQUIRE(element_coords == gold);
  }
}

// This might be a problem if test ordering is not guaranteed..
TEST_CASE("close session")
{
  ml = get_session(ml);

  REQUIRE(ml->is_open());

  REQUIRE_NOTHROW(ml->close());
}