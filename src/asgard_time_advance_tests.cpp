#include "tests_general.hpp"

using namespace asgard;

int main(int argc, char *argv[])
{
  initialize_distribution();

  int result = Catch::Session().run(argc, argv);

  finalize_distribution();

  return result;
}

static auto const time_advance_base_dir = gold_base_dir / "time_advance";

// NOTE: when using this template the precision is inferred from the type
//       of the tolerance factor, make sure the type of the factor is correct
template<typename P>
void time_advance_test(prog_opts const &opts,
                       std::filesystem::path const &filepath,
                       P const tolerance_factor)
{
  auto const num_ranks = get_num_ranks();
  if (num_ranks > 1 and opts.step_method and opts.step_method.value() == time_method::imp)
  {
    // distributed implicit stepping not implemented
    return;
  }

  prog_opts silent_opts = opts;

  silent_opts.ignore_exact = true;

  discretization_manager<P> disc(make_PDE<P>(silent_opts));

  // -- time loop
  for (auto i : indexof(disc.final_time_step()))
  {
    disc.advance_time(1);

    fk::vector<P> f = disc.current_state();

    auto const file_path =
        filepath.parent_path() /
        (filepath.filename().string() + std::to_string(i) + ".dat");
    auto const gold = read_vector_from_txt_file<P>(file_path);

    // each rank generates partial answer
    int64_t const dof  = fm::ipow(disc.degree() + 1, disc.get_pde().num_dims());
    auto const subgrid = disc.get_grid().get_subgrid(get_rank());
    REQUIRE((subgrid.col_stop + 1) * dof - 1 <= gold.size());
    auto const my_gold = fk::vector<P, mem_type::const_view>(
        gold, subgrid.col_start * dof, (subgrid.col_stop + 1) * dof - 1);
    rmse_comparison(my_gold, f, tolerance_factor);
  }
}

std::string get_level_string(std::vector<int> const &levels)
{
  std::string s = "";
  for (auto l : levels)
    s += std::to_string(l) + "_";
  return s;
}

TEST_CASE("adaptive time advance")
{
  if (!is_active() || get_num_ranks() == 2 || get_num_ranks() == 3)
  {
    return;
  }

  SECTION("fokkerplanck1_pitch_E case1 explicit")
  {
    auto constexpr tol_factor = get_tolerance<default_precision>(100);

    auto const gold_base =
        time_advance_base_dir / "fokkerplanck1_4p1a_ad_sg_l4_d4_t";

    auto opts = make_opts("-p fokkerplanck_1d_pitch_E_case1 -d 3 -l 4 -n 5 -s expl -a 1.E-4");

    // we do not gracefully handle coarsening below number of active ranks yet
    if (get_num_ranks() == 1)
    {
      time_advance_test(opts, gold_base, tol_factor);
    }
  }

  SECTION("fokkerplanck1_pitch_E case2 explicit")
  {
    auto const tol_factor = get_tolerance<default_precision>(10);

    auto const gold_base =
        time_advance_base_dir / "fokkerplanck1_pitch_E_case2_ad_sg_l4_d4_t";

    auto opts = make_opts("-p fokkerplanck_1d_pitch_E_case2 -d 3 -l 4 -n 5 -m 8 -a 1.E-4");

    // we do not gracefully handle coarsening below number of active ranks yet
    if (get_num_ranks() == 1)
    {
      time_advance_test(opts, gold_base, tol_factor);
    }
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_pitch_C", "[time_advance]",
                   test_precs)
{
  if (!is_active())
  {
    return;
  }

  auto constexpr tol_factor = get_tolerance<TestType>(200);

  SECTION("fokkerplanck_1d_pitch_C, level 2, degree 1, sparse grid")
  {
    auto const gold_base =
        time_advance_base_dir / "fokkerplanck1_4p2_sg_l2_d2_t";

    auto opts = make_opts("-p fokkerplanck_1d_pitch_C -d 1 -l 2 -n 5");

    time_advance_test(opts, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_4p3", "[time_advance]",
                   test_precs)
{
  if (!is_active())
  {
    return;
  }

  SECTION("fokkerplanck_1d_4p3, level 2, degree 1, sparse grid")
  {
    auto constexpr tol_factor = get_tolerance<TestType>(10);

    auto const gold_base =
        time_advance_base_dir / "fokkerplanck1_4p3_sg_l2_d2_t";

    auto opts = make_opts("-p fokkerplanck_1d_4p3 -d 1 -l 2 -n 5");

    time_advance_test(opts, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_pitch_E_case1",
                   "[time_advance]", test_precs)
{
  if (!is_active())
  {
    return;
  }

  auto constexpr tol_factor = get_tolerance<TestType>(100);

  SECTION("fokkerplanck_1d_pitch_E_case1, level 2, degree 1, sparse grid")
  {
    auto const gold_base =
        time_advance_base_dir / "fokkerplanck1_4p1a_sg_l2_d2_t";

    auto opts = make_opts("-p fokkerplanck_1d_pitch_E_case1 -d 1 -l 2 -n 5");

    time_advance_test(opts, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_pitch_E_case2",
                   "[time_advance]", test_precs)
{
  if (!is_active())
  {
    return;
  }

  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("fokkerplanck_1d_pitch_E_case2, level 2, degree 1, sparse grid")
  {
    auto const gold_base =
        time_advance_base_dir / "fokkerplanck1_pitch_E_case2_sg_l2_d2_t";

    auto opts = make_opts("-p fokkerplanck_1d_pitch_E_case2 -d 1 -l 2 -n 5");

    time_advance_test(opts, gold_base, tol_factor);
  }
}

// explicit time advance is not a fruitful approach to this problem
TEMPLATE_TEST_CASE("implicit time advance - fokkerplanck_2d_complete_case4",
                   "[time_advance]", test_precs)
{
  if (!is_active() || get_num_ranks() == 2 || get_num_ranks() == 3)
  {
    return;
  }

  std::string pde_choice = "fokkerplanck_2d_complete_case4";

  SECTION("fokkerplanck_2d_complete_case4, level 3, degree 2, sparse grid")
  {
    auto constexpr tol_factor = get_tolerance<TestType>(1e5);

    auto const gold_base =
        time_advance_base_dir / "fokkerplanck2_complete_implicit_sg_l3_d3_t";

    auto opts = make_opts("-p fokkerplanck_2d_complete_case4 -d 2 -l 3 -n 5 -s impl -sv direct");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("fokkerplanck_2d_complete_case4, level 4, degree 2, sparse grid")
  {
    auto constexpr tol_factor = get_tolerance<TestType>(1e5);

    auto const gold_base =
        time_advance_base_dir / "fokkerplanck2_complete_implicit_sg_l4_d3_t";

    auto opts = make_opts("-p fokkerplanck_2d_complete_case4 -d 2 -l 4 -n 5 -s impl -sv direct");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("fokkerplanck_2d_complete_case4, level 5, degree 2, sparse grid")
  {
    auto constexpr tol_factor = get_tolerance<TestType>(1e5);

    auto const gold_base =
        time_advance_base_dir / "fokkerplanck2_complete_implicit_sg_l5_d3_t";

    auto opts = make_opts("-p fokkerplanck_2d_complete_case4 -d 2 -l 5 -n 5 -s impl -sv direct");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION(
      "fokkerplanck_2d_complete_case4, implicit/non-uniform level, degree 2, "
      "sparse grid")
  {
    std::vector<int> const levels{2, 3};
    auto constexpr tol_factor = get_tolerance<TestType>(1e5);

    auto const gold_base =
        time_advance_base_dir / ("fokkerplanck2_complete_implicit_sg_l" +
                                 get_level_string(levels) + "d3_t");

    auto opts = make_opts("-p fokkerplanck_2d_complete_case4 -d 2 -n 5 -s impl -sv direct");

    opts.start_levels = levels;

    time_advance_test(opts, gold_base, tol_factor);
  }
}
