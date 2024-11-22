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
  if (num_ranks > 1 and opts.step_method and opts.step_method.value() == time_advance::method::imp)
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
    advance_time(disc, 1);

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

TEMPLATE_TEST_CASE("time advance - diffusion 2", "[time_advance]", test_precs)
{
  if (!is_active())
  {
    return;
  }

  SECTION("diffusion2, explicit, sparse grid, level 2, degree 1")
  {
    auto constexpr tol_factor = get_tolerance<TestType>(100);

    auto const gold_base = time_advance_base_dir / "diffusion2_sg_l2_d2_t";

    auto opts = make_opts("-p diffusion_2 -d 1 -l 2 -n 5");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("diffusion2, explicit, sparse grid, level 3, degree 2")
  {
    auto constexpr tol_factor = get_tolerance<TestType>(100);

    auto const gold_base = time_advance_base_dir / "diffusion2_sg_l3_d3_t";

    auto opts = make_opts("-p diffusion_2 -d 2 -l 3 -n 5");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("diffusion2, explicit, sparse grid, level 4, degree 3")
  {
    auto constexpr tol_factor = get_tolerance<TestType>(1000000);

    auto const gold_base = time_advance_base_dir / "diffusion2_sg_l4_d4_t";

    auto opts = make_opts("-p diffusion_2 -d 3 -l 4 -n 5");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("diffusion2, explicit/non-uniform level, sparse grid, degree 1")
  {
    auto constexpr tol_factor = get_tolerance<TestType>(100);

    std::vector<int> const levels{4, 5};
    auto const gold_base =
        time_advance_base_dir /
        ("diffusion2_sg_l" + get_level_string(levels) + "d2_t");

    auto opts = make_opts("-p diffusion_2 -d 1 -n 5");

    opts.start_levels = levels;
    time_advance_test(opts, gold_base, tol_factor);
  }
}

TEST_CASE("adaptive time advance")
{
  if (!is_active() || get_num_ranks() == 2 || get_num_ranks() == 3)
  {
    return;
  }

  SECTION("diffusion 2 implicit")
  {
    auto const tol_factor = get_tolerance<default_precision>(1000);

    auto const gold_base =
        time_advance_base_dir / "diffusion2_ad_implicit_sg_l3_d4_t";

    auto opts = make_opts("-p diffusion_2 -d 3 -l 3 -n 5 -s impl -a 0.05");

    // temporarily disable test for MPI due to table elements < num ranks
    if (get_num_ranks() == 1)
    {
      time_advance_test(opts, gold_base, tol_factor);
    }
  }
  SECTION("diffusion 2 explicit")
  {
    auto const tol_factor = get_tolerance<default_precision>(1000);

    auto const gold_base = time_advance_base_dir / "diffusion2_ad_sg_l3_d4_t";

    auto opts = make_opts("-p diffusion_2 -d 3 -l 3 -n 5 -s expl -a 0.05");

    // temporarily disable test for MPI due to table elements < num ranks
    if (get_num_ranks() == 1)
    {
      time_advance_test(opts, gold_base, tol_factor);
    }
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

  SECTION("continuity 2 explicit")
  {
    auto const tol_factor = get_tolerance<default_precision>(100);

    auto const gold_base = time_advance_base_dir / "continuity2_ad_sg_l3_d4_t";

    auto opts = make_opts("-p continuity_2 -d 3 -l 3 -n 5 -m 8 -s expl -a 1.E-3");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("continuity 2 explicit, l2 norm")
  {
    // Gold data was calculated with L^{\infty} norm
    auto const tol_factor = static_cast<default_precision>(0.00001);

    auto const gold_base = time_advance_base_dir / "continuity2_ad_sg_l3_d4_t";

    auto opts = make_opts("-p continuity_2 -d 3 -l 3 -n 5 -m 8 -s expl -a 1.E-3");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("continuity 2 explicit")
  {
    auto const tol_factor = get_tolerance<default_precision>(100);

    auto const gold_base = time_advance_base_dir / "continuity2_ad_sg_l3_d4_t";

    auto opts = make_opts("-p continuity_2 -d 3 -l 3 -n 5 -a 1.E-3");

    opts.max_levels = {6, 8};

    time_advance_test(opts, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - diffusion 1", "[time_advance]", test_precs)
{
  if (!is_active())
  {
    return;
  }

  SECTION("diffusion1, explicit, sparse grid, level 3, degree 2")
  {
    auto const gold_base      = time_advance_base_dir / "diffusion1_sg_l3_d3_t";
    auto constexpr tol_factor = get_tolerance<TestType>(100);

    auto opts = make_opts("-p diffusion_1 -d 2 -l 3 -n 5");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("diffusion1, explicit, sparse grid, level 4, degree 3")
  {
    auto const gold_base  = time_advance_base_dir / "diffusion1_sg_l4_d4_t";
    auto const tol_factor = get_tolerance<TestType>(100000);

    auto opts = make_opts("-p diffusion_1 -d 3 -l 4 -n 5");

    time_advance_test(opts, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 1", "[time_advance]", test_precs)
{
  if (!is_active())
  {
    return;
  }

  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("continuity1, explicit, level 2, degree 1, sparse grid")
  {
    auto const gold_base = time_advance_base_dir / "continuity1_sg_l2_d2_t";

    auto opts = make_opts("-p continuity_1 -d 1 -l 2 -n 5");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("continuity1, explicit, level 2, degree 1, full grid")
  {
    auto const gold_base = time_advance_base_dir / "continuity1_fg_l2_d2_t";

    auto opts = make_opts("-p continuity_1 -d 1 -l 2 -n 5 -g dense");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("continuity1, explicit, level 4, degree 2, sparse grid")
  {
    auto const gold_base = time_advance_base_dir / "continuity1_sg_l4_d3_t";

    auto opts = make_opts("-p continuity_1 -d 2 -l 4 -n 5");

    time_advance_test(opts, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 2", "[time_advance]", test_precs)
{
  if (!is_active())
  {
    return;
  }

  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("continuity2, explicit, level 2, degree 1, sparse grid")
  {
    auto const gold_base = time_advance_base_dir / "continuity2_sg_l2_d2_t";

    auto opts = make_opts("-p continuity_2 -d 1 -l 2 -n 5");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("continuity2, explicit, level 2, degree 1, full grid")
  {
    auto const gold_base = time_advance_base_dir / "continuity2_fg_l2_d2_t";

    auto opts = make_opts("-p continuity_2 -d 1 -l 2 -n 5 -g dense");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("continuity2, explicit, level 4, degree 2, sparse grid")
  {
    auto const gold_base = time_advance_base_dir / "continuity2_sg_l4_d3_t";

    auto opts = make_opts("-p continuity_2 -d 2 -l 4 -n 5");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("continuity2, explicit/non-uniform level, full grid, degree 2")
  {
    std::vector<int> const levels{3, 4};

    auto const gold_base =
        time_advance_base_dir /
        ("continuity2_fg_l" + get_level_string(levels) + "d3_t");

    auto opts = make_opts("-p continuity_2 -d 2 -n 5 -g dense");

    opts.start_levels = levels;

    time_advance_test(opts, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 3", "[time_advance]", test_precs)
{
  if (!is_active())
  {
    return;
  }

  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("continuity3, explicit, level 2, degree 1, sparse grid")
  {
    auto const gold_base = time_advance_base_dir / "continuity3_sg_l2_d2_t";

    auto opts = make_opts("-p continuity_3 -d 1 -l 2 -n 5");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("continuity3, explicit, level 4, degree 2, sparse grid")
  {
    auto const gold_base = time_advance_base_dir / "continuity3_sg_l4_d3_t";

    auto opts = make_opts("-p continuity_3 -d 2 -l 4 -n 5");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("continuity3, explicit/non-uniform level, degree 3, sparse grid")
  {
    std::vector<int> const levels{3, 4, 2};

    auto const gold_base =
        time_advance_base_dir /
        ("continuity3_sg_l" + get_level_string(levels) + "d4_t");

    auto opts = make_opts("-p continuity_3 -d 3 -n 5");

    opts.start_levels = levels;

    time_advance_test(opts, gold_base, get_tolerance<TestType>(10));
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 6", "[time_advance]", test_precs)
{
  if (!is_active())
  {
    return;
  }

  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("continuity6, level 2, degree 2, sparse grid")
  {
    auto const gold_base = time_advance_base_dir / "continuity6_sg_l2_d3_t";

    auto opts = make_opts("-p continuity_6 -d 2 -l 2 -n 5");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("continuity6, explicit/non-uniform level, degree 3, sparse grid")
  {
    std::vector<int> const levels{2, 3, 2, 3, 3, 2};
    auto const gold_base =
        time_advance_base_dir /
        ("continuity6_sg_l" + get_level_string(levels) + "d2_t");

    auto opts = make_opts("-p continuity_6 -d 1 -n 5");

    opts.start_levels = levels;

    time_advance_test(opts, gold_base, tol_factor);
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

TEMPLATE_TEST_CASE("implicit time advance - diffusion 1", "[time_advance]",
                   test_precs)
{
  if (!is_active() || get_num_ranks() == 2 || get_num_ranks() == 3)
  {
    return;
  }

  auto constexpr tol_factor = get_tolerance<TestType>(100);

  SECTION("diffusion1, implicit, sparse grid, level 4, degree 3")
  {
    auto const gold_base =
        time_advance_base_dir / "diffusion1_implicit_sg_l4_d4_t";

    auto opts = make_opts("-p diffusion_1 -d 3 -l 4 -n 5 -s impl -sv direct");

    time_advance_test(opts, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("implicit time advance - diffusion 2", "[time_advance]",
                   test_precs)
{
  if (!is_active() || get_num_ranks() == 2 || get_num_ranks() == 3)
  {
    return;
  }

  auto constexpr tol_factor = get_tolerance<TestType>(100);

  SECTION("diffusion2, implicit, sparse grid, level 3, degree 2")
  {
    auto const gold_base =
        time_advance_base_dir / "diffusion2_implicit_sg_l3_d3_t";

    auto opts = make_opts("-p diffusion_2 -d 2 -l 3 -n 5 -s impl -sv direct");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("diffusion2, implicit, sparse grid, level 4, degree 2")
  {
    auto const gold_base =
        time_advance_base_dir / "diffusion2_implicit_sg_l4_d3_t";

    auto opts = make_opts("-p diffusion_2 -d 2 -l 4 -n 5 -s impl -sv direct");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("diffusion2, implicit, sparse grid, level 5, degree 2")
  {
    auto const gold_base =
        time_advance_base_dir / "diffusion2_implicit_sg_l5_d3_t";

    auto opts = make_opts("-p diffusion_2 -d 2 -l 5 -n 5 -s impl -sv direct");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("diffusion2, implicit/non-uniform level, degree 1, sparse grid")
  {
    std::vector<int> const levels = {4, 5};
    auto const gold_base =
        time_advance_base_dir /
        ("diffusion2_implicit_sg_l" + get_level_string(levels) + "d2_t");

    auto opts = make_opts("-p diffusion_2 -d 1 -n 5 -s impl -sv direct");

    opts.start_levels = levels;

    time_advance_test(opts, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("implicit time advance - continuity 1", "[time_advance]",
                   test_precs)
{
  if (!is_active() || get_num_ranks() == 2 || get_num_ranks() == 3)
  {
    return;
  }

  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("continuity1, level 2, degree 1, sparse grid")
  {
    auto const gold_base =
        time_advance_base_dir / "continuity1_implicit_l2_d2_t";

    auto opts = make_opts("-p continuity_1 -d 1 -l 2 -n 5 -s impl -sv direct");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("continuity1, level 4, degree 2, sparse grid")
  {
    auto const gold_base =
        time_advance_base_dir / "continuity1_implicit_l4_d3_t";

    auto opts = make_opts("-p continuity_1 -d 2 -l 4 -n 5 -s impl -sv direct");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("continuity1, level 4, degree 2, sparse grid, iterative")
  {
    auto const gold_base =
        time_advance_base_dir / "continuity1_implicit_l4_d3_t";

    auto opts = make_opts("-p continuity_1 -d 2 -l 4 -n 5 -s impl -sv direct");

    time_advance_test(opts, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("implicit time advance - continuity 2", "[time_advance]",
                   test_precs)
{
  if (!is_active() || get_num_ranks() == 2 || get_num_ranks() == 3)
  {
    return;
  }

  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("continuity2, level 2, degree 1, sparse grid")
  {
    auto const gold_base =
        time_advance_base_dir / "continuity2_implicit_l2_d2_t";

    auto opts = make_opts("-p continuity_2 -d 1 -l 2 -n 5 -s impl -sv direct");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("continuity2, level 4, degree 2, sparse grid")
  {
    auto const gold_base =
        time_advance_base_dir / "continuity2_implicit_l4_d3_t";
    auto opts = make_opts("-p continuity_2 -d 2 -l 4 -n 5 -s impl -sv direct");

    time_advance_test(opts, gold_base, tol_factor);
  }

  SECTION("continuity2, level 4, degree 2, sparse grid, iterative")
  {
    auto constexpr temp_tol_factor = get_tolerance<TestType>(10);

    auto const continuity2_base_dir =
        time_advance_base_dir / "continuity2_implicit_l4_d3_t";

    auto opts = make_opts("-p continuity_2 -d 2 -l 4 -n 5 -s impl -sv direct");

    time_advance_test(opts, continuity2_base_dir, temp_tol_factor);
  }
  SECTION("continuity2, implicit/non-uniform level, degree 2, full grid")
  {
    std::vector<int> const levels = {3, 4};
    auto const gold_base =
        time_advance_base_dir /
        ("continuity2_implicit_fg_l" + get_level_string(levels) + "d3_t");

    auto opts = make_opts("-p continuity_2 -d 2 -n 5 -s impl -sv direct -g dense");

    opts.start_levels = levels;

    time_advance_test(opts, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("IMEX time advance - landau", "[imex]", test_precs)
{
  // Disable test for MPI - IMEX needs to be tested further with MPI
  if (!is_active() || get_num_ranks() > 1)
  {
    return;
  }

  TestType constexpr gmres_tol =
      std::is_same_v<TestType, double> ? 1.0e-8 : 1.0e-6;
  TestType constexpr tolerance = // error tolerance
      std::is_same_v<TestType, double> ? 1.0e-9 : 1.0e-3;

  auto opts = make_opts("-p landau -d 2 -l 4 -n 100 -s imex -sv gmres -g dense -dt 0.019634954084936");

  opts.isolver_tolerance = gmres_tol;

  discretization_manager disc(make_PDE<TestType>(opts));

  auto const &pde = disc.get_pde();

  TestType E_pot_initial = 0.0;
  TestType E_kin_initial = 0.0;

  // -- time loop
  for (auto i : indexof(disc.final_time_step()))
  {
    advance_time(disc, 1);

    // compute the E potential and kinetic energy
    fk::vector<TestType> E_field_sq(pde.E_field);
    for (auto &e : E_field_sq)
    {
      e = e * e;
    }
    dimension<TestType> const &dim = pde.get_dimensions()[0];

    TestType E_pot = calculate_integral(E_field_sq, dim);
    TestType E_kin = calculate_integral(disc.get_moments()[2].get_realspace_moment(),
                                        dim);
    if (i == 0)
    {
      E_pot_initial = E_pot;
      E_kin_initial = E_kin;
    }

    // calculate the absolute relative total energy
    TestType E_relative =
        std::fabs((E_pot + E_kin) - (E_pot_initial + E_kin_initial));
    REQUIRE(E_relative <= tolerance);
  }

  parameter_manager<TestType>::get_instance().reset();
}

#ifdef ASGARD_ENABLE_DOUBLE
TEMPLATE_TEST_CASE("IMEX time advance - twostream", "[imex]", double)
{
  // Disable test for MPI - IMEX needs to be tested further with MPI
  if (!is_active() || get_num_ranks() > 1)
  {
    return;
  }

  std::string const pde_choice = "two_stream";

  TestType constexpr tolerance =
      std::is_same_v<TestType, double> ? 1.0e-9 : 1.0e-5;

  auto opts = make_opts("-p two_stream -d 2 -l 5 -n 20 -s imex -sv gmres -g dense -dt 6.25e-3");

  discretization_manager disc(make_PDE<TestType>(opts));

  auto const &pde = disc.get_pde();

  TestType E_pot_initial = 0.0;
  TestType E_kin_initial = 0.0;

  // -- time loop
  for (auto i : indexof(disc.final_time_step()))
  {
    advance_time(disc, 1);

    // compute the E potential and kinetic energy
    fk::vector<TestType> E_field_sq(pde.E_field);
    for (auto &e : E_field_sq)
    {
      e = e * e;
    }
    dimension<TestType> const &dim = pde.get_dimensions()[0];

    auto &moments = disc.get_moments();

    TestType E_pot = calculate_integral(E_field_sq, dim);
    TestType E_kin = calculate_integral(moments[2].get_realspace_moment(), dim);
    if (i == 0)
    {
      E_pot_initial = E_pot;
      E_kin_initial = E_kin;
    }

    // calculate the absolute relative total energy
    TestType E_relative =
        std::fabs((E_pot + E_kin) - (E_pot_initial + E_kin_initial));
    // REQUIRE(E_relative <= tolerance);

    // calculate integral of moments
    fk::vector<TestType> mom0 = moments[0].get_realspace_moment();
    fk::vector<TestType> mom1 = moments[1].get_realspace_moment();

    TestType n_total = calculate_integral(fm::scal(TestType{2.0}, mom0), dim);

    fk::vector<TestType> n_times_u(mom0.size());
    for (auto j : indexof(n_times_u))
      n_times_u[j] = mom0[j] * mom1[j];

    TestType nu_total = calculate_integral(n_times_u, dim);

    // n total should be close to 6.28
    REQUIRE((n_total - 6.283185) <= 1.0e-4);

    // n*u total should be 0
    REQUIRE(nu_total <= 1.0e-14);

    // total relative energy change drops and stabilizes around 2.0e-5
    REQUIRE(E_relative <= 5.5e-5);

    if (i > 0 && i < 100)
    {
      // check the initial slight energy decay before it stabilizes
      // Total energy at time step 1:   5.4952938
      // Total energy at time step 100: 5.4952734
      REQUIRE(E_relative >= tolerance);
    }
  }

  parameter_manager<TestType>::get_instance().reset();
}

// MIRO ERROR: This causes a problem!!!!
TEMPLATE_TEST_CASE("IMEX time advance - twostream - ASG", "[imex][adapt]",
                   double)
{
  // Disable test for MPI - IMEX needs to be tested further with MPI
  if (!is_active() || get_num_ranks() > 1)
  {
    return;
  }

  int const degree = 2;
  int const levels = 5;

  TestType constexpr tolerance =
      std::is_same_v<TestType, double> ? 1.0e-9 : 1.0e-5;

  auto opts = make_opts("-p two_stream -d 2 -l 5 -m 5 -n 10 -s imex -dt 6.25e-3 -a 1.0e-6 -an linf");

  discretization_manager disc(make_PDE<TestType>(opts));

  auto const &pde = disc.get_pde();

  TestType E_pot_initial = 0.0;
  TestType E_kin_initial = 0.0;

  // number of DOF for the FG case: ((degree + 1) * 2^level)^2 = 9.216e3
  int const fg_dof = fm::ipow((degree + 1) * fm::ipow2(levels), 2);

  // -- time loop
  for (auto i : indexof(disc.final_time_step()))
  {
    advance_time(disc, 1);

    auto &moments = disc.get_moments();

    // compute the E potential and kinetic energy
    fk::vector<TestType> E_field_sq(pde.E_field);
    for (auto &e : E_field_sq)
    {
      e = e * e;
    }
    dimension<TestType> const &dim = pde.get_dimensions()[0];

    TestType E_pot = calculate_integral(E_field_sq, dim);
    TestType E_kin = calculate_integral(moments[2].get_realspace_moment(), dim);
    if (i == 0)
    {
      E_pot_initial = E_pot;
      E_kin_initial = E_kin;
    }

    // calculate the absolute relative total energy
    TestType E_relative =
        std::fabs((E_pot + E_kin) - (E_pot_initial + E_kin_initial));
    // REQUIRE(E_relative <= tolerance);

    // calculate integral of moments
    fk::vector<TestType> mom0 = moments[0].get_realspace_moment();
    fk::vector<TestType> mom1 = moments[1].get_realspace_moment();

    TestType n_total = calculate_integral(fm::scal(TestType{2.0}, mom0), dim);

    fk::vector<TestType> n_times_u(mom0.size());
    for (int j = 0; j < n_times_u.size(); j++)
    {
      n_times_u[j] = mom0[j] * mom1[j];
    }

    TestType nu_total = calculate_integral(n_times_u, dim);

    // n total should be close to 6.28
    REQUIRE((n_total - 6.283185) <= 1.0e-4);

    // n*u total should be 0
    REQUIRE(nu_total <= 1.0e-14);

    // total relative energy change drops and stabilizes around 2.0e-5
    REQUIRE(E_relative <= 5.5e-5);

    if (i > 0 && i < 100)
    {
      // check the initial slight energy decay before it stabilizes
      // Total energy at time step 1:   5.4952938
      // Total energy at time step 100: 5.4952734
      REQUIRE(E_relative >= tolerance);
    }

    // for this configuration, the DOF of ASG / DOF of FG should be between
    // 60-65%. Testing against 70% is conservative but will capture issues with
    // adaptivity
    REQUIRE(static_cast<TestType>(disc.current_state().size()) / fg_dof <= 0.70);
  }

  parameter_manager<TestType>::get_instance().reset();
}
#endif

TEMPLATE_TEST_CASE("IMEX time advance - relaxation1x1v", "[imex]", test_precs)
{
  // Disable test for MPI - IMEX needs to be tested further with MPI
  if (!is_active() || get_num_ranks() > 1)
  {
    return;
  }

  std::vector<int> const levels{0, 4};

  TestType constexpr gmres_tol =
      std::is_same_v<TestType, double> ? 1.0e-10 : 1.0e-6;

  // the expected L2 from analytical solution after the maxwellian has relaxed
  // rel tolerance for comparing l2
  TestType constexpr tolerance = std::is_same_v<TestType, double> ? 1.0e-3 : 5.0e-3;

  auto opts = make_opts("-p relaxation_1x1v -d 2 -n 10 -s imex -sv gmres -dt 5.0e-4 -g dense");

  opts.start_levels      = levels;
  opts.isolver_tolerance = gmres_tol;

  discretization_manager disc(make_PDE<TestType>(opts));

  // -- time loop
  int64_t const num_final = disc.final_time_step();

  disc.set_final_time_step(0);

  // -- time loop
  for (auto i : indexof(num_final))
  {
    disc.add_time_steps(1);

    advance_time(disc);

    fk::vector<TestType> f_val = disc.current_state();

    // get analytic solution at final time step to compare
    if (i == opts.num_time_steps.value() - 1)
    {
      fk::vector<TestType> const analytic_solution = disc.get_exact_solution().value();

      // calculate L2 error between simulation and analytical solution
      TestType const L2 = nrm2_dist(f_val, analytic_solution);
      TestType const relative_error =
          TestType{100.0} * (L2 / fm::nrm2(analytic_solution));
      auto const [l2_errors, relative_errors] =
          asgard::gather_errors<TestType>(L2, relative_error);
      expect(l2_errors.size() == relative_errors.size());
      for (auto const &l2 : l2_errors)
        REQUIRE(l2 <= tolerance);
    }
  }

  parameter_manager<TestType>::get_instance().reset();
}

TEMPLATE_TEST_CASE("IMEX time advance - relaxation1x2v", "[!mayfail][imex]",
                   test_precs)
{
  // Disable test for MPI - IMEX needs to be tested further with MPI
  if (!is_active() || get_num_ranks() > 1)
  {
    return;
  }

  auto opts = make_opts("-p relaxation_1x2v -d 2 -n 10 -s imex -sv gmres -dt 5.0e-4 -g dense");

  opts.start_levels = {0, 4, 4};

  auto const pde = make_PDE<TestType>(opts);

  parameter_manager<TestType>::get_instance().reset();

  // TODO
  REQUIRE(false);
}

TEMPLATE_TEST_CASE("IMEX time advance - relaxation1x3v", "[!mayfail][imex]",
                   test_precs)
{
  // Disable test for MPI - IMEX needs to be tested further with MPI
  if (!is_active() || get_num_ranks() > 1)
  {
    return;
  }

  auto opts = make_opts("-p relaxation_1x3v -d 2 -n 10 -s imex -sv gmres -dt 5.0e-4 -g dense");

  opts.start_levels = {0, 4, 4, 4};

  auto const pde = make_PDE<TestType>(opts);

  parameter_manager<TestType>::get_instance().reset();

  // TODO
  REQUIRE(false);
}

/*****************************************************************************
 * Testing the ability to split a matrix into multiple calls
 *****************************************************************************/
#ifndef KRON_MODE_GLOBAL
template<typename prec>
void test_memory_mode(imex_flag imex)
{
  if (get_num_ranks() > 1) // this is a one-rank test
    return;

  // make some PDE, no need to be too specific
  verbosity_level verb = verbosity_level::quiet;

  auto opts = make_opts("-p two_stream -d 2 -l 5");

  discretization_manager<prec> disc(make_PDE<prec>(opts));

  auto const &pde  = disc.get_pde();
  auto const &grid = disc.get_grid();

  fk::vector<prec> x = disc.current_state();

  // one means that all data fits in memory and only one call will be made
  constexpr bool force_sparse = true;

  kron_sparse_cache spcache_null1, spcache_one;
  memory_usage memory_one =
      compute_mem_usage(pde, grid, imex, spcache_null1);

  coefficient_matrices<prec> &cmats = disc.get_cmatrices();
  connection_patterns const &conn   = disc.get_conn();

  auto mat_one = make_local_kronmult_matrix(
      pde, cmats, conn, grid, memory_one, imex_flag::unspecified, spcache_null1, verb);
  memory_usage spmemory_one = compute_mem_usage(
      pde, grid, imex, spcache_one, 6, 2147483646, force_sparse);
  auto spmat_one = make_local_kronmult_matrix(
      pde, cmats, conn, grid, spmemory_one, imex, spcache_one, verb, force_sparse);

  kron_sparse_cache spcache_null2, spcache_multi;
  memory_usage memory_multi =
      compute_mem_usage(pde, grid, imex, spcache_null2, 0, 8000);

  auto mat_multi = make_local_kronmult_matrix(
      pde, cmats, conn, grid, memory_multi, imex, spcache_null2, verb);
  memory_usage spmemory_multi = compute_mem_usage(
      pde, grid, imex, spcache_multi, 6, 8000, force_sparse);
  auto spmat_multi = make_local_kronmult_matrix(
      pde, cmats, conn, grid, spmemory_multi, imex, spcache_multi, verb, force_sparse);

  REQUIRE(mat_one.is_onecall());
  REQUIRE(spmat_one.is_onecall());
  REQUIRE(not spmat_multi.is_onecall());

  fk::vector<prec> y_one(mat_one.output_size());
  fk::vector<prec> y_multi(mat_multi.output_size());
  fk::vector<prec> y_spone(spmat_one.output_size());
  fk::vector<prec> y_spmulti(spmat_multi.output_size());
  REQUIRE(y_one.size() == y_multi.size());
  REQUIRE(y_one.size() == y_spmulti.size());
  REQUIRE(y_one.size() == y_spone.size());

#ifdef ASGARD_USE_CUDA
  fk::vector<prec, mem_type::owner, resource::device> xdev(y_one.size());
  fk::vector<prec, mem_type::owner, resource::device> ydev(y_multi.size());
  mat_one.set_workspace(xdev, ydev);
  mat_multi.set_workspace(xdev, ydev);
  spmat_one.set_workspace(xdev, ydev);
  spmat_multi.set_workspace(xdev, ydev);
#endif
#ifdef ASGARD_USE_GPU_MEM_LIMIT
  // allocate large enough vectors, total size is 24MB
  cudaStream_t load_stream;
  cudaStreamCreate(&load_stream);
  auto worka = fk::vector<int, mem_type::owner, resource::device>(1048576);
  auto workb = fk::vector<int, mem_type::owner, resource::device>(1048576);
  auto irowa = fk::vector<int, mem_type::owner, resource::device>(262144);
  auto irowb = fk::vector<int, mem_type::owner, resource::device>(262144);
  auto icola = fk::vector<int, mem_type::owner, resource::device>(262144);
  auto icolb = fk::vector<int, mem_type::owner, resource::device>(262144);
  mat_multi.set_workspace_ooc(worka, workb, load_stream);
  spmat_multi.set_workspace_ooc(worka, workb, load_stream);
  mat_multi.set_workspace_ooc_sparse(irowa, irowb, icola, icolb);
  spmat_multi.set_workspace_ooc_sparse(irowa, irowb, icola, icolb);
#endif

  mat_one.apply(2.0, x.data(), 0.0, y_one.data());
  mat_multi.apply(2.0, x.data(), 0.0, y_multi.data());
  spmat_one.apply(2.0, x.data(), 0.0, y_spone.data());
  spmat_multi.apply(2.0, x.data(), 0.0, y_spmulti.data());

  rmse_comparison(y_one, y_multi, prec{10});
  rmse_comparison(y_one, y_spone, prec{10});
  rmse_comparison(y_one, y_spmulti, prec{10});

  fk::vector<prec> y2_one(y_one);
  fk::vector<prec> y2_multi(y_multi);
  fk::vector<prec> y2_spone(y_spone);
  fk::vector<prec> y2_spmulti(y_spmulti);

  mat_one.apply(2.5, y_one.data(), 3.0, y2_one.data());
  mat_multi.apply(2.5, y_multi.data(), 3.0, y2_multi.data());
  spmat_one.apply(2.5, y_spone.data(), 3.0, y2_spone.data());
  spmat_multi.apply(2.5, y_spmulti.data(), 3.0, y2_spmulti.data());

  rmse_comparison(y2_one, y2_multi, prec{10});
  rmse_comparison(y_one, y_spone, prec{10});
  rmse_comparison(y_one, y_spmulti, prec{10});

  parameter_manager<prec>::get_instance().reset();
#ifdef ASGARD_USE_GPU_MEM_LIMIT
  cudaStreamDestroy(load_stream);
#endif
}

TEMPLATE_TEST_CASE("testing multi imex unspecified", "unspecified", test_precs)
{
  test_memory_mode<TestType>(imex_flag::unspecified);
}

TEMPLATE_TEST_CASE("testing multi imex implicit", "imex_implicit", test_precs)
{
  test_memory_mode<TestType>(imex_flag::imex_implicit);
}

TEMPLATE_TEST_CASE("testing multi imex explicit", "imex_explicit", test_precs)
{
  test_memory_mode<TestType>(imex_flag::imex_explicit);
}
#endif
