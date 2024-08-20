
#include "tests_general.hpp"

static auto constexpr adapt_threshold = 1e-4;

static auto const adapt_base_dir = gold_base_dir / "adapt";

using namespace asgard;

struct distribution_test_init
{
  void set_my_rank(const int rank) { my_rank = rank; }
  void set_num_ranks(const int size) { num_ranks = size; }
  int get_my_rank() const { return my_rank; }
  int get_num_ranks() const { return num_ranks; }

private:
  int my_rank;
  int num_ranks;
};
static distribution_test_init distrib_test_info;

int main(int argc, char *argv[])
{
  auto const [rank, total_ranks] = initialize_distribution();
  distrib_test_info.set_my_rank(rank);
  distrib_test_info.set_num_ranks(total_ranks);

  int result = Catch::Session().run(argc, argv);

  finalize_distribution();

  return result;
}

template<typename P>
void test_adapt(prog_opts const &opts, std::filesystem::path gold_base)
{
  auto const prefix = gold_base.filename().string();
  gold_base.remove_filename();
  auto const fval_orig_path    = gold_base / (prefix + "orig.dat");
  auto const fval_refine_path  = gold_base / (prefix + "refine.dat");
  auto const fval_coarse_path  = gold_base / (prefix + "coarse.dat");
  auto const table_refine_path = gold_base / (prefix + "refine_table.dat");
  auto const table_coarse_path = gold_base / (prefix + "coarse_table.dat");

  auto const fval_orig   = read_vector_from_txt_file<P>(fval_orig_path);
  auto const gold_coarse = read_vector_from_txt_file<P>(fval_coarse_path);
  auto const gold_refine = [fval_refine_path]() {
    auto gold = read_vector_from_txt_file<P>(fval_refine_path);
    for (auto i = 0; i < gold.size(); ++i)
    {
      // matlab stores new refined coefficients as 1e-15 (0 deletes from sparse
      // vect)
      if (std::abs(gold(i)) < 1e-14)
      {
        gold(i) = 0.0;
      }
    }
    return gold;
  }();

  auto const test_tables = [](elements::table const &test,
                              fk::matrix<int> const &gold) {
    REQUIRE(test.size() == gold.nrows());
    for (int64_t i = 0; i < test.size(); ++i)
    {
      auto const &test_coords = test.get_coords(i);
      fk::vector<int> const gold_coords =
          gold.extract_submatrix(i, 0, 1, gold.ncols());
      REQUIRE(test_coords == gold_coords);
    }
  };

  auto const gold_coarse_table =
      read_matrix_from_txt_file<int>(table_coarse_path);
  auto const gold_refine_table =
      fk::matrix<int>(read_matrix_from_txt_file<int>(table_refine_path));

  auto const pde = make_PDE<P>(opts);
  // using second opts after reset from the pde
  auto const &opts2 = pde->options();

  adapt::distributed_grid<P> refine_grid(*pde);
  auto const my_subgrid   = refine_grid.get_subgrid(get_rank());
  auto const segment_size = element_segment_size(*pde);
  auto const my_fval_orig =
      fval_orig.extract(my_subgrid.col_start * segment_size,
                        (my_subgrid.col_stop + 1) * segment_size - 1);

  auto const test_refine = refine_grid.refine(my_fval_orig, opts2);
  adapt::distributed_grid<P> coarse_grid(*pde);
  auto const test_coarse = coarse_grid.coarsen(my_fval_orig, opts2);
  test_tables(coarse_grid.get_table(), gold_coarse_table);
  test_tables(refine_grid.get_table(), gold_refine_table);

  auto const refine_subgrid = refine_grid.get_subgrid(get_rank());
  fk::vector<P, mem_type::const_view> const my_gold_refine(
      gold_refine, refine_subgrid.col_start * segment_size,
      (refine_subgrid.col_stop + 1) * segment_size - 1);
  REQUIRE(test_refine == my_gold_refine);

  auto const coarsen_subgrid = coarse_grid.get_subgrid(get_rank());
  fk::vector<P, mem_type::const_view> const my_gold_coarse(
      gold_coarse, coarsen_subgrid.col_start * segment_size,
      (coarsen_subgrid.col_stop + 1) * segment_size - 1);
  REQUIRE(test_coarse == my_gold_coarse);
}

TEMPLATE_TEST_CASE("adapt - 1d, scattered coarsen/refine", "[adapt]",
                   test_precs)
{
  if (!is_active())
  {
    return;
  }

  auto opts = make_opts("-p continuity_1 -d 2 -l 4 -m 8");

  opts.adapt_threshold = adapt_threshold;
  opts.anorm           = adapt_norm::linf;

  test_adapt<TestType>(opts, adapt_base_dir / "continuity1_l4_d3_");
}

TEMPLATE_TEST_CASE("adapt - 2d, all zero", "[adapt]", test_precs)
{
  if (!is_active())
  {
    return;
  }

  auto opts = make_opts("-p continuity_2 -d 1 -l 5 -m 8");

  opts.adapt_threshold = adapt_threshold;
  opts.anorm           = adapt_norm::linf;

  // temporarily disable test for MPI due to table elements < num ranks
  if (get_num_ranks() == 1)
  {
    test_adapt<default_precision>(opts, adapt_base_dir / "continuity2_l5_d2_");
  }
}

TEMPLATE_TEST_CASE("adapt - 3d, scattered, contiguous refine/adapt", "[adapt]",
                   test_precs)
{
  if (!is_active())
  {
    return;
  }

  auto opts = make_opts("-p continuity_3 -d 3 -l 4 -m 8");

  opts.adapt_threshold = adapt_threshold;
  opts.anorm           = adapt_norm::linf;

  test_adapt<TestType>(opts, adapt_base_dir / "continuity3_l4_d4_");
}

template<typename P>
void test_initial(prog_opts const &opts, std::string const &gold_filepath)
{
  auto const gold = [&gold_filepath]() {
    auto raw = read_vector_from_txt_file<P>(gold_filepath);
    // matlab stores new refined coefficients as 1e-15 (0 deletes from sparse
    // vector)
    std::replace_if(
        raw.begin(), raw.end(),
        [](P old_value) { return std::abs(old_value) < 1.e-14; }, 0.0);
    return raw;
  }();

  discretization_manager<P> disc(make_PDE<P>(opts));

  fk::vector<P> const test = disc.current_state();

  REQUIRE(gold.size() >= test.size());

  auto constexpr tol_factor = get_tolerance<P>(100);
  auto const my_subgrid     = disc.get_grid().get_subgrid(get_rank());
  auto const segment_size   = disc.get_hiermanip().block_size();
  fk::vector<P, mem_type::const_view> const my_gold(
      gold, my_subgrid.col_start * segment_size,
      (my_subgrid.col_stop + 1) * segment_size - 1);
  rmse_comparison(my_gold, test, tol_factor);
}

TEMPLATE_TEST_CASE("initial - diffusion 1d", "[adapt]", test_precs)
{
  auto opts = make_opts("-p diffusion_1 -d 3 -l 3 -m 8");

  opts.adapt_threshold = adapt_threshold;
  opts.anorm           = adapt_norm::linf;

  // don't test this in the MPI case -- too small to split table
  if (get_num_ranks() == 1)
  {
    test_initial<TestType>(opts,
                           adapt_base_dir / "diffusion1_l3_d4_initial.dat");
  }
}

TEMPLATE_TEST_CASE("initial - diffusion 2d", "[adapt]", test_precs)
{
  if (!is_active())
  {
    return;
  }

  auto opts = make_opts("-p diffusion_2 -d 2 -l 2 -m 8");

  opts.adapt_threshold = adapt_threshold;
  opts.anorm           = adapt_norm::linf;

  test_initial<TestType>(opts,
                         adapt_base_dir / "diffusion2_l2_d3_initial.dat");
}
