#include "distribution.hpp"
#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "tests_general.hpp"
#include "transformations.hpp"
#include <numeric>

template<typename P>
void test_combine_dimensions(PDE<P> const &pde, int const lev, int const deg,
                             P const time = 1.0, int const num_ranks = 1,
                             bool const full_grid = false)
{
  int const dims = pde.num_dims;
  std::string const filename =
      "../testing/generated-inputs/transformations/combine_dim_dim" +
      std::to_string(dims) + "_deg" + std::to_string(deg) + "_lev" +
      std::to_string(lev) + "_" + (full_grid ? "fg" : "sg") + ".dat";

  // FIXME assuming uniform degree across dims
  dimension const dim = pde.get_dimensions()[0];

  std::string const grid_str = full_grid ? "-f" : "";
  options const o            = make_options(
      {"-d", std::to_string(deg), "-l", std::to_string(lev), grid_str});

  element_table const t(o, dims);

  std::vector<fk::vector<P>> vectors;
  P counter = 1.0;
  for (int i = 0; i < pde.num_dims; ++i)
  {
    int const vect_size         = dims * static_cast<int>(std::pow(2, lev));
    fk::vector<P> const vect_1d = [&counter, vect_size] {
      fk::vector<P> vect(vect_size);
      std::iota(vect.begin(), vect.end(), static_cast<P>(counter));
      counter += vect.size();
      return vect;
    }();
    vectors.push_back(vect_1d);
  }
  distribution_plan const plan = get_plan(num_ranks, t);

  fk::vector<P> const gold = fk::vector<P>(read_vector_from_txt_file(filename));
  fk::vector<P> test(gold.size());
  for (auto const &[rank, grid] : plan)
  {
    int const rank_start =
        grid.row_start * static_cast<int>(std::pow(deg, dims));
    int const rank_stop =
        (grid.row_stop + 1) * static_cast<int>(std::pow(deg, dims)) - 1;
    fk::vector<P, mem_type::view> const gold_partial(gold, rank_start,
                                                     rank_stop);
    fk::vector<P> const test_partial = combine_dimensions(
        deg, t, plan.at(rank).row_start, plan.at(rank).row_stop, vectors, time);
    REQUIRE(test_partial == gold_partial);
    test.set_subvector(rank_start, test_partial);
  }
  REQUIRE(test == gold);
}

TEMPLATE_TEST_CASE("combine dimensions", "[transformations]", double, float)
{
  SECTION("combine dimensions, dim = 2, deg = 2, lev = 3, 1 rank")
  {
    int const lev       = 3;
    int const deg       = 2;
    auto const pde      = make_PDE<TestType>(PDE_opts::continuity_2, lev, deg);
    TestType const time = 2.0;
    test_combine_dimensions(*pde, lev, deg, time);
  }

  SECTION("combine dimensions, dim = 2, deg = 2, lev = 3, 8 ranks")
  {
    int const lev       = 3;
    int const deg       = 2;
    auto const pde      = make_PDE<TestType>(PDE_opts::continuity_2, lev, deg);
    int const num_ranks = 8;
    TestType const time = 2.0;
    test_combine_dimensions(*pde, lev, deg, time, num_ranks);
  }

  SECTION("combine dimensions, dim = 3, deg = 3, lev = 2, full grid")
  {
    int const lev        = 2;
    int const deg        = 3;
    auto const pde       = make_PDE<TestType>(PDE_opts::continuity_3, lev, deg);
    int const num_ranks  = 20;
    TestType const time  = 2.5;
    bool const full_grid = true;
    test_combine_dimensions(*pde, lev, deg, time, num_ranks, full_grid);
  }
}

TEMPLATE_TEST_CASE("forward multi-wavelet transform", "[transformations]",
                   double, float)
{
  auto const relaxed_comparison = [](auto const first, auto const second) {
    auto first_it = first.begin();
    std::for_each(second.begin(), second.end(), [&first_it](auto &second_elem) {
      REQUIRE(Approx(*first_it++)
                  .epsilon(std::numeric_limits<TestType>::epsilon() * 1e4) ==
              second_elem);
    });
  };

  SECTION("transform(2, 2, -1, 1, double)")
  {
    int const degree     = 2;
    int const levels     = 2;
    auto const double_it = [](fk::vector<TestType> x, TestType t) {
      ignore(t);
      return x * static_cast<TestType>(2.0);
    };

    dimension const dim =
        make_PDE<TestType>(PDE_opts::continuity_1, levels, degree)
            ->get_dimensions()[0];
    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(
            "../testing/generated-inputs/transformations/forward_transform_" +
            std::to_string(degree) + "_" + std::to_string(levels) +
            "_neg1_pos1_double.dat"));

    fk::vector<TestType> const test =
        forward_transform<TestType>(dim, double_it);
    relaxed_comparison(gold, test);
  }

  SECTION("transform(3, 4, -2.0, 2.0, double plus)")
  {
    int const degree       = 3;
    int const levels       = 4;
    auto const double_plus = [](fk::vector<TestType> x, TestType t) {
      ignore(t);
      return x + (x * static_cast<TestType>(2.0));
    };

    dimension const dim =
        make_PDE<TestType>(PDE_opts::continuity_2, levels, degree)
            ->get_dimensions()[1];

    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(
            "../testing/generated-inputs/transformations/forward_transform_" +
            std::to_string(degree) + "_" + std::to_string(levels) +
            "_neg2_pos2_doubleplus.dat"));

    fk::vector<TestType> const test =
        forward_transform<TestType>(dim, double_plus);

    relaxed_comparison(gold, test);
  }
}
