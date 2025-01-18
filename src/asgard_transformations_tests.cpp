#include "tests_general.hpp"

#include "asgard_small_mats.hpp"

static auto const transformations_base_dir = gold_base_dir / "transformations";

using namespace asgard;

template<typename P>
void test_combine_dimensions(PDE<P> const &pde, P const time = 1.0,
                             int const num_ranks  = 1,
                             bool const full_grid = false)
{
  int const dims = pde.num_dims();

  dimension const dim = pde.get_dimensions()[0];
  int const lev       = dim.get_level();
  int const degree    = dim.get_degree();

  std::string const filename =
      "combine_dim_dim" + std::to_string(dims) + "_deg" + std::to_string(degree + 1) +
      "_lev" + std::to_string(lev) + "_" + (full_grid ? "fg" : "sg") + ".dat";

  elements::table const t(pde);

  std::vector<std::vector<P>> vectors;
  P counter = 1.0;
  for (int i = 0; i < pde.num_dims(); ++i)
  {
    int const vect_size         = dims * fm::ipow2(lev);
    std::vector<P> const vect_1d = [&counter, vect_size] {
      std::vector<P> vect(vect_size);
      std::iota(vect.begin(), vect.end(), static_cast<P>(counter));
      counter += vect.size();
      return vect;
    }();
    vectors.push_back(vect_1d);
  }
  distribution_plan const plan = get_plan(num_ranks, t);

  fk::vector<P> const gold =
      read_vector_from_txt_file<P>(transformations_base_dir / filename);
  fk::vector<P> test(gold.size());
  for (auto const &[rank, grid] : plan)
  {
    int const rank_start =
        grid.row_start * fm::ipow(degree + 1, dims);
    int const rank_stop =
        (grid.row_stop + 1) * fm::ipow(degree + 1, dims) - 1;
    fk::vector<P, mem_type::const_view> const gold_partial(gold, rank_start,
                                                           rank_stop);
    std::vector<P> test_partial(gold_partial.size());

    combine_dimensions(
        degree, t, plan.at(rank).row_start, plan.at(rank).row_stop, vectors, test_partial.data());
    for (auto &t : test_partial)
      t *= time;
    fk::vector<P> fk_test_partial(test_partial);
    REQUIRE(fk_test_partial == gold_partial);
    test.set_subvector(rank_start, fk_test_partial);
  }
  REQUIRE(test == gold);
}

TEMPLATE_TEST_CASE("fast-transform", "[transformations]", test_precs)
{
  std::minstd_rand park_miller(42);
  std::uniform_real_distribution<TestType> unif(-1.0, 1.0);

  for (int nbatch = 1; nbatch < 5; nbatch++) {
    for (int level = 0; level < 5; level++) {
      for (int degree = 0; degree < 4; degree++)
      {
        hierarchy_manipulator<TestType> hier(degree, 1, {-2,}, {1,}); // dims 1

        int const pdof    = (degree + 1);
        int64_t const num = fm::ipow2(level);

        std::vector<TestType> ref(nbatch * num * pdof);
        std::vector<TestType> hp(nbatch * num * pdof);

        for (auto &x : ref)
          x = unif(park_miller);

        std::vector<TestType> fp(num * pdof);
        for (int b : indexof(nbatch)) // forward project
        {
          TestType *r = ref.data() + b * pdof; // batch begin
          for (int i : indexof(num))
            std::copy_n(r + i * nbatch * pdof, pdof, fp.data() + i * pdof);

          hier.project1d(level, fp); // to hierarchical

          TestType *h = hp.data() + b * pdof; // write out in hp
          for (int i : indexof(num))
            std::copy_n(fp.data() + i * pdof, pdof, h + i * nbatch * pdof);
        }

        if (level > 0)
          REQUIRE(fm::diff_inf(ref, hp) > 1.E-2); // sanity check, did we transform anything

        hier.reconstruct1d(nbatch, level, span2d<TestType>(pdof, nbatch * num, hp.data()));

        REQUIRE(fm::diff_inf(ref, hp) < 5.E-6); // inverse transform should get us back
      }
    }
  }
}
