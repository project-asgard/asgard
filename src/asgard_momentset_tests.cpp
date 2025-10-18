#include "asgard_test_macros.hpp"

using namespace asgard;

void test_moment() {
  current_test name_("create/compare moments");
  tassert(moment(-1).num_dims() == 0);
  tassert(moment(0).num_dims() == 1);
  tassert(moment(0, 2).num_dims() == 2);
  tassert(moment(0, 0, 3).num_dims() == 3);

  tassert(moment(2) == moment(2));
  tassert(moment(0, 0) == moment(0, 0));
  tassert(moment(0, 0, 3) == moment(0, 0, 3));
  tassert(moment(0, 0, 3) != moment(0, -1, 3));
  tassert(moment(2, 3) != moment(1, 3));
  tassert(moment(2, 3) != moment(2, 1));

  tassert(moment(0).action == moment::regular);
  tassert(moment(0, 0).action == moment::regular);
  tassert(moment(0, 0, 0).action == moment::regular);
  tassert(moment(3, moment::interpolatory).action == moment::interpolatory);
  tassert(moment(0, 0, moment::inactive).action == moment::inactive);

  // moment ids
  tassert(moment_id(2).get() == 2);
  tassert(moment_id(5)() == 5);
  tassert(moment_id(3) == moment_id(3));
  tassert(moment_id(1) != moment_id(2));
}

void test_moment_list() {
  current_test name_("moment list");
  {
    moments_list list;
    tassert(list.empty());
    tassert(list.num_moms() == 0);
    tassert(list.size() == 0);

    moment_id id = list.get_add_id({0, 0});
    tassert(list.size() == 1);
    tassert(id() == 0);
    tassert(list[0] == moment(0, 0));
    tassert(list[0] != moment(0, 1));
    tassert(list[id] == moment(0, 0));
  }{
    moments_list list;
    auto id0 = list.get_add_id(0);
    auto id1 = list.get_add_id(2);
    auto id2 = list.get_add_id(0);
    static_assert(std::is_same_v<decltype(id0), moment_id>);
    tassert(list.size() == 2);
    tassert(id0 == id2);
    tassert(id0 != id1);
    auto const &clist = list;
    tassert(clist.get_id(2) == id1);
    tassert(clist.have_all_dimension(1));
    tassert(!clist.have_all_dimension(2));

    terror_message(clist.get_id(3), "cannot find the specified moment");

    id0 = list.get_add_id({0, 2});
    tassert(list.size() == 3);
    tassert(!clist.have_all_dimension(1));
    tassert(!clist.have_all_dimension(2));
    tassert(!clist.have_all_dimension(3));
  }{
    moments_list super;
    super.get_add_id(0);
    auto id1 = super.get_add_id(5);
    auto id2 = super.get_add_id(3);
    super.get_add_id(4);
    moments_list list;
    list.get_add_id(3);
    list.get_add_id(5);
    std::vector<moment_id> ref = {id1, id2};
    std::vector<moment_id> val = list.find_as_subset_of(super);
    tassert(ref.size() == val.size());
    for (auto const &id : ref) {
      tassert(std::any_of(val.begin(), val.end(),
              [&](moment_id const &a) -> bool { return (a == id); }));
    }
  }
}

int main(int argc, char **argv) {

  libasgard_runtime running_(argc, argv);

  all_tests global_("momentset", " testing functionality");

  test_moment();
  test_moment_list();

  return 0;
}
