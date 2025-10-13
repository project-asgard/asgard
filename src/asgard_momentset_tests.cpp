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

  // moment ids
  static_assert(not std::is_default_constructible_v<moment_id>);
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

    moment_id id = list.get_id({0, 0});
    tassert(list.size() == 1);
    tassert(id() == 0);
    tassert(list[0] == moment(0, 0));
    tassert(list[0] != moment(0, 1));
    tassert(list[id] == moment(0, 0));
  }{
    moments_list list;
    auto id0 = list.get_id(0);
    auto id1 = list.get_id(2);
    auto id2 = list.get_id(0);
    static_assert(std::is_same_v<decltype(id0), moment_id>);
    tassert(list.size() == 2);
    tassert(id0 == id2);
    tassert(id0 != id1);
    auto const &clist = list;
    tassert(clist.get_id(2) == id1);
    tassert(clist.have_all_dimension(1));
    tassert(!clist.have_all_dimension(2));

    terror_message(clist.get_id(3), "cannot find the specified moment");

    id0 = list.get_id({0, 2});
    tassert(list.size() == 3);
    tassert(!clist.have_all_dimension(1));
    tassert(!clist.have_all_dimension(2));
    tassert(!clist.have_all_dimension(3));
  }
}

int main(int argc, char **argv) {

  libasgard_runtime running_(argc, argv);

  all_tests global_("momentset", " testing functionality");

  test_moment();
  test_moment_list();

  return 0;
}
