#pragma warning(push, 0)

#define BOOST_TEST_MODULE HalideTest
#include <boost/test/unit_test.hpp>

#pragma warning(pop)

BOOST_AUTO_TEST_CASE(init_test_main){
  BOOST_TEST(0 == int(1));
}
