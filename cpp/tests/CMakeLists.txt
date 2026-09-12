add_executable(
    unit_tests
    test_estimator.cpp
    test_lie.cpp
    test_gtsam.cpp
)

target_link_libraries(unit_tests PRIVATE core io gtsam GTest::gtest_main)
target_compile_definitions(unit_tests PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
include(GoogleTest)
gtest_discover_tests(unit_tests)

add_executable(
    integration_tests
    test_integration_tum.cpp
)

target_link_libraries(integration_tests PRIVATE core io gtsam GTest::gtest_main)
target_compile_definitions(integration_tests PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
include(GoogleTest)
gtest_discover_tests(integration_tests)
