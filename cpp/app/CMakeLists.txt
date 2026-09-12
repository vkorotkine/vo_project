add_executable(main main.cpp)
target_link_libraries(main PRIVATE core io rerun_sdk)


add_executable(gtsam_test gtsam_test.cpp)
target_link_libraries(gtsam_test PRIVATE core io gtsam)