find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# add_library(
#     slam
#     io/Types.cpp
# )
# target_include_directories(slam PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
# target_link_libraries(())

add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(app)
add_subdirectory(types)
