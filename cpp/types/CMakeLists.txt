add_library(types INTERFACE)
target_include_directories(
    types INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(types INTERFACE opencv_core  Eigen3::Eigen )