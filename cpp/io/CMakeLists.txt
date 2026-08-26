add_library(io Dataset.cpp)
target_include_directories(
    io PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(io PUBLIC opencv_core core)