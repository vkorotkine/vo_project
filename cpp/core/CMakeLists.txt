add_library(core Estimator.cpp  Frontend.cpp Checks.cpp)
target_include_directories(core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(core PUBLIC Eigen3::Eigen gtsam opencv_core opencv_highgui opencv_imgcodecs opencv_features2d opencv_calib3d)
