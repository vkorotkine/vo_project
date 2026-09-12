cmake_minimum_required(VERSION 3.16)
project(vio_project)

set(CMAKE_CXX_STANDARD 17)

message("Building!")

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0)
FetchContent_MakeAvailable(googletest)

include(FetchContent)
FetchContent_Declare(rerun_sdk URL
  https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

find_package(GTSAM REQUIRED)

enable_testing()
add_subdirectory(cpp)
add_subdirectory(cpp/tests)
