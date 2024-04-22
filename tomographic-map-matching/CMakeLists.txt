cmake_minimum_required(VERSION 3.5)
project(tomographic_map_matching CXX)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(PCL REQUIRED )
find_package(OpenMP REQUIRED)
find_package(spdlog REQUIRED)
find_package(teaserpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)

# Library
add_library(${PROJECT_NAME}_lib SHARED
  src/map_matcher_base.cpp
  src/consensus.cpp
  src/teaser_orb.cpp
)
target_link_libraries(${PROJECT_NAME}_lib
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
  ${OpenCV_LIBS}
  spdlog::spdlog
  teaserpp::teaser_registration
)
target_include_directories(${PROJECT_NAME}_lib PRIVATE include)

# Demo node
add_executable(${PROJECT_NAME}_standalone src/map_matching.cpp)
target_link_libraries(${PROJECT_NAME}_standalone
  ${PROJECT_NAME}_lib
  spdlog::spdlog
  gflags
)
target_include_directories(${PROJECT_NAME}_standalone PRIVATE include)
