cmake_minimum_required(VERSION 2.8.3)
set(PROJECT_NAME DeVio)
project(${PROJECT_NAME})

set(CMAKE_BUILD_TYPE release)

set(CMAKE_CXX_FLAGS "-std=c++11")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fpic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")

#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
MESSAGE("PROJECT_SOURCE_DIR " ${PROJECT_SOURCE_DIR})

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -mssse3" )
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -mssse3" )

find_package(Eigen3  REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)


include_directories(${EIGEN3_INCLUDE_DIR}
        ${CERES_INCLUDE_DIRS})


#Build the main lib
add_subdirectory(src)


add_executable(testPinholeProjectFactor  test/testPinholeProjectFactor.cpp)
target_link_libraries(testPinholeProjectFactor ${PROJECT_NAME} )



add_executable(testMarginalization  test/testMarginalization.cpp)
target_link_libraries(testMarginalization ${PROJECT_NAME} )


