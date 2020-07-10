cmake_minimum_required(VERSION 3.13)

set(BLT_CXX_STD "c++14" CACHE STRING "Version of C++ standard")

set(test_laplacexy_sources
   test-laplacexy.cxx
)

set(test_laplacexy_depends bout++ PETSc)

if(HAVE_CUDA)
   list(APPEND test_laplacexy_depends cuda)
endif ()

blt_add_executable(
   NAME test_laplacexy
   SOURCES ${test_laplacexy_sources}
   INCLUDES 
     ${PROJECT_SOURCE_DIR}/include
     ${PROJECT_SOURCE_DIR}/externalpackages/fmt/include
     ${PROJECT_SOURCE_DIR}/externalpackages/mpark.variant/include
   DEFINES 
      "BOUT_VERSION_STRING=\"${BOUT_FULL_VERSION}\""
      "BOUT_VERSION_DOUBLE=${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}${PROJECT_VERSION_PATCH}"
   DEPENDS_ON
     ${test_laplacexy_depends})



