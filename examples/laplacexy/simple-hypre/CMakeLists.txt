cmake_minimum_required(VERSION 3.13)

set(BLT_CXX_STD "c++14" CACHE STRING "Version of C++ standard")
if(NOT BOUT_HAS_HYPRE)
  message(FATAL_ERROR "This example requires BOUT++ to have been built with HYPRE")
endif()

set(test_laplacexy_hypre_sources
   test-laplacexy-hypre.cxx
)

set(test_laplacexy_hypre_depends bout++ PETSc HYPRE)

if(HAVE_CUDA)
   set(test_laplacexy_hypre_depends ${test_laplacexy_hypre_depends} cuda)
endif ()

message(STATUS "test_laplacexy_hypre_depends: ${test_laplacexy_hypre_depends}")
blt_add_executable(
   NAME test_laplacexy_hypre
   SOURCES ${test_laplacexy_hypre_sources}
   INCLUDES 
     ${PROJECT_SOURCE_DIR}/include
     ${PROJECT_SOURCE_DIR}/externalpackages/fmt/include
     ${PROJECT_SOURCE_DIR}/externalpackages/mpark.variant/include
   DEFINES 
      "BOUT_VERSION_STRING=\"${BOUT_FULL_VERSION}\""
      "BOUT_VERSION_DOUBLE=${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}${PROJECT_VERSION_PATCH}"
      BOUT_HAS_HYPRE
   DEPENDS_ON
     ${test_laplacexy_hypre_depends})



