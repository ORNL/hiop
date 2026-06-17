# Build EVLOSER embedded backend library

set(EVLOSER_SRC
  RefactorizationSolver.cpp
  MatrixCsr.cpp
  IterativeRefinement.cpp
  KrylovSolverKernels.cu
)
set_source_files_properties(${EVLOSER_SRC} PROPERTIES LANGUAGE CUDA)

add_library(EVLOSER STATIC ${EVLOSER_SRC})
target_include_directories(EVLOSER INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(EVLOSER PRIVATE KLU hiop_cuda)
