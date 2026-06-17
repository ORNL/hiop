# Build EVLOSER embedded backend library

set(EVLOSER_SRC
  RefactorizationSolver.cpp
  MatrixCsr.cpp
)

if(HIOP_USE_CUDA)
  list(APPEND EVLOSER_SRC
    IterativeRefinement.cpp
    KrylovSolverKernels.cu
  )
  set_source_files_properties(${EVLOSER_SRC} PROPERTIES LANGUAGE CUDA)
endif()

add_library(EVLOSER STATIC ${EVLOSER_SRC})

if(HIOP_USE_CUDA)
  target_compile_definitions(EVLOSER PRIVATE HIOP_USE_CUDA)
endif()

if(HIOP_USE_HIP)
  target_compile_definitions(EVLOSER PRIVATE HIOP_USE_HIP)
endif()

target_include_directories(EVLOSER INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(EVLOSER PRIVATE KLU)

if(HIOP_USE_CUDA)
  target_link_libraries(EVLOSER PRIVATE hiop_cuda)
endif()

if(HIOP_USE_HIP)
  target_link_libraries(EVLOSER PRIVATE hipsparse hipsolver hipblas)
endif()
