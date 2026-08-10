set(hiopSparseEx_INTERFACE_HEADERS
  NlpSparseEx1.hpp
  NlpSparseEx2.hpp
  NlpSparseEx3.hpp
  NlpSparseEx4.hpp
  )

add_executable(NlpSparseEx1.exe NlpSparseEx1.cpp NlpSparseEx1Driver.cpp)
target_link_libraries(NlpSparseEx1.exe HiOp::HiOp)

add_executable(NlpSparseEx2.exe NlpSparseEx2.cpp NlpSparseEx2Driver.cpp)
target_link_libraries(NlpSparseEx2.exe HiOp::HiOp)

add_executable(NlpSparseEx3.exe NlpSparseEx3.cpp NlpSparseEx3Driver.cpp)
target_link_libraries(NlpSparseEx3.exe HiOp::HiOp)

add_executable(NlpSparseEx4.exe NlpSparseEx4.cpp NlpSparseEx4Driver.cpp)
target_link_libraries(NlpSparseEx4.exe HiOp::HiOp)

if(HIOP_USE_RAJA)
  if(HIOP_USE_GPU AND (HIOP_USE_CUDA OR HIOP_USE_HIP) AND HIOP_USE_EVLOSER)
    if(HIOP_USE_CUDA)
      set_source_files_properties(
        NlpSparseRajaEx2.cpp
        NlpSparseRajaEx2Driver.cpp
        PROPERTIES LANGUAGE CUDA
      )
    endif()
  
    add_executable(NlpSparseRajaEx2.exe  NlpSparseRajaEx2Driver.cpp  NlpSparseRajaEx2.cpp)
    target_link_libraries(NlpSparseRajaEx2.exe HiOp::HiOp)
    install(TARGETS NlpSparseRajaEx2.exe DESTINATION bin)
    list(APPEND hiopSparseEx_INTERFACE_HEADERS NlpSparseRajaEx2.hpp)
  endif()
endif()

if(HIOP_BUILD_SHARED)
  add_executable(NlpSparseCEx1.exe NlpSparseCEx1.c)
  target_link_libraries(NlpSparseCEx1.exe HiOp::HiOp)
  if(HIOP_BUILD_FORTRAN_EXAMPLE)
    add_executable(NlpSparseFEx1.exe NlpSparseFEx1.F90)
    target_link_libraries(NlpSparseFEx1.exe HiOp::HiOp)
    add_executable(NlpSparseHS071.exe NlpSparseExHS071.f)
    target_link_libraries(NlpSparseHS071.exe HiOp::HiOp)
  endif()
endif()

install(
  FILES ${hiopSparseEx_INTERFACE_HEADERS}
  DESTINATION include
  )

##########################################################
# CMake Tests
##########################################################
add_test(NAME NlpSparse1_1 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-selfcheck")
add_test(NAME NlpSparse1_2 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-fr" "-selfcheck")
if(HIOP_USE_EVLOSER)
  add_test(NAME NlpSparse1_EVLOSER COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-evloser" "-selfcheck")

  # Fail if EVLOSER uses an unexpected backend or iterative-refinement mode.
  if(HIOP_USE_CUDA)
    set_tests_properties(NlpSparse1_EVLOSER PROPERTIES FAIL_REGULAR_EXPRESSION "Refactorization: klu;Refactorization: glu;Refactorization: rocSolverRf;Use IR: no")
  elseif(HIOP_USE_HIP)
    set_tests_properties(NlpSparse1_EVLOSER PROPERTIES FAIL_REGULAR_EXPRESSION "Refactorization: klu;Refactorization: glu;Refactorization: rf;Use IR: no")
  else()
    set_tests_properties(NlpSparse1_EVLOSER PROPERTIES FAIL_REGULAR_EXPRESSION "Refactorization: glu;Refactorization: rf;Refactorization: rocSolverRf;Use IR: yes")
  endif()
endif()
if(HIOP_USE_PARDISO)
  add_test(NAME NlpSparse1_4 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-pardiso" "-selfcheck")
endif(HIOP_USE_PARDISO)
if(HIOP_USE_GINKGO)
  add_test(NAME NlpSparse1_5 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-ginkgo" "-selfcheck")
  if(HIOP_USE_CUDA)
      add_test(NAME NlpSparse1_6 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-ginkgo_cuda" "-selfcheck")
  endif(HIOP_USE_CUDA)
  if(HIOP_USE_HIP)
      add_test(NAME NlpSparse1_7 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-ginkgo_hip" "-selfcheck")
  endif(HIOP_USE_HIP)
endif(HIOP_USE_GINKGO)
add_test(NAME NlpSparse2_1 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-selfcheck")
add_test(NAME NlpSparse2_2 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-inertiafree" "-selfcheck")
if(HIOP_USE_EVLOSER)
  add_test(NAME NlpSparse2_EVLOSER COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-evloser" "-inertiafree" "-selfcheck")

  if(HIOP_USE_CUDA)
    add_test(NAME NlpSparse2_EVLOSER_GLU COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-evloser_cuda_glu" "-inertiafree" "-selfcheck")
    add_test(NAME NlpSparse2_EVLOSER_RF COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-evloser_cuda_rf" "-inertiafree" "-selfcheck")
  endif()

  if(HIOP_USE_HIP)
    add_test(NAME NlpSparse2_EVLOSER_HIP COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-evloser_hip" "-inertiafree" "-selfcheck")
  endif()
endif()
if(HIOP_USE_CUDA AND HIOP_USE_COINHSL)
  add_test(NAME NlpSparse2_3 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-cusolver" "-inertiafree" "-selfcheck")
endif()
if(HIOP_USE_GINKGO)
  add_test(NAME NlpSparse2_4 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-ginkgo" "-inertiafree" "-selfcheck")
  if(HIOP_USE_CUDA)
      add_test(NAME NlpSparse2_5 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-ginkgo_cuda" "-inertiafree" "-selfcheck")
  endif(HIOP_USE_CUDA)
  if(HIOP_USE_HIP)
      add_test(NAME NlpSparse2_6 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-ginkgo_hip" "-inertiafree" "-selfcheck")
  endif(HIOP_USE_HIP)
endif(HIOP_USE_GINKGO)

if(HIOP_USE_RAJA AND HIOP_USE_GPU)
  if(HIOP_USE_CUDA)
    if(HIOP_USE_EVLOSER)
      add_test(NAME NlpSparseRaja2_EVLOSER_GLU COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseRajaEx2.exe>" "500" "-inertiafree" "-selfcheck" "-evloser_cuda_glu")
      add_test(NAME NlpSparseRaja2_EVLOSER_RF COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseRajaEx2.exe>" "500" "-inertiafree" "-selfcheck" "-evloser_cuda_rf")
    endif()
  endif()
  if(HIOP_USE_HIP AND HIOP_USE_EVLOSER)
    add_test(NAME NlpSparseRaja2_EVLOSER_HIP COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseRajaEx2.exe>" "500" "-inertiafree" "-selfcheck" "-evloser_hip")
  endif()
endif()

add_test(NAME NlpSparse3_1 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx3.exe>" "500" "-selfcheck")
if(HIOP_BUILD_SHARED AND NOT HIOP_USE_GPU )
  add_test(NAME NlpSparseCinterface COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseCEx1.exe>")
  if(HIOP_BUILD_FORTRAN_EXAMPLE)
    add_test(NAME NlpSparseFinterface COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseFEx1.exe>")
  endif()
endif()

