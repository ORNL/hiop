//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop.
// HiOp is released under the BSD 3-clause license
// (https://opensource.org/licenses/BSD-3-Clause). Please also read “Additional
// BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the disclaimer below. ii. Redistributions in
// binary form must reproduce the above copyright notice, this list of
// conditions and the disclaimer (as noted below) in the documentation and/or
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
// THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S.
// Department of Energy (DOE). This work was produced at Lawrence Livermore
// National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National
// Security, LLC nor any of their employees, makes any warranty, express or
// implied, or assumes any liability or responsibility for the accuracy,
// completeness, or usefulness of any information, apparatus, product, or
// process disclosed, or represents that its use would not infringe
// privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or
// services by trade name, trademark, manufacturer or otherwise does not
// necessarily constitute or imply its endorsement, recommendation, or favoring
// by the United States Government or Lawrence Livermore National Security,
// LLC. The views and opinions of authors expressed herein do not necessarily
// state or reflect those of the United States Government or Lawrence Livermore
// National Security, LLC, and shall not be used for advertising or product
// endorsement purposes.

/**
 * @file hiopLinSolverSparseEVLOSER.cpp
 */

#include "hiopLinSolverSparseEVLOSER.hpp"

#include "LinAlgFactory.hpp"
#include "hiopLinSolver.hpp"
#include "hiopMatrixSparse.hpp"
#include "hiopVector.hpp"

#include "hiopCppStdUtils.hpp"

#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>

#if defined(HIOP_USE_CUDA)
#include <cuda_runtime.h>
#include <resolve/LinSolverDirectCuSolverGLU.hpp>
#include <resolve/LinSolverDirectCuSolverRf.hpp>
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#elif defined(HIOP_USE_HIP)
#include <hip/hip_runtime.h>
#include <resolve/LinSolverDirectRocSolverRf.hpp>
#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

namespace hiop
{
namespace
{

#if defined(HIOP_USE_CUDA)

bool copy_device_to_host(hiopNlpFormulation* nlp, void* destination, const void* source, size_t bytes, const char* operation)
{
  const cudaError_t status = cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost);

  if(status == cudaSuccess) {
    return true;
  }

  nlp->log->printf(hovError, "CUDA failure during %s: %s\n", operation, cudaGetErrorString(status));

  return false;
}

bool copy_device_to_device(hiopNlpFormulation* nlp,
                           void* destination,
                           const void* source,
                           size_t bytes,
                           const char* operation)
{
  const cudaError_t status = cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice);

  if(status == cudaSuccess) {
    return true;
  }

  nlp->log->printf(hovError, "CUDA failure during %s: %s\n", operation, cudaGetErrorString(status));

  return false;
}

#elif defined(HIOP_USE_HIP)

bool copy_device_to_host(hiopNlpFormulation* nlp, void* destination, const void* source, size_t bytes, const char* operation)
{
  const hipError_t status = hipMemcpy(destination, source, bytes, hipMemcpyDeviceToHost);

  if(status == hipSuccess) {
    return true;
  }

  nlp->log->printf(hovError, "HIP failure during %s: %s\n", operation, hipGetErrorString(status));

  return false;
}

bool copy_device_to_device(hiopNlpFormulation* nlp,
                           void* destination,
                           const void* source,
                           size_t bytes,
                           const char* operation)
{
  const hipError_t status = hipMemcpy(destination, source, bytes, hipMemcpyDeviceToDevice);

  if(status == hipSuccess) {
    return true;
  }

  nlp->log->printf(hovError, "HIP failure during %s: %s\n", operation, hipGetErrorString(status));

  return false;
}

#endif

#if defined(HIOP_USE_CUDA) || defined(HIOP_USE_HIP)

/**
 * @brief Map elements of one array to the other.
 */
template<typename T, typename I>
__global__ void mapArraysKernel(T* dst, const T* src, const I* mapidx, I n)
{
  I tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid < n) {
    dst[tid] = src[mapidx[tid]];
  }
}

/**
 * @brief Add HiOp's separate diagonal entries to CSR values.
 */
template<typename T, typename I>
__global__ void addToArrayKernel(T* dst, const T* src, const I* mapidx, I n, I nnz)
{
  I tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid < n) {
    if(mapidx[tid] != -1) {
      dst[mapidx[tid]] += src[nnz - n + tid];
    }
  }
}

#endif

}  // namespace

hiopLinSolverSymSparseEVLOSER::hiopLinSolverSymSparseEVLOSER(const int& n, const int& nnz, hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparse(n, nnz, nlp),
      M_host_(nullptr),
      use_device_(false),
      n_(n),
      nnz_(0),
      ordering_(1),
      index_convert_CSR2Triplet_host_(nullptr),
      index_convert_extra_Diag2CSR_host_(nullptr),
      index_convert_CSR2Triplet_device_(nullptr),
      index_convert_extra_Diag2CSR_device_(nullptr),
      factorizationSetupSucc_(0),
      is_first_call_(true),
      matrix_(nullptr),
      rhs_(nullptr),
      solution_(nullptr),
      factorization_solver_(nullptr)
#if defined(HIOP_USE_CUDA)
      ,
      cuda_workspace_(nullptr),
      cuda_glu_solver_(nullptr),
      cuda_rf_solver_(nullptr)
#elif defined(HIOP_USE_HIP)
      ,
      hip_workspace_(nullptr),
      hip_rf_solver_(nullptr)
#endif
      ,
      refactorization_mode_(RefactorizationMode::CPU_KLU)
{
  const std::string mem_space = nlp_->options->GetString("mem_space");

  if(mem_space == "host" || mem_space == "default") {
    use_device_ = false;
  } else if(mem_space == "device") {
#if defined(HIOP_USE_CUDA) || defined(HIOP_USE_HIP)
    use_device_ = true;

    M_host_ = LinearAlgebraFactory::create_matrix_sparse("default", n, n, nnz);
#else
    nlp_->log->printf(hovError,
                      "EVLOSER device execution requires a CUDA or HIP build.\n");
    std::abort();
#endif
  } else {
    nlp_->log->printf(hovError,
                      "Memory space %s is incompatible with EVLOSER.\n",
                      mem_space.c_str());
    std::abort();
  }

  const std::string ordering = nlp_->options->GetString("linear_solver_sparse_ordering");

  if(ordering == "amd_ssparse") {
    ordering_ = 0;
  } else if(ordering == "colamd_ssparse") {
    ordering_ = 1;
  } else {
    nlp_->log->printf(hovWarning,
                      "Ordering %s is not supported by EVLOSER; "
                      "using colamd_ssparse.\n",
                      ordering.c_str());
    ordering_ = 1;
  }

  const std::string factorization = nlp_->options->GetString("resolve_factorization");

  if(factorization != "klu") {
    nlp_->log->printf(hovWarning,
                      "Factorization %s is not supported by EVLOSER; "
                      "using KLU.\n",
                      factorization.c_str());
  }

  const std::string refactorization = nlp_->options->GetString("resolve_refactorization");

  factorization_solver_ = new ReSolve::LinSolverDirectKLU();
  factorization_solver_->setOrdering(ordering_);
  factorization_solver_->setHaltIfSingular(true);


#if defined(HIOP_USE_CUDA)
  if(use_device_) {
    cuda_workspace_ = new ReSolve::LinAlgWorkspaceCUDA();
    cuda_workspace_->initializeHandles();

    if(refactorization == "rf") {
      refactorization_mode_ = RefactorizationMode::CUDA_RF;

      cuda_rf_solver_ =
          new ReSolve::LinSolverDirectCuSolverRf(cuda_workspace_);
    } else {
      if(refactorization != "glu") {
        nlp_->log->printf(
            hovWarning,
            "Unsupported CUDA refactorization %s; using GLU.\n",
            refactorization.c_str()
        );
      }

      refactorization_mode_ = RefactorizationMode::CUDA_GLU;

      cuda_glu_solver_ =
          new ReSolve::LinSolverDirectCuSolverGLU(cuda_workspace_);
    }
  }
#elif defined(HIOP_USE_HIP)
  if(use_device_) {
    if(refactorization == "glu") {
      nlp_->log->printf(
          hovWarning,
          "GLU is unavailable with HIP; using rocSolverRf.\n"
      );
    } else if(refactorization != "rf") {
      nlp_->log->printf(
          hovWarning,
          "Unsupported HIP refactorization %s; using rocSolverRf.\n",
          refactorization.c_str()
      );
    }

    refactorization_mode_ = RefactorizationMode::HIP_RF;

    hip_workspace_ = new ReSolve::LinAlgWorkspaceHIP();
    hip_workspace_->initializeHandles();

    hip_rf_solver_ =
        new ReSolve::LinSolverDirectRocSolverRf(hip_workspace_);
  }
#endif

  rhs_ = new ReSolve::vector::Vector(n_);
  solution_ = new ReSolve::vector::Vector(n_);

  const auto vector_memory = use_device_ ? ReSolve::memory::DEVICE : ReSolve::memory::HOST;

  rhs_->allocate(vector_memory);
  solution_->allocate(vector_memory);

  if(rhs_->getData(vector_memory) == nullptr || solution_->getData(vector_memory) == nullptr) {
    nlp_->log->printf(hovError, "Failed to allocate ReSolve vectors.\n");
    std::abort();
  }

  nlp_->log->printf(hovSummary, "Ordering: %d\n", ordering_);

  nlp_->log->printf(hovSummary, "Factorization: klu\n");

  switch(refactorization_mode_) {
    case RefactorizationMode::CPU_KLU:
      nlp_->log->printf(
          hovSummary,
          "Refactorization: klu\n"
      );
      break;

#if defined(HIOP_USE_CUDA)
    case RefactorizationMode::CUDA_GLU:
      nlp_->log->printf(
          hovSummary,
          "Refactorization: glu\n"
      );
      break;

    case RefactorizationMode::CUDA_RF:
      nlp_->log->printf(
          hovSummary,
          "Refactorization: rf\n"
      );
      break;
#elif defined(HIOP_USE_HIP)
    case RefactorizationMode::HIP_RF:
      nlp_->log->printf(
          hovSummary,
          "Refactorization: rocSolverRf\n"
      );
      break;
#endif
  }

  nlp_->log->printf(hovSummary, "Use IR: no\n");
}

hiopLinSolverSymSparseEVLOSER::~hiopLinSolverSymSparseEVLOSER()
{
#if defined(HIOP_USE_CUDA)
  delete cuda_glu_solver_;
  cuda_glu_solver_ = nullptr;

  delete cuda_rf_solver_;
  cuda_rf_solver_ = nullptr;
#elif defined(HIOP_USE_HIP)
  delete hip_rf_solver_;
  hip_rf_solver_ = nullptr;
#endif

  delete factorization_solver_;
  factorization_solver_ = nullptr;

  delete rhs_;
  rhs_ = nullptr;

  delete solution_;
  solution_ = nullptr;

  delete matrix_;
  matrix_ = nullptr;

#if defined(HIOP_USE_CUDA)
  delete cuda_workspace_;
  cuda_workspace_ = nullptr;
#elif defined(HIOP_USE_HIP)
  delete hip_workspace_;
  hip_workspace_ = nullptr;
#endif

  delete M_host_;
  M_host_ = nullptr;

  delete[] index_convert_CSR2Triplet_host_;
  index_convert_CSR2Triplet_host_ = nullptr;

  delete[] index_convert_extra_Diag2CSR_host_;
  index_convert_extra_Diag2CSR_host_ = nullptr;

#if defined(HIOP_USE_CUDA)
  if(index_convert_CSR2Triplet_device_ != nullptr) {
    cudaFree(index_convert_CSR2Triplet_device_);
    index_convert_CSR2Triplet_device_ = nullptr;
  }

  if(index_convert_extra_Diag2CSR_device_ != nullptr) {
    cudaFree(index_convert_extra_Diag2CSR_device_);
    index_convert_extra_Diag2CSR_device_ = nullptr;
  }
#elif defined(HIOP_USE_HIP)
  if(index_convert_CSR2Triplet_device_ != nullptr) {
    hipFree(index_convert_CSR2Triplet_device_);
    index_convert_CSR2Triplet_device_ = nullptr;
  }

  if(index_convert_extra_Diag2CSR_device_ != nullptr) {
    hipFree(index_convert_extra_Diag2CSR_device_);
    index_convert_extra_Diag2CSR_device_ = nullptr;
  }
#endif
}

int hiopLinSolverSymSparseEVLOSER::matrixChanged()
{
  assert(M_ != nullptr);
  assert(n_ == M_->n());
  assert(M_->n() == M_->m());
  assert(n_ > 0);
  assert(factorization_solver_ != nullptr);

  nlp_->runStats.linsolv.tmFactTime.start();

  if(is_first_call_) {
    if(firstCall() != 0) {
      nlp_->runStats.linsolv.tmFactTime.stop();
      return -1;
    }
  } else {
    if(update_matrix_values() != 0) {
      nlp_->runStats.linsolv.tmFactTime.stop();
      return -1;
    }
  }

  int status = 0;

  if(factorizationSetupSucc_ == 0) {
    status = factorization_solver_->factorize();

    if(status != 0) {
      nlp_->log->printf(
          hovWarning,
          "EVLOSER KLU factorization failed. Regularizing ...\n"
      );

      factorizationSetupSucc_ = 0;

      nlp_->runStats.linsolv.tmFactTime.stop();
      return -1;
    }

    status = setup_refactorization_solver();

    if(status == 0) {
      switch(refactorization_mode_) {
        case RefactorizationMode::CPU_KLU:
          break;

#if defined(HIOP_USE_CUDA)
        case RefactorizationMode::CUDA_GLU:
          // GLU setup performs its initial numeric factorization.
          break;

        case RefactorizationMode::CUDA_RF:
          // RF setup imports and analyzes the KLU factors.
          // Perform the initial RF numeric refactorization.
          status = refactorize_selected_solver();
          break;
#elif defined(HIOP_USE_HIP)
        case RefactorizationMode::HIP_RF:
          // HIP RF setup analyzes the imported KLU factors.
          // Perform the initial RF numeric refactorization.
          status = refactorize_selected_solver();
          break;
#endif
      }
    }

    if(status != 0) {
      nlp_->log->printf(
          hovWarning,
          "EVLOSER refactorization solver setup or initial "
          "refactorization failed. Regularizing ...\n"
      );

      factorizationSetupSucc_ = 0;

      nlp_->runStats.linsolv.tmFactTime.stop();
      return -1;
    }

    factorizationSetupSucc_ = 1;

    nlp_->log->printf(
        hovScalars,
        "EVLOSER factorization setup successful.\n"
    );
  } else {
    status = refactorize_selected_solver();

    if(status != 0) {
      nlp_->log->printf(
          hovWarning,
          "EVLOSER refactorization failed. Regularizing ...\n"
      );

      factorizationSetupSucc_ = 0;

      nlp_->runStats.linsolv.tmFactTime.stop();
      return -1;
    }
  }

  nlp_->runStats.linsolv.tmFactTime.stop();
  return 0;
}

bool hiopLinSolverSymSparseEVLOSER::solve(hiopVector& x)
{
  assert(M_ != nullptr);
  assert(n_ == M_->n());
  assert(M_->n() == M_->m());
  assert(n_ > 0);
  assert(x.get_size() == M_->n());

  if(factorizationSetupSucc_ == 0) {
    nlp_->log->printf(
        hovError,
        "EVLOSER solve requested without a valid factorization.\n"
    );
    return false;
  }

  double* x_data = x.local_data();

  if(x_data == nullptr) {
    nlp_->log->printf(
        hovError,
        "Failed to access the HiOp solve vector.\n"
    );
    return false;
  }

  nlp_->runStats.linsolv.tmTriuSolves.start();

#if defined(HIOP_USE_CUDA) || defined(HIOP_USE_HIP)
  if(use_device_) {
    if(rhs_->copyFromExternal(
           x_data,
           ReSolve::memory::DEVICE,
           ReSolve::memory::DEVICE
       ) != 0) {
      nlp_->log->printf(
          hovError,
          "Failed to copy the device right-hand side into ReSolve.\n"
      );

      nlp_->runStats.linsolv.tmTriuSolves.stop();
      return false;
    }

    if(solve_selected_solver() != 0) {
      nlp_->log->printf(
          hovError,
          "EVLOSER device solve failed.\n"
      );

      nlp_->runStats.linsolv.tmTriuSolves.stop();
      return false;
    }

    const double* solution_data =
        solution_->getData(ReSolve::memory::DEVICE);

    if(solution_data == nullptr) {
      nlp_->log->printf(
          hovError,
          "Failed to access the ReSolve device solution.\n"
      );

      nlp_->runStats.linsolv.tmTriuSolves.stop();
      return false;
    }

    if(!copy_device_to_device(
           nlp_,
           x_data,
           solution_data,
           sizeof(double) * n_,
           "copying the ReSolve solution into HiOp"
       )) {
      nlp_->runStats.linsolv.tmTriuSolves.stop();
      return false;
    }

    nlp_->runStats.linsolv.tmTriuSolves.stop();
    return true;
  }
#endif

  double* rhs_data =
      rhs_->getData(ReSolve::memory::HOST);

  if(rhs_data == nullptr) {
    nlp_->log->printf(
        hovError,
        "Failed to access the ReSolve host right-hand side.\n"
    );

    nlp_->runStats.linsolv.tmTriuSolves.stop();
    return false;
  }

  std::copy(
      x_data,
      x_data + n_,
      rhs_data
  );

  rhs_->setDataUpdated(ReSolve::memory::HOST);

  if(solve_selected_solver() != 0) {
    nlp_->log->printf(
        hovError,
        "EVLOSER host solve failed.\n"
    );

    nlp_->runStats.linsolv.tmTriuSolves.stop();
    return false;
  }

  const double* solution_data =
      solution_->getData(ReSolve::memory::HOST);

  if(solution_data == nullptr) {
    nlp_->log->printf(
        hovError,
        "Failed to access the ReSolve host solution.\n"
    );

    nlp_->runStats.linsolv.tmTriuSolves.stop();
    return false;
  }

  std::copy(
      solution_data,
      solution_data + n_,
      x_data
  );

  nlp_->runStats.linsolv.tmTriuSolves.stop();
  return true;
}

int hiopLinSolverSymSparseEVLOSER::firstCall()
{
  assert(M_ != nullptr);
  assert(n_ == M_->n());
  assert(M_->n() == M_->m());
  assert(n_ > 0);
  assert(factorization_solver_ != nullptr);

#if defined(HIOP_USE_CUDA) || defined(HIOP_USE_HIP)
  if(use_device_) {
    assert(M_host_ != nullptr);

    if(!copy_device_to_host(nlp_,
                            M_host_->M(),
                            M_->M(),
                            sizeof(double) * M_->numberOfNonzeros(),
                            "copying EVLOSER values to the host")) {
      return -1;
    }

    if(!copy_device_to_host(nlp_,
                            M_host_->i_row(),
                            M_->i_row(),
                            sizeof(index_type) * M_->numberOfNonzeros(),
                            "copying EVLOSER row indices to the host")) {
      return -1;
    }

    if(!copy_device_to_host(nlp_,
                            M_host_->j_col(),
                            M_->j_col(),
                            sizeof(index_type) * M_->numberOfNonzeros(),
                            "copying EVLOSER column indices to the host")) {
      return -1;
    }
  }
#endif

  compute_nnz();

  // Clean up partial allocations before retrying first-call setup.
  delete matrix_;
  matrix_ = nullptr;

  delete[] index_convert_CSR2Triplet_host_;
  index_convert_CSR2Triplet_host_ = nullptr;

  delete[] index_convert_extra_Diag2CSR_host_;
  index_convert_extra_Diag2CSR_host_ = nullptr;

#if defined(HIOP_USE_CUDA)
  if(index_convert_CSR2Triplet_device_ != nullptr) {
    const cudaError_t status =
        cudaFree(index_convert_CSR2Triplet_device_);

    if(status != cudaSuccess) {
      nlp_->log->printf(
          hovError,
          "CUDA failure freeing the CSR-to-triplet mapping: %s\n",
          cudaGetErrorString(status)
      );
      return -1;
    }

    index_convert_CSR2Triplet_device_ = nullptr;
  }

  if(index_convert_extra_Diag2CSR_device_ != nullptr) {
    const cudaError_t status =
        cudaFree(index_convert_extra_Diag2CSR_device_);

    if(status != cudaSuccess) {
      nlp_->log->printf(
          hovError,
          "CUDA failure freeing the diagonal-to-CSR mapping: %s\n",
          cudaGetErrorString(status)
      );
      return -1;
    }

    index_convert_extra_Diag2CSR_device_ = nullptr;
  }
#elif defined(HIOP_USE_HIP)
  if(index_convert_CSR2Triplet_device_ != nullptr) {
    const hipError_t status =
        hipFree(index_convert_CSR2Triplet_device_);

    if(status != hipSuccess) {
      nlp_->log->printf(
          hovError,
          "HIP failure freeing the CSR-to-triplet mapping: %s\n",
          hipGetErrorString(status)
      );
      return -1;
    }

    index_convert_CSR2Triplet_device_ = nullptr;
  }

  if(index_convert_extra_Diag2CSR_device_ != nullptr) {
    const hipError_t status =
        hipFree(index_convert_extra_Diag2CSR_device_);

    if(status != hipSuccess) {
      nlp_->log->printf(
          hovError,
          "HIP failure freeing the diagonal-to-CSR mapping: %s\n",
          hipGetErrorString(status)
      );
      return -1;
    }

    index_convert_extra_Diag2CSR_device_ = nullptr;
  }
#endif

  matrix_ =
      new ReSolve::matrix::Csr(
          n_,
          n_,
          nnz_,
          true,
          true
      );

  if(matrix_->allocateMatrixData(ReSolve::memory::HOST) != 0) {
    nlp_->log->printf(
        hovError,
        "Failed to allocate the ReSolve CSR matrix.\n"
    );
    return -1;
  }

  if(set_csr_indices_values() != 0) {
    nlp_->log->printf(
        hovError,
        "Failed to construct the ReSolve CSR matrix.\n"
    );
    return -1;
  }

  if(matrix_->setUpdated(ReSolve::memory::HOST) != 0) {
    nlp_->log->printf(
        hovError,
        "Failed to mark the ReSolve matrix as updated.\n"
    );
    return -1;
  }

#if defined(HIOP_USE_CUDA) || defined(HIOP_USE_HIP)
  if(use_device_) {
    if(matrix_->allocateMatrixData(ReSolve::memory::DEVICE) != 0) {
      nlp_->log->printf(
          hovError,
          "Failed to allocate ReSolve matrix device storage.\n"
      );
      return -1;
    }

    if(matrix_->syncData(ReSolve::memory::DEVICE) != 0) {
      nlp_->log->printf(
          hovError,
          "Failed to copy the ReSolve matrix to device memory.\n"
      );
      return -1;
    }
  }
#endif

  if(factorization_solver_->setup(matrix_) != 0) {
    nlp_->log->printf(
        hovError,
        "ReSolve KLU setup failed.\n"
    );
    return -1;
  }

  if(factorization_solver_->analyze() != 0) {
    nlp_->log->printf(
        hovError,
        "ReSolve KLU symbolic analysis failed.\n"
    );
    return -1;
  }

  is_first_call_ = false;
  return 0;
}

int hiopLinSolverSymSparseEVLOSER::update_matrix_values()
{
  assert(M_ != nullptr);
  assert(matrix_ != nullptr);

#if defined(HIOP_USE_CUDA) || defined(HIOP_USE_HIP)
  if(use_device_) {
    double* values =
        matrix_->getValues(ReSolve::memory::DEVICE);

    const double* source_values = M_->M();

    if(values == nullptr ||
       source_values == nullptr ||
       index_convert_CSR2Triplet_device_ == nullptr ||
       index_convert_extra_Diag2CSR_device_ == nullptr) {
      nlp_->log->printf(
          hovError,
          "Failed to access EVLOSER device matrix data.\n"
      );
      return -1;
    }

    const int blocksize = 512;
    int gridsize = (nnz_ + blocksize - 1) / blocksize;

    mapArraysKernel<double, int>
        <<<gridsize, blocksize>>>(
            values,
            source_values,
            index_convert_CSR2Triplet_device_,
            nnz_
        );

#if defined(HIOP_USE_CUDA)
    cudaError_t launch_status = cudaGetLastError();

    if(launch_status != cudaSuccess) {
      nlp_->log->printf(
          hovError,
          "CUDA failure launching the CSR value-mapping kernel: %s\n",
          cudaGetErrorString(launch_status)
      );
      return -1;
    }
#elif defined(HIOP_USE_HIP)
    hipError_t launch_status = hipGetLastError();

    if(launch_status != hipSuccess) {
      nlp_->log->printf(
          hovError,
          "HIP failure launching the CSR value-mapping kernel: %s\n",
          hipGetErrorString(launch_status)
      );
      return -1;
    }
#endif

    gridsize = (n_ + blocksize - 1) / blocksize;

    addToArrayKernel<double, int>
        <<<gridsize, blocksize>>>(
            values,
            source_values,
            index_convert_extra_Diag2CSR_device_,
            n_,
            M_->numberOfNonzeros()
        );

#if defined(HIOP_USE_CUDA)
    launch_status = cudaGetLastError();

    if(launch_status != cudaSuccess) {
      nlp_->log->printf(
          hovError,
          "CUDA failure launching the diagonal-update kernel: %s\n",
          cudaGetErrorString(launch_status)
      );
      return -1;
    }
#elif defined(HIOP_USE_HIP)
    launch_status = hipGetLastError();

    if(launch_status != hipSuccess) {
      nlp_->log->printf(
          hovError,
          "HIP failure launching the diagonal-update kernel: %s\n",
          hipGetErrorString(launch_status)
      );
      return -1;
    }
#endif

    if(matrix_->setUpdated(ReSolve::memory::DEVICE) != 0) {
      nlp_->log->printf(
          hovError,
          "Failed to mark EVLOSER device matrix values as updated.\n"
      );
      return -1;
    }

    if(factorizationSetupSucc_ == 0) {
      if(matrix_->syncData(ReSolve::memory::HOST) != 0) {
        nlp_->log->printf(
            hovError,
            "Failed to synchronize EVLOSER matrix values to the host.\n"
        );
        return -1;
      }
    }

    return 0;
  }
#endif

  double* values =
      matrix_->getValues(ReSolve::memory::HOST);

  if(values == nullptr) {
    nlp_->log->printf(
        hovError,
        "Failed to access EVLOSER host matrix values.\n"
    );
    return -1;
  }

  for(int k = 0; k < nnz_; ++k) {
    values[k] = M_->M()[index_convert_CSR2Triplet_host_[k]];
  }

  for(int i = 0; i < n_; ++i) {
    if(index_convert_extra_Diag2CSR_host_[i] != -1) {
      values[index_convert_extra_Diag2CSR_host_[i]] += M_->M()[M_->numberOfNonzeros() - n_ + i];
    }
  }

  if(matrix_->setUpdated(ReSolve::memory::HOST) != 0) {
    nlp_->log->printf(
       hovError,
       "Failed to mark EVLOSER host matrix values as updated.\n"
    );
    return -1;
  }

  return 0;
}

hiopMatrixSparse*
hiopLinSolverSymSparseEVLOSER::host_matrix() const
{
  return use_device_ ? M_host_ : M_;
}

void hiopLinSolverSymSparseEVLOSER::compute_nnz()
{
  hiopMatrixSparse* source = host_matrix();
  assert(source != nullptr);

  nnz_ = n_;

  for(int k = 0; k < source->numberOfNonzeros() - n_; ++k) {
    if(source->i_row()[k] != source->j_col()[k]) {
      nnz_ += 2;
    }
  }
}

int hiopLinSolverSymSparseEVLOSER::set_csr_indices_values()
{
  assert(M_ != nullptr);
  assert(matrix_ != nullptr);

  hiopMatrixSparse* source = host_matrix();
  assert(source != nullptr);

  ReSolve::index_type* row_ptr = matrix_->getRowData(ReSolve::memory::HOST);

  ReSolve::index_type* col_idx = matrix_->getColData(ReSolve::memory::HOST);

  double* values = matrix_->getValues(ReSolve::memory::HOST);

  std::fill(row_ptr, row_ptr + n_ + 1, 0);

  for(int k = 0; k < source->numberOfNonzeros() - n_; ++k) {
    if(source->i_row()[k] != source->j_col()[k]) {
      row_ptr[source->i_row()[k] + 1]++;
      row_ptr[source->j_col()[k] + 1]++;
    }
  }

  for(int i = 0; i < n_; ++i) {
    row_ptr[i + 1]++;
  }

  for(int i = 1; i < n_ + 1; ++i) {
    row_ptr[i] += row_ptr[i - 1];
  }

  assert(nnz_ == row_ptr[n_]);

  index_convert_CSR2Triplet_host_ = new int[nnz_];

  index_convert_extra_Diag2CSR_host_ = new int[n_];

  int* nnz_each_row_tmp = new int[n_]{0};

  int total_nnz_tmp{0};
  int nnz_tmp{0};
  int rowID_tmp{0};
  int colID_tmp{0};

  for(int i = 0; i < n_; ++i) {
    index_convert_extra_Diag2CSR_host_[i] = -1;
  }

  for(int k = 0; k < source->numberOfNonzeros() - n_; ++k) {
    rowID_tmp = source->i_row()[k];
    colID_tmp = source->j_col()[k];

    if(rowID_tmp == colID_tmp) {
      nnz_tmp = nnz_each_row_tmp[rowID_tmp] + row_ptr[rowID_tmp];

      col_idx[nnz_tmp] = colID_tmp;
      values[nnz_tmp] = source->M()[k];

      index_convert_CSR2Triplet_host_[nnz_tmp] = k;

      values[nnz_tmp] += source->M()[source->numberOfNonzeros() - n_ + rowID_tmp];

      index_convert_extra_Diag2CSR_host_[rowID_tmp] = nnz_tmp;

      nnz_each_row_tmp[rowID_tmp]++;
      total_nnz_tmp++;
    } else {
      nnz_tmp = nnz_each_row_tmp[rowID_tmp] + row_ptr[rowID_tmp];

      col_idx[nnz_tmp] = colID_tmp;
      values[nnz_tmp] = source->M()[k];

      index_convert_CSR2Triplet_host_[nnz_tmp] = k;

      nnz_tmp = nnz_each_row_tmp[colID_tmp] + row_ptr[colID_tmp];

      col_idx[nnz_tmp] = rowID_tmp;
      values[nnz_tmp] = source->M()[k];

      index_convert_CSR2Triplet_host_[nnz_tmp] = k;

      nnz_each_row_tmp[rowID_tmp]++;
      nnz_each_row_tmp[colID_tmp]++;
      total_nnz_tmp += 2;
    }
  }

  for(int i = 0; i < n_; ++i) {
    if(nnz_each_row_tmp[i] != row_ptr[i + 1] - row_ptr[i]) {
      assert(nnz_each_row_tmp[i] == row_ptr[i + 1] - row_ptr[i] - 1);

      nnz_tmp = nnz_each_row_tmp[i] + row_ptr[i];

      col_idx[nnz_tmp] = i;

      values[nnz_tmp] = source->M()[source->numberOfNonzeros() - n_ + i];

      index_convert_CSR2Triplet_host_[nnz_tmp] = source->numberOfNonzeros() - n_ + i;

      total_nnz_tmp++;

      std::vector<int> permutation(row_ptr[i + 1] - row_ptr[i]);

      std::iota(permutation.begin(), permutation.end(), 0);

      std::sort(permutation.begin(), permutation.end(), [&](int a, int b) {
        return col_idx[a + row_ptr[i]] < col_idx[b + row_ptr[i]];
      });

      reorder(values + row_ptr[i], permutation, row_ptr[i + 1] - row_ptr[i]);

      reorder(index_convert_CSR2Triplet_host_ + row_ptr[i], permutation, row_ptr[i + 1] - row_ptr[i]);

      std::sort(col_idx + row_ptr[i], col_idx + row_ptr[i + 1]);
    }
  }

  assert(total_nnz_tmp == nnz_);

#if defined(HIOP_USE_CUDA)
  if(use_device_) {
    cudaError_t status =
        cudaMalloc(
            reinterpret_cast<void**>(
                &index_convert_CSR2Triplet_device_
            ),
            sizeof(int) * nnz_
        );

    if(status != cudaSuccess) {
      nlp_->log->printf(
          hovError,
          "CUDA failure allocating the CSR-to-triplet mapping: %s\n",
          cudaGetErrorString(status)
      );

      delete[] nnz_each_row_tmp;
      return -1;
    }

    status =
        cudaMalloc(
            reinterpret_cast<void**>(
                &index_convert_extra_Diag2CSR_device_
            ),
            sizeof(int) * n_
        );

    if(status != cudaSuccess) {
      nlp_->log->printf(
          hovError,
          "CUDA failure allocating the diagonal-to-CSR mapping: %s\n",
          cudaGetErrorString(status)
      );

      cudaFree(index_convert_CSR2Triplet_device_);
      index_convert_CSR2Triplet_device_ = nullptr;

      delete[] nnz_each_row_tmp;
      return -1;
    }

    status =
        cudaMemcpy(
            index_convert_CSR2Triplet_device_,
            index_convert_CSR2Triplet_host_,
            sizeof(int) * nnz_,
            cudaMemcpyHostToDevice
        );

    if(status != cudaSuccess) {
      nlp_->log->printf(
          hovError,
          "CUDA failure copying the CSR-to-triplet mapping: %s\n",
          cudaGetErrorString(status)
      );

      cudaFree(index_convert_CSR2Triplet_device_);
      cudaFree(index_convert_extra_Diag2CSR_device_);
      index_convert_CSR2Triplet_device_ = nullptr;
      index_convert_extra_Diag2CSR_device_ = nullptr;

      delete[] nnz_each_row_tmp;
      return -1;
    }

    status =
        cudaMemcpy(
            index_convert_extra_Diag2CSR_device_,
            index_convert_extra_Diag2CSR_host_,
            sizeof(int) * n_,
            cudaMemcpyHostToDevice
        );

    if(status != cudaSuccess) {
      nlp_->log->printf(
          hovError,
          "CUDA failure copying the diagonal-to-CSR mapping: %s\n",
          cudaGetErrorString(status)
      );

      cudaFree(index_convert_CSR2Triplet_device_);
      cudaFree(index_convert_extra_Diag2CSR_device_);
      index_convert_CSR2Triplet_device_ = nullptr;
      index_convert_extra_Diag2CSR_device_ = nullptr;

      delete[] nnz_each_row_tmp;
      return -1;
    }
  }
#elif defined(HIOP_USE_HIP)
  if(use_device_) {
    hipError_t status =
        hipMalloc(
            reinterpret_cast<void**>(
                &index_convert_CSR2Triplet_device_
            ),
            sizeof(int) * nnz_
        );

    if(status != hipSuccess) {
      nlp_->log->printf(
          hovError,
          "HIP failure allocating the CSR-to-triplet mapping: %s\n",
          hipGetErrorString(status)
      );

      delete[] nnz_each_row_tmp;
      return -1;
    }

    status =
        hipMalloc(
            reinterpret_cast<void**>(
                &index_convert_extra_Diag2CSR_device_
            ),
            sizeof(int) * n_
        );

    if(status != hipSuccess) {
      nlp_->log->printf(
          hovError,
          "HIP failure allocating the diagonal-to-CSR mapping: %s\n",
          hipGetErrorString(status)
      );

      hipFree(index_convert_CSR2Triplet_device_);
      index_convert_CSR2Triplet_device_ = nullptr;

      delete[] nnz_each_row_tmp;
      return -1;
    }

    status =
        hipMemcpy(
            index_convert_CSR2Triplet_device_,
            index_convert_CSR2Triplet_host_,
            sizeof(int) * nnz_,
            hipMemcpyHostToDevice
        );

    if(status != hipSuccess) {
      nlp_->log->printf(
          hovError,
          "HIP failure copying the CSR-to-triplet mapping: %s\n",
          hipGetErrorString(status)
      );

      hipFree(index_convert_CSR2Triplet_device_);
      hipFree(index_convert_extra_Diag2CSR_device_);
      index_convert_CSR2Triplet_device_ = nullptr;
      index_convert_extra_Diag2CSR_device_ = nullptr;

      delete[] nnz_each_row_tmp;
      return -1;
    }

    status =
        hipMemcpy(
            index_convert_extra_Diag2CSR_device_,
            index_convert_extra_Diag2CSR_host_,
            sizeof(int) * n_,
            hipMemcpyHostToDevice
        );

    if(status != hipSuccess) {
      nlp_->log->printf(
          hovError,
          "HIP failure copying the diagonal-to-CSR mapping: %s\n",
          hipGetErrorString(status)
      );

      hipFree(index_convert_CSR2Triplet_device_);
      hipFree(index_convert_extra_Diag2CSR_device_);
      index_convert_CSR2Triplet_device_ = nullptr;
      index_convert_extra_Diag2CSR_device_ = nullptr;

      delete[] nnz_each_row_tmp;
      return -1;
    }
  }
#endif

  delete[] nnz_each_row_tmp;
  return 0;
}

int hiopLinSolverSymSparseEVLOSER::setup_refactorization_solver()
{
  if(refactorization_mode_ == RefactorizationMode::CPU_KLU) {
    return 0;
  }

#if defined(HIOP_USE_CUDA) || defined(HIOP_USE_HIP)
  auto* L =
      dynamic_cast<ReSolve::matrix::Csr*>(
          factorization_solver_->getLFactor()
      );

  auto* U =
      dynamic_cast<ReSolve::matrix::Csr*>(
          factorization_solver_->getUFactor()
      );

  ReSolve::index_type* P =
      factorization_solver_->getPOrdering();

  ReSolve::index_type* Q =
      factorization_solver_->getQOrdering();

  if(L == nullptr ||
     U == nullptr ||
     P == nullptr ||
     Q == nullptr) {
    nlp_->log->printf(
        hovError,
        "Failed to extract KLU factors for EVLOSER.\n"
    );
    return -1;
  }
#endif

  switch(refactorization_mode_) {
    case RefactorizationMode::CPU_KLU:
      return 0;

#if defined(HIOP_USE_CUDA)
    case RefactorizationMode::CUDA_GLU:
      assert(cuda_glu_solver_ != nullptr);

      return cuda_glu_solver_->setup(
          matrix_,
          L,
          U,
          P,
          Q
      );

    case RefactorizationMode::CUDA_RF:
      assert(cuda_rf_solver_ != nullptr);

      if(L->getRowData(ReSolve::memory::DEVICE) == nullptr ||
        L->getColData(ReSolve::memory::DEVICE) == nullptr ||
        L->getValues(ReSolve::memory::DEVICE) == nullptr) {
        if(L->allocateMatrixData(ReSolve::memory::DEVICE) != 0) {
          nlp_->log->printf(
              hovError,
              "Failed to allocate device storage for the EVLOSER KLU L factor.\n"
          );
          return -1;
        }
      }

      if(U->getRowData(ReSolve::memory::DEVICE) == nullptr ||
        U->getColData(ReSolve::memory::DEVICE) == nullptr ||
        U->getValues(ReSolve::memory::DEVICE) == nullptr) {
        if(U->allocateMatrixData(ReSolve::memory::DEVICE) != 0) {
          nlp_->log->printf(
              hovError,
              "Failed to allocate device storage for the EVLOSER KLU U factor.\n"
          );
          return -1;
        }
      }

      return cuda_rf_solver_->setup(
          matrix_,
          L,
          U,
          P,
          Q,
          rhs_
      );
#elif defined(HIOP_USE_HIP)
    case RefactorizationMode::HIP_RF:
      assert(hip_rf_solver_ != nullptr);

      return hip_rf_solver_->setup(
          matrix_,
          L,
          U,
          P,
          Q,
          rhs_
      );
#endif
  }

  return -1;
}

int hiopLinSolverSymSparseEVLOSER::refactorize_selected_solver()
{
  switch(refactorization_mode_) {
    case RefactorizationMode::CPU_KLU:
      assert(factorization_solver_ != nullptr);
      return factorization_solver_->refactorize();

#if defined(HIOP_USE_CUDA)
    case RefactorizationMode::CUDA_GLU:
      assert(cuda_glu_solver_ != nullptr);
      return cuda_glu_solver_->refactorize();

    case RefactorizationMode::CUDA_RF:
      assert(cuda_rf_solver_ != nullptr);
      return cuda_rf_solver_->refactorize();
#elif defined(HIOP_USE_HIP)
    case RefactorizationMode::HIP_RF:
      assert(hip_rf_solver_ != nullptr);
      return hip_rf_solver_->refactorize();
#endif
  }

  return -1;
}

int hiopLinSolverSymSparseEVLOSER::solve_selected_solver()
{
  switch(refactorization_mode_) {
    case RefactorizationMode::CPU_KLU:
      assert(factorization_solver_ != nullptr);
      return factorization_solver_->solve(
          rhs_,
          solution_
      );

#if defined(HIOP_USE_CUDA)
    case RefactorizationMode::CUDA_GLU:
      assert(cuda_glu_solver_ != nullptr);
      return cuda_glu_solver_->solve(
          rhs_,
          solution_
      );

    case RefactorizationMode::CUDA_RF:
      assert(cuda_rf_solver_ != nullptr);
      return cuda_rf_solver_->solve(
          rhs_,
          solution_
      );
#elif defined(HIOP_USE_HIP)
    case RefactorizationMode::HIP_RF:
      assert(hip_rf_solver_ != nullptr);
      return hip_rf_solver_->solve(
          rhs_,
          solution_
      );
#endif
  }

  return -1;
}

}  // namespace hiop
