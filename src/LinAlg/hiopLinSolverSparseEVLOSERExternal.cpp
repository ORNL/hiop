#include "hiopLinSolverSparseEVLOSERProvider.hpp"

#include "LinAlgFactory.hpp"
#include "hiopLinSolver.hpp"
#include "hiopMatrixSparse.hpp"
#include "hiopVector.hpp"

#include "hiop_blasdefs.hpp"

#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA)
#include <cuda_runtime.h>
#include <resolve/LinSolverDirectCuSolverRf.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#elif defined(HIOP_USE_HIP) || defined(HAVE_HIP)
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

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA)

bool copy_device_to_host(hiopNlpFormulation* nlp,
                         void* destination,
                         const void* source,
                         size_t bytes,
                         const char* operation)
{
  const cudaError_t status =
      cudaMemcpy(
          destination,
          source,
          bytes,
          cudaMemcpyDeviceToHost);

  if(status == cudaSuccess) {
    return true;
  }

  nlp->log->printf(
      hovError,
      "CUDA failure during %s: %s\n",
      operation,
      cudaGetErrorString(status));

  return false;
}

bool copy_device_to_device(hiopNlpFormulation* nlp,
                           void* destination,
                           const void* source,
                           size_t bytes,
                           const char* operation)
{
  const cudaError_t status =
      cudaMemcpy(
          destination,
          source,
          bytes,
          cudaMemcpyDeviceToDevice);

  if(status == cudaSuccess) {
    return true;
  }

  nlp->log->printf(
      hovError,
      "CUDA failure during %s: %s\n",
      operation,
      cudaGetErrorString(status));

  return false;
}

#elif defined(HIOP_USE_HIP) || defined(HAVE_HIP)

bool copy_device_to_host(hiopNlpFormulation* nlp,
                         void* destination,
                         const void* source,
                         size_t bytes,
                         const char* operation)
{
  const hipError_t status =
      hipMemcpy(
          destination,
          source,
          bytes,
          hipMemcpyDeviceToHost);

  if(status == hipSuccess) {
    return true;
  }

  nlp->log->printf(
      hovError,
      "HIP failure during %s: %s\n",
      operation,
      hipGetErrorString(status));

  return false;
}

bool copy_device_to_device(hiopNlpFormulation* nlp,
                           void* destination,
                           const void* source,
                           size_t bytes,
                           const char* operation)
{
  const hipError_t status =
      hipMemcpy(
          destination,
          source,
          bytes,
          hipMemcpyDeviceToDevice);

  if(status == hipSuccess) {
    return true;
  }

  nlp->log->printf(
      hovError,
      "HIP failure during %s: %s\n",
      operation,
      hipGetErrorString(status));

  return false;
}

#endif

class hiopLinSolverSparseEVLOSERExternal final
    : public hiopLinSolverSparseEVLOSERProvider
{
public:
  hiopLinSolverSparseEVLOSERExternal(const int& n,
                                    const int& nnz,
                                    hiopNlpFormulation* nlp);

  ~hiopLinSolverSparseEVLOSERExternal() override;

  int matrixChanged(hiopMatrixSparse& matrix) override;
  bool solve(hiopVector& x) override;

private:
  int firstCall();
  int update_matrix_values();
  void compute_nnz();
  void set_csr_indices_values();

  int reset_solver();
  hiopMatrixSparse* host_matrix() const;

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  int setup_gpu_refactorization();
#endif

private:
  hiopNlpFormulation* nlp_;

  ReSolve::LinSolverDirectKLU* solver_;

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA)
  ReSolve::LinSolverDirectCuSolverRf* rf_solver_;
#elif defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  ReSolve::LinSolverDirectRocSolverRf* rf_solver_;
  ReSolve::LinAlgWorkspaceHIP* hip_workspace_;
#endif

  ReSolve::matrix::Csr* matrix_;
  ReSolve::vector::Vector* rhs_;
  ReSolve::vector::Vector* solution_;

  hiopMatrixSparse* M_;
  hiopMatrixSparse* M_host_;

  bool use_device_;

  int n_;
  int nnz_;
  int ordering_;

  int* index_convert_CSR2Triplet_host_;
  int* index_convert_extra_Diag2CSR_host_;

  bool is_first_call_;
  bool factorization_valid_;
};

hiopLinSolverSparseEVLOSERExternal::
    hiopLinSolverSparseEVLOSERExternal(const int& n,
                                       const int& nnz,
                                       hiopNlpFormulation* nlp)
    : nlp_{nlp},
      solver_{nullptr},
#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA)
      rf_solver_{nullptr},
#elif defined(HIOP_USE_HIP) || defined(HAVE_HIP)
      rf_solver_{nullptr},
      hip_workspace_{nullptr},
#endif
      matrix_{nullptr},
      rhs_{nullptr},
      solution_{nullptr},
      M_{nullptr},
      M_host_{nullptr},
      use_device_{false},
      n_{n},
      nnz_{0},
      ordering_{1},
      index_convert_CSR2Triplet_host_{nullptr},
      index_convert_extra_Diag2CSR_host_{nullptr},
      is_first_call_{true},
      factorization_valid_{false}
{
  const std::string mem_space =
      nlp_->options->GetString("mem_space");

  if(mem_space == "host" || mem_space == "default") {
    use_device_ = false;
  } else if(mem_space == "device") {
#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
    use_device_ = true;

    M_host_ =
        LinearAlgebraFactory::create_matrix_sparse(
            "default",
            n,
            n,
            nnz);
#else
    nlp_->log->printf(
        hovError,
        "External EVLOSER device execution requires a CUDA or HIP "
        "build.\n");
    std::abort();
#endif
  } else {
    nlp_->log->printf(
        hovError,
        "Memory space %s is incompatible with the external EVLOSER "
        "provider.\n",
        mem_space.c_str());
    std::abort();
  }

  const std::string ordering =
      nlp_->options->GetString(
          "linear_solver_sparse_ordering");

  if(ordering == "amd_ssparse") {
    ordering_ = 0;
  } else if(ordering == "colamd_ssparse") {
    ordering_ = 1;
  } else {
    nlp_->log->printf(
        hovWarning,
        "Ordering %s not compatible with the external EVLOSER "
        "provider, using default ...\n",
        ordering.c_str());
    ordering_ = 1;
  }

  const std::string factorization =
      nlp_->options->GetString("resolve_factorization");

  if(factorization != "klu") {
    nlp_->log->printf(
        hovWarning,
        "Factorization %s not compatible with the external EVLOSER "
        "provider, using KLU ...\n",
        factorization.c_str());
  }

  const std::string refactorization =
      nlp_->options->GetString("resolve_refactorization");

  if(use_device_ && refactorization != "rf") {
    nlp_->log->printf(
        hovWarning,
        "External EVLOSER device execution supports RF "
        "refactorization. Using rf instead of %s.\n",
        refactorization.c_str());
  }

  solver_ = new ReSolve::LinSolverDirectKLU();
  solver_->setOrdering(ordering_);
  solver_->setHaltIfSingular(true);

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA)
  if(use_device_) {
    rf_solver_ =
        new ReSolve::LinSolverDirectCuSolverRf();
  }
#elif defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  if(use_device_) {
    hip_workspace_ =
        new ReSolve::LinAlgWorkspaceHIP();

    hip_workspace_->initializeHandles();

    rf_solver_ =
        new ReSolve::LinSolverDirectRocSolverRf(
            hip_workspace_);
  }
#endif

  rhs_ = new ReSolve::vector::Vector(n_);
  solution_ = new ReSolve::vector::Vector(n_);

  const auto vector_memory =
      use_device_
          ? ReSolve::memory::DEVICE
          : ReSolve::memory::HOST;

  rhs_->allocate(vector_memory);
  solution_->allocate(vector_memory);

  if(rhs_->getData(vector_memory) == nullptr ||
     solution_->getData(vector_memory) == nullptr) {
    nlp_->log->printf(
        hovError,
        "Failed to allocate external ReSolve vectors.\n");
    std::abort();
  }

  nlp_->log->printf(
      hovSummary,
      "EVLOSER provider: external ReSolve\n");

  nlp_->log->printf(
      hovSummary,
      "Ordering: %d\n",
      ordering_);

  nlp_->log->printf(
      hovSummary,
      "Factorization: klu\n");

  if(use_device_) {
    nlp_->log->printf(
        hovSummary,
        "Refactorization: rf\n");
  }

  nlp_->log->printf(
      hovSummary,
      "Use IR: no\n");
}

hiopLinSolverSparseEVLOSERExternal::
    ~hiopLinSolverSparseEVLOSERExternal()
{
#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA)
  delete rf_solver_;
  rf_solver_ = nullptr;
#elif defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  delete rf_solver_;
  rf_solver_ = nullptr;

  delete hip_workspace_;
  hip_workspace_ = nullptr;
#endif

  delete solver_;
  solver_ = nullptr;

  delete matrix_;
  matrix_ = nullptr;

  delete rhs_;
  rhs_ = nullptr;

  delete solution_;
  solution_ = nullptr;

  delete M_host_;
  M_host_ = nullptr;

  delete[] index_convert_CSR2Triplet_host_;
  index_convert_CSR2Triplet_host_ = nullptr;

  delete[] index_convert_extra_Diag2CSR_host_;
  index_convert_extra_Diag2CSR_host_ = nullptr;
}

int hiopLinSolverSparseEVLOSERExternal::matrixChanged(
    hiopMatrixSparse& matrix)
{
  M_ = &matrix;

  assert(n_ == M_->n());
  assert(M_->n() == M_->m());
  assert(n_ > 0);

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

  if(!factorization_valid_) {
    status = solver_->factorize();

    if(status != 0) {
      nlp_->log->printf(
          hovWarning,
          "External ReSolve KLU factorization failed. "
          "Regularizing ...\n");

      factorization_valid_ = false;
      reset_solver();

      nlp_->runStats.linsolv.tmFactTime.stop();
      return -1;
    }

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
    if(use_device_) {
      status = setup_gpu_refactorization();

      if(status == 0) {
        status = rf_solver_->refactorize();
      }

      if(status != 0) {
        nlp_->log->printf(
            hovWarning,
            "External ReSolve RF setup or initial "
            "refactorization failed. Regularizing ...\n");

        factorization_valid_ = false;
        reset_solver();

        nlp_->runStats.linsolv.tmFactTime.stop();
        return -1;
      }
    }
#endif

    factorization_valid_ = true;

    nlp_->log->printf(
        hovScalars,
        "External ReSolve KLU factorization successful.\n");
  } else {
#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
    if(use_device_) {
      status = rf_solver_->refactorize();
    } else
#endif
    {
      status = solver_->refactorize();
    }

    if(status != 0) {
      nlp_->log->printf(
          hovWarning,
          "External ReSolve refactorization failed. "
          "Regularizing ...\n");

      factorization_valid_ = false;
      reset_solver();

      nlp_->runStats.linsolv.tmFactTime.stop();
      return -1;
    }
  }

  nlp_->runStats.linsolv.tmFactTime.stop();
  return 0;
}

bool hiopLinSolverSparseEVLOSERExternal::solve(
    hiopVector& x)
{
  assert(M_ != nullptr);
  assert(n_ == M_->n());
  assert(M_->n() == M_->m());
  assert(n_ > 0);
  assert(x.get_size() == M_->n());

  if(!factorization_valid_) {
    nlp_->log->printf(
        hovError,
        "External ReSolve solve requested without a valid "
        "factorization.\n");
    return false;
  }

  nlp_->runStats.linsolv.tmTriuSolves.start();

  double* x_data = x.local_data();

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  if(use_device_) {
    if(rhs_->copyFromExternal(
           x_data,
           ReSolve::memory::DEVICE,
           ReSolve::memory::DEVICE) != 0) {
      nlp_->log->printf(
          hovError,
          "Failed to copy the device right-hand side into ReSolve.\n");

      nlp_->runStats.linsolv.tmTriuSolves.stop();
      return false;
    }

    if(rf_solver_->solve(rhs_, solution_) != 0) {
      nlp_->log->printf(
          hovError,
          "External ReSolve GPU RF solve failed.\n");

      nlp_->runStats.linsolv.tmTriuSolves.stop();
      return false;
    }

    double* solution_data =
        solution_->getData(ReSolve::memory::DEVICE);

    if(solution_data == nullptr) {
      nlp_->log->printf(
          hovError,
          "Failed to access the external ReSolve device solution.\n");

      nlp_->runStats.linsolv.tmTriuSolves.stop();
      return false;
    }

    if(!copy_device_to_device(
           nlp_,
           x_data,
           solution_data,
           sizeof(double) * n_,
           "copying the ReSolve solution into HiOp")) {
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
        "Failed to access the external ReSolve right-hand side.\n");

    nlp_->runStats.linsolv.tmTriuSolves.stop();
    return false;
  }

  std::copy(
      x_data,
      x_data + n_,
      rhs_data);

  rhs_->setDataUpdated(ReSolve::memory::HOST);

  if(solver_->solve(rhs_, solution_) != 0) {
    nlp_->log->printf(
        hovError,
        "External ReSolve KLU solve failed.\n");

    nlp_->runStats.linsolv.tmTriuSolves.stop();
    return false;
  }

  const double* solution_data =
      solution_->getData(ReSolve::memory::HOST);

  if(solution_data == nullptr) {
    nlp_->log->printf(
        hovError,
        "Failed to access the external ReSolve solution.\n");

    nlp_->runStats.linsolv.tmTriuSolves.stop();
    return false;
  }

  std::copy(
      solution_data,
      solution_data + n_,
      x_data);

  nlp_->runStats.linsolv.tmTriuSolves.stop();
  return true;
}

int hiopLinSolverSparseEVLOSERExternal::firstCall()
{
  assert(M_ != nullptr);

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  if(use_device_) {
    if(!copy_device_to_host(
           nlp_,
           M_host_->M(),
           M_->M(),
           sizeof(double) * M_->numberOfNonzeros(),
           "copying EVLOSER values to the host")) {
      return -1;
    }

    if(!copy_device_to_host(
           nlp_,
           M_host_->i_row(),
           M_->i_row(),
           sizeof(index_type) *
               M_->numberOfNonzeros(),
           "copying EVLOSER row indices to the host")) {
      return -1;
    }

    if(!copy_device_to_host(
           nlp_,
           M_host_->j_col(),
           M_->j_col(),
           sizeof(index_type) *
               M_->numberOfNonzeros(),
           "copying EVLOSER column indices to the host")) {
      return -1;
    }
  }
#endif

  compute_nnz();

  matrix_ = new ReSolve::matrix::Csr(
      n_,
      n_,
      nnz_,
      true,
      true);

  if(matrix_->allocateMatrixData(
         ReSolve::memory::HOST) != 0) {
    nlp_->log->printf(
        hovError,
        "Failed to allocate the external ReSolve CSR matrix.\n");
    return -1;
  }

  set_csr_indices_values();

  if(matrix_->setUpdated(
         ReSolve::memory::HOST) != 0) {
    nlp_->log->printf(
        hovError,
        "Failed to mark the external ReSolve matrix as updated.\n");
    return -1;
  }

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  if(use_device_ &&
     (((matrix_->getRowData(ReSolve::memory::DEVICE) == nullptr ||
        matrix_->getColData(ReSolve::memory::DEVICE) == nullptr ||
        matrix_->getValues(ReSolve::memory::DEVICE) == nullptr) &&
       matrix_->allocateMatrixData(ReSolve::memory::DEVICE) != 0) ||
      matrix_->syncData(ReSolve::memory::DEVICE) != 0)) {
    nlp_->log->printf(
        hovError,
        "Failed to copy the external ReSolve matrix to device memory.\n");
    return -1;
  }
#endif

  if(solver_->setup(matrix_) != 0) {
    nlp_->log->printf(
        hovError,
        "External ReSolve KLU setup failed.\n");
    return -1;
  }

  if(solver_->analyze() != 0) {
    nlp_->log->printf(
        hovError,
        "External ReSolve KLU symbolic analysis failed.\n");
    return -1;
  }

  is_first_call_ = false;
  return 0;
}

int hiopLinSolverSparseEVLOSERExternal::
    update_matrix_values()
{
  assert(M_ != nullptr);
  assert(matrix_ != nullptr);

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  if(use_device_) {
    if(!copy_device_to_host(
           nlp_,
           M_host_->M(),
           M_->M(),
           sizeof(double) * M_->numberOfNonzeros(),
           "updating EVLOSER host matrix values")) {
      return -1;
    }
  }
#endif

  hiopMatrixSparse* source = host_matrix();
  assert(source != nullptr);

  double* values =
      matrix_->getValues(ReSolve::memory::HOST);

  for(int k = 0; k < nnz_; ++k) {
    values[k] =
        source->M()[index_convert_CSR2Triplet_host_[k]];
  }

  for(int i = 0; i < n_; ++i) {
    if(index_convert_extra_Diag2CSR_host_[i] != -1) {
      values[index_convert_extra_Diag2CSR_host_[i]] +=
          source->M()[
              source->numberOfNonzeros() - n_ + i];
    }
  }

  if(matrix_->setUpdated(
         ReSolve::memory::HOST) != 0) {
    return -1;
  }

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  if(use_device_ &&
     (((matrix_->getRowData(ReSolve::memory::DEVICE) == nullptr ||
        matrix_->getColData(ReSolve::memory::DEVICE) == nullptr ||
        matrix_->getValues(ReSolve::memory::DEVICE) == nullptr) &&
       matrix_->allocateMatrixData(ReSolve::memory::DEVICE) != 0) ||
      matrix_->syncData(ReSolve::memory::DEVICE) != 0)) {
    return -1;
  }
#endif

  return 0;
}

hiopMatrixSparse*
hiopLinSolverSparseEVLOSERExternal::host_matrix() const
{
  return use_device_ ? M_host_ : M_;
}

void hiopLinSolverSparseEVLOSERExternal::compute_nnz()
{
  hiopMatrixSparse* source = host_matrix();
  assert(source != nullptr);

  nnz_ = n_;

  for(int k = 0;
      k < source->numberOfNonzeros() - n_;
      ++k) {
    if(source->i_row()[k] != source->j_col()[k]) {
      nnz_ += 2;
    }
  }
}

void hiopLinSolverSparseEVLOSERExternal::
    set_csr_indices_values()
{
  assert(M_ != nullptr);
  assert(matrix_ != nullptr);

  hiopMatrixSparse* source = host_matrix();
  assert(source != nullptr);

  ReSolve::index_type* row_ptr =
      matrix_->getRowData(ReSolve::memory::HOST);

  ReSolve::index_type* col_idx =
      matrix_->getColData(ReSolve::memory::HOST);

  double* values =
      matrix_->getValues(ReSolve::memory::HOST);

  std::fill(row_ptr, row_ptr + n_ + 1, 0);

  for(int k = 0;
      k < source->numberOfNonzeros() - n_;
      ++k) {
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

  index_convert_CSR2Triplet_host_ =
      new int[nnz_];

  index_convert_extra_Diag2CSR_host_ =
      new int[n_];

  int* nnz_each_row_tmp = new int[n_]{0};

  int total_nnz_tmp{0};
  int nnz_tmp{0};
  int rowID_tmp{0};
  int colID_tmp{0};

  for(int i = 0; i < n_; ++i) {
    index_convert_extra_Diag2CSR_host_[i] = -1;
  }

  for(int k = 0;
      k < source->numberOfNonzeros() - n_;
      ++k) {
    rowID_tmp = source->i_row()[k];
    colID_tmp = source->j_col()[k];

    if(rowID_tmp == colID_tmp) {
      nnz_tmp =
          nnz_each_row_tmp[rowID_tmp] +
          row_ptr[rowID_tmp];

      col_idx[nnz_tmp] = colID_tmp;
      values[nnz_tmp] = source->M()[k];

      index_convert_CSR2Triplet_host_[nnz_tmp] = k;

      values[nnz_tmp] +=
          source->M()[
              source->numberOfNonzeros() -
              n_ +
              rowID_tmp];

      index_convert_extra_Diag2CSR_host_[rowID_tmp] =
          nnz_tmp;

      nnz_each_row_tmp[rowID_tmp]++;
      total_nnz_tmp++;
    } else {
      nnz_tmp =
          nnz_each_row_tmp[rowID_tmp] +
          row_ptr[rowID_tmp];

      col_idx[nnz_tmp] = colID_tmp;
      values[nnz_tmp] = source->M()[k];

      index_convert_CSR2Triplet_host_[nnz_tmp] = k;

      nnz_tmp =
          nnz_each_row_tmp[colID_tmp] +
          row_ptr[colID_tmp];

      col_idx[nnz_tmp] = rowID_tmp;
      values[nnz_tmp] = source->M()[k];

      index_convert_CSR2Triplet_host_[nnz_tmp] = k;

      nnz_each_row_tmp[rowID_tmp]++;
      nnz_each_row_tmp[colID_tmp]++;
      total_nnz_tmp += 2;
    }
  }

  for(int i = 0; i < n_; ++i) {
    if(nnz_each_row_tmp[i] !=
       row_ptr[i + 1] - row_ptr[i]) {
      assert(
          nnz_each_row_tmp[i] ==
          row_ptr[i + 1] - row_ptr[i] - 1);

      nnz_tmp =
          nnz_each_row_tmp[i] +
          row_ptr[i];

      col_idx[nnz_tmp] = i;

      values[nnz_tmp] =
          source->M()[
              source->numberOfNonzeros() - n_ + i];

      index_convert_CSR2Triplet_host_[nnz_tmp] =
          source->numberOfNonzeros() - n_ + i;

      total_nnz_tmp++;

      std::vector<int> permutation(
          row_ptr[i + 1] - row_ptr[i]);

      std::iota(
          permutation.begin(),
          permutation.end(),
          0);

      std::sort(
          permutation.begin(),
          permutation.end(),
          [&](int a, int b) {
            return col_idx[a + row_ptr[i]] <
                   col_idx[b + row_ptr[i]];
          });

      reorder(
          values + row_ptr[i],
          permutation,
          row_ptr[i + 1] - row_ptr[i]);

      reorder(
          index_convert_CSR2Triplet_host_ +
              row_ptr[i],
          permutation,
          row_ptr[i + 1] - row_ptr[i]);

      std::sort(
          col_idx + row_ptr[i],
          col_idx + row_ptr[i + 1]);
    }
  }

  assert(total_nnz_tmp == nnz_);

  delete[] nnz_each_row_tmp;
}

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)

int hiopLinSolverSparseEVLOSERExternal::
    setup_gpu_refactorization()
{
  assert(rf_solver_ != nullptr);

  auto* L =
      dynamic_cast<ReSolve::matrix::Csr*>(
          solver_->getLFactor());

  auto* U =
      dynamic_cast<ReSolve::matrix::Csr*>(
          solver_->getUFactor());

  ReSolve::index_type* P =
      solver_->getPOrdering();

  ReSolve::index_type* Q =
      solver_->getQOrdering();

  if(L == nullptr ||
     U == nullptr ||
     P == nullptr ||
     Q == nullptr) {
    nlp_->log->printf(
        hovError,
        "Failed to extract KLU factors for external "
        "ReSolve RF.\n");
    return -1;
  }

  if(L->allocateMatrixData(ReSolve::memory::DEVICE) != 0 ||
     U->allocateMatrixData(ReSolve::memory::DEVICE) != 0) {
    nlp_->log->printf(
        hovError,
        "Failed to allocate device storage for external "
        "ReSolve KLU factors.\n");
    return -1;
  }

  return rf_solver_->setup(
      matrix_,
      L,
      U,
      P,
      Q,
      rhs_);
}

#endif

int hiopLinSolverSparseEVLOSERExternal::reset_solver()
{
#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA)
  if(use_device_) {
    delete rf_solver_;
    rf_solver_ =
        new ReSolve::LinSolverDirectCuSolverRf();
  }
#elif defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  if(use_device_) {
    delete rf_solver_;
    rf_solver_ =
        new ReSolve::LinSolverDirectRocSolverRf(
            hip_workspace_);
  }
#endif

  delete solver_;

  solver_ = new ReSolve::LinSolverDirectKLU();
  solver_->setOrdering(ordering_);
  solver_->setHaltIfSingular(true);

  if(solver_->setup(matrix_) != 0) {
    nlp_->log->printf(
        hovError,
        "External ReSolve KLU recovery setup failed.\n");
    return -1;
  }

  if(solver_->analyze() != 0) {
    nlp_->log->printf(
        hovError,
        "External ReSolve KLU recovery analysis failed.\n");
    return -1;
  }

  return 0;
}

}  // namespace

hiopLinSolverSparseEVLOSERProvider*
create_hiop_evloser_provider(const int& n,
                             const int& nnz,
                             hiopNlpFormulation* nlp)
{
  return new hiopLinSolverSparseEVLOSERExternal(
      n,
      nnz,
      nlp);
}

}  // namespace hiop