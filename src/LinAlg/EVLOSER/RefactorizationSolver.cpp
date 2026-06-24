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
 * @file RefactorizationSolver.cpp
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 * @author Slaven Peles <peless@ornl.gov>, ORNL
 *
 */

#include "MatrixCsr.hpp"
#include "RefactorizationSolver.hpp"

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
#include "IterativeRefinement.hpp"
#endif

#include "klu.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>


#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
#define checkGpuErrors(val) evloserCheckGpuError((val), __FILE__, __LINE__)
#endif

namespace EVLOSER
{
#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
namespace
{

struct KluFactorData
{
  int nnzL{0};
  int nnzU{0};
  std::vector<int> Lp;
  std::vector<int> Li;
  std::vector<double> Lx;
  std::vector<int> Up;
  std::vector<int> Ui;
  std::vector<double> Ux;
};

struct HostCsrFactor
{
  std::vector<int> rowptr;
  std::vector<int> colind;
  std::vector<double> values;
};

bool validate_csc_factor(const char* name, int n, int nnz, const std::vector<int>& colptr, const std::vector<int>& rowind, bool silent_output)
{
  auto report = [&](const std::string& message) {
    if(!silent_output) {
      std::cout << "[EVLOSER] Invalid KLU " << name << " factor: " << message << "\n";
    }
    return false;
  };

  if(n <= 0) {
    return report("factor dimension must be positive");
  }

  if(nnz < 0) {
    return report("number of nonzeros is negative");
  }

  if(static_cast<int>(colptr.size()) != n + 1) {
    return report("column pointer size does not match dimension");
  }

  if(static_cast<int>(rowind.size()) != nnz) {
    return report("row index size does not match nnz");
  }

  if(colptr[0] != 0) {
    return report("column pointer must start at zero");
  }

  for(int col = 0; col < n; ++col) {
    if(colptr[col] > colptr[col + 1]) {
      return report("column pointer is not monotone");
    }
  }

  if(colptr[n] != nnz) {
    return report("final column pointer does not match nnz");
  }

  for(int k = 0; k < nnz; ++k) {
    if(rowind[k] < 0 || rowind[k] >= n) {
      return report("row index out of range");
    }
  }

  return true;
}

bool extract_klu_factors(klu_numeric* numeric,
                         klu_symbolic* symbolic,
                         klu_common& common,
                         int n,
                         KluFactorData& factors,
                         bool silent_output)
{
  factors.nnzL = numeric->lnz;
  factors.nnzU = numeric->unz;

  factors.Lp.assign(n + 1, 0);
  factors.Li.assign(factors.nnzL, 0);
  factors.Lx.assign(factors.nnzL, 0.0);
  factors.Up.assign(n + 1, 0);
  factors.Ui.assign(factors.nnzU, 0);
  factors.Ux.assign(factors.nnzU, 0.0);

  const int ok = klu_extract(numeric,
                             symbolic,
                             factors.Lp.data(),
                             factors.Li.data(),
                             factors.Lx.data(),
                             factors.Up.data(),
                             factors.Ui.data(),
                             factors.Ux.data(),
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             &common);

  if(ok == 0) {
    if(!silent_output) {
      std::cout << "[EVLOSER] klu_extract failed while preparing GPU RF setup\n";
    }
    return false;
  }

  return validate_csc_factor("L", n, factors.nnzL, factors.Lp, factors.Li, silent_output) &&
         validate_csc_factor("U", n, factors.nnzU, factors.Up, factors.Ui, silent_output);
}

HostCsrFactor convert_csc_to_csr(int n,
                                 int nnz,
                                 const std::vector<int>& colptr,
                                 const std::vector<int>& rowind,
                                 const std::vector<double>& values)
{
  HostCsrFactor csr;
  csr.rowptr.assign(n + 1, 0);
  csr.colind.assign(nnz, 0);
  csr.values.assign(nnz, 0.0);

  for(int col = 0; col < n; ++col) {
    for(int k = colptr[col]; k < colptr[col + 1]; ++k) {
      csr.rowptr[rowind[k] + 1]++;
    }
  }

  for(int row = 0; row < n; ++row) {
    csr.rowptr[row + 1] += csr.rowptr[row];
  }

  std::vector<int> offsets = csr.rowptr;
  for(int col = 0; col < n; ++col) {
    for(int k = colptr[col]; k < colptr[col + 1]; ++k) {
      const int row = rowind[k];
      const int dest = offsets[row]++;
      csr.colind[dest] = col;
      csr.values[dest] = values[k];
    }
  }

  return csr;
}

bool validate_host_csr_factor(const char* name, int n, int nnz, const HostCsrFactor& csr, bool silent_output)
{
  auto report = [&](const std::string& message) {
    if(!silent_output) {
      std::cout << "[EVLOSER] Invalid host CSR " << name << " factor: " << message << "\n";
    }
    return false;
  };

  if(static_cast<int>(csr.rowptr.size()) != n + 1) {
    return report("row pointer size does not match dimension");
  }

  if(static_cast<int>(csr.colind.size()) != nnz || static_cast<int>(csr.values.size()) != nnz) {
    return report("column/value array size does not match nnz");
  }

  if(csr.rowptr[0] != 0) {
    return report("row pointer must start at zero");
  }

  for(int row = 0; row < n; ++row) {
    if(csr.rowptr[row] > csr.rowptr[row + 1]) {
      return report("row pointer is not monotone");
    }
  }

  if(csr.rowptr[n] != nnz) {
    return report("final row pointer does not match nnz");
  }

  for(int k = 0; k < nnz; ++k) {
    if(csr.colind[k] < 0 || csr.colind[k] >= n) {
      return report("column index out of range");
    }
  }

  return true;
}

}  // namespace
#endif
RefactorizationSolver::RefactorizationSolver(int n, ExecutionMode execution_mode)
    : n_(n),
      execution_mode_(execution_mode)
{
  mat_A_csr_ = new MatrixCsr(execution_mode_);
#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  if(execution_mode_ == ExecutionMode::CUDA || execution_mode_ == ExecutionMode::HIP) {
    // handles
    cusparseCreate(&handle_);
    cusolverSpCreate(&handle_cusolver_);
    cublasCreate(&handle_cublas_);

    // descriptors
    cusparseCreateMatDescr(&descr_A_);
    cusparseSetMatType(descr_A_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_A_, CUSPARSE_INDEX_BASE_ZERO);
  }
#endif

  // Allocate host mirror for the solution vector
  hostx_ = new double[n_];
#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  if(execution_mode_ == ExecutionMode::CUDA || execution_mode_ == ExecutionMode::HIP) {
    // Allocate solution and rhs vectors
    checkGpuErrors(evloserGpuMalloc((void**)&devx_, n_ * sizeof(double)));
    checkGpuErrors(evloserGpuMalloc((void**)&devr_, n_ * sizeof(double)));
  }
#endif
}

RefactorizationSolver::~RefactorizationSolver()
{
#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  delete ir_;
#endif
  delete mat_A_csr_;

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  if(execution_mode_ == ExecutionMode::CUDA || execution_mode_ == ExecutionMode::HIP) {
    // Delete workspaces and handles
    if(d_work_ != nullptr) {
      (void)evloserGpuFree(d_work_);
    }
    cusparseDestroy(handle_);
    cusolverSpDestroy(handle_cusolver_);
    cublasDestroy(handle_cublas_);
    cusparseDestroyMatDescr(descr_A_);
  }
#endif

  // Delete host mirror for the solution vector
  delete[] hostx_;

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  if(execution_mode_ == ExecutionMode::CUDA || execution_mode_ == ExecutionMode::HIP) {
    // Delete residual and solution vectors
    if(devr_ != nullptr) {
      (void)evloserGpuFree(devr_);
    }
    if(devx_ != nullptr) {
      (void)evloserGpuFree(devx_);
    }

    // Delete matrix descriptor used in cuSolverGLU setup
    if(cusolver_glu_enabled_) {
      cusparseDestroyMatDescr(descr_M_);
      cusolverSpDestroyGluInfo(info_M_);
    }

    if(cusolver_rf_enabled_) {
      if(d_P_ != nullptr) {
        (void)evloserGpuFree(d_P_);
      }
      if(d_Q_ != nullptr) {
        (void)evloserGpuFree(d_Q_);
      }
      if(d_T_ != nullptr) {
        (void)evloserGpuFree(d_T_);
      }
    }
  }
#endif
  if(Numeric_ != nullptr) {
    klu_free_numeric(&Numeric_, &Common_);
  }

  if(Symbolic_ != nullptr) {
    klu_free_symbolic(&Symbolic_, &Common_);
  }
  delete[] mia_;
  delete[] mja_;
}

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
void RefactorizationSolver::enable_iterative_refinement()
{
  if(execution_mode_ == ExecutionMode::CPU) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] Iterative refinement is unavailable in CPU execution mode.\n";
    }
    return;
  }

  if(ir_ == nullptr) {
    ir_ = new IterativeRefinement();
  }

  iterative_refinement_enabled_ = (ir_ != nullptr);
}

void RefactorizationSolver::disable_iterative_refinement()
{
  delete ir_;
  ir_ = nullptr;
  iterative_refinement_enabled_ = false;
  use_ir_ = "no";
}

bool RefactorizationSolver::iterative_refinement_active() const
{
  return execution_mode_ != ExecutionMode::CPU && iterative_refinement_enabled_ && ir_ != nullptr && use_ir_ == "yes";
}

// TODO: Refactor to only pass mat_A_csr_ to setup_system_matrix; n and nnz can be read from mat_A_csr_
void RefactorizationSolver::setup_iterative_refinement_matrix(int n, int nnz)
{
  if(!iterative_refinement_active()) {
    return;
  }

  ir_->setup_system_matrix(n, nnz, mat_A_csr_->device_irows(), mat_A_csr_->device_jcols(), mat_A_csr_->device_vals());
}

// TODO: Can this function be merged with setup_iterative_refinement_matrix ?
void RefactorizationSolver::configure_iterative_refinement(cusparseHandle_t cusparse_handle,
                                                           cublasHandle_t cublas_handle,
                                                           evloserRfHandle_t cusolverrf_handle,
                                                           int n,
                                                           double* d_T,
                                                           int* d_P,
                                                           int* d_Q,
                                                           double* devx,
                                                           double* devr)
{
  if(!iterative_refinement_active()) {
    return;
  }

  ir_->setup(cusparse_handle, cublas_handle, cusolverrf_handle, n, d_T, d_P, d_Q, devx, devr);
}
#endif

bool RefactorizationSolver::validate_system_matrix(const char* caller) const
{
  if(mat_A_csr_ == nullptr) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] Invalid matrix in " << caller << ": matrix object is null\n";
    }
    return false;
  }

  return mat_A_csr_->validate_host_structure(caller, silent_output_) &&
         mat_A_csr_->validate_host_values(caller, silent_output_);
}

bool RefactorizationSolver::validate_klu_factorization(const char* caller) const
{
  if(Symbolic_ == nullptr) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] Invalid KLU factorization in " << caller << ": symbolic factor is null\n";
    }
    return false;
  }

  if(Numeric_ == nullptr) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] Invalid KLU factorization in " << caller << ": numeric factor is null\n";
    }
    return false;
  }

  if(Symbolic_->n != n_) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] Invalid KLU factorization in " << caller << ": symbolic dimension "
                << Symbolic_->n << " does not match solver dimension " << n_ << "\n";
    }
    return false;
  }

  if(Numeric_->n != n_) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] Invalid KLU factorization in " << caller << ": numeric dimension "
                << Numeric_->n << " does not match solver dimension " << n_ << "\n";
    }
    return false;
  }

  if(Symbolic_->Q == nullptr || Numeric_->Pnum == nullptr) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] Invalid KLU factorization in " << caller
                << ": missing permutation data\n";
    }
    return false;
  }

  return true;
}

bool RefactorizationSolver::validate_solution(const double* solution, const char* caller) const
{
  const char* caller_name = caller == nullptr ? "unknown caller" : caller;

  if(solution == nullptr) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] Invalid vector in " << caller_name << ": pointer is null\n";
    }
    return false;
  }

  for(int i = 0; i < n_; ++i) {
    if(!std::isfinite(solution[i])) {
      if(!silent_output_) {
        std::cout << "[EVLOSER] Invalid vector in " << caller_name
                  << ": entry " << i << " is not finite\n";
      }
      return false;
    }
  }

  return true;
}

double RefactorizationSolver::compute_klu_residual(const double* rhs,
                                                   const double* solution) const
{
  if(rhs == nullptr || solution == nullptr || mat_A_csr_ == nullptr) {
    return std::numeric_limits<double>::infinity();
  }

  const int* rowptr = mat_A_csr_->host_irows();
  const int* colind = mat_A_csr_->host_jcols();
  const double* values = mat_A_csr_->host_vals();

  long double residual_inf = 0.0L;
  long double matrix_inf = 0.0L;
  long double solution_inf = 0.0L;
  long double rhs_inf = 0.0L;

  for(int i = 0; i < n_; ++i) {
    solution_inf =
        std::max(solution_inf, std::abs(static_cast<long double>(solution[i])));
    rhs_inf =
        std::max(rhs_inf, std::abs(static_cast<long double>(rhs[i])));
  }

  for(int row = 0; row < n_; ++row) {
    long double row_sum = 0.0L;
    long double matrix_vector_product = 0.0L;

    for(int k = rowptr[row]; k < rowptr[row + 1]; ++k) {
      const long double value = static_cast<long double>(values[k]);

      row_sum += std::abs(value);
      matrix_vector_product +=
          value * static_cast<long double>(solution[colind[k]]);
    }

    matrix_inf = std::max(matrix_inf, row_sum);
    residual_inf =
        std::max(residual_inf,
                 std::abs(matrix_vector_product -
                          static_cast<long double>(rhs[row])));
  }

  const long double scale =
      std::max(1.0L, matrix_inf * solution_inf + rhs_inf);

  const long double normalized_residual = residual_inf / scale;

  if(!std::isfinite(normalized_residual)) {
    return std::numeric_limits<double>::infinity();
  }

  return static_cast<double>(normalized_residual);
}

klu_numeric* RefactorizationSolver::factor_klu_numeric(const char* caller)
{
  if(!validate_system_matrix(caller)) {
    return nullptr;
  }

  if(Symbolic_ == nullptr || Symbolic_->n != n_) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] " << caller
                << " requires valid KLU symbolic analysis.\n";
    }
    return nullptr;
  }

  klu_common trial_common = Common_;
  trial_common.status = KLU_OK;

  klu_numeric* numeric =
      klu_factor(mat_A_csr_->host_irows(),
                 mat_A_csr_->host_jcols(),
                 mat_A_csr_->host_vals(),
                 Symbolic_,
                 &trial_common);

  const int status = trial_common.status;

  if(numeric == nullptr || status != KLU_OK) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] " << caller
                << " failed with KLU status " << status << "\n";
    }

    if(numeric != nullptr) {
      klu_free_numeric(&numeric, &trial_common);
    }

    return nullptr;
  }
  return numeric;
}

bool RefactorizationSolver::solve_klu_candidate(
    klu_numeric* numeric,
    const double* rhs,
    std::vector<double>& solution,
    double& residual,
    const char* caller)
{
  residual = std::numeric_limits<double>::infinity();

  if(numeric == nullptr || rhs == nullptr) {
    return false;
  }

  if(!validate_solution(rhs, caller)) {
    return false;
  }

  solution.assign(rhs, rhs + n_);

  klu_common trial_common = Common_;
  trial_common.status = KLU_OK;

  const int ok =
      klu_solve(Symbolic_,
                numeric,
                n_,
                1,
                solution.data(),
                &Common_);

  const int status = Common_.status;

  if(ok == 0 || status != KLU_OK) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] " << caller
                << " failed with KLU status " << status << "\n";
    }
    return false;
  }

  if(!validate_solution(solution.data(), caller)) {
    return false;
  }

  residual = compute_klu_residual(rhs, solution.data());

  if(!std::isfinite(residual)) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] " << caller
                << " produced a non-finite residual.\n";
    }
    return false;
  }

  return true;
}

bool RefactorizationSolver::solve_cpu_with_recovery(double* dx)
{
  last_klu_recovery_action_ = KluRecoveryAction::None;

  if(dx == nullptr) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] KLU solve received a null right-hand side.\n";
    }
    last_klu_recovery_action_ = KluRecoveryAction::Failed;
    return false;
  }

  if(!validate_system_matrix("KLU solve") ||
     !validate_solution(dx, "KLU right-hand side")) {
    last_klu_recovery_action_ = KluRecoveryAction::Failed;
    return false;
  }

  if(Symbolic_ == nullptr || Symbolic_->n != n_) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] KLU solve requires valid symbolic analysis.\n";
    }

    last_klu_recovery_action_ = KluRecoveryAction::Failed;
    return false;
  }

  /*
   * A failed klu_refactor() may leave Common_.status nonzero. When recovery
   * is pending, allow the fresh-factorization path to run instead of
   * rejecting the solve based on that previous status.
   */
  if(!klu_refactor_pending_validation_ &&
     !validate_klu_factorization("KLU solve")) {
    last_klu_recovery_action_ = KluRecoveryAction::Failed;
    return false;
  }

  const std::vector<double> rhs(dx, dx + n_);

  std::vector<double> refactor_solution;
  double residual_refactor =
      std::numeric_limits<double>::infinity();

  bool refactor_solution_usable = false;

  /*
   * With no pending value-only refactorization, Numeric_ already represents
   * a fresh factorization. Solve normally and apply only the loose safety
   * limit.
   */
  if(!klu_refactor_pending_validation_) {
    if(!solve_klu_candidate(Numeric_,
                            rhs.data(),
                            refactor_solution,
                            residual_refactor,
                            "KLU solve")) {
      last_klu_recovery_action_ = KluRecoveryAction::Failed;
      return false;
    }

    if(residual_refactor > klu_residual_safety_limit_) {
      if(!silent_output_) {
        std::cout << "[EVLOSER] KLU residual "
                  << residual_refactor
                  << " exceeds safety limit "
                  << klu_residual_safety_limit_ << "\n";
      }

      last_klu_recovery_action_ = KluRecoveryAction::Failed;
      return false;
    }

    std::copy(refactor_solution.begin(),
              refactor_solution.end(),
              dx);

    return true;
  }

  if(klu_refactor_succeeded_) {
    refactor_solution_usable =
        solve_klu_candidate(Numeric_,
                            rhs.data(),
                            refactor_solution,
                            residual_refactor,
                            "KLU refactorized solve");
  }

  const bool fresh_factorization_required =
      !refactor_solution_usable ||
      residual_refactor > klu_suspicious_residual_threshold_;

  if(!fresh_factorization_required) {
    if(residual_refactor > klu_residual_safety_limit_) {
      if(!silent_output_) {
        std::cout << "[EVLOSER] KLU refactorized residual "
                  << residual_refactor
                  << " exceeds safety limit "
                  << klu_residual_safety_limit_ << "\n";
      }

      klu_refactor_pending_validation_ = false;
      klu_refactor_succeeded_ = false;
      last_klu_recovery_action_ = KluRecoveryAction::Failed;
      return false;
    }

    std::copy(refactor_solution.begin(),
              refactor_solution.end(),
              dx);

    klu_refactor_pending_validation_ = false;
    klu_refactor_succeeded_ = false;
    last_klu_recovery_action_ =
        KluRecoveryAction::RefactorAccepted;

    return true;
  }

  if(!silent_output_ && refactor_solution_usable) {
    std::cout << "[EVLOSER] KLU refactorized residual "
              << residual_refactor
              << " exceeds suspicious-result threshold "
              << klu_suspicious_residual_threshold_
              << "; trying fresh factorization.\n";
  }

  klu_numeric* full_numeric =
      factor_klu_numeric("KLU recovery factorization");

  std::vector<double> full_solution;
  double residual_full =
      std::numeric_limits<double>::infinity();

  const bool full_solution_usable =
      full_numeric != nullptr &&
      solve_klu_candidate(full_numeric,
                          rhs.data(),
                          full_solution,
                          residual_full,
                          "KLU recovery solve");

  const bool refactor_candidate_safe =
      refactor_solution_usable &&
      residual_refactor <= klu_residual_safety_limit_;

  const bool full_candidate_safe =
      full_solution_usable &&
      residual_full <= klu_residual_safety_limit_;

  if(!refactor_candidate_safe && !full_candidate_safe) {
    if(full_numeric != nullptr) {
      klu_free_numeric(&full_numeric, &Common_);
    }

    if(!silent_output_) {
      std::cout << "[EVLOSER] KLU recovery produced no candidate "
                   "within the residual safety limit.\n";
    }

    klu_refactor_pending_validation_ = false;
    klu_refactor_succeeded_ = false;
    last_klu_recovery_action_ = KluRecoveryAction::Failed;

    return false;
  }

  const bool full_factor_materially_better =
      full_candidate_safe &&
      refactor_candidate_safe &&
      residual_full <
          klu_improvement_ratio_ * residual_refactor &&
      residual_refactor - residual_full >
          klu_minimum_improvement_;

  const bool keep_full_factors =
      full_candidate_safe &&
      (!refactor_candidate_safe ||
       full_factor_materially_better);

  if(keep_full_factors) {
    if(Numeric_ != nullptr) {
      klu_free_numeric(&Numeric_, &Common_);
    }

    Numeric_ = full_numeric;
    full_numeric = nullptr;

    std::copy(full_solution.begin(),
              full_solution.end(),
              dx);

    last_klu_recovery_action_ =
        KluRecoveryAction::FullFactorAccepted;
  } else {
    /*
     * Retain the refactored factors unless the fresh factorization is
     * materially better. For the current solve, return whichever finite
     * candidate has the smaller residual.
     */
    if(full_candidate_safe &&
       residual_full < residual_refactor) {
      std::copy(full_solution.begin(),
                full_solution.end(),
                dx);
    } else {
      std::copy(refactor_solution.begin(),
                refactor_solution.end(),
                dx);
    }

    if(full_numeric != nullptr) {
      klu_free_numeric(&full_numeric, &Common_);
    }

    last_klu_recovery_action_ =
        KluRecoveryAction::RefactorRetained;
  }

  klu_refactor_pending_validation_ = false;
  klu_refactor_succeeded_ = false;

  return true;
}

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
bool RefactorizationSolver::checkEvloserRfStatus(evloserRfStatus_t status, const char* caller) const
{
  if(status == evloserRfSuccess) {
    return true;
  }

  if(!silent_output_) {
    std::cout << "[EVLOSER] " << caller << " failed with GPU RF status " << status << "\n";
  }

  return false;
}

int RefactorizationSolver::resetEvloserRfValues(const char* caller)
{
  sp_status_ = evloserRfResetValues(n_,
                                     nnz_,
                                     mat_A_csr_->device_irows(),
                                     mat_A_csr_->device_jcols(),
                                     mat_A_csr_->device_vals(),
                                     d_P_,
                                     d_Q_,
                                     handle_rf_);

  if(!checkEvloserRfStatus(sp_status_, caller)) {
    return -1;
  }

  checkGpuErrors(evloserGpuDeviceSynchronize());
  return 0;
}

int RefactorizationSolver::analyzeEvloserRf(const char* caller)
{
  sp_status_ = evloserRfAnalyze(handle_rf_);
  return checkEvloserRfStatus(sp_status_, caller) ? 0 : -1;
}

int RefactorizationSolver::refactorizeEvloserRf(const char* caller)
{
  sp_status_ = evloserRfRefactor(handle_rf_);
  return checkEvloserRfStatus(sp_status_, caller) ? 0 : -1;
}
#endif

int RefactorizationSolver::setup_factorization()
{
  if(fact_ != "klu") {
    assert(false && "Only KLU is available for the first factorization.");
    return -1;
  }

  if(!validate_system_matrix("KLU analysis")) {
    return -1;
  }

  klu_refactor_pending_validation_ = false;
  klu_refactor_succeeded_ = false;
  last_klu_recovery_action_ = KluRecoveryAction::None;

  // A new matrix structure invalidates both existing KLU states.
  if(Numeric_ != nullptr) {
    klu_free_numeric(&Numeric_, &Common_);
  }

  if(Symbolic_ != nullptr) {
    klu_free_symbolic(&Symbolic_, &Common_);
  }

  if(initializeKLU() != 0) {
    return -1;
  }

  Symbolic_ = klu_analyze(n_,
                          mat_A_csr_->host_irows(),
                          mat_A_csr_->host_jcols(),
                          &Common_);

  if(Symbolic_ == nullptr || Common_.status != KLU_OK) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] KLU symbolic analysis failed with status "
                << Common_.status << "\n";
    }
    return -1;
  }

  return 0;
}

int RefactorizationSolver::factorize()
{
  klu_numeric* fresh_numeric =
      factor_klu_numeric("KLU factorization");

  if(fresh_numeric == nullptr) {
    return -1;
  }

  if(Numeric_ != nullptr) {
    klu_free_numeric(&Numeric_, &Common_);
  }

  Numeric_ = fresh_numeric;

  klu_refactor_pending_validation_ = false;
  klu_refactor_succeeded_ = false;
  last_klu_recovery_action_ = KluRecoveryAction::None;
  is_first_solve_ = true;

  return validate_klu_factorization("KLU factorization")
             ? 0
             : -1;
}

void RefactorizationSolver::setup_refactorization()
{
  if(!validate_system_matrix("refactorization setup")) {
    return;
  }

  if(execution_mode_ == ExecutionMode::CPU) {
    // KLU numeric state is already available from factorize().
    return;
  }

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  if(refact_ == "glu") {
    if(execution_mode_ != ExecutionMode::CUDA) {
      if(!silent_output_) {
        std::cout << "[EVLOSER] GLU refactorization requires CUDA execution mode.\n";
      }
      return;
    }

    if(initializeCusolverGLU() != 0) {
      return;
    }

    if(refactorizationSetupCusolverGLU() != 0) {
      return;
    }
  } else if(refact_ == "rf") {
    if(initializeCusolverRf() != 0 || refactorizationSetupCusolverRf() != 0) {
      assert(false && "EVLOSER RF setup failed.");
      return;
    }
    if(iterative_refinement_active()) {
      configure_iterative_refinement(handle_, handle_cublas_, handle_rf_, n_, d_T_, d_P_, d_Q_, devx_, devr_);
    }
  } else {  // for future -
    assert(0 && "Only glu and rf refactorizations available.\n");
  }

#endif
}

int RefactorizationSolver::refactorize()
{
  if(!validate_system_matrix("refactorization")) {
    if(execution_mode_ == ExecutionMode::CPU) {
      klu_refactor_pending_validation_ = false;
      klu_refactor_succeeded_ = false;
      last_klu_recovery_action_ = KluRecoveryAction::Failed;
    }
    return -1;
  }

  if(execution_mode_ == ExecutionMode::CPU) {
    if(!validate_klu_factorization("KLU refactorization")) {
      klu_refactor_pending_validation_ = false;
      klu_refactor_succeeded_ = false;
      last_klu_recovery_action_ = KluRecoveryAction::Failed;
      return -1;
    }

    klu_refactor_pending_validation_ = true;
    klu_refactor_succeeded_ = false;
    last_klu_recovery_action_ = KluRecoveryAction::None;

    const int ok =
        klu_refactor(mat_A_csr_->host_irows(),
                     mat_A_csr_->host_jcols(),
                     mat_A_csr_->host_vals(),
                     Symbolic_,
                     Numeric_,
                     &Common_);

    klu_refactor_succeeded_ =
        ok != 0 && Common_.status == KLU_OK;

    if(!klu_refactor_succeeded_ && !silent_output_) {
      std::cout
          << "[EVLOSER] KLU refactorization failed with status "
          << Common_.status
          << "; fresh factorization will be attempted during solve.\n";
    }

    /*
     * A failed KLU refactorization is recoverable. Continue to the solve,
     * where a fresh numeric factorization will be attempted.
     */
    return 0;
  }

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  if(refact_ == "glu") {
    if(execution_mode_ != ExecutionMode::CUDA) {
      if(!silent_output_) {
        std::cout
            << "[EVLOSER] GLU refactorization requires CUDA execution mode.\n";
      }
      return -1;
    }

    sp_status_ = cusolverSpDgluReset(handle_cusolver_,
                                     n_,
                                     /* A is original matrix */
                                     nnz_,
                                     descr_A_,
                                     mat_A_csr_->device_vals(),
                                     mat_A_csr_->device_irows(),
                                     mat_A_csr_->device_jcols(),
                                     info_M_);

    sp_status_ =
        cusolverSpDgluFactor(handle_cusolver_, info_M_, d_work_);
  } else {
    if(refact_ == "rf") {
      if(resetEvloserRfValues("GPU RF reset values") != 0) {
        return -1;
      }

      if(refactorizeEvloserRf("GPU RF refactorization") != 0) {
        return -1;
      }
    }
  }

  return 0;
#endif

  if(!silent_output_) {
    std::cout
        << "[EVLOSER] Selected refactorization backend is unavailable "
           "in this build.\n";
  }

  return -1;
}

bool RefactorizationSolver::triangular_solve(double* dx, double tol)
{
  if(execution_mode_ == ExecutionMode::CPU) {
    (void)tol;
    return solve_cpu_with_recovery(dx);
  }

  if(dx == nullptr) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] Solve received a null right-hand side.\n";
    }
    return false;
  }

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  if(refact_ == "glu") {
    if(execution_mode_ != ExecutionMode::CUDA) {
      if(!silent_output_) {
        std::cout << "[EVLOSER] GLU solve requires CUDA execution mode.\n";
      }
      return false;
    }

    checkGpuErrors(evloserGpuMemcpy(devr_, dx, sizeof(double) * n_, evloserMemcpyDeviceToDevice));
    sp_status_ = cusolverSpDgluSolve(handle_cusolver_,
                                     n_,
                                     /* A is original matrix */
                                     nnz_,
                                     descr_A_,
                                     mat_A_csr_->device_vals(),
                                     mat_A_csr_->device_irows(),
                                     mat_A_csr_->device_jcols(),
                                     devr_, /* right hand side */
                                     dx,    /* left hand side */
                                     &ite_refine_succ_,
                                     &r_nrminf_,
                                     info_M_,
                                     d_work_);
    if(sp_status_ != 0) {
      if(!silent_output_) {
        std::cout << "GLU solve failed with status: " << sp_status_ << "\n";
      }
      return false;
    }
    return true;
  }

  if(refact_ == "rf") {
    // First solve is performed on CPU
    if(is_first_solve_) {
      checkGpuErrors(evloserGpuMemcpy(hostx_,
                                    dx,
                                    sizeof(double) * n_,
                                    evloserMemcpyDeviceToHost));

      const int ok = klu_solve(Symbolic_,
                              Numeric_,
                              n_,
                              1,
                              hostx_,
                              &Common_);

      if(ok == 0 || Common_.status != KLU_OK) {
        if(!silent_output_) {
          std::cout << "[EVLOSER] Initial KLU solve failed with status "
                    << Common_.status << "\n";
        }
        return false;
      }

      if(!validate_solution(hostx_, "initial KLU solve")) {
        return false;
      }

      checkGpuErrors(evloserGpuMemcpy(dx,
                                    hostx_,
                                    sizeof(double) * n_,
                                    evloserMemcpyHostToDevice));

      is_first_solve_ = false;
      return true;
    }

    checkGpuErrors(evloserGpuMemcpy(devr_, dx, sizeof(double) * n_, evloserMemcpyDeviceToDevice));

    // Each next solve is performed on GPU
    sp_status_ = evloserRfSolve(handle_rf_,
                                 d_P_,
                                 d_Q_,
                                 1,
                                 d_T_,
                                 n_,
                                 dx,
                                 n_);
    if(sp_status_ != 0) {
      if(!silent_output_) std::cout << "Rf solve failed with status: " << sp_status_ << "\n";
      return false;
    }

    if(iterative_refinement_active()) {
      // Set tolerance based on barrier parameter mu
      ir_->set_tol(tol);

      ir_->fgmres(dx, devr_);
      if(!silent_output_ && (ir_->getFinalResidalNorm() > tol * ir_->getBNorm())) {
        std::cout << "[Warning] Iterative refinement did not converge!\n";
        std::cout << "\t Iterative refinement tolerance " << tol << "\n";
        std::cout << "\t Relative solution error        " << ir_->getFinalResidalNorm() / ir_->getBNorm() << "\n";
        std::cout << "\t fgmres: init residual norm: " << ir_->getInitialResidalNorm() << "\n"
                  << "\t final residual norm:        " << ir_->getFinalResidalNorm() << "\n"
                  << "\t number of iterations:       " << ir_->getFinalNumberOfIterations() << "\n";
      }
    }
    return true;
  }

  if(!silent_output_) {
    std::cout << "Unknown refactorization " << refact_ << ", exiting\n";
  }
  return false;
#else

  (void)dx;
  (void)tol;

  if(!silent_output_) {
    std::cout << "[EVLOSER] GPU triangular solve is unavailable in this build.\n";
  }

  return false;

#endif
}


#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
// helper private function needed for format conversion
int RefactorizationSolver::createM(const int n,
                                   const int /* nnzL */,
                                   const int* Lp,
                                   const int* Li,
                                   const int /* nnzU */,
                                   const int* Up,
                                   const int* Ui)
{
  int row;
  for(int i = 0; i < n; ++i) {
    // go through EACH COLUMN OF L first
    for(int j = Lp[i]; j < Lp[i + 1]; ++j) {
      row = Li[j];
      // BUT dont count diagonal twice, important
      if(row != i) {
        mia_[row + 1]++;
      }
    }
    // then each column of U
    for(int j = Up[i]; j < Up[i + 1]; ++j) {
      row = Ui[j];
      mia_[row + 1]++;
    }
  }
  // then organize mia_;
  mia_[0] = 0;
  for(int i = 1; i < n + 1; i++) {
    mia_[i] += mia_[i - 1];
  }

  std::vector<int> Mshifts(n, 0);
  for(int i = 0; i < n; ++i) {
    // go through EACH COLUMN OF L first
    for(int j = Lp[i]; j < Lp[i + 1]; ++j) {
      row = Li[j];
      if(row != i) {
        // place (row, i) where it belongs!
        mja_[mia_[row] + Mshifts[row]] = i;
        Mshifts[row]++;
      }
    }
    // each column of U next
    for(int j = Up[i]; j < Up[i + 1]; ++j) {
      row = Ui[j];
      mja_[mia_[row] + Mshifts[row]] = i;
      Mshifts[row]++;
    }
  }
  return 0;
}
#endif

int RefactorizationSolver::initializeKLU()
{
  if(klu_defaults(&Common_) == 0) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] klu_defaults failed.\n";
    }
    return -1;
  }

  // TODO: consider making these user-configurable.
  Common_.btf = 0;
  Common_.ordering = ordering_;  // COLAMD=1; AMD=0
  Common_.tol = 0.1;
  Common_.scale = -1;
  Common_.halt_if_singular = 1;

  if(Common_.status != KLU_OK) {
    if(!silent_output_) {
      std::cout << "[EVLOSER] Invalid KLU initialization status "
                << Common_.status << "\n";
    }
    return -1;
  }

  return 0;
}

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
int RefactorizationSolver::initializeCusolverGLU()
{
#if defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  std::cerr << "EVLOSER GLU refactorization is not supported on HIP. Use RF instead.\n";
  return -1;
#endif

  cusparseCreateMatDescr(&descr_M_);
  cusparseSetMatType(descr_M_, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr_M_, CUSPARSE_INDEX_BASE_ZERO);

  // info (data structure where factorization is stored)
  // this is done in the constructor - however, this function might be called more than once
  cusolverSpDestroyGluInfo(info_M_);
  cusolverSpCreateGluInfo(&info_M_);

  cusolver_glu_enabled_ = true;
  return 0;
}

int RefactorizationSolver::initializeCusolverRf()
{
  if(!checkEvloserRfStatus(evloserRfCreate(&handle_rf_), "evloserRfCreate")) {
    return -1;
  }

#if defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  /*
   * hipSOLVER RF uses the default RF settings. Some CUDA RF tuning calls are
   * not portable to HIP.
   */
#else
  sp_status_ = evloserRfSetAlgs(handle_rf_, evloserRfFactorizationAlg2, evloserRfTriangularSolveAlg2);
  if(!checkEvloserRfStatus(sp_status_, "evloserRfSetAlgs")) {
    return -1;
  }

  sp_status_ = evloserRfSetMatrixFormat(handle_rf_, evloserRfMatrixFormatCsr, evloserRfUnitDiagonalStoredL);
  if(!checkEvloserRfStatus(sp_status_, "evloserRfSetMatrixFormat")) {
    return -1;
  }

  sp_status_ = evloserRfSetResetValuesFastMode(handle_rf_, evloserRfResetValuesFastModeOn);
  if(!checkEvloserRfStatus(sp_status_, "evloserRfSetResetValuesFastMode")) {
    return -1;
  }

  const double boost = 1e-12;
  const double zero = 1e-14;

  sp_status_ = evloserRfSetNumericProperties(handle_rf_, zero, boost);
  if(!checkEvloserRfStatus(sp_status_, "evloserRfSetNumericProperties")) {
    return -1;
  }
#endif

  cusolver_rf_enabled_ = true;
  return 0;
}

// call if both the matrix and the nnz structure changed or if convergence is
// poor while using refactorization.
int RefactorizationSolver::refactorizationSetupCusolverGLU()
{
#if defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  std::cerr << "EVLOSER GLU refactorization is not supported on HIP. Use RF instead.\n";
  return -1;
#endif

  // for now this ONLY WORKS if proceeded by KLU. Might be worth decoupling
  // later

  // get sizes
  const int nnzL = Numeric_->lnz;
  const int nnzU = Numeric_->unz;

  const int nnzM = (nnzL + nnzU - n_);

  /* parse the factorization */

  mia_ = new int[n_ + 1]{0};
  mja_ = new int[nnzM]{0};
  int* Lp = new int[n_ + 1];
  int* Li = new int[nnzL];
  // we can't use nullptr instead od Lx and Ux because it causes SEG FAULT. It
  // seems like a waste of memory though.

  double* Lx = new double[nnzL];
  int* Up = new int[n_ + 1];
  int* Ui = new int[nnzU];

  double* Ux = new double[nnzU];

  (void)klu_extract(Numeric_,
                       Symbolic_,
                       Lp,
                       Li,
                       Lx,
                       Up,
                       Ui,
                       Ux,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       &Common_);
  createM(n_, nnzL, Lp, Li, nnzU, Up, Ui);

  delete[] Lp;
  delete[] Li;
  delete[] Lx;
  delete[] Up;
  delete[] Ui;
  delete[] Ux;

  /* setup GLU */
  sp_status_ = cusolverSpDgluSetup(handle_cusolver_,
                                   n_,
                                   nnz_,
                                   descr_A_,
                                   mat_A_csr_->host_irows(),  // kRowPtr_,
                                   mat_A_csr_->host_jcols(),  // jCol_,
                                   Numeric_->Pnum,                /* base-0 */
                                   Symbolic_->Q,                  /* base-0 */
                                   nnzM,                          /* nnzM */
                                   descr_M_,
                                   mia_,
                                   mja_,
                                   info_M_);

  sp_status_ = cusolverSpDgluBufferSize(handle_cusolver_, info_M_, &size_M_);
  assert(CUSOLVER_STATUS_SUCCESS == sp_status_);

  buffer_size_ = size_M_;
  checkGpuErrors(evloserGpuMalloc((void**)&d_work_, buffer_size_));

  sp_status_ = cusolverSpDgluAnalysis(handle_cusolver_, info_M_, d_work_);
  assert(CUSOLVER_STATUS_SUCCESS == sp_status_);

  // reset and refactor so factors are ON THE GPU

  sp_status_ = cusolverSpDgluReset(handle_cusolver_,
                                   n_,
                                   /* A is original matrix */
                                   nnz_,
                                   descr_A_,
                                   mat_A_csr_->device_vals(),
                                   mat_A_csr_->device_irows(),
                                   mat_A_csr_->device_jcols(),
                                   info_M_);

  assert(CUSOLVER_STATUS_SUCCESS == sp_status_);
  sp_status_ = cusolverSpDgluFactor(handle_cusolver_, info_M_, d_work_);
  return 0;
}

int RefactorizationSolver::refactorizationSetupCusolverRf()
{
  // For now this path requires a prior KLU factorization.
  if(!validate_klu_factorization("cuSOLVER RF setup")) {
    return -1;
  }

  KluFactorData factors;
  if(!extract_klu_factors(Numeric_, Symbolic_, Common_, n_, factors, silent_output_)) {
    return -1;
  }

  HostCsrFactor L_csr = convert_csc_to_csr(n_, factors.nnzL, factors.Lp, factors.Li, factors.Lx);
  HostCsrFactor U_csr = convert_csc_to_csr(n_, factors.nnzU, factors.Up, factors.Ui, factors.Ux);

  if(!validate_host_csr_factor("L", n_, factors.nnzL, L_csr, silent_output_)) {
    return -1;
  }

  if(!validate_host_csr_factor("U", n_, factors.nnzU, U_csr, silent_output_)) {
    return -1;
  }

  checkGpuErrors(evloserGpuMalloc(&d_P_, n_ * sizeof(int)));
  checkGpuErrors(evloserGpuMalloc(&d_Q_, n_ * sizeof(int)));
  checkGpuErrors(evloserGpuMalloc(&d_T_, n_ * sizeof(double)));

  checkGpuErrors(evloserGpuMemcpy(d_P_, Numeric_->Pnum, n_ * sizeof(int), evloserMemcpyHostToDevice));
  checkGpuErrors(evloserGpuMemcpy(d_Q_, Symbolic_->Q, n_ * sizeof(int), evloserMemcpyHostToDevice));

  sp_status_ = evloserRfSetupHost(n_,
                                   nnz_,
                                   mat_A_csr_->host_irows(),
                                   mat_A_csr_->host_jcols(),
                                   mat_A_csr_->host_vals(),
                                   factors.nnzL,
                                   L_csr.rowptr.data(),
                                   L_csr.colind.data(),
                                   L_csr.values.data(),
                                   factors.nnzU,
                                   U_csr.rowptr.data(),
                                   U_csr.colind.data(),
                                   U_csr.values.data(),
                                   Numeric_->Pnum,
                                   Symbolic_->Q,
                                   handle_rf_);
  if(!checkEvloserRfStatus(sp_status_, "evloserRfSetupHost")) {
    return -1;
  }

  if(analyzeEvloserRf("GPU RF analysis") != 0) {
    return -1;
  }

  return refactorizeEvloserRf("GPU RF initial refactorization");
}

// Error checking utility for GPU backend calls
// KS: might later become part of src/Utils, putting it here for now
template<typename T>
void RefactorizationSolver::evloserCheckGpuError(T result, const char* const file, int const line)
{
  if(result != evloserGpuSuccess) {
    std::cout << "GPU backend error at " << file << ":" << line
              << ", error# " << static_cast<int>(result)
              << ": " << evloserGpuGetErrorString(result) << "\n";
    assert(false);
  }
}
#endif
}  // namespace EVLOSER
