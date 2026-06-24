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
 * @file hiopLinSolverSparseEVLOSER.hpp
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 * @author Slaven Peles <peless@ornl.gov>, ORNL
 *
 */

#ifndef HIOP_LINSOLVER_EVLOSER
#define HIOP_LINSOLVER_EVLOSER

#include "hiopLinSolver.hpp"
#include "hiopMatrixSparseTriplet.hpp"
#include "evloser_execution_mode.hpp"
#include <unordered_map>

/** Implements the sparse linear solver class using the EVLOSER interface
 *  to the embedded EVLOSER backend.
 *
 * @ingroup LinearSolvers
 */

namespace EVLOSER
{
// Forward declaration of inner IR class
class IterativeRefinement;
class MatrixCsr;
class RefactorizationSolver;
}  // namespace EVLOSER

namespace hiop
{

class hiopLinSolverSymSparseEVLOSER : public hiopLinSolverSymSparse
{
public:
  // constructor
  hiopLinSolverSymSparseEVLOSER(const int& n, const int& nnz, hiopNlpFormulation* nlp);
  virtual ~hiopLinSolverSymSparseEVLOSER();

  /**
   * @brief Triggers a refactorization of the matrix, if necessary.
   * Overload from base class.
   * In this case, KLU (SuiteSparse) is used to refactor
   */
  virtual int matrixChanged();

  /**
   * @brief Solves a linear system.
   *
   * @param x is on entry the right hand side(s) of the system to be solved.
   *
   * @post On exit `x` is overwritten with the solution(s).
   */
  virtual bool solve(hiopVector& x_);

  /** Multiple rhs not supported yet */
  virtual bool solve(hiopMatrix& /* x */)
  {
    assert(false && "not yet supported");
    return false;
  }

protected:
  const EVLOSER::ExecutionMode execution_mode_;  ///< Selected EVLOSER execution path
  EVLOSER::RefactorizationSolver* solver_;

  int m_;    ///< number of rows of the whole matrix
  int n_;    ///< number of cols of the whole matrix
  int nnz_;  ///< number of nonzeros in the matrix

  // Mapping on the host
  int* index_convert_CSR2Triplet_host_;
  int* index_convert_extra_Diag2CSR_host_;

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  // Mapping on the device
  int* index_convert_CSR2Triplet_device_;
  int* index_convert_extra_Diag2CSR_device_;
#endif

  // Algorithm control flags
  int factorizationSetupSucc_;
  bool is_first_call_;

  hiopMatrixSparse* M_host_{nullptr};  ///< Host mirror for the KKT matrix

  /** called the very first time a matrix is factored. Perform KLU
   * factorization, allocate all aux variables
   *
   * @note Converts HiOp triplet matrix to CSR format.
   */
  virtual void firstCall();

  /**
   * @brief Updates matrix values from HiOp object.
   *
   * @note This function maps data from HiOp supplied matrix M_ to data structures
   * used by the linear solver.
   */
  void update_matrix_values();

  /** Function to compute nnz and set row pointers */
  void compute_nnz();
  /** Function to compute column indices and matrix values arrays */
  void set_csr_indices_values();

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
  template<typename T>
  void hiopCheckGpuError(T result, const char* const file, int const line);
#endif
};

}  // namespace hiop

#endif
