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

#include <cassert>

/**
 * @brief Implements the HiOp sparse linear solver interface using the
 * configured EVLOSER backend provider.
 *
 * The concrete provider is selected when HiOp is configured. Backend-specific
 * matrix storage, factorization state, and execution resources are owned by
 * the provider rather than by this HiOp-facing wrapper.
 *
 * @ingroup LinearSolvers
 */

namespace hiop
{

class hiopLinSolverSparseEVLOSERProvider;

class hiopLinSolverSymSparseEVLOSER : public hiopLinSolverSymSparse
{
public:
  hiopLinSolverSymSparseEVLOSER(const int& n,
                               const int& nnz,
                               hiopNlpFormulation* nlp);

  virtual ~hiopLinSolverSymSparseEVLOSER();

  /**
   * @brief Notifies the configured provider that the matrix changed.
   */
  virtual int matrixChanged();

  /**
   * @brief Solves a linear system.
   *
   * @param x On entry, the right-hand side of the system.
   *
   * @post On exit, `x` is overwritten with the solution.
   */
  virtual bool solve(hiopVector& x);

  /** Multiple right-hand sides are not supported yet. */
  virtual bool solve(hiopMatrix& /* x */)
  {
    assert(false && "not yet supported");
    return false;
  }

protected:
  hiopLinSolverSparseEVLOSERProvider* provider_;
};

}  // namespace hiop

#endif