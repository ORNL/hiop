#include "hiopKKTLinSysSparse.hpp"

#include <resolve/hykkt/HyKKTSolver.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>

#ifdef HIOP_USE_CUDA
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#endif

#ifdef HIOP_USE_HIP
#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>
#endif

namespace hiop
{

/* *************************************************************************
 * For class hiopKKTLinSysCompressedSparseXDYcYdHyKKT
 * *************************************************************************
 */
hiopKKTLinSysCompressedSparseXDYcYdHyKKT::hiopKKTLinSysCompressedSparseXDYcYdHyKKT(hiopNlpFormulation* nlp)
    : hiopKKTLinSysCompressedSparseXDYcYd(nlp),
      hykkt_solver_{nullptr},
      H_{nullptr},
      D_s_{nullptr},
      J_{nullptr},
      J_d_{nullptr},
#ifdef HIOP_USE_CUDA
      cuda_workspace_{nullptr},
#endif
#ifdef HIOP_USE_HIP
      hip_workspace_{nullptr},
#endif
      matrix_handler_{nullptr},
      vector_handler_{nullptr}
{}

hiopKKTLinSysCompressedSparseXDYcYdHyKKT::~hiopKKTLinSysCompressedSparseXDYcYdHyKKT()
{
  delete hykkt_solver_;

  delete H_;
  delete D_s_;
  delete J_;
  delete J_d_;

  delete vector_handler_;
  delete matrix_handler_;

#ifdef HIOP_USE_CUDA
  delete cuda_workspace_;
#endif

#ifdef HIOP_USE_HIP
  delete hip_workspace_;
#endif
}

bool hiopKKTLinSysCompressedSparseXDYcYdHyKKT::build_kkt_matrix(const hiopPDPerturbation& pdreg)
{
  return hiopKKTLinSysCompressedSparseXDYcYd::build_kkt_matrix(pdreg);
}

bool hiopKKTLinSysCompressedSparseXDYcYdHyKKT::solveCompressed(hiopVector& rx,
                                                               hiopVector& rd,
                                                               hiopVector& ryc,
                                                               hiopVector& ryd,
                                                               hiopVector& dx,
                                                               hiopVector& dd,
                                                               hiopVector& dyc,
                                                               hiopVector& dyd)
{
  return hiopKKTLinSysCompressedSparseXDYcYd::solveCompressed(rx, rd, ryc, ryd, dx, dd, dyc, dyd);
}

int hiopKKTLinSysCompressedSparseXDYcYdHyKKT::factorizeWithCurvCheck()
{
  return hiopKKTLinSysCurvCheck::factorizeWithCurvCheck();
}

}  // namespace hiop
