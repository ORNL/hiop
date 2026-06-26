#ifndef HIOP_LINSOLVER_SPARSE_EVLOSER_PROVIDER
#define HIOP_LINSOLVER_SPARSE_EVLOSER_PROVIDER

namespace hiop
{

class hiopMatrixSparse;
class hiopNlpFormulation;
class hiopVector;

/**
 * @brief External ReSolve backend boundary for the HiOp EVLOSER sparse solver.
 *
 * The implementation owns ReSolve matrix storage, solver objects, execution
 * resources, and matrix-conversion state.
 */
class hiopLinSolverSparseEVLOSERProvider
{
public:
  virtual ~hiopLinSolverSparseEVLOSERProvider() = default;

  /**
   * @brief Updates and factorizes or refactorizes the supplied HiOp matrix.
   *
   * @return Zero on success or a negative value when HiOp should regularize
   * the matrix.
   */
  virtual int matrixChanged(hiopMatrixSparse& matrix) = 0;

  /**
   * @brief Solves the current linear system in place.
   *
   * @param x On entry, the right-hand side. On exit, the solution.
   */
  virtual bool solve(hiopVector& x) = 0;
};

/**
 * @brief Creates the external ReSolve-backed EVLOSER implementation.
 */
hiopLinSolverSparseEVLOSERProvider*
create_hiop_evloser_provider(const int& n,
                             const int& nnz,
                             hiopNlpFormulation* nlp);

}  // namespace hiop

#endif
