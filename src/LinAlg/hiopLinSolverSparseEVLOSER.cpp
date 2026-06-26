#include "hiopLinSolverSparseEVLOSER.hpp"
#include "hiopLinSolverSparseEVLOSERProvider.hpp"

#include <cassert>

namespace hiop
{

hiopLinSolverSymSparseEVLOSER::hiopLinSolverSymSparseEVLOSER(
    const int& n,
    const int& nnz,
    hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparse(n, nnz, nlp),
      provider_{create_hiop_evloser_provider(n, nnz, nlp)}
{
  assert(provider_ != nullptr);
}

hiopLinSolverSymSparseEVLOSER::~hiopLinSolverSymSparseEVLOSER()
{
  delete provider_;
  provider_ = nullptr;
}

int hiopLinSolverSymSparseEVLOSER::matrixChanged()
{
  assert(provider_ != nullptr);
  assert(M_ != nullptr);

  return provider_->matrixChanged(*M_);
}

bool hiopLinSolverSymSparseEVLOSER::solve(hiopVector& x)
{
  assert(provider_ != nullptr);

  return provider_->solve(x);
}

}  // namespace hiop
