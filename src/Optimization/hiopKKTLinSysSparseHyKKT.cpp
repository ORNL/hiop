#include "hiopKKTLinSysSparse.hpp"

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

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

namespace
{
#ifdef HIOP_USE_GPU

template<typename T, typename I>
__global__ void map_triplet_to_csr(T* dst, const T* src, const I* map, I n)
{
  const I tid = static_cast<I>(blockDim.x * blockIdx.x + threadIdx.x);

  if(tid < n) {
    const I triplet = map[tid];
    dst[tid] = triplet >= 0 ? src[triplet] : T{0};
  }
}

template<typename T, typename I>
__global__ void add_diagonal_to_csr(T* dst, const T* diagonal, const I* map, I n)
{
  const I tid = static_cast<I>(blockDim.x * blockIdx.x + threadIdx.x);

  if(tid < n) {
    dst[map[tid]] += diagonal[tid];
  }
}

bool copy_mapping_to_device(int*& dst, const int* src, size_t count)
{
#ifdef HIOP_USE_CUDA
  if(cudaMalloc(reinterpret_cast<void**>(&dst), count * sizeof(int)) != cudaSuccess) {
    return false;
  }

  if(cudaMemcpy(dst, src, count * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(dst);
    dst = nullptr;
    return false;
  }
#elif defined(HIOP_USE_HIP)
  if(hipMalloc(reinterpret_cast<void**>(&dst), count * sizeof(int)) != hipSuccess) {
    return false;
  }

  if(hipMemcpy(dst, src, count * sizeof(int), hipMemcpyHostToDevice) != hipSuccess) {
    hipFree(dst);
    dst = nullptr;
    return false;
  }
#endif

  return true;
}

#endif

struct CsrEntry
{
  int row;
  int col;
  int triplet;
};

/**
 * Build a sorted ReSolve CSR structure and a mapping back to HiOp triplet entries.
 *
 * Symmetric matrices are expanded to full CSR. When ensure_diagonal is true,
 * missing diagonal entries are inserted with a triplet mapping of -1.
 */
bool build_csr_structure(int nrows,
                         int ncols,
                         const int* triplet_rows,
                         const int* triplet_cols,
                         int triplet_nnz,
                         bool symmetric,
                         bool ensure_diagonal,
                         ReSolve::matrix::Csr*& matrix,
                         int*& csr_to_triplet,
                         int*& diag_to_csr)
{
  std::vector<CsrEntry> entries;
  entries.reserve(static_cast<size_t>(symmetric ? 2 * triplet_nnz + nrows : triplet_nnz));

  std::vector<bool> has_diagonal(static_cast<size_t>(nrows), false);

  for(int k = 0; k < triplet_nnz; ++k) {
    const int row = triplet_rows[k];
    const int col = triplet_cols[k];

    entries.push_back({row, col, k});

    if(row == col) {
      has_diagonal[static_cast<size_t>(row)] = true;
    } else if(symmetric) {
      entries.push_back({col, row, k});
    }
  }

  if(ensure_diagonal) {
    for(int i = 0; i < nrows; ++i) {
      if(!has_diagonal[static_cast<size_t>(i)]) {
        entries.push_back({i, i, -1});
      }
    }
  }

  std::sort(entries.begin(), entries.end(), [](const CsrEntry& a, const CsrEntry& b) {
    return a.row < b.row || (a.row == b.row && a.col < b.col);
  });

  std::vector<ReSolve::index_type> row_ptr(static_cast<size_t>(nrows + 1), 0);
  std::vector<ReSolve::index_type> col_idx(entries.size());
  std::vector<double> values(entries.size(), 0.0);

  csr_to_triplet = new int[entries.size()];

  if(ensure_diagonal) {
    diag_to_csr = new int[static_cast<size_t>(nrows)];
    std::fill(diag_to_csr, diag_to_csr + nrows, -1);
  }

  for(size_t k = 0; k < entries.size(); ++k) {
    const auto& entry = entries[k];

    row_ptr[static_cast<size_t>(entry.row + 1)]++;
    col_idx[k] = entry.col;
    csr_to_triplet[k] = entry.triplet;

    if(ensure_diagonal && entry.row == entry.col) {
      diag_to_csr[entry.row] = static_cast<int>(k);
    }
  }

  std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.begin());

  matrix = new ReSolve::matrix::Csr(
      nrows, ncols, static_cast<ReSolve::index_type>(entries.size()), symmetric, true);

  if(matrix->allocateMatrixData(ReSolve::memory::HOST) != 0) {
    return false;
  }

  return matrix->copyFromExternal(row_ptr.data(),
                                  col_idx.data(),
                                  values.data(),
                                  ReSolve::memory::HOST,
                                  ReSolve::memory::HOST) == 0;
}

}  // namespace

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
      vector_handler_{nullptr},
      H_csr_to_triplet_host_{nullptr},
      H_diag_to_csr_host_{nullptr},
      J_csr_to_triplet_host_{nullptr},
      J_d_csr_to_triplet_host_{nullptr}
#ifdef HIOP_USE_GPU
      ,
      H_csr_to_triplet_device_{nullptr},
      H_diag_to_csr_device_{nullptr},
      J_csr_to_triplet_device_{nullptr},
      J_d_csr_to_triplet_device_{nullptr}
#endif
{}

hiopKKTLinSysCompressedSparseXDYcYdHyKKT::~hiopKKTLinSysCompressedSparseXDYcYdHyKKT()
{
  delete hykkt_solver_;

  delete H_;
  delete D_s_;
  delete J_;
  delete J_d_;

  delete[] H_csr_to_triplet_host_;
  delete[] H_diag_to_csr_host_;
  delete[] J_csr_to_triplet_host_;
  delete[] J_d_csr_to_triplet_host_;

#ifdef HIOP_USE_CUDA
  if(H_csr_to_triplet_device_) cudaFree(H_csr_to_triplet_device_);
  if(H_diag_to_csr_device_) cudaFree(H_diag_to_csr_device_);
  if(J_csr_to_triplet_device_) cudaFree(J_csr_to_triplet_device_);
  if(J_d_csr_to_triplet_device_) cudaFree(J_d_csr_to_triplet_device_);
#endif

#ifdef HIOP_USE_HIP
  if(H_csr_to_triplet_device_) hipFree(H_csr_to_triplet_device_);
  if(H_diag_to_csr_device_) hipFree(H_diag_to_csr_device_);
  if(J_csr_to_triplet_device_) hipFree(J_csr_to_triplet_device_);
  if(J_d_csr_to_triplet_device_) hipFree(J_d_csr_to_triplet_device_);
#endif

  delete vector_handler_;
  delete matrix_handler_;

#ifdef HIOP_USE_CUDA
  delete cuda_workspace_;
#endif

#ifdef HIOP_USE_HIP
  delete hip_workspace_;
#endif
}

bool hiopKKTLinSysCompressedSparseXDYcYdHyKKT::initialize_matrix_blocks()
{
  assert(HessSp_);
  assert(Jac_cSp_);
  assert(Jac_dSp_);

  const int nx = HessSp_->n();
  const int neq = Jac_cSp_->m();
  const int nineq = Jac_dSp_->m();

  std::vector<int> H_rows(static_cast<size_t>(HessSp_->numberOfNonzeros()));
  std::vector<int> H_cols(static_cast<size_t>(HessSp_->numberOfNonzeros()));
  std::vector<double> H_vals(static_cast<size_t>(HessSp_->numberOfNonzeros()));

  HessSp_->copy_to(H_rows.data(), H_cols.data(), H_vals.data());

  if(!build_csr_structure(nx,
                          nx,
                          H_rows.data(),
                          H_cols.data(),
                          HessSp_->numberOfNonzeros(),
                          true,
                          true,
                          H_,
                          H_csr_to_triplet_host_,
                          H_diag_to_csr_host_)) {
    nlp_->log->printf(hovError, "Failed to construct the ReSolve HyKKT Hessian block.\n");
    return false;
  }

  std::vector<int> J_rows(static_cast<size_t>(Jac_cSp_->numberOfNonzeros()));
  std::vector<int> J_cols(static_cast<size_t>(Jac_cSp_->numberOfNonzeros()));
  std::vector<double> J_vals(static_cast<size_t>(Jac_cSp_->numberOfNonzeros()));

  const_cast<hiopMatrixSparse*>(Jac_cSp_)->copy_to(J_rows.data(), J_cols.data(), J_vals.data());

  int* unused_diag{nullptr};

  if(!build_csr_structure(neq,
                          nx,
                          J_rows.data(),
                          J_cols.data(),
                          Jac_cSp_->numberOfNonzeros(),
                          false,
                          false,
                          J_,
                          J_csr_to_triplet_host_,
                          unused_diag)) {
    nlp_->log->printf(hovError, "Failed to construct the ReSolve HyKKT equality Jacobian block.\n");
    return false;
  }

  std::vector<int> J_d_rows(static_cast<size_t>(Jac_dSp_->numberOfNonzeros()));
  std::vector<int> J_d_cols(static_cast<size_t>(Jac_dSp_->numberOfNonzeros()));
  std::vector<double> J_d_vals(static_cast<size_t>(Jac_dSp_->numberOfNonzeros()));

  const_cast<hiopMatrixSparse*>(Jac_dSp_)->copy_to(J_d_rows.data(), J_d_cols.data(), J_d_vals.data());

  if(!build_csr_structure(nineq,
                          nx,
                          J_d_rows.data(),
                          J_d_cols.data(),
                          Jac_dSp_->numberOfNonzeros(),
                          false,
                          false,
                          J_d_,
                          J_d_csr_to_triplet_host_,
                          unused_diag)) {
    nlp_->log->printf(hovError, "Failed to construct the ReSolve HyKKT inequality Jacobian block.\n");
    return false;
  }

  std::vector<ReSolve::index_type> D_s_rows(static_cast<size_t>(nineq + 1));
  std::vector<ReSolve::index_type> D_s_cols(static_cast<size_t>(nineq));
  std::vector<double> D_s_vals(static_cast<size_t>(nineq), 0.0);

  std::iota(D_s_rows.begin(), D_s_rows.end(), 0);
  std::iota(D_s_cols.begin(), D_s_cols.end(), 0);

  D_s_ = new ReSolve::matrix::Csr(nineq, nineq, nineq, false, true);

  if(D_s_->allocateMatrixData(ReSolve::memory::HOST) != 0 ||
     D_s_->copyFromExternal(D_s_rows.data(),
                            D_s_cols.data(),
                            D_s_vals.data(),
                            ReSolve::memory::HOST,
                            ReSolve::memory::HOST) != 0) {
    nlp_->log->printf(hovError, "Failed to construct the ReSolve HyKKT slack diagonal block.\n");
    return false;
  }
#ifdef HIOP_USE_GPU
  if(nlp_->options->GetString("mem_space") == "device") {
    if(H_->allocateMatrixData(ReSolve::memory::DEVICE) != 0 ||
       D_s_->allocateMatrixData(ReSolve::memory::DEVICE) != 0 ||
       J_->allocateMatrixData(ReSolve::memory::DEVICE) != 0 ||
       J_d_->allocateMatrixData(ReSolve::memory::DEVICE) != 0) {
      nlp_->log->printf(hovError, "Failed to allocate ReSolve HyKKT device matrix storage.\n");
      return false;
    }

    if(H_->syncData(ReSolve::memory::DEVICE) != 0 ||
       D_s_->syncData(ReSolve::memory::DEVICE) != 0 ||
       J_->syncData(ReSolve::memory::DEVICE) != 0 ||
       J_d_->syncData(ReSolve::memory::DEVICE) != 0) {
      nlp_->log->printf(hovError, "Failed to copy ReSolve HyKKT matrix structure to the device.\n");
      return false;
    }

    if(!copy_mapping_to_device(H_csr_to_triplet_device_,
                               H_csr_to_triplet_host_,
                               static_cast<size_t>(H_->getNnz())) ||
       !copy_mapping_to_device(H_diag_to_csr_device_,
                               H_diag_to_csr_host_,
                               static_cast<size_t>(nx)) ||
       !copy_mapping_to_device(J_csr_to_triplet_device_,
                               J_csr_to_triplet_host_,
                               static_cast<size_t>(J_->getNnz())) ||
       !copy_mapping_to_device(J_d_csr_to_triplet_device_,
                               J_d_csr_to_triplet_host_,
                               static_cast<size_t>(J_d_->getNnz()))) {
      nlp_->log->printf(hovError, "Failed to copy ReSolve HyKKT matrix mappings to the device.\n");
      return false;
    }
  }
#endif

  return true;
}

bool hiopKKTLinSysCompressedSparseXDYcYdHyKKT::update_matrix_blocks()
{
  assert(H_);
  assert(D_s_);
  assert(J_);
  assert(J_d_);
  assert(HessSp_);
  assert(Jac_cSp_);
  assert(Jac_dSp_);
  assert(Hx_);
  assert(Hd_);

  const std::string mem_space = nlp_->options->GetString("mem_space");

#ifdef HIOP_USE_GPU
  if(mem_space == "device") {
    double* H_values = H_->getValues(ReSolve::memory::DEVICE);
    double* D_s_values = D_s_->getValues(ReSolve::memory::DEVICE);
    double* J_values = J_->getValues(ReSolve::memory::DEVICE);
    double* J_d_values = J_d_->getValues(ReSolve::memory::DEVICE);

    const double* Hess_values = HessSp_->M();
    const double* Jac_c_values = Jac_cSp_->M();
    const double* Jac_d_values = Jac_dSp_->M();
    const double* Hx_values = Hx_->local_data();
    const double* Hd_values = Hd_->local_data();

    if(!H_values || !D_s_values || !J_values || !J_d_values ||
       !Hess_values || !Jac_c_values || !Jac_d_values || !Hx_values || !Hd_values) {
      nlp_->log->printf(hovError, "Failed to access HyKKT device matrix block values.\n");
      return false;
    }

    constexpr unsigned int blocksize = 512;

    unsigned int gridsize =
        (static_cast<unsigned int>(H_->getNnz()) + blocksize - 1) / blocksize;

    map_triplet_to_csr<double, int>
        <<<gridsize, blocksize>>>(H_values,
                                 Hess_values,
                                 H_csr_to_triplet_device_,
                                 static_cast<int>(H_->getNnz()));

    gridsize =
        (static_cast<unsigned int>(Hx_->get_size()) + blocksize - 1) / blocksize;

    add_diagonal_to_csr<double, int>
        <<<gridsize, blocksize>>>(H_values,
                                 Hx_values,
                                 H_diag_to_csr_device_,
                                 static_cast<int>(Hx_->get_size()));

    gridsize =
        (static_cast<unsigned int>(J_->getNnz()) + blocksize - 1) / blocksize;

    map_triplet_to_csr<double, int>
        <<<gridsize, blocksize>>>(J_values,
                                 Jac_c_values,
                                 J_csr_to_triplet_device_,
                                 static_cast<int>(J_->getNnz()));

    gridsize =
        (static_cast<unsigned int>(J_d_->getNnz()) + blocksize - 1) / blocksize;

    map_triplet_to_csr<double, int>
        <<<gridsize, blocksize>>>(J_d_values,
                                 Jac_d_values,
                                 J_d_csr_to_triplet_device_,
                                 static_cast<int>(J_d_->getNnz()));

#ifdef HIOP_USE_CUDA
    if(cudaGetLastError() != cudaSuccess ||
       cudaMemcpy(D_s_values,
                  Hd_values,
                  static_cast<size_t>(D_s_->getNnz()) * sizeof(double),
                  cudaMemcpyDeviceToDevice) != cudaSuccess) {
      nlp_->log->printf(hovError, "Failed to update ReSolve HyKKT matrix blocks on CUDA.\n");
      return false;
    }
#elif defined(HIOP_USE_HIP)
    if(hipGetLastError() != hipSuccess ||
       hipMemcpy(D_s_values,
                 Hd_values,
                 static_cast<size_t>(D_s_->getNnz()) * sizeof(double),
                 hipMemcpyDeviceToDevice) != hipSuccess) {
      nlp_->log->printf(hovError, "Failed to update ReSolve HyKKT matrix blocks on HIP.\n");
      return false;
    }
#endif

    if(H_->setUpdated(ReSolve::memory::DEVICE) != 0 ||
       D_s_->setUpdated(ReSolve::memory::DEVICE) != 0 ||
       J_->setUpdated(ReSolve::memory::DEVICE) != 0 ||
       J_d_->setUpdated(ReSolve::memory::DEVICE) != 0) {
      nlp_->log->printf(hovError, "Failed to mark ReSolve HyKKT device matrix blocks as updated.\n");
      return false;
    }

    return true;
  }
#endif

  double* H_values = H_->getValues(ReSolve::memory::HOST);
  double* D_s_values = D_s_->getValues(ReSolve::memory::HOST);
  double* J_values = J_->getValues(ReSolve::memory::HOST);
  double* J_d_values = J_d_->getValues(ReSolve::memory::HOST);

  if(!H_values || !D_s_values || !J_values || !J_d_values) {
    nlp_->log->printf(hovError, "Failed to access ReSolve HyKKT matrix values.\n");
    return false;
  }

  const double* Hess_values = HessSp_->M();
  const double* Jac_c_values = Jac_cSp_->M();
  const double* Jac_d_values = Jac_dSp_->M();
  const double* Hx_values = Hx_->local_data();
  const double* Hd_values = Hd_->local_data();

  if(!Hess_values || !Jac_c_values || !Jac_d_values || !Hx_values || !Hd_values) {
    nlp_->log->printf(hovError, "Failed to access HiOp KKT matrix block values.\n");
    return false;
  }

  for(ReSolve::index_type k = 0; k < H_->getNnz(); ++k) {
    const int triplet = H_csr_to_triplet_host_[k];
    H_values[k] = triplet >= 0 ? Hess_values[triplet] : 0.0;
  }

  for(index_type i = 0; i < Hx_->get_size(); ++i) {
    assert(H_diag_to_csr_host_[i] >= 0);
    H_values[H_diag_to_csr_host_[i]] += Hx_values[i];
  }

  for(ReSolve::index_type k = 0; k < J_->getNnz(); ++k) {
    J_values[k] = Jac_c_values[J_csr_to_triplet_host_[k]];
  }

  for(ReSolve::index_type k = 0; k < J_d_->getNnz(); ++k) {
    J_d_values[k] = Jac_d_values[J_d_csr_to_triplet_host_[k]];
  }

  for(ReSolve::index_type k = 0; k < D_s_->getNnz(); ++k) {
    D_s_values[k] = Hd_values[k];
  }

  if(H_->setUpdated(ReSolve::memory::HOST) != 0 ||
     D_s_->setUpdated(ReSolve::memory::HOST) != 0 ||
     J_->setUpdated(ReSolve::memory::HOST) != 0 ||
     J_d_->setUpdated(ReSolve::memory::HOST) != 0) {
    nlp_->log->printf(hovError, "Failed to mark ReSolve HyKKT matrix blocks as updated.\n");
    return false;
  }

  return true;
}

bool hiopKKTLinSysCompressedSparseXDYcYdHyKKT::build_kkt_matrix(const hiopPDPerturbation& pdreg)
{
  delta_wx_ = perturb_calc_->get_curr_delta_wx();
  delta_wd_ = perturb_calc_->get_curr_delta_wd();
  delta_cc_ = perturb_calc_->get_curr_delta_cc();
  delta_cd_ = perturb_calc_->get_curr_delta_cd();

  HessSp_ = dynamic_cast<hiopMatrixSparse*>(Hess_);
  Jac_cSp_ = dynamic_cast<const hiopMatrixSparse*>(Jac_c_);
  Jac_dSp_ = dynamic_cast<const hiopMatrixSparse*>(Jac_d_);

  if(!HessSp_ || !Jac_cSp_ || !Jac_dSp_) {
    assert(false);
    return false;
  }

  const size_type nx = HessSp_->n();
  const size_type nd = Jac_dSp_->m();

  if(nullptr == Hx_) {
    Hx_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nx);
    assert(Hx_);
  }
  Hx_->startingAtCopyFromStartingAt(0, *Dx_, 0);
  Hx_->axpy(1., *delta_wx_);

  if(nullptr == Hd_) {
    Hd_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nd);
    assert(Hd_);
  }
  Hd_->startingAtCopyFromStartingAt(0, *Dd_, 0);
  Hd_->axpy(1., *delta_wd_);

  // HyKKT currently has no matrix blocks for the dual regularization terms.
  if(delta_cc_->infnorm() != 0.0 || delta_cd_->infnorm() != 0.0) {
    nlp_->log->printf(hovError,
                      "ReSolve HyKKT does not support nonzero dual regularization.\n");
    return false;
  }

  if(nullptr == H_) {
    if(!initialize_matrix_blocks()) {
      return false;
    }
  }

  return update_matrix_blocks();
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
