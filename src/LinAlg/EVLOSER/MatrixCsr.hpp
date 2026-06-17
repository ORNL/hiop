#pragma once

namespace EVLOSER
{

class MatrixCsr
{
public:
  MatrixCsr();
  ~MatrixCsr();

  /// Allocate device and host row-pointer storage for an n-by-n CSR matrix.
  void allocate_size(int n);

  /// Allocate device and host column-index/value storage for the current CSR matrix.
  void allocate_nnz(int nnz);

  /// Release all owned device and host CSR storage.
  void clear_data();

  /// Return the matrix dimension.
  int n() const { return n_; }

  /// Return the number of stored nonzeros.
  int nnz() const { return nnz_; }

  /// Return true when the required device CSR arrays have been allocated.
  bool has_device_storage() const;

  /// Return true when the required host CSR mirror arrays have been allocated.
  bool has_host_mirror() const;

  /**
   * @brief Validate the host-side CSR structure before factorization/refactorization.
   *
   * Checks row-pointer monotonicity, final nnz consistency, and column-index bounds.
   *
   * @param caller Name of the caller used in diagnostic messages.
   * @param silent_output Suppress diagnostic output when true.
   * @return true if the host CSR structure is valid.
   */
  bool validate_host_structure(const char* caller, bool silent_output) const;

  /// Return device row-pointer storage.
  int* device_irows() { return irows_; }

  /// Return const device row-pointer storage.
  const int* device_irows() const { return irows_; }

  /// Return device column-index storage.
  int* device_jcols() { return jcols_; }

  /// Return const device column-index storage.
  const int* device_jcols() const { return jcols_; }

  /// Return device value storage.
  double* device_vals() { return vals_; }

  /// Return const device value storage.
  const double* device_vals() const { return vals_; }

  /// Return host row-pointer mirror storage.
  int* host_irows() { return irows_host_; }

  /// Return const host row-pointer mirror storage.
  const int* host_irows() const { return irows_host_; }

  /// Return host column-index mirror storage.
  int* host_jcols() { return jcols_host_; }

  /// Return const host column-index mirror storage.
  const int* host_jcols() const { return jcols_host_; }

  /// Return host value mirror storage.
  double* host_vals() { return vals_host_; }

  /// Return const host value mirror storage.
  const double* host_vals() const { return vals_host_; }

  /// Copy host-side CSR arrays into device storage.
  void update_from_host_mirror();

  /// Copy device CSR arrays into the host mirror.
  void copy_to_host_mirror();

private:
  int n_{0};
  int nnz_{0};

  int* irows_{nullptr};
  int* jcols_{nullptr};
  double* vals_{nullptr};

  int* irows_host_{nullptr};
  int* jcols_host_{nullptr};
  double* vals_host_{nullptr};

  /**
   * @brief Check for CUDA errors.
   *
   * @tparam T - type of the result
   * @param result - result value
   * @param file   - file name where the error occured
   * @param line   - line at which the error occured
   */
  template<typename T>
  void evloserCheckCudaError(T result, const char* const file, int const line);
};

}  // namespace EVLOSER
