#pragma once
#include <cstdint>

/**
 * Integer type of the LAPACK/BLAS library Xatu is linked against, used by Xatu's own direct LAPACK calls.
 * Default (LP64): 32-bit. With an ILP64 library (64-bit integers) build with `make ILP64=1`, which defines
 * XATU_ILP64 together with ARMA_BLAS_64BIT_INT, so that Armadillo's calls and Xatu's own use the same width.
 *
 * Why it matters: a 32-bit LAPACK computes workspace sizes in 32-bit integers. zheevd (Armadillo's eig_sym)
 * needs a real workspace of 1 + 5N + 2N^2, which overflows above N = 32766, so with LP64 'diag' is limited to
 * that dimension (larger problems are switched to zheevr, whose workspace is O(N)). With ILP64 there is no
 * such limit.
 */
#ifdef XATU_ILP64
  #ifndef ARMA_BLAS_64BIT_INT
    #error "XATU_ILP64 requires ARMA_BLAS_64BIT_INT (Armadillo must pass 64-bit integers to the same library)"
  #endif
  typedef long long lapack_int;   // same type as Armadillo's blas_int under ARMA_BLAS_64BIT_INT
  static_assert(sizeof(lapack_int) == 8, "ILP64 needs a 64-bit lapack_int");
  // No 32-bit workspace limit for zheevd
  const int64_t ZHEEVD_MAX_DIM = INT64_MAX;
#else
  #ifdef ARMA_BLAS_64BIT_INT
    #error "ARMA_BLAS_64BIT_INT without XATU_ILP64: Xatu's own LAPACK calls would pass 32-bit integers (use make ILP64=1)"
  #endif
  typedef int lapack_int;
  // Largest N with 1 + 5N + 2N^2 <= 2^31 - 1 (real workspace of zheevd with eigenvectors)
  const int64_t ZHEEVD_MAX_DIM = 32766;
#endif
