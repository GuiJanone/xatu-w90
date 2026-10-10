#define ARMA_MAX_ELEM 0x200000000ULL
#define ARMA_64BIT_WORD
#include <armadillo>
#include <functional>

namespace xatu {
    void davidson_method(arma::vec&, arma::cx_mat&, const arma::cx_mat&, int neigval = 4, double tol = 1E-8);
    
    void davidson_method_new(arma::vec&, arma::cx_mat&, const arma::cx_mat&, int neigval = 4, double tol = 1E-8);
    
    
    void diagonalize_partial(arma::vec&, arma::cx_mat&, arma::cx_mat&, int neigval = 4, bool preserve_H = false);
    void diagonalize_partial_range(arma::vec&, arma::cx_mat&, arma::cx_mat&, int il, int iu);
#ifdef XATU_ILP64
    void check_lapack_ilp64();
#endif

    // Dense helpers on the lower triangle of a matrix (the strict upper triangle is neither read nor written)
    long long cholesky_lower_inplace(arma::cx_mat& A);                    // A = L L^H, LAPACK info returned
    void solve_lower(const arma::cx_mat& L, arma::cx_mat& X, bool conjtrans); // X <- L^-1 X  or  L^-H X
    void multiply_lower(const arma::cx_mat& L, arma::cx_mat& X);           // X <- L X

    // Block Krylov solver for the extremal eigenpairs of a Hermitian operator known only through its action on
    // blocks of vectors (see davidson.cpp). Returns false if it did not converge.
    struct KrylovReport { int iterations = 0; long long applications = 0; long long basis = 0; long long block = 0; double residual = 0.0; };
    long long krylov_block_size(long long n);
    bool krylov_applicable(long long n, long long nwanted);
    bool block_krylov_extremal(const std::function<void(arma::cx_mat&)>& op, long long n, long long nlow, long long nhigh,
                               arma::vec& mu, arma::cx_mat& X, double tol, KrylovReport& report);
    
}
