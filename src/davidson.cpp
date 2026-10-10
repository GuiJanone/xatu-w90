#include "xatu/davidson.hpp"
#include "xatu/lapack_int.hpp"

namespace xatu {

void davidson_method(
    arma::vec& eigval, 
    arma::cx_mat& eigvec, 
    const arma::cx_mat& mat, 
    int neigval, 
    double tol){

    int max_iterations = mat.n_rows/2;
    int k = neigval * 2;
        
    if(!mat.is_hermitian()){
        throw std::invalid_argument("davidson_method: provided matrix must be hermitian");
    }
    
    arma::cx_mat guess_eigvec = arma::eye<arma::cx_mat>(mat.n_rows, k);
    arma::cx_mat proyected_matrix;
    arma::vec aux_eigval = arma::ones(neigval);
    arma::cx_mat Q, R;

    for(int i = 0; i < max_iterations; i++){

        // QR decomposition
        Q.clear();
        arma::qr_econ(Q, R, guess_eigvec);

        // Proyect matrix on
        proyected_matrix = Q.t()*mat*Q;
        arma::eig_sym(eigval, eigvec, proyected_matrix);

        // Add new eigvec to guess
        eigvec = Q.cols(0, (i + 1)*k - 1) * eigvec.cols(0, k - 1);
        for (int j = 0; j < k; j++){
            arma::cx_mat w = (mat - eigval(j)*arma::eye<arma::cx_mat>(mat.n_rows, mat.n_cols))*eigvec.col(j);
            arma::cx_vec normalized_w = w/(eigval(j) - mat(j, j));
            Q.insert_cols(Q.n_cols, normalized_w);
        }

        // Check convergence
        if(arma::norm(eigval.subvec(0, neigval - 1) - aux_eigval) < tol){
            break;
        }

        // Prepare data for next iteration
        aux_eigval = eigval.subvec(0, neigval - 1);
        guess_eigvec.clear();
        guess_eigvec = Q;

        if (i == max_iterations - 1){
            std::cout << "Reached maximum number of iterations" << std::endl;
        }
    }

    // Store final eigenvectors
    eigval = eigval.subvec(0, neigval - 1);
    arma::cout << eigvec.n_rows << arma::endl;
    arma::cout << eigvec.n_cols << arma::endl;
};


// Attempt at optimizing davidson_method with claude
void davidson_method_new(
    arma::vec&          eigval,
    arma::cx_mat&       eigvec,
    const arma::cx_mat& mat,
    int                 neigval,
    double              tol)
{
    const int n       = mat.n_rows;
    const int max_sub = std::min(n, std::max(neigval * 10, 50));
    const int max_iter = 300;
    
    if (!mat.is_hermitian(1e-10))
        throw std::invalid_argument("davidson_method: matrix must be Hermitian");
    
    // Cache diagonal for preconditioner
    arma::cx_vec diag = mat.diag();
    
    // ----------------------------------------------------------------
    // Initial subspace: use neigval unit vectors (not 2*neigval).
    // Starting with the identity columns biases toward the natural
    // basis which can break degeneracies — use a random unitary start
    // for robustness with degenerate eigenvalues.
    // ----------------------------------------------------------------
    arma::cx_mat V(n, neigval, arma::fill::zeros);
    {
        arma::cx_mat rnd = arma::randn<arma::cx_mat>(n, neigval);
        arma::cx_mat Qinit, Rinit;
        arma::qr_econ(Qinit, Rinit, rnd);
        V = Qinit;
    }
    
    arma::vec    ritz_old = arma::vec(neigval, arma::fill::value(1e10));
    arma::cx_mat AV(n, 0);       // mat*V, grown incrementally
    arma::vec    theta;
    arma::cx_mat s;
    
    for (int iter = 0; iter < max_iter; iter++) {
        
        // ----------------------------------------------------------------
        // Orthonormalise V (thin QR). On first iter V is already
        // orthonormal; subsequent iters may add nearly-dependent vectors.
        // ----------------------------------------------------------------
        {
            arma::cx_mat Q, R;
            arma::qr_econ(Q, R, V);
            // Detect and discard numerically rank-deficient columns
            arma::vec diag_R = arma::abs(R.diag());
            double    thresh  = diag_R(0) * n * 1e-14;
            int       rank    = (int)arma::sum(diag_R > thresh);
            V = Q.cols(0, rank - 1);
        }
        
        // ----------------------------------------------------------------
        // Incremental mat-vec: only multiply columns added since last iter
        // ----------------------------------------------------------------
        int old_cols = (int)AV.n_cols;
        int new_cols = (int)V.n_cols;
        AV.resize(n, new_cols);
        for (int c = old_cols; c < new_cols; c++)
            AV.col(c) = mat * V.col(c);
        
        // ----------------------------------------------------------------
        // Rayleigh-Ritz projection and diagonalisation
        // H_proj is Hermitian by construction (V^H A V with A Hermitian)
        // ----------------------------------------------------------------
        arma::cx_mat H_proj = V.t() * AV;
        // Enforce exact Hermitian symmetry to avoid eig_sym drift
        H_proj = 0.5 * (H_proj + H_proj.t());
        arma::eig_sym(theta, s, H_proj);
        
        // Ritz vectors in full space for all neigval targets
        arma::cx_mat ritz = V * s.cols(0, neigval - 1);
        
        // ----------------------------------------------------------------
        // Convergence: check residual norm for each target vector,
        // not just eigenvalue difference. This correctly handles
        // degenerate subspaces where eigenvalues converge before
        // eigenvectors are properly separated.
        // ----------------------------------------------------------------
        arma::cx_mat AX = AV * s.cols(0, neigval - 1);
        arma::vec res_norms(neigval);
        for (int j = 0; j < neigval; j++){
            arma::cx_vec r = AX.col(j) - theta(j) * ritz.col(j);
            res_norms(j) = arma::norm(r);
        }
        
        if (iter > 0 && arma::max(res_norms) < tol) {
            eigval = theta.subvec(0, neigval - 1);
            eigvec = ritz;
            return;
        }
        ritz_old = theta.subvec(0, neigval - 1);
        
        // ----------------------------------------------------------------
        // Subspace restart: collapse to neigval Ritz vectors
        // ----------------------------------------------------------------
        if ((int)V.n_cols >= max_sub) {
            AV = AV * s.cols(0, neigval - 1);  // A*(V*s) = (AV)*s
            V  = ritz;
            continue;
        }
        
        // ----------------------------------------------------------------
        // Correction vectors for ALL neigval targets (not k=2*neigval).
        // Using more targets than needed inflates the subspace and can
        // introduce spurious mixing of degenerate states.
        // Preconditioner: t_j = (D - theta_j I)^{-1} r_j
        // ----------------------------------------------------------------
        for (int j = 0; j < neigval; j++) {
            arma::cx_vec r = AX.col(j) - theta(j) * ritz.col(j);
            
            arma::cx_vec t(n);
            for (int row = 0; row < n; row++) {
                std::complex<double> denom = diag(row) - theta(j);
                t(row) = (std::abs(denom) > 1e-10) ? r(row) / denom : r(row);
            }
            
            // Orthogonalise against current V (double Gram-Schmidt for stability)
            t -= V * (V.t() * t);
            t -= V * (V.t() * t);
            double nt = arma::norm(t);
            if (nt > 1e-12)
                V.insert_cols(V.n_cols, t / nt);
        }
    }
    
    std::cerr << "davidson_method: did not converge in " << max_iter
    << " iterations. Returning best approximation." << std::endl;
    eigval = theta.subvec(0, neigval - 1);
    eigvec = V * s.cols(0, neigval - 1);
};

// Direct LAPACK call to zheevr, computing only nstates lowest eigenvalues
// This avoids both the Armadillo element limit and the full workspace allocation
extern "C" void zheevr_(char*,               // JOBZ
                        char*,               // RANGE  
                        char*,               // UPLO
                        lapack_int*,                // N
                        std::complex<double>*, // A
                        lapack_int*,                // LDA
                        double*,             // VL
                        double*,             // VU
                        lapack_int*,                // IL
                        lapack_int*,                // IU
                        double*,             // ABSTOL
                        lapack_int*,                // M
                        double*,             // W
                        std::complex<double>*, // Z
                        lapack_int*,                // LDZ
                        lapack_int*,                // ISUPPZ
                        std::complex<double>*, // WORK
                        lapack_int*,                // LWORK
                        double*,             // RWORK
                        lapack_int*,                // LRWORK
                        lapack_int*,                // IWORK
                        lapack_int*,                // LIWORK
                        lapack_int*);               // INFO

void diagonalize_partial(arma::vec& eigval, arma::cx_mat& eigvec, 
                         arma::cx_mat& H, int nstates, bool preserve_H){
    
    arma::cx_mat* Hptr = &H;
    arma::cx_mat  Hcopy;
    if(preserve_H){
        Hcopy = H;
        Hptr  = &Hcopy;
    }
    
    
    
    lapack_int n   = Hptr->n_rows;
    lapack_int lda = n, ldz = n;
    // nstates = 0 (the -n 0 convention for "all states") or more than the dimension: return every eigenpair.
    // Without this, IU = 0 < IL made LAPACK reject the call, which crashed TDA runs above dimension 32766,
    // where 'diag' is switched to zheevr automatically (32-bit LAPACK).
    if(nstates <= 0 || nstates > n) nstates = n;
    lapack_int il  = 1, iu = nstates;
    lapack_int m_found;
    double abstol = 0.0, vl = 0.0, vu = 0.0;
    lapack_int lwork = -1, lrwork = -1, liwork = -1, info;
    
    // W must have length n (LAPACK): with an index range zheevr bisects (dstebz), which stores every eigenvalue
    // tied at the boundary before discarding the extra ones, so -n inside a degenerate level wrote past an
    // nstates-long array (heap corruption, segfault). Trimmed to the m_found eigenpairs after the call.
    eigval.set_size(n);
    eigvec.resize(n, nstates);
    std::vector<lapack_int> isuppz(2*n);
    
    // Workspace query
    std::complex<double> work_query;
    double rwork_query;
    lapack_int iwork_query;
    char V='V', I='I', U='U';
    
    zheevr_(&V, &I, &U, &n,
            Hptr->memptr(), &lda,
            &vl, &vu, &il, &iu, &abstol, &m_found,
            eigval.memptr(), eigvec.memptr(), &ldz,
            isuppz.data(),
            &work_query, &lwork, &rwork_query, &lrwork,
            &iwork_query, &liwork, &info);
    
    if(info != 0)
        throw std::runtime_error("zheevr workspace query failed with info="
        + std::to_string(info));
    
    lwork  = (lapack_int)work_query.real();
    lrwork = (lapack_int)rwork_query;
    liwork = iwork_query;
    
    arma::cx_vec     work(lwork);
    arma::vec        rwork(lrwork);
    std::vector<lapack_int> iwork(liwork);
    
    zheevr_(&V, &I, &U, &n,
            Hptr->memptr(), &lda,
            &vl, &vu, &il, &iu, &abstol, &m_found,
            eigval.memptr(), eigvec.memptr(), &ldz,
            isuppz.data(),
            work.memptr(),  &lwork,
            rwork.memptr(), &lrwork,
            iwork.data(),   &liwork, &info);
    
    if(info != 0)
        throw std::runtime_error("zheevr failed with info="
        + std::to_string(info));
    
    if(m_found < nstates)
        std::cerr << "Warning: zheevr found only " << m_found
        << " of " << nstates << " requested eigenvalues." << std::endl;
    eigval.resize(m_found);
    eigvec.resize(n, m_found);
}
                         


/**
 * Eigenpairs il..iu (1-based, ascending) of the Hermitian matrix H with LAPACK zheevr (RANGE = 'I').
 * H is destroyed. Used by the full BSE, whose wanted eigenvalues sit in the middle of the spectrum.
 */
void diagonalize_partial_range(arma::vec& eigval, arma::cx_mat& eigvec, arma::cx_mat& H, int il_in, int iu_in){

    lapack_int il = il_in, iu = iu_in;
    lapack_int n   = H.n_rows;
    lapack_int lda = n, ldz = n;
    lapack_int m_found = 0, info = 0;
    lapack_int nwant = iu - il + 1;
    double abstol = 0.0, vl = 0.0, vu = 0.0;
    lapack_int lwork = -1, lrwork = -1, liwork = -1;
    eigval.set_size(n);
    eigvec.set_size(n, nwant);
    std::vector<lapack_int> isuppz(2*nwant);
    std::complex<double> work_query;
    double rwork_query;
    lapack_int iwork_query;
    char V='V', I='I', U='U';
    zheevr_(&V, &I, &U, &n, H.memptr(), &lda, &vl, &vu, &il, &iu, &abstol, &m_found,
            eigval.memptr(), eigvec.memptr(), &ldz, isuppz.data(),
            &work_query, &lwork, &rwork_query, &lrwork, &iwork_query, &liwork, &info);
    if(info != 0)
        throw std::runtime_error("zheevr workspace query failed with info=" + std::to_string(info));
    lwork  = (lapack_int)work_query.real();
    lrwork = (lapack_int)rwork_query;
    liwork = iwork_query;
    arma::cx_vec     work(lwork);
    arma::vec        rwork(lrwork);
    std::vector<lapack_int> iwork(liwork);
    zheevr_(&V, &I, &U, &n, H.memptr(), &lda, &vl, &vu, &il, &iu, &abstol, &m_found,
            eigval.memptr(), eigvec.memptr(), &ldz, isuppz.data(),
            work.memptr(), &lwork, rwork.memptr(), &lrwork, iwork.data(), &liwork, &info);
    if(info != 0)
        throw std::runtime_error("zheevr failed with info=" + std::to_string(info));
    if(m_found < nwant)
        std::cerr << "Warning: zheevr found only " << m_found << " of " << nwant << " requested eigenvalues." << std::endl;
    eigval.resize(m_found);
    eigvec.resize(n, m_found);
}

#ifdef XATU_ILP64
/**
 * Stops unless the linked LAPACK really uses 64-bit integers (ILP64 build). A 32-bit library handed 64-bit
 * integers reads the low half of every scalar argument on a little-endian machine, so a wrong link can run and
 * return wrong results (integer arrays such as pivots are misread) instead of failing. The zheevd workspace query
 * for N = 40000 must report a real workspace of at least 1 + 5N + 2N^2 = 3200200001 > 2^31 - 1, which a 32-bit
 * LAPACK cannot represent.
 */
void check_lapack_ilp64(){
    static bool checked = false;
    if(checked) return;
    char V = 'V', U = 'U';
    lapack_int n = 40000, lwork = -1, lrwork = -1, liwork = -1, info = 0;
    std::complex<double> a[1], work[1];
    double w[1], rwork[1] = {0.0};
    lapack_int iwork[1] = {0};
    arma::lapack::heevd(&V, &U, &n, a, &n, w, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
    const double expected = 1.0 + 5.0*(double)n + 2.0*(double)n*(double)n;
    if(info != 0 || rwork[0] < expected - 0.5){
        throw std::runtime_error("Xatu was built with ILP64=1 but the linked LAPACK does not use 64-bit integers "
                                 "(zheevd workspace query for N = 40000 returned " + std::to_string(rwork[0]) +
                                 ", info = " + std::to_string(info) + "). Relink against an ILP64 LAPACK/BLAS.");
    }
    checked = true;
}
#endif

}