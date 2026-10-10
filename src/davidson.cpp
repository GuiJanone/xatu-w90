#include "xatu/davidson.hpp"
#include "xatu/lapack_int.hpp"
#include <random>
#include <algorithm>

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


// ---------------------------------------------------------------------------------------------------------------
// Lowest states from the inverse of a positive definite matrix (TDA: H = L L^H; full BSE: K = L L^H, S^-1 = L^-1 s3 L^-H)
// ---------------------------------------------------------------------------------------------------------------
extern "C" {
    void zpotrf_(char* uplo, lapack_int* n, std::complex<double>* a, lapack_int* lda, lapack_int* info);
    void ztrsm_(char* side, char* uplo, char* transa, char* diag, lapack_int* m, lapack_int* n, std::complex<double>* alpha,
                std::complex<double>* a, lapack_int* lda, std::complex<double>* b, lapack_int* ldb);
    void ztrmm_(char* side, char* uplo, char* transa, char* diag, lapack_int* m, lapack_int* n, std::complex<double>* alpha,
                std::complex<double>* a, lapack_int* lda, std::complex<double>* b, lapack_int* ldb);
}

long long cholesky_lower_inplace(arma::cx_mat& A){
    char lo = 'L';
    lapack_int n = A.n_rows, info = 0;
    zpotrf_(&lo, &n, A.memptr(), &n, &info);
    return info;
}

void solve_lower(const arma::cx_mat& L, arma::cx_mat& X, bool conjtrans){
    char sd = 'L', lo = 'L', tr = conjtrans ? 'C' : 'N', dg = 'N';
    lapack_int n = L.n_rows, b = X.n_cols;
    std::complex<double> one(1.0, 0.0);
    ztrsm_(&sd, &lo, &tr, &dg, &n, &b, &one, const_cast<std::complex<double>*>(L.memptr()), &n, X.memptr(), &n);
}

void multiply_lower(const arma::cx_mat& L, arma::cx_mat& X){
    char sd = 'L', lo = 'L', tr = 'N', dg = 'N';
    lapack_int n = L.n_rows, b = X.n_cols;
    std::complex<double> one(1.0, 0.0);
    ztrmm_(&sd, &lo, &tr, &dg, &n, &b, &one, const_cast<std::complex<double>*>(L.memptr()), &n, X.memptr(), &n);
}

// Block size: the operator is two triangular solves with an n x n factor, whose cost is dominated by reading the
// factor once per block, so blocks are as wide as the subspace allows (measured: 64 -> 256 halves the time).
long long krylov_block_size(long long n){
    return std::max(16LL, std::min(256LL, n/16));
}

// Worth it (and safe) only when the subspace is a small part of the space; otherwise dense LAPACK is used.
bool krylov_applicable(long long n, long long nwanted){
    long long b = krylov_block_size(n);
    return nwanted > 0 && 4*(nwanted + 4*b) <= n;
}

/**
 * Extremal eigenpairs of a Hermitian operator A, given only X <- A X on blocks: the nlow algebraically smallest and
 * the nhigh largest eigenvalues (ascending in mu) with orthonormal eigenvectors X.
 * @details Thick-restart block Krylov (Rayleigh-Ritz on a basis expanded with the residuals of the unconverged
 * wanted Ritz pairs, which lie in the next Krylov block). Converged when ||A x - mu x|| <= tol |mu| for every wanted
 * pair; the eigenvalue error is then O(tol^2 |mu|). Used with A = H^-1 (TDA, lowest states = largest mu) and
 * A = S^-1 = L^-1 s3 L^-H (full BSE, the modes closest to zero frequency = both ends of the spectrum of S^-1), where
 * A X costs two triangular solves with a Cholesky factor that is already available, so the tridiagonal reduction of
 * a dense eigensolver (memory bound, poor thread scaling) is avoided.
 */
bool block_krylov_extremal(const std::function<void(arma::cx_mat&)>& op, long long n, long long nlow, long long nhigh,
                           arma::vec& mu, arma::cx_mat& X, double tol, KrylovReport& report){
    const long long b = krylov_block_size(n);
    const long long kl = nlow > 0 ? nlow + b : 0, kh = nhigh > 0 ? nhigh + b : 0;
    const long long keep = std::min(n, kl + kh), Dmax = std::min(n, keep + 2*b);
    const long long nw = nlow + nhigh;
    const int maxit = 300, patience = 40;   // give up after 40 iterations without halving the worst residual
    double best = 1E300;
    int lastGain = 0;
    report = KrylovReport();
    report.block = b;

    // Deterministic random start (a local generator: Armadillo's global RNG is left alone)
    std::mt19937_64 gen(20261010);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::cx_mat V(n, keep), Rq;
    for (arma::uword j = 0; j < V.n_cols; j++)
        for (arma::uword i = 0; i < V.n_rows; i++) V(i, j) = std::complex<double>(nd(gen), nd(gen));
    arma::qr_econ(V, Rq, arma::cx_mat(V));
    arma::cx_mat AV = V;
    op(AV);
    report.applications = keep;

    for (int it = 1; it <= maxit; it++){
        report.iterations = it;
        arma::cx_mat H = V.t()*AV;
        H = 0.5*(H + H.t());
        arma::vec th;
        arma::cx_mat Y;
        arma::eig_sym(th, Y, H);
        const long long D = V.n_cols;
        arma::uvec w(nw);
        for (long long a = 0; a < nlow; a++) w(a) = a;
        for (long long a = 0; a < nhigh; a++) w(nlow + a) = D - nhigh + a;
        arma::cx_mat Xw = V*Y.cols(w);
        arma::cx_mat R = AV*Y.cols(w);
        for (long long c = 0; c < nw; c++) R.col(c) -= th(w(c))*Xw.col(c);
        std::vector<std::pair<double, arma::uword>> unc;
        double worst = 0.0;
        for (long long c = 0; c < nw; c++){
            double rel = arma::norm(R.col(c))/std::max(std::abs(th(w(c))), 1E-300);
            worst = std::max(worst, rel);
            if (rel > tol) unc.push_back({rel, (arma::uword)c});
        }
        report.residual = worst;
        report.basis = D;
        if (unc.empty()){
            mu = th(w);
            X = std::move(Xw);
            return true;
        }
        if (worst < 0.5*best){ best = worst; lastGain = it; }
        if (it - lastGain > patience) break;   // stagnation (e.g. an ill-conditioned operator)
        if (D + b > Dmax){
            // Thick restart: keep the kl lowest and kh highest Ritz vectors
            arma::uvec kk(keep);
            for (long long a = 0; a < kl; a++) kk(a) = a;
            for (long long a = 0; a < kh; a++) kk(kl + a) = D - kh + a;
            V = V*Y.cols(kk);
            AV = AV*Y.cols(kk);
        }
        // Expand with the residuals of the least converged wanted pairs
        std::sort(unc.rbegin(), unc.rend());
        long long nb = std::min<long long>(b, unc.size());
        arma::uvec pick(nb);
        for (long long a = 0; a < nb; a++) pick(a) = unc[a].second;
        arma::cx_mat Wn = R.cols(pick);
        for (int pass = 0; pass < 2; pass++) Wn -= V*(V.t()*Wn);
        arma::cx_mat Q;
        arma::qr_econ(Q, Rq, Wn);
        arma::vec rd = arma::abs(Rq.diag());
        arma::uvec good = arma::find(rd > 1E-10*std::max(1E-300, rd.max()));
        if (good.n_elem == 0 || (long long)V.n_cols + (long long)good.n_elem > n) break;
        Q = Q.cols(good);
        arma::cx_mat AQ = Q;
        op(AQ);
        report.applications += Q.n_cols;
        V = arma::join_rows(V, Q);
        AV = arma::join_rows(AV, AQ);
    }
    return false;
}

}
