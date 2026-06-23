#include "xatu/davidson.hpp"

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

// Direct LAPACK call to zheevr, computing only nstates lowest eigenvalues
// This avoids both the Armadillo element limit and the full workspace allocation
extern "C" void zheevr_(char*,               // JOBZ
                        char*,               // RANGE  
                        char*,               // UPLO
                        int*,                // N
                        std::complex<double>*, // A
                        int*,                // LDA
                        double*,             // VL
                        double*,             // VU
                        int*,                // IL
                        int*,                // IU
                        double*,             // ABSTOL
                        int*,                // M
                        double*,             // W
                        std::complex<double>*, // Z
                        int*,                // LDZ
                        int*,                // ISUPPZ
                        std::complex<double>*, // WORK
                        int*,                // LWORK
                        double*,             // RWORK
                        int*,                // LRWORK
                        int*,                // IWORK
                        int*,                // LIWORK
                        int*);               // INFO

void diagonalize_partial(arma::vec& eigval, arma::cx_mat& eigvec, 
                         const arma::cx_mat& H, int nstates){
    int n = H.n_rows;
    arma::cx_mat Hcopy = H;  // zheevr overwrites input
    int lda = n, ldz = n;
    int il = 1, iu = nstates;
    int m_found;
    double abstol = 0.0, vl = 0.0, vu = 0.0;
    int lwork = -1, lrwork = -1, liwork = -1, info;
    
    eigval.resize(nstates);
    eigvec.resize(n, nstates);
    std::vector<int> isuppz(2*n);  // safe upper bound
    
    // workspace query
    std::complex<double> work_query;
    double rwork_query;
    int iwork_query;
    char V='V', I='I', U='U';
    
    zheevr_(&V, &I, &U, &n,
            Hcopy.memptr(), &lda,
            &vl, &vu, &il, &iu, &abstol, &m_found,
            eigval.memptr(), eigvec.memptr(), &ldz,
            isuppz.data(),
            &work_query, &lwork, &rwork_query, &lrwork,
            &iwork_query, &liwork, &info);
    
    if(info != 0)
        throw std::runtime_error("zheevr workspace query failed with info=" 
        + std::to_string(info));
    
    lwork  = (int)work_query.real();
    lrwork = (int)rwork_query;
    liwork = iwork_query;
    
    arma::cx_vec work(lwork);
    arma::vec rwork(lrwork);
    std::vector<int> iwork(liwork);
    
    zheevr_(&V, &I, &U, &n,
            Hcopy.memptr(), &lda,
            &vl, &vu, &il, &iu, &abstol, &m_found,
            eigval.memptr(), eigvec.memptr(), &ldz,
            isuppz.data(),
            work.memptr(), &lwork, rwork.memptr(), &lrwork,
            iwork.data(), &liwork, &info);
    
    if(info != 0)
        throw std::runtime_error("zheevr failed with info=" 
        + std::to_string(info));
    
    if(m_found < nstates)
        std::cerr << "Warning: zheevr found only " << m_found 
        << " of " << nstates << " requested eigenvalues." << std::endl;
}

}