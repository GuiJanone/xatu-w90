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

// // Direct LAPACK call to zheevr, computing only nstates lowest eigenvalues
// // This avoids both the Armadillo element limit and the full workspace allocation
// extern "C" void zheevr_(char*, char*, char*, int*, std::complex<double>*, 
//                         int*, double*, double*, int*, int*, double*, int*,
//                         double*, std::complex<double>*, int*, int64_t*, 
//                         std::complex<double>*, int*, double*, int*, int*, int*);
// 
// void diagonalize_partial(arma::vec& eigval, arma::cx_mat& eigvec, const arma::cx_mat& H, int nstates){
//     int n = H.n_rows;
//     int lda = n, ldz = n;
//     int il = 1, iu = nstates; // compute states 1 through nstates
//     int m_found;
//     double abstol = 1e-10, vl = 0, vu = 0;
//     int lwork = -1, lrwork = -1, liwork = -1, info;
//     
//     eigval.resize(nstates);
//     eigvec.resize(n, nstates);
//     arma::ivec isuppz(2*nstates);
//     
//     // workspace query
//     std::complex<double> work_query;
//     double rwork_query;
//     int iwork_query;
//     char V='V', I='I', U='U';
//     zheevr_(&V, &I, &U, &n, 
//             const_cast<std::complex<double>*>(H.memptr()), &lda,
//             &vl, &vu, &il, &iu, &abstol, &m_found,
//             eigval.memptr(), eigvec.memptr(), &ldz,
//             isuppz.memptr(),
//             &work_query, &lwork, &rwork_query, &lrwork, 
//             &iwork_query, &liwork, &info);
//     
//     lwork  = (int)work_query.real();
//     lrwork = (int)rwork_query;
//     liwork = iwork_query;
//     
//     arma::cx_vec work(lwork);
//     arma::vec rwork(lrwork);
//     arma::ivec iwork(liwork);
//     
//     zheevr_(&V, &I, &U, &n,
//             const_cast<std::complex<double>*>(H.memptr()), &lda,
//             &vl, &vu, &il, &iu, &abstol, &m_found,
//             eigval.memptr(), eigvec.memptr(), &ldz,
//             isuppz.memptr(),
//             work.memptr(), &lwork, rwork.memptr(), &lrwork,
//             iwork.memptr(), &liwork, &info);
//     
//     if(info != 0)
//         throw std::runtime_error("zheevr failed with info=" + std::to_string(info));
// }

}