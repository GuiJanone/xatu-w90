#define ARMA_MAX_ELEM 0x200000000ULL
#define ARMA_64BIT_WORD
#include <armadillo>

namespace xatu {
    void davidson_method(arma::vec&, arma::cx_mat&, const arma::cx_mat&, int neigval = 4, double tol = 1E-8);
    
    void davidson_method_new(arma::vec&, arma::cx_mat&, const arma::cx_mat&, int neigval = 4, double tol = 1E-8);
    
    
    void diagonalize_partial(arma::vec&, arma::cx_mat&, arma::cx_mat&, int neigval = 4, bool preserve_H = false);
    
}
