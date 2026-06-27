#ifndef RESULTTB_HPP
#define RESULTTB_HPP

#pragma once
#define ARMA_MAX_ELEM 0x200000000ULL
#define ARMA_64BIT_WORD
#include <armadillo>
#include "xatu/SystemTB.hpp"
#include "xatu/Result.hpp"

extern "C" {
    void skubo_w_(int* nR, arma::uword* norb, arma::uword* norb_ex, int* nv, int* nc, int* filling, double* scissor,
                  double* Rvec, double* bravaisLattice, double* motif, 
                  std::complex<double>* hhop, double* shop, arma::uword* nk, double* rkx, 
                  double* rky, double* rkz, std::complex<double>* fk_ex, arma::uword* ldfk, double* e_ex, 
                  double* eigval_stack, std::complex<double>* eigvec_stack);

    void exciton_oscillator_strength_(int* nR, arma::uword* norb, arma::uword* norb_ex, int* nv, int* nc, int* filling, 
                  double* Rvec, double* bravaisLattice, double* motif, 
                  std::complex<double>* hhop, double* shop, arma::uword* nk, double* rkx, 
                  double* rky, double* rkz, std::complex<double>* fk_ex, arma::uword* ldfk, double* e_ex, 
                  double* eigval_stack, std::complex<double>* eigvec_stack, std::complex<double>* vme,
                  std::complex<double>* vme_ex, int* convert_to_au);
}

namespace xatu {

class ExcitonTB;

class ResultTB : public Result<SystemTB> {

    public:
        //// Constructor
        ResultTB(ExcitonTB* exciton_, arma::vec& eigval_, arma::cx_mat& eigvec_);
        ~ResultTB() = default;
        
        //// Observables
        // First recover hidden methods from Result<SystemTB>
        using Result<SystemTB>::spinX; // Add overload for spinX
        using Result<SystemTB>::writeRealspaceAmplitude;

        // Define additional methods
        arma::cx_vec spinX(const arma::cx_vec&);
        arma::cx_mat cachedSpinHole_;
        arma::cx_mat cachedSpinElectron_;
        std::vector<arma::cx_mat> spinHoleBlocks_;    // nk blocks of size npairs×npairs
        std::vector<arma::cx_mat> spinElectronBlocks_;
        bool spinMatricesInitialized_ = false;
        
        void initializeSpinMatrices();
        arma::mat velocity(int);
        arma::cx_vec velocitySingleParticle(int, int, int, std::string);
        arma::cx_mat excitonOscillatorStrength();
        double realSpaceWavefunction(const arma::cx_vec&, int, int,
                                     const arma::rowvec&, const arma::rowvec&);

        // Output and plotting
        void writeRealspaceAmplitude(const arma::cx_vec&, int, const arma::rowvec&, FILE*, int ncells = 3);
        void writeAbsorptionSpectrum();        
        
    private:
        arma::cx_vec addExponential(arma::cx_vec&, const arma::rowvec&);
};


}

#endif