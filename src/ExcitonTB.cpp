#include <math.h>
#include <sys/resource.h>
#include <iomanip>
#include "xatu/ExcitonTB.hpp"
#include "xatu/utils.hpp"
#include "xatu/davidson.hpp"

using namespace arma;
using namespace std::chrono;

namespace xatu {

/**
 * Method to set the values of the attributes of an exciton object.
 * @param ncell Number of unit cells per axis.
 * @param bands Vector with the indices of the bands that form the exciton.
 * @param parameters Dielectric constants and screening length.
 * @param Q Center-of-mass momentum of the exciton.
 * @return void 
 */
void ExcitonTB::initializeExcitonAttributes(int ncell, const arma::ivec& bands, 
                                      const arma::rowvec& parameters, const arma::rowvec& Q){
    this->ncell_      = ncell;
    this->totalCells_ = pow(ncell, system_->ndim);
    this->Q_          = Q;
    this->bands_      = bands;
    // if there are <= than 3 elements, it's hubbard
    if (parameters.n_elem  >= 1) {
        this->hubbardU_    = parameters(0);
        this->hubbardU1_   = 0.0;
        this->hubbarddist_ = 0.0;
    }
    if (parameters.n_elem  >= 2) {
        this->hubbardU1_   = parameters(0);
        this->hubbarddist_ = parameters(1);
    }
    if (parameters.n_elem  >= 3) {
        this->hubbardU1_   = parameters(1);
        this->hubbarddist_ = parameters(2);
    }
    // if there are 5 elements, it's keldysh'
    if (parameters.n_elem  >= 5) {
        this->eps_m_       = parameters(0);
        this->eps_s_       = parameters(1);
        this->r0_          = parameters(2);
        this->ry_          = parameters(3);
        this->rz_          = parameters(4);
    } 
    // if there are more than 5 elements, it's both (not tested)
    if (parameters.n_elem  >= 6) {
        this->hubbardU_    = parameters(5);
        this->hubbardU1_   = parameters(5);
        this->hubbarddist_ = 0.0; 
    }
    if (parameters.n_elem  >= 7) {
        this->hubbardU1_   = parameters(5);
        this->hubbarddist_ = parameters(6); 
    }
    if (parameters.n_elem  >= 8) {
        this->hubbardU1_   = parameters(6);
        this->hubbarddist_ = parameters(7); 
    }
    this->cutoff_     = ncell/2.5; // Default value, seems to preserve crystal point group

    if((parameters.n_elem > 3) && (r0 == 0)){
        throw std::invalid_argument("Error: r0 must be non-zero");
    }

}

/**
 * Method to set the attributes of an exciton object from a ExcitonConfiguration object.
 * @details Overload of the method to use a configuration object. Based on the parametric method.
 * @param cfg ExcitonConfiguration object from parsed file.
 * @return void
 */
void ExcitonTB::initializeExcitonAttributes(const ExcitonConfiguration& cfg){

    uint64_t ncell        = cfg.excitonInfo.ncell;
    uint64_t nbands       = cfg.excitonInfo.nbands;
    arma::ivec bands = cfg.excitonInfo.bands;
    arma::rowvec parameters = arma::conv_to<arma::rowvec>::from(join_cols(cfg.excitonInfo.eps, cfg.excitonInfo.hubbardU));
    arma::rowvec Q   = cfg.excitonInfo.Q;

    if (bands.empty()){
        bands = arma::regspace<arma::ivec>(- nbands + 1, nbands);
    }

    initializeExcitonAttributes(ncell, bands, parameters, Q);

    std::vector<arma::s64> valence, conduction;
    for(arma::uword i = 0; i < bands.n_elem; i++){
        if (bands(i) <= 0){
            valence.push_back(bands(i) + system->fermiLevel);
        }
        else{
            conduction.push_back(bands(i) + system->fermiLevel);
        }
    }
    this->valenceBands_ = arma::ivec(valence);
    this->conductionBands_ = arma::ivec(conduction);
    this->bandList_ = arma::conv_to<arma::uvec>::from(arma::join_cols(valenceBands, conductionBands));

    // Set flags
    this->exchange = cfg.excitonInfo.exchange;
    this->selfenergy = cfg.excitonInfo.selfenergy;
    this->scissor_ = cfg.excitonInfo.scissor;
    this->mode_    = cfg.excitonInfo.mode;
    this->nReciprocalVectors_ = cfg.excitonInfo.nReciprocalVectors;
    this->regularization_ = cfg.excitonInfo.regularization;
    if (regularization_ == 0){
        regularization_ = system->a;
    }
    this->potential_ = cfg.excitonInfo.potential;
    this->exchangePotential_ = cfg.excitonInfo.exchangePotential;
    this->selfenergyPotential_ = cfg.excitonInfo.selfenergyPotential;
    
    this->tammdancoff_ = cfg.excitonInfo.tammdancoff;
    
    this->bandTracking_ = cfg.excitonInfo.bandTracking;
    this->bandTrackingThreshold_ = cfg.excitonInfo.bandTrackingThreshold;
}

/**
 * Exciton constructor from a SystemConfiguration object and a vector with the bands that form
 * the exciton, as well as the other parameters.
 * @param config SystemConfiguration object from config file.
 * @param ncell Number of unit cells along each axis.
 * @param bands Vector with the indices of the bands that form the exciton.
 * @param parameters Vector with dielectric constants and screening length.
 * @param Q Center-of-mass momentum.
 */
ExcitonTB::ExcitonTB(const SystemConfiguration& config, int ncell, const arma::ivec& bands, 
                     const arma::rowvec& parameters, const arma::rowvec& Q) {

    system_.reset(new SystemTB(config));

    initializeExcitonAttributes(ncell, bands, parameters, Q);

    if (bands.n_elem > excitonbasisdim){
        cout << "Error: Number of bands cannot be higher than actual material bands" << endl;
        exit(1);
    }

    // arma::ivec is implemented with typedef s64
    std::vector<arma::s64> valence, conduction;
    for(arma::uword i = 0; i < bands.n_elem; i++){
        if (bands(i) <= 0){
            valence.push_back(bands(i) + system->fermiLevel);
        }
        else{
            conduction.push_back(bands(i) + system->fermiLevel);
        }
    }
    this->valenceBands_ = arma::ivec(valence);
    this->conductionBands_ = arma::ivec(conduction);
    this->bandList_ = arma::conv_to<arma::uvec>::from(arma::join_cols(valenceBands, conductionBands));
};

/**
 * Exciton constructor from a SystemConfiguration object. One specifies the number of valence and conduction
 * bands, as well as the other parameters.
 * @param config SystemConfiguration object from config file.
 * @param ncell Number of unit cells along each axis.
 * @param nbands Number of bands (same number for both valence and conduction) that form the exciton.
 * @param nrmbands Number of bands to be removed with respect to the Fermi level. 
 * @param parameters Vector with dielectric constants and screening length.
 * @param Q Center-of-mass momentum.
 */
ExcitonTB::ExcitonTB(const SystemConfiguration& config, int ncell, int nbands, int nrmbands, 
                     const arma::rowvec& parameters, const arma::rowvec& Q) : 
                     ExcitonTB(config, ncell, {}, parameters, Q){
    
    if (2*nbands > system->basisdim){
        cout << "Error: Number of bands cannot be higher than actual material bands" << endl;
        exit(1);
    }

    int fermiLevel = system->fermiLevel;
    this->valenceBands_ = arma::regspace<arma::ivec>(fermiLevel - nbands - nrmbands + 1, 
                                                     fermiLevel - nrmbands);
    this->conductionBands_ = arma::regspace<arma::ivec>(fermiLevel + 1 + nrmbands, 
                                                        fermiLevel + nbands + nrmbands);
    this->bands_ = arma::join_cols(valenceBands, conductionBands) - fermiLevel;
    this->bandList_ = arma::conv_to<arma::uvec>::from(arma::join_cols(valenceBands, conductionBands));
};


/**
 * Exciton constructor from SystemConfiguration and ExcitonConfiguration.
 * @param config SystemConfiguration object.
 * @param excitonConfig ExcitonConfiguration object.
 */ 
ExcitonTB::ExcitonTB(const SystemConfiguration& config, const ExcitonConfiguration& excitonConfig){

    system_.reset(new SystemTB(config));
    initializeExcitonAttributes(excitonConfig);
}

ExcitonTB::ExcitonTB(std::shared_ptr<SystemTB> sys, int ncell, const arma::ivec& bands, 
                     const arma::rowvec& parameters, const arma::rowvec& Q){

    
    system_ = sys;
    initializeExcitonAttributes(ncell, bands, parameters, Q);

    if (bands.n_elem > system->basisdim){
        cout << "Error: Number of bands cannot be higher than actual material bands" << endl;
        exit(1);
    }

    // arma::ivec is implemented with typedef s64
    std::vector<arma::s64> valence, conduction;
    for(int i = 0; i < bands.n_elem; i++){
        if (bands(i) <= 0){
            valence.push_back(bands(i) + system->fermiLevel);
        }
        else{
            conduction.push_back(bands(i) + system->fermiLevel);
        }
    }
    this->valenceBands_ = arma::ivec(valence);
    this->conductionBands_ = arma::ivec(conduction);
    this->bandList_ = arma::conv_to<arma::uvec>::from(arma::join_cols(valenceBands, conductionBands));
};

/**
 * Exciton constructor from an already initialized System object, and all the exciton parameters.
 * @param system System object where excitons are computed.
 * @param ncell Number of unit cells along each axis.
 * @param nbands Number of bands (same for valence and conduction) that form the exciton.
 * @param nrmbands Number of bands to be removed with respect to the Fermi level. 
 * @param parameters Dielectric constant and screening length.
 * @param Q Center-of-mass momentum of the exciton.
 */
ExcitonTB::ExcitonTB(std::shared_ptr<SystemTB> sys, int ncell, int nbands, int nrmbands, 
                     const arma::rowvec& parameters, const arma::rowvec& Q) : 
                     ExcitonTB(sys, ncell, {}, parameters, Q) {
    
    if (2*nbands > system->basisdim){
        cout << "Error: Number of bands cannot be higher than actual material bands" << endl;
        exit(1);
    }

    int fermiLevel = system->fermiLevel;
    this->valenceBands_ = arma::regspace<arma::ivec>(fermiLevel - nbands - nrmbands + 1, 
                                                     fermiLevel - nrmbands);
    this->conductionBands_ = arma::regspace<arma::ivec>(fermiLevel + 1 + nrmbands, 
                                                        fermiLevel + nbands + nrmbands);
    this->bands_ = arma::join_cols(valenceBands, conductionBands) - fermiLevel;
    this->bandList_ = arma::conv_to<arma::uvec>::from(arma::join_cols(valenceBands, conductionBands));
};

/** 
 * Exciton destructor.
 * @details Used mainly for debugging; the message should be removed at some point.
 */
// ExcitonTB::~ExcitonTB(){};


/* ------------------------------ Setters ------------------------------ */

/**
 * Method to set the parameters of the Keldysh potential, namely the environmental
 * dielectric constants and the effective screening lengths.
 * @param parameters Vector with 3 to 5 parameters: '{eps_m, eps_s, r0[, ry[, rz]]}'.
 * @return void
 */
void ExcitonTB::setParameters(const arma::rowvec& parameters) {
    if (parameters.n_elem < 3) {
        std::cout << "parameters array must be at least 3D (eps_m, eps_s, r0)" << std::endl;
        return;
    }

    eps_m_ = parameters(0);
    eps_s_ = parameters(1);
    r0_    = parameters(2);

    // Set ry
    if (parameters.n_elem >= 4) {
        ry_ = parameters(3);
    } else {
        ry_ = r0_;
    }

    // Set rz
    if (parameters.n_elem >= 5) {
        rz_ = parameters(4);
    } else {
        rz_ = 0.5 * (r0_ + ry_);
    }
}


/**
 * Sets the parameters of the Keldysh potential.
 * @param eps_m Dielectric constant of embedding medium.
 * @param eps_s Dielectric constant of substrate.
 * @param r0 Effective screeening length.
 * @return void 
 */
void ExcitonTB::setParameters(double eps_m, double eps_s, double r0, double ry, double rz){
    eps_m_ = eps_m;
    eps_s_ = eps_s;
    r0_    = r0;

    // Set ry
    if (ry != 0) {
        ry_ = ry;
    } else {
        ry_ = r0;
    }

    // Set rz
    if (rz != 0) {
        rz_ = rz;
    } else {
        rz_ = 0.5 * (r0 + ry);
    }
}

/**
 * Sets the gauge used for the Bloch basis, either 'lattice' or 'atomic'.
 * @param gauge Gause to be used, default to 'lattice'.
 * @return void
 */
void ExcitonTB::setGauge(std::string gauge){
    if(gauge != "lattice" && gauge != "atomic"){
        throw std::invalid_argument("setGauge(): gauge must be either lattice or atomic");
    }
    this->gauge_ = gauge;
}

/**
 * Sets the type of calculation used to obtain the exciton spectrum. It can be 'realspace' (default) 
 * or 'reciprocalspace'.
 * @param mode Calculation model.
 * @return void 
 */
void ExcitonTB::setMode(std::string mode){
    if(mode != "realspace" && mode != "reciprocalspace"){
        throw std::invalid_argument("setMode(): mode must be either realspace or reciprocalspace");
    }
    this->mode_ = mode;
}

/**
 * Sets the number of reciprocal vectors to use if the exciton calculation is set to 'reciprocalspace'.
 * @param nReciprocalVector Number of reciprocal vectors to sum over.
 * @return void 
 */
void ExcitonTB::setReciprocalVectors(int nReciprocalVectors){
    if(nReciprocalVectors < 0){
        throw std::invalid_argument("setReciprocalVectors(): given number must be positive");
    }
    this->nReciprocalVectors_ = nReciprocalVectors;
}

/**
 * Sets the regularization distance for the Coulomb potential divergence at r=0.
 * @param regularization Distance in Angstroms.
 * @return void
*/
void ExcitonTB::setRegularization(double regularization){
    this->regularization_ = regularization;
}

/**
 * Sets the bandtracking flag.
 * @bool bandTracking.
 * @return void
 */
void ExcitonTB::setBandTracking(bool bandTracking){
    bandTracking_ = bandTracking;
}
void ExcitonTB::setBandTrackingThreshold(double threshold){
    bandTrackingThreshold_ = threshold;
}
/*---------------------------------------- Potentials ----------------------------------------*/

/** 
 * Purpose: Compute Struve function H0(x).
 * Source: http://jean-pierre.moreau.pagesperso-orange.fr/Cplus/mstvh0_cpp.txt 
 * @param X x --- Argument of H0(x) ( x ò 0 )
 * @param SH0 SH0 --- H0(x). The return value is written to the direction of the pointer.
*/
void ExcitonTB::STVH0(double X, double *SH0) {
    double A0,BY0,P0,Q0,R,S,T,T2,TA0;
	int K, KM;

        S=1.0;
        R=1.0;
        if (X <= 20.0) {
           A0=2.0*X/PI;
           for (K=1; K<61; K++) {
              R=-R*X/(2.0*K+1.0)*X/(2.0*K+1.0);
              S=S+R;
              if (fabs(R) < fabs(S)*1.0e-12) goto e15;
           }
    e15:       *SH0=A0*S;
        }
        else {
           KM=int(0.5*(X+1.0));
           if (X >= 50.0) KM=25;
           for (K=1; K<=KM; K++) {
              R=-R*pow((2.0*K-1.0)/X,2);
              S=S+R;
              if (fabs(R) < fabs(S)*1.0e-12) goto e25;
           }
    e25:       T=4.0/X;
           T2=T*T;
           P0=((((-.37043e-5*T2+.173565e-4)*T2-.487613e-4)*T2+.17343e-3)*T2-0.1753062e-2)*T2+.3989422793;
           Q0=T*(((((.32312e-5*T2-0.142078e-4)*T2+0.342468e-4)*T2-0.869791e-4)*T2+0.4564324e-3)*T2-0.0124669441);
           TA0=X-0.25*PI;
           BY0=2.0/sqrt(X)*(P0*sin(TA0)+Q0*cos(TA0));
           *SH0=2.0/(PI*X)*S+BY0;
        }
}


/** 
 * Calculate value of interaction potential (Keldysh). Units are eV.
 * @details If the distance is zero, then the interaction is renormalized to be V(a) since
 * V(0) is infinite, where a is the lattice parameter. Also, for r > cutoff the interaction is taken to be zero.
 * @param r Distance at which we evaluate the potential.
 * @return Value of Keldysh potential, V(r).
 */
double ExcitonTB::keldysh(arma::rowvec r){
    double eps_bar = (eps_m + eps_s)/2;
    double SH0;
    double cutoff = arma::norm(system->bravaisLattice.row(0)) * cutoff_ + 1E-5;
    arma::rowvec R0 = {r0,ry,rz};
    double R = arma::norm(r/R0);
    double r_norm = arma::norm(r);
    double r0avg = (r0 + ry + rz)/3;
    double potential_value;
    if(r_norm < 1E-10){
        STVH0(regularization/r0, &SH0);
        potential_value = ec/(8E-10*eps0*eps_bar*r0avg)*(SH0 - y0(regularization/r0avg));
    }
    else if (r_norm > cutoff){
        potential_value = 0.0;
    }
    else{
        STVH0(R, &SH0);
        potential_value = ec/(8E-10*eps0*eps_bar*r0avg)*(SH0 - y0(R));
    };

    return potential_value;
};

/**
 * Coulomb potential in real space.
 * @param r Distance at which we evaluate the potential.
 * @param regularization Regularization distance to remove divergence at r=0.
 * @return Value of Coulomb potential, V(r).
 */
double ExcitonTB::coulomb(arma::rowvec r){
    double cutoff = arma::norm(system->bravaisLattice.row(0)) * cutoff_ + 1E-5;
    double R = abs(arma::norm(r));
/*    if (R > cutoff){
        return 0.0;
    }
    return (R != 0) ? ec/(4E-10*PI*eps0*R) : ec*1E10/(4*PI*eps0*regularization);   */ 
    double eps_bar = (eps_m + eps_s)/2;
    double potential_value;
    if(R < 1E-5){
        potential_value =ec/(4E-10*PI*eps0*eps_bar*regularization);
    }
    else if (R > 2.5*cutoff){
        potential_value = 0.0;
    }
    else{
        potential_value = ec*1E10/(4*PI*eps0*eps_bar*R);   
    };

    return potential_value;
}

/**
 * Hubbard potential in real space.
 * @param r Distance at which we evaluate the potential.
 * @param strength Hubbard U parameter.
 * @return Value of Coulomb potential, V(r).
 */


double ExcitonTB::hubbard(arma::rowvec r){
    double R = abs(arma::norm(r));
    if (R > hubbarddist){
        return 0.0;
    }
    if ((R > 0.0) && (R <= hubbarddist)){
        return hubbardU1;
    }
    return hubbardU;
}

/**
 * Method to select the potential to be used in the of the exciton calculation.
 * @param potential Potential to be used in the direct term.
 * @return Pointer to function representing the potential.
 */

potptr ExcitonTB::selectPotential(std::string potential){
    if(potential == "keldysh"){
        return &ExcitonTB::keldysh;
    }
    else if(potential == "coulomb"){
        return &ExcitonTB::coulomb;
    }
    else if(potential == "hubbard"){
        return &ExcitonTB::hubbard;
    }
    else{
        throw std::invalid_argument("selectPotential(): potential must be either 'keldysh', 'coulomb' or 'hubbard'");
    }
}



/*---------------------------------------- Fourier transforms ----------------------------------------*/

/**
 * Evaluates the Fourier transform of the Keldysh potential, which is an analytical expression.
 * @param q kpoint where we evaluate the FT.
 * @return Fourier transform of the potential at q, FT[V](q).
 */
double ExcitonTB::keldyshFT(arma::rowvec q){
    double radius = cutoff*arma::norm(system->reciprocalLattice.row(0));
    double potential = 0;
    double eps_bar = (eps_m + eps_s)/2;
    double eps = arma::norm(system->reciprocalLattice.row(0))/totalCells;

    double qnorm = arma::norm(q);
    if (qnorm < eps){
        potential = 0;
    }
    else{
        potential = 1/(qnorm*(1 + r0*qnorm));
    }
    
    potential = potential*ec*1E10/(2*eps0*eps_bar*system->unitCellArea*totalCells);
    return potential;
}

/**
 * Routine to compute the lattice Fourier transform with the potential displaced by some
 * vectors of the motif.
 * @param fAtomIndex Index of first atom of the motif.
 * @param sAtomIndex Index of second atom of the motif.
 * @param k kpoint where we evaluate the FT.
 * @param cells Matrix with the unit cells over which we sum to compute the lattice FT.
 * @param potential Pointer to potential function.
 * @return Motif lattice Fourier transform of the Keldysh potential at k.
 */
std::complex<double> ExcitonTB::motifFourierTransform(int fAtomIndex, int sAtomIndex, const arma::rowvec& k, 
                                                      const arma::mat& cells, potptr potential){

    std::complex<double> imag(0,1);
    std::complex<double> Vk = 0.0;
    arma::rowvec firstAtom = system->motif.row(fAtomIndex).subvec(0, 2);
    arma::rowvec secondAtom = system->motif.row(sAtomIndex).subvec(0, 2);

    for(int n = 0; n < cells.n_rows; n++){
        arma::rowvec cell = cells.row(n);
        // double module = arma::norm(cell + firstAtom - secondAtom);
        arma::rowvec dist = cell + firstAtom - secondAtom;
        Vk += (this->*potential)(dist)*std::exp(imag*arma::dot(k, cell));
    }
    Vk /= pow(totalCells, 1);

    return Vk;
}

/**
 * Method to compute the motif FT matrix at a given k vector.
 * @param k k vector where we compute the motif FT.
 * @param cells Matrix of unit cells over which the motif FT is computed.
 * @param potential Pointer to potential function.
 * @return void
 */
arma::cx_mat ExcitonTB::motifFTMatrix(const arma::rowvec& k, const arma::mat& cells, potptr potential){
    // Uses hermiticity of V
    int natoms = system->natoms;
    arma::cx_mat motifFT = arma::zeros<arma::cx_mat>(natoms, natoms);

    for(int fAtomIndex = 0; fAtomIndex < natoms; fAtomIndex++){
        for(int sAtomIndex = fAtomIndex; sAtomIndex < natoms; sAtomIndex++){
            motifFT(fAtomIndex, sAtomIndex) = motifFourierTransform(fAtomIndex, sAtomIndex, k, cells, potential);
            motifFT(sAtomIndex, fAtomIndex) = conj(motifFT(fAtomIndex, sAtomIndex));
        }   
    }

    return motifFT;
}

/**
 * Method to extend the motif Fourier transform matrix to match the dimension of the
 * one-particle basis. 
 * @param motifFT Matrix storing the motif Fourier transform to be extended.
 * @return Extended matrix.
 */
arma::cx_mat ExcitonTB::extendMotifFT(const arma::cx_mat& motifFT){
    arma::cx_mat extendedMFT = arma::zeros<arma::cx_mat>(system->basisdim, system->basisdim);
    int rowIterator = 0;
    int colIterator = 0;
    for(unsigned int atom_index_r = 0; atom_index_r < system->motif.n_rows; atom_index_r++){
        int species_r = system->motif.row(atom_index_r)(3);
        int norbitals_r = system->orbitals(species_r);
        colIterator = 0;
        for(unsigned int atom_index_c = 0; atom_index_c < system->motif.n_rows; atom_index_c++){
            int species_c = system->motif.row(atom_index_c)(3);
            int norbitals_c = system->orbitals(species_c);
            extendedMFT.submat(rowIterator, colIterator, 
                               rowIterator + norbitals_r - 1, colIterator + norbitals_c - 1) = 
                          motifFT(atom_index_r, atom_index_c) * arma::ones(norbitals_r, norbitals_c);
            colIterator += norbitals_c;
        }
        rowIterator += norbitals_r;
    }

    return extendedMFT;
}


/*------------------------------------ Interaction matrix elements ------------------------------------*/

/** 
 * Real space implementation of interaction term, valid for both direct and exchange.
 * To compute the direct term, the expected order is (ck,v'k',c'k',vk).
 * For the exchange term, the order is (ck,v'k',vk,c'k').
 * @param coefsK1 First eigenstate vector.
 * @param coefsK2 Second eigenstate vector.
 * @param coefsK3 Third eigenstate vector.
 * @param coefsK4 Fourth eigenstate vector.
 * @param motifFT Motif Fourier transform.
 * @return Interaction term.
 */
std::complex<double> ExcitonTB::realSpaceInteractionTerm(const arma::cx_vec& coefsK1, 
                                     const arma::cx_vec& coefsK2,
                                     const arma::cx_vec& coefsK3, 
                                     const arma::cx_vec& coefsK4,
                                     const arma::cx_mat& motifFT){
    
    arma::cx_vec firstCoefArray =  arma::conj(coefsK1) % coefsK3;
    arma::cx_vec secondCoefArray = arma::conj(coefsK2) % coefsK4;
    /* Old implementation; one below should be faster */
    // std::complex<double> term = arma::dot(firstCoefArray, extendMotifFT(motifFT) * secondCoefArray);

    /* Instead of extending the motifFT matrix, reduce the coefs vectors */
    arma::cx_vec reducedFirstCoefArray = arma::zeros<arma::cx_vec>(system->natoms);
    arma::cx_vec reducedSecondCoefArray = arma::zeros<arma::cx_vec>(system->natoms);

    int iterator = 0;
    for(unsigned int atom_index = 0; atom_index < system->motif.n_rows; atom_index++){
        int norbitals = system->orbitals(system->motif.row(atom_index)(3));

        reducedFirstCoefArray(atom_index) = arma::sum(firstCoefArray.subvec(iterator, iterator + norbitals - 1));
        reducedSecondCoefArray(atom_index) = arma::sum(secondCoefArray.subvec(iterator, iterator + norbitals - 1));

        iterator += norbitals;
    }

    std::complex<double> term = arma::dot(reducedFirstCoefArray, motifFT * reducedSecondCoefArray);

    return term;
};

/**
 * Reciprocal space implementation of interaction term, valid for both direct and exchange.
 * @param coefsK Vector of eigenstate |v,k>.
 * @param coefsK2 Vector of eigenstate |v',k'>.
 * @param coefsKQ Vector of eigenstate |c,k+Q>.
 * @param coefsK2Q Vector of eigenstate |c',k'+Q>.
 * @param k kpoint corresponding to k.
 * @param k2 kpoint corresponding to k'.
 * @param kQ kpoint corresponding to k + Q.
 * @param k2Q kpoint corresponding to k' + Q.
 * @return Interaction term.
 */
std::complex<double> ExcitonTB::reciprocalInteractionTerm(const arma::cx_vec& coefsK, 
                                     const arma::cx_vec& coefsK2,
                                     const arma::cx_vec& coefsKQ, 
                                     const arma::cx_vec& coefsK2Q,
                                     const arma::rowvec& k, 
                                     const arma::rowvec& k2,
                                     const arma::rowvec& kQ, 
                                     const arma::rowvec& k2Q,
                                     int nrcells){
    
    std::complex<double> Ic, Iv;
    std::complex<double> term = 0;
    double radius = cutoff * arma::norm(system->reciprocalLattice.row(0));
    arma::mat reciprocalVectors = system_->truncateReciprocalSupercell(nrcells, radius);

    for(int i = 0; i < reciprocalVectors.n_rows; i++){
        auto G = reciprocalVectors.row(i);

        Ic = blochCoherenceFactor(coefsKQ, coefsK2Q, kQ, k2Q, G);
        Iv = blochCoherenceFactor(coefsK, coefsK2, k, k2, G);

        // must change this to include Q2DRK...
        term += Ic*conj(Iv)*keldyshFT(k - k2 + G);
    }

    return term;
};

/**
 * Calculation of Bloch coherence factors, required to compute the interaction terms in reciprocal space.
 * @param coefs1 Vector of eigenstate |n,k>.
 * @param coefs2 Vector of eigenstate |n',k'>.
 * @param k1 kpoint k.
 * @param k2 kpoint k'.
 * @param G Reciprocal lattice vector used to compute the coherence factor.
 * @return Bloch coherence factor I evaluated at G for states |nk>, |n'k'>.
 */
std::complex<double> ExcitonTB::blochCoherenceFactor(const arma::cx_vec& coefs1, const arma::cx_vec& coefs2,
                                                    const arma::rowvec& k1, const arma::rowvec& k2,
                                                    const arma::rowvec& G){

    std::complex<double> imag(0, 1);
    arma::cx_vec coefs = arma::conj(coefs1) % coefs2;
    arma::cx_vec phases = arma::ones<arma::cx_vec>(system->basisdim);
    int index_min = 0;
    int index_max = -1;

    for(int i = 0; i < system->natoms; i++){
        int species = system->motif.row(i)(3);
        arma::rowvec atomPosition = system->motif.row(i).subvec(0, 2); 

        index_max += system->orbitals(species);
        phases.subvec(index_min, index_max) *= exp(imag*arma::dot(k1 - k2 + G, atomPosition));

        index_min = index_max + 1;
    }

    std::complex<double> factor = arma::dot(coefs, phases);

    return factor;
}


/*------------------------------------ Electron-hole pair basis ------------------------------------*/

/**
 * Method to generate a basis which is a subset of the basis considered for the
 * exciton. Its main purpose is to allow computation of Fermi golden rule between
 * two specified subbasis. 
 * @param bands Band subset of the originally specified for the exciton.
 * @return Matrix with the states corresponding to the specified subset.
 */
arma::imat ExcitonTB::specifyBasisSubset(const arma::ivec& bands){

    // Check if given bands vector corresponds to subset of bands
    try{
        for (const auto& band : bands){
            for (const auto& reference_band : bandList){
                if ((band + system->fermiLevel - reference_band) == 0) {
                    continue;
                }
            }
            throw "Error: Given band list must be a subset of the exciton one"; 
        };
    }
    catch (std::string e){
        std::cerr << e;
    };

    int reducedBasisDim = system->nk*bands.n_elem;
    std::vector<arma::s64> valence, conduction;
    for(int i = 0; i < bands.n_elem; i++){
        if (bands(i) <= 0){
            valence.push_back(bands(i) + system->fermiLevel);
        }
        else{
            conduction.push_back(bands(i) + system->fermiLevel);
        }
    }
    arma::ivec valenceBands = arma::ivec(valence);
    arma::ivec conductionBands = arma::ivec(conduction);

    arma::imat states = createBasis(conductionBands, valenceBands);

    return states;
}

/**
 * Criterium to fix the phase of the single-particle eigenstates after diagonalization.
 * @details The prescription we take here is to impose that the sum of all the coefficients is real.
 * @return Fixed coefficients. 
 */
arma::cx_mat ExcitonTB::fixGlobalPhase(arma::cx_mat& coefs){

    arma::cx_rowvec sums = arma::sum(coefs);
    std::complex<double> imag(0, 1);
    for(int j = 0; j < sums.n_elem; j++){
        double phase = arg(sums(j));
        coefs.col(j) *= exp(-imag*phase);
    }

    return coefs;
}

/*------------------------------------ Initializers ------------------------------------*/

/**
 * Method to initialize the motif Fourier transform for all possible motif combination 
 * at a given kpoint.
 * @param i Index of kpoint.
 * @param cells Matrix of unit cells over which the motif FT is computed.
 * @param potential Pointer to potential function.
 * @return void
 */
void ExcitonTB::initializeMotifFT(int i, const arma::mat& cells, potptr potential){
    ftMotifStack.slice(i) = motifFTMatrix(system->meshBZ.row(i), cells, potential);
}


/**
 * Main method to compute all the relevant single-particle quantities (bands, eigenstates and fourier transforms),
 * to compute the Bethe-Salpeter equation. Updated to keep track of band spin and prevent weird relabelings whenever bandtracking is set to true.
 * @details It precomputes and saves the relevant data in the heap for later computations.
 * @param triangular Boolean to specify whether the Hamiltonian matrices are triangular (default = false).
 * @return void
 */ 
void ExcitonTB::initializeResultsH0(){
    int nTotalBands = bandList.n_elem;
    double radius = arma::norm(system->bravaisLattice.row(0)) * cutoff_;
    arma::mat cells = system_->truncateSupercell(ncell, radius);
    int nk = system->nk;
    int natoms = system->natoms;
    int basisdim = system->basisdim;
    this->eigvecKStack_  = arma::cx_cube(basisdim, nTotalBands, nk);
    this->eigvecKQStack_ = arma::cx_cube(basisdim, nTotalBands, nk);
    this->eigvalKStack_  = arma::mat(nTotalBands, nk);
    this->eigvalKQStack_ = arma::mat(nTotalBands, nk);
    this->ftMotifStack   = arma::cx_cube(natoms, natoms, system->meshBZ.n_rows);
    this->ftMotifQ       = arma::cx_mat(natoms, natoms);
    if(this->selfenergy){
        this->ftMotifQ3 = arma::cx_cube(natoms, natoms, system->meshBZ.n_rows);      
    }
    arma::vec auxEigVal(basisdim);
    arma::cx_mat auxEigvec(basisdim, basisdim);
    
    const int historySize = 4;
    std::vector<arma::cx_mat> prevEigvecs,  prevEigvecsQ;
    std::vector<arma::vec>    prevSpinZs,   prevSpinZsQ;
    std::vector<arma::vec>    prevEigvals,  prevEigvalsQ;
    
    // Progress bar variables
    int step = 1;
    int displayNext = step;
    int percent = 0;
    system_->calculateInverseReciprocalMatrix();
    std::complex<double> imag(0, 1);
    std::cout << "Diagonalizing H0 for all k points... " << std::flush;
    for (int i = 0; i < nk; i++){
        arma::rowvec k = system->kpoints.row(i);
        system->solveBands(k, auxEigVal, auxEigvec);
        
        if (bandTracking_ && !prevEigvecs.empty()) {
            system->trackBands(prevEigvecs, prevSpinZs, prevEigvals,
                               auxEigvec, auxEigVal, bandTrackingThreshold_);
        }
        if (bandTracking_) {
            arma::vec spinZ(basisdim);
            for (int ib = 0; ib < basisdim; ib++){
                spinZ(ib) = system->expectedSpinZValue(auxEigvec.col(ib));
            }
            prevEigvecs.push_back(auxEigvec);
            prevSpinZs.push_back(spinZ);
            prevEigvals.push_back(auxEigVal);
            if ((int)prevEigvecs.size() > historySize) {
                prevEigvecs.erase(prevEigvecs.begin());
                prevSpinZs.erase(prevSpinZs.begin());
                prevEigvals.erase(prevEigvals.begin());
            }
            if (i == 0) {
                std::cout << "Band tracking active, checking spin consistency..." << std::endl;
                std::cout << "basisdim=" << basisdim << " nbands=" << bandList.n_elem << std::endl;
                for (int ib = 0; ib < basisdim; ib++) {
                    std::cout << "  band=" << ib 
                    << " Sz=" << system->expectedSpinZValue(auxEigvec.col(ib))
                    << " Eval=" << auxEigVal(ib) << std::endl;
                }
            }
            
            for (int ib = 0; ib < basisdim - 2; ib++) {
                double szA = system->expectedSpinZValue(auxEigvec.col(ib));
                double szB = system->expectedSpinZValue(auxEigvec.col(ib + 2));
                if (std::abs(szA - szB) < 0.1) {
                    if (auxEigVal(ib) > auxEigVal(ib + 2) + 1e-6) {
                        std::cerr << "WARNING: energy inversion at k=" << i
                        << " bands " << ib << "," << ib + 2
                        << " Eval=" << auxEigVal(ib)
                        << "," << auxEigVal(ib + 2) << std::endl;
                    }
                }
            }
        }
        
        
        
        auxEigvec = fixGlobalPhase(auxEigvec);
        eigvalKStack_.col(i) = auxEigVal(bandList);
        eigvecKStack_.slice(i) = auxEigvec.cols(bandList);
        
        if(arma::norm(Q) != 0){
            arma::rowvec kQ = system->kpoints.row(i) + Q;
            system->solveBands(kQ, auxEigVal, auxEigvec);
            
            if (bandTracking_ && !prevEigvecsQ.empty()) {
                system->trackBands(prevEigvecsQ, prevSpinZsQ, prevEigvalsQ,
                                   auxEigvec, auxEigVal, bandTrackingThreshold_);
            }
            if (bandTracking_) {
                arma::vec spinZQ(basisdim);
                for (int ib = 0; ib < basisdim; ib++){
                    spinZQ(ib) = system->expectedSpinZValue(auxEigvec.col(ib));
                }
                prevEigvecsQ.push_back(auxEigvec);
                prevSpinZsQ.push_back(spinZQ);
                prevEigvalsQ.push_back(auxEigVal);
                if ((int)prevEigvecsQ.size() > historySize) {
                    prevEigvecsQ.erase(prevEigvecsQ.begin());
                    prevSpinZsQ.erase(prevSpinZsQ.begin());
                    prevEigvalsQ.erase(prevEigvalsQ.begin());
                }
            }
            
            auxEigvec = fixGlobalPhase(auxEigvec);
            eigvalKQStack_.col(i) = auxEigVal(bandList);
            eigvecKQStack_.slice(i) = auxEigvec.cols(bandList);
        }
        else{
            eigvecKQStack_.slice(i) = eigvecKStack_.slice(i);
            eigvalKQStack_.col(i) = eigvalKStack_.col(i);
        };
    };
    std::cout << "Done" << std::endl;
    if(this->mode == "realspace"){
        std::cout << "Computing lattice Fourier transform... " << std::flush;
        potptr directPotential = selectPotential(this->potential_);
        #pragma omp parallel for
        for (unsigned int i = 0; i < system->meshBZ.n_rows; i++){
            initializeMotifFT(i, cells, directPotential);
            if(this->selfenergy){
                if(arma::norm(Q) != 0){
                    potptr selfenergyPotential = selectPotential(this->selfenergyPotential_);
                    this->ftMotifQ3.slice(i) = motifFTMatrix(system->kpoints.row(i) - this->Q, cells, selfenergyPotential);
                }
                else {
                    potptr selfenergyPotential = selectPotential(this->selfenergyPotential_);
                    this->ftMotifQ3.slice(i) = this->ftMotifStack.slice(i);
                };
            }
        }
        std::cout << "Done\n" << std::endl;
    }
    if(this->exchange){
        potptr exchangePotential = selectPotential(this->exchangePotential_);
        this->ftMotifQ = motifFTMatrix(this->Q, cells, exchangePotential);
    }
};

/**
 * Routine to initialize the required variables to construct the Bethe-Salpeter Hamiltonian.
 * @param triangular Boolean to specify whether the single-particle Hamiltonian matrices are triangular.
 * @return void.
 */
void ExcitonTB::initializeHamiltonian(){

    if(bands.empty()){
        throw std::invalid_argument("Error: Exciton object must have some bands");
    }
    if(system->nk == 0){
        throw std::invalid_argument("Error: BZ mesh must be initialized first");
    }

    if (this->regularization_ < 1E-10){
        regularization_ = system->a;
    }

    this->excitonbasisdim_ = system->nk*valenceBands.n_elem*conductionBands.n_elem;
    this->totalCells_ = pow(ncell*system->factor, system->ndim);

    std::cout << "Initializing basis for BSE... " << std::flush;
    initializeBasis();
    generateBandDictionary();

    initializeResultsH0();
}

/**
 * Method to initialize the BSE.
 * @details Calls the more general routine which allows
 * to specify a subset of the complete basis.
 */ 
void ExcitonTB::BShamiltonian(){
    arma::imat basis = {};
    BShamiltonian(basis);
}

/**
 * Routine to compute the self-energy contribution to the bands used in the exciton computation.
 * @param Qtoggle toggle on whether to use Q in calculations. it is only set to zero when computing the correction to the band structure for exporting.
 * @param k/kp_index index of k/k' point for the computation of the shifted mot 
 * @param coefsK/Kp k/k' points in the definition of self-energy contribution
 * @return self energy contribution
 */
std::complex<double> ExcitonTB::selfenergyTerm(bool Qtoggle, uint64_t k_index, uint64_t kp_index, const arma::cx_vec& coefsK, const arma::cx_vec& coefsKp){

    std::complex<double> self = 0.0;
    if (this->selfenergy){
        uint64_t k_index_0 = system_->findEquivalentPointBZ(arma::rowvec{0,0,0}, ncell);
        arma::cx_mat motifFT0 = ftMotifStack.slice(k_index_0);
        arma::cx_vec coefsK3;
        arma::cx_mat motifFT3;
        if (mode == "realspace"){
            for (int v3 = 0; v3 < (int)valenceBands.n_elem; v3++){
                for (uint64_t k3_index = 0; k3_index < system->nk; k3_index++){
                    if (gauge == "atomic"){
                        coefsK3 = system_->latticeToAtomicGauge(eigvecKStack.slice(k3_index).col(v3), system->kpoints.row(k3_index));
                    }
                    else{
                        coefsK3 = eigvecKStack.slice(k3_index).col(v3);
                    }
                    uint64_t effective_k3_index = system_->findEquivalentPointBZ(system->kpoints.row(k3_index) - system->kpoints.row(k_index), ncell);
                    arma::cx_mat motifFT3 = (Qtoggle) ? this->ftMotifQ3.slice(effective_k3_index) : ftMotifStack.slice(effective_k3_index);
                    // if (Qtoggle){
                    //     arma::cx_mat motifFT3 = this->ftMotifQ3.slice(effective_k3_index);
                    // }
                    // else{
                    //     arma::cx_mat motifFT3 = ftMotifStack.slice(effective_k3_index);
                    //
                    // }
                    self = self + realSpaceInteractionTerm(coefsK, coefsK3 , coefsKp, coefsK3, motifFT0) - realSpaceInteractionTerm(coefsK, coefsK3, coefsK3, coefsKp, motifFT3);
                    
                };
            };
            return self;
        }
        else if (mode == "reciprocalspace"){
            for (int v3 = 0; v3 < (int)valenceBands.n_elem; v3++){
                for (uint64_t k3_index = 0; k3_index < system->nk; k3_index++){
                    if (gauge == "atomic"){
                        coefsK3 = system_->latticeToAtomicGauge(eigvecKStack.slice(k3_index).col(v3), system->kpoints.row(k3_index));
                    }
                    else{
                        coefsK3 = eigvecKStack.slice(k3_index).col(v3);
                    }
                    arma::rowvec k  = system->kpoints.row(k_index);
                    arma::rowvec k3 = system->kpoints.row(k3_index);
                    
                    // Direct (Hartree-like) term: q=0, analogous to motifFT0
                    std::complex<double> directTerm = reciprocalInteractionTerm(coefsK, coefsK3, coefsKp, coefsK3,k, k3, k, k3, this->nReciprocalVectors);
                    
                    // Exchange (Fock-like) term: q=k3-k, analogous to motifFT3
                    std::complex<double> exchangeTerm = reciprocalInteractionTerm(coefsK, coefsK3, coefsK3, coefsKp,k, k3, k3, k, this->nReciprocalVectors);
                    
                    self = self + directTerm - exchangeTerm;
                }
            }
            return self;
        }
        return self;
    }
    else{
        return self;
    }
};

/**
 * Initialize BSE hamiltonian matrix and kinetic matrix.
 * @details Instead of calculating the energies and coeficients dinamically, which
 * is too expensive, instead we first calculate those for each k, save them
 * in the heap, and then call them consecutively as we build the matrix.
 * Analogously, we calculate the Fourier transform of the potential beforehand,
 * saving it in the stack so that it can be later called in the matrix element
 * calculation.
 * Also note that this routine involves a omp parallelization when building the matrix.
 * @param basis Subset of the exciton basis to build the BSE. If none, defaults to
 * the complete or original basis.
 * @return void
 */
void ExcitonTB::BShamiltonian(const arma::imat& basis){
    
    arma::imat basisStates = this->basisStates;
    if (!basis.is_empty()){
        basisStates = basis;
    };
    uint64_t basisDimBSE = basisStates.n_rows;
    
    double estimated_gb = (!this->tammdancoff_) 
    ? (useCholesky_ ? 6.0*(double)basisDimBSE/2*(double)basisDimBSE/2*16.0/(1ULL<<30)
    : 6.0*(double)basisDimBSE*2*(double)basisDimBSE*2*8.0/(1ULL<<30))
    : (double)basisDimBSE*(double)basisDimBSE*16.0/(1ULL<<30);
    std::cout << "BSE dimension: " << ((!this->tammdancoff_) ? 2*basisDimBSE : basisDimBSE) << std::endl;
    std::cout << "Estimated memory requirement for Bethe-Salpeter matrix: " << estimated_gb << " GB"<< std::endl;
    std::cout << "Initializing Bethe-Salpeter matrix... " << std::flush;
    
    HBS_ = (!this->tammdancoff_) ? arma::zeros<cx_mat>(2*basisDimBSE, 2*basisDimBSE) : arma::zeros<cx_mat>(basisDimBSE, basisDimBSE); 

    //set blocks as 1x1 matrices for memory saving if tammdancoff approximation is being used
    HBSres_  = (!this->tammdancoff_) ? arma::zeros<cx_mat>(basisDimBSE, basisDimBSE) : arma::zeros<cx_mat>(1, 1); 
    HBScoup_ = (!this->tammdancoff_) ? arma::zeros<cx_mat>(basisDimBSE, basisDimBSE) : arma::zeros<cx_mat>(1, 1); 
    
    
    
    // To be able to parallelize over the triangular matrix, we build
    uint64_t loopLength = basisDimBSE*(basisDimBSE + 1)/2.;
    
    // https://stackoverflow.com/questions/242711/algorithm-for-index-numbers-of-triangular-matrix-coefficients
    #pragma omp parallel for
    for(uint64_t n = 0; n < loopLength; n++){
        
        arma::cx_vec coefsK, coefsK2, coefsKQ, coefsK2Q;
        arma::cx_vec coefsKsw, coefsK2sw, coefsKQsw, coefsK2Qsw;
        
        uint64_t ii = loopLength - 1 - n;
        uint64_t m  = floor((sqrt(8*ii + 1) - 1)/2);
        uint64_t i = basisDimBSE - 1 - m;
        uint64_t j = basisDimBSE - 1 - ii + m*(m+1)/2;
        
        uint64_t k_index = basisStates(i, 2);
        uint64_t v = bandToIndex[basisStates(i, 0)];
        uint64_t c = bandToIndex[basisStates(i, 1)];
        uint64_t kQ_index = k_index;
        
        uint64_t k2_index = basisStates(j, 2);
        uint64_t v2 = bandToIndex[basisStates(j, 0)];
        uint64_t c2 = bandToIndex[basisStates(j, 1)];
        uint64_t k2Q_index = k2_index;
        // Using the atomic gauge
        if(gauge == "atomic"){
            coefsK = system_->latticeToAtomicGauge(eigvecKStack.slice(k_index).col(v), system->kpoints.row(k_index));
            coefsKQ = system_->latticeToAtomicGauge(eigvecKQStack.slice(kQ_index).col(c), system->kpoints.row(kQ_index));
            coefsK2 = system_->latticeToAtomicGauge(eigvecKStack.slice(k2_index).col(v2), system->kpoints.row(k2_index));
            coefsK2Q = system_->latticeToAtomicGauge(eigvecKQStack.slice(k2Q_index).col(c2), system->kpoints.row(k2Q_index));
        }
        else{
            coefsK = eigvecKStack.slice(k_index).col(v);
            coefsKQ = eigvecKQStack.slice(kQ_index).col(c);
            coefsK2 = eigvecKStack.slice(k2_index).col(v2);
            coefsK2Q = eigvecKQStack.slice(k2Q_index).col(c2);
        }
        
        std::complex<double> D, X, selfcond, selfval = 0.0;
        std::complex<double> Dcoup, Xcoup = 0.0;
        // std::complex<double> Dares, Xares = 0.0;
        if (mode == "realspace"){
            uint64_t effective_k_index = system_->findEquivalentPointBZ(system->kpoints.row(k2_index) - system->kpoints.row(k_index), ncell);
            arma::cx_mat motifFT = ftMotifStack.slice(effective_k_index);
            // Direct and exchange terms for resonant block of BSE matrix
            D = realSpaceInteractionTerm(coefsKQ, coefsK2, coefsK2Q, coefsK, motifFT);
            if(this->exchange){
                X = realSpaceInteractionTerm(coefsKQ, coefsK2, coefsK, coefsK2Q, this->ftMotifQ);
            }
            //self-energy terms
            if(this->selfenergy){
                bool testval = (kQ_index==k2Q_index and v==v2);
                bool testcond = (k_index==k2_index and c==c2);
                if (testval or testcond){
                    if (testval){
                        selfcond = selfenergyTerm(true, kQ_index, k2_index, coefsKQ, coefsK2Q);
                    }
                    if (testcond){
                        selfval = selfenergyTerm(false, k2_index, k_index, coefsK2, coefsK);
                    }
                }
            }
            if(!this->tammdancoff_){
                // Hcoup terms correspond to a swap c2<->v2
                coefsKsw = eigvecKStack.slice(k_index).col(c);
                coefsKQsw = eigvecKQStack.slice(kQ_index).col(v);
                coefsK2sw = eigvecKStack.slice(k2_index).col(c2);
                coefsK2Qsw = eigvecKQStack.slice(k2Q_index).col(v2);
                
                Dcoup = realSpaceInteractionTerm(coefsK, coefsK2Qsw, coefsK2sw, coefsKQ, motifFT);
                if(this->exchange){
                    Xcoup = realSpaceInteractionTerm(coefsK, coefsK2Qsw, coefsKQ, coefsK2sw, this->ftMotifQ);
                }                
            }
        }
        else if (mode == "reciprocalspace"){
            arma::rowvec k = system->kpoints.row(k_index);
            arma::rowvec k2 = system->kpoints.row(k2_index);
            D = reciprocalInteractionTerm(coefsK, coefsK2, coefsKQ, coefsK2Q, k, k2, k, k2, this->nReciprocalVectors);
            if(this->exchange){
                X = reciprocalInteractionTerm(coefsK2Q, coefsK2, coefsKQ, coefsK, k2 + Q, k2, k + Q, k, this->nReciprocalVectors);
            }
            // Self-energy terms — analogous to real-space branch
            if(this->selfenergy){
                bool testval  = (kQ_index == k2Q_index and v  == v2);
                bool testcond = (k_index  == k2_index  and c  == c2);
                if (testval or testcond){
                    if (testval){
                        selfcond = selfenergyTerm(true,  kQ_index, k2_index, coefsKQ,  coefsK2Q);
                    }
                    if (testcond){
                        selfval  = selfenergyTerm(false, k2_index, k_index,  coefsK2,  coefsK);
                    }
                }
            }
            if(!this->tammdancoff_){
                // Coupling block — swap c2<->v2
                coefsKsw   = eigvecKStack.slice(k_index).col(c);
                coefsKQsw  = eigvecKQStack.slice(kQ_index).col(v);
                coefsK2sw  = eigvecKStack.slice(k2_index).col(c2);
                coefsK2Qsw = eigvecKQStack.slice(k2Q_index).col(v2);
                
                Dcoup = reciprocalInteractionTerm(coefsK, coefsK2Qsw, coefsK2sw, coefsKQ, k, k2 + Q, k2, k + Q, this->nReciprocalVectors);
                if(this->exchange){
                    Xcoup = reciprocalInteractionTerm(coefsK, coefsK2Qsw, coefsKQ, coefsK2sw, k, k2 + Q, k + Q, k2, this->nReciprocalVectors);
                }
            }
        }
        
        if (i == j){
            if(!this->tammdancoff_){
                HBSres_(i, j) = (this->scissor + (eigvalKQStack.col(kQ_index)(c) + selfcond) - (eigvalKStack.col(k_index)(v) + selfval))/2.
                - (D - X)/2.;
                
                HBScoup_(i, j) = - (Dcoup - Xcoup)/2.;
            }
            else if(this->tammdancoff_){
                HBS_(i, j) = (this->scissor + (eigvalKQStack.col(kQ_index)(c) + selfcond) - (eigvalKStack.col(k_index)(v) + selfval))/2.
                - (D - X)/2.;
            }
        }
        else{
            if(!this->tammdancoff_){
                HBSres_(i, j)  = - (D - X);
                HBScoup_(i, j) = - (Dcoup - Xcoup);
            }
            else if(this->tammdancoff_){
                HBS_(i, j)  = - (D - X);
            }
        };
    }
    if(!this->tammdancoff_){
        HBSres_  = HBSres_  + HBSres_.t();
        HBScoup_ = HBScoup_ + HBScoup_.t();
        
        
        // We don't explicitly compute the antiresonat block nor the antires-res block as
        // HBS_ = join_rows( join_cols( HBSres_, -(HBScoup_.t()) ), join_cols( HBScoup_, HBSares_ ) );
        // instead we use the explicit properties of the BSE matrix to construct them from the other two
        // 10.1103/PhysRevB.92.045209   
         
        // Attempt Cholesky reduction to standard Hermitian problem.
        // Valid when (A-B) is positive definite (positive single-particle gap).
        // Stores L, Linv, ApB for use in diagonalizeRaw() instead of full HBS_.
        arma::cx_mat AmB = HBSres_ - HBScoup_;
        arma::cx_mat ApB = HBSres_ + HBScoup_;
        
        if(arma::chol(choleskyL_, AmB, "lower")){
                        
            HBSres_.reset();
            HBScoup_.reset();
            
            choleskyLinv_ = arma::inv(arma::trimatl(choleskyL_));
            HBS_ = choleskyL_.t() * ApB * choleskyL_;
            useCholesky_ = true;
            
            AmB.reset();
            ApB.reset();
            
            // DIAGNOSTIC STUFF
            // Armadillo "lower" gives AmB = L * L.t() or L.t() * L?
            // arma::cx_mat recon_LLt = choleskyL_ * choleskyL_.t();
            // arma::cx_mat recon_LtL = choleskyL_.t() * choleskyL_;
            // double err_LLt = arma::norm(recon_LLt - AmB, "fro") / arma::norm(AmB, "fro");
            // double err_LtL = arma::norm(recon_LtL - AmB, "fro") / arma::norm(AmB, "fro");
            // std::cout << "Cholesky convention check:" << std::endl;
            // std::cout << "  ||L*L† - AmB|| / ||AmB|| = " << err_LLt << std::endl;
            // std::cout << "  ||L†*L - AmB|| / ||AmB|| = " << err_LtL << std::endl;
            // --- End verification ---
        }
        else {
            // Fallback: A-B not positive definite
            // Build full pseudo-Hermitian matrix as before
            AmB.reset();
            ApB.reset();
            choleskyL_.reset();
            choleskyLinv_.reset();
            
            
            std::cout << "Warning: Cholesky reduction failed (A-B not positive definite). "
            << "Falling back to full BSE matrix." << std::endl;
            HBS_ = join_rows( join_cols( HBSres_, -(HBScoup_.t()) ),
                              join_cols( HBScoup_, -(HBSres_.t()) ) );
            useCholesky_ = false;
            HBSres_.reset();
            HBScoup_.reset();
            
        }
    }
    else if(this->tammdancoff_){
        HBS_ = HBS_ + HBS_.t();
    }
    std::cout << "Done" << std::endl;
};

/**
 * Routine to write the self energy contribution to each band to a file. Creates a matrix named selfen with nk rows and bands columns.
 * Each row is organized as selfenergy(band0) selfenergy(band1) ....
 * @param file Pointer to file.
 * 
 * 
 * @return void
 */
void ExcitonTB::writeBandSelfEnergy(FILE* file){
    arma::cx_vec coefsK;
    arma::cx_mat selfen = arma::zeros<cx_mat>(system->nk, (int)bands.n_elem);
    for (uint64_t k_index = 0; k_index < system->nk; k_index++){
        arma::rowvec kpoint = system->kpoints.row(k_index);
        fprintf(file, "%11.7lf\t%11.7lf\t%11.7lf\t", kpoint(0), kpoint(1), kpoint(2));
        
        for (int bandindex = 0; bandindex < (int)bands.n_elem; bandindex++){
            if (gauge == "atomic"){
                coefsK = system_->latticeToAtomicGauge(eigvecKStack.slice(k_index).col(bandindex), system->kpoints.row(k_index));
            }
            else{
                coefsK = eigvecKStack.slice(k_index).col(bandindex);
            }
            selfen(k_index, bandindex) = selfenergyTerm(false, k_index, k_index, coefsK, coefsK);
            fprintf(file, "%11.7lf\t%11.7lf\t", real(selfen(k_index, bandindex)), imag(selfen(k_index, bandindex)));
        }
        fprintf(file, "\n");
    }
};


/**
 * Routine to diagonalize the BSE and return a Result object.
 * @param method Method to diagonalize the BSE, either 'diag' (standard diagonalization) 
 * 'davidson' (iterative diagonalization) or 'sparse' (Lanczos).
 * @param nstates Number of states to be stored from the diagonalization.
 * @return Result object storing the exciton energies and states.
 */ 
ResultTB* ExcitonTB::diagonalizeRaw(std::string method, int nstates){

    if (HBS.empty() || HBS.is_zero()){
        throw std::invalid_argument("diagonalizeRaw(): BSE Hamiltonian is not initialized.");
    }

    arma::vec eigval;
    arma::cx_mat eigvec;

    uint64_t basisDimBSE = this->basisStates.n_rows;
    double estimated_gb;
    if(method == "zheevr"){
        estimated_gb = (
            16.0 * (double)basisDimBSE * basisDimBSE             // input matrix (destroyed in-place)
            + 16.0 * (double)basisDimBSE * 66.0         // LWORK: realistic for RANGE='I'
            + 8.0  * (double)basisDimBSE * 24.0         // LRWORK
            + 16.0 * (double)basisDimBSE * nstates      // eigenvector output
        ) / (1ULL << 30);
        if(useCholesky_){
            // reconstruction stage: XpY+XmY+eigvec (transient) then X+Y+eigvec_full (peak)
            double recon_gb = (16.0*basisDimBSE*nstates*2          // X, Y peak
                            + 16.0*(2*basisDimBSE)*(2*nstates)  // eigvec_full
                            ) / (1ULL << 30);
            estimated_gb = std::max(estimated_gb, recon_gb);
        }
    }
    else if(method == "diag"){
        estimated_gb = (!this->tammdancoff_ && !useCholesky_)
        ? (3.0 * 4.0 * (double)basisDimBSE * basisDimBSE * 8.0) / (1ULL << 30)  // full BSE, eig_gen
        : (3.0 *       (double)basisDimBSE * basisDimBSE * 8.0) / (1ULL << 30); // TDA, eig_sym
    }
    else if(method == "davidson"){
        int max_sub = std::max(10 * nstates, 50);
        estimated_gb = (
            3.0 * (double)basisDimBSE * max_sub * 16.0   // V, AV, ritz
        ) / (1ULL << 30);
    }
    
    std::cout << "Estimated memory requirement for BSE diagonalization: " << estimated_gb << " GB"<< std::endl;
    
    // Hard limit from Armadillo's internal 2^31 element check (https://gitlab.com/conradsnicta/armadillo-code/-/blob/15.4.x/include/armadillo_bits/memory.hpp?ref_type=heads#L47)
    // Something is being cast as a signed 32 bit integer. Still trying to figure out what.
    // N*N must fit, so N_max = floor(sqrt(2^31)) = 46340
    const int64_t ARMA_HARD_LIMIT = 46340;
    
    //still want to look more into it, it might have been removed via the #define ARMA_64BIT_WORD flag
    
    if (method == "diag" && basisDimBSE > ARMA_HARD_LIMIT){
        std::cout << "BSE dimension " << basisDimBSE 
        << " exceeds Armadillo's single-allocation limit (max ~46340)."
        << " Switching to Zheevr." << std::endl;
        method = "zheevr";
    }
    
    std::cout << "Solving BSE with ";
    if (method == "diag"){
        std::cout << "exact diagonalization... " << std::flush;
        try {
            if(!this->tammdancoff_){
                if(useCholesky_){
                    // Full BSE, Cholesky path:
                    // HBS_ already stores C = L^{-1}(A+B)L^{-†} (Hermitian, half dimension)
                    // Eigenvalues of C are E^2
                    arma::eig_sym(eigval, eigvec, HBS);
                    
                    // get number of elements of eigval. same check for all diagonalization procedures
                    int nall = (int)eigval.n_elem;
                    
                    eigval = arma::sqrt(arma::clamp(eigval, 0.0, eigval.max()));
                    // double floorThreshold = std::max(1e-12, 1e-10 * eigval.max());
                    // arma::uword nBad = arma::find(eigval < floorThreshold).eval().n_elem;
                    // if(nBad > 0){
                    //     std::cerr << "Warning: " << nBad << " exciton state(s) have near-zero excitation "
                    //     "energy (< " << floorThreshold << " eV); X/Y decomposition is "
                    //     "numerically marginal for these states." << std::endl;
                    // }
                    // eigval = arma::clamp(eigval, floorThreshold, eigval.max());
                    // eigval = arma::sqrt(eigval);
                    
                    int dim = choleskyL_.n_rows;
                    arma::cx_mat XpY = choleskyL_        * eigvec;  // L * v       (= X+Y, before scaling)
                    arma::cx_mat XmY = choleskyLinv_.t() * eigvec;  // L^{-†} * v  (= X-Y, before scaling)
                    
                    eigvec.reset();               // eigvec no longer needed once XpY/XmY exist
                    choleskyL_.reset();
                    choleskyLinv_.reset();
                                                    
                    for(int i = 0; i < nall; i++){
                        XpY.col(i) /= std::sqrt(eigval(i));
                        XmY.col(i) *= std::sqrt(eigval(i));
                    }
                    arma::cx_mat X = 0.5 * (XpY + XmY);
                    arma::cx_mat Y = 0.5 * (XpY - XmY);
                    
                    XpY.reset(); 
                    XmY.reset();     // same as above
                    
                    // std::cout <<"\n"<< arma::norm(Y)/arma::norm(X)<<"\n"<< std::endl;
                    // for(int i = 0; i < std::min(nall, 5); i++){
                    //     double xn = arma::norm(X.col(i)); // magnitude of X for state i
                    //     double yn = arma::norm(Y.col(i));
                    //     std::cout << "state " << i << ": X'X-Y'Y = " << xn*xn - yn*yn << std::endl;
                    // }
                    arma::cx_mat eigvec_full(2*dim, 2*nall, arma::fill::zeros);
                    eigvec_full.submat(0,    0,     dim-1,   nall-1) = -arma::fliplr(arma::conj(Y));
                    eigvec_full.submat(dim,  0,     2*dim-1, nall-1) =  arma::fliplr(arma::conj(X));
                    eigvec_full.submat(0,    nall,  dim-1,   2*nall-1) = X;
                    eigvec_full.submat(dim,  nall,  2*dim-1, 2*nall-1) = Y;
                    
                    X.reset(); 
                    Y.reset();
                    eigvec = std::move(eigvec_full);
                    
                    arma::vec eigval_full(2*nall);
                    eigval_full.subvec(0,     nall-1)   = -arma::flipud(eigval);
                    eigval_full.subvec(nall,  2*nall-1) =  eigval;
                    eigval = std::move(eigval_full);
                    
                    
                    //DIAGNOSTIC cholesky eigvals should be mirrored
                    // int nall2 = (int)eigval.n_elem / 2;
                    // std::cout << "Cholesky eigval check (first 3 pairs):" << std::endl;
                    // for(int i = 0; i < std::min(3, nall2); i++){
                    //     std::cout << "  antires[" << i << "]=" << eigval(nall2 - i - 1)
                    //     << "  res[" << i << "]=" << eigval(nall2 + i) << std::endl;
                    // }
                }
                else{
                    // Full BSE, Cholesky failed
                    // HBS_ is the full pseudo-Hermitian 2N x 2N matrix.
                    std::cout << "(non-Hermitian fallback)... " << std::flush;
                    arma::cx_vec cx_eigval;
                    arma::uvec   sorted_indices;
                    arma::eig_gen(cx_eigval, eigvec, HBS);
                    eigval = arma::real(cx_eigval);
                    
                    sorted_indices = arma::sort_index(eigval, "ascend");
                    eigval = eigval(sorted_indices);
                    eigvec = eigvec.cols(sorted_indices);
                }
                
            }
            else if(this->tammdancoff_){
                arma::eig_sym(eigval, eigvec, HBS);
            }
        } catch (const std::exception& e) {
            std::cerr << "Diagonalization failed: " << e.what() << std::endl;
            std::cerr << "BSE matrix stats:" << std::endl;
            std::cerr << "  size: " << HBS.n_rows << " x " << HBS.n_cols << std::endl;
            std::cerr << "  memory needed: " << estimated_gb << " GB " << std::endl;
            std::cerr << "  is_hermitian check: " << arma::approx_equal(HBS, HBS.t(), "absdiff", 1e-10) << std::endl;
            std::cerr << "  norm: " << arma::norm(HBS, "fro") << std::endl;
            std::cerr << "  has_nan: " << HBS.has_nan() << std::endl;
            std::cerr << "  has_inf: " << HBS.has_inf() << std::endl;
            throw;
        }
    }
    else if (method == "zheevr"){
        std::cout << "partial diagonalization (zheevr)... " << std::flush;
        
        if(this->tammdancoff_){
            arma::cx_mat HBS_mutable = std::move(HBS_);
            diagonalize_partial(eigval, eigvec, HBS_mutable, nstates);
        }
        else{
            if(useCholesky_){
                arma::cx_mat HBS_mutable = std::move(HBS_);
                diagonalize_partial(eigval, eigvec, HBS_mutable, nstates);
                
                
                int nall = (int)eigval.n_elem;
                
                eigval = arma::sqrt(arma::clamp(eigval, 0.0, eigval.max()));
                // double floorThreshold = std::max(1e-12, 1e-10 * eigval.max());
                // arma::uword nBad = arma::find(eigval < floorThreshold).eval().n_elem;
                // if(nBad > 0){
                //     std::cerr << "Warning: " << nBad << " exciton state(s) have near-zero excitation "
                //     "energy (< " << floorThreshold << " eV); X/Y decomposition is "
                //     "numerically marginal for these states." << std::endl;
                // }
                // eigval = arma::clamp(eigval, floorThreshold, eigval.max());
                // eigval = arma::sqrt(eigval);
                
                int dim = choleskyL_.n_rows;

                arma::cx_mat XpY = choleskyL_        * eigvec;  // L * v       (= X+Y, before scaling)
                arma::cx_mat XmY = choleskyLinv_.t() * eigvec;  // L^{-†} * v  (= X-Y, before scaling)
                
                eigvec.reset();               // eigvec no longer needed once XpY/XmY exist
                choleskyL_.reset();
                choleskyLinv_.reset();
                
                for(int i = 0; i < nall; i++){
                    XpY.col(i) /= std::sqrt(eigval(i));
                    XmY.col(i) *= std::sqrt(eigval(i));
                }
                arma::cx_mat X = 0.5 * (XpY + XmY);
                arma::cx_mat Y = 0.5 * (XpY - XmY);
                
                XpY.reset(); 
                XmY.reset();     // same as above
                
                arma::cx_mat eigvec_full(2*dim, 2*nall, arma::fill::zeros);
                eigvec_full.submat(0,    0,     dim-1,   nall-1) = -arma::fliplr(arma::conj(Y));
                eigvec_full.submat(dim,  0,     2*dim-1, nall-1) =  arma::fliplr(arma::conj(X));
                eigvec_full.submat(0,    nall,  dim-1,   2*nall-1) = X;
                eigvec_full.submat(dim,  nall,  2*dim-1, 2*nall-1) = Y;
                
                X.reset(); 
                Y.reset();
                eigvec = std::move(eigvec_full);
                
                arma::vec eigval_full(2*nall);
                eigval_full.subvec(0,     nall-1)   = -arma::flipud(eigval);
                eigval_full.subvec(nall,  2*nall-1) =  eigval;
                eigval = std::move(eigval_full);
            }
            else{
                throw std::runtime_error(
                    "diagonalizeRaw(): zheevr for full BSE requires positive-definite "
                    "(A-B). Cholesky decomposition failed for this system.");
            }
        }
    }
    else if (method == "davidson"){
        std::cout << "Davidson method... " << std::flush;
        
        if(this->tammdancoff_){
            davidson_method_new(eigval, eigvec, HBS, nstates);
        }
        else{
            if(useCholesky_){
                
                davidson_method(eigval, eigvec, HBS, nstates);
                
                
                int nall = (int)eigval.n_elem;
                
                eigval = arma::sqrt(arma::clamp(eigval, 0.0, eigval.max()));
                // double floorThreshold = std::max(1e-12, 1e-10 * eigval.max());
                // arma::uword nBad = arma::find(eigval < floorThreshold).eval().n_elem;
                // if(nBad > 0){
                //     std::cerr << "Warning: " << nBad << " exciton state(s) have near-zero excitation "
                //     "energy (< " << floorThreshold << " eV); X/Y decomposition is "
                //     "numerically marginal for these states." << std::endl;
                // }
                // eigval = arma::clamp(eigval, floorThreshold, eigval.max());
                // eigval = arma::sqrt(eigval);
                
                int dim = choleskyL_.n_rows;
                
                arma::cx_mat XpY = choleskyL_        * eigvec;  // L * v       (= X+Y, before scaling)
                arma::cx_mat XmY = choleskyLinv_.t() * eigvec;  // L^{-†} * v  (= X-Y, before scaling)
                
                eigvec.reset();               // eigvec no longer needed once XpY/XmY exist
                choleskyL_.reset();
                choleskyLinv_.reset();
                
                for(int i = 0; i < nall; i++){
                    XpY.col(i) /= std::sqrt(eigval(i));
                    XmY.col(i) *= std::sqrt(eigval(i));
                }
                arma::cx_mat X = 0.5 * (XpY + XmY);
                arma::cx_mat Y = 0.5 * (XpY - XmY);
                
                XpY.reset(); 
                XmY.reset();     // same as above
                
                arma::cx_mat eigvec_full(2*dim, 2*nall, arma::fill::zeros);
                eigvec_full.submat(0,    0,     dim-1,   nall-1) = -arma::fliplr(arma::conj(Y));
                eigvec_full.submat(dim,  0,     2*dim-1, nall-1) =  arma::fliplr(arma::conj(X));
                eigvec_full.submat(0,    nall,  dim-1,   2*nall-1) = X;
                eigvec_full.submat(dim,  nall,  2*dim-1, 2*nall-1) = Y;
                
                X.reset(); 
                Y.reset();
                eigvec = std::move(eigvec_full);
                
                arma::vec eigval_full(2*nall);
                eigval_full.subvec(0,     nall-1)   = -arma::flipud(eigval);
                eigval_full.subvec(nall,  2*nall-1) =  eigval;
                eigval = std::move(eigval_full);
                
            }
            else{
                // Non-positive-definite fallback: pseudo-Hermitian Davidson
                // Not implemented — warn and throw
                throw std::runtime_error(
                    "diagonalizeRaw(): Davidson for full BSE requires positive-definite "
                    "(A-B). Cholesky decomposition failed for this system. "
                    "Use method='diag' with the non-Hermitian fallback.");
            }
        }
    }
    else if (method == "sparse"){
        std::cout << "Lanczos method... " << std::flush;
        
        if(!this->tammdancoff_){
            throw std::runtime_error(
                "diagonalizeRaw(): Lanczos method is only supported with TDA. "
                "Use method='diag' or method='davidson/zheevr' for full BSE.");
        }
        
        arma::cx_vec cx_eigval;
        eigs_opts opts;
        opts.maxiter = 10000;
        opts.tol     = 1e-10;
        
        arma::eigs_gen(cx_eigval, eigvec, arma::sp_cx_mat(HBS), nstates, "sr", opts);
        
        double max_imag = arma::max(arma::abs(arma::imag(cx_eigval)));
        if(max_imag > 1e-6){
            std::cerr << "Warning: Lanczos returned significant imaginary eigenvalue "
            << "components (max=" << max_imag << "). "
            << "Results may be inaccurate for degenerate states. "
            << "Consider using method='diag' or method='zheevr/davidson'." << std::endl;
        }
        eigval = arma::real(cx_eigval);
        
        // eigs_gen does not guarantee sorted output
        arma::uvec idx = arma::sort_index(eigval, "ascend");
        eigval = eigval(idx);
        eigvec = eigvec.cols(idx);
    }
    HBS_.reset();
    std::cout << "Done" << std::endl;

    return new ResultTB(this, eigval, eigvec);
}

/**
 * Routine to diagonalize the BSE and return a Result object.
 * @details Wrapper for the diagonalizeRaw method, which returns a raw pointer.
 * We wrap it into a unique pointer to avoid memory leaks.
 * @param method Method to diagonalize the BSE, either 'diag' (standard diagonalization) 
 * 'davidson' (iterative diagonalization) or 'sparse' (Lanczos).
 * @param nstates Number of states to be stored from the diagonalization.
 * @return Result object storing the exciton energies and states.
 */ 
std::unique_ptr<ResultTB> ExcitonTB::diagonalize(std::string method, int nstates){
    return std::unique_ptr<ResultTB>(diagonalizeRaw(method, nstates));
}

// ------------- Routines to compute Fermi Golden Rule -------------

/**
 * Method to compute density of states associated to non-interacting electron-hole pairs.
 * Considers only the bands defined as the basis for excitons.
 * @param energy Energy at which we evaluate the pair DoS.
 * @param delta Broadening used to smooth the DoS.
 * @return DoS at E.
 */
double ExcitonTB::pairDensityOfStates(double energy, double delta) const {
    
    double dos = 0;
    for(int v = 0; v < (int)valenceBands.n_elem; v++){
        for(int c = 0; c < (int)conductionBands.n_elem; c++){
            for(int i = 0; i < system->nk; i++){

                arma::uword vband = bandToIndex.at(valenceBands(v)); // Unsigned integer 
                arma::uword cband = bandToIndex.at(conductionBands(c));

                double stateEnergy = eigvalKStack.col(i)(cband) - eigvalKStack.col(i)(vband);
                dos += -PI*imag(rGreenF(energy, delta, stateEnergy));
            };
        }
    }
    dos /= (system->a * system->nk);

    return dos;
}


/** 
 * Routine to compute and write to a file the density of states of non-interacting e-h pairs.
 * @param file Pointer to file.
 * @param delta DoS broadening.
 * @param n Number of points on which we evaluate the DoS.
 * @return void
 */
void ExcitonTB::writePairDOS(FILE* file, double delta, int n){

    double eMin = eigvalKStack.min();
    double eMax = eigvalKQStack.max();
    arma::vec energies = arma::linspace(0, (eMax - eMin)*1.1, n);
    for (double energy : energies){
        double dos = pairDensityOfStates(energy, delta);
        fprintf(file, "%f\t%f\n", energy, dos);
    }
}


/**
 * Routine to compute the non-interacting electron-hole edge pair associated to a given energy.
 * @details We run a search algorithm to find which k value matches the given energy.
 * @param energy Energy at which we want the non-interacting e-h pair.
 * @param gapEnergy Values of the gap at all kpoints.
 * @param side Whether to obtain the pair at +k or -k.
 * @return Vec of coefficients in e-h pair basis associated to desired pair.
 */
cx_vec ExcitonTB::ehPairCoefs(double energy, const vec& gapEnergy, std::string side){

    cx_vec coefs = arma::zeros<cx_vec>(system->nk);
    int closestKindex = -1;
    double eDiff;
    double currentEnergy = gapEnergy(0) - energy;

    for(int n = 1; n < system->nk/2; n++){
        
        eDiff = gapEnergy(n) - energy;
        if(abs(eDiff) < abs(currentEnergy)){
            closestKindex = n;
            currentEnergy = eDiff;
        };
    };
    std::cout << closestKindex << std::endl;
    std::cout << "Selected k: " << system->kpoints(closestKindex) << "\t" << closestKindex << std::endl;
    std::cout << "Closest gap energy: " << gapEnergy(closestKindex) << std::endl;
    // By virtue of band symmetry, we expect n < nk/2
    double dispersion = PI/(16*system->a);
    if(side == "left"){
        coefs(closestKindex) = 1.;
    }
    else if(side == "right"){
        coefs(system->nk - 1 - closestKindex) = 1.;
    }

    std::cout << "Energy gap (-k): " << gapEnergy(closestKindex) << std::endl;
    std::cout << "Energy gap (k): " << gapEnergy(system->nk - 1 - closestKindex) << std::endl;

    return coefs;
};

/**
 * Method to compute the transition rate from one exciton to a general non-interacting electron-hole pair.
 * @param targetExciton Exciton object representing the final system->
 * @param initialState Exciton state (coefficients) from which the transition happens.
 * @param finalState Final exciton state in the transition.
 * @param energy Energy of the initial exciton state.
 * @return Transition rate from initialState to finalState.
 */ 
double ExcitonTB::fermiGoldenRule(const ExcitonTB& targetExciton, 
                                  const arma::cx_vec& initialState, 
                                  const arma::cx_vec& finalState, double energy){

    double transitionRate = 0;
    arma::imat initialBasis = basisStates;
    arma::imat finalBasis = targetExciton.basisStates;
    cx_mat W = arma::zeros<cx_mat>(finalBasis.n_rows, initialBasis.n_rows);

    // -------- Main loop (W initialization) --------
    #pragma omp parallel for schedule(static, 1) collapse(2)
    for (arma::uword i = 0; i < finalBasis.n_rows; i++){
        for (int j = 0; j < initialBasis.n_rows; j++){

            arma::cx_vec coefsK, coefsK2, coefsKQ, coefsK2Q;

            int vf = targetExciton.bandToIndex.at(finalBasis(i, 0));
            int cf = targetExciton.bandToIndex.at(finalBasis(i, 1));
            double kf_index = finalBasis(i, 2);
            
            int vi = bandToIndex[initialBasis(j, 0)];
            int ci = bandToIndex[initialBasis(j, 1)];
            double ki_index = initialBasis(j, 2);

            // Using the atomic gauge
            if(gauge == "atomic"){
                coefsK = system_->latticeToAtomicGauge(
                    targetExciton.eigvecKStack.slice(kf_index).col(vf), system->kpoints.row(kf_index));
                coefsKQ = system_->latticeToAtomicGauge(
                    targetExciton.eigvecKQStack.slice(kf_index).col(cf), system->kpoints.row(kf_index));
                coefsK2 = system_->latticeToAtomicGauge(
                    eigvecKStack.slice(ki_index).col(vi), system->kpoints.row(ki_index));
                coefsK2Q = system_->latticeToAtomicGauge(
                    eigvecKQStack.slice(ki_index).col(ci), system->kpoints.row(ki_index));
            }
            else{
                coefsK = targetExciton.eigvecKStack.slice(kf_index).col(vf);
                coefsKQ = targetExciton.eigvecKQStack.slice(kf_index).col(cf);
                coefsK2 = eigvecKStack.slice(ki_index).col(vi);
                coefsK2Q = eigvecKQStack.slice(ki_index).col(ci);
            }

            std::complex<double> D, X;
            if (mode == "realspace"){
                int effective_k_index = system_->findEquivalentPointBZ(
                    system->kpoints.row(ki_index) - system->kpoints.row(kf_index), ncell);
                arma::cx_mat motifFT = ftMotifStack.slice(effective_k_index);
                D = realSpaceInteractionTerm(coefsKQ, coefsK2, coefsK2Q, coefsK, motifFT);
                X = 0;

            }
            else if (mode == "reciprocalspace"){
                arma::rowvec k = system->kpoints.row(kf_index);
                arma::rowvec k2 = system->kpoints.row(ki_index);
                D = reciprocalInteractionTerm(coefsK, coefsK2, coefsKQ, coefsK2Q, k, k2, k, k2, this->nReciprocalVectors);
                X = 0;
            }
            
            W(i, j) = - (D - X);                
        };
    };

    double delta = 2.4/(2*ncell); // Adjust delta depending on number of k points
    double rho = targetExciton.pairDensityOfStates(energy, delta);
    cout << "DoS value: " << rho << endl;
    double hbar = 6.582119624E-16; // Units are eV*s

    transitionRate = 2*PI*std::norm(arma::cdot(finalState, W*initialState))*rho/hbar;

    return transitionRate;
}


/**
 * Method to identify a k point corresponding to a non-interacting electron-hole pair in the defined system
 * with the energy specified.
 * @param targetExciton Exciton object representing the general final states in the transition.
 * @param energy Energy of the initial state.
 * @param side Whether the transition takes place to an state with +k or -k.
 * @param increasing Used to specify whether the gap increases or decreases with k.
 * @return k vector of the equivalent electron-hole pair.
*/
arma::rowvec ExcitonTB::findElectronHolePair(const ExcitonTB& targetExciton,
                                             double energy, std::string side, bool increasing) {
    double n = 10;
    arma::rowvec min_k, max_k, kmin, kmax;
    arma::uword nk = system->nk;
    if (side == "right") {
        min_k = system->kpoints.row(nk/2);
        max_k = -system->kpoints.row(0);
    } else if (side == "left") {
        max_k = system->kpoints.row(0);
        min_k = system->kpoints.row(nk/2 - 1);
    }
    
    arma::rowvec k;
    double threshold = 1E-8;
    arma::vec eigval;
    arma::cx_mat eigvec;
    int currentIndex = 0;
    double currentEnergy = 0, vEnergy, cEnergy, gap, prevGap = 0;
    double prevEnergy = currentEnergy;
    const int historySize = 4;
    
    while (abs(currentEnergy - energy) > threshold) {
        std::vector<arma::cx_mat> prevEigvecs,  prevEigvecsQ;
        std::vector<arma::vec>    prevSpinZs,   prevSpinZsQ;
        std::vector<arma::vec>    prevEigvals,  prevEigvalsQ;
        
        for (double i = 0; i <= n; i++) {
            k = min_k * (1 - i/n) + max_k * i/n;
            
            targetExciton.system->solveBands(k, eigval, eigvec);
            if (bandTracking_ && !prevEigvecs.empty()) {
                targetExciton.system->trackBands(prevEigvecs, prevSpinZs, prevEigvals,
                                                 eigvec, eigval, bandTrackingThreshold_);
            }
            if (bandTracking_) {
                int nb = eigvec.n_cols;
                arma::vec sz(nb);
                for (int ib = 0; ib < nb; ib++)
                    sz(ib) = targetExciton.system->expectedSpinZValue(eigvec.col(ib));
                prevEigvecs.push_back(eigvec);
                prevSpinZs.push_back(sz);
                prevEigvals.push_back(eigval);
                if ((int)prevEigvecs.size() > historySize) {
                    prevEigvecs.erase(prevEigvecs.begin());
                    prevSpinZs.erase(prevSpinZs.begin());
                    prevEigvals.erase(prevEigvals.begin());
                }
            }
            
            arma::vec eigvalBands = eigval(targetExciton.bandList);
            vEnergy = eigvalBands(0);
            
            if (arma::norm(Q) != 0) {
                arma::rowvec kQ = k + Q;
                arma::cx_mat eigvecQ;
                arma::vec    eigvalQ;
                targetExciton.system->solveBands(kQ, eigvalQ, eigvecQ);
                if (bandTracking_ && !prevEigvecsQ.empty()) {
                    targetExciton.system->trackBands(prevEigvecsQ, prevSpinZsQ, prevEigvalsQ,
                                                     eigvecQ, eigvalQ, bandTrackingThreshold_);
                }
                if (bandTracking_) {
                    int nb = eigvecQ.n_cols;
                    arma::vec szQ(nb);
                    for (int ib = 0; ib < nb; ib++)
                        szQ(ib) = targetExciton.system->expectedSpinZValue(eigvecQ.col(ib));
                    prevEigvecsQ.push_back(eigvecQ);
                    prevSpinZsQ.push_back(szQ);
                    prevEigvalsQ.push_back(eigvalQ);
                    if ((int)prevEigvecsQ.size() > historySize) {
                        prevEigvecsQ.erase(prevEigvecsQ.begin());
                        prevSpinZsQ.erase(prevSpinZsQ.begin());
                        prevEigvalsQ.erase(prevEigvalsQ.begin());
                    }
                }
                eigvalBands = eigvalQ(targetExciton.bandList);
            }
            
            cEnergy = eigvalBands(1);
            gap = cEnergy - vEnergy;
            
            if (!increasing && (gap <= energy) && (prevGap > energy)) {
                currentIndex = i;
                currentEnergy = gap;
                kmin = min_k * (1 - (currentIndex - 1)/n) + max_k * (currentIndex - 1)/n;
                kmax = min_k * (1 - (currentIndex + 1)/n) + max_k * (currentIndex + 1)/n;
            }
            if (increasing && (gap > energy) && (prevGap <= energy)) {
                currentIndex = i;
                currentEnergy = gap;
                kmin = min_k * (1 - (currentIndex - 1)/n) + max_k * (currentIndex - 1)/n;
                kmax = min_k * (1 - (currentIndex + 1)/n) + max_k * (currentIndex + 1)/n;
            }
            prevGap = gap;
        }
        
        k = min_k * (1 - currentIndex/n) + max_k * currentIndex/n;
        min_k = kmin;
        max_k = kmax;
        arma::cout << "Current edge pair energy: " << currentEnergy << arma::endl;
        arma::cout << "Target energy: " << energy << "\n" << arma::endl;
        if (currentEnergy == prevEnergy) n += 1;
        prevEnergy = currentEnergy;
    }
    
    arma::cout << "k: " << k << arma::endl;
    return k;
}

/**
 * Method to compute the transition to an edge e-h pair with the same energy (up to some error) as the bulk exciton.
 * @param targetExciton Exciton object representing the general final states in the transition.
 * @param initialState Exciton state from which the transition happens.
 * @param energy Energy of the initial state.
 * @param side Whether the transition takes place to an state with +k or -k.
 * @param increasing Used to specify whether the gap increases or decreases with k.
 * @return Transition rate from the initial exciton state to a non-interacting e-h pair.
 */
double ExcitonTB::edgeFermiGoldenRule(const ExcitonTB& targetExciton, 
                                      const arma::cx_vec& initialState, 
                                      double energy, std::string side, bool increasing){

    double transitionRate = 0;
    arma::imat initialBasis = basisStates;

    arma::rowvec k = findElectronHolePair(targetExciton, energy, side, increasing);

    arma::vec eigval;
    arma::cx_mat eigvec;
    arma::cx_vec coefsK, coefsKQ;

    targetExciton.system->solveBands(k, eigval, eigvec);

    eigvec = fixGlobalPhase(eigvec);
    eigvec = eigvec.cols(targetExciton.bandList);
    coefsK = eigvec.col(0);

    if(arma::norm(Q) != 0){
        arma::rowvec kQ = k + Q;
        targetExciton.system->solveBands(kQ, eigval, eigvec);

        eigvec = fixGlobalPhase(eigvec);
        eigvec = eigvec.cols(targetExciton.bandList);
    }
    coefsKQ = eigvec.col(1);

    bool computeOccupations = true;
    if (computeOccupations){
        //////// Specific for Bi ribbon; must be deleted afterwards.
        int N = targetExciton.system->basisdim;
        double l_e_edge_occ = arma::norm(coefsKQ.subvec(0, 15));
        double r_e_edge_occ = arma::norm(coefsKQ.subvec(N - 16, N - 1));
        double l_h_edge_occ = arma::norm(coefsK.subvec(0, 15));
        double r_h_edge_occ = arma::norm(coefsK.subvec(N - 16, N - 1));

        std::cout << "left e occ.: " << l_e_edge_occ << "\nright e occ: " << r_e_edge_occ << std::endl;
        std::cout << "Total e occ.: " << std::sqrt(l_e_edge_occ*l_e_edge_occ + r_e_edge_occ*r_e_edge_occ) << arma::endl;
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "left h occ.: " << l_h_edge_occ << "\nright h occ: " << r_h_edge_occ << std::endl;
        std::cout << "Total h occ.: " << std::sqrt(l_h_edge_occ*l_h_edge_occ + r_h_edge_occ*r_h_edge_occ) << arma::endl;
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "Total e-h pair edge occu.: " << std::sqrt(l_e_edge_occ*l_e_edge_occ + r_e_edge_occ*r_e_edge_occ) + 
                    std::sqrt(l_h_edge_occ*l_h_edge_occ + r_h_edge_occ*r_h_edge_occ) << std::endl;
    }
    
    // Now compute motif FT using k of edge pair
    double radius = arma::norm(system->bravaisLattice.row(0)) * cutoff_;
    arma::mat cells = system_->truncateSupercell(ncell, radius);
    potptr potential = selectPotential(this->potential_);

    int natoms = system->natoms;
    arma::cx_cube ftMotifStack = arma::cx_cube(natoms, natoms, system->kpoints.n_rows);
    
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < system->nk; i++){
        for(int fAtomIndex = 0; fAtomIndex < natoms; fAtomIndex++){
            for(int sAtomIndex = fAtomIndex; sAtomIndex < natoms; sAtomIndex++){
                ftMotifStack(fAtomIndex, sAtomIndex, i) = 
                motifFourierTransform(fAtomIndex, sAtomIndex, system->kpoints.row(i) - k, cells, potential);
                ftMotifStack(sAtomIndex, fAtomIndex, i) = conj(ftMotifStack(fAtomIndex, sAtomIndex, i));
            }   
        }
    }
    
    arma::cx_vec W = arma::zeros<arma::cx_vec>(initialBasis.n_rows);

    // -------- Main loop (W initialization) --------
    // #pragma omp parallel for
    for (int i = 0; i < initialBasis.n_rows; i++){

        arma::cx_vec coefsK2, coefsK2Q;
        
        int vi = bandToIndex[initialBasis(i, 0)];
        int ci = bandToIndex[initialBasis(i, 1)];
        double ki_index = initialBasis(i, 2);

        // Using the atomic gauge
        if(gauge == "atomic"){
            coefsK2 = system_->latticeToAtomicGauge(eigvecKStack.slice(ki_index).col(vi), system->kpoints.row(ki_index));
            coefsK2Q = system_->latticeToAtomicGauge(eigvecKQStack.slice(ki_index).col(ci), system->kpoints.row(ki_index));
        }
        else{
            coefsK2 = eigvecKStack.slice(ki_index).col(vi);
            coefsK2Q = eigvecKQStack.slice(ki_index).col(ci);
        }

        std::complex<double> D, X;
        if (mode == "realspace"){
            arma::cx_mat motifFT = ftMotifStack.slice(ki_index);
            D = realSpaceInteractionTerm(coefsKQ, coefsK2, coefsK2Q, coefsK, motifFT);
            X = 0;

        }
        else if (mode == "reciprocalspace"){
            arma::rowvec k2 = system->kpoints.row(ki_index);
            D = reciprocalInteractionTerm(coefsK, coefsK2, coefsKQ, coefsK2Q, k, k2, k, k2, this->nReciprocalVectors);
            X = 0;
        }
        
        W(i) = - (D - X);                
    };

    double delta = 2.0/targetExciton.system->nk; // Adjust delta depending on number of k points
    double rho = targetExciton.pairDensityOfStates(energy, delta);
    cout << "DoS value: " << rho << endl;
    double hbar = 6.582119624E-16; // Units are eV*s

    transitionRate = (ncell*system->a)*2*PI*std::norm(arma::dot(W, initialState))*rho/hbar;


    return transitionRate;
}

/**
 * Method to print information about the exciton.
 * @return void 
 */
void ExcitonTB::printInformation(){
    cout << std::left << std::setw(30) << "Number of cells: " << ncell << endl;
    cout << std::left << std::setw(30) << "Valence bands:";
    for (int i = 0; i < valenceBands.n_elem; i++){
        cout << valenceBands(i) << " ";
    }
    cout << endl;

    cout << std::left << std::setw(30) << "Conduction bands: ";
    for (int i = 0; i < conductionBands.n_elem; i++){
        cout << conductionBands(i) << " ";
    }
    cout << "\n" << endl;

    cout << std::left << std::setw(30) << "Gauge used: " << gauge << endl;
    cout << std::left << std::setw(30) << "Calculation mode: " << mode << endl;
    if(mode == "reciprocalspace"){
        cout << std::left << std::setw(30) << "nG: " << nReciprocalVectors << endl;
    }
    cout << std::left << std::setw(30) << "Potential: " << potential_ << endl;
    if(exchange){
        cout << std::left << std::setw(30) << "Exchange: " << (exchange ? "True" : "False") << endl;
        cout << std::left << std::setw(30) << "Exchange potential: " << exchangePotential_ << endl;
    }
    if(selfenergy){
        cout << std::left << std::setw(30) << "Self-Energy: " << (selfenergy ? "True" : "False") << endl;
        cout << std::left << std::setw(30) << "Self-Energy potential: " << selfenergyPotential_ << endl;
    }
    if(!tammdancoff_){
        cout << std::left << std::setw(30) << "Tamm-Dancoff Approximation: " << (tammdancoff_ ? "True" : "False") << endl;
    }
    if(bandTracking_){
        cout << std::left << std::setw(30) << "Band Tracking: " << (bandTracking_ ? "True" : "False") << endl;
    }
    if(arma::norm(Q) > 1E-7){
        cout << std::left << std::setw(30) << "Q: "; 
        for (auto qi : Q){
            cout << qi << "  ";
        }
        cout << endl;
    }
    cout << std::left << std::setw(30) << "Scissor cut: " << scissor_ << endl;
}

}
