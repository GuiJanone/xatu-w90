#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "xatu/ExcitonHDF5.hpp"

#ifdef ARMA_USE_HDF5
#include <hdf5.h>
#endif

namespace xatu {

#ifdef ARMA_USE_HDF5
namespace {

/**
 * Owns one HDF5 identifier and releases it with the matching H5*close routine.
 * Construction fails loudly if the HDF5 call that produced the identifier failed.
 */
class H5Id {
    public:
        H5Id(hid_t id, herr_t (*closer)(hid_t), const std::string& what) : id_(id), closer_(closer) {
            if (id_ < 0){
                throw std::runtime_error("HDF5 export: could not " + what);
            }
        }
        H5Id(H5Id&& other) noexcept : id_(other.id_), closer_(other.closer_) { other.id_ = -1; }
        H5Id(const H5Id&) = delete;
        H5Id& operator=(const H5Id&) = delete;
        ~H5Id(){ if (id_ >= 0) closer_(id_); }
        operator hid_t() const { return id_; }
    private:
        hid_t id_;
        herr_t (*closer_)(hid_t);
};

void check(herr_t status, const std::string& what){
    if (status < 0){
        throw std::runtime_error("HDF5 export: could not " + what);
    }
}

/**
 * Open archive plus the derived datatypes shared by all datasets.
 * @details Complex numbers are stored as the compound {r, i} of two doubles, the layout of
 * std::complex<double>; h5py reads it directly as numpy complex128. Booleans use the int8
 * enum {FALSE, TRUE}, which h5py reads as numpy bool.
 */
class Archive {
    public:
        H5Id file;
        H5Id complexType;
        H5Id boolType;
        H5Id stringType;
        int level;

        Archive(const std::string& filename, int compression) :
            file(createFile(filename), H5Fclose, "create file '" + filename + "'"),
            complexType(H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>)), H5Tclose, "create complex type"),
            boolType(H5Tenum_create(H5T_NATIVE_INT8), H5Tclose, "create bool type"),
            stringType(H5Tcopy(H5T_C_S1), H5Tclose, "create string type"),
            level(compression) {

            check(H5Tinsert(complexType, "r", 0, H5T_NATIVE_DOUBLE), "build complex type");
            check(H5Tinsert(complexType, "i", sizeof(double), H5T_NATIVE_DOUBLE), "build complex type");
            int8_t no = 0, yes = 1;
            check(H5Tenum_insert(boolType, "FALSE", &no), "build bool type");
            check(H5Tenum_insert(boolType, "TRUE", &yes), "build bool type");
            check(H5Tset_size(stringType, H5T_VARIABLE), "build string type");
            check(H5Tset_cset(stringType, H5T_CSET_UTF8), "build string type");

            if (level > 0 && H5Zfilter_avail(H5Z_FILTER_DEFLATE) <= 0){
                std::cout << "Warning: HDF5 library has no gzip filter, writing uncompressed archive." << std::endl;
                level = 0;
            }
        }

    private:
        // Readable by any HDF5 >= 1.10 (and h5py), with the compact group storage of 1.8.
        static hid_t createFile(const std::string& filename){
            H5Id fapl(H5Pcreate(H5P_FILE_ACCESS), H5Pclose, "create file access list");
            check(H5Pset_libver_bounds(fapl, H5F_LIBVER_V18, H5F_LIBVER_V110), "set library version bounds");
            H5Id fcpl(H5Pcreate(H5P_FILE_CREATE), H5Pclose, "create file creation list");
            check(H5Pset_link_creation_order(fcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED),
                  "track creation order of the root group");
            check(H5Pset_attr_creation_order(fcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED),
                  "track creation order of the root attributes");
            return H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, fcpl, fapl);
        }
};

/**
 * Creates a group whose children and attributes are also indexed by creation order.
 */
H5Id createGroup(hid_t parent, const std::string& name){
    H5Id gcpl(H5Pcreate(H5P_GROUP_CREATE), H5Pclose, "create group property list");
    check(H5Pset_link_creation_order(gcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED),
          "track creation order of group '" + name + "'");
    check(H5Pset_attr_creation_order(gcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED),
          "track attribute creation order of group '" + name + "'");
    return H5Id(H5Gcreate2(parent, name.c_str(), H5P_DEFAULT, gcpl, H5P_DEFAULT), H5Gclose,
                "create group '" + name + "'");
}

/**
 * Writes a dataset in C (row-major) order. Datasets of 64 or more elements are chunked
 * (about 1 MiB per chunk along the first axis) and compressed with shuffle + gzip.
 * @param dims Shape of the dataset; empty for a scalar.
 */
void writeDataset(const Archive& archive, hid_t loc, const std::string& name, hid_t type,
                  const std::vector<hsize_t>& dims, const void* data){

    H5Id space(dims.empty() ? H5Screate(H5S_SCALAR)
                            : H5Screate_simple((int)dims.size(), dims.data(), nullptr),
               H5Sclose, "create dataspace for '" + name + "'");
    H5Id dcpl(H5Pcreate(H5P_DATASET_CREATE), H5Pclose, "create dataset property list");

    hsize_t nelem = 1;
    for (hsize_t d : dims) nelem *= d;
    if (archive.level > 0 && !dims.empty() && nelem >= 64){
        hsize_t rowElems = nelem / dims[0];
        hsize_t rows = (1 << 20) / (H5Tget_size(type) * rowElems);
        std::vector<hsize_t> chunk = dims;
        chunk[0] = std::max<hsize_t>(1, std::min<hsize_t>(dims[0], rows));
        check(H5Pset_chunk(dcpl, (int)chunk.size(), chunk.data()), "chunk '" + name + "'");
        check(H5Pset_shuffle(dcpl), "set shuffle filter on '" + name + "'");
        check(H5Pset_deflate(dcpl, archive.level), "set gzip filter on '" + name + "'");
    }

    H5Id dset(H5Dcreate2(loc, name.c_str(), type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT),
              H5Dclose, "create dataset '" + name + "'");
    check(H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), "write dataset '" + name + "'");
}

void writeScalar(const Archive& a, hid_t loc, const std::string& name, double value){
    writeDataset(a, loc, name, H5T_NATIVE_DOUBLE, {}, &value);
}

void writeVector(const Archive& a, hid_t loc, const std::string& name, const arma::vec& v){
    writeDataset(a, loc, name, H5T_NATIVE_DOUBLE, {v.n_elem}, v.memptr());
}

void writeVector(const Archive& a, hid_t loc, const std::string& name, const std::vector<int64_t>& v){
    writeDataset(a, loc, name, H5T_NATIVE_INT64, {v.size()}, v.data());
}

void writeVector(const Archive& a, hid_t loc, const std::string& name, const arma::cx_vec& v){
    writeDataset(a, loc, name, a.complexType, {v.n_elem}, v.memptr());
}

// Armadillo is column-major: transpose so that the dataset reads M[row, col].
void writeMatrix(const Archive& a, hid_t loc, const std::string& name, const arma::mat& m){
    arma::mat t = m.t();
    writeDataset(a, loc, name, H5T_NATIVE_DOUBLE, {m.n_rows, m.n_cols}, t.memptr());
}

void writeMatrix(const Archive& a, hid_t loc, const std::string& name, const arma::imat& m){
    arma::imat t = m.t();
    writeDataset(a, loc, name, H5T_NATIVE_INT64, {m.n_rows, m.n_cols}, t.memptr());
}

void writeAttribute(hid_t loc, const std::string& name, hid_t type,
                    const std::vector<hsize_t>& dims, const void* data){
    H5Id space(dims.empty() ? H5Screate(H5S_SCALAR)
                            : H5Screate_simple((int)dims.size(), dims.data(), nullptr),
               H5Sclose, "create dataspace for attribute '" + name + "'");
    H5Id attr(H5Acreate2(loc, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT),
              H5Aclose, "create attribute '" + name + "'");
    check(H5Awrite(attr, type, data), "write attribute '" + name + "'");
}

void attribute(const Archive&, hid_t loc, const std::string& name, double value){
    writeAttribute(loc, name, H5T_NATIVE_DOUBLE, {}, &value);
}

void attribute(const Archive&, hid_t loc, const std::string& name, int64_t value){
    writeAttribute(loc, name, H5T_NATIVE_INT64, {}, &value);
}

void attribute(const Archive& a, hid_t loc, const std::string& name, int value){
    attribute(a, loc, name, (int64_t)value);
}

void attribute(const Archive& a, hid_t loc, const std::string& name, bool value){
    int8_t v = value ? 1 : 0;
    writeAttribute(loc, name, a.boolType, {}, &v);
}

void attribute(const Archive& a, hid_t loc, const std::string& name, const std::string& value){
    const char* p = value.c_str();
    writeAttribute(loc, name, a.stringType, {}, &p);
}

void attribute(const Archive& a, hid_t loc, const std::string& name, const char* value){
    attribute(a, loc, name, std::string(value));
}

void attribute(const Archive&, hid_t loc, const std::string& name, const std::vector<double>& v){
    if (v.empty()) return;
    writeAttribute(loc, name, H5T_NATIVE_DOUBLE, {v.size()}, v.data());
}

void attribute(const Archive&, hid_t loc, const std::string& name, const std::vector<int64_t>& v){
    if (v.empty()) return;
    writeAttribute(loc, name, H5T_NATIVE_INT64, {v.size()}, v.data());
}

template <typename V>
std::vector<double> toDoubles(const V& v){
    return std::vector<double>(v.begin(), v.end());
}

template <typename V>
std::vector<int64_t> toInts(const V& v){
    std::vector<int64_t> out;
    for (auto x : v) out.push_back((int64_t)x);
    return out;
}

std::string timestamp(){
    std::time_t now = std::time(nullptr);
    char buffer[64];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S%z", std::localtime(&now));
    return buffer;
}

} // namespace
#endif

/**
 * Writes the exciton results to an HDF5 archive.
 * @details Layout (lengths in the units of the system file, Angstrom for model and Wannier90
 * files; energies in eV; all arrays in C order, indices 0-based unless stated):
 *   /                  attributes: calculation parameters (bands, k mesh, potential, ...)
 *   /system            lattice, reciprocal lattice, motif, orbitals per species
 *   /kpoints           (nk, 3) BZ mesh used in the BSE
 *   /bands             energies (nk, nbands) [and eigenvectors (nk, nbands, norb) with states]
 *   /basis             (dim, 3) electron-hole basis: valence band, conduction band, k index
 *   /realspace         sites where the real-space w.f. is evaluated (with -r)
 *   /summary           energies, degeneracy groups [, spin] of all written states
 *   /excitons/<NNNN>   one group per state, 1-based and ordered by energy: eigval [, state,
 *                      state_antiresonant, kwf, rswf, spin]
 * @param exciton Exciton object after diagonalization.
 * @param results Results of the diagonalization.
 * @param config Exciton configuration the calculation was run with.
 * @param options Selection of contents and provenance information.
 */
void writeExcitonHDF5(ExcitonTB& exciton, ResultTB& results, const ExcitonConfiguration& config,
                      const HDF5ExportOptions& options){
#ifndef ARMA_USE_HDF5
    (void)exciton; (void)results; (void)config; (void)options;
    throw std::runtime_error("Xatu was not compiled with HDF5 support (build with 'make HDF5=1')");
#else
    const auto& system = results.system;
    const auto& info = config.excitonInfo;

    int offset = results.resonantOffset();
    int navail = (int)results.eigval.n_elem - offset;
    int nwrite = (options.nstates == 0) ? navail : std::min(options.nstates, navail);
    if (nwrite <= 0){
        throw std::invalid_argument("writeExcitonHDF5: no resonant states to write");
    }
    if (options.writeSpin && system->basisdim % 2 != 0){
        throw std::invalid_argument("writeExcitonHDF5: spin requested but the system basis does not include spin");
    }
    if (options.writeRSWF && (options.holeIndex < 0 || options.holeIndex >= system->natoms)){
        throw std::invalid_argument("writeExcitonHDF5: hole index " + std::to_string(options.holeIndex) +
                                    " out of range [0, " + std::to_string(system->natoms) + ")");
    }

    uint64_t dim = exciton.excitonbasisdim;
    uint64_t nk = system->nk;
    int npairs = exciton.valenceBands.n_elem * exciton.conductionBands.n_elem;
    bool hasY = !exciton.TDA && results.eigvec.n_rows == 2*dim;

    Archive archive(options.filename, options.compression);
    hid_t root = archive.file;

    // ------------------------- Calculation parameters -------------------------
    attribute(archive, root, "format", "xatu-excitons");
    attribute(archive, root, "format_version", 1);
    attribute(archive, root, "created", timestamp());
    attribute(archive, root, "label", info.label);
    attribute(archive, root, "system_file", options.systemFile);
    attribute(archive, root, "exciton_file", options.excitonFile);
    attribute(archive, root, "command_line", options.commandLine);
    attribute(archive, root, "units", "energies in eV; lengths in the units of the system file "
                                      "(Angstrom for model and Wannier90 files); k in inverse length");

    attribute(archive, root, "ncells", info.ncell);
    attribute(archive, root, "submesh_factor", info.submeshFactor);
    attribute(archive, root, "kpoint_shift", toDoubles(info.shift));
    attribute(archive, root, "n_kpoints", (int64_t)nk);
    attribute(archive, root, "Q", toDoubles(exciton.Q));
    attribute(archive, root, "band_list", toInts(exciton.bands));
    attribute(archive, root, "valence_bands", toInts(exciton.valenceBands));
    attribute(archive, root, "conduction_bands", toInts(exciton.conductionBands));
    attribute(archive, root, "fermi_level", system->fermiLevel);

    attribute(archive, root, "potential", info.potential);
    if (!info.eps.empty()){
        attribute(archive, root, "eps_m", exciton.eps_m);
        attribute(archive, root, "eps_s", exciton.eps_s);
        attribute(archive, root, "r0", exciton.r0);
        attribute(archive, root, "ry", exciton.ry);
        attribute(archive, root, "rz", exciton.rz);
    }
    attribute(archive, root, "hubbard_U", toDoubles(info.hubbardU));
    attribute(archive, root, "exchange", info.exchange);
    if (info.exchange){
        attribute(archive, root, "exchange_potential", info.exchangePotential);
    }
    attribute(archive, root, "selfenergy", info.selfenergy);
    if (info.selfenergy){
        attribute(archive, root, "selfenergy_potential", info.selfenergyPotential);
    }
    attribute(archive, root, "mode", exciton.mode);
    if (exciton.mode == "reciprocalspace"){
        attribute(archive, root, "n_reciprocal_vectors", exciton.nReciprocalVectors);
    }
    attribute(archive, root, "gauge", exciton.gauge);
    attribute(archive, root, "cutoff", exciton.cutoff);
    attribute(archive, root, "regularization", exciton.regularization);
    attribute(archive, root, "scissor", exciton.scissor);
    attribute(archive, root, "tamm_dancoff", exciton.TDA);
    attribute(archive, root, "band_tracking", exciton.bandTracking);
    if (exciton.bandTracking){
        attribute(archive, root, "band_tracking_threshold", exciton.bandTrackingThreshold);
    }

    attribute(archive, root, "diagonalization_method", options.method);
    attribute(archive, root, "energy_cutoff", options.energyCutoff);
    attribute(archive, root, "exciton_basis_dim", (int64_t)dim);
    attribute(archive, root, "n_excitons_computed", navail);
    attribute(archive, root, "n_excitons", nwrite);

    // --------------------------------- System ---------------------------------
    {
        H5Id group = createGroup(root, "system");
        attribute(archive, group, "ndim", system->ndim);
        attribute(archive, group, "natoms", system->natoms);
        attribute(archive, group, "basisdim", system->basisdim);
        attribute(archive, group, "filling", system->filling);
        attribute(archive, group, "unit_cell_area", system->unitCellArea);
        writeMatrix(archive, group, "bravais_lattice", system->bravaisLattice);
        writeMatrix(archive, group, "reciprocal_lattice", system->reciprocalLattice);
        writeMatrix(archive, group, "motif", system->motif);
        writeVector(archive, group, "orbitals", toInts(system->orbitals));
        H5Id motif(H5Dopen2(group, "motif", H5P_DEFAULT), H5Dclose, "reopen 'motif'");
        attribute(archive, motif, "columns", "x, y, z, species index (row of 'orbitals')");
    }

    // --------------------------- k mesh, bands, basis ---------------------------
    writeMatrix(archive, root, "kpoints", system->kpoints);
    {
        H5Id group = createGroup(root, "bands");
        attribute(archive, group, "description", "single-particle bands used in the BSE, in the order of "
                  "'indices' (absolute 0-based band numbers); energies[k, b], eigenvectors[k, b, orbital]");
        writeVector(archive, group, "indices", toInts(exciton.bandList));
        // eigvalKStack is (nbands, nk) column-major, i.e. (nk, nbands) in C order; same for the cube.
        writeDataset(archive, group, "energies", H5T_NATIVE_DOUBLE,
                     {nk, exciton.eigvalKStack.n_rows}, exciton.eigvalKStack.memptr());
        bool finiteQ = arma::norm(exciton.Q) > 1E-7;
        if (finiteQ){
            writeDataset(archive, group, "energies_kQ", H5T_NATIVE_DOUBLE,
                         {nk, exciton.eigvalKQStack.n_rows}, exciton.eigvalKQStack.memptr());
        }
        if (options.writeStates){
            const arma::cx_cube& v = exciton.eigvecKStack;
            writeDataset(archive, group, "eigenvectors", archive.complexType,
                         {v.n_slices, v.n_cols, v.n_rows}, v.memptr());
            if (finiteQ){
                const arma::cx_cube& vq = exciton.eigvecKQStack;
                writeDataset(archive, group, "eigenvectors_kQ", archive.complexType,
                             {vq.n_slices, vq.n_cols, vq.n_rows}, vq.memptr());
            }
        }
    }
    if (options.writeStates){
        writeMatrix(archive, root, "basis", exciton.basisStates);
        H5Id basis(H5Dopen2(root, "basis", H5P_DEFAULT), H5Dclose, "reopen 'basis'");
        attribute(archive, basis, "columns", "valence band, conduction band (absolute 0-based band numbers), "
                  "k index (row of /kpoints); valence fastest, then conduction, then k");
    }

    // ------------------------- Real-space evaluation sites -------------------------
    arma::mat cells;
    arma::rowvec holeCell = {0., 0., 0.};
    if (options.writeRSWF){
        double radius = arma::norm(system->bravaisLattice.row(0)) * options.ncellsRSWF;
        cells = system->truncateSupercell(exciton.ncell, radius);
        int natoms = system->natoms;
        arma::mat positions(cells.n_rows*natoms, 3);
        arma::mat cellOfSite(cells.n_rows*natoms, 3);
        std::vector<int64_t> atomOfSite(cells.n_rows*natoms);
        for (arma::uword c = 0; c < cells.n_rows; c++){
            for (int atom = 0; atom < natoms; atom++){
                arma::uword site = atom + c*natoms;
                positions.row(site) = system->motif.row(atom).subvec(0, 2) + cells.row(c);
                cellOfSite.row(site) = cells.row(c);
                atomOfSite[site] = atom;
            }
        }
        H5Id group = createGroup(root, "realspace");
        attribute(archive, group, "description", "electron sites of the real-space w.f.: position = "
                  "motif[atom] + cell; rswf[site] = probability of the electron at the site with the hole "
                  "on motif atom 'hole_index' of the origin cell");
        attribute(archive, group, "hole_index", options.holeIndex);
        arma::rowvec holePosition = system->motif.row(options.holeIndex).subvec(0, 2) + holeCell;
        attribute(archive, group, "hole_position", toDoubles(holePosition));
        attribute(archive, group, "radius_cells", options.ncellsRSWF);
        writeMatrix(archive, group, "positions", positions);
        writeMatrix(archive, group, "cell", cellOfSite);
        writeVector(archive, group, "atom", atomOfSite);
    }

    // --------------------------------- Excitons ---------------------------------
    arma::vec energies = results.eigval.subvec(offset, offset + nwrite - 1);

    // Same chained criterion as the printed table: |E_i - E_{i-1}| < 10^-precision.
    std::vector<int64_t> degeneracyGroup(nwrite);
    double threshold = std::pow(10.0, -options.precision);
    degeneracyGroup[0] = 1;
    for (int i = 1; i < nwrite; i++){
        bool degenerate = std::abs(energies(i) - energies(i - 1)) < threshold;
        degeneracyGroup[i] = degeneracyGroup[i - 1] + (degenerate ? 0 : 1);
    }

    H5Id summary = createGroup(root, "summary");
    attribute(archive, summary, "description", "tables over all written states; row i is /excitons group i+1");
    attribute(archive, summary, "degeneracy_precision", options.precision);
    writeVector(archive, summary, "energies", energies);
    writeVector(archive, summary, "degeneracy_group", degeneracyGroup);

    H5Id excitons = createGroup(root, "excitons");
    attribute(archive, excitons, "description", "one group per resonant state, named by its 1-based position "
              "in ascending energy (zero padded)");

    arma::mat spinTable;
    if (options.writeSpin){
        spinTable = arma::mat(nwrite, 3);
    }

    int width = std::max(4, (int)std::to_string(nwrite).size());
    for (int i = 0; i < nwrite; i++){
        std::string number = std::to_string(i + 1);
        std::string name = std::string(width - number.size(), '0') + number;
        H5Id state = createGroup(excitons, name);
        attribute(archive, state, "index", i + 1);
        attribute(archive, state, "degeneracy_group", degeneracyGroup[i]);
        writeScalar(archive, state, "eigval", energies(i));

        int col = offset + i;
        arma::cx_vec X, Y;
        results.splitXY(col, X, Y);

        if (options.writeStates){
            writeVector(archive, state, "state", X);
            if (hasY){
                writeVector(archive, state, "state_antiresonant", Y);
            }
        }

        if (options.writeKWF){
            // Probability per k point, sum_{v,c} |X_vck|^2 - |Y_vck|^2; sums to 1 over the mesh.
            arma::vec kwf(nk, arma::fill::zeros);
            for (uint64_t k = 0; k < nk; k++){
                for (int p = 0; p < npairs; p++){
                    kwf(k) += std::norm(X(npairs*k + p)) - std::norm(Y(npairs*k + p));
                }
            }
            writeVector(archive, state, "kwf", kwf);
        }

        if (options.writeRSWF){
            std::cout << "Writing real-space w.f. of state " << i + 1 << " out of " << nwrite << std::endl;
            int natoms = system->natoms;
            arma::vec rswf(cells.n_rows*natoms);
            #pragma omp parallel for
            for (arma::uword c = 0; c < cells.n_rows; c++){
                arma::rowvec cell = cells.row(c);
                for (int atom = 0; atom < natoms; atom++){
                    double amplitude = results.realSpaceWavefunction(X, atom, options.holeIndex, cell, holeCell);
                    if (hasY){
                        amplitude -= results.realSpaceWavefunction(Y, atom, options.holeIndex, cell, holeCell);
                    }
                    rswf(atom + c*natoms) = amplitude;
                }
            }
            writeVector(archive, state, "rswf", rswf);
        }

        if (options.writeSpin){
            arma::cx_vec spin = results.spinX(col);
            arma::vec s = arma::real(spin);
            spinTable.row(i) = s.t();
            writeVector(archive, state, "spin", s);
        }
    }

    if (options.writeSpin){
        writeMatrix(archive, summary, "spin", spinTable);
        H5Id spin(H5Dopen2(summary, "spin", H5P_DEFAULT), H5Dclose, "reopen 'spin'");
        attribute(archive, spin, "columns", "total, hole, electron (S_z)");
    }

    check(H5Fflush(archive.file, H5F_SCOPE_GLOBAL), "flush archive");
#endif
}

}
