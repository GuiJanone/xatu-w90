#pragma once
#define ARMA_MAX_ELEM 0x200000000ULL
#define ARMA_64BIT_WORD
#include <armadillo>
#include <string>
#include "xatu/ExcitonTB.hpp"
#include "xatu/ExcitonConfiguration.hpp"

namespace xatu {

/**
 * Selects what writeExcitonHDF5 stores in the archive.
 * @details The energies, the calculation parameters, the system, the k-point mesh and
 * the single-particle energies of the bands used are always written. Everything else is
 * opt-in, mirroring the text outputs of the CLI (-c, -k, -r, -s).
 */
struct HDF5ExportOptions {
    // Name of the archive (overwritten if it exists)
    std::string filename;
    // Number of resonant exciton states to write, in ascending energy (0: all available)
    int nstates = 0;
    // Exciton coefficients (resonant block, and anti-resonant block without TDA)
    bool writeStates = false;
    // Reciprocal-space probability density on the BZ mesh
    bool writeKWF = false;
    // Real-space electron probability density with the hole fixed
    bool writeRSWF = false;
    // Total, hole and electron spin
    bool writeSpin = false;
    // Motif index of the atom holding the hole (real-space w.f.)
    int holeIndex = 0;
    // Radius, in units of |a1|, of the region where the real-space w.f. is evaluated
    int ncellsRSWF = 8;
    // gzip level 0-9 (0 disables compression)
    int compression = 4;
    // Decimals used to group degenerate states (same criterion as the printed table)
    int precision = 6;
    // Provenance, stored as attributes
    std::string method;
    double energyCutoff = 0.0;
    std::string systemFile;
    std::string excitonFile;
    std::string commandLine;
};

void writeExcitonHDF5(ExcitonTB&, ResultTB&, const ExcitonConfiguration&, const HDF5ExportOptions&);

}
