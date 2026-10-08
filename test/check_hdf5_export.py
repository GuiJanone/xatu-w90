#!/usr/bin/env python3
"""Check the HDF5 exciton archive (xatu -H) against the text outputs of the same run.

Each case runs bin/xatu twice on the same input, once writing the text files (-e -c -k -r -s)
and once writing the archive (-H), and compares eigenvalues, states, k and real-space wave
functions and spin to the precision of the text format. Degenerate states are compared through
quantities that do not depend on the basis chosen inside the degenerate subspace (projector,
summed densities, summed spin), since the two runs are separate diagonalizations. It also checks
the archive's own consistency (normalization, kwf from the states, basis ordering, layout).

Needs h5py and numpy, and a binary built with 'make HDF5=1'. Exits with status 1 on any failure.

Usage: python3 test/check_hdf5_export.py [--xatu bin/xatu] [--keep]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "examples", "material_models")

failures = []


def report(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


# ----------------------------------------------------------------------------------------
# Text parsers
# ----------------------------------------------------------------------------------------

def read_eigval(path):
    lines = open(path).read().split()
    nprint = int(lines[2])
    return np.array([float(x) for x in lines[3:3 + nprint]])


def read_states(path):
    with open(path) as f:
        dim = int(f.readline())
        basis = np.array([f.readline().split() for _ in range(dim)], dtype=float)
        states = []
        for line in f:
            vals = np.array(line.split(), dtype=float)
            if vals.size:
                states.append(vals[0::2] + 1j * vals[1::2])
    return basis, np.array(states)


def read_blocks(path, skip_header):
    blocks, current = [], []
    for line in open(path):
        if line.startswith("#"):
            blocks.append(np.array(current, dtype=float))
            current = []
        elif skip_header and line.startswith("k"):
            continue
        else:
            current.append(line.split())
    return blocks


def read_spin(path):
    rows = [line.split() for line in open(path)][1:]
    return np.array(rows, dtype=float)[:, 1:]


# ----------------------------------------------------------------------------------------
# Comparisons
# ----------------------------------------------------------------------------------------

def groups(labels):
    out = {}
    for i, g in enumerate(labels):
        out.setdefault(int(g), []).append(i)
    return list(out.values())


def reduced(points, reciprocal):
    """Fractional coordinates of k points in the reciprocal basis, modulo 1."""
    coords = np.linalg.lstsq(reciprocal.T, points.T, rcond=None)[0].T
    return coords - np.round(coords)


def run_case(xatu, workdir, name, model, config_lines, flags, args):
    print(f"\n=== {name} ===")
    for mode in ("text", "h5"):
        label = f"{name}_{mode}"
        cfg = os.path.join(workdir, f"{label}_exciton.txt")
        with open(cfg, "w") as f:
            # A block is only stored when the next '#' line is read, so the file ends with one.
            f.write("# label\n" + label + "\n" + config_lines + "#\n")
        out = flags + (["-H"] if mode == "h5" else ["-e"])
        cmd = [xatu, os.path.join(MODELS, model), cfg] + args + out
        res = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        if res.returncode != 0:
            report(f"{name}: xatu {mode} run", False, res.stderr[-2000:] + res.stdout[-2000:])
            return
    text = os.path.join(workdir, f"{name}_text")
    f = h5py.File(os.path.join(workdir, f"{name}_h5.h5"), "r")

    n = int(f.attrs["n_excitons"])
    ex = f["excitons"]
    names = list(ex.keys())
    energies = f["summary/energies"][()]
    deg = groups(f["summary/degeneracy_group"][()])
    width = max(4, len(str(n)))
    report(f"{name}: one group per state, named by energy rank",
           names == [str(i + 1).zfill(width) for i in range(n)] and np.all(np.diff(energies) >= 0),
           f"{len(names)} groups")
    report(f"{name}: per-state eigval = summary table",
           all(ex[k]["eigval"][()] == energies[i] for i, k in enumerate(names)))

    e_text = read_eigval(text + ".eigval")
    report(f"{name}: eigenvalues vs .eigval", e_text.size == n and np.max(np.abs(e_text - energies)) < 1e-6,
           f"max|diff| {np.max(np.abs(e_text - energies)):.1e} eV")

    tda = bool(f.attrs["tamm_dancoff"])
    kpoints = f["kpoints"][()]
    npairs = len(f.attrs["valence_bands"]) * len(f.attrs["conduction_bands"])

    if "-c" in flags:
        basis_text, states_text = read_states(text + ".states")
        basis = f["basis"][()]
        kb = kpoints[basis[:, 2]]
        report(f"{name}: basis vs .states",
               np.array_equal(basis[:, :2], basis_text[:, 3:5].astype(int)) and np.max(np.abs(kb - basis_text[:, :3])) < 1e-6)
        X = np.array([ex[k]["state"][()] for k in names])
        has_y = "state_antiresonant" in ex[names[0]]
        report(f"{name}: anti-resonant block present iff no TDA", has_y == (not tda))
        Y = np.array([ex[k]["state_antiresonant"][()] for k in names]) if has_y else np.zeros_like(X)
        norm = np.sum(np.abs(X) ** 2, axis=1) - np.sum(np.abs(Y) ** 2, axis=1)
        report(f"{name}: states normalised (|X|^2 - |Y|^2 = 1)", np.max(np.abs(norm - 1)) < 1e-10,
               f"max dev {np.max(np.abs(norm - 1)):.1e}")
        worst = 0.0
        for g in deg:
            P_h5 = X[g].T @ X[g].conj()
            P_tx = states_text[g].T @ states_text[g].conj()
            worst = max(worst, np.max(np.abs(P_h5 - P_tx)))
        report(f"{name}: states vs .states (projector per degenerate group)", worst < 1e-6, f"max|dP| {worst:.1e}")
        same = sum(np.max(np.abs(X[i] - states_text[i])) < 1e-6 for i in range(n))
        print(f"      ({same}/{n} states also equal one by one)")
        ev = f["bands/eigenvectors"]
        report(f"{name}: band eigenvectors shape (nk, nbands, norb)",
               ev.shape == (len(kpoints), len(f["bands/indices"]), int(f["system"].attrs["basisdim"])))
        if "-k" in flags:
            kwf_from_x = (np.abs(X) ** 2 - np.abs(Y) ** 2).reshape(n, len(kpoints), npairs).sum(axis=2)
            kwf = np.array([ex[k]["kwf"][()] for k in names])
            report(f"{name}: kwf = sum over pairs of |X|^2 - |Y|^2", np.max(np.abs(kwf - kwf_from_x)) < 1e-14)

    if "-k" in flags:
        kwf = np.array([ex[k]["kwf"][()] for k in names])
        if tda:
            report(f"{name}: kwf sums to 1", np.max(np.abs(kwf.sum(axis=1) - 1)) < 1e-10)
        submesh = int(f.attrs["submesh_factor"])
        blocks = read_blocks(text + ".kwf", skip_header=(submesh != 1))
        dk = np.linalg.norm(kpoints[1] - kpoints[0])
        rec = f["system/reciprocal_lattice"][()]
        red_k = reduced(kpoints, rec)
        worst = 0.0
        # Map every (possibly replicated) text point to its mesh point.
        b0 = blocks[0]
        red_t = reduced(b0[:, :3], rec)
        d = np.abs(red_t[:, None, :] - red_k[None, :, :])
        d = np.minimum(d, 1 - d).max(axis=2)
        idx = np.argmin(d, axis=1)
        report(f"{name}: .kwf points map onto /kpoints", np.max(d[np.arange(len(idx)), idx]) < 1e-5)
        for g in deg:
            t = sum(blocks[i][:, 3] for i in g)
            h = sum(kwf[i] for i in g)[idx] / dk
            worst = max(worst, np.max(np.abs(t - h)) / np.max(np.abs(h)))
        report(f"{name}: kwf vs .kwf (summed per degenerate group, text / |k1-k0|)", worst < 1e-7,
               f"max rel {worst:.1e}")

    if "-r" in flags:
        blocks = read_blocks(text + ".rswf", skip_header=False)
        pos = f["realspace/positions"][()]
        hole = f["realspace"].attrs["hole_position"]
        rswf = np.array([ex[k]["rswf"][()] for k in names])
        report(f"{name}: hole position and sites vs .rswf",
               np.max(np.abs(blocks[0][0, :2] - hole[:2])) < 1e-7 and np.max(np.abs(blocks[0][1:, :2] - pos[:, :2])) < 1e-7)
        worst = 0.0
        for g in deg:
            t = sum(blocks[i][1:, 2] for i in g)
            h = sum(rswf[i] for i in g)
            worst = max(worst, np.max(np.abs(t - h)) / np.max(np.abs(h)))
        report(f"{name}: rswf vs .rswf (summed per degenerate group)", worst < 1e-9, f"max rel {worst:.1e}")

    if "-s" in flags:
        spin_text = read_spin(text + ".spin")
        spin = f["summary/spin"][()]
        report(f"{name}: per-state spin = summary table",
               all(np.array_equal(ex[k]["spin"][()], spin[i]) for i, k in enumerate(names)))
        worst = max(np.max(np.abs(spin_text[g].sum(axis=0) - spin[g].sum(axis=0))) for g in deg)
        report(f"{name}: spin vs .spin (summed per degenerate group)", worst < 1e-6, f"max|diff| {worst:.1e}")
    f.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xatu", default=os.path.join(ROOT, "bin", "xatu"))
    parser.add_argument("--keep", action="store_true", help="keep the work directory")
    args = parser.parse_args()
    xatu = os.path.abspath(args.xatu)
    workdir = tempfile.mkdtemp(prefix="xatu_h5check_")

    hbn = "# ncells\n18\n# bands\n1\n# Dielectric\n1 1 10\n"
    run_case(xatu, workdir, "hbn_tda", "hBN.model", hbn, ["-c", "-k", "-r", "1", "-r", "3"], ["-n", "12"])
    run_case(xatu, workdir, "hbn_full_bse", "hBN.model", hbn + "# tammdancoff\nfalse\n",
             ["-c", "-k", "-r", "0", "-r", "2"], ["-n", "8"])
    run_case(xatu, workdir, "hbn_spinful", "hBN_spinful.model", "# ncells\n12\n# bands\n2\n# Dielectric\n1 1 10\n",
             ["-c", "-k", "-s"], ["-n", "16"])
    run_case(xatu, workdir, "mos2_submesh", "MoS2.model",
             "# ncells\n12\n# submesh\n2\n# shift\n0.6628  -1.1480 0\n# bandList\n-1 0 1 2\n# Dielectric\n1 4 13.55\n",
             ["-c", "-k", "-s"], ["-n", "10"])

    if args.keep:
        print(f"\nwork directory kept: {workdir}")
    else:
        shutil.rmtree(workdir)
    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
