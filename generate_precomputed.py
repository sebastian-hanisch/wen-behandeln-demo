"""Erzeugt `precomputed_pools.npz`: je (Standorttyp, Regel, Fensterlänge) ein Pool von POOL_SIZE unabhängigen Standort-Fenstern.

    _venvs/test/Scripts/python.exe generate_precomputed.py        # etwa eine Minute, nutzt alle Kerne

Jeder Pool ist ein eigener SplitMix64-Strom, dessen Seed nur vom Schlüssel abhängt (wbh_pools.pool_seed): das Ergebnis hängt weder von der Zahl der
Prozesse noch von der Plattform ab. Die Servicegrade werden als uint16 (Wert * 65535) gespeichert."""

import multiprocessing as mp
import sys
import time

import numpy as np

import wbh_pools as P


def _one(key):
    return P.quantize(P.generate_pool(key))


def main():
    t0 = time.time()
    keys = P.all_keys()
    with mp.Pool() as pool:
        rows = pool.map(_one, keys, chunksize=4)
    np.savez_compressed(P.POOLS_PATH, keys=np.array([P.key_name(k) for k in keys]), values=np.stack(rows))
    print(f"{P.POOLS_PATH.name}: {len(keys)} Pools à {rows[0].shape[0]} Fenster, {time.time() - t0:.0f} s, {P.POOLS_PATH.stat().st_size / 1024:.0f} KB",
          file=sys.stderr)


if __name__ == "__main__":
    main()
