"""Ein fortsetzbarer Zufallsstrom auf SplitMix64 (Portfolio-Konvention) und das Mischen von Seeds.

`Stream(seed)` liefert nacheinander Gleichverteilte, ganze Zahlen und Normalverteilte. Jeder Aufruf setzt den Strom fort: die Zahlen sind genau die des
skalaren `SplitMix64(seed)` (Test: tests/test_stream.py), nur blockweise vektorisiert. `mix_seed(*teile)` macht aus mehreren ganzen Zahlen (Welt, Population,
Wiederholung, ...) einen 64-Bit-Seed, damit jede Zelle der Messreihen einen eigenen, von der Zahl der Prozesse unabhängigen Strom hat."""

import math

import numpy as np

from wbh_rng import _GAMMA, _MASK, SplitMix64, uniform_block


def mix_seed(*parts):
    """64-Bit-Seed aus ganzen Zahlen: jede Zahl wird in den Zustand gemischt und einmal durch SplitMix64 geschickt."""
    h = 0x243F6A8885A308D3
    for p in parts:
        h = SplitMix64((h ^ (int(p) & _MASK)) & _MASK).next()
    return h


class Stream:
    def __init__(self, seed):
        self.seed = int(seed) & _MASK
        self.pos = 0

    def uniform(self, shape=()):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        count = int(math.prod(shape)) if shape else 1
        block = uniform_block((self.seed + self.pos * _GAMMA) & _MASK, count)
        self.pos += count
        return block.reshape(shape) if shape else block[0]

    def integers(self, high, shape=()):
        """Ganze Zahlen in [0, high), gleichverteilt (floor(u * high))."""
        return np.floor(self.uniform(shape) * high).astype(np.int64)

    def normal(self, shape=()):
        """Standardnormalverteilt (Box-Muller; je zwei Gleichverteilte ein Paar)."""
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        count = int(math.prod(shape)) if shape else 1
        half = (count + 1) // 2
        u = self.uniform((2, half))
        r = np.sqrt(-2.0 * np.log(1.0 - u[0]))
        z = np.concatenate([r * np.cos(2.0 * np.pi * u[1]), r * np.sin(2.0 * np.pi * u[1])])[:count]
        return z.reshape(shape) if shape else z[0]
