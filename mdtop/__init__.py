"""Store topological information about a system to simulate."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("mdtop")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

from mdtop._const import AMINO_ACID_NAMES, ION_RES_NAMES, WATER_RES_NAMES
from mdtop._top import Atom, Bond, Chain, Residue, Topology

__all__ = [
    "__version__",
    "ION_RES_NAMES",
    "WATER_RES_NAMES",
    "AMINO_ACID_NAMES",
    "Atom",
    "Bond",
    "Chain",
    "Residue",
    "Topology",
]
