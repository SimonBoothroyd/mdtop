"""Common constants."""

WATER_RES_NAMES: frozenset[str] = frozenset(
    ["HOH", "WAT", "TIP", "TIP2", "TIP3", "TIP4"]
)
"""Common water residue names."""

ION_RES_NAMES: frozenset[str] = frozenset(
    [
        "NA+",
        "NA",
        "K+",
        "K",
        "LI+",
        "LI",
        "CL-",
        "CL",
        "BR-",
        "BR",
        "I-",
        "I",
        "F-",
        "F",
        "MG+2",
        "MG",
        "CA+2",
        "CA",
        "ZN+2",
        "ZN",
        "FE+3",
        "FE+2",
        "FE",
    ]
)
"""Residue names for common ions."""

AMINO_ACID_NAMES: frozenset[str] = frozenset(
    [
        "ACE",
        "NME",
        "NMA",
        "ALA",
        "CYS",
        "ASP",
        "GLU",
        "PHE",
        "GLY",
        "HIS",
        "ILE",
        "LYS",
        "LEU",
        "MET",
        "ASN",
        "PRO",
        "GLN",
        "ARG",
        "SER",
        "THR",
        "VAL",
        "TRP",
        "TYR",
        "CYD",
        "CYZ",
        "HID",
        "HIE",
        "HIP",
    ]
)
"""Common natural amino acid residue names."""

__all__ = ["WATER_RES_NAMES", "ION_RES_NAMES", "AMINO_ACID_NAMES"]
