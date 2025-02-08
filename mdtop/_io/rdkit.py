"""Convert between RDKit and ``mdtop`` objects."""

import collections
import typing

import numpy
import openmm.unit

from mdtop import Topology
from mdtop._const import AMINO_ACID_NAMES

if typing.TYPE_CHECKING:
    from rdkit import Chem


_EXCLUDED_ATOM_PROPS = {
    "_Name",
    "_CIPCode",
    "_CIPRank",
    "_ChiralityPossible",
    "_MolFileRLabel",
    "_ReactionDegreeChanged",
    "_protected",
    "dummyLabel",
    "molAtomMapNumber",
    "molfileAlias",
    "molFileValue",
    "molFileInversionFlag",
    "molRxnComponent",
    "molRxnRole",
    "smilesSymbol",
    "__computedProps",
    "isImplicit",
}
"""Exclude 'magic' atom properties when setting metadata from RDKit mols."""
_EXCLUDED_BOND_PROPS = {
    "_MolFileBondType",
}
"""Exclude 'magic' bond properties when setting metadata from RDKit mols."""
_EXCLUDED_MOL_PROPS = {
    "MolFileComments",
    "MolFileInfo",
    "_MolFileChiralFlag",
    "_Name",
    "_smilesAtomOutputOrder",
    "_smilesBondOutputOrder",
}
"""Exclude 'magic' molecule properties when setting metadata from RDKit mols."""


def _set_meta(
    obj: typing.Union["Chem.Mol", "Chem.Atom", "Chem.Bond"],
    meta: dict[str, str | float | int | bool | None],
):
    """Set metadata on an RDKit object."""
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, str):
            obj.SetProp(key, value)
        elif isinstance(value, float):
            obj.SetDoubleProp(key, value)
        elif isinstance(value, bool):
            obj.SetBoolProp(key, value)
        elif isinstance(value, int):
            obj.SetIntProp(key, value)
        else:
            raise ValueError(f"Unsupported metadata type for {key}: {value}")


def from_rdkit(mol: "Chem.Mol", residue_name: str = "LIG", chain: str = "") -> Topology:
    """Create a topology from an RDKit molecule.

    Args:
        mol: The RDKit molecule to convert.
        residue_name: The residue name to use for the ligand.
        chain: The chain ID to use for the ligand.

    Returns:
        The converted topology.
    """
    from rdkit import Chem

    mol = Chem.AddHs(mol)
    Chem.Kekulize(mol)

    topology = Topology()
    topology.add_chain(chain)
    residue = topology.add_residue(residue_name, 1, "", topology.chains[0])

    symbol_counter = collections.defaultdict(int)

    for atom_rd in mol.GetAtoms():
        if atom_rd.GetPDBResidueInfo() is not None:
            name = atom_rd.GetPDBResidueInfo().GetName()
        elif atom_rd.HasProp("_Name"):
            name = atom_rd.GetProp("_Name")
        else:
            symbol = atom_rd.GetSymbol()
            symbol_counter[symbol] += 1

            name = f"{symbol}{symbol_counter[symbol]}".ljust(4, "x")

        atom = topology.add_atom(
            name=name,
            atomic_num=atom_rd.GetAtomicNum(),
            formal_charge=atom_rd.GetFormalCharge(),
            serial=atom_rd.GetIdx() + 1,
            residue=residue,
        )
        atom.meta = {
            k: v
            for k, v in atom_rd.GetPropsAsDict().items()
            if k not in _EXCLUDED_ATOM_PROPS
        }

    for bond_rd in mol.GetBonds():
        bond = topology.add_bond(
            idx_1=bond_rd.GetBeginAtomIdx(),
            idx_2=bond_rd.GetEndAtomIdx(),
            order=int(bond_rd.GetBondTypeAsDouble()),
        )
        bond.meta = {
            k: v
            for k, v in bond_rd.GetPropsAsDict().items()
            if k not in _EXCLUDED_BOND_PROPS
        }

    topology.meta = {
        k: v for k, v in mol.GetPropsAsDict().items() if k not in _EXCLUDED_MOL_PROPS
    }

    if mol.GetNumConformers() >= 1:
        xyz = mol.GetConformer().GetPositions()
        topology.xyz = numpy.array(xyz) * openmm.unit.angstrom

    return topology


def to_rdkit(topology: Topology) -> "Chem.Mol":
    """Convert the Topology to an RDKit Mol object.

    Notes:
        * Currently this requires formal charges to be set on the atoms, and
          formal bond orders to be set on the bonds.

    Returns:
        The RDKit Mol object.
    """
    from rdkit import Chem

    mol = Chem.RWMol()
    atoms_rd = []

    for atom in topology.atoms:
        if atom.formal_charge is None:
            raise ValueError("Formal charges must be set on all atoms.")

        atom_rd = Chem.Atom(atom.atomic_num)
        atom_rd.SetFormalCharge(atom.formal_charge)
        atom_rd.SetProp("_Name", atom.name)

        _set_meta(atom_rd, atom.meta)

        res_info = Chem.AtomPDBResidueInfo(
            atom.name,
            atom.serial,
            "",
            atom.residue.name,
            atom.residue.seq_num,
            atom.residue.chain.id,
            atom.residue.insertion_code,
            isHeteroAtom=atom.residue.name not in AMINO_ACID_NAMES,
        )
        atom_rd.SetPDBResidueInfo(res_info)

        atoms_rd.append(mol.AddAtom(atom_rd))

    bond_order_to_type = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    for bond in topology.bonds:
        if bond.order is None:
            raise ValueError("Formal bond orders must be set on all bonds.")
        if bond.order not in bond_order_to_type:
            raise NotImplementedError(f"Bond order {bond.order} is not supported.")

        mol.AddBond(bond.idx_1, bond.idx_2, bond_order_to_type[bond.order])

        bond_rd = mol.GetBondBetweenAtoms(bond.idx_1, bond.idx_2)
        _set_meta(bond_rd, bond.meta)

    if topology.xyz is not None:
        xyz = topology.xyz.value_in_unit(openmm.unit.angstrom)
        conf = Chem.Conformer(len(atoms_rd))

        for idx, pos in enumerate(xyz):
            conf.SetAtomPosition(idx, pos)

        mol.AddConformer(conf, assignId=True)

    _set_meta(mol, topology.meta)

    Chem.SanitizeMol(mol)
    return Chem.Mol(mol)
