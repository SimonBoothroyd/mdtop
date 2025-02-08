"""Convert between OpenEye and ``mdtop`` objects."""

import collections
import logging
import typing

import numpy
import openmm.unit

from mdtop import Topology, box_from_geometry, box_to_geometry

if typing.TYPE_CHECKING:
    from openeye import oechem

_LOGGER = logging.getLogger("mdtop")


_EXCLUDED_ATOM_PROPS = set()
"""Exclude 'magic' atom properties when setting metadata from OpenEye mols."""
_EXCLUDED_MOL_PROPS = {
    "OE_unit_cell",
    "OE_spacegroup",
    "OE_spacegroup_name",
    "OE_symm_enabled",
}


_OEObj = typing.Union[
    "oechem.OEMol", "oechem.OEConfBase", "oechem.OEAtomBase", "oechem.OEBondBase"
]


def _set_meta(obj: _OEObj, meta: dict[str, str | float | int | bool | None]):
    """Set metadata on an OpenEye object."""
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, str):
            obj.SetStringData(key, value)
        elif isinstance(value, float):
            obj.SetDoubleData(key, value)
        elif isinstance(value, bool):
            obj.SetBoolData(key, value)
        elif isinstance(value, int):
            obj.SetIntData(key, value)
        else:
            raise ValueError(f"Unsupported metadata type: {type(value)}")


def _extract_meta(
    obj: _OEObj, exclude: set[str]
) -> dict[str, str | float | int | bool]:
    """Extract metadata from an OpenEye object."""
    from openeye import oechem

    meta = {}

    for pair in obj.GetDataIter():
        tag = oechem.OEGetTag(pair.GetTag())
        if len(tag) == 0 or tag in exclude:
            continue
        meta[tag] = pair.GetData()

    return meta


def _find_atoms(mol: "oechem.OEMol"):
    """Find atoms in the OpenEye molecule, and partition them by chain and residue."""
    from openeye import oechem

    atoms_by_chain = collections.defaultdict(lambda: collections.defaultdict(list))
    res_lookup = {}

    for i, atom_oe in enumerate(mol.GetAtoms()):
        atomic_num = atom_oe.GetAtomicNum()
        formal_charge = atom_oe.GetFormalCharge()

        name = atom_oe.GetName()

        res_name, res_num, chain_id = "", 0, ""
        insertion_code, serial = "", None

        if oechem.OEHasResidue(atom_oe):
            res_oe: oechem.OEResidue = oechem.OEAtomGetResidue(atom_oe)

            res_name, res_num = res_oe.GetName(), res_oe.GetResidueNumber()

            insertion_code = res_oe.GetInsertCode()
            serial = res_oe.GetSerialNumber()

            chain_id = res_oe.GetChainID()

        res_key = (res_name, res_num, insertion_code)

        assert res_key not in res_lookup or res_name == res_lookup[res_key]
        res_lookup[(chain_id, res_num, insertion_code)] = res_name

        meta = _extract_meta(atom_oe, _EXCLUDED_ATOM_PROPS)
        atoms_by_chain[chain_id][(res_num, insertion_code)].append(
            (i, name, atomic_num, formal_charge, serial, meta)
        )

    return atoms_by_chain, res_lookup


def from_openeye(mol: "oechem.OEMol") -> "Topology":
    """Create a topology from an OpenEye molecule.

    Args:
        mol: The RDKit molecule to convert.

    Returns:
        The converted topology.
    """
    from openeye import oechem

    if oechem.OEHasImplicitHydrogens(mol):
        assert oechem.OEAddExplicitHydrogens(mol)

    atoms_by_chain, res_lookup = _find_atoms(mol)
    atom_idx_old_to_new = {}

    topology = Topology()

    for chain_id, residues in atoms_by_chain.items():
        chain = topology.add_chain(chain_id)

        for (res_num, insertion_code), atoms in residues.items():
            res_name = res_lookup[(chain_id, res_num, insertion_code)]
            res = topology.add_residue(res_name, res_num, insertion_code, chain)

            for idx_old, name, atomic_num, formal_charge, serial, meta in atoms:
                atom = topology.add_atom(name, atomic_num, formal_charge, serial, res)
                atom.meta = meta

                atom_idx_old_to_new[idx_old] = atom.index

    if any(idx_old != idx_new for idx_old, idx_new in atom_idx_old_to_new.items()):
        _LOGGER.warning("Atoms were re-ordered so residues are contiguous.")

    for bond_oe in mol.GetBonds():
        idx_1 = atom_idx_old_to_new[bond_oe.GetBgnIdx()]
        idx_2 = atom_idx_old_to_new[bond_oe.GetEndIdx()]

        order = bond_oe.GetOrder()

        bond = topology.add_bond(idx_1, idx_2, order)
        bond.meta = _extract_meta(bond_oe, set())

    topology.meta = _extract_meta(mol, _EXCLUDED_MOL_PROPS)

    if oechem.OEHasCrystalSymmetry(mol):
        symm = oechem.OECrystalSymmetryParams()
        oechem.OEGetCrystalSymmetry(symm, mol)

        topology.box = box_from_geometry(
            symm.GetA(),
            symm.GetB(),
            symm.GetC(),
            symm.GetAlpha(),
            symm.GetBeta(),
            symm.GetGamma(),
        )

    if mol.NumConfs() == 1:
        coords_dict = mol.GetCoords()
        topology.xyz = numpy.array([coords_dict[i] for i in range(mol.NumAtoms())])
    elif mol.NumConfs() > 1:
        raise NotImplementedError("Multiple conformers are not supported.")

    return topology


def to_openeye(topology: Topology) -> "oechem.OEMol":
    """Convert the Topology to an OpenEye molecule object.

    Notes:
        * Currently this requires formal charges to be set on the atoms, and
          formal bond orders to be set on the bonds.

    Returns:
        The OpenEye molecule.
    """
    from openeye import oechem

    mol = oechem.OEMol()
    atoms_oe = []

    for atom in topology.atoms:
        if atom.formal_charge is None:
            raise ValueError("Formal charges must be set on all atoms.")

        atom_oe: oechem.OEAtomBase = mol.NewAtom(atom.atomic_num)
        atom_oe.SetFormalCharge(atom.formal_charge)
        atom_oe.SetName(atom.name)

        _set_meta(atom_oe, atom.meta)

        res_info = oechem.OEAtomGetResidue(atom_oe)
        res_info.SetName(atom.residue.name)
        res_info.SetResidueNumber(atom.residue.seq_num)
        res_info.SetChainID(atom.residue.chain.id)
        if atom.serial is not None:
            res_info.SetSerialNumber(atom.serial)
        res_info.SetInsertCode(atom.residue.insertion_code)

        oechem.OEAtomSetResidue(atom_oe, res_info)

        atoms_oe.append(atom_oe)

    for bond in topology.bonds:
        if bond.order is None:
            raise ValueError("Formal bond orders must be set on all bonds.")

        bond_oe = mol.NewBond(atoms_oe[bond.idx_1], atoms_oe[bond.idx_2], bond.order)
        _set_meta(bond_oe, bond.meta)

    if topology.xyz is not None:
        xyz = topology.xyz.value_in_unit(openmm.unit.angstrom)

        coords = oechem.OEFloatArray(3 * topology.n_atoms)

        for idx, pos in enumerate(xyz):
            coords[idx * 3] = pos[0]
            coords[idx * 3 + 1] = pos[1]
            coords[idx * 3 + 2] = pos[2]

        mol.DeleteConfs()
        mol.NewConf(coords)

    if topology.box is not None:
        a, b, c, alpha, beta, gamma = box_to_geometry(topology.box)
        assert oechem.OESetCrystalSymmetry(
            mol, a, b, c, alpha, beta, gamma, oechem.OEGetSpaceGroupNumber("P1"), True
        )

    _set_meta(mol, topology.meta)

    oechem.OEFindRingAtomsAndBonds(mol)
    return mol
