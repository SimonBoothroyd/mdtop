"""Topology representations"""

import collections
import copy
import logging
import pathlib
import typing
import warnings

import numpy
import openmm.app

from mdtop._const import AMINO_ACID_NAMES
from mdtop._sel import select

if typing.TYPE_CHECKING:
    from openeye import oechem
    from rdkit import Chem

_LOGGER = logging.getLogger("mdtop")

_RDKIT_EXCLUDED_ATOM_PROPS = {
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
_RDKIT_EXCLUDED_BOND_PROPS = {
    "_MolFileBondType",
}
"""Exclude 'magic' bond properties when setting metadata from RDKit mols."""
_RDKIT_EXCLUDED_MOL_PROPS = {
    "MolFileComments",
    "MolFileInfo",
    "_MolFileChiralFlag",
    "_Name",
    "_smilesAtomOutputOrder",
    "_smilesBondOutputOrder",
}
"""Exclude 'magic' molecule properties when setting metadata from RDKit mols."""


def _set_rd_meta(
    obj: typing.Union["Chem.Mol", "Chem.Atom", "Chem.Bond"],
    meta: dict[str, str | float | int | bool],
):
    """Set metadata on an RDKit object."""
    for key, value in meta.items():
        if isinstance(value, str):
            obj.SetProp(key, value)
        elif isinstance(value, float):
            obj.SetDoubleProp(key, value)
        elif isinstance(value, bool):
            obj.SetBoolProp(key, value)
        elif isinstance(value, int):
            obj.SetIntProp(key, value)
        else:
            raise ValueError(f"Unsupported metadata type: {type(value)}")


def _set_oe_meta(
    obj: typing.Union["oechem.OEMol", "oechem.OEAtomBase", "oechem.OEBondBase"],
    meta: dict[str, str | float | int | bool],
):
    """Set metadata on an OpenEye object."""
    for key, value in meta.items():
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


def _sanitize_array(
    value: numpy.ndarray | openmm.unit.Quantity | None, shape: tuple[int, int]
) -> openmm.unit.Quantity | None:
    """Sanitize an array, converting it to a Quantity with units of Å if necessary."""

    if value is None:
        return
    if isinstance(value, openmm.unit.Quantity):
        value = value.value_in_unit(openmm.unit.angstrom)
    if not isinstance(value, numpy.ndarray):
        value = numpy.array(value)
    if value.shape != shape:
        raise ValueError(f"expected shape {shape}, got {value.shape}")

    return value * openmm.unit.angstrom


class Atom:
    """Represents atoms and virtual sites stored in a topology."""

    __slots__ = (
        "name",
        "atomic_num",
        "formal_charge",
        "serial",
        "meta",
        "_residue",
        "_index",
    )

    def __init__(
        self, name: str, atomic_num: int, formal_charge: int | None, serial: int
    ):
        self.name: str = name
        """The name of the atom."""

        self.atomic_num: int = atomic_num
        """The atomic number, or 0 if this is a virtual site."""
        self.formal_charge: int | None = formal_charge
        """The formal charge on the atom."""

        self.serial: int = serial
        """The index of this atom in its original source (e.g. the serial defined
        in a PDB). This may not be zero-index or contiguous."""

        self.meta: dict[str, str | float | int | bool] = {}
        """Extra metadata associated with the atom."""

        self._residue: typing.Optional["Residue"] = None
        self._index: int | None = None

    @property
    def symbol(self) -> str:
        """The chemical symbol of the atom, or 'X' if this is a virtual site."""
        return (
            "X"
            if self.atomic_num == 0
            else openmm.app.Element.getByAtomicNumber(self.atomic_num).symbol
        )

    @property
    def residue(self) -> typing.Optional["Residue"]:
        """The residue that the atom belongs to."""
        return self._residue

    @property
    def chain(self) -> typing.Optional["Chain"]:
        """The chain that the atom belongs to."""
        return None if self.residue is None else self._residue.chain

    @property
    def index(self) -> int | None:
        """The index of the atom in the parent topology"""
        return self._index

    def __repr__(self):
        return (
            f"Atom("
            f"name='{self.name}', "
            f"atomic_num={self.atomic_num}, "
            f"formal_charge={self.formal_charge}, "
            f"serial={self.serial})"
        )


class Bond:
    """Represents a bond between two atoms."""

    __slots__ = ("_idx_1", "_idx_2", "order", "meta")

    def __init__(self, idx_1: int, idx_2: int, order: int | None):
        self._idx_1 = idx_1
        self._idx_2 = idx_2
        self.order = order
        """The formal bond order"""

        self.meta: dict[str, str | float | int | bool] = {}
        """Extra metadata associated with the bond."""

    @property
    def idx_1(self) -> int:
        """The index of the first atom."""
        return self._idx_1

    @property
    def idx_2(self) -> int:
        """The index of the second atom."""
        return self._idx_2

    def __repr__(self):
        return f"Bond(idx_1={self.idx_1}, idx_2={self.idx_2}, order={self.order})"


class Residue:
    """Represents residues stored in a topology."""

    __slots__ = ("name", "seq_num", "_chain", "_atoms", "_index")

    def __init__(self, name: str, seq_num: int):
        self.name = name
        """The name of the residue."""
        self.seq_num = seq_num
        """The sequence number of the residue."""

        self._chain: typing.Optional["Chain"] = None
        self._atoms: list[Atom] = []

        self._index: int | None = None

    @property
    def chain(self) -> typing.Optional["Chain"]:
        """The chain the residue belongs to (if any)."""
        return self._chain

    @property
    def topology(self) -> typing.Optional["Topology"]:
        """The topology the residue belongs to (if any)."""
        return None if self._chain is None else self._chain.topology

    @property
    def atoms(self) -> tuple[Atom, ...]:
        """The atoms associated with the residue."""
        return tuple(self._atoms)

    @property
    def n_atoms(self) -> int:
        """The number of atoms in the residue."""
        return len(self._atoms)

    @property
    def index(self) -> int | None:
        """The index of the residue in the parent topology"""
        return self._index

    def __repr__(self):
        return f"Residue(name='{self.name}', seq_num={self.seq_num})"


class Chain:
    """Represents chains stored in a topology."""

    __slots__ = ("id", "_topology", "_residues", "_index")

    def __init__(self, id_: str):
        self.id = id_
        """The ID of the chain."""

        self._topology: typing.Optional["Topology"] | None = None
        self._residues: list[Residue] = []

        self._index: int | None = None

    @property
    def topology(self) -> typing.Optional["Topology"]:
        """The topology the chain belongs to (if any)."""
        return self._topology

    @property
    def residues(self) -> tuple[Residue, ...]:
        """The residues associated with the chain."""
        return tuple(self._residues)

    @property
    def n_residues(self) -> int:
        """The number of chains in the chain."""
        return len(self._residues)

    @property
    def atoms(self) -> tuple[Atom, ...]:
        """The atoms associated with the chain."""
        return tuple(atom for residue in self._residues for atom in residue.atoms)

    @property
    def n_atoms(self) -> int:
        """The number of atoms in the chain."""
        return sum(residue.n_atoms for residue in self._residues)

    @property
    def index(self) -> int | None:
        """The index of the chain in the parent topology"""
        return self._index

    def __repr__(self):
        return f"Chain(id='{self.id}')"


class Topology:
    """Represents topological information about a system."""

    __slots__ = ("_chains", "_bonds", "_n_atoms", "_n_residues", "_xyz", "_box", "meta")

    def __init__(self):
        self._chains: list[Chain] = []
        self._bonds: list[Bond] = []

        self._n_atoms: int = 0
        self._n_residues: int = 0

        self._xyz: openmm.unit.Quantity | None = None
        self._box: openmm.unit.Quantity | None = None

        self.meta: dict[str, str | float | int | bool] = {}
        """Extra metadata associated with the atom."""

    @property
    def chains(self) -> tuple[Chain, ...]:
        """The chains associated with the topology."""
        return tuple(self._chains)

    @property
    def n_chains(self) -> int:
        """The number of chains in the topology."""
        return len(self.chains)

    @property
    def residues(self) -> tuple[Residue, ...]:
        """The residues associated with the topology."""
        return tuple(residue for chain in self.chains for residue in chain.residues)

    @property
    def n_residues(self) -> int:
        """The number of residues in the topology."""
        return self._n_residues

    @property
    def atoms(self) -> tuple[Atom, ...]:
        """The atoms associated with the topology."""
        return tuple(atom for residue in self.residues for atom in residue.atoms)

    @property
    def n_atoms(self) -> int:
        """The number of atoms in the topology."""
        return self._n_atoms

    @property
    def bonds(self) -> tuple[Bond, ...]:
        """The bonds associated with the topology."""
        return tuple(self._bonds)

    @property
    def n_bonds(self) -> int:
        """The number of bonds in the topology."""
        return len(self.bonds)

    @property
    def xyz(self) -> openmm.unit.Quantity | None:
        """The coordinates of the atoms in the topology."""
        return self._xyz

    @xyz.setter
    def xyz(self, value: openmm.unit.Quantity | numpy.ndarray | None):
        """Set the coordinates of the atoms in the topology. Assumes units of Å if
        a raw array is passed."""
        self._xyz = _sanitize_array(value, (self.n_atoms, 3))

    @property
    def box(self) -> openmm.unit.Quantity | None:
        """The box vectors of the simulation box."""
        return self._box

    @box.setter
    def box(self, value: openmm.unit.Quantity):
        """Set the box vectors. Assumes units of Å if a raw array is passed."""
        self._box = _sanitize_array(value, (3, 3))

    def add_chain(self, id_: str) -> Chain:
        """Add a new chain to the topology.

        Args:
            id_: The ID of the chain to add.

        Returns:
             The newly created chain.
        """
        chain = Chain(id_=id_)
        chain._topology = self
        chain._index = self.n_chains

        self._chains.append(chain)

        self._n_atoms += chain.n_atoms
        self._n_residues += chain.n_residues

        return chain

    def add_residue(self, name: str, seq_num: int | None, chain: Chain) -> Residue:
        """Add a new residue to the topology.

        Args:
            name: The name of the residue to add
            seq_num: The sequence number of the residue. If ``None``, the index in the
                topology will be used.
            chain: The parent chain to add to.

        Returns:
             The newly created residue.
        """

        if chain.topology != self:
            raise ValueError(f"{chain} does not belong to this topology.")

        seq_num = int(self.n_residues if seq_num is None else seq_num)

        residue = Residue(name=name, seq_num=seq_num)
        residue._chain = chain
        residue._index = self.n_residues

        chain._residues.append(residue)

        self._n_atoms += residue.n_atoms
        self._n_residues += 1

        return residue

    def add_atom(
        self,
        name: str,
        atomic_num: int,
        formal_charge: int | None,
        serial: int | None,
        residue: Residue,
    ) -> Atom:
        """Add a new atom to the topology.

        Args:
            name: The name of the atom to add
            atomic_num: The atomic number of the atom to add, or 0 for virtual sites.
            formal_charge: The formal charge on the atom (if defined).
            serial: The index of this atom in its original source (e.g. the serial
                defined in a PDB), which may not be zero-index or contiguous. If
                ``None``, the index in the topology will be used.
            residue: The parent residue to add to.

        Returns:
            The newly created atom
        """
        if residue.topology != self:
            raise ValueError(f"{residue} does not belong to this topology.")

        serial = int(self.n_atoms if serial is None else serial)

        atom = Atom(
            name=name, atomic_num=atomic_num, formal_charge=formal_charge, serial=serial
        )
        atom._residue = residue
        atom._index = self.n_atoms

        residue._atoms.append(atom)

        self._n_atoms += 1

        return atom

    def add_bond(self, idx_1: int, idx_2: int, order: int | None) -> Bond:
        """Add a new bond to the topology.

        Args:
            idx_1: The index of the first atom.
            idx_2: The index of the second atom.
            order: The formal bond order (if defined).

        Returns:
            The newly created bond.
        """

        if idx_1 >= self.n_atoms:
            raise ValueError("Index 1 is out of range.")
        if idx_2 >= self.n_atoms:
            raise ValueError("Index 2 is out of range.")

        bond = Bond(idx_1=idx_1, idx_2=idx_2, order=order)
        self._bonds.append(bond)

        return bond

    @classmethod
    def from_openmm(cls, topology_omm: openmm.app.Topology) -> "Topology":
        """Create a topology from an OpenMM topology.

        Args:
            topology_omm: The OpenMM topology to convert.

        Returns:
            The converted topology.
        """
        topology = cls()

        for chain_omm in topology_omm.chains():
            chain = topology.add_chain(chain_omm.id)

            for residue_omm in chain_omm.residues():
                residue = topology.add_residue(residue_omm.name, residue_omm.id, chain)

                for atom_omm in residue_omm.atoms():
                    is_v_site = atom_omm.element is None

                    topology.add_atom(
                        atom_omm.name,
                        atom_omm.element.atomic_number if not is_v_site else 0,
                        None if is_v_site else getattr(atom_omm, "formalCharge", None),
                        atom_omm.id,
                        residue,
                    )

        for bond_omm in topology_omm.bonds():
            order = bond_omm.order

            if order is None and bond_omm.type is not None:
                raise NotImplementedError

            topology.add_bond(bond_omm.atom1.index, bond_omm.atom2.index, order)

        if topology_omm.getPeriodicBoxVectors() is not None:
            box = topology_omm.getPeriodicBoxVectors().value_in_unit(
                openmm.unit.angstrom
            )
            topology.box = numpy.array(box) * openmm.unit.angstrom

        return topology

    def to_openmm(self) -> openmm.app.Topology:
        """Convert the topology to an OpenMM topology.

        Returns:
            The OpenMM topology.
        """
        topology_omm = openmm.app.Topology()

        atoms_omm = []

        for chain in self.chains:
            chain_omm = topology_omm.addChain(chain.id)

            for residue in chain.residues:
                residue_omm = topology_omm.addResidue(
                    residue.name, chain_omm, str(residue.seq_num)
                )

                for atom in residue.atoms:
                    element = (
                        None
                        if atom.atomic_num == 0
                        else openmm.app.Element.getByAtomicNumber(atom.atomic_num)
                    )

                    atom_omm = topology_omm.addAtom(
                        atom.name, element, residue_omm, str(atom.serial)
                    )

                    if hasattr(atom_omm, "formalCharge"):
                        atom_omm.formalCharge = atom.formal_charge

                    atoms_omm.append(atom_omm)

        bond_order_to_type = {
            1: openmm.app.Single,
            2: openmm.app.Double,
            3: openmm.app.Triple,
        }

        for bond in self.bonds:
            topology_omm.addBond(
                atoms_omm[bond.idx_1],
                atoms_omm[bond.idx_2],
                bond_order_to_type[bond.order] if bond.order is not None else None,
                bond.order,
            )

        if self.box is not None:
            topology_omm.setPeriodicBoxVectors(self.box)

        return topology_omm

    @classmethod
    def from_rdkit(
        cls, mol: "Chem.Mol", residue_name: str = "LIG", chain: str = ""
    ) -> "Topology":
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

        topology = cls()
        topology.add_chain(chain)
        residue = topology.add_residue(residue_name, 1, topology.chains[0])

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
                if k not in _RDKIT_EXCLUDED_ATOM_PROPS
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
                if k not in _RDKIT_EXCLUDED_BOND_PROPS
            }

        topology.meta = {
            k: v
            for k, v in mol.GetPropsAsDict().items()
            if k not in _RDKIT_EXCLUDED_MOL_PROPS
        }

        if mol.GetNumConformers() >= 1:
            xyz = mol.GetConformer().GetPositions()
            topology.xyz = numpy.array(xyz) * openmm.unit.angstrom

        return topology

    def to_rdkit(self) -> "Chem.Mol":
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

        for atom in self.atoms:
            if atom.formal_charge is None:
                raise ValueError("Formal charges must be set on all atoms.")

            atom_rd = Chem.Atom(atom.atomic_num)
            atom_rd.SetFormalCharge(atom.formal_charge)
            atom_rd.SetProp("_Name", atom.name)

            _set_rd_meta(atom_rd, atom.meta)

            res_info = Chem.AtomPDBResidueInfo(
                atom.name,
                atom.serial,
                "",
                atom.residue.name,
                atom.residue.seq_num,
                atom.residue.chain.id,
                isHeteroAtom=atom.residue.name not in AMINO_ACID_NAMES,
            )
            atom_rd.SetPDBResidueInfo(res_info)

            atoms_rd.append(mol.AddAtom(atom_rd))

        bond_order_to_type = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
        }
        for bond in self.bonds:
            if bond.order is None:
                raise ValueError("Formal bond orders must be set on all bonds.")
            if bond.order not in bond_order_to_type:
                raise NotImplementedError(f"Bond order {bond.order} is not supported.")

            mol.AddBond(bond.idx_1, bond.idx_2, bond_order_to_type[bond.order])

            bond_rd = mol.GetBondBetweenAtoms(bond.idx_1, bond.idx_2)
            _set_rd_meta(bond_rd, bond.meta)

        if self.xyz is not None:
            xyz = self.xyz.value_in_unit(openmm.unit.angstrom)
            conf = Chem.Conformer(len(atoms_rd))

            for idx, pos in enumerate(xyz):
                conf.SetAtomPosition(idx, pos)

            mol.AddConformer(conf, assignId=True)

        _set_rd_meta(mol, self.meta)

        Chem.SanitizeMol(mol)
        return Chem.Mol(mol)

    @classmethod
    def from_openeye(cls, mol: "oechem.OEMol") -> "Topology":  # pragma: no cover
        """Create a topology from an OpenEye molecule.

        Args:
            mol: The RDKit molecule to convert.

        Returns:
            The converted topology.
        """
        from openeye import oechem

        if oechem.OEHasImplicitHydrogens(mol):
            assert oechem.OEAddExplicitHydrogens(mol)

        atoms_by_chain = collections.defaultdict(lambda: collections.defaultdict(list))

        res_num_to_name = {}
        res_num_to_chain_id = {}

        topology = cls()

        for i, atom_oe in enumerate(mol.GetAtoms()):
            atomic_num = atom_oe.GetAtomicNum()
            formal_charge = atom_oe.GetFormalCharge()

            name = atom_oe.GetName()

            res_name = ""
            res_num = 0

            chain_id = ""

            serial = None

            if oechem.OEHasResidue(atom_oe):
                residue_oe: oechem.OEResidue = oechem.OEAtomGetResidue(atom_oe)

                res_name = residue_oe.GetName()
                res_num = residue_oe.GetResidueNumber()

                serial = residue_oe.GetSerialNumber()

                chain_id = residue_oe.GetChainID()

            res_num_to_name[res_num] = res_name
            res_num_to_chain_id[res_num] = chain_id

            assert (
                res_num not in res_num_to_name
                or res_name == res_num_to_name[res_num]
                or chain_id not in res_num_to_chain_id
                or chain_id != res_num_to_chain_id[res_num]
            )

            meta = {
                oechem.OEGetTag(pair.GetTag()): pair.GetData()
                for pair in atom_oe.GetDataIter()
                if len(oechem.OEGetTag(pair.GetTag())) > 0
            }

            atoms_by_chain[chain_id][res_num].append(
                (i, name, atomic_num, formal_charge, serial, meta)
            )

        atom_idx_old_to_new = {}

        for chain_id, residues in atoms_by_chain.items():
            chain = topology.add_chain(chain_id)

            for res_num, atoms in residues.items():
                residue = topology.add_residue(res_num_to_name[res_num], res_num, chain)

                for (
                    idx_old,
                    name,
                    atomic_num,
                    formal_charge,
                    serial,
                    meta,
                ) in atoms:
                    atom = topology.add_atom(
                        name, atomic_num, formal_charge, serial, residue
                    )
                    atom.meta = meta

                    atom_idx_old_to_new[idx_old] = atom.index

        if any(idx_old != idx_new for idx_old, idx_new in atom_idx_old_to_new.items()):
            _LOGGER.warning("Atoms were re-ordered so residues are contiguous.")

        for bond_oe in mol.GetBonds():
            idx_1 = atom_idx_old_to_new[bond_oe.GetBgnIdx()]
            idx_2 = atom_idx_old_to_new[bond_oe.GetEndIdx()]

            order = bond_oe.GetOrder()

            bond = topology.add_bond(idx_1, idx_2, order)
            bond.meta = {
                oechem.OEGetTag(pair.GetTag()): pair.GetData()
                for pair in bond_oe.GetDataIter()
                if len(oechem.OEGetTag(pair.GetTag())) > 0
            }

        topology.meta = {
            oechem.OEGetTag(pair.GetTag()): pair.GetData()
            for pair in mol.GetDataIter()
            if len(oechem.OEGetTag(pair.GetTag())) > 0
        }

        if mol.NumConfs() == 1:
            coords_dict = mol.GetCoords()
            topology.xyz = numpy.array([coords_dict[i] for i in range(mol.NumAtoms())])
        elif mol.NumConfs() > 1:
            raise NotImplementedError("Multiple conformers are not supported.")

        return topology

    def to_openeye(self) -> "oechem.OEMol":  # pragma: no cover
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

        for atom in self.atoms:
            if atom.formal_charge is None:
                raise ValueError("Formal charges must be set on all atoms.")

            atom_oe: oechem.OEAtomBase = mol.NewAtom(atom.atomic_num)
            atom_oe.SetFormalCharge(atom.formal_charge)
            atom_oe.SetName(atom.name)

            _set_oe_meta(atom_oe, atom.meta)

            res_info = oechem.OEAtomGetResidue(atom_oe)
            res_info.SetName(atom.residue.name)
            res_info.SetResidueNumber(atom.residue.seq_num)
            res_info.SetChainID(atom.residue.chain.id)
            res_info.SetSerialNumber(atom.serial)

            oechem.OEAtomSetResidue(atom_oe, res_info)

            atoms_oe.append(atom_oe)

        for bond in self.bonds:
            if bond.order is None:
                raise ValueError("Formal bond orders must be set on all bonds.")

            bond_oe = mol.NewBond(
                atoms_oe[bond.idx_1], atoms_oe[bond.idx_2], bond.order
            )
            _set_oe_meta(bond_oe, bond.meta)

        if self.xyz is not None:
            xyz = self.xyz.value_in_unit(openmm.unit.angstrom)

            coords = oechem.OEFloatArray(3 * self.n_atoms)

            for idx, pos in enumerate(xyz):
                coords[idx * 3] = pos[0]
                coords[idx * 3 + 1] = pos[1]
                coords[idx * 3 + 2] = pos[2]

            mol.DeleteConfs()
            mol.NewConf(coords)

        _set_oe_meta(mol, self.meta)

        oechem.OEFindRingAtomsAndBonds(mol)
        return mol

    @classmethod
    def _from_pdb(cls, path: pathlib.Path) -> "Topology":
        """Load the topology from a PDB file."""
        pdb = openmm.app.PDBFile(str(path))

        topology = cls.from_openmm(pdb.topology)

        xyz = pdb.positions.value_in_unit(openmm.unit.angstrom)
        topology.xyz = numpy.array(xyz) * openmm.unit.angstrom

        return topology

    @classmethod
    def from_file(cls, path: pathlib.Path | str) -> "Topology":
        """Load the topology from a file.

        Notes:
            * Currently PDB, SDF, and MOL2 files are supported.

        Args:
            path: The path to the file to load.

        Returns:
            The loaded topology.
        """
        path = pathlib.Path(path)

        if path.suffix.lower() == ".pdb":
            return cls._from_pdb(path)
        elif path.suffix.lower() in {".mol", ".sdf"}:
            from rdkit import Chem

            mol = Chem.MolFromMolFile(str(path), removeHs=False)
            return cls.from_rdkit(mol)
        elif path.suffix.lower() == ".mol2":
            from rdkit import Chem

            mol = Chem.MolFromMol2File(str(path), removeHs=False)
            return cls.from_rdkit(mol)

        raise NotImplementedError(f"{path.suffix} files are not supported.")

    def to_file(self, path: pathlib.Path | str):
        """Write the topology to a file.

        Notes:
            * Currently PDB, MOL, and SDF files are supported.
            * SDF / MOL writing requires that all atoms have formal charges set, and
              all bonds have formal bond orders set, as reading and writing is via
              RDKit.
            * Not all metadata will be preserved when writing to files, including
              residue and chain information.

        Args:
            path: The path to write the topology to.
        """
        path = pathlib.Path(path)

        if path.suffix.lower() == ".pdb":
            xyz = self.xyz if self.xyz is not None else numpy.zeros((self.n_atoms, 3))
            openmm.app.PDBFile.writeFile(self.to_openmm(), xyz, str(path))
            return
        elif path.suffix.lower() in {".mol", ".sdf"}:
            from rdkit import Chem

            with Chem.SDWriter(str(path)) as writer:
                writer.write(self.to_rdkit())
            return

        raise NotImplementedError(f"{path.suffix} files are not supported.")

    def _select_amber(self, expr: str) -> numpy.ndarray | None:
        try:
            import parmed.amber
        except ImportError:
            return None

        try:
            topology_pmd = parmed.openmm.load_topology(self.to_openmm())
            result = parmed.amber.AmberMask(topology_pmd, expr).Selection()

            warnings.warn(
                "Using an Amber style selection mask is deprecated. Please use the "
                "PyMol style selection language instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            return numpy.array(tuple(i for i, matches in enumerate(result) if matches))
        except parmed.exceptions.MaskError:
            return

    def select(self, expr: str) -> numpy.ndarray:
        """Select atoms from the topology using a selection expression.

        The selection expression should be expressed in terms of the PyMol
        selection language. For example, to select all atoms in chain A:

        ```python
        selection = topology.select("chain A")
        ```

        or all atoms within 5 Å of the ligand:

        ```python
        selection = topology.select("all within 5 of resn LIG")
        ```

        Notes:
            An Amber-style selection mask can also be used, but this is deprecated
            and will be removed in a future version.

        Args:
            expr: The selection expression.
        """
        idxs = self._select_amber(expr)

        if idxs is not None:
            return idxs

        return select(self, expr)

    def subset(self, idxs: typing.Iterable[int]) -> "Topology":
        """Create a subset of the topology.

        Args:
            idxs: The indices of the atoms to include in the subset. Note the order of
                the atoms in the subset will be the same as in the original topology,
                regardless of the order of the indices.

        Returns:
            The subset of the topology.
        """
        idxs = numpy.array(idxs)
        idxs_unique = set(idxs)

        if len(idxs_unique) != len(idxs):
            raise ValueError("Indices are not unique.")

        subset = Topology()

        idx_old_to_new = {}

        for chain in self.chains:
            has_chain = any(
                atom.index in idxs_unique
                for residue in chain.residues
                for atom in residue.atoms
            )

            if not has_chain:
                continue

            chain_new = subset.add_chain(chain.id)

            for residue in chain.residues:
                has_residue = any(atom.index in idxs_unique for atom in residue.atoms)

                if not has_residue:
                    continue

                residue_new = subset.add_residue(
                    residue.name, residue.seq_num, chain_new
                )

                for atom in residue.atoms:
                    if atom.index not in idxs_unique:
                        continue

                    atom_new = subset.add_atom(
                        atom.name,
                        atom.atomic_num,
                        atom.formal_charge,
                        atom.serial,
                        residue_new,
                    )
                    idx_old_to_new[atom.index] = atom_new.index

        for bond in self.bonds:
            if bond.idx_1 not in idxs_unique or bond.idx_2 not in idxs_unique:
                continue

            subset.add_bond(
                idx_old_to_new[bond.idx_1], idx_old_to_new[bond.idx_2], bond.order
            )

        subset.box = self.box
        subset.xyz = None if self.xyz is None else self.xyz[idxs]

        return subset

    @classmethod
    def merge(cls, *topologies: "Topology") -> "Topology":
        """Merge multiple topologies.

        Notes:
            * The box vectors of the first topology will be used.
            * Topologies without coordinates will be treated as if they have all zero
              coordinates.

        Args:
            topologies: The topologies to merge together.

        Returns:
            The merged topology.
        """

        if len(topologies) == 0:
            return cls()

        merged = copy.deepcopy(topologies[0])

        for topology in topologies[1:]:
            merged += topology

        return merged

    def __iadd__(self, other: "Topology"):
        if not isinstance(other, Topology):
            raise TypeError("Can only combine topologies.")

        idx_offset = self.n_atoms

        xyz_a = (
            None if self.xyz is None else self.xyz.value_in_unit(openmm.unit.angstrom)
        )
        xyz_b = (
            None if other.xyz is None else other.xyz.value_in_unit(openmm.unit.angstrom)
        )

        if xyz_a is None and xyz_b is not None:
            xyz_a = numpy.zeros((self.n_atoms, 3), dtype=float)
        if xyz_b is None and xyz_a is not None:
            xyz_b = numpy.zeros((other.n_atoms, 3), dtype=float)

        self._chains.extend(other.chains)
        self._n_atoms += other.n_atoms
        self._n_residues += other.n_residues

        for idx, atom in enumerate(self.atoms):
            atom._index = idx
        for idx, residue in enumerate(self.residues):
            residue._index = idx

        if xyz_a is not None and xyz_b is not None:
            self.xyz = numpy.vstack((xyz_a, xyz_b)) * openmm.unit.angstrom

        for bond in other.bonds:
            self.add_bond(bond.idx_1 + idx_offset, bond.idx_2 + idx_offset, bond.order)

        return self

    def __add__(self, other: "Topology") -> "Topology":
        combined = copy.deepcopy(self)
        combined += other
        return combined

    def __getitem__(self, item) -> "Topology":
        if isinstance(item, int):
            idxs = numpy.array([item])
        elif isinstance(item, str):
            idxs = self.select(item)
        elif isinstance(item, slice):
            idxs = numpy.arange(self.n_atoms)[item]
        elif isinstance(item, (tuple, list)):
            idxs = numpy.array(item)
        elif isinstance(item, numpy.ndarray):
            idxs = item
        else:
            raise TypeError(f"Invalid index type: {type(item)}")

        return self.subset(idxs.flatten())

    def __repr__(self):
        return (
            f"Topology("
            f"n_chains={self.n_chains}, "
            f"n_residues={self.n_residues}, "
            f"n_atoms={self.n_atoms})"
        )
