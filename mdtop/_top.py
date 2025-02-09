"""Topology representations"""

import collections
import copy
import logging
import pathlib
import typing
import warnings

import numpy
import openmm.app

from mdtop._sel import select

if typing.TYPE_CHECKING:
    from openeye import oechem
    from rdkit import Chem

_LOGGER = logging.getLogger("mdtop")


def _sanitize_array(
    value: numpy.ndarray | openmm.unit.Quantity | None, shape: tuple[int, int]
) -> openmm.unit.Quantity | None:
    """Sanitize an array, converting it to a Quantity with units of Å if necessary."""

    if value is None:
        return None
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

    __slots__ = ("name", "seq_num", "insertion_code", "_chain", "_atoms", "_index")

    def __init__(self, name: str, seq_num: int, insertion_code: str = ""):
        self.name = name
        """The name of the residue."""
        self.seq_num = seq_num
        """The sequence number of the residue."""
        self.insertion_code = insertion_code
        """The insertion code of the residue."""

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
        return (
            f"Residue(name='{self.name}', "
            f"seq_num={self.seq_num} "
            f"insertion_code='{self.insertion_code}')"
        )


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

    def add_residue(
        self, name: str, seq_num: int | None, insertion_code: str, chain: Chain
    ) -> Residue:
        """Add a new residue to the topology.

        Args:
            name: The name of the residue to add
            seq_num: The sequence number of the residue. If ``None``, the index in the
                topology will be used.
            insertion_code: The insertion code of the residue.
            chain: The parent chain to add to.

        Returns:
             The newly created residue.
        """

        if chain.topology != self:
            raise ValueError(f"{chain} does not belong to this topology.")

        seq_num = int(self.n_residues if seq_num is None else seq_num)

        residue = Residue(name=name, seq_num=seq_num, insertion_code=insertion_code)
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
        from mdtop._io.openmm import from_openmm

        return from_openmm(topology_omm)

    def to_openmm(self) -> openmm.app.Topology:
        """Convert the topology to an OpenMM topology.

        Returns:
            The OpenMM topology.
        """
        from mdtop._io.openmm import to_openmm

        return to_openmm(self)

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
        from mdtop._io.rdkit import from_rdkit

        return from_rdkit(mol, residue_name, chain)

    def to_rdkit(self) -> "Chem.Mol":
        """Convert the Topology to an RDKit Mol object.

        Notes:
            * Currently this requires formal charges to be set on the atoms, and
              formal bond orders to be set on the bonds.

        Returns:
            The RDKit Mol object.
        """
        from mdtop._io.rdkit import to_rdkit

        return to_rdkit(self)

    @classmethod
    def from_openeye(cls, mol: "oechem.OEMol") -> "Topology":  # pragma: no cover
        """Create a topology from an OpenEye molecule.

        Args:
            mol: The RDKit molecule to convert.

        Returns:
            The converted topology.
        """

        from mdtop._io.openeye import from_openeye

        return from_openeye(mol)

    def to_openeye(self) -> "oechem.OEMol":  # pragma: no cover
        """Convert the Topology to an OpenEye molecule object.

        Notes:
            * Currently this requires formal charges to be set on the atoms, and
              formal bond orders to be set on the bonds.

        Returns:
            The OpenEye molecule.
        """
        from mdtop._io.openeye import to_openeye

        return to_openeye(self)

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
                    residue.name, residue.seq_num, "", chain_new
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
            if bond.idx_1 not in idx_old_to_new or bond.idx_2 not in idx_old_to_new:
                continue

            subset.add_bond(
                idx_old_to_new[bond.idx_1], idx_old_to_new[bond.idx_2], bond.order
            )

        subset.box = self.box
        subset.xyz = None if self.xyz is None else self.xyz[idxs]

        return subset

    def split(self) -> list["Topology"]:
        """Split the topology into multiple topologies, such that bound atoms are
        in the same topology.

        This is useful, for example, when your topology contains multiple ligands,
        and you want to split them into separate topologies.

        Returns:
            An list of the split topologies.
        """

        open_list = set(range(self.n_atoms))
        closed_list = set()

        neighs = {idx: set() for idx in range(self.n_atoms)}

        for bond in self.bonds:
            neighs[bond.idx_1].add(bond.idx_2)
            neighs[bond.idx_2].add(bond.idx_1)

        frags = []

        while len(open_list) > 0:
            idx = open_list.pop()

            queue = collections.deque([idx])
            frag = []

            closed_list.add(idx)

            while len(queue) > 0:
                idx = queue.popleft()
                frag.append(idx)

                for neigh in neighs[idx]:
                    if neigh in closed_list:
                        continue

                    queue.append(neigh)
                    closed_list.add(neigh)

            frags.append(frag)
            open_list -= set(frag)

        return [self.subset(frag) for frag in frags]

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

        self._chains.extend(copy.deepcopy(other.chains))
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
