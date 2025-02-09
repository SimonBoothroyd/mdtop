import numpy
import openmm
import openmm.app
import parmed
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from mdtop import Atom, Bond, Chain, Residue, Topology, box_to_geometry


def compare_topologies(top_a: Topology, top_b: Topology):
    assert top_a.n_chains == top_b.n_chains
    assert top_a.n_residues == top_b.n_residues
    assert top_a.n_atoms == top_b.n_atoms
    assert top_a.n_bonds == top_b.n_bonds
    assert top_a.meta == top_b.meta

    for chain_orig, chain_rt in zip(top_a.chains, top_b.chains, strict=True):
        assert chain_orig.id == chain_rt.id
        assert chain_orig.n_residues == chain_rt.n_residues
        assert chain_orig.n_atoms == chain_rt.n_atoms

        for residue_orig, residue_rt in zip(
            chain_orig.residues, chain_rt.residues, strict=True
        ):
            assert residue_orig.name == residue_rt.name
            assert residue_orig.seq_num == residue_rt.seq_num
            assert residue_orig.n_atoms == residue_rt.n_atoms

            for atom_orig, atom_rt in zip(
                residue_orig.atoms, residue_rt.atoms, strict=True
            ):
                assert atom_orig.name == atom_rt.name
                assert atom_orig.atomic_num == atom_rt.atomic_num
                assert atom_orig.formal_charge == atom_rt.formal_charge
                assert atom_orig.serial == atom_rt.serial
                assert atom_orig.meta == atom_rt.meta

    for bond_orig, bond_rt in zip(top_a.bonds, top_b.bonds, strict=True):
        assert bond_orig.idx_1 == bond_rt.idx_1
        assert bond_orig.idx_2 == bond_rt.idx_2
        assert bond_orig.order == bond_rt.order
        assert bond_orig.meta == bond_rt.meta

    assert (top_a.xyz is None) == (top_b.xyz is None)

    if top_a.xyz is not None:
        assert numpy.allclose(
            top_a.xyz.value_in_unit(openmm.unit.angstrom),
            top_b.xyz.value_in_unit(openmm.unit.angstrom),
        )

    assert (top_a.box is None) == (top_b.box is None)

    if top_a.box is not None:
        assert numpy.allclose(
            top_a.box.value_in_unit(openmm.unit.angstrom),
            top_b.box.value_in_unit(openmm.unit.angstrom),
        )


def test_atom_properties():
    atom = Atom(name="C", atomic_num=6, formal_charge=0, serial=1)
    assert atom.name == "C"
    assert atom.atomic_num == 6
    assert atom.formal_charge == 0
    assert atom.serial == 1
    assert atom.symbol == "C"


def test_bond_properties():
    bond = Bond(idx_1=0, idx_2=1, order=1)
    assert bond.idx_1 == 0
    assert bond.idx_2 == 1
    assert bond.order == 1


def test_residue_properties():
    residue = Residue(name="ALA", seq_num=1)
    assert residue.name == "ALA"
    assert residue.seq_num == 1
    assert residue.n_atoms == 0
    assert len(residue.atoms) == 0

    atom = Atom(name="C", atomic_num=6, formal_charge=0, serial=1)
    residue._atoms.append(atom)
    assert residue.n_atoms == 1


def test_chain_properties():
    chain = Chain(id_="A")
    assert chain.id == "A"
    assert chain.n_residues == 0
    assert len(chain.residues) == 0
    assert chain.n_atoms == 0
    assert len(chain.atoms) == 0
    assert chain.index is None

    residue = Residue(name="ALA", seq_num=1)
    chain._residues.append(residue)
    assert chain.n_residues == 1

    atom = Atom(name="C", atomic_num=6, formal_charge=0, serial=1)
    residue._atoms.append(atom)
    assert chain.n_atoms == 1


def test_topology_add_chain():
    topology = Topology()
    chain = topology.add_chain(id_="A")
    assert len(topology.chains) == 1
    assert chain.id == "A"
    assert topology.n_chains == 1


def test_topology_add_residue():
    topology = Topology()
    chain = topology.add_chain(id_="A")
    residue = topology.add_residue(
        name="ALA", seq_num=1, insertion_code="", chain=chain
    )
    assert len(chain.residues) == 1
    assert residue.name == "ALA"
    assert residue.seq_num == 1
    assert topology.n_residues == 1


def test_topology_add_atom():
    topology = Topology()
    chain = topology.add_chain(id_="A")
    residue = topology.add_residue(
        name="ALA", seq_num=1, insertion_code="", chain=chain
    )
    atom = topology.add_atom(
        name="C", atomic_num=6, formal_charge=0, serial=1, residue=residue
    )
    assert len(residue.atoms) == 1
    assert atom.name == "C"
    assert topology.n_atoms == 1


def test_topology_add_bond():
    topology = Topology()
    chain = topology.add_chain(id_="A")
    residue1 = topology.add_residue(
        name="ALA", seq_num=1, insertion_code="", chain=chain
    )
    atom1 = topology.add_atom(
        name="C", atomic_num=6, formal_charge=0, serial=1, residue=residue1
    )
    residue2 = topology.add_residue(
        name="GLY", seq_num=2, insertion_code="", chain=chain
    )
    atom2 = topology.add_atom(
        name="N", atomic_num=7, formal_charge=0, serial=2, residue=residue2
    )
    topology.add_bond(idx_1=atom1.index, idx_2=atom2.index, order=1)
    assert topology.n_bonds == 1
    assert len(topology.bonds) == 1


def test_topology_invalid_add_bond():
    topology = Topology()
    topology.add_chain("A")
    topology.add_residue("ALA", seq_num=1, insertion_code="", chain=topology.chains[0])
    topology.add_atom("C", 6, 0, 1, topology.residues[0])

    with pytest.raises(ValueError, match="Index 1 is out of range."):
        topology.add_bond(idx_1=10, idx_2=0, order=1)

    with pytest.raises(ValueError, match="Index 2 is out of range."):
        topology.add_bond(idx_1=0, idx_2=10, order=1)


def test_topology_omm_roundtrip(test_data_dir):
    pdb = openmm.app.PDBFile(str(test_data_dir / "protein.pdb"))
    topology_original = Topology.from_openmm(pdb.topology)

    topology_omm = topology_original.to_openmm()
    topology_roundtrip = Topology.from_openmm(topology_omm)

    compare_topologies(topology_roundtrip, topology_original)


def test_topology_rdkit_roundtrip():
    mol: Chem.Mol = Chem.AddHs(Chem.MolFromSmiles("O=Cc1ccccc1[N+](=O)[O-]"))
    assert mol is not None, "Failed to create RDKit molecule from SMILES."
    expected_smiles = Chem.MolToSmiles(mol, canonical=True)

    mol.SetProp("MolStrProp", "mol-a")
    mol.SetDoubleProp("MolDblProp", 1.0)
    mol.SetIntProp("MolIntProp", 2)
    mol.SetBoolProp("MolBoolProp", True)

    atom = mol.GetAtomWithIdx(0)
    atom.SetProp("AtomStrProp", "atom-a")
    atom.SetDoubleProp("AtomDblProp", 3.0)
    atom.SetIntProp("AtomIntProp", 4)
    atom.SetBoolProp("AtomBoolProp", False)

    bond = mol.GetBondWithIdx(0)
    bond.SetProp("BondStrProp", "bond-a")
    bond.SetDoubleProp("BondDblProp", 5.0)
    bond.SetIntProp("BondIntProp", 6)
    bond.SetBoolProp("BondBoolProp", True)

    AllChem.EmbedMolecule(mol)
    expected_coords = numpy.array(mol.GetConformer().GetPositions())

    topology = Topology.from_rdkit(mol, "ABC", "E")

    assert topology.meta == {
        "MolStrProp": "mol-a",
        "MolDblProp": 1.0,
        "MolIntProp": 2,
        "MolBoolProp": True,
    }
    assert topology.atoms[0].meta == {
        "AtomStrProp": "atom-a",
        "AtomDblProp": 3.0,
        "AtomIntProp": 4,
        "AtomBoolProp": False,
    }
    assert topology.bonds[0].meta == {
        "BondStrProp": "bond-a",
        "BondDblProp": 5.0,
        "BondIntProp": 6,
        "BondBoolProp": True,
    }

    roundtrip_mol = topology.to_rdkit()
    roundtrip_smiles = Chem.MolToSmiles(roundtrip_mol, canonical=True)
    roundtrip_coords = numpy.array(roundtrip_mol.GetConformer().GetPositions())

    assert roundtrip_mol.GetProp("MolStrProp") == "mol-a"
    assert roundtrip_mol.GetDoubleProp("MolDblProp") == 1.0
    assert roundtrip_mol.GetIntProp("MolIntProp") == 2
    assert roundtrip_mol.GetBoolProp("MolBoolProp") is True

    assert roundtrip_mol.GetAtomWithIdx(0).GetProp("AtomStrProp") == "atom-a"
    assert roundtrip_mol.GetAtomWithIdx(0).GetDoubleProp("AtomDblProp") == 3.0
    assert roundtrip_mol.GetAtomWithIdx(0).GetIntProp("AtomIntProp") == 4
    assert roundtrip_mol.GetAtomWithIdx(0).GetBoolProp("AtomBoolProp") is False

    assert roundtrip_mol.GetBondWithIdx(0).GetProp("BondStrProp") == "bond-a"
    assert roundtrip_mol.GetBondWithIdx(0).GetDoubleProp("BondDblProp") == 5.0
    assert roundtrip_mol.GetBondWithIdx(0).GetIntProp("BondIntProp") == 6
    assert roundtrip_mol.GetBondWithIdx(0).GetBoolProp("BondBoolProp") is True

    assert expected_smiles == roundtrip_smiles

    assert expected_coords.shape == roundtrip_coords.shape
    assert numpy.allclose(expected_coords, roundtrip_coords)


def test_topology_rdkit_unique_names():
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))

    topology = Topology.from_rdkit(mol, "ABC", "E")
    names = [atom.name for atom in topology.atoms]

    expected_names = ["C1xx", "H1xx", "H2xx", "H3xx", "H4xx"]
    assert names == expected_names


def test_topology_oe_roundtrip(tmp_path):
    pytest.importorskip("openeye")

    from openeye import oechem

    mol: oechem.OEMol = oechem.OEMol()
    oechem.OESmilesToMol(mol, "O=Cc1ccccc1[N+](=O)[O-]")
    oechem.OEAddExplicitHydrogens(mol)

    with oechem.oemolostream(str(tmp_path / "molecule.sdf")) as ofs:
        oechem.OEWriteMolecule(ofs, mol)
    with oechem.oemolistream(str(tmp_path / "molecule.sdf")) as ifs:
        assert oechem.OEReadMolecule(ifs, mol)

    assert oechem.OESetCrystalSymmetry(
        mol,
        10.0,
        20.0,
        30.0,
        70.0,
        80.0,
        95.0,
        oechem.OEGetSpaceGroupNumber("P1"),
        True,
    )

    expected_smiles = oechem.OEMolToSmiles(mol)
    expected_coords = numpy.arange(mol.NumAtoms() * 3).reshape(-1, 3) * 0.1

    mol.DeleteConfs()
    mol.NewConf(oechem.OEFloatArray(expected_coords.flatten().tolist()))

    mol.SetStringData("MolStrProp", "mol-a")
    mol.SetDoubleData("MolDblProp", 1.0)
    mol.SetIntData("MolIntProp", 2)
    mol.SetBoolData("MolBoolProp", True)

    atom = [*mol.GetAtoms()][0]
    atom.SetStringData("AtomStrProp", "atom-a")
    atom.SetDoubleData("AtomDblProp", 3.0)
    atom.SetIntData("AtomIntProp", 4)
    atom.SetBoolData("AtomBoolProp", False)

    bond = [*mol.GetBonds()][0]
    bond.SetStringData("BondStrProp", "bond-a")
    bond.SetDoubleData("BondDblProp", 5.0)
    bond.SetIntData("BondIntProp", 6)
    bond.SetBoolData("BondBoolProp", True)

    topology = Topology.from_openeye(mol)

    assert topology.box is not None

    a, b, c, alpha, beta, gamma = box_to_geometry(topology.box)
    assert numpy.allclose(a, 10.0)
    assert numpy.allclose(b, 20.0)
    assert numpy.allclose(c, 30.0)
    assert numpy.allclose(alpha, 70.0)
    assert numpy.allclose(beta, 80.0)
    assert numpy.allclose(gamma, 95.0)

    assert topology.meta == {
        "MolStrProp": "mol-a",
        "MolDblProp": 1.0,
        "MolIntProp": 2,
        "MolBoolProp": True,
    }
    assert topology.atoms[0].meta == {
        "AtomStrProp": "atom-a",
        "AtomDblProp": 3.0,
        "AtomIntProp": 4,
        "AtomBoolProp": False,
    }
    assert topology.bonds[0].meta == {
        "BondStrProp": "bond-a",
        "BondDblProp": 5.0,
        "BondIntProp": 6,
        "BondBoolProp": True,
    }

    roundtrip_mol = topology.to_openeye()
    roundtrip_smiles = oechem.OEMolToSmiles(roundtrip_mol)

    assert roundtrip_mol.GetStringData("MolStrProp") == "mol-a"
    assert roundtrip_mol.GetDoubleData("MolDblProp") == 1.0
    assert roundtrip_mol.GetIntData("MolIntProp") == 2
    assert roundtrip_mol.GetBoolData("MolBoolProp") is True

    roundtrip_atom = [*roundtrip_mol.GetAtoms()][0]
    assert roundtrip_atom.GetData("AtomStrProp") == "atom-a"
    assert roundtrip_atom.GetDoubleData("AtomDblProp") == 3.0
    assert roundtrip_atom.GetIntData("AtomIntProp") == 4
    assert roundtrip_atom.GetBoolData("AtomBoolProp") is False

    roundtrip_bond = [*roundtrip_mol.GetBonds()][0]
    assert roundtrip_bond.GetData("BondStrProp") == "bond-a"
    assert roundtrip_bond.GetDoubleData("BondDblProp") == 5.0
    assert roundtrip_bond.GetIntData("BondIntProp") == 6
    assert roundtrip_bond.GetBoolData("BondBoolProp") is True

    assert expected_smiles == roundtrip_smiles

    assert roundtrip_mol.NumConfs() == 1

    roundtrip_coord_dict = roundtrip_mol.GetCoords()
    roundtrip_coords = numpy.array(
        [roundtrip_coord_dict[i] for i in range(roundtrip_mol.NumAtoms())]
    )

    assert oechem.OEHasCrystalSymmetry(roundtrip_mol)
    a, b, c, alpha, beta, gamma, *_ = oechem.OEGetCrystalSymmetry(roundtrip_mol)

    assert numpy.allclose(a, 10.0)
    assert numpy.allclose(b, 20.0)
    assert numpy.allclose(c, 30.0)
    assert numpy.allclose(alpha, 70.0)
    assert numpy.allclose(beta, 80.0)
    assert numpy.allclose(gamma, 95.0)

    assert expected_coords.shape == roundtrip_coords.shape
    assert numpy.allclose(expected_coords, roundtrip_coords)


@pytest.mark.parametrize(
    "suffix, write_fn",
    [
        (".mol", Chem.MolToMolFile),
        (".sdf", Chem.MolToMolFile),
        (".pdb", Chem.MolToPDBFile),
    ],
)
def test_topology_from_file(tmp_path, suffix, write_fn):
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    mol_path = tmp_path / f"mol{suffix}"

    write_fn(mol, str(mol_path))
    top = Topology.from_file(mol_path)

    symbols = [atom.symbol for atom in top.atoms]
    assert symbols == ["C", "H", "H", "H", "H"]


def test_topology_from_file_mol2(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    mol_path = tmp_path / "mol.mol2"

    top_omm = Topology.from_rdkit(mol, "UNK", "A").to_openmm()

    for a in top_omm.atoms():
        a.name = a.element.symbol

    parmed.openmm.load_topology(top_omm).save(str(mol_path))
    top = Topology.from_file(mol_path)

    symbols = [atom.symbol for atom in top.atoms]
    assert symbols == ["C", "H", "H", "H", "H"]


def test_topology_pdb_roundtrip(tmp_path, test_data_dir):
    topology_original = Topology.from_file(test_data_dir / "protein.pdb")

    topology_original.to_file(tmp_path / "protein.pdb")
    topology_roundtrip = Topology.from_file(tmp_path / "protein.pdb")

    compare_topologies(topology_roundtrip, topology_original)


@pytest.mark.parametrize("format", ["mol", "sdf"])
def test_topology_sdf_roundtrip(format, tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))

    topology_original = Topology.from_rdkit(mol, "LIG", "")
    topology_original.xyz = numpy.zeros((5, 3)) * openmm.unit.angstrom

    topology_original.to_file(tmp_path / f"mol.{format}")
    topology_roundtrip = Topology.from_file(tmp_path / f"mol.{format}")

    compare_topologies(topology_roundtrip, topology_original)


def test_topology_pdb_roundtrip_with_v_sites(tmp_path):
    topology = Topology()
    chain = topology.add_chain("A")
    residue = topology.add_residue("LIG", 1, "", chain)
    topology.add_atom("H1", 1, 0, 1, residue)
    topology.add_atom("Cl1", 17, 0, 2, residue)
    topology.add_atom("X1", 0, 0, 3, residue)

    pdb_path = tmp_path / "ligand.pdb"

    topology.to_file(pdb_path)
    topology_roundtrip = Topology.from_file(pdb_path)

    assert topology_roundtrip.atoms[0].symbol == "H"
    assert topology_roundtrip.atoms[1].symbol == "Cl"
    assert topology_roundtrip.atoms[2].symbol == "X"


def test_topology_select():
    topology = Topology()

    chain_a = topology.add_chain("A")
    res_a = topology.add_residue("ALA", 1, "", chain_a)
    topology.add_atom("C1", 6, 0, 1, res_a)

    selection = topology.select("c. A and r. ALA")
    assert numpy.allclose(selection, numpy.array([0]))


def test_topology_select_amber():
    topology = Topology()

    chain_a = topology.add_chain("A")
    res_a = topology.add_residue("ACE", 1, "", chain_a)
    topology.add_atom("H1", 1, 0, 1, res_a)
    topology.add_atom("CH3", 6, 0, 2, res_a)
    topology.add_atom("H2", 1, 0, 3, res_a)
    topology.add_atom("H3", 1, 0, 4, res_a)
    topology.add_atom("C", 6, 0, 5, res_a)
    topology.add_atom("O", 8, 0, 6, res_a)

    selection = topology.select(":ACE & !@/H")
    assert numpy.allclose(selection, numpy.array([1, 4, 5]))


def test_topology_subset():
    topology = Topology()

    chain_a = topology.add_chain("A")
    res_a = topology.add_residue("ALA", 1, "", chain_a)
    topology.add_atom("C1", 6, 0, 1, res_a)
    topology.add_atom("C2", 6, 0, 2, res_a)
    res_b = topology.add_residue("MET", 2, "", chain_a)
    topology.add_atom("C3", 6, 0, 3, res_b)
    topology.add_atom("C4", 6, 0, 4, res_b)
    topology.add_residue("TYR", 2, "", chain_a)

    chain_b = topology.add_chain("B")
    res_d = topology.add_residue("GLY", 3, "", chain_b)
    topology.add_atom("C5", 6, 0, 5, res_d)
    chain_c = topology.add_chain("C")
    res_e = topology.add_residue("SER", 4, "", chain_c)
    topology.add_atom("C6", 6, 0, 6, res_e)

    topology.add_bond(0, 1, 1)
    topology.add_bond(0, 5, 1)

    subset = topology.subset([0, 3, 5])

    assert subset.n_chains == 2
    assert [c.id for c in subset.chains] == ["A", "C"]

    assert subset.n_residues == 3
    assert [r.name for r in subset.residues] == ["ALA", "MET", "SER"]

    assert subset.n_atoms == 3
    assert [a.name for a in subset.atoms] == ["C1", "C4", "C6"]

    assert subset.n_bonds == 1
    assert subset.bonds[0].idx_1 == 0
    assert subset.bonds[0].idx_2 == 2


@pytest.mark.parametrize(
    "n_atoms,bonds,expected",
    [
        # Single connected component (chain: 0-1-2-3-4)
        (5, [(0, 1), (1, 2), (2, 3), (3, 4)], [[0, 1, 2, 3, 4]]),
        # Two disconnected components: 0-1-2 and 3-4
        (5, [(0, 1), (1, 2), (3, 4)], [[0, 1, 2], [3, 4]]),
        # No edges: each atom is its own fragment
        (3, [], [[0], [1], [2]]),
    ],
)
def test_topology_split(n_atoms, bonds, expected):
    """
    Tests the find_fragments function with different configurations.
    """
    topology = Topology()
    chain = topology.add_chain("A")
    residue = topology.add_residue("ALA", 1, "", chain)

    for i in range(n_atoms):
        topology.add_atom("C", 6, 0, i, residue)

    for bond in bonds:
        topology.add_bond(*bond, 1)

    frags = topology.split()
    frags = [sorted(a.serial for a in frag.atoms) for frag in frags]

    assert frags == expected, f"Expected {expected}, got {frags}"


@pytest.mark.parametrize(
    "xyz",
    [
        numpy.arange(6).reshape(-1, 3) * openmm.unit.angstrom,
        numpy.arange(6).reshape(-1, 3).tolist() * openmm.unit.angstrom,
        numpy.arange(6).reshape(-1, 3),
        numpy.arange(6).reshape(-1, 3).tolist(),
    ],
)
def test_topology_xyz_setter(xyz):
    topology = Topology()
    topology.add_chain("A")
    topology.add_residue("ALA", 1, "", topology.chains[-1])
    topology.add_atom("C", 6, 0, 1, topology.residues[-1])
    topology.add_atom("C", 6, 0, 2, topology.residues[-1])
    topology.xyz = xyz

    expected_xyz = numpy.arange(topology.n_atoms * 3).reshape(-1, 3)

    assert isinstance(topology.xyz, openmm.unit.Quantity)

    xyz_array = topology.xyz.value_in_unit(openmm.unit.angstrom)
    assert isinstance(xyz_array, numpy.ndarray)
    assert xyz_array.shape == expected_xyz.shape
    assert numpy.allclose(xyz_array, expected_xyz)

    with pytest.raises(ValueError, match="expected shape"):
        topology.xyz = numpy.zeros((0, 3))


def test_topology_xyz_setter_none():
    topology = Topology()
    topology.xyz = numpy.zeros((0, 3)) * openmm.unit.angstrom
    topology.xyz = None
    assert topology.xyz is None


@pytest.mark.parametrize(
    "box",
    [
        numpy.eye(3) * openmm.unit.angstrom,
        numpy.eye(3).tolist() * openmm.unit.angstrom,
        numpy.eye(3),
    ],
)
def test_topology_box_setter(box):
    topology = Topology()
    topology.box = box

    expected_box = numpy.eye(3)

    assert isinstance(topology.box, openmm.unit.Quantity)

    box_array = topology.box.value_in_unit(openmm.unit.angstrom)
    assert isinstance(box_array, numpy.ndarray)
    assert box_array.shape == expected_box.shape
    assert numpy.allclose(box_array, expected_box)

    with pytest.raises(ValueError, match="expected shape"):
        topology.box = numpy.zeros((0, 3))


def test_topology_merge():
    topology1 = Topology()
    topology1.add_chain(id_="A")
    topology1.add_residue("ALA", 1, "", topology1.chains[0])
    topology1.add_atom("C", 6, 0, 1, topology1.residues[-1])
    topology1.add_residue("GLY", 2, "", topology1.chains[0])
    topology1.add_atom("N", 7, 0, 2, topology1.residues[-1])
    topology1.add_chain(id_="B")
    topology1.add_residue("SER", 1, "", topology1.chains[1])
    topology1.add_atom("O", 8, 0, 3, topology1.residues[-1])
    topology1.add_bond(0, 1, 1)
    topology1.xyz = (
        numpy.arange(topology1.n_atoms * 3).reshape(-1, 3) * openmm.unit.angstrom
    )

    topology2 = Topology()
    topology2.add_chain(id_="C")
    topology2.add_residue("VAL", 1, "", topology2.chains[0])
    topology2.add_atom("CA", 6, 0, 1, topology2.residues[-1])
    topology2.add_residue("GLU", 2, "", topology2.chains[0])
    topology2.add_atom("CB", 6, 0, 2, topology2.residues[-1])
    topology2.add_bond(0, 1, 1)
    topology2.xyz = None

    merged_topology = Topology.merge(topology1, topology2)

    assert topology1.n_chains == 2
    assert topology2.n_chains == 1

    assert merged_topology.n_chains == 3
    assert merged_topology.n_chains == 3
    assert merged_topology.n_residues == topology1.n_residues + topology2.n_residues
    assert merged_topology.n_atoms == topology1.n_atoms + topology2.n_atoms
    assert merged_topology.n_bonds == topology1.n_bonds + topology2.n_bonds

    assert merged_topology.chains[0].id == "A"
    assert merged_topology.chains[1].id == "B"
    assert merged_topology.chains[2].id == "C"

    chain_a_residues = merged_topology.chains[0].residues
    assert chain_a_residues[0].name == "ALA"
    assert chain_a_residues[1].name == "GLY"
    chain_b_residues = merged_topology.chains[1].residues
    assert chain_b_residues[0].name == "SER"
    chain_c_residues = merged_topology.chains[2].residues
    assert chain_c_residues[0].name == "VAL"
    assert chain_c_residues[1].name == "GLU"
    assert merged_topology.bonds[0].idx_1 == 0
    assert merged_topology.bonds[0].idx_2 == 1
    assert merged_topology.bonds[1].idx_1 == topology1.n_atoms
    assert merged_topology.bonds[1].idx_2 == topology1.n_atoms + 1

    expected_xyz = numpy.vstack(
        [
            numpy.arange(topology1.n_atoms * 3).reshape(-1, 3),
            numpy.zeros((topology2.n_atoms, 3)),
        ]
    )
    assert merged_topology.xyz.shape == expected_xyz.shape
    assert numpy.allclose(merged_topology.xyz, expected_xyz)

    for i, atom in enumerate(merged_topology.atoms):
        assert atom.index == i
    for i, residue in enumerate(merged_topology.residues):
        assert residue.index == i


def test_topology_merge_with_b_coords():
    topology1 = Topology()
    topology1.add_chain(id_="A")
    topology1.add_residue("ALA", 1, "", topology1.chains[0])
    topology1.add_atom("C", 6, 0, 1, topology1.residues[-1])
    topology1.add_atom("N", 7, 0, 2, topology1.residues[-1])
    topology1.xyz = None

    topology2 = Topology()
    topology2.add_chain(id_="C")
    topology2.add_residue("ALA", 1, "", topology2.chains[0])
    topology2.add_atom("CA", 6, 0, 1, topology2.residues[-1])
    topology2.xyz = numpy.array([[1.0, 2.0, 3.0]]) * openmm.unit.angstrom

    merged_topology = Topology.merge(topology1, topology2)

    expected_xyz = (
        numpy.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        * openmm.unit.angstrom
    )
    assert merged_topology.xyz.shape == expected_xyz.shape
    assert numpy.allclose(merged_topology.xyz, expected_xyz)


def test_topology_add():
    topology1 = Topology()
    topology1.add_chain(id_="A")
    topology1.add_residue("ALA", 1, "", topology1.chains[0])
    topology1.add_atom("C", 6, 0, 1, topology1.residues[-1])
    topology1.xyz = numpy.array([[1.0, 2.0, 3.0]]) * openmm.unit.angstrom

    topology2 = Topology()
    topology2.add_chain(id_="C")
    topology2.add_residue("ALA", 1, "", topology2.chains[0])
    topology2.add_atom("CA", 6, 0, 1, topology2.residues[-1])
    topology2.xyz = numpy.array([[4.0, 5.0, 6.0]]) * openmm.unit.angstrom

    merged_topology = topology1 + topology2

    expected_xyz = (
        numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * openmm.unit.angstrom
    )
    assert merged_topology.xyz.shape == expected_xyz.shape
    assert numpy.allclose(merged_topology.xyz, expected_xyz)


@pytest.mark.parametrize(
    "item, expected",
    [
        ("idx. 1+5", numpy.array([0, 4])),
        (3, numpy.array([3])),
        (slice(0, 4), numpy.array([0, 1, 2, 3])),
        (numpy.array([1, 3, 5]), numpy.array([1, 3, 5])),
        ([1, 3, 5], numpy.array([1, 3, 5])),
    ],
)
def test_topology_slice(mocker, item, expected):
    mock_subset = mocker.patch("mdtop.Topology.subset", autospec=True)

    topology = Topology()
    topology.add_chain("A")
    topology.add_residue("ALA", 1, "", topology.chains[-1])

    for i in range(10):
        topology.add_atom("C", 6, 0, i + 1, topology.residues[-1])

    topology.__getitem__(item)

    mock_subset.assert_called_once()

    idxs = mock_subset.call_args.args[1]
    assert isinstance(idxs, numpy.ndarray)

    assert idxs.shape == expected.shape
    assert numpy.allclose(idxs, expected)


def test_repr():
    topology = Topology()
    topology.add_chain("A")
    topology.add_residue("ALA", 1, "", topology.chains[-1])
    topology.add_atom("C", 6, 0, 1, topology.residues[-1])
    topology.add_atom("C", 6, 0, 2, topology.residues[-1])
    topology.add_bond(0, 1, 1)

    assert repr(topology) == "Topology(n_chains=1, n_residues=1, n_atoms=2)"
    assert repr(topology.chains[0]) == "Chain(id='A')"
    assert (
        repr(topology.residues[0]) == "Residue(name='ALA', seq_num=1 insertion_code='')"
    )
    assert (
        repr(topology.atoms[0])
        == "Atom(name='C', atomic_num=6, formal_charge=0, serial=1)"
    )
    assert repr(topology.bonds[0]) == "Bond(idx_1=0, idx_2=1, order=1)"
