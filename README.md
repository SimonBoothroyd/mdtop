<h1 align="center">mdtop</h1>

<p align="center">A simple package for storing topological information about a system to simulate.</p>

<p align="center">
  <a href="https://github.com/SimonBoothroyd/mdtop/actions?query=workflow%3Aci">
    <img alt="ci" src="https://github.com/SimonBoothroyd/mdtop/actions/workflows/ci.yaml/badge.svg" />
  </a>
  <a href="https://codecov.io/gh/SimonBoothroyd/mdtop/branch/main">
    <img alt="coverage" src="https://codecov.io/gh/SimonBoothroyd/mdtop/branch/main/graph/badge.svg" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="license" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>

---

The `mdtop` framework aims to provide a simple, modern, and flexible way to store
topological information about a system to simulate. While this may be 'yet another
`Topology` object', it is designed to take the best of  all worlds from the fantastic
`mdtraj`, `parmed`, and `openmm` packages, and provide a clean way to store atom
data, merge multiple topologies, and also select subsets using the ubiquitous `PyMol`
atom selection language.

## Installation

This package can be installed using `conda` (or `mamba`, a faster version of `conda`):

```shell
mamba install -c conda-forge mdtop
```

## Getting Started

### Creating a Topology
Topologies can be created from a variety of sources. However, the most robust workflows
typically involve:

* **Creating a topology from OpenMM**: Use the `Topology.from_openmm()` method to import
  an existing OpenMM topology containing, for example, a protein structure.
* **Creating a topology from an RDKit molecule**: Use the `Topology.from_rdkit()` method
  to create a topology from an RDKit molecule representation.
* **Loading a topology from a file**: Use the `Topology.from_file()` method to import
  an existing PDB, SDF, or MOL2 file.

```python
from openmm.app import PDBFile
from rdkit import Chem

from mdtop import Topology

# Load a protein topology from OpenMM
protein_top_omm = PDBFile("protein.pdb").topology
protein_top = Topology.from_openmm(protein_top_omm)
# OR
protein_top = Topology.from_file("protein.pdb")

# Load a ligand using RDKit
ligand_rd = Chem.MolFromMolFile("ligand.sdf")
ligand_top = Topology.from_rdkit(ligand_rd)
# OR
ligand_top = Topology.from_file("ligand.sdf")

# Merge the protein and ligand topologies
system_top = Topology.merge(protein_top, ligand_top)
# OR
system_top = protein_top + ligand_top
```

### Atom Selection

Subsets of atoms can be selected using a (for now) subset of the
[PyMol atom selection language]((https://pymolwiki.org/index.php/Selection_Algebra)).

For example, to select all atoms in chain A:

```python
selection = system_top.select("chain A")
```

or all atoms within 5 Å of the ligand:

```python
atom_idxs = system_top.select("all within 5. of resn LIG")
```

A subset of the topology can then be created using the `subset()` method:

```python
subset = system_top.subset(atom_idxs)
```

or by indexing directly:

```python
subset = system_top["chain A"]
# OR
subset = system_top[atom_idxs]
```

### Exporting Topologies

Topologies can be converted back into OpenMM or RDKit formats for further analysis or
simulation.

```python
# Export to OpenMM topology
system_top_omm = system_top.to_openmm()

# Export to RDKit molecule - this currently only works for topologies that contain
# full formal charge and bond order information.
mol_rd = ligand_top.to_rdkit()
```

They can also be directly written to a file:

```python
system_top.to_file("system.pdb")
```
