"""Convert between OpenMM and ``mdtop`` objects."""

import numpy
import openmm.app

from mdtop import Topology


def from_openmm(topology_omm: openmm.app.Topology) -> Topology:
    """Create a topology from an OpenMM topology.

    Args:
        topology_omm: The OpenMM topology to convert.

    Returns:
        The converted topology.
    """
    topology = Topology()

    for chain_omm in topology_omm.chains():
        chain = topology.add_chain(chain_omm.id)

        for res_omm in chain_omm.residues():
            res = topology.add_residue(
                res_omm.name, res_omm.id, res_omm.insertionCode, chain
            )

            for atom_omm in res_omm.atoms():
                is_v_site = atom_omm.element is None

                topology.add_atom(
                    atom_omm.name,
                    atom_omm.element.atomic_number if not is_v_site else 0,
                    None if is_v_site else getattr(atom_omm, "formalCharge", None),
                    atom_omm.id,
                    res,
                )

    for bond_omm in topology_omm.bonds():
        order = bond_omm.order

        if order is None and bond_omm.type is not None:
            raise NotImplementedError("Bond type is not None but order is None")

        topology.add_bond(bond_omm.atom1.index, bond_omm.atom2.index, order)

    if topology_omm.getPeriodicBoxVectors() is not None:
        box = topology_omm.getPeriodicBoxVectors().value_in_unit(openmm.unit.angstrom)
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

        for res in chain.residues:
            res_omm = topology_omm.addResidue(
                res.name,
                chain_omm,
                str(res.seq_num),
                res.insertion_code,
            )

            for atom in res.atoms:
                element = (
                    None
                    if atom.atomic_num == 0
                    else openmm.app.Element.getByAtomicNumber(atom.atomic_num)
                )
                atom_omm = topology_omm.addAtom(
                    atom.name, element, res_omm, str(atom.serial)
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
        if bond.order == 0:
            continue

        topology_omm.addBond(
            atoms_omm[bond.idx_1],
            atoms_omm[bond.idx_2],
            bond_order_to_type[bond.order] if bond.order is not None else None,
            bond.order,
        )

    if self.box is not None:
        topology_omm.setPeriodicBoxVectors(self.box)

    return topology_omm
