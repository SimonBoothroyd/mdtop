import numpy
import openmm.unit
import scipy

_EPSILON = 1e-6
"""A small value used for numerical comparisons."""


def compute_pairwise_distances(
    xyz_a: numpy.ndarray,
    xyz_b: numpy.ndarray,
    box: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """
    Computes all pairwise distances between particles in a periodic simulation box.

    Args:
        xyz_a: The coordinates of the first set of particles with
            ``shape=(n_atoms_a, 3)``.
        xyz_b: The coordinates of the second set of particles with
            ``shape=(n_atoms_a, 3)``.
        box: The box vectors of the simulation box with ``shape=(3, 3)``.

    Returns:
        The pairwise distances with ``shape=(n_atoms_a, n_atoms_b)``.
    """
    if box is None:
        return scipy.spatial.distance.cdist(xyz_a, xyz_b, "euclidean")

    if not numpy.allclose(box, numpy.diag(numpy.diagonal(box))):
        raise NotImplementedError("only orthogonal boxes are supported.")

    box_flat = numpy.diag(box)
    box_inv = 1.0 / box_flat

    delta = xyz_a[:, None, :] - xyz_b[None, :, :]
    delta -= numpy.floor(delta * box_inv[None, None, :] + 0.5) * box_flat[None, None, :]

    return numpy.linalg.norm(delta, axis=-1)


def box_to_geometry(
    box: openmm.unit.Quantity,
) -> tuple[float, float, float, float, float, float]:
    """Convert a box to its geometry parameters.

    Args:
        box: The box vectors with ``shape=(3, 3)``.

    Returns:
        The box lengths and angles in the order ``(a, b, c, alpha, beta, gamma)``,
        where all lengths are in angstroms and all angles are in degrees.
    """

    box = box.value_in_unit(openmm.unit.angstrom)
    assert box.shape == (3, 3)

    a, b, c = box[0], box[1], box[2]

    lengths = numpy.linalg.norm([a, b, c], axis=1)

    alpha = numpy.rad2deg(numpy.arccos(numpy.dot(b, c) / (lengths[1] * lengths[2])))
    beta = numpy.rad2deg(numpy.arccos(numpy.dot(c, a) / (lengths[2] * lengths[0])))
    gamma = numpy.rad2deg(numpy.arccos(numpy.dot(a, b) / (lengths[0] * lengths[1])))

    # round the box vectors so that future checks for orthorhombic boxes are more robust
    for vec in [a, b, c]:
        vec[numpy.abs(vec) < _EPSILON] = 0.0

    return lengths[0], lengths[1], lengths[2], alpha, beta, gamma


def box_from_geometry(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> openmm.unit.Quantity:
    """Convert box geometry parameters to a box.

    Args:
        a: The length [Å] of the first box vector.
        b: The length [Å] of the second box vector.
        c: The length [Å] of the third box vector.
        alpha: The angle [deg] between the second and third box vectors.
        beta: The angle [deg] between the third and first box vectors.
        gamma: The angle [deg] between the first and second box vectors.

    Returns:
        The box vectors with ``shape=(3, 3)``.
    """

    alpha, beta, gamma = numpy.deg2rad([alpha, beta, gamma])

    cx = c * numpy.cos(beta)
    cy = c * (numpy.cos(alpha) - numpy.cos(beta) * numpy.cos(gamma)) / numpy.sin(gamma)
    cz = numpy.sqrt(c**2 - cx**2 - cy**2)

    a_vec = numpy.array([a, 0, 0])
    b_vec = numpy.array([b * numpy.cos(gamma), b * numpy.sin(gamma), 0])
    c_vec = numpy.array([cx, cy, cz])

    return numpy.array([a_vec, b_vec, c_vec]) * openmm.unit.angstrom
