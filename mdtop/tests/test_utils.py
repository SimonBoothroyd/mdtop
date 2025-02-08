import numpy
import openmm.unit
import pytest

from mdtop._utils import box_from_geometry, box_to_geometry, compute_pairwise_distances


def test_compute_pairwise_distances():
    xyz_a = numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    xyz_b = numpy.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    box = numpy.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])

    dists = compute_pairwise_distances(xyz_a, xyz_b, box)
    expected_dists = numpy.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    assert dists.shape == expected_dists.shape
    assert numpy.allclose(dists, expected_dists)

    dists = compute_pairwise_distances(xyz_a, xyz_b, None)
    expected_dists = numpy.array([[2.0, 1.0], [1.0, 0.0], [3.0, 2.0]])

    assert dists.shape == expected_dists.shape
    assert numpy.allclose(dists, expected_dists)


def test_compute_pairwise_distances_triclinic():
    box = numpy.array([[3.0, 1.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])

    with pytest.raises(
        NotImplementedError, match="only orthogonal boxes are supported."
    ):
        compute_pairwise_distances(numpy.zeros((0, 3)), numpy.zeros((0, 3)), box)


def test_box_from_geometry_ortho():
    box = box_from_geometry(1.0, 2.0, 3.0, 90.0, 90.0, 90.0)
    expected_box = numpy.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])

    assert box.shape == expected_box.shape
    assert numpy.allclose(box.value_in_unit(openmm.unit.angstrom), expected_box)

    a, b, c, alpha, beta, gamma = box_to_geometry(box)

    assert numpy.allclose(a, 1.0)
    assert numpy.allclose(b, 2.0)
    assert numpy.allclose(c, 3.0)
    assert numpy.allclose(alpha, 90.0)
    assert numpy.allclose(beta, 90.0)
    assert numpy.allclose(gamma, 90.0)


def test_box_from_geometry_non_other():
    box = box_from_geometry(1.0, 2.0, 3.0, 80.0, 85.0, 95.0)

    expected_box = numpy.array(
        [
            [1.0, 0.0, 0.0],
            [-0.17431149, 1.9923894, 0.0],
            [0.26146723, 0.54580987, 2.93832035],
        ]
    )

    assert box.shape == expected_box.shape
    assert numpy.allclose(box.value_in_unit(openmm.unit.angstrom), expected_box)

    a, b, c, alpha, beta, gamma = box_to_geometry(box)

    assert numpy.allclose(a, 1.0)
    assert numpy.allclose(b, 2.0)
    assert numpy.allclose(c, 3.0)
    assert numpy.allclose(alpha, 80.0)
    assert numpy.allclose(beta, 85.0)
    assert numpy.allclose(gamma, 95.0)
