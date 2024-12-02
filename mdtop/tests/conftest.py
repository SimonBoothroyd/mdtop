import pathlib

import pytest


@pytest.fixture
def test_data_dir() -> pathlib.Path:
    """Return the path to the test data directory."""
    return pathlib.Path(__file__).parent / "data"
