import os
import pytest

@pytest.fixture
def rootdir():
    """Add path to test data as fixture. """
    return os.path.dirname(os.path.abspath("tests/test_data"))