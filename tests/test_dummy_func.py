from src import dummy_function


def test_dummy_func() -> None:
    """A dummy test case for a dummy function."""
    assert dummy_function(3) == 4
