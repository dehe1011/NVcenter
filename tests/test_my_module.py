import pytest
from src import my_add, my_mul


@pytest.mark.parametrize("x, y, expected", [(1, 2, 3), (4, -2, 2)])
def test_my_add(x, y, expected):
    """
    Tests my_add().
    """
    assert my_add(x, y) == expected


@pytest.mark.parametrize("x, y, expected", [(1,2,2), (4,-2,-8)] )
def test_my_mul(x, y, expected):
    """
    Tests my_mul().
    """
    assert my_mul(x, y) == expected
