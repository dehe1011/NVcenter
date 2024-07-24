import pytest
from src import my_add

@pytest.mark.parametrize("x, y, expected", [(1,2,3), (4,-2,2)] )
def test_my_add(x, y, expected):
    assert my_add(x, y) == expected

if __name__ == "__main__":
    pytest.main()
