import pytest

from vista import colors


def test_invalid_color_str_single_char():
    with pytest.raises(ValueError):
        colors.string_to_rgb('x')


def test_invalid_color_str():
    with pytest.raises(ValueError):
        colors.string_to_rgb('not a color')
