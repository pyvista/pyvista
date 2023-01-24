import pytest

import pyvista as pv
from pyvista.plotting.volume_property import VolumeProperty


@pytest.fixture()
def vol_prop():
    return VolumeProperty()


def test_volume_lookup_table(vol_prop):
    lut = pv.LookupTable(cmap='bwr')
    lut.apply_opacity([1.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.3])
    orig = vol_prop.GetRGBTransferFunction()
    vol_prop.apply_lookup_table(lut)
    assert vol_prop.GetRGBTransferFunction() is not orig


def test_interpolation_type(vol_prop):
    assert isinstance(vol_prop.interpolation_type, str)

    for interpolation_type in ['nearest', 'linear']:
        vol_prop.interpolation_type = interpolation_type
        assert vol_prop.interpolation_type == interpolation_type

    with pytest.raises(ValueError, match='must be either'):
        vol_prop.interpolation_type = 'not valid'


def test_volume_property_shade(vol_prop):
    assert isinstance(vol_prop.shade, bool)
    vol_prop.shade = True
    assert vol_prop.shade is True
    vol_prop.shade = False
    assert vol_prop.shade is False


def test_volume_independent_components(vol_prop):
    assert isinstance(vol_prop.independent_components, bool)
    vol_prop.independent_components = True
    assert vol_prop.independent_components is True
    vol_prop.independent_components = False
    assert vol_prop.independent_components is False


def test_volume_property_ambient(vol_prop):
    assert isinstance(vol_prop.ambient, float)
    value = 0.45
    vol_prop.ambient = value
    assert vol_prop.ambient == value


def test_volume_property_diffuse(vol_prop):
    assert isinstance(vol_prop.diffuse, float)
    value = 0.45
    vol_prop.diffuse = value
    assert vol_prop.diffuse == value


def test_volume_property_specular(vol_prop):
    assert isinstance(vol_prop.specular, float)
    value = 0.45
    vol_prop.specular = value
    assert vol_prop.specular == value


def test_volume_property_specular_power(vol_prop):
    assert isinstance(vol_prop.specular_power, float)
    value = 0.45
    vol_prop.specular_power = value
    assert vol_prop.specular_power == value


def test_volume_property_copy(vol_prop):
    vol_prop.ambient = 1.0
    vol_prop_copy = vol_prop.copy()
    assert vol_prop_copy.ambient == vol_prop.ambient


def test_volume_property_repr(vol_prop):
    assert 'Interpolation type:' in repr(vol_prop)
    assert "nearest" in repr(vol_prop)
