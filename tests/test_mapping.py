from utils.mapping import get_alexbot_sector, get_relative_prediction_angle


def test_relative_angle_with_rotor():
    assert get_relative_prediction_angle(30, 10) == 20


def test_relative_angle_without_rotor():
    assert get_relative_prediction_angle(370, None) == 10


def test_sector_wraparound():
    name, nums = get_alexbot_sector(350)
    assert name == 'VECINOS DEL CERO'
    assert 0 in nums
