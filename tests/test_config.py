import pytest

from utils.config import parse_manual_roi


def test_parse_manual_roi_ok():
    roi = parse_manual_roi('10,20,300,400')
    assert roi.top == 10
    assert roi.left == 20
    assert roi.width == 300
    assert roi.height == 400


def test_parse_manual_roi_bad():
    with pytest.raises(ValueError):
        parse_manual_roi('10,20,0,400')
