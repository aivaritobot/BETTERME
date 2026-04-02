from engine.physics import CylinderPhysics
from engine.vision import RouletteVision


def test_cylinder_sector_resolution():
    physics = CylinderPhysics()
    assert physics.get_sector(22) == 'Voisins'
    assert physics.get_sector(1) == 'Orphelins'
    assert physics.get_sector(36) == 'Tier'


def test_cylinder_physical_trend_detection():
    physics = CylinderPhysics()
    history = [22, 18, 29, 1, 22]
    assert physics.predict_physical_zone(history) == 'Voisins'


def test_ocr_number_extraction():
    assert RouletteVision._extract_number(' 17\n') == 17
    assert RouletteVision._extract_number('xx') is None
    assert RouletteVision._extract_number('99') is None
