from engine.physics import CylinderPhysics, UniversalCylinderMap
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


def test_mode_switch_token_detection_and_neighbors():
    assert RouletteVision._extract_token('WINNER 00') == '00'
    mapping = UniversalCylinderMap(mode='American')
    neighbors = mapping.get_neighbors('00', span=1)
    assert neighbors == [1, '00', 27]


def test_stability_buffer_promotes_after_500ms_window():
    vision = RouletteVision(stable_ms=500, min_stable_samples=3)
    assert vision._promote_stable_token('17', now=1.0) is None
    assert vision._promote_stable_token('17', now=1.2) is None
    assert vision._promote_stable_token('17', now=1.51) == '17'
