from __future__ import annotations

import argparse
import logging
import time

from utils.config import load_roi_from_config, parse_manual_roi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='ALEXBOT V3 - Prototipo de análisis visual de ruleta')
    parser.add_argument('--source', choices=['screen', 'video'], default='screen')
    parser.add_argument('--video-path', default=None, help='Ruta de video local para demo reproducible')
    parser.add_argument('--config', default='config.json', help='Ruta a config de calibración')
    parser.add_argument('--roi-manual', default=None, help='ROI manual: top,left,width,height')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--max-frames', type=int, default=0, help='0 = infinito. Útil para smoke tests.')
    parser.add_argument('--overlay', action='store_true', default=False, help='Mostrar ventana de overlay (requiere backend GUI)')
    return parser


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s | %(levelname)s | %(message)s',
    )


def run(args: argparse.Namespace) -> int:
    setup_logging(args.log_level)
    logger = logging.getLogger('alexbot')

    # Imports pesados/lábiles en runtime para que --help siempre funcione sin depender de OpenCV GUI.
    try:
        from engine.physics import AlexBotPhysics
        from engine.vision import AlexBotVision
        from utils.mapping import get_relative_prediction_angle
    except Exception as exc:  # noqa: BLE001
        logger.error('No se pudo cargar módulos de runtime: %s', exc)
        return 1

    show_overlay = None
    cv2 = None
    if args.overlay:
        try:
            import cv2 as _cv2
            from ui.overlay import show_alex_overlay as _show_alex_overlay

            cv2 = _cv2
            show_overlay = _show_alex_overlay
        except Exception as exc:  # noqa: BLE001
            logger.warning('Overlay deshabilitado (GUI/OpenCV no disponible): %s', exc)
            args.overlay = False

    try:
        roi = parse_manual_roi(args.roi_manual) if args.roi_manual else load_roi_from_config(args.config)
    except Exception as exc:  # noqa: BLE001
        logger.error('No se pudo cargar ROI: %s', exc)
        return 1

    try:
        vision = AlexBotVision(roi=roi, source=args.source, video_path=args.video_path)
    except Exception as exc:  # noqa: BLE001
        logger.error('No se pudo iniciar vision: %s', exc)
        return 1

    brain = AlexBotPhysics()

    logger.info('Inicio OK | source=%s | roi=(%s,%s,%s,%s) | overlay=%s', args.source, roi.top, roi.left, roi.width, roi.height, args.overlay)

    frame_count = 0
    last_log = 0.0

    try:
        while True:
            ball_angle, rotor_angle, frame, debug = vision.get_alex_data()
            if frame is None:
                logger.info('Fuente finalizada (EOF).')
                break

            brain.update(ball_angle, rotor_angle)
            pred = brain.get_prediction()
            relative_prediction = None
            if pred is not None:
                relative_prediction = get_relative_prediction_angle(pred.ball_pred, pred.rotor_pred)

            if args.overlay and show_overlay is not None:
                show_overlay(frame, relative_prediction, telemetry=pred, debug=debug)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    logger.info('Salida solicitada por usuario.')
                    break

            now = time.time()
            if now - last_log > 1.5:
                logger.info(
                    'frame=%d ball=%s rotor=%s pred=%s conf=%s',
                    frame_count,
                    f'{ball_angle:.2f}' if ball_angle is not None else 'None',
                    f'{rotor_angle:.2f}' if rotor_angle is not None else 'None',
                    f'{relative_prediction:.2f}' if relative_prediction is not None else 'None',
                    f'{pred.confidence:.2f}' if pred is not None else 'None',
                )
                last_log = now

            frame_count += 1
            if args.max_frames > 0 and frame_count >= args.max_frames:
                logger.info('max_frames alcanzado: %d', args.max_frames)
                break
    finally:
        vision.close()
        if args.overlay and cv2 is not None:
            cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    cli = build_parser().parse_args()
    raise SystemExit(run(cli))
