from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def main() -> int:
    root = Path(__file__).resolve().parents[1]

    py_files = [
        'main.py',
        'run_demo.py',
        'calibrate.py',
        'engine/vision.py',
        'engine/physics.py',
        'ui/overlay.py',
        'utils/config.py',
        'utils/mapping.py',
    ]

    checks = [
        ['python', '-m', 'py_compile', *py_files],
        ['pytest', '-q'],
        ['python', 'main.py', '--help'],
        ['python', 'tools/generate_demo_video.py', '--output', 'demo.mp4', '--frames', '30', '--width', '300', '--height', '300'],
        ['python', 'main.py', '--source', 'video', '--video-path', 'demo.mp4', '--roi-manual', '0,0,300,300', '--max-frames', '20'],
    ]

    failed = 0
    for cmd in checks:
        code, out = run(cmd)
        status = 'PASS' if code == 0 else 'FAIL'
        print(f'[{status}]', ' '.join(cmd))
        print(out.strip())
        print('-' * 80)
        if code != 0:
            failed += 1

    demo = root / 'demo.mp4'
    if demo.exists():
        demo.unlink()

    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
