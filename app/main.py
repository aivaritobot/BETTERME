from __future__ import annotations

from app.controller import AppController


def main() -> int:
    return AppController().run()


if __name__ == "__main__":
    raise SystemExit(main())
