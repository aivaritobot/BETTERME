from __future__ import annotations

from app.gui import BetterMeDesktopApp


class AppController:
    """Single official desktop entrypoint controller."""

    def __init__(self):
        self.app = BetterMeDesktopApp()

    def run(self) -> int:
        self.app.run()
        return 0
