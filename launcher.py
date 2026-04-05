"""Legacy launcher entrypoint.

Now delegates to the desktop application so users can open BETTERME by double click
without interacting with a terminal.
"""
from __future__ import annotations

from app.main import main


if __name__ == "__main__":
    raise SystemExit(main())
