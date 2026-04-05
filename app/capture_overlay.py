from __future__ import annotations

import tkinter as tk
from typing import Callable

from app.models import CaptureROI


class CaptureOverlay:
    def __init__(
        self,
        root: tk.Tk,
        initial_roi: CaptureROI,
        on_roi_change: Callable[[CaptureROI], None],
        on_lock_change: Callable[[bool], None],
    ):
        self._on_roi_change = on_roi_change
        self._on_lock_change = on_lock_change
        self._locked = False

        self.window = tk.Toplevel(root)
        self.window.title("BETTERME · Área de captura")
        self.window.attributes("-topmost", True)
        self.window.geometry(f"{initial_roi.width}x{initial_roi.height}+{initial_roi.left}+{initial_roi.top}")
        self.window.minsize(240, 240)
        self.window.configure(bg="#18b38f")

        frame = tk.Frame(self.window, bg="#101425", bd=2, relief="ridge")
        frame.pack(fill="both", expand=True, padx=4, pady=4)

        tk.Label(
            frame,
            text="Área de captura",
            font=("Segoe UI", 11, "bold"),
            bg="#101425",
            fg="#b3efd9",
        ).pack(anchor="nw", padx=10, pady=(8, 2))
        tk.Label(
            frame,
            text="Mueve o redimensiona esta ventana.",
            font=("Segoe UI", 9),
            bg="#101425",
            fg="#7f92be",
        ).pack(anchor="nw", padx=10)

        controls = tk.Frame(frame, bg="#101425")
        controls.pack(side="bottom", fill="x", padx=10, pady=8)
        self.lock_btn = tk.Button(
            controls,
            text="Lock Area",
            command=self.toggle_lock,
            bg="#253454",
            fg="#d6e0ff",
            relief="flat",
            padx=8,
        )
        self.lock_btn.pack(side="left")

        self.reset_btn = tk.Button(
            controls,
            text="Reset",
            command=self._reset_default,
            bg="#253454",
            fg="#d6e0ff",
            relief="flat",
            padx=8,
        )
        self.reset_btn.pack(side="left", padx=6)

        self.window.bind("<Configure>", self._handle_configure)
        self._on_roi_change(self.current_roi())

    def _handle_configure(self, _event=None) -> None:
        if self._locked:
            return
        self._on_roi_change(self.current_roi())

    def toggle_lock(self) -> None:
        self._locked = not self._locked
        if self._locked:
            self.window.resizable(False, False)
            self.lock_btn.configure(text="Unlock Area")
        else:
            self.window.resizable(True, True)
            self.lock_btn.configure(text="Lock Area")
            self._on_roi_change(self.current_roi())
        self._on_lock_change(self._locked)

    def set_locked(self, locked: bool) -> None:
        if self._locked == locked:
            return
        self.toggle_lock()

    def _reset_default(self) -> None:
        self.window.geometry("820x820+240+120")
        self.window.resizable(True, True)
        self._locked = False
        self.lock_btn.configure(text="Lock Area")
        self._on_lock_change(False)
        self._on_roi_change(self.current_roi())

    def current_roi(self) -> CaptureROI:
        return CaptureROI(
            left=self.window.winfo_x(),
            top=self.window.winfo_y(),
            width=max(80, self.window.winfo_width()),
            height=max(80, self.window.winfo_height()),
        )

    def lift(self) -> None:
        self.window.lift()

    def destroy(self) -> None:
        self.window.destroy()
