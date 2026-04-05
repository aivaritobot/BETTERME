from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from app.capture_overlay import CaptureOverlay
from app.config_store import SettingsManager
from app.models import AppStatus, CaptureROI, RuntimeSettings
from app.session import SessionController


class BetterMeDesktopApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BETTERME Desktop · Research / Demo / Audit")
        self.root.configure(bg="#0d1222")

        self.settings = SettingsManager(Path("app_state.json"))
        self.root.geometry(self.settings.ui.geometry)

        self._build_styles()
        self._build_ui()

        self.session = SessionController(Path("config.json"), self._emit)
        self.session.update_settings(self.settings.runtime)

        self.overlay = CaptureOverlay(
            root=self.root,
            initial_roi=self.settings.capture,
            on_roi_change=self._on_overlay_roi,
            on_lock_change=self._on_overlay_lock,
        )
        self.overlay.set_locked(self.settings.ui.capture_locked)
        self.root.after(50, self.overlay.lift)

        self._set_status(AppStatus.READY.value, "Listo para iniciar")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_styles(self) -> None:
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure("Root.TFrame", background="#0d1222")
        s.configure("Card.TFrame", background="#151c33")
        s.configure("Header.TLabel", background="#151c33", foreground="#f4f7ff", font=("Segoe UI", 16, "bold"))
        s.configure("Text.TLabel", background="#151c33", foreground="#b8c4e9", font=("Segoe UI", 10))

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, style="Root.TFrame", padding=12)
        root_frame.pack(fill="both", expand=True)

        left = ttk.Frame(root_frame, style="Card.TFrame", padding=14)
        right = ttk.Frame(root_frame, style="Card.TFrame", padding=14)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="BETTERME", style="Header.TLabel").pack(anchor="w")
        ttk.Label(left, text="Asistente visual de laboratorio", style="Text.TLabel").pack(anchor="w", pady=(0, 8))

        self.status_var = tk.StringVar(value="Ready")
        self.status_msg_var = tk.StringVar(value="")
        ttk.Label(left, text="Estado", style="Text.TLabel").pack(anchor="w")
        ttk.Label(left, textvariable=self.status_var, style="Text.TLabel").pack(anchor="w")
        ttk.Label(left, textvariable=self.status_msg_var, style="Text.TLabel").pack(anchor="w", pady=(0, 8))

        source_row = ttk.Frame(left, style="Card.TFrame")
        source_row.pack(fill="x", pady=4)
        ttk.Label(source_row, text="Fuente", style="Text.TLabel").pack(side="left")
        self.source_var = tk.StringVar(value=self.settings.runtime.source)
        src = ttk.Combobox(source_row, textvariable=self.source_var, state="readonly", values=["screen", "0", "1", "rtsp"])
        src.pack(side="left", padx=10)
        src.bind("<<ComboboxSelected>>", lambda _e: self._save_runtime())

        bankroll_row = ttk.Frame(left, style="Card.TFrame")
        bankroll_row.pack(fill="x", pady=4)
        ttk.Label(bankroll_row, text="Bankroll", style="Text.TLabel").pack(side="left")
        self.bankroll_var = tk.StringVar(value=str(self.settings.runtime.bankroll))
        b = ttk.Entry(bankroll_row, textvariable=self.bankroll_var, width=12)
        b.pack(side="left", padx=10)
        b.bind("<FocusOut>", lambda _e: self._save_runtime())

        self.voice_var = tk.BooleanVar(value=self.settings.runtime.voice)
        ttk.Checkbutton(left, text="Voz", variable=self.voice_var, command=self._save_runtime).pack(anchor="w", pady=2)

        mode_row = ttk.Frame(left, style="Card.TFrame")
        mode_row.pack(fill="x", pady=4)
        ttk.Label(mode_row, text="Inferencia", style="Text.TLabel").pack(side="left")
        self.inference_mode_var = tk.StringVar(value=self.settings.runtime.inference_mode)
        inf = ttk.Combobox(
            mode_row,
            textvariable=self.inference_mode_var,
            state="readonly",
            values=["analytic", "reactive"],
            width=12,
        )
        inf.pack(side="left", padx=10)
        inf.bind("<<ComboboxSelected>>", lambda _e: self._save_runtime())

        weight_row = ttk.Frame(left, style="Card.TFrame")
        weight_row.pack(fill="x", pady=4)
        ttk.Label(weight_row, text="Peso sesión", style="Text.TLabel").pack(side="left")
        self.execution_weight_var = tk.StringVar(value=str(self.settings.runtime.execution_weight))
        w = ttk.Combobox(weight_row, textvariable=self.execution_weight_var, state="readonly", values=["3", "5", "10", "50", "100"], width=8)
        w.pack(side="left", padx=10)
        w.bind("<<ComboboxSelected>>", lambda _e: self._save_runtime())

        self.mode_var = tk.StringVar(value="Research / Demo / Audit")
        ttk.Label(left, text="Modo activo", style="Text.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Label(left, textvariable=self.mode_var, style="Text.TLabel").pack(anchor="w")

        buttons = ttk.Frame(left, style="Card.TFrame")
        buttons.pack(fill="x", pady=10)
        self.start_btn = ttk.Button(buttons, text="Start", command=self.start)
        self.pause_btn = ttk.Button(buttons, text="Pause", command=self.pause, state="disabled")
        self.resume_btn = ttk.Button(buttons, text="Resume", command=self.resume, state="disabled")
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.reset_btn = ttk.Button(buttons, text="Reset", command=self.reset)
        self.unlock_btn = ttk.Button(buttons, text="Reposition / Unlock capture area", command=self.unlock_capture)

        self.start_btn.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        self.pause_btn.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        self.resume_btn.grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        self.stop_btn.grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        self.reset_btn.grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        self.unlock_btn.grid(row=1, column=2, padx=2, pady=2, sticky="ew")

        for c in range(3):
            buttons.grid_columnconfigure(c, weight=1)

        settings_btn = ttk.Button(left, text="Settings", command=self._show_settings)
        settings_btn.pack(fill="x", pady=(4, 0))

        self.advanced_var = tk.BooleanVar(value=self.settings.ui.advanced_mode)
        ttk.Checkbutton(left, text="Advanced", variable=self.advanced_var, command=self._toggle_advanced).pack(anchor="w", pady=(10, 2))

        ttk.Label(right, text="Panel de eventos", style="Header.TLabel").pack(anchor="w")
        self.chat = tk.Text(right, height=14, bg="#0a0f1f", fg="#dbe4ff", relief="flat", wrap="word")
        self.chat.pack(fill="both", expand=True, pady=(8, 10))
        self.chat.configure(state="disabled")
        self._chat("App iniciada. Ajusta el área de captura y pulsa Start.")

        metrics = ttk.Frame(right, style="Card.TFrame")
        metrics.pack(fill="x")
        self.signal_var = tk.StringVar(value="—")
        self.conf_var = tk.StringVar(value="—")
        self.top_var = tk.StringVar(value="—")
        ttk.Label(metrics, text="Señal:", style="Text.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(metrics, textvariable=self.signal_var, style="Text.TLabel").grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(metrics, text="Confianza:", style="Text.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(metrics, textvariable=self.conf_var, style="Text.TLabel").grid(row=1, column=1, sticky="w", padx=8)
        ttk.Label(metrics, text="Top:", style="Text.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Label(metrics, textvariable=self.top_var, style="Text.TLabel").grid(row=2, column=1, sticky="w", padx=8)

        self.health_text = tk.Text(right, height=5, bg="#0a0f1f", fg="#8ea2d8", relief="flat", wrap="word")
        self.health_text.pack(fill="x", pady=(8, 0))

        self.advanced_panel = tk.Text(right, height=8, bg="#060a16", fg="#7f8cb0", relief="flat", wrap="word")
        self.advanced_panel.pack(fill="x", pady=(8, 0))
        if not self.advanced_var.get():
            self.advanced_panel.pack_forget()

    def _chat(self, msg: str) -> None:
        self.chat.configure(state="normal")
        self.chat.insert("end", f"• {msg}\n")
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def _log_advanced(self, msg: str) -> None:
        self.advanced_panel.insert("end", msg + "\n")
        self.advanced_panel.see("end")

    def _emit(self, event: str, payload: dict) -> None:
        self.root.after(0, lambda: self._handle_event(event, payload))

    def _handle_event(self, event: str, payload: dict) -> None:
        if event == "status":
            self._set_status(payload.get("status", "Ready"), payload.get("message", ""))
            self._chat(payload.get("message", ""))
            self._sync_buttons(payload.get("status", ""))
        elif event == "metrics":
            should = bool(payload.get("should_bet", False))
            self.signal_var.set("APOSTAR" if should else "ESPERAR")
            self.conf_var.set(f"{float(payload.get('confidence', 0.0)):.1%}")
            top = payload.get("top_numbers", [])
            self.top_var.set(" ".join(str(x) for x in top[:5]) if top else "—")
        elif event == "health":
            self.health_text.delete("1.0", "end")
            for item in payload.get("checks", []):
                self.health_text.insert("end", f"{item['name']}: {item['value']}\n")
        elif event == "error":
            self._set_status(payload.get("status", "Error"), payload.get("title", "Error"))
            self._chat(f"{payload.get('title', 'Error')} {payload.get('hint', '')}")
            messagebox.showerror("BETTERME", f"{payload.get('title')}\n\n{payload.get('hint')}")
            self._log_advanced(payload.get("detail", ""))
            self._sync_buttons("Error")
        elif event == "log":
            self._log_advanced(payload.get("message", ""))

    def _set_status(self, status: str, message: str) -> None:
        self.status_var.set(status)
        self.status_msg_var.set(message)

    def _sync_buttons(self, status: str) -> None:
        capturing = status == AppStatus.CAPTURING.value
        paused = status == AppStatus.PAUSED.value
        self.start_btn.configure(state="disabled" if capturing or paused else "normal")
        self.pause_btn.configure(state="normal" if capturing else "disabled")
        self.resume_btn.configure(state="normal" if paused else "disabled")
        self.stop_btn.configure(state="normal" if capturing or paused else "disabled")

    def _save_runtime(self) -> None:
        try:
            bankroll = float(self.bankroll_var.get().strip())
        except Exception:
            bankroll = 100.0
            self.bankroll_var.set("100")
        self.settings.runtime = RuntimeSettings(
            source=self.source_var.get(),
            bankroll=bankroll,
            voice=bool(self.voice_var.get()),
            inference_mode=str(self.inference_mode_var.get() or "analytic"),
            execution_weight=int(self.execution_weight_var.get() or 10),
        )
        self.session.update_settings(self.settings.runtime)
        self.settings.save()

    def _on_overlay_roi(self, roi: CaptureROI) -> None:
        self.settings.capture = roi
        self.session.update_roi(roi)
        self.settings.save()

    def _on_overlay_lock(self, locked: bool) -> None:
        self.settings.ui.capture_locked = locked
        self.settings.save()

    def start(self) -> None:
        self._save_runtime()
        self.session.start()

    def pause(self) -> None:
        self.session.pause()

    def resume(self) -> None:
        self.session.resume()

    def stop(self) -> None:
        self.session.stop()

    def reset(self) -> None:
        self.session.reset()
        self.signal_var.set("—")
        self.conf_var.set("—")
        self.top_var.set("—")
        self._chat("Estado reiniciado.")

    def unlock_capture(self) -> None:
        if self.settings.ui.capture_locked:
            self.overlay.toggle_lock()
        self.overlay.lift()
        self._chat("Área de captura lista para mover/redimensionar.")

    def _show_settings(self) -> None:
        messagebox.showinfo(
            "Settings",
            "La configuración básica se guarda automáticamente.\n"
            "Usa Advanced para ver logs técnicos.",
        )

    def _toggle_advanced(self) -> None:
        self.settings.ui.advanced_mode = bool(self.advanced_var.get())
        self.settings.save()
        if self.advanced_var.get():
            self.advanced_panel.pack(fill="x", pady=(8, 0))
        else:
            self.advanced_panel.pack_forget()

    def run(self) -> None:
        self.root.mainloop()

    def _on_close(self) -> None:
        self.session.stop()
        self.settings.ui.geometry = self.root.geometry()
        self.settings.save()
        self.overlay.destroy()
        self.root.destroy()
