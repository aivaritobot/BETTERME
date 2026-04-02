"""
BETTERME Launcher — Interfaz gráfica sin necesidad de terminal.
Doble clic en BetterMe.command (macOS) para abrir.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

BASE_DIR = Path(__file__).parent

BG = "#12121e"
BG2 = "#1c1c30"
ACCENT = "#00d4aa"
DANGER = "#e74c3c"
FG = "#e0e0e0"
FG_DIM = "#777"
FONT = "Helvetica"


class BetterMeLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BETTERME — Asistente de Ruleta")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self._process: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI BUILD
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = self.root

        # Title
        tk.Label(root, text="BETTERME", font=(FONT, 26, "bold"),
                 bg=BG, fg=ACCENT).pack(pady=(20, 0))
        tk.Label(root, text="Asistente de Ruleta en Vivo  (experimental)",
                 font=(FONT, 10), bg=BG, fg=FG_DIM).pack(pady=(0, 14))

        # ---- SOURCE ----
        src_frame = self._section(root, "Fuente de video")

        self.source_var = tk.StringVar(value="screen")
        sources = [
            ("screen", "Pantalla completa  (auto-detecta la ruleta)"),
            ("0",      "Webcam 0"),
            ("1",      "Webcam 1"),
            ("rtsp",   "Stream RTSP"),
            ("file",   "Archivo de video"),
        ]
        for val, label in sources:
            tk.Radiobutton(
                src_frame, text=label, variable=self.source_var, value=val,
                bg=BG2, fg=FG, selectcolor=BG, activebackground=BG2,
                activeforeground=ACCENT, font=(FONT, 10),
                command=self._on_source_change,
            ).pack(anchor="w", padx=10, pady=1)

        # RTSP URL row
        self._rtsp_row = tk.Frame(src_frame, bg=BG2)
        self._rtsp_row.pack(fill="x", padx=10, pady=(4, 2))
        tk.Label(self._rtsp_row, text="URL:", bg=BG2, fg=FG_DIM,
                 font=(FONT, 9), width=6).pack(side="left")
        self.rtsp_entry = tk.Entry(self._rtsp_row, bg=BG, fg=FG, width=42,
                                   insertbackground=ACCENT, relief="flat", bd=4)
        self.rtsp_entry.insert(0, "rtsp://user:pass@host:554/stream")
        self.rtsp_entry.pack(side="left", padx=4)

        # File path row
        self._file_row = tk.Frame(src_frame, bg=BG2)
        self._file_row.pack(fill="x", padx=10, pady=(2, 6))
        tk.Label(self._file_row, text="Archivo:", bg=BG2, fg=FG_DIM,
                 font=(FONT, 9), width=6).pack(side="left")
        self.file_entry = tk.Entry(self._file_row, bg=BG, fg=FG, width=34,
                                   insertbackground=ACCENT, relief="flat", bd=4)
        self.file_entry.pack(side="left", padx=4)
        tk.Button(self._file_row, text="Examinar", command=self._browse_file,
                  bg=BG, fg=ACCENT, relief="flat",
                  font=(FONT, 9), cursor="hand2").pack(side="left")

        self._on_source_change()

        # ---- PARAMS ----
        p_frame = self._section(root, "Parámetros")

        # Bankroll
        row1 = tk.Frame(p_frame, bg=BG2)
        row1.pack(fill="x", padx=10, pady=4)
        tk.Label(row1, text="Bankroll (€/$):", bg=BG2, fg=FG,
                 font=(FONT, 10), width=20, anchor="w").pack(side="left")
        self.bankroll_var = tk.StringVar(value="100")
        tk.Entry(row1, textvariable=self.bankroll_var, width=10,
                 bg=BG, fg=FG, insertbackground=ACCENT,
                 relief="flat", bd=4).pack(side="left", padx=4)

        # Opciones de modo
        row2 = tk.Frame(p_frame, bg=BG2)
        row2.pack(fill="x", padx=10, pady=4)
        self.audit_only_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row2, text="Solo auditoría (sin ventana de video)",
                       variable=self.audit_only_var,
                       bg=BG2, fg=FG, selectcolor=BG,
                       activebackground=BG2, activeforeground=ACCENT,
                       font=(FONT, 10)).pack(anchor="w")

        row3 = tk.Frame(p_frame, bg=BG2)
        row3.pack(fill="x", padx=10, pady=(4, 8))
        self.voice_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row3, text="Activar voz (TTS)",
                       variable=self.voice_var,
                       bg=BG2, fg=FG, selectcolor=BG,
                       activebackground=BG2, activeforeground=ACCENT,
                       font=(FONT, 10)).pack(anchor="w")

        # ---- MONTE CARLO ----
        mc_frame = self._section(root, "Simulación")
        mc_row = tk.Frame(mc_frame, bg=BG2)
        mc_row.pack(fill="x", padx=10, pady=6)
        tk.Label(mc_row, text="Monte Carlo 10k spins:", bg=BG2, fg=FG,
                 font=(FONT, 10)).pack(side="left")
        tk.Button(mc_row, text="Ejecutar simulación", command=self._run_simulate,
                  bg=BG, fg=ACCENT, relief="flat",
                  font=(FONT, 9), cursor="hand2").pack(side="left", padx=8)

        # ---- BUTTONS ----
        btn_frame = tk.Frame(root, bg=BG)
        btn_frame.pack(pady=12)

        self.start_btn = tk.Button(
            btn_frame, text="  INICIAR  ", command=self.start,
            bg=ACCENT, fg=BG, font=(FONT, 13, "bold"),
            relief="flat", padx=18, pady=8, cursor="hand2",
        )
        self.start_btn.pack(side="left", padx=8)

        self.stop_btn = tk.Button(
            btn_frame, text="  DETENER  ", command=self.stop,
            bg=DANGER, fg="white", font=(FONT, 13, "bold"),
            relief="flat", padx=18, pady=8, cursor="hand2",
            state="disabled",
        )
        self.stop_btn.pack(side="left", padx=8)

        # ---- RESULTS ----
        res_frame = self._section(root, "Resultados en vivo")

        metrics = [
            ("Confianza (Shannon)", "conf_disp",    "—"),
            ("Entropía (bits)",     "entropy_disp", "—"),
            ("Edge estimado",       "edge_disp",    "—"),
            ("Top 5 números",       "top_disp",     "—"),
            ("Tilt factor",         "tilt_disp",    "—"),
            ("Apuesta Kelly",       "bet_disp",     "—"),
            ("Profit est. 1h",      "profit_disp",  "—"),
            ("Señal",               "signal_disp",  "—"),
        ]
        self._metric_vars: dict[str, tk.StringVar] = {}
        for label, key, default in metrics:
            row = tk.Frame(res_frame, bg=BG2)
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=f"{label}:", bg=BG2, fg=FG_DIM,
                     font=(FONT, 10), width=20, anchor="e").pack(side="left")
            var = tk.StringVar(value=default)
            self._metric_vars[key] = var
            tk.Label(row, textvariable=var, bg=BG2, fg=ACCENT,
                     font=(FONT, 11, "bold"), width=28,
                     anchor="w").pack(side="left", padx=6)

        # ---- LOG ----
        log_frame = self._section(root, "Log")
        self.log_text = tk.Text(
            log_frame, height=7, bg="#0b0b18", fg="#666",
            font=("Courier", 9), state="disabled", wrap="word",
            bd=0, relief="flat",
        )
        sb = tk.Scrollbar(log_frame, command=self.log_text.yview, bg=BG2)
        self.log_text.configure(yscrollcommand=sb.set)
        self.log_text.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        sb.pack(side="right", fill="y")

        tk.Label(root, text="Solo para uso experimental/laboratorio",
                 font=(FONT, 8), bg=BG, fg=FG_DIM).pack(pady=(4, 12))

    def _section(self, parent, title: str) -> tk.Frame:
        outer = tk.LabelFrame(
            parent, text=f"  {title}  ", bg=BG2, fg=FG_DIM,
            font=(FONT, 9), bd=1, relief="groove",
        )
        outer.pack(fill="x", padx=16, pady=6)
        return outer

    # ------------------------------------------------------------------
    # EVENT HANDLERS
    # ------------------------------------------------------------------

    def _on_source_change(self):
        src = self.source_var.get()
        state_rtsp = "normal" if src == "rtsp" else "disabled"
        state_file = "normal" if src == "file" else "disabled"
        for w in self._rtsp_row.winfo_children():
            try:
                w.configure(state=state_rtsp)
            except tk.TclError:
                pass
        for w in self._file_row.winfo_children():
            try:
                w.configure(state=state_file)
            except tk.TclError:
                pass

    def _browse_file(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("Todos",  "*.*"),
            ]
        )
        if path:
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, path)

    # ------------------------------------------------------------------
    # PROCESS MANAGEMENT
    # ------------------------------------------------------------------

    def _build_source(self) -> str:
        src = self.source_var.get()
        if src == "rtsp":
            return self.rtsp_entry.get().strip()
        if src == "file":
            return self.file_entry.get().strip()
        return src  # "screen", "0", "1"

    def _build_cmd(self, extra_args: list[str] | None = None) -> list[str]:
        cmd = [
            sys.executable, "-u",
            str(BASE_DIR / "main.py"),
        ]
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    def start(self):
        source = self._build_source()
        if not source:
            self._log("Error: selecciona una fuente de video válida.")
            return

        bankroll = self.bankroll_var.get().strip() or "100"
        cmd = self._build_cmd([
            "--source", source,
            "--bankroll", bankroll,
        ])
        if self.audit_only_var.get():
            cmd.append("--audit-only")
        if self.voice_var.get():
            cmd.append("--voice")

        self._log(f"Iniciando: source={source}  bankroll={bankroll}")
        self._launch(cmd)

    def _run_simulate(self):
        cmd = self._build_cmd(["--simulate"])
        self._log("Ejecutando Monte Carlo 10k runs...")
        self._launch(cmd)

    def _launch(self, cmd: list[str]):
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(BASE_DIR),
            )
        except Exception as exc:
            self._log(f"No se pudo iniciar el proceso: {exc}")
            return

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        self._reader_thread = threading.Thread(
            target=self._read_output, daemon=True
        )
        self._reader_thread.start()
        self.root.after(600, self._poll_process)

    def stop(self):
        if self._process is not None:
            self._log("Deteniendo...")
            try:
                self._process.terminate()
            except Exception:
                pass
            self._process = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._reset_metrics()

    def _read_output(self):
        if self._process is None or self._process.stdout is None:
            return
        for line in self._process.stdout:
            line = line.rstrip("\n")
            if line.startswith("RESULT:"):
                self._parse_result(line[7:])
            else:
                self.root.after(0, self._log, line)

    def _parse_result(self, payload: str):
        try:
            data = json.loads(payload)
        except Exception:
            return
        self.root.after(0, self._update_metrics, data)

    def _update_metrics(self, data: dict):
        conf = data.get("confidence", 0)
        self._metric_vars["conf_disp"].set(f"{conf:.1%}")

        entropy = data.get("entropy_bits", 0)
        max_e = 5.209
        pct = entropy / max_e * 100
        self._metric_vars["entropy_disp"].set(f"{entropy:.3f} bits  ({pct:.0f}% max)")

        edge = data.get("edge", 0)
        self._metric_vars["edge_disp"].set(f"{edge:.3f}")

        top = data.get("top_numbers", [])
        self._metric_vars["top_disp"].set(
            "  ".join(str(n) for n in top[:5]) if top else "—"
        )

        tilt = data.get("tilt_factor", 1.0)
        self._metric_vars["tilt_disp"].set(f"{tilt:.3f}")

        safe = data.get("safe_mode", False)
        bet = data.get("bet_amount", 0)
        label = f"${bet:.2f}  [SAFE MODE]" if safe else f"${bet:.2f}"
        self._metric_vars["bet_disp"].set(label)

        profit = data.get("expected_profit_1h", 0)
        self._metric_vars["profit_disp"].set(f"${profit:.2f}/h  (est.)")

        should = data.get("should_bet", False)
        signal_text = "APOSTAR" if should else "Esperar"
        signal_color = ACCENT if should else FG_DIM
        self._metric_vars["signal_disp"].set(signal_text)
        for w in self.root.winfo_children():
            self._set_signal_color(w, signal_color)

    def _set_signal_color(self, widget, color: str):
        """Busca el label de señal y actualiza su color."""
        try:
            if isinstance(widget, tk.Label):
                if widget.cget("textvariable") == str(self._metric_vars["signal_disp"]):
                    widget.configure(fg=color)
                    return
            for child in widget.winfo_children():
                self._set_signal_color(child, color)
        except Exception:
            pass

    def _reset_metrics(self):
        for var in self._metric_vars.values():
            var.set("—")

    def _poll_process(self):
        if self._process is not None and self._process.poll() is not None:
            self._log("Proceso finalizado.")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self._process = None
            return
        if self._process is not None:
            self.root.after(600, self._poll_process)

    # ------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    BetterMeLauncher().run()
